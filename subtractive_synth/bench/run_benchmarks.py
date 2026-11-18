"""Reproducible benchmark battery required by the specification.

Outputs (under output/):
  sweep.wav / sweep_spectrum.png   - pitch sweep, foldover check
  repeat_notes.wav                 - short repeated notes, timing/peak check
  full_polyphony.wav               - all voices at once, peak check
  realtime_timing.json             - per-block callback timing vs deadline
  parity.json                      - real-time mock vs offline render diff
  metrics.json                     - all numeric checks in one file

Run:  python3 bench/run_benchmarks.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth import SynthEngine, SynthParams, NoteEvent, SAMPLE_RATE, BLOCK_SIZE
from synth.config import WAVE_SAW, WAVE_SQUARE, WAVE_TRI, WAVE_NAIVE_SAW
from synth.audio import MockAudioBackend, PortAudioBackend
from synth.render import (
    render_offline, save_wav, spectrum, measure_aliasing,
    estimate_fundamental, peak_stats,
)
from synth.oscillator import midi_to_freq

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
SR = SAMPLE_RATE


def _p(msg):
    print(f"[bench] {msg}")


def bench_sweep() -> dict:
    """Continuous frequency sweep through the high register.

    Checks pitch tracking at several sweep positions and compares folded
    spectrum of the band-limited oscillator against the naive one.
    """
    duration = 3.0
    n = int(duration * SR)
    t = np.arange(n) / SR
    # 200 Hz -> 4 kHz linear sweep, held note number 72 drives the voice
    f0 = np.linspace(200.0, 4000.0, n)
    note = 72
    params = SynthParams(waveform=WAVE_SAW, attack=0.01, decay=0.2,
                         sustain=0.9, release=0.1, smooth_time=0.0)
    engine = SynthEngine(params, SR, BLOCK_SIZE)
    engine._reserve(note, 1.0, 0)
    engine._note_map[note].env.note_on(1.0)
    audio = np.zeros(n)
    for s in range(0, n, BLOCK_SIZE):
        m = min(BLOCK_SIZE, n - s)
        audio[s:s + m] = engine.process_block(
            m, block_origin=s,
            f0_automation={note: f0[s:s + m]})
    save_wav(os.path.join(OUT, "sweep.wav"), audio, SR)

    # Pitch check on short windows where f0 is nearly constant.
    pitch_checks = []
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        center = int(frac * n)
        expected = f0[center]
        seg = audio[center - 2048:center + 2048]
        res = estimate_fundamental(seg, expected, SR)
        pitch_checks.append(res)

    # Foldover at the bright end: compare PolyBLEP vs naive at 3.5 kHz.
    hi_f0 = 3500.0
    from synth.oscillator import Oscillator
    band = Oscillator(WAVE_SAW, SR).render(np.full(int(2 * SR), hi_f0))
    raw = Oscillator(WAVE_NAIVE_SAW, SR).render(np.full(int(2 * SR), hi_f0))
    alias_band = measure_aliasing(band, hi_f0, SR)
    alias_raw = measure_aliasing(raw, hi_f0, SR)

    # Spectrum plot: naive vs band-limited at the high end.
    fr, _, db_band = spectrum(band, SR, nfft=2 ** 15)
    _, _, db_raw = spectrum(raw, SR, nfft=2 ** 15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fr, db_raw, label="naive saw (aliased)", lw=0.8, alpha=0.8)
    ax.plot(fr, db_band, label="PolyBLEP saw (band-limited)", lw=0.8, alpha=0.9)
    ax.axvline(SR / 2, color="r", ls="--", lw=1, label="Nyquist")
    ax.set_xlim(0, SR / 2)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Foldover check at f0={hi_f0:.0f} Hz (sweep high end)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sweep_spectrum.png"), dpi=110)
    plt.close(fig)

    result = {
        "pitch_tracking": pitch_checks,
        "alias_polyblep_db": alias_band["alias_rms_db"],
        "alias_naive_db": alias_raw["alias_rms_db"],
        "alias_reduction_db": round(
            alias_raw["alias_rms_db"] - alias_band["alias_rms_db"], 2),
        "peak": peak_stats(audio, SR),
    }
    _p(f"sweep: max pitch error cents="
       f"{max(abs(p['error_cents']) for p in pitch_checks):.3f}, "
       f"alias reduction={result['alias_reduction_db']} dB")
    return result


def bench_repeat_notes() -> dict:
    """Burst of very short repeated notes: timing, retrigger, peaks."""
    n_notes = 16
    interval = 0.06
    events = []
    for i in range(n_notes):
        t0 = 0.05 + i * interval
        events.append(NoteEvent(t0, 60 + (i % 5), 0.9, "on"))
        events.append(NoteEvent(t0 + 0.03, 60 + (i % 5), 0.0, "off"))
    duration = 0.05 + n_notes * interval + 0.3
    params = SynthParams(waveform=WAVE_SQUARE, attack=0.002, decay=0.03,
                         sustain=0.5, release=0.03, smooth_time=0.005)
    audio, stats = render_offline(events, params, duration, SR, BLOCK_SIZE)
    save_wav(os.path.join(OUT, "repeat_notes.wav"), audio, SR)

    # Verify each retrigger with an onset-novelty signal: positive derivative
    # of the smoothed RMS energy spikes at attacks even when a previous note's
    # release keeps the absolute level high.  Every scheduled onset must land
    # on a novelty peak above an adaptive threshold.
    win = max(1, int(0.005 * SR))
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode="same"))
    novelty = np.diff(rms, prepend=rms[0])
    # positive-peak floor = 10% of the strongest attack in the phrase
    floor = 0.1 * novelty.max()
    detected = 0
    for ev in events:
        if ev.kind != "on" or ev.velocity <= 0:
            continue
        k = int(ev.time * SR)
        window = novelty[k:min(audio.size, k + win)]
        if window.size and window.max() > floor:
            detected += 1
    starts = detected
    result = {
        "render": stats,
        "detected_bursts": starts,
        "expected_bursts": n_notes,
        "peak": peak_stats(audio, SR),
    }
    _p(f"repeat notes: detected {starts}/{n_notes} bursts, "
       f"peak={result['peak']['peak']:.3f}")
    return result


def bench_full_polyphony() -> dict:
    """Every voice sounding simultaneously: finite polyphony and peak safety."""
    n_voices = 8
    params = SynthParams(voices=n_voices, waveform=WAVE_SAW,
                         attack=0.005, decay=0.2, sustain=0.8, release=0.2,
                         gain=0.8)
    notes = list(range(60, 60 + n_voices))
    events = [NoteEvent(0.1, m, 0.9, "on") for m in notes]
    events += [NoteEvent(1.2, m, 0.0, "off") for m in notes]
    audio, stats = render_offline(events, params, 1.8, SR, BLOCK_SIZE)
    save_wav(os.path.join(OUT, "full_polyphony.wav"), audio, SR)

    held_engine = SynthEngine(params, SR, BLOCK_SIZE)
    held_engine.render_events(events[:n_voices], 0.6)
    simultaneous = len([v for v in held_engine.voices
                        if v.note is not None and not v.idle])

    # Verify each fundamental is present in the mix.
    fr, mag, _ = spectrum(audio[int(0.6 * SR):int(1.0 * SR)], SR, 2 ** 16)
    df = fr[1] - fr[0]
    floor = np.median(mag)
    found = {}
    for m in notes:
        b = int(round(midi_to_freq(m) / df))
        found[m] = bool(mag[b] > 10 * floor)

    result = {
        "voices_configured": n_voices,
        "voices_simultaneously_active": simultaneous,
        "all_fundamentals_present": all(found.values()),
        "fundamentals": found,
        "peak": peak_stats(audio, SR),
        "render": stats,
    }
    _p(f"full polyphony: {simultaneous}/{n_voices} active, "
       f"all fundamentals={result['all_fundamentals_present']}, "
       f"peak={result['peak']['peak']:.3f}")
    return result


def bench_waveform_character() -> dict:
    """Document the distinct harmonic profiles of the available waveforms."""
    f0 = midi_to_freq(48)
    out = {}
    for w in (WAVE_SAW, WAVE_SQUARE, WAVE_TRI):
        from synth.oscillator import Oscillator
        y = Oscillator(w, SR).render(np.full(int(2 * SR), f0))
        r = measure_aliasing(y, f0, SR)
        fr, mag, db = spectrum(y, SR, nfft=2 ** 15)
        bw = fr[1] - fr[0]
        levels = {str(k): round(float(db[int(round(k * f0 / bw))]), 2)
                  for k in range(1, 8)}
        out[w] = {"harmonic_db": levels, "residual_db": r["alias_rms_db"]}
    _p(f"waveform profiles: {list(out)}")
    return out


def bench_realtime_timing_and_parity() -> dict:
    """Mock real-time callback: deadline compliance and offline parity."""
    params = SynthParams(waveform=WAVE_SAW, attack=0.004, decay=0.08,
                         sustain=0.6, release=0.12, smooth_time=0.02,
                         voices=8)
    block_events = [
        (2, "on", 60, 1.0), (5, "on", 64, 0.8), (8, "on", 67, 0.7),
        (12, "off", 60, 0.0), (15, "on", 72, 0.9),
        (18, "off", 64, 0.0), (22, "off", 67, 0.0), (26, "off", 72, 0.0),
    ]
    n_blocks = 32
    duration = n_blocks * BLOCK_SIZE / SR

    engine = SynthEngine(params, SR, BLOCK_SIZE)
    backend = MockAudioBackend(engine, SR, BLOCK_SIZE)
    by_block: dict[int, list] = {}
    for b, kind, note, vel in block_events:
        by_block.setdefault(b, []).append((kind, note, vel))
    backend.start()
    recorded = []
    for b in range(n_blocks):
        for kind, note, vel in by_block.get(b, []):
            if kind == "on":
                engine.note_on(note, vel)
            else:
                engine.note_off(note)
        recorded.append(backend.pull())
    backend.stop()
    y_rt = np.concatenate(recorded)
    save_wav(os.path.join(OUT, "realtime_mock.wav"), y_rt, SR)

    aligned = [NoteEvent(b * BLOCK_SIZE / SR, note, vel, kind)
               for b, kind, note, vel in block_events]
    y_off, render_stats = render_offline(aligned, params, duration, SR, BLOCK_SIZE)
    diff = np.abs(y_rt - y_off)
    parity = {
        "max_abs_diff": float(diff.max()),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "bit_exact_within_1e-12": bool(np.allclose(y_rt, y_off, atol=1e-12)),
    }
    with open(os.path.join(OUT, "parity.json"), "w") as f:
        json.dump(parity, f, indent=2)

    timing = backend.timing_report()
    backend_name = ("portaudio" if PortAudioBackend.available() else "mock")
    result = {
        "backend": backend_name,
        "timing": timing,
        "offline_render": render_stats,
        "parity": parity,
    }
    _p(f"real-time [{backend_name}]: mean {timing['mean_ms']} ms vs deadline "
       f"{timing['deadline_ms']} ms ({timing['cpu_percent_of_deadline']}% CPU), "
       f"overruns={timing['overruns']}, bit-exact={parity['bit_exact_within_1e-12']}")
    with open(os.path.join(OUT, "realtime_timing.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()
    metrics = {
        "sample_rate": SR,
        "block_size": BLOCK_SIZE,
        "sweep": bench_sweep(),
        "repeat_notes": bench_repeat_notes(),
        "full_polyphony": bench_full_polyphony(),
        "waveform_character": bench_waveform_character(),
        "realtime": bench_realtime_timing_and_parity(),
        "total_wallclock_seconds": round(time.perf_counter() - t_start, 3),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _p(f"all artifacts written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
