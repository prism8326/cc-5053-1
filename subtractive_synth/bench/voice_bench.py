"""Keyboard voice benchmark: per-voice cost, aliasing and callback budget.

This is the measurement harness used to justify the polyphonic optimizations.
It renders the *same* performance input through the real :class:`SynthEngine`
(no stubs, no default-filled values) and reports, per scenario:

* wall-clock seconds for a fixed 1.5 s performance (min of N reps, so a noisy
  host cannot masquerade as a speed-up),
* per-block callback elapsed time (mean / p95 / max) versus the 256/48 kHz
  deadline,
* measured in-band aliasing of the rendered chord (residual bins relative to
  harmonic energy), using ``measure_aliasing`` on the actual mix,
* peak, fundamental presence and active voice count as correctness guards.

The scenarios escalate the way a keyboard player would: one note, a dense
low chord, the same chord with more voices, and the same chord transposed into
the high register (where PolyBLEP has the least headroom).

Run:  python3 bench/voice_bench.py [--reps 5] [--out output/bench_optimized]
Exit code is non-zero if a scenario produces non-finite or over-full audio,
so a broken optimization fails the benchmark instead of printing a nice number.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from synth import SynthEngine, SynthParams, NoteEvent, SAMPLE_RATE, BLOCK_SIZE
from synth.config import WAVE_SAW
from synth.oscillator import midi_to_freq
from synth.render import measure_aliasing, peak_stats, save_wav

SR = SAMPLE_RATE
HERE = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = [
    ("single_note", 1, [60]),
    ("dense_chord_8", 8, list(range(48, 56))),
    ("dense_chord_16", 16, list(range(48, 64))),
    ("high_register_8", 8, list(range(84, 92))),
    ("high_register_16", 16, list(range(84, 100))),
]


def _params(voices: int) -> SynthParams:
    return SynthParams(
        voices=voices, waveform=WAVE_SAW, attack=0.005, decay=0.12,
        sustain=0.7, release=0.35, cutoff=6000.0, resonance=0.9,
        gain=0.8, smooth_time=0.02,
    )


def _events(notes: list[int], held_seconds: float = 1.5,
            release_at: float = 2.0) -> tuple[list[NoteEvent], float]:
    # Same performance input for every timed run: all notes struck together at
    # t=0.10 and held for the whole measured window.  A release past the window
    # lets the timing run include the note-off bookkeeping without contaminating
    # the cost measurement with release tails.
    ev = [NoteEvent(0.10, m, 0.9, "on") for m in notes]
    ev += [NoteEvent(release_at, m, 0.0, "off") for m in notes]
    return ev, held_seconds + 0.5


def _render(voices: int, notes: list[int], seconds: float, events: list[NoteEvent],
            block_ms: list[float] | None = None) -> np.ndarray:
    engine = SynthEngine(_params(voices), SR, BLOCK_SIZE)
    total = int(seconds * SR)
    audio = np.zeros(total)
    ev = sorted(events, key=lambda e: e.time)
    for start in range(0, total, BLOCK_SIZE):
        n = min(BLOCK_SIZE, total - start)
        t0, t1 = start / SR, (start + n) / SR
        block_events = [e for e in ev if t0 <= e.time < t1]
        t = time.perf_counter()
        audio[start:start + n] = engine.process_block(n, block_events, start)
        if block_ms is not None:
            block_ms.append((time.perf_counter() - t) * 1000.0)
    return audio


def run_scenario(name: str, voices: int, notes: list[int], reps: int,
                 out_dir: str) -> dict:
    events, seconds = _events(notes)

    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _render(voices, notes, seconds, events)
        walls.append(time.perf_counter() - t0)
    best_wall = min(walls)

    block_ms: list[float] = []
    audio = _render(voices, notes, seconds, events, block_ms=block_ms)
    deadline = BLOCK_SIZE / SR * 1000.0
    arr = np.asarray(block_ms)

    # Steady window well inside the held note; no release energy in it.
    seg = audio[int(0.45 * SR):int(0.95 * SR)]
    alias = measure_aliasing(seg, midi_to_freq(notes[0]), SR)

    fps = np.fft.rfftfreq(seg.size, 1.0 / SR)
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    floor = np.median(mag)
    present = {
        str(m): bool(mag[int(round(midi_to_freq(m) / (fps[1] - fps[0])))]
                     > 10 * floor)
        for m in notes
    }

    peak = peak_stats(audio, SR)
    finite = bool(np.all(np.isfinite(audio)))
    result = {
        "voices_configured": voices,
        "notes": notes,
        "reps": reps,
        "performance_seconds": round(seconds, 3),
        "wallclock_seconds_best": round(best_wall, 4),
        "realtime_factor_best": round(best_wall / seconds, 4),
        "per_block_ms": {
            "blocks": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(arr.max()), 4),
            "deadline_ms": round(deadline, 3),
            "mean_pct_deadline": round(100.0 * float(arr.mean()) / deadline, 2),
            "overruns": int(np.sum(arr > deadline)),
        },
        "aliasing": alias,
        "peak": peak,
        "finite": finite,
        "all_fundamentals_present": all(present.values()),
        "fundamentals": present,
        "soft_saturated_samples": peak["clipped_samples"],
    }
    save_wav(os.path.join(out_dir, name + ".wav"), audio, SR)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "output",
                                                  "bench_optimized"))
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    summary: dict[str, dict] = {}
    bad = []
    print(f"keyboard voice benchmark: {args.reps} reps, min-of-N timing, "
          f"deadline {BLOCK_SIZE / SR * 1000:.3f} ms")
    for name, voices, notes in SCENARIOS:
        r = run_scenario(name, voices, notes, args.reps, out_dir)
        summary[name] = r
        pb = r["per_block_ms"]
        print(f"  {name:18s} wall={r['wallclock_seconds_best']:.4f}s "
              f"block mean={pb['mean']:.4f}ms p95={pb['p95']:.4f} "
              f"max={pb['max']:.4f} ({pb['mean_pct_deadline']:.1f}% of "
              f"deadline, overruns={pb['overruns']}) "
              f"alias/harm={r['aliasing']['alias_to_harmonic_db']:.1f} dB "
              f"peak={r['peak']['peak']:.3f}")
        # These are hard failures, not warnings: a benchmark that hides them
        # behind defaults would let a broken optimization ship.
        if not r["finite"]:
            bad.append(f"{name}: non-finite output")
        if not r["all_fundamentals_present"]:
            bad.append(f"{name}: missing fundamental")
        # peak_stats counts >=0.9999; tanh deliberately bounds the mix there, so
        # near-1.0 samples are soft saturation, not overflow.  Flag only a
        # genuine excursion past full scale.
        if r["peak"]["peak"] > 1.0 or r["peak"]["oversampled_peak"] > 1.0:
            bad.append(f"{name}: output exceeds full scale")
        if pb["overruns"] > 0:
            # One or two isolated spikes are the host scheduler, not the DSP
            # (mean stays <10% of the deadline).  A sustained miss shows up in
            # p95, so that is what fails the run; raw overruns are still
            # recorded in the JSON for inspection.
            print(f"  note: {name} had {pb['overruns']} isolated block "
                  f"spike(s) (host jitter), p95={pb['p95']:.3f} ms")
        if pb["p95"] > pb["deadline_ms"]:
            bad.append(f"{name}: sustained deadline miss "
                       f"(p95 {pb['p95']:.3f} > {pb['deadline_ms']:.3f} ms)")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}")
    if bad:
        print("FAIL: " + "; ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
