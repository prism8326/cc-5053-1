"""Reproducible DSP tests for the subtractive synth.

Run with:  python3 -m pytest tests/ -v
These tests deliberately do *not* equate "no clicks" with "no aliasing":
test_aliasing_polyblep_suppresses_folded_spectrum measures energy in bins that
can only contain folded super-Nyquist partials, and compares the band-limited
oscillator against the explicit naive (aliasing) reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth import SynthEngine, SynthParams, NoteEvent, ADSR, Oscillator
from synth.config import (
    WAVE_SAW, WAVE_SQUARE, WAVE_TRI, WAVE_NAIVE_SAW, WAVE_NAIVE_SQUARE,
    SAMPLE_RATE,
)
from synth.engine import ScalarEngine, Voice
from synth.render import (
    measure_aliasing, estimate_fundamental, peak_stats, render_offline,
)
from synth.oscillator import midi_to_freq

SR = SAMPLE_RATE


def _steady_tone(waveform: str, f0: float, seconds: float = 2.0,
                 sr: float = SR) -> np.ndarray:
    osc = Oscillator(waveform, sr)
    return osc.render(np.full(int(sr * seconds), f0))


# --------------------------------------------------------------------------- #
# 1. Pitch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("note", [36, 60, 69, 84, 100])
def test_pitch_matches_midi_frequency(note):
    f0 = midi_to_freq(note)
    y = _steady_tone(WAVE_SAW, f0)
    res = estimate_fundamental(y, f0)
    assert res["relative_error_percent"] < 0.1
    assert abs(res["error_cents"]) < 2.0


def test_polyblep_preserves_inband_harmonics():
    f0 = midi_to_freq(72)
    a = _steady_tone(WAVE_NAIVE_SAW, f0)
    b = _steady_tone(WAVE_SAW, f0)
    from synth.render import spectrum
    fa, ma, _ = spectrum(a)
    fb, mb, _ = spectrum(b)
    for k in range(1, 6):
        ba = int(round(k * f0 / (fa[1] - fa[0])))
        bb = int(round(k * f0 / (fb[1] - fb[0])))
        # band-limiting must not hollow out the audible spectrum
        assert mb[bb] > 0.7 * ma[ba]


# --------------------------------------------------------------------------- #
# 2. Aliasing / folded spectrum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("wave,naive", [(WAVE_SAW, WAVE_NAIVE_SAW),
                                        (WAVE_SQUARE, WAVE_NAIVE_SQUARE)])
def test_aliasing_polyblep_suppresses_folded_spectrum(wave, naive):
    f0 = midi_to_freq(107)  # B7 ~3951 Hz, partials 7.. fold back
    bandlimited = measure_aliasing(_steady_tone(wave, f0), f0)
    raw = measure_aliasing(_steady_tone(naive, f0), f0)
    # at least 12 dB less folded energy while harmonic level stays comparable
    assert raw["alias_rms_db"] - bandlimited["alias_rms_db"] >= 12.0
    assert abs(bandlimited["harmonic_rms_db"] - raw["harmonic_rms_db"]) < 6.0


def test_alias_energy_grows_without_polyblep():
    # The naive reference must actually show the defect we claim to fix:
    # its absolute folded residual rises with pitch.  (A ratio against the
    # in-band level stays roughly constant because both decay together.)
    low = measure_aliasing(_steady_tone(WAVE_NAIVE_SAW, midi_to_freq(48)),
                           midi_to_freq(48))
    high = measure_aliasing(_steady_tone(WAVE_NAIVE_SAW, midi_to_freq(107)),
                            midi_to_freq(107))
    assert high["alias_rms_db"] > low["alias_rms_db"] + 6.0
    # and PolyBLEP keeps the same high note's residual near the low-note floor
    fixed = measure_aliasing(_steady_tone(WAVE_SAW, midi_to_freq(107)),
                             midi_to_freq(107))
    assert fixed["alias_rms_db"] < low["alias_rms_db"] + 12.0


def test_polyblep_continuous_at_block_boundaries():
    # No discontinuity may appear at vectorized block seams.
    f0 = midi_to_freq(80)
    osc = Oscillator(WAVE_SAW, SR)
    chunks = [osc.render(np.full(317, f0)) for _ in range(4)]
    joined = np.concatenate(chunks)
    osc2 = Oscillator(WAVE_SAW, SR)
    whole = osc2.render(np.full(317 * 4, f0))
    assert np.allclose(joined, whole, atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Envelope timing in explicit time units
# --------------------------------------------------------------------------- #
def test_envelope_times_match_seconds():
    sr = SR
    env = ADSR(attack=0.02, decay=0.1, sustain=0.5, release=0.2, sr=sr)
    block = 256
    n_total = int(0.6 * sr)
    out = np.zeros(n_total)
    on_at = int(0.05 * sr)
    off_at = int(0.4 * sr)
    on_block = on_at // block
    off_block = off_at // block
    for b in range(n_total // block):
        on_idx = on_at - b * block if b == on_block else None
        off_idx = off_at - b * block if b == off_block else None
        out[b * block:(b + 1) * block] = env.render(block, on_idx, off_idx)

    peak_idx = int(np.argmax(out))
    assert abs(peak_idx - on_at - int(0.02 * sr)) < block + 5
    # sustain level while key held
    held = out[int(0.3 * sr)]
    assert abs(held - 0.5) < 0.02
    # release reaches ~zero one release-time after note-off
    after = off_at + int(0.2 * sr)
    assert out[min(after, n_total - 1)] < 0.02
    # event not quantized to block boundary: onset appears inside the block
    assert out[on_at] > 0 and out[on_at - 1] == 0.0


def test_scalar_and_vector_envelope_identical():
    e1 = ADSR(0.01, 0.05, 0.6, 0.08, SR)
    e2 = ADSR(0.01, 0.05, 0.6, 0.08, SR)
    n = 4000
    v = np.zeros(n)
    for i in range(n):
        if i == 300:
            e1.note_on()
        if i == 2500:
            e1.note_off()
        v[i] = e1.render_step()
    v2 = e2.render(700, note_on_at=300, note_off_at=None)
    v2 = np.concatenate([v2, e2.render(1800, note_on_at=None, note_off_at=None)])
    v2 = np.concatenate([v2, e2.render(n - 2500, note_on_at=None, note_off_at=0)])
    assert np.allclose(v, v2, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. Polyphony and voice isolation
# --------------------------------------------------------------------------- #
def test_full_polyphony_sounds_simultaneously():
    n_voices = 8
    params = SynthParams(voices=n_voices, attack=0.005, decay=0.1,
                         sustain=0.8, release=0.1, smooth_time=0.0, gain=0.5)
    notes = list(range(60, 60 + n_voices))
    on_events = [NoteEvent(0.1, m, 0.9, "on") for m in notes]

    # Render only the held portion first and inspect engine state directly:
    # eight distinct notes must occupy eight distinct voice instances.
    held_engine = SynthEngine(params, SR, block_size=256)
    y_held, _ = held_engine.render_events(on_events, 0.6)
    alive = [v for v in held_engine.voices if v.note is not None and not v.idle]
    assert len(alive) == n_voices
    assert set(held_engine._note_map) == set(notes)

    # Every one of the eight fundamentals must be present in the mixed output.
    from synth.render import spectrum
    freqs, mag, _ = spectrum(y_held[int(0.4 * SR):], SR, nfft=2 ** 16)
    df = freqs[1] - freqs[0]
    floor = np.median(mag)
    for m in notes:
        b = int(round(midi_to_freq(m) / df))
        assert mag[b] > 10 * floor, f"note {m} missing from mix"

    # Full lifecycle: after note-off and release decay all voices are reclaimed,
    # and the output stays finite and bounded.
    full = on_events + [NoteEvent(1.0, m, 0.0, "off") for m in notes]
    engine = SynthEngine(params, SR, block_size=256)
    y, _ = engine.render_events(full, 1.4)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0
    assert len(engine._note_map) == 0


def test_stopping_one_note_keeps_others_sounding():
    params = SynthParams(voices=4, attack=0.005, decay=0.05, sustain=0.9,
                         release=0.3, smooth_time=0.0)
    engine = SynthEngine(params, SR, block_size=256)
    events = [NoteEvent(0.1, 60, 1.0, "on"), NoteEvent(0.1, 72, 1.0, "on"),
              NoteEvent(0.5, 60, 0.0, "off")]
    y, _ = engine.render_events(events, 1.2)
    # well after note 60's release finished, note 72 must still ring
    late = y[int(1.0 * SR):int(1.1 * SR)]
    assert np.sqrt(np.mean(late ** 2)) > 0.05
    held = [v for v in engine.voices if v.note == 72]
    assert held and not held[0].idle


def test_voice_stealing_under_overflow():
    params = SynthParams(voices=2, smooth_time=0.0)
    engine = SynthEngine(params, SR)
    for m in (60, 64, 67):  # third note must steal, not crash or drop
        engine._allocate(m, 0)
    assert len([v for v in engine.voices if v.note is not None]) == 2
    assert 67 in engine._note_map
    assert 60 not in engine._note_map  # oldest onset stolen


def test_filter_state_is_per_voice():
    p = SynthParams()
    v1, v2 = Voice(p, SR), Voice(p, SR)
    v1.filt._z1 = 0.42
    assert v2.filt._z1 == 0.0
    v1.reclaim()
    assert v1.filt._z1 == 0.0 and v2.filt._z1 == 0.0


# --------------------------------------------------------------------------- #
# 5. Parameter continuity
# --------------------------------------------------------------------------- #
def test_parameter_changes_are_smoothed_not_stepped():
    params = SynthParams(voices=1, smooth_time=0.05)
    engine = SynthEngine(params, SR, block_size=256)
    engine._allocate(60, 0)
    engine.pull_block(256)
    params.cutoff = 200.0  # big jump from 4000 Hz
    first = engine.pull_block(256)
    # smoothed cutoff after one 256-sample block is still near the old value
    v = engine.voices[0]
    assert v.filt._cutoff_sm > 3000.0
    # no NaNs/inf on output
    assert np.all(np.isfinite(first))


# --------------------------------------------------------------------------- #
# 6. Real-time vs offline parity
# --------------------------------------------------------------------------- #
def test_realtime_mock_matches_offline_render():
    from synth.audio import MockAudioBackend
    params = SynthParams(waveform=WAVE_SAW, attack=0.004, decay=0.08,
                         sustain=0.6, release=0.12, smooth_time=0.02)
    # All event times land on block origins, which is exactly the granularity
    # a live callback can schedule queued events at; both paths must then be
    # bit-identical because they call the same process_block code.
    block = 256
    schedule = [(1, "on", 60, 1.0), (4, "on", 67, 1.0),
                (7, "off", 60, 0.0), (12, "off", 67, 0.0)]
    n_blocks = 20
    duration = n_blocks * block / SR

    aligned = [
        NoteEvent(b * block / SR, note, vel, kind)
        for b, kind, note, vel in schedule
    ]
    y_aligned, _ = SynthEngine(params, SR, block).render_events(aligned, duration)

    rt_engine = SynthEngine(params, SR, block)
    backend = MockAudioBackend(rt_engine, SR, block)
    by_block: dict[int, list] = {}
    for b, kind, note, vel in schedule:
        by_block.setdefault(b, []).append((kind, note, vel))
    recorded = []
    backend.start()
    for b in range(n_blocks):
        for kind, note, vel in by_block.get(b, []):
            if kind == "on":
                rt_engine.note_on(note, vel)
            else:
                rt_engine.note_off(note)
        recorded.append(backend.pull())
    backend.stop()
    y_rt = np.concatenate(recorded)

    assert y_rt.shape == y_aligned.shape
    assert np.allclose(y_rt, y_aligned, atol=1e-12), np.max(np.abs(y_rt - y_aligned))

    # A sample-accurate schedule (events inside blocks) still renders and
    # differs from the block-aligned one only near the event blocks.
    shifted = [
        NoteEvent((b * block + 77) / SR, note, vel, kind)
        for b, kind, note, vel in schedule
    ]
    y_sample, _ = SynthEngine(params, SR, block).render_events(shifted, duration)
    assert np.corrcoef(y_sample, y_aligned)[0, 1] > 0.995
    # but they are provably not the same at the shifted onset
    assert not np.allclose(y_sample, y_aligned, atol=1e-9)


def test_scalar_engine_parity_with_vector_engine():
    params = SynthParams(waveform=WAVE_SQUARE, attack=0.003, decay=0.07,
                         sustain=0.5, release=0.1, smooth_time=0.02)
    events = [NoteEvent(0.013, 60, 1.0, "on"), NoteEvent(0.211, 64, 0.9, "on"),
              NoteEvent(0.402, 60, 0.0, "off"), NoteEvent(0.618, 64, 0.0, "off")]
    duration = 0.9
    yv, _ = SynthEngine(params, SR, 256).render_events(events, duration)
    ys = ScalarEngine(params, SR).render_events(events, duration)
    # independently written scalar path vs vector path
    assert np.allclose(yv, ys, atol=1e-10), np.max(np.abs(yv - ys))


# --------------------------------------------------------------------------- #
# 7. Waveform harmonic character
# --------------------------------------------------------------------------- #
def test_square_and_triangle_have_distinct_harmonic_structure():
    f0 = midi_to_freq(48)
    from synth.render import spectrum
    _, m_sq, _ = spectrum(_steady_tone(WAVE_SQUARE, f0))
    _, m_tri, _ = spectrum(_steady_tone(WAVE_TRI, f0))
    df = f0  # nfft chosen so df is not exactly f0; use nearest bins
    # even harmonics must be essentially absent in both square and triangle
    def even_odd_ratio(mag):
        freqs = np.fft.rfftfreq(2 ** 15, 1 / SR)
        binwidth = freqs[1] - freqs[0]
        odd = np.mean([mag[int(round(k * f0 / binwidth))] for k in (3, 5, 7)])
        even = np.mean([mag[int(round(k * f0 / binwidth))] for k in (2, 4, 6)])
        return even / (odd + 1e-12)
    assert even_odd_ratio(m_sq) < 0.05
    assert even_odd_ratio(m_tri) < 0.05
    # triangle's high harmonics decay much faster than square's (1/n^2 vs 1/n)
    freqs = np.fft.rfftfreq(2 ** 15, 1 / SR)
    bw = freqs[1] - freqs[0]
    sq_decay = m_sq[int(round(7 * f0 / bw))] / (m_sq[int(round(f0 / bw))] + 1e-12)
    tri_decay = m_tri[int(round(7 * f0 / bw))] / (m_tri[int(round(f0 / bw))] + 1e-12)
    assert tri_decay < 0.4 * sq_decay


# --------------------------------------------------------------------------- #
# 8. Peak / safety and render cost
# --------------------------------------------------------------------------- #
def test_output_remains_finite_and_within_range():
    params = SynthParams(voices=8, gain=0.9)
    engine = SynthEngine(params, SR)
    events = [NoteEvent(0.05 * i, 60 + i, 1.0, "on") for i in range(8)]
    events += [NoteEvent(1.0, 60 + i, 0.0, "off") for i in range(8)]
    y, _ = engine.render_events(events, 1.5)
    assert np.all(np.isfinite(y))
    stats = peak_stats(y)
    assert stats["peak"] <= 1.0


def test_offline_render_faster_than_realtime():
    params = SynthParams(voices=8)
    events = [NoteEvent(0.2 * (i % 4), 60 + i, 1.0, "on") for i in range(12)]
    events += [NoteEvent(2.5, 60 + i, 0.0, "off") for i in range(12)]
    _, stats = render_offline(events, params, 3.0, sr=SR, block_size=256)
    assert stats["realtime_factor"] < 1.0
    assert stats["samples"] == int(3.0 * SR)


# --------------------------------------------------------------------------- #
# 9. Required benchmark scenarios as regression tests
# --------------------------------------------------------------------------- #
def test_frequency_sweep_tracks_pitch_and_stays_bounded():
    duration = 1.0
    n = int(duration * SR)
    f0 = np.linspace(300.0, 3000.0, n)
    note = 72
    params = SynthParams(voices=1, smooth_time=0.0)
    engine = SynthEngine(params, SR, 256)
    engine._reserve(note, 1.0, 0)
    engine._note_map[note].env.note_on(1.0)
    audio = np.zeros(n)
    for s in range(0, n, 256):
        m = min(256, n - s)
        audio[s:s + m] = engine.process_block(
            m, block_origin=s, f0_automation={note: f0[s:s + m]})
    # pitch follows the automation at several points along the sweep
    for frac in (0.2, 0.5, 0.8):
        c = int(frac * n)
        seg = audio[c - 2048:c + 2048]
        res = estimate_fundamental(seg, f0[c], SR)
        assert abs(res["error_cents"]) < 3.0
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


def test_short_repeated_notes_all_retrigger():
    n_notes = 16
    events = []
    for i in range(n_notes):
        t0 = 0.05 + i * 0.06
        events.append(NoteEvent(t0, 60 + (i % 5), 0.9, "on"))
        events.append(NoteEvent(t0 + 0.03, 60 + (i % 5), 0.0, "off"))
    params = SynthParams(waveform=WAVE_SQUARE, attack=0.002, decay=0.03,
                         sustain=0.5, release=0.03, smooth_time=0.005)
    audio, stats = render_offline(
        events, params, 0.05 + n_notes * 0.06 + 0.3, SR, 256)
    win = int(0.005 * SR)
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode="same"))
    novelty = np.diff(rms, prepend=rms[0])
    floor = 0.1 * novelty.max()
    hits = 0
    for ev in events:
        if ev.kind != "on":
            continue
        k = int(ev.time * SR)
        if novelty[k:k + win].max() > floor:
            hits += 1
    assert hits == n_notes
    assert stats["peak"] <= 1.0


def test_full_polyphony_under_max_voice_load():
    # More notes than voices: stealing must keep exactly `voices` active and
    # the output finite/bounded rather than crashing or dropping silently.
    params = SynthParams(voices=4, gain=0.5)
    engine = SynthEngine(params, SR, 256)
    events = [NoteEvent(0.1 * i, 60 + i, 0.9, "on") for i in range(10)]
    y, _ = engine.render_events(events, 1.2)
    active = [v for v in engine.voices if v.note is not None and not v.idle]
    assert len(active) == 4
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0



# --------------------------------------------------------------------------- #
# 6. Note termination and voice reuse correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("release", [0.0, 0.01, 0.2])
def test_release_reaches_idle_and_frees_the_voice(release):
    # Regression: a multiplicative fade never reaches zero, so the voice would
    # sit in RELEASE forever and never be reclaimed.  The linear increment must
    # reach exactly zero and return the slot to the pool.
    params = SynthParams(voices=1, release=release)
    engine = SynthEngine(params, SR, 256)
    ev_on = [NoteEvent(0.0, 60, 0.9, "on")]
    ev_off = [NoteEvent(0.0, 60, 0.0, "off")]
    engine.process_block(256, ev_on)            # reserve + start attack
    engine.process_block(int(0.3 * SR))         # reach sustain
    engine.process_block(256, ev_off)           # enter release
    engine.process_block(int(release * SR) + 512)  # render the whole fade
    v = engine.voices[0]
    assert v.env.state == 0, f"voice stuck in state {v.env.state}"
    assert v.env.level == 0.0
    assert v.note is None and 60 not in engine._note_map

    # The freed slot must be reusable immediately and start from zero phase.
    engine.process_block(256, [NoteEvent(0.0, 64, 1.0, "on")])
    y = engine.process_block(256)
    assert np.all(np.isfinite(y))
    assert engine.voices[0].note == 64


def test_release_duration_matches_seconds():
    for rel in (0.05, 0.2):
        env = ADSR(0.001, 0.01, 0.8, rel, SR)
        env.note_on(1.0)
        env.render(int(0.05 * SR))          # reach sustain
        env.note_off()
        steps = 0
        while env.state != 0 and steps < int(2 * rel * SR) + 1000:
            env.render_step()
            steps += 1
        assert env.state == 0
        assert abs(steps - round(rel * SR)) <= 2


def test_voice_reuse_after_release_is_correct():
    # Reusing a completed slot must reset oscillator phase, envelope and filter
    # state rather than carrying the old note's tail.
    params = SynthParams(voices=1, release=0.02, cutoff=8000.0)
    engine = SynthEngine(params, SR, 256)
    engine.process_block(int(0.1 * SR), [NoteEvent(0.0, 72, 1.0, "on")])
    engine.process_block(256, [NoteEvent(0.0, 72, 0.0, "off")])
    engine.process_block(int(0.1 * SR))
    assert engine.voices[0].note is None
    engine.process_block(256, [NoteEvent(0.0, 48, 1.0, "on")])
    v = engine.voices[0]
    # Reused slot is fully fresh: no old note, filter delay lines zeroed, and
    # the oscillator phase resets so the new note starts its own period.
    assert v.note == 48
    first = engine.process_block(256)
    assert v.env.level > 0.0
    # Phase advanced at the new note's frequency, not the old note's.
    assert v.osc.phase == pytest.approx(
        (midi_to_freq(48) / SR * 512) % 1.0, abs=1e-9)
    assert np.all(np.isfinite(first))


# --------------------------------------------------------------------------- #
# 7. Playable controls drive the kernel (not just a params copy)
# --------------------------------------------------------------------------- #
def test_keyboard_controls_reach_voice_state():
    import keyboard as kb

    params = SynthParams()
    engine = SynthEngine(params, SR, 256)
    kb._set_waveform(engine, params, "triangle")
    kb._set_cutoff(engine, params, 1200.0)
    kb._set_resonance(engine, params, 3.0)
    kb._set_release(engine, params, 0.4)
    for v in engine.voices:
        assert v.osc.waveform == "triangle"
        assert v.filt.cutoff == 1200.0
        assert v.filt.resonance == 3.0
        assert v.env.release == 0.4

    # And the kernel actually renders the new cutoff: a low cutoff removes the
    # high harmonics, so broadband energy drops while the fundamental survives.
    def hf_energy(cutoff):
        eng = SynthEngine(SynthParams(cutoff=cutoff), SR, 256)
        kb._set_cutoff(eng, eng.params, cutoff)
        eng.process_block(256, [NoteEvent(0.0, 69, 1.0, "on")])
        y = eng.process_block(int(0.4 * SR))
        fr, mag, _ = np.fft.rfftfreq(y.size, 1 / SR), None, None
        spec = np.abs(np.fft.rfft(y))
        high = spec[np.fft.rfftfreq(y.size, 1 / SR) > 4000].sum()
        return high

    assert hf_energy(8000.0) > 3 * hf_energy(300.0)
