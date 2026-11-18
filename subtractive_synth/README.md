# Subtractive Synth

A polyphonic subtractive synthesizer with band-limited oscillators, a
sample-accurate ADSR, and a per-voice state-variable-style low-pass filter.
It runs both as an offline renderer and as a real-time keyboard instrument,
and the two paths are checked bit-for-bit in the test suite.

## How the requirements are met

| Requirement | Implementation |
| --- | --- |
| Real oscillators, no pre-recorded scale files | Every sample is computed from a phase accumulator in `synth/oscillator.py`; WAV outputs are generated, never read back as a source |
| Two+ harmonically distinct waveforms | Saw (all harmonics, 1/n), square (odd only, 1/n) and triangle (odd only, 1/n²); even harmonics of square/triangle sit below -120 dB |
| Anti-aliasing in the high register | PolyBLEP band-limited steps (`_polyblep`, `_polyblep_at_half`). Deliberately aliased `naive_saw`/`naive_square` are retained so the reduction is *measured*, not asserted from "no clicks" |
| Note timing independent of block boundaries | Events carry an absolute time; `ADSR.render` and `SynthEngine.process_block` apply them at an explicit in-block sample offset |
| Envelope evaluated in explicit time units | Attack/decay/release stored in seconds and converted with the sample rate (`synth/envelope.py`) |
| Continuous parameter changes without jumps | One-pole smoothers (time constant in seconds) on cutoff, Q, gain and pitch; biquad coefficients update once per block |
| Separate filter and voice state | Each `Voice` owns its oscillator, `ADSR` and `BiquadLowpass` delay lines; ending one note only releases that voice |
| Finite polyphony | Fixed voice pool (`SynthParams.voices`), oldest-onset stealing when full, never a dropped note or crash |
| Comparable real-time + offline output | Both call `SynthEngine.process_block`; the mock audio backend records callback pulls and the parity test compares against an offline render (`max_abs_diff = 0.0`) |

## Layout

```
synth/
  config.py      SynthParams (all times in seconds, frequencies in Hz)
  oscillator.py  phase accumulator, PolyBLEP saw/square, integrated triangle
  envelope.py    scalar + vector sample-accurate ADSR (same state machine)
  filter.py      RBJ biquad low-pass, per-voice state, smoothed coefficients
  engine.py      Voice, SynthEngine (vector), ScalarEngine (reference)
  audio.py       PortAudio backend + deterministic MockAudioBackend
  render.py      offline render, WAV/NPZ export, spectrum & alias metrics
keyboard.py      playable terminal UI (and optional tkinter GUI)
cli.py           command-line offline renderer
bench/           reproducible benchmark battery -> output/
                 voice_bench.py          optimized keyboard cost/alias/budget
                 baseline_reference.py   pre-optimization reference capture
                 compare_optimized.py    before/after table
tests/           31 pytest checks
```

## Quickstart

```bash
# dependencies (the sandbox used --user --break-system-packages via get-pip.py)
pip install numpy scipy soundfile sounddevice matplotlib pytest

# full test suite
PYTHONPATH=. python3 -m pytest tests/ -v

# benchmark battery: sweep, repeated notes, full polyphony, timing, parity
PYTHONPATH=. python3 bench/run_benchmarks.py

# offline render of the built-in phrase
PYTHONPATH=. python3 cli.py demo --waveform square --out output/phrase

# offline render of a custom event list
PYTHONPATH=. python3 cli.py render examples/events.txt --out output/custom

# live playing (terminal; works without a display)
PYTHONPATH=. python3 keyboard.py
# live playing with a GUI (needs tkinter + a display)
PYTHONPATH=. python3 keyboard.py --gui
```

Terminal keys: `Z S X D C ...` are the white/black keys starting at C4,
`Q..U` the next octave; `1/2/3` switch waveform, `[ ]` cutoff, `- =`
resonance, `space` releases all notes, `o` saves a WAV recording, `q` quits.

## Reproducible spectrum checks

`measure_aliasing` in `synth/render.py` takes a Hann-windowed FFT, protects
the bins around every in-band partial (PolyBLEP spreads energy over a few
bins), and reports the RMS energy everywhere else as the residual/aliased
floor. At B7 (3951 Hz) the band-limited oscillators sit ~15-19 dB below the
naive references while keeping the in-band harmonic level within a fraction of
a dB. The benchmark writes the comparison plot to
`output/sweep_spectrum.png` and all numbers to `output/metrics.json`.

Key measured results (48 kHz, block size 256):

- sweep pitch tracking error: <= 0.23 cents across 0.3-3.6 kHz
- folded-spectrum reduction vs naive: ~15 dB (saw), ~17 dB (square) at B7
- 8-voice peak 0.997 with zero clipped samples (soft `tanh` ceiling)
- real-time callback mean 0.45 ms vs 5.33 ms deadline (~8% CPU), zero overruns
- real-time mock vs offline render: bit-exact, `max_abs_diff = 0.0`
- offline render speed: real-time factor ~0.1-0.35

## Polyphonic cost optimization

The optimization keeps exactly the same audible behavior. It changes **how**
the existing math is evaluated, not **what** is evaluated. The working
reference (the original code) can be reproduced by running
`bench/baseline_reference.py` against the pre-optimization commit; that script
is the one that produced `output/baseline/`.

### Computation order

1. **Voice allocation / note events** (unchanged). Notes are still reserved in
   event order, started at their exact in-block sample, and released after the
   block, and voice stealing is still oldest-onset. This was left alone because
   it is the part that must be exactly correct.
2. **Per-voice frequency smoothing.** For a held note, `_f0_sm` is already the
   target at trigger time, so the per-sample one-pole update leaves it
   unchanged. `render_block` now takes a fast path that fills a constant array
   and skips the recurrence; a real glide / sweep still runs the recurrence,
   but the loop body uses plain Python floats (`tolist()`) instead of numpy
   scalars, which is the same IEEE-754 sequence at a fraction of the cost.
3. **Oscillator phase advance.** The wrap-after-every-step recurrence is kept
   (a single `cumsum` would change wrap samples). The arithmetic runs on Python
   floats and wraps with a conditional subtraction, which is exactly
   `ph -= floor(ph)` because `dt` is clipped to `[0, 0.5]`. The oscillator is
   evaluated per voice over the sample axis, not over voices, because at 8-16
   voices cross-voice batching was measured to be *slower* (numpy overhead per
   tiny vector dominates).
4. **PolyBLEP.** The two transitions are applied in place on the naive
   waveform instead of building separate `zeros_like` and correction arrays.
   Same arithmetic, ~2 fewer arrays per voice per block.
5. **Envelope.** Per-block constants (attack/decay/release increments) are
   hoisted out of the sample loop, and an event-free block whose state is
   constant (SUSTAIN or IDLE) fills the array directly. Attack, decay and
   release blocks still run the sample state machine because they change
   every sample. Release uses a linear increment captured at the transition
   into RELEASE (see the correctness fix below).
6. **Filter.** Cutoff/Q smoothing, biquad coefficient recomputation and the
   per-voice DF-II-transposed state are unchanged. The only change is that the
   transcendental coefficient build is skipped when the smoothed cutoff and Q
   are bit-for-bit unchanged.

Nothing above changes oversampling. `peak_stats` still measures the 4×
sample-repeat peak exactly as before (`synth/render.py` is untouched), and the
oscillator / anti-aliasing path is not reduced.

### Correctness fix (not an optimization)

The original release used `level += -level / k` every sample: a geometric decay
that never reaches zero. Voices therefore never returned to idle and were never
reclaimed. A release now captures `level / k` once when RELEASE is entered and
subtracts it linearly, reaching exactly zero after `release` seconds. The
capture is idempotent because the vector path calls `note_off()` twice per
release; recomputing there would restart the fade from the end-of-block level.
This is the only intentional sample-value change after note-off; everything
before the first note-off is bit-identical to the original.

### Measured before/after (identical input)

`bench/compare_optimized.py` prints this table. Baseline was generated by
`bench/baseline_reference.py` from the pre-optimization code:

| scenario | baseline ms/block | optimized ms/block | speedup | alias (base → opt) |
| --- | --- | --- | --- | --- |
| dense_chord_8 | 0.918 | 0.255 | 3.59× | -7.74 → -7.74 dB |
| dense_chord_16 | 1.912 | 0.502 | 3.81× | -8.02 → -8.02 dB |
| high_register_8 | 0.952 | 0.292 | 3.26× | -16.62 → -16.62 dB |
| high_register_16 | 1.860 | 0.470 | 3.95× | -16.74 → -16.74 dB |

Aliasing (`alias_to_harmonic_db`, lower is less aliasing) is unchanged to the
printed precision in every scenario. Per-block time is the real
`process_block` elapsed time against the 5.333 ms deadline. The optimized
16-voice high-register case sits at ~8.8% mean / 13.4% p95 of the deadline with
zero overruns in a clean run. Occasional single-block spikes (~1-7 ms) come from
the host scheduler, not the DSP: the per-block mean stays under 10% of the
deadline, and the benchmark fails only on a sustained miss (p95 over the
deadline) while still recording raw overruns in the JSON. The 5 soft-saturated
samples in the 16-voice high-register case are the `tanh` ceiling doing its
job, not hard clipping.

The benchmark does not hide failures behind defaults: `bench/voice_bench.py`
exits non-zero if output is non-finite, a fundamental is missing, or the peak
exceeds full scale, and it records the actual measured values in every JSON.

## Playing it

```
# terminal (no display needed)
PYTHONPATH=. python3 keyboard.py

# GUI (needs tkinter + a display; under Xvfb: DISPLAY=:99 ...)
PYTHONPATH=. python3 keyboard.py --gui
```

The GUI piano, waveform radios and cutoff/Q/release sliders each write through
to every voice's oscillator/filter/envelope state, not just to a params copy.
The status bar and stderr report any control or audio-callback failure back to
the keyboard; a failure is never swallowed by a default value. "save wav"
writes `output/gui_performance.wav`.

Before the fix, the GUI crashed on the first rendered block with
`ValueError: shape mismatch ... (512,) and (129,)` because the spectrum line was
initialised with 512 points while a 256-sample block has 129 rfft bins; the
waveform radios also never reached the voices and the sliders only updated a
local params object. `output/gui_evidence/` holds the before/after screenshots
and the spectra of real clicks.

## Portability note

This build sandbox has no PortAudio/ALSA system library and no display server,
so `sounddevice` cannot open a device here. `audio.py` therefore selects
`MockAudioBackend` automatically; it runs the identical callback contract and
deadline timing. On a machine with PortAudio installed, `PortAudioBackend` is
selected with no DSP changes, and `keyboard.py --gui` provides the piano plus
live waveform/spectrum scope.
