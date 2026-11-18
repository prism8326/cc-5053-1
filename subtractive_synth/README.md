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
tests/           25 pytest checks
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

## Portability note

This build sandbox has no PortAudio/ALSA system library and no display server,
so `sounddevice` cannot open a device here. `audio.py` therefore selects
`MockAudioBackend` automatically; it runs the identical callback contract and
deadline timing. On a machine with PortAudio installed, `PortAudioBackend` is
selected with no DSP changes, and `keyboard.py --gui` provides the piano plus
live waveform/spectrum scope.
