# GUI performance evidence

Captured by actually clicking the tkinter piano in `keyboard.py --gui` under
Xvfb (`xdotool` real mouse events), not by calling the engine directly.

## Before the fix (`before_spectrum_shape_crash.png`)

The original GUI raised on the first rendered block:

```
ValueError: shape mismatch: objects cannot be broadcast to a single shape.
Mismatch is between arg 0 with shape (512,) and arg 1 with shape (129,).
```

(1008+ repeats in the log; the window shows a blank spectrum.) For the same
reason the waveform radios never updated `params.waveform` and the sliders only
wrote to a local params object, so neither reached the voice kernel.

## After the fix (`gui_after_*.png`, `gui_*.wav`, `spectra_gui.png`)

Same C4 key pressed with a real mouse click, three waveform selections set
through the radio buttons. All three produce sound and a live spectrum, and the
status bar reflects the change (`wave=saw`, `wave=square`, `wave=triangle`).

Measured from the saved GUI WAVs (`gui_spectra.json`):

| waveform | measured peak Hz | even harmonics | odd harmonics |
| --- | --- | --- | --- |
| saw | 261.1 | present (-18.3 dB at 2f) | present |
| square | 260.9 | absent (2f -73.3 dB) | present |
| triangle | 261.1 | absent (2f -62.0 dB) | present, 1/n² decay |

C4 = 261.63 Hz, so the pitch is correct. Square/triangle even-harmonic
suppression is the expected signature and confirms the radio buttons reached
the oscillator. Cutoff and release sliders were also exercised: dragging cutoff
to ~153 Hz audibly and spectrally mutes the harmonics of a 261 Hz note.

Reproduce:

```
DISPLAY=:99 python3 keyboard.py --gui     # Xvfb + tkinter needed
xdotool mousemove 31 363 mousedown 1      # press C4
xdotool mousemove 45 569 click 1          # "save wav"
```
