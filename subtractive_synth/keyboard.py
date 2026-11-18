"""Playable keyboard front-end.

Two interchangeable UIs drive the same :class:`~synth.engine.SynthEngine`:

* :func:`run_terminal` — stdin single-key playing, works over SSH / in this
  sandbox where no display server or PortAudio library exists.  Audio is still
  synthesized in real-time-sized blocks through the mock backend; pressing ``r``
  writes what you played to a WAV so it can be auditioned elsewhere.
* :func:`run_gui` — tkinter piano + parameter sliders + live scope/spectrum,
  used automatically when a display and tkinter are available.
"""

from __future__ import annotations

import os
import sys
import termios
import tty

import numpy as np

from synth import SynthEngine, SynthParams, SAMPLE_RATE, BLOCK_SIZE
from synth.audio import make_backend, MockAudioBackend
from synth.render import save_wav

# Computer keyboard -> MIDI note (bottom row = white keys, upper row = sharps)
KEY_TO_NOTE = {
    "z": 60, "s": 61, "x": 62, "d": 63, "c": 64, "v": 65, "g": 66,
    "b": 67, "h": 68, "n": 69, "j": 70, "m": 71, ",": 72, "l": 73,
    ".": 74, ";": 75, "/": 76,
    "q": 72, "2": 73, "w": 74, "3": 75, "e": 76, "r": 77, "5": 78,
    "t": 79, "6": 80, "y": 81, "7": 82, "u": 83, "i": 84,
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


class _KeyReader:
    """Raw single-key stdin reader with proper terminal restoration.

    Falls back to plain buffered reads when stdin is not a TTY (piped script
    input), so the same UI can be exercised non-interactively.
    """

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._tty = os.isatty(self._fd)
        self._old = termios.tcgetattr(self._fd) if self._tty else None

    def __enter__(self):
        if self._tty:
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._tty:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get(self) -> str:
        ch = sys.stdin.read(1)
        return ch.lower() if ch else "q"


def run_terminal(out_dir: str, sr: float = SAMPLE_RATE) -> None:
    params = SynthParams()
    engine = SynthEngine(params, sr=sr, block_size=BLOCK_SIZE)
    backend = make_backend(engine, sr, BLOCK_SIZE)
    recording: list[np.ndarray] = []
    backend.start()

    help_text = (
        "\n=== Subtractive synth (terminal) ===\n"
        "Play: bottom row Z..M (white keys) and S D G H J L (sharps), "
        "Q..I jumps an octave up\n"
        "Commands:\n"
        "  1/2/3  waveform: saw / square / triangle\n"
        "  [ / ]  filter cutoff down / up    - / = resonance down / up\n"
        "  space  release all held notes      o  save recording to WAV\n"
        "  ?      status                      q  quit\n"
    )
    print(help_text)
    pressed: dict[str, int] = {}
    try:
        with _KeyReader() as reader:
            while True:
                ch = reader.get()
                if ch == "q":
                    break
                if ch in KEY_TO_NOTE:
                    note = KEY_TO_NOTE[ch]
                    if ch in pressed:
                        # Repeated strike: retrigger cleanly instead of ignoring.
                        engine.note_off(note)
                    pressed[ch] = note
                    engine.note_on(note, 0.9)
                    block = backend.pull(record=isinstance(backend, MockAudioBackend))
                    if isinstance(backend, MockAudioBackend):
                        recording.append(block)
                    print(f"\rON  {note_name(note)} ({note})      ", end="", flush=True)
                    continue
                if ch == "o":
                    if isinstance(backend, MockAudioBackend) and recording:
                        audio = np.concatenate(recording)
                        path = os.path.join(out_dir, "terminal_performance.wav")
                        save_wav(path, audio, sr)
                        print(f"\rsaved {path} ({audio.size / sr:.3f} s)      ",
                              end="", flush=True)
                    continue
                # key release isn't detectable in cbreak; an explicit space
                # releases every held note
                if ch == " ":
                    for note in list(pressed.values()):
                        engine.note_off(note)
                    pressed.clear()
                    print("\rall notes released      ", end="", flush=True)
                elif ch == "1":
                    _set_waveform(engine, params, "saw")
                    print("\rwaveform = saw         ", end="", flush=True)
                elif ch == "2":
                    _set_waveform(engine, params, "square")
                    print("\rwaveform = square      ", end="", flush=True)
                elif ch == "3":
                    _set_waveform(engine, params, "triangle")
                    print("\rwaveform = triangle    ", end="", flush=True)
                elif ch == "[":
                    _set_cutoff(engine, params, params.cutoff * 0.8)
                    print(f"\rcutoff = {params.cutoff:.0f} Hz      ", end="", flush=True)
                elif ch == "]":
                    _set_cutoff(engine, params, params.cutoff * 1.25)
                    print(f"\rcutoff = {params.cutoff:.0f} Hz      ", end="", flush=True)
                elif ch == "-":
                    _set_resonance(engine, params, params.resonance * 0.85)
                    print(f"\rQ = {params.resonance:.2f}            ", end="", flush=True)
                elif ch == "=":
                    _set_resonance(engine, params, params.resonance * 1.18)
                    print(f"\rQ = {params.resonance:.2f}            ", end="", flush=True)
                elif ch == ",":
                    pass  # comma is also a note key; commands avoid overlap
                elif ch == "?":
                    print("\n" + _status(engine, params, backend))
    finally:
        backend.stop()
        print("\nSession ended. " + backend.timing_report().__repr__())


# ---------------------------------------------------------------------- #
# Parameter pushes: write both the shared params object *and* every voice's
# own DSP state.  The voices never read ``params`` after construction, so a
# control that only updated ``params`` would look right in the status line
# while the kernel kept rendering the old value.
# ---------------------------------------------------------------------- #
def _set_waveform(engine: SynthEngine, params: SynthParams, waveform: str) -> None:
    params.waveform = waveform
    params.validate(engine.sr)
    for v in engine.voices:
        v.osc.waveform = waveform


def _set_cutoff(engine: SynthEngine, params: SynthParams, cutoff_hz: float) -> None:
    cutoff_hz = float(min(max(cutoff_hz, 1.0), engine.sr * 0.499))
    params.cutoff = cutoff_hz
    for v in engine.voices:
        v.filt.set_params(cutoff=cutoff_hz)


def _set_resonance(engine: SynthEngine, params: SynthParams, q: float) -> None:
    q = float(min(max(q, 1e-4), 40.0))
    params.resonance = q
    for v in engine.voices:
        v.filt.set_params(resonance=q)


def _set_release(engine: SynthEngine, params: SynthParams, seconds: float) -> None:
    seconds = max(float(seconds), 0.0)
    params.release = seconds
    for v in engine.voices:
        v.env.set_times(release=seconds)


def _refresh_voices(engine: SynthEngine, params: SynthParams) -> None:
    _set_waveform(engine, params, params.waveform)


def _status(engine, params, backend) -> str:
    active = [note_name(n) for n in engine._note_map]
    return (f"wave={params.waveform} cutoff={params.cutoff:.0f} Hz "
            f"Q={params.resonance:.2f} release={params.release:.2f}s "
            f"active={active or 'none'}\n"
            f"timing={backend.timing_report()}")


def run_gui(out_dir: str, sr: float = SAMPLE_RATE) -> None:
    import tkinter as tk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    params = SynthParams()
    engine = SynthEngine(params, sr=sr, block_size=BLOCK_SIZE)
    backend = make_backend(engine, sr, BLOCK_SIZE)
    backend.start()

    root = tk.Tk()
    root.title("Subtractive Synth")
    scope = np.zeros(BLOCK_SIZE * 4)
    spec_bins = BLOCK_SIZE // 2 + 1
    spec_freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / sr)

    fig = Figure(figsize=(7, 3.2))
    ax_wave = fig.add_subplot(2, 1, 1)
    ax_spec = fig.add_subplot(2, 1, 2)
    line, = ax_wave.plot(scope, lw=0.7)
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_title("waveform")
    spec_line, = ax_spec.plot(spec_freqs, np.full(spec_bins, -100.0), lw=0.7)
    ax_spec.set_title("spectrum (dB)")
    ax_spec.set_ylim(-100, 5)
    ax_spec.set_xlim(0, sr / 2)
    ax_spec.set_xlabel("Hz")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    panel = tk.Frame(root)
    panel.pack(side=tk.BOTTOM, fill=tk.X)

    # Every failure is reported back here *and* printed: the control panel is
    # part of the keyboard, so a kernel exception must be visible next to the
    # key that caused it instead of silently killing the update loop.
    status = tk.StringVar(value="ready")
    status_label = tk.Label(root, textvariable=status, anchor="w",
                            fg="#0a0", bg="#111", font=("TkFixedFont", 9))
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    failures: list[str] = []

    def report(kind: str, ok: bool = True) -> None:
        if ok:
            status.set(f"{kind}  |  wave={params.waveform} "
                       f"cutoff={params.cutoff:.0f}Hz Q={params.resonance:.2f} "
                       f"release={params.release:.3f}s gain={params.gain:.2f}")
        else:
            status_label.configure(fg="#f33")
            failures.append(kind)
            print(f"[keyboard] {kind}")

    def guard(fn):
        def wrapped(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                report(f"{type(exc).__name__}: {exc}", ok=False)
        return wrapped

    wave_var = tk.StringVar(value=params.waveform)

    @guard
    def on_wave():
        _set_waveform(engine, params, wave_var.get())
        report("waveform")

    for i, w in enumerate(("saw", "square", "triangle")):
        tk.Radiobutton(panel, text=w, value=w, variable=wave_var,
                       command=on_wave).grid(row=0, column=i, padx=4)

    def add_scale(label, key, lo, hi, row, log=False):
        var = tk.DoubleVar(value=getattr(params, key))

        @guard
        def on_change(val, key=key, lo=lo, hi=hi, log=log):
            val = float(val)
            if log:
                val = lo * (hi / lo) ** val
            if key == "cutoff":
                _set_cutoff(engine, params, val)
            elif key == "resonance":
                _set_resonance(engine, params, val)
            elif key == "release":
                _set_release(engine, params, val)
            report(key)

        sc = tk.Scale(panel, label=label, from_=0, to=1, resolution=0.001,
                      orient=tk.HORIZONTAL, length=160, variable=var,
                      command=on_change)
        sc.set((np.log(getattr(params, key) / lo) / np.log(hi / lo)) if log
               else getattr(params, key))
        sc.grid(row=row, column=4, padx=4)
        return sc

    add_scale("cutoff", "cutoff", 100, sr / 2.1, 0, log=True)
    add_scale("Q", "resonance", 0.1, 12, 1)
    add_scale("release(s)", "release", 0.01, 1.0, 2)

    held = set()

    @guard
    def press(note):
        if note not in held:
            held.add(note)
            engine.note_on(note, 0.9)
            report(f"on {note_name(note)}")

    @guard
    def release(note):
        held.discard(note)
        engine.note_off(note)
        report(f"off {note_name(note)}")

    keys_frame = tk.Frame(root)
    keys_frame.pack(fill=tk.X)
    for i, note in enumerate(range(60, 73)):
        btn = tk.Button(keys_frame, text=note_name(note), width=4, height=3,
                        bg="#222" if "#" in note_name(note) else "#eee",
                        fg="#eee" if "#" in note_name(note) else "#000")
        btn.grid(row=0, column=i, padx=1)
        btn.bind("<ButtonPress-1>", lambda e, n=note: press(n))
        btn.bind("<ButtonRelease-1>", lambda e, n=note: release(n))

    recorded: list[np.ndarray] = []

    @guard
    def save_perf():
        if not recorded:
            raise RuntimeError("nothing recorded yet")
        audio = np.concatenate(recorded)
        path = os.path.join(out_dir, "gui_performance.wav")
        save_wav(path, audio, sr)
        recorded.clear()
        report(f"saved {os.path.basename(path)} ({audio.size / sr:.2f}s)")

    tk.Button(panel, text="save wav", command=save_perf).grid(row=2, column=0)
    tk.Button(panel, text="release all",
              command=guard(lambda: [release(n) for n in list(held)])).grid(
        row=2, column=1)

    def tick():
        try:
            block = backend.pull()
        except Exception as exc:  # noqa: BLE001
            report(f"pull: {type(exc).__name__}: {exc}", ok=False)
            root.after(50, tick)
            return
        if isinstance(backend, MockAudioBackend):
            recorded.append(block.copy())
        scope[:-BLOCK_SIZE] = scope[BLOCK_SIZE:]
        scope[-BLOCK_SIZE:] = block
        line.set_ydata(scope)
        sp = 20 * np.log10(np.abs(np.fft.rfft(block * np.hanning(block.size)))
                           + 1e-9)
        spec_line.set_ydata(sp)
        canvas.draw_idle()
        root.after(20, tick)

    root.after(20, tick)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (backend.stop(), root.destroy()))
    try:
        root.mainloop()
    finally:
        backend.stop()
        print(f"[keyboard] backend={backend.name} "
              f"timing={backend.timing_report()}")
        if failures:
            print(f"[keyboard] {len(failures)} control failure(s): {failures}")


def main() -> None:
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    gui = "--gui" in sys.argv
    if gui:
        try:
            import tkinter  # noqa: F401
            if not os.environ.get("DISPLAY"):
                raise RuntimeError("no DISPLAY")
            run_gui(out_dir)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"GUI unavailable ({exc}); falling back to terminal.")
    run_terminal(out_dir)


if __name__ == "__main__":
    main()
