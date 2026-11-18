"""Command-line offline renderer.

Produces a WAV plus a JSON sidecar with all numeric checks (pitch, aliasing,
peaks, render cost), so the offline output can be compared directly with a
real-time mock recording of the same event list (see parity.json from the
benchmark battery).

Examples
--------
Render a built-in demo phrase:
    python3 cli.py demo --waveform square --out output/phrase

Render an explicit MIDI-style event list (time,note,velocity; vel 0 = off):
    python3 cli.py render events.txt --out output/custom

Events file rows:  <seconds> <midi_note> <velocity>   (# comments allowed)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from synth import SynthParams, NoteEvent, SAMPLE_RATE, BLOCK_SIZE
from synth.config import WAVE_SAW, WAVE_SQUARE, WAVE_TRI
from synth.render import (
    render_offline, save_wav, measure_aliasing, estimate_fundamental,
    peak_stats,
)
from synth.oscillator import midi_to_freq


def parse_events(path: str) -> list[NoteEvent]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            t, note, vel = line.split()[:3]
            t, note, vel = float(t), int(note), float(vel)
            events.append(NoteEvent(t, note, vel,
                                    "on" if vel > 0 else "off"))
    return events


def demo_events() -> tuple[list[NoteEvent], float]:
    # A short C major arpeggio phrase with overlapping notes and a rest.
    seq = [
        (0.00, 60, 1.0), (0.00, 72, 0.0),
        (0.25, 60, 0.0), (0.25, 64, 1.0),
        (0.50, 64, 0.0), (0.50, 67, 1.0),
        (0.75, 67, 0.0), (0.75, 72, 1.0),
        (1.10, 72, 0.0), (1.20, 72, 0.9),
        (1.45, 76, 0.9), (1.70, 79, 0.9),
        (2.10, 79, 0.0), (2.10, 76, 0.0), (2.10, 72, 0.0),
    ]
    events = [NoteEvent(t, n, v, "on" if v > 0 else "off") for t, n, v in seq]
    return events, 2.8


def analyze(audio: np.ndarray, events: list[NoteEvent], sr: float) -> dict:
    ons = [e for e in events if e.kind == "on" and e.velocity > 0]
    first_note = ons[0].note
    # steady window well after the first attack
    start = int((ons[0].time + 0.15) * sr)
    seg = audio[start:start + 2 ** 15]
    pitch = estimate_fundamental(seg, midi_to_freq(first_note), sr)
    alias = measure_aliasing(seg, midi_to_freq(first_note), sr)
    return {
        "pitch": pitch,
        "aliasing": alias,
        "peak": peak_stats(audio, sr),
        "finite": bool(np.all(np.isfinite(audio))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline subtractive synth renderer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--waveform", choices=[WAVE_SAW, WAVE_SQUARE, WAVE_TRI],
                       default=WAVE_SAW)
        p.add_argument("--cutoff", type=float, default=4000.0)
        p.add_argument("--resonance", type=float, default=0.707)
        p.add_argument("--attack", type=float, default=0.005)
        p.add_argument("--decay", type=float, default=0.15)
        p.add_argument("--sustain", type=float, default=0.7)
        p.add_argument("--release", type=float, default=0.25)
        p.add_argument("--gain", type=float, default=0.8)
        p.add_argument("--voices", type=int, default=8)
        p.add_argument("--sr", type=float, default=SAMPLE_RATE)
        p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
        p.add_argument("--out", required=True, help="output path without extension")

    p_demo = sub.add_parser("demo", help="render the built-in phrase")
    add_common(p_demo)
    p_render = sub.add_parser("render", help="render an events file")
    add_common(p_render)
    p_render.add_argument("events_file")

    args = ap.parse_args()
    params = SynthParams(
        waveform=args.waveform, attack=args.attack, decay=args.decay,
        sustain=args.sustain, release=args.release, cutoff=args.cutoff,
        resonance=args.resonance, gain=args.gain, voices=args.voices,
    )
    if args.cmd == "demo":
        events, duration = demo_events()
    else:
        events = parse_events(args.events_file)
        duration = max(e.time for e in events) + args.release + 0.5

    audio, render_stats = render_offline(
        events, params, duration, args.sr, args.block_size)
    checks = analyze(audio, events, args.sr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wav_path = args.out + ".wav"
    json_path = args.out + ".json"
    save_wav(wav_path, audio, args.sr)
    report = {
        "params": vars(params),
        "sample_rate": args.sr,
        "block_size": args.block_size,
        "render": render_stats,
        "checks": checks,
        "wav": os.path.abspath(wav_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {wav_path} and {json_path}")


if __name__ == "__main__":
    main()
