"""Polyphonic synth engine with separate per-voice state and two render paths.

Design rules required by the specification:

* Each :class:`Voice` owns its oscillator, envelope and filter state.  Ending a
  note only releases that voice; every other ringing voice is untouched.
* Polyphony is finite (``params.voices``).  When all slots are busy the oldest
  note (earliest onset) is retriggered — voice stealing, never a crash or a
  silently dropped note.
* Note on/off events are resolved at an explicit sample index inside a block,
  so note timing is independent of block boundaries.
* Two paths run the same DSP: a vectorized block path used both by the offline
  renderer and the real-time audio callback, and a scalar per-sample path used
  as an independently-written reference in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SynthParams, SAMPLE_RATE, BLOCK_SIZE
from .oscillator import Oscillator, midi_to_freq
from .envelope import ADSR, IDLE
from .filter import BiquadLowpass


@dataclass
class NoteEvent:
    """A MIDI note event at an absolute sample position."""
    time: float          # seconds
    note: int            # MIDI note number
    velocity: float      # 0..1; velocity 0 is equivalent to note-off
    kind: str = "on"     # "on" / "off"


class Voice:
    def __init__(self, params: SynthParams, sr: float):
        self.sr = sr
        self.params = params
        self.osc = Oscillator(params.waveform, sr)
        self.env = ADSR(params.attack, params.decay, params.sustain,
                        params.release, sr)
        self.filt = BiquadLowpass(sr, params.cutoff, params.resonance,
                                  params.smooth_time)
        self.note: int | None = None
        self.velocity = 1.0
        self.onset_sample = -1
        self._f0_sm = midi_to_freq(60)
        self._f0_a = 1.0
        self._has_note = False

    # --------------------------------------------------------------- #
    # Vector block path
    # --------------------------------------------------------------- #
    def render_block(self, n: int, note_on_at: int | None,
                     note_off_at: int | None,
                     f0_automation: np.ndarray | None) -> np.ndarray:
        p = self.params

        # Target frequency per sample (fixed note or a recorded sweep).
        if self._has_note:
            target = midi_to_freq(self.note)
        else:
            target = self._f0_sm
        f0_target = np.full(n, target) if f0_automation is None else f0_automation

        # One-pole glide/smoothing, block-constant coefficient: identical math
        # in the scalar reference path.
        if p.smooth_time > 0.0:
            a = 1.0 - np.exp(-n / (p.smooth_time * self.sr))
        else:
            a = 1.0
        f0 = np.empty(n)
        sm = self._f0_sm
        for i in range(n):
            sm += a * (f0_target[i] - sm)
            f0[i] = sm
        self._f0_sm = sm

        x = self.osc.render(f0)
        env = self.env.render(n, note_on_at, note_off_at,
                              velocity=self.velocity if note_on_at is not None else None)
        y = self.filt.process(x * env)
        return y * self.velocity

    # --------------------------------------------------------------- #
    # Scalar reference path
    # --------------------------------------------------------------- #
    def prepare_scalar_block(self, n: int) -> None:
        # Block-constant coefficient updates for the filter and f0 smoother.
        self.filt.prepare_block(n)
        if self.params.smooth_time > 0.0:
            self._f0_a = 1.0 - np.exp(-n / (self.params.smooth_time * self.sr))
        else:
            self._f0_a = 1.0

    def render_scalar_step(self, f0_target: float) -> float:
        self._f0_sm += self._f0_a * (f0_target - self._f0_sm)
        x = self.osc.step(self._f0_sm)
        e = self.env.render_step()
        y = self.filt.process_step(x * e)
        return y * self.velocity

    def trigger(self, note: int, velocity: float, sample_index: int) -> None:
        self.note = note
        self.velocity = float(np.clip(velocity, 0.0, 1.0))
        self.onset_sample = sample_index
        self._has_note = True
        self._f0_sm = midi_to_freq(note)
        self.osc.reset()
        self.env.note_on(self.velocity)

    def release(self) -> None:
        self.env.note_off()

    @property
    def idle(self) -> bool:
        return self.env.state == IDLE

    def reclaim(self) -> None:
        """Full reset when a voice is reused or fully finished."""
        self.env.state = IDLE
        self.env.level = 0.0
        self.osc.reset()
        self.filt.reset()
        self.note = None
        self._has_note = False


class SynthEngine:
    """Vectorized block engine shared by offline rendering and live audio."""

    def __init__(self, params: SynthParams | None = None,
                 sr: float = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.params = params or SynthParams()
        self.params.validate(sr)
        self.sr = float(sr)
        self.block_size = int(block_size)
        self.voices = [Voice(self.params, self.sr) for _ in range(self.params.voices)]
        # note -> voice, so a note-off finds the exact sounding instance
        self._note_map: dict[int, Voice] = {}
        self._onset_counter = 0
        self._gain_sm = self.params.gain

    # ------------------------------------------------------------------ #
    # Voice allocation
    # ------------------------------------------------------------------ #
    def _reserve(self, note: int, velocity: float, sample_index: int) -> Voice:
        """Claim a voice slot *without* starting the envelope.

        The envelope starts inside :meth:`render_block` at the exact in-block
        event offset.  Activating it here would make every later note in the
        same block see the slot as busy and pile onto one voice via stealing.
        """
        if note in self._note_map:
            return self._note_map[note]
        for v in self.voices:
            if v.note is None:
                break
        else:
            v = min(self.voices, key=lambda x: x.onset_sample)
            self._note_map.pop(v.note, None)
        v.note = note
        v.velocity = float(np.clip(velocity, 0.0, 1.0))
        v.onset_sample = sample_index
        v._has_note = True
        v._f0_sm = midi_to_freq(note)
        v.osc.reset()
        v.env.state = IDLE
        v.env.level = 0.0
        v.env._vel = v.velocity
        self._note_map[note] = v
        self._onset_counter += 1
        v.onset_sample = self._onset_counter
        return v

    def _allocate(self, note: int, sample_index: int,
                  velocity: float = 1.0) -> Voice:
        # Reuse a slot already holding this note (retrigger).
        if note in self._note_map:
            v = self._note_map[note]
            v.trigger(note, velocity, sample_index)
            return v
        v = self._reserve(note, velocity, sample_index)
        v.env.note_on(v.velocity)
        return v

    def _release_note(self, note: int) -> None:
        v = self._note_map.pop(note, None)
        if v is not None:
            v.release()

    # ------------------------------------------------------------------ #
    # Block processing
    # ------------------------------------------------------------------ #
    def process_block(self, n: int, events: list[NoteEvent] | None = None,
                      block_origin: int = 0,
                      f0_automation: dict[int, np.ndarray] | None = None
                      ) -> np.ndarray:
        """Render one block of ``n`` samples.

        ``events`` are absolute-time events clipped to this block; internally
        they are converted to sample offsets.  ``f0_automation`` maps MIDI note
        to a length-n per-sample Hz array (used by the sweep test).
        """
        events = sorted(events or [], key=lambda e: e.time)
        # events_at[offset] -> list of (note, kind, velocity)
        events_at: dict[int, list[tuple[int, str, float]]] = {}
        for ev in events:
            idx = int(round(ev.time * self.sr)) - block_origin
            if 0 <= idx < n:
                events_at.setdefault(idx, []).append((ev.note, ev.kind, ev.velocity))

        # Reserve slots in event order while every envelope stays idle, so
        # multiple notes in one block get distinct voices.  The envelope is
        # started by render_block at the exact in-block event offset.
        first_ons: dict[int, int] = {}
        first_offs: dict[int, int] = {}
        for idx in sorted(events_at):
            for note, kind, vel in events_at[idx]:
                if kind == "on" and vel > 0:
                    first_ons.setdefault(note, idx)
                    if note not in self._note_map:
                        self._reserve(note, vel, block_origin + idx)
                else:
                    first_offs.setdefault(note, idx)

        out = np.zeros(n)
        for v in self.voices:
            if v.note is None:
                continue
            on_at = first_ons.get(v.note)
            off_at = first_offs.get(v.note)
            auto = None
            if f0_automation and v.note in f0_automation:
                auto = f0_automation[v.note]
            out += v.render_block(n, on_at, off_at, auto)

        # Releases take effect after rendering (envelope itself applies the
        # release slope starting at the exact event sample).
        for note, idx in first_offs.items():
            if note in self._note_map:
                self._release_note(note)

        # Master gain with block-constant one-pole smoothing (no jumps).
        if self.params.smooth_time > 0.0:
            a = 1.0 - np.exp(-n / (self.params.smooth_time * self.sr))
        else:
            a = 1.0
        self._gain_sm += a * (self.params.gain - self._gain_sm)
        out *= self._gain_sm
        # Soft ceiling: N simultaneous voices sum beyond +/-1; tanh keeps the
        # peak bounded without hard clipping at a sample.
        out = np.tanh(out)

        # Reclaim voices whose release has fully decayed.  Idle slots that
        # have never held a note (note is None) are simply skipped, so the
        # bookkeeping of other sounding voices is never touched.
        for v in self.voices:
            if v.note is not None and v.env.state == IDLE and v.env.level == 0.0:
                self._note_map.pop(v.note, None)
                v.reclaim()
        return out

    # ------------------------------------------------------------------ #
    # Offline rendering
    # ------------------------------------------------------------------ #
    def render_events(self, events: list[NoteEvent], duration: float,
                      f0_automation: dict[int, np.ndarray] | None = None
                      ) -> tuple[np.ndarray, float]:
        """Render an event list to a mono float64 array. Returns (audio, seconds)."""
        total = int(round(duration * self.sr))
        out = np.zeros(total)
        ev_sorted = sorted(events, key=lambda e: e.time)
        for start in range(0, total, self.block_size):
            n = min(self.block_size, total - start)
            t0, t1 = start / self.sr, (start + n) / self.sr
            block_events = [e for e in ev_sorted if t0 <= e.time < t1]
            auto = None
            if f0_automation:
                auto = {note: arr[start:start + n] for note, arr in f0_automation.items()}
            out[start:start + n] = self.process_block(n, block_events, start, auto)
        return out, total / self.sr

    # ------------------------------------------------------------------ #
    # Live control (called from an audio thread or a mock backend)
    # ------------------------------------------------------------------ #
    def note_on(self, note: int, velocity: float = 1.0) -> None:
        self._pending = getattr(self, "_pending", [])
        self._pending.append(("on", int(note), float(velocity)))

    def note_off(self, note: int) -> None:
        self._pending = getattr(self, "_pending", [])
        self._pending.append(("off", int(note), 0.0))

    def pull_block(self, n: int | None = None) -> np.ndarray:
        """Audio-callback style pull; queued events land at sample offset 0."""
        n = n or self.block_size
        pending = getattr(self, "_pending", [])
        events = [
            NoteEvent(0.0, note, vel, "on" if cmd == "on" else "off")
            for cmd, note, vel in pending
        ]
        self._pending = []
        return self.process_block(n, events, 0)


class ScalarEngine:
    """Per-sample reference implementation of the same architecture.

    Kept deliberately simple and independent of the vector path; the parity
    test drives both engines with identical schedules and compares samples.
    """

    def __init__(self, params: SynthParams | None = None, sr: float = SAMPLE_RATE):
        self.params = params or SynthParams()
        self.sr = float(sr)
        self.voices = [Voice(self.params, self.sr) for _ in range(self.params.voices)]
        self._note_map: dict[int, Voice] = {}
        self._gain_sm = self.params.gain
        self._block_size = BLOCK_SIZE

    def _reserve(self, note: int, velocity: float, sample_index: int) -> Voice:
        if note in self._note_map:
            return self._note_map[note]
        for v in self.voices:
            if v.note is None:
                break
        else:
            v = min(self.voices, key=lambda x: x.onset_sample)
            self._note_map.pop(v.note, None)
        v.note = note
        v.velocity = float(np.clip(velocity, 0.0, 1.0))
        v.onset_sample = sample_index
        v._has_note = True
        v._f0_sm = midi_to_freq(note)
        v.osc.reset()
        v.env.state = IDLE
        v.env.level = 0.0
        v.env._vel = v.velocity
        self._note_map[note] = v
        return v

    def render_events(self, events: list[NoteEvent], duration: float) -> np.ndarray:
        total = int(round(duration * self.sr))
        on_at: dict[int, int] = {}
        off_at: dict[int, int] = {}
        vel_at: dict[int, float] = {}
        for ev in events:
            idx = int(round(ev.time * self.sr))
            if ev.kind == "on" and ev.velocity > 0:
                on_at[ev.note] = idx
                vel_at[ev.note] = ev.velocity
            else:
                off_at[ev.note] = idx

        out = np.zeros(total)
        block = self._block_size
        for start in range(0, total, block):
            n = min(block, total - start)
            end = start + n

            # Mirror the vector path: events are resolved to slot reservations
            # first, envelopes stay idle until their exact in-block sample.
            block_on = {note: idx - start for note, idx in on_at.items()
                        if start <= idx < end}
            block_off = {note: idx - start for note, idx in off_at.items()
                         if start <= idx < end}
            for note, off_idx in sorted(block_on.items(), key=lambda kv: kv[1]):
                if note not in self._note_map:
                    self._reserve(note, vel_at.get(note, 1.0), start + off_idx)

            # Only voices that will actually produce sound in this block get
            # the block-constant coefficient updates — same rule as the vector
            # path, which never touches idle slots.
            for v in self.voices:
                if v.note is not None:
                    v.prepare_scalar_block(n)

            for i in range(n):
                k = start + i
                for note, off_idx in block_on.items():
                    if off_idx == i:
                        v = self._note_map[note]
                        # Use the velocity stored at reservation time; calling
                        # note_on() with a fresh argument would drop it.
                        v.env.note_on(v.velocity)
                for note, off_idx in block_off.items():
                    if off_idx == i and note in self._note_map:
                        self._note_map[note].release()
                for v in self.voices:
                    if v.note is None:
                        continue
                    target = midi_to_freq(v.note)
                    out[k] += v.render_scalar_step(target)

            for note in block_off:
                self._note_map.pop(note, None)
            # Reclaim finished voices exactly like the vector path.
            for v in self.voices:
                if v.note is not None and v.env.state == IDLE and v.env.level == 0.0:
                    self._note_map.pop(v.note, None)
                    v.reclaim()

            if self.params.smooth_time > 0.0:
                a = 1.0 - np.exp(-n / (self.params.smooth_time * self.sr))
            else:
                a = 1.0
            self._gain_sm += a * (self.params.gain - self._gain_sm)
            out[start:end] = np.tanh(out[start:end] * self._gain_sm)
        return out
