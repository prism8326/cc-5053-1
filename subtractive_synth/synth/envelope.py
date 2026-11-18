"""Sample-accurate ADSR envelope.

Times are always supplied in seconds; slopes are derived from the sample rate
so envelope behavior is identical regardless of how many samples a processing
block contains.  Note on/off events take effect at an explicit sample index,
never at the next block boundary.

Two implementations of the exact same state machine are provided:

* :meth:`ADSR.render_step` — scalar, one sample at a time (real-time path /
  reference),
* :meth:`ADSR.render` — vectorized over a block with an event index (offline
  path).

The test suite checks they return bit-identical output.
"""

from __future__ import annotations

import numpy as np

IDLE, ATTACK, DECAY, SUSTAIN, RELEASE = range(5)

_STATE_NAMES = {
    IDLE: "idle", ATTACK: "attack", DECAY: "decay",
    SUSTAIN: "sustain", RELEASE: "release",
}


class ADSR:
    def __init__(self, attack=0.005, decay=0.15, sustain=0.7, release=0.25,
                 sr: float = 48000.0):
        self.attack = max(float(attack), 0.0)
        self.decay = max(float(decay), 0.0)
        self.sustain = float(sustain)
        self.release = max(float(release), 0.0)
        self.sr = float(sr)
        self.state = IDLE
        self.level = 0.0

    def set_times(self, attack=None, decay=None, sustain=None, release=None) -> None:
        if attack is not None:
            self.attack = max(float(attack), 0.0)
        if decay is not None:
            self.decay = max(float(decay), 0.0)
        if sustain is not None:
            self.sustain = float(sustain)
        if release is not None:
            self.release = max(float(release), 0.0)

    def note_on(self, velocity: float = 1.0) -> None:
        # Retriggering starts from the current level, which avoids a click
        # when a voice is stolen while still ringing.
        self._vel = float(np.clip(velocity, 0.0, 1.0))
        self.state = ATTACK

    def note_off(self) -> None:
        if self.state != IDLE:
            self.state = RELEASE

    @property
    def active(self) -> bool:
        return self.state != IDLE

    # ------------------------------------------------------------------ #
    # Scalar path
    # ------------------------------------------------------------------ #
    def _slope(self, delta: float, n: int) -> float:
        return delta / max(n, 1)

    def render_step(self) -> float:
        s = self.state
        if s == IDLE:
            self.level = 0.0
        elif s == ATTACK:
            n = max(int(round(self.attack * self.sr)), 1)
            self.level += self._slope(self._vel - 0.0, n)
            if self.level >= self._vel:
                self.level = self._vel
                self.state = DECAY
        elif s == DECAY:
            target = self.sustain * self._vel
            n = max(int(round(self.decay * self.sr)), 1)
            self.level += self._slope(target - self._vel, n)
            if self.level <= target:
                self.level = target
                self.state = SUSTAIN
        elif s == SUSTAIN:
            self.level = self.sustain * self._vel
        elif s == RELEASE:
            n = max(int(round(self.release * self.sr)), 1)
            self.level += self._slope(0.0 - self.level, n)
            if self.level <= 0.0:
                self.level = 0.0
                self.state = IDLE
        return self.level

    # ------------------------------------------------------------------ #
    # Vector path with sample-accurate events
    # ------------------------------------------------------------------ #
    def render(self, n: int, note_on_at: int | None = None,
               note_off_at: int | None = None,
               velocity: float | None = None) -> np.ndarray:
        """Render ``n`` samples; events fire at sample offsets inside the block.

        ``note_on_at`` / ``note_off_at`` are offsets in [0, n] (or None).  An
        event at offset ``k`` affects sample ``k`` onward — no block quantization.
        A note-on adopts ``velocity`` when given, otherwise the velocity stored
        by the preceding :meth:`note_on` / reservation.
        """
        out = np.empty(n, dtype=np.float64)
        state = self.state
        level = self.level
        vel = float(velocity) if velocity is not None else getattr(self, "_vel", 1.0)

        for i in range(n):
            if note_on_at is not None and i == note_on_at:
                state = ATTACK
            if note_off_at is not None and i == note_off_at and state != IDLE:
                state = RELEASE

            if state == IDLE:
                level = 0.0
            elif state == ATTACK:
                k = max(int(round(self.attack * self.sr)), 1)
                level += vel / k
                if level >= vel:
                    level = vel
                    state = DECAY
            elif state == DECAY:
                target = self.sustain * vel
                k = max(int(round(self.decay * self.sr)), 1)
                level += (target - vel) / k
                if level <= target:
                    level = target
                    state = SUSTAIN
            elif state == SUSTAIN:
                level = self.sustain * vel
            elif state == RELEASE:
                k = max(int(round(self.release * self.sr)), 1)
                level += -level / k
                if level <= 0.0:
                    level = 0.0
                    state = IDLE
            out[i] = level

        self.state = state
        self.level = level
        self._vel = vel
        return out
