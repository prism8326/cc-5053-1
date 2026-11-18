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
        # Linear fade increment captured when RELEASE is entered.  A
        # multiplicative ``level -= level/k`` fade never reaches zero, so the
        # voice would never be reclaimed; a captured increment reaches zero
        # after exactly ``release`` seconds.
        self._rel_step = None

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
        self._rel_step = None

    def note_off(self) -> None:
        if self.state != IDLE:
            # The vector path applies the release inside render() at the exact
            # event sample and then calls note_off() again after the block;
            # recomputing the increment there would restart the fade from the
            # end-of-block level.  Only the transition into RELEASE sets it.
            if self.state != RELEASE:
                self._rel_step = self._release_increment(self.level)
            self.state = RELEASE

    def _release_increment(self, level: float) -> float:
        k = max(int(round(self.release * self.sr)), 1)
        return level / k

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
            if self._rel_step is None:
                self._rel_step = self._release_increment(self.level)
            self.level -= self._rel_step
            if self.level <= 0.0:
                self.level = 0.0
                self.state = IDLE
                self._rel_step = None
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
        rel_step = self._rel_step
        vel = float(velocity) if velocity is not None else getattr(self, "_vel", 1.0)

        # Fast paths for blocks with no event at all.  In SUSTAIN every sample
        # is exactly ``sustain * vel`` and in IDLE every sample is exactly 0,
        # so filling the array is bit-identical to running the state machine;
        # it just skips the per-sample branch chain for held/idle voices.
        if note_on_at is None and note_off_at is None:
            if state == SUSTAIN:
                sus = self.sustain * vel
                out.fill(sus)
                self.level = sus
                self._vel = vel
                return out
            if state == IDLE:
                out.fill(0.0)
                self.level = 0.0
                self._vel = vel
                return out

        # Hoist the per-sample-count divisions out of the loop.  They depend
        # only on attack/decay/release and the sample rate, so recomputing them
        # on every sample was pure overhead.  Python floats run the identical
        # IEEE-754 operations the old np.float64 scalars did, in the same
        # order, so the output is bit-identical.
        k_a = max(int(round(self.attack * self.sr)), 1)
        k_d = max(int(round(self.decay * self.sr)), 1)
        k_r = max(int(round(self.release * self.sr)), 1)
        step_a = vel / k_a
        sus_level = self.sustain * vel

        for i in range(n):
            if note_on_at is not None and i == note_on_at:
                state = ATTACK
                rel_step = None
            if note_off_at is not None and i == note_off_at and state != IDLE:
                state = RELEASE
                rel_step = level / k_r

            if state == IDLE:
                level = 0.0
            elif state == ATTACK:
                level += step_a
                if level >= vel:
                    level = vel
                    state = DECAY
            elif state == DECAY:
                level += (sus_level - vel) / k_d
                if level <= sus_level:
                    level = sus_level
                    state = SUSTAIN
            elif state == SUSTAIN:
                level = sus_level
            elif state == RELEASE:
                if rel_step is None:
                    rel_step = level / k_r
                level -= rel_step
                if level <= 0.0:
                    level = 0.0
                    state = IDLE
                    rel_step = None
            out[i] = level

        self.state = state
        self.level = level
        self._rel_step = rel_step
        self._vel = vel
        return out
