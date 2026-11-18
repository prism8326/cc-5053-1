"""Band-limited oscillators built on a phase accumulator with PolyBLEP.

The naive ``2*phase - 1`` / sign-based waveforms contain ideal discontinuities
whose partials extend past Nyquist and fold back into the audible band.  This
module removes that aliasing with a piecewise polynomial band-limited step
(PolyBLEP, see Valimaki et al., "Discrete-Time Synthesis of the Sawtooth
Waveform", 2006).  The naive oscillators are kept on purpose so the test suite
can measure the aliasing difference instead of merely asserting "no clicks".

Phase lives in [0, 1); one period per unit phase.  All render functions accept
per-sample frequency arrays, so pitch sweeps stay correct at every sample.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .config import (
    WAVE_SAW, WAVE_SQUARE, WAVE_TRI, WAVE_NAIVE_SAW, WAVE_NAIVE_SQUARE,
    SAMPLE_RATE,
)


def midi_to_freq(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def naive_saw(phase: np.ndarray) -> np.ndarray:
    """Ideal discontinuous saw, full of aliasing — reference only."""
    return 2.0 * phase - 1.0


def naive_square(phase: np.ndarray) -> np.ndarray:
    """Ideal discontinuous square, full of aliasing — reference only."""
    return np.where(phase < 0.5, -1.0, 1.0)


def _polyblep(phase: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Band-limited step correction around phase 0 (and, via shifted input, 0.5)."""
    out = np.zeros_like(phase)
    dt = np.maximum(dt, 1e-12)

    left = phase < dt
    if np.any(left):
        x = phase[left] / dt[left]
        out[left] = 2.0 * x - x * x - 1.0

    right = phase > 1.0 - dt
    if np.any(right):
        x = (phase[right] - 1.0) / dt[right]
        out[right] = x * x + 2.0 * x + 1.0
    return out


def _polyblep_at_half(phase: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """BLEP correction for the transition at phase 0.5.

    Equivalent to ``_polyblep((phase + 0.5) % 1.0, dt)`` but written without
    the modular remap, so the branch each sample takes is explicit:

    * phase in [0.5, 0.5+dt)  -> shifted phase in [0, dt)      (left branch)
    * phase in [0.5-dt, 0.5)  -> shifted phase in [1-dt, 1)    (right branch)
    """
    out = np.zeros_like(phase)
    dt = np.maximum(dt, 1e-12)

    left = (phase >= 0.5) & (phase < 0.5 + dt)
    if np.any(left):
        x = (phase[left] - 0.5) / dt[left]
        out[left] = 2.0 * x - x * x - 1.0

    right = (phase >= 0.5 - dt) & (phase < 0.5)
    if np.any(right):
        x = (phase[right] - 0.5) / dt[right]  # shifted (u-1)/dt, u=phase+0.5
        out[right] = x * x + 2.0 * x + 1.0
    return out


def polyblep_saw(phase: np.ndarray, dt: np.ndarray) -> np.ndarray:
    # Written in-place rather than ``naive_saw - _polyblep(...)`` so the two
    # temporary arrays (zeros_like + naive result) disappear.  The arithmetic
    # at every touched sample is the same subtraction, so output is identical.
    out = 2.0 * phase - 1.0
    dt = np.maximum(dt, 1e-12)

    left = phase < dt
    if np.any(left):
        x = phase[left] / dt[left]
        out[left] -= 2.0 * x - x * x - 1.0

    right = phase > 1.0 - dt
    if np.any(right):
        x = (phase[right] - 1.0) / dt[right]
        out[right] -= x * x + 2.0 * x + 1.0
    return out


def polyblep_square(phase: np.ndarray, dt: np.ndarray) -> np.ndarray:
    # The square has two opposite-direction jumps per period.  The -1 -> +1
    # jump at phase 0 needs the subtractive BLEP (same as the saw), while the
    # +1 -> -1 jump at phase 0.5 needs the additive one.  Any other sign
    # combination doubles the signal level or leaves the folded residual.
    out = np.where(phase < 0.5, -1.0, 1.0)
    dt = np.maximum(dt, 1e-12)

    left = phase < dt
    if np.any(left):
        x = phase[left] / dt[left]
        out[left] -= 2.0 * x - x * x - 1.0

    right = phase > 1.0 - dt
    if np.any(right):
        x = (phase[right] - 1.0) / dt[right]
        out[right] -= x * x + 2.0 * x + 1.0

    left_h = (phase >= 0.5) & (phase < 0.5 + dt)
    if np.any(left_h):
        x = (phase[left_h] - 0.5) / dt[left_h]
        out[left_h] += 2.0 * x - x * x - 1.0

    right_h = (phase >= 0.5 - dt) & (phase < 0.5)
    if np.any(right_h):
        x = (phase[right_h] - 0.5) / dt[right_h]
        out[right_h] += x * x + 2.0 * x + 1.0
    return out


# Leak corner for the triangle integrator: far below any playable note, it only
# stops the accumulator from drifting away on rounding error.
_TRI_LEAK_HZ = 20.0


class Oscillator:
    """Stateful oscillator: one phase accumulator (+ integrator for triangle).

    Use :meth:`render` on per-sample frequency arrays (offline path), or
    :meth:`step` sample by sample (scalar reference path).  Both run the same
    math and must agree to floating-point tolerance.
    """

    def __init__(self, waveform: str = WAVE_SAW, sr: float = SAMPLE_RATE):
        self.waveform = waveform
        self.sr = sr
        self.phase = 0.0
        self._tri_state = 0.0
        self._tri_g = 1.0 / (1.0 + 2.0 * np.pi * _TRI_LEAK_HZ / sr)

    def reset(self) -> None:
        self.phase = 0.0
        self._tri_state = 0.0

    def _advance(self, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = np.clip(f0 / self.sr, 0.0, 0.5)
        # Wrap after every step, exactly like the scalar path: a single cumsum
        # followed by one wrap would accumulate phase rounding error and land
        # on the other side of a discontinuity at wrap samples.
        #
        # The recurrence runs on Python floats (``tolist``) rather than numpy
        # scalars: the arithmetic is the identical IEEE-754 sequence, but
        # avoiding a numpy scalar per add/wrap is ~3x faster.  ``dt`` is clipped
        # to [0, 0.5] and phase stays in [0, 1), so at most one wrap can occur
        # per step; the single conditional subtraction is exactly
        # ``ph -= floor(ph)`` here, and it preserves wrap-after-every-step.
        steps = dt.tolist()
        phase = [0.0] * len(steps)
        ph = self.phase
        for i, step_dt in enumerate(steps):
            ph += step_dt
            if ph >= 1.0:
                ph -= 1.0
            phase[i] = ph
        self.phase = ph
        return np.asarray(phase), dt

    def render(self, f0: np.ndarray) -> np.ndarray:
        """Render one block; ``f0`` is a per-sample frequency array in Hz."""
        f0 = np.asarray(f0, dtype=np.float64)
        phase, dt = self._advance(f0)

        if self.waveform == WAVE_NAIVE_SAW:
            return naive_saw(phase)
        if self.waveform == WAVE_NAIVE_SQUARE:
            return naive_square(phase)
        if self.waveform == WAVE_SAW:
            return polyblep_saw(phase, dt)
        if self.waveform == WAVE_SQUARE:
            return polyblep_square(phase, dt)
        if self.waveform == WAVE_TRI:
            square = polyblep_square(phase, dt)
            return self._integrate_triangle(square, f0)
        raise ValueError(f"unknown waveform {self.waveform!r}")

    def _integrate_triangle(self, square: np.ndarray, f0: np.ndarray) -> np.ndarray:
        # Leaky integration of the band-limited square yields a band-limited
        # triangle (integration turns the square's odd 1/n spectrum into 1/n^2).
        c = 2.0 * np.pi * f0 / self.sr
        u = c * square
        y, zf = lfilter([1.0], [1.0, -self._tri_g], u, zi=[self._tri_state])
        self._tri_state = float(zf[0])
        return (2.0 / np.pi) * y

    def step(self, f0: float) -> float:
        """Scalar reference implementation, one sample at a time.

        Phase is advanced first and the output is evaluated at the resulting
        phase, matching :meth:`render` exactly (which accumulates phase before
        evaluating the waveform).
        """
        dt = min(max(f0 / self.sr, 0.0), 0.5)
        total = self.phase + dt
        self.phase = total - np.floor(total)
        phase = self.phase

        if self.waveform == WAVE_NAIVE_SAW:
            return 2.0 * phase - 1.0
        if self.waveform == WAVE_NAIVE_SQUARE:
            return -1.0 if phase < 0.5 else 1.0

        dta = np.array([dt])
        pa = np.array([phase])
        if self.waveform == WAVE_SAW:
            return float(polyblep_saw(pa, dta)[0])
        if self.waveform == WAVE_SQUARE:
            return float(polyblep_square(pa, dta)[0])
        if self.waveform == WAVE_TRI:
            sq = float(polyblep_square(pa, dta)[0])
            self._tri_state = self._tri_g * self._tri_state + (2.0 * np.pi * f0 / self.sr) * sq
            return (2.0 / np.pi) * self._tri_state
        raise ValueError(f"unknown waveform {self.waveform!r}")
