"""RBJ biquad low-pass filter (Cookbook formulae), one state per voice.

Filter state (the two delay lines) is stored inside every voice instance, so
stopping one voice never touches another voice's filter.  Cutoff/resonance are
smoothed with an explicit one-pole (time constant in seconds) and coefficients
are recomputed once per processing block; both the scalar real-time path and
the vector offline path use identical coefficient values at every block, which
keeps their outputs bit-identical.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class BiquadLowpass:
    def __init__(self, sr: float = 48000.0, cutoff: float = 4000.0,
                 resonance: float = 0.70710678, smooth_time: float = 0.02):
        self.sr = float(sr)
        self.cutoff = float(cutoff)
        self._cutoff_sm = float(cutoff)
        self.resonance = float(resonance)
        self._res_sm = float(resonance)
        self.smooth_time = float(smooth_time)
        # Direct-form II transposed state
        self._z1 = 0.0
        self._z2 = 0.0
        self.b = np.ones(3)
        self.a = np.array([1.0, 0.0, 0.0])
        self._last_cutoff = float("nan")
        self._last_res = float("nan")
        self._coeff_for(self._cutoff_sm, self._res_sm)
        self._last_cutoff = self._cutoff_sm
        self._last_res = self._res_sm

    def set_params(self, cutoff: float | None = None,
                   resonance: float | None = None,
                   smooth_time: float | None = None) -> None:
        if cutoff is not None:
            self.cutoff = float(cutoff)
        if resonance is not None:
            self.resonance = float(resonance)
        if smooth_time is not None:
            self.smooth_time = float(smooth_time)

    def reset(self) -> None:
        self._z1 = self._z2 = 0.0
        self._cutoff_sm = self.cutoff
        self._res_sm = self.resonance
        self._coeff_for(self._cutoff_sm, self._res_sm)
        self._last_cutoff = self._cutoff_sm
        self._last_res = self._res_sm

    def _coeff_for(self, f0: float, q: float) -> None:
        f0 = float(np.clip(f0, 1.0, self.sr * 0.499))
        q = max(float(q), 1e-4)
        w0 = 2.0 * np.pi * f0 / self.sr
        cos_w, sin_w = np.cos(w0), np.sin(w0)
        alpha = sin_w / (2.0 * q)
        a0 = 1.0 + alpha
        self.b = np.array([(1.0 - cos_w) / 2.0, 1.0 - cos_w,
                           (1.0 - cos_w) / 2.0]) / a0
        self.a = np.array([1.0, -2.0 * cos_w / a0, (1.0 - alpha) / a0])

    def _advance_smoothers(self, block_len: int) -> None:
        # One update per block, same value for every sample in the block:
        # equivalent to an lfilter with a block-constant coefficient.
        if self.smooth_time > 0.0:
            n = block_len
            alpha = 1.0 - np.exp(-n / (self.smooth_time * self.sr))
        else:
            alpha = 1.0
        self._cutoff_sm += alpha * (self.cutoff - self._cutoff_sm)
        self._res_sm += alpha * (self.resonance - self._res_sm)
        # Skip the transcendental coefficient build when the smoothed cutoff
        # and Q are bit-for-bit unchanged (e.g. smooth_time == 0, or a held
        # note whose smoother has underflowed to its target).  Reusing the
        # previous b/a is exactly equivalent because nothing changed.
        if self._cutoff_sm != self._last_cutoff or self._res_sm != self._last_res:
            self._coeff_for(self._cutoff_sm, self._res_sm)
            self._last_cutoff = self._cutoff_sm
            self._last_res = self._res_sm

    def process(self, x: np.ndarray) -> np.ndarray:
        """Vector path for one block; preserves delay-line state."""
        self._advance_smoothers(len(x))
        y, zf = lfilter(self.b, self.a, x, zi=[self._z1, self._z2])
        self._z1, self._z2 = float(zf[0]), float(zf[1])
        return y

    def process_step(self, x: float) -> float:
        """Scalar reference path; coefficient update happens once per block."""
        # The block-constant coefficient update is driven by prepare_block()
        # in the scalar engine so it matches the vector path exactly.
        y = self.b[0] * x + self._z1
        self._z1 = self.b[1] * x - self.a[1] * y + self._z2
        self._z2 = self.b[2] * x - self.a[2] * y
        return y

    def prepare_block(self, block_len: int) -> None:
        self._advance_smoothers(block_len)
