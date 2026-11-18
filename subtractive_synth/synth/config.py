"""Global constants and synthesizer parameter container.

All time-valued parameters are stored in **seconds** and converted to sample
counts inside the DSP code, so envelopes and smoothing never silently depend
on a block size.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

SAMPLE_RATE = 48000.0
BLOCK_SIZE = 256

WAVE_SAW = "saw"
WAVE_SQUARE = "square"
WAVE_TRI = "triangle"
WAVE_NAIVE_SAW = "naive_saw"   # deliberately aliased, used only by tests
WAVE_NAIVE_SQUARE = "naive_square"


@dataclass
class SynthParams:
    """User-facing synth parameters.

    Times are seconds, frequencies are Hz, gains are linear (0..~1),
    resonance Q is dimensionless.
    """

    waveform: str = WAVE_SAW
    attack: float = 0.005
    decay: float = 0.15
    sustain: float = 0.7
    release: float = 0.25
    cutoff: float = 4000.0
    resonance: float = 0.707
    gain: float = 0.8
    # time constant of the one-pole smoother applied to continuously
    # adjustable parameters (cutoff / gain / f0 glide), seconds
    smooth_time: float = 0.02
    voices: int = 8

    def with_(self, **changes) -> "SynthParams":
        return replace(self, **changes)

    def validate(self, sr: float = SAMPLE_RATE) -> None:
        if self.waveform not in (
            WAVE_SAW, WAVE_SQUARE, WAVE_TRI, WAVE_NAIVE_SAW, WAVE_NAIVE_SQUARE
        ):
            raise ValueError(f"unknown waveform {self.waveform!r}")
        for name in ("attack", "decay", "release", "smooth_time"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 seconds")
        if not 0.0 <= self.sustain <= 1.5:
            raise ValueError("sustain out of range")
        if not 1.0 <= self.cutoff <= sr / 2.0:
            raise ValueError("cutoff must lie within (0, Nyquist)")
        if self.resonance <= 0:
            raise ValueError("resonance must be positive")
        if self.voices < 1:
            raise ValueError("need at least one voice")
