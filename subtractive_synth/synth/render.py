"""Offline rendering, file export and reproducible spectrum diagnostics."""

from __future__ import annotations

import time

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE
from .engine import SynthEngine, NoteEvent


def render_offline(events: list[NoteEvent], params, duration: float,
                   sr: float = SAMPLE_RATE, block_size: int = 256,
                   f0_automation=None) -> tuple[np.ndarray, dict]:
    """Render an event list; returns float64 mono audio and timing stats."""
    engine = SynthEngine(params, sr=sr, block_size=block_size)
    t0 = time.perf_counter()
    audio, rendered_seconds = engine.render_events(
        events, duration, f0_automation=f0_automation)
    elapsed = time.perf_counter() - t0
    stats = {
        "render_seconds": round(elapsed, 4),
        "audio_seconds": round(rendered_seconds, 4),
        "realtime_factor": round(elapsed / rendered_seconds, 4),
        "samples": int(audio.size),
        "peak": float(np.max(np.abs(audio))),
    }
    return audio, stats


def save_wav(path: str, audio: np.ndarray, sr: float = SAMPLE_RATE) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio / peak * 0.999
    sf.write(path, audio.astype(np.float32), int(sr), subtype="FLOAT")


def save_npz(path: str, **arrays) -> None:
    np.savez(path, **arrays)


def peak_stats(audio: np.ndarray, sr: float = SAMPLE_RATE) -> dict:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    # True-peak style check: 4x oversampled peak catches inter-sample peaks.
    up = np.repeat(audio, 4)
    return {
        "peak": round(peak, 6),
        "oversampled_peak": round(float(np.max(np.abs(up))), 6),
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 6),
        "clipped_samples": int(np.sum(np.abs(audio) >= 0.9999)),
    }


def spectrum(audio: np.ndarray, sr: float = SAMPLE_RATE,
             nfft: int = 2 ** 15, start: float | None = None,
             duration: float | None = None):
    """Hann-windowed magnitude spectrum (linear + dB) of a steady segment."""
    if start is None:
        start = max(0, (audio.size - nfft) // 2)
    else:
        start = int(start * sr)
    if duration is not None:
        nfft = min(nfft, int(duration * sr))
    seg = audio[start:start + nfft]
    if seg.size < 8:
        raise ValueError("spectrum segment too short")
    window = np.hanning(seg.size)
    mag = np.abs(np.fft.rfft(seg * window)) * 2.0 / np.sum(window)
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    db = 20.0 * np.log10(mag + 1e-12)
    return freqs, mag, db


def measure_aliasing(audio: np.ndarray, f0: float, sr: float = SAMPLE_RATE,
                     nfft: int = 2 ** 15, guard_bins: int = 3) -> dict:
    """Quantify in-band harmonics vs. residual energy away from every partial.

    For a periodic tone at ``f0`` every partial sits at k*f0.  PolyBLEP spreads
    each partial's energy over a few FFT bins, so bins within ``guard_bins`` of
    any partial below Nyquist count as in-band content.  Everything else in the
    spectrum is residual/aliased energy: for the naive discontinuous oscillator
    that residual is dominated by folded super-Nyquist partials, while the
    band-limited oscillator drives it toward the numerical floor.
    """
    freqs, mag, db = spectrum(audio, sr, nfft=nfft)
    df = freqs[1] - freqs[0]

    def bin_at(f: float) -> int:
        return int(round(f / df))

    protected = np.zeros(mag.size, dtype=bool)
    harmonic_bins: list[int] = []
    k = 1
    while k * f0 < sr / 2.0:
        b = bin_at(k * f0)
        harmonic_bins.append(b)
        lo, hi = max(0, b - guard_bins), min(mag.size, b + guard_bins + 1)
        protected[lo:hi] = True
        k += 1

    residual_bins = np.flatnonzero(~protected)
    # Skip the lowest bins: sidelobes of the windowed fundamental live there.
    residual_bins = residual_bins[residual_bins > guard_bins + 1]

    harm_rms = float(np.sqrt(np.mean(mag[harmonic_bins] ** 2)))
    res_rms = float(np.sqrt(np.mean(mag[residual_bins] ** 2))) if residual_bins.size else 0.0
    return {
        "f0": f0,
        "n_harmonics_measured": len(harmonic_bins),
        "n_residual_bins_measured": int(residual_bins.size),
        "harmonic_rms_db": round(20 * np.log10(harm_rms + 1e-12), 2),
        "alias_rms_db": round(20 * np.log10(res_rms + 1e-12), 2),
        "alias_to_harmonic_db": round(
            20 * np.log10((res_rms + 1e-12) / (harm_rms + 1e-12)), 2),
    }


def estimate_fundamental(audio: np.ndarray, expected_f0: float,
                         sr: float = SAMPLE_RATE) -> dict:
    """Pitch check via parabolic-interpolated peak near the expected f0."""
    freqs, mag, _ = spectrum(audio, sr, nfft=2 ** 16)
    df = freqs[1] - freqs[0]
    center = int(round(expected_f0 / df))
    lo, hi = max(1, center - 8), min(mag.size - 1, center + 8)
    j = lo + int(np.argmax(mag[lo:hi]))
    # parabolic interpolation
    a, b, c = mag[j - 1], mag[j], mag[j + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
    measured = freqs[j] + shift * df
    cents = 1200.0 * np.log2(measured / expected_f0)
    return {
        "expected_hz": round(expected_f0, 3),
        "measured_hz": round(float(measured), 3),
        "error_cents": round(float(cents), 3),
        "relative_error_percent": round(
            100.0 * abs(measured - expected_f0) / expected_f0, 4),
    }
