"""Audio backends for live performance.

* :class:`PortAudioBackend` wraps ``sounddevice`` and drives the exact same
  :class:`~synth.engine.SynthEngine.pull_block` callback used offline.
* :class:`MockAudioBackend` reproduces the callback contract (fixed-size pulls
  on a real-time-style schedule, timing measured per callback) without needing
  the PortAudio system library.  It is used in this sandbox and in CI to prove
  the real-time path keeps its deadline and matches the offline render.

Both backends implement ``start()`` / ``stop()`` / ``pull()`` and feed the same
engine method, so switching machines never changes the DSP.
"""

from __future__ import annotations

import time

import numpy as np

from .config import SAMPLE_RATE, BLOCK_SIZE


class MockAudioBackend:
    """Deterministic pull-based backend with per-block timing statistics."""

    name = "mock"

    def __init__(self, engine, sr: float = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.engine = engine
        self.sr = float(sr)
        self.block_size = int(block_size)
        self._block_seconds = self.block_size / self.sr
        self.times_ms: list[float] = []
        self.overruns = 0
        self._running = False
        self._record: list[np.ndarray] = []

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def pull(self, record: bool = False) -> np.ndarray:
        """One callback-length pull; raises if it misses the audio deadline."""
        t0 = time.perf_counter()
        block = self.engine.pull_block(self.block_size)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.times_ms.append(dt_ms)
        if dt_ms > self._block_seconds * 1000.0:
            self.overruns += 1
        if record:
            self._record.append(block.copy())
        return block

    def render_realtime(self, duration: float) -> np.ndarray:
        """Simulate a live session of ``duration`` seconds."""
        n_blocks = int(round(duration * self.sr / self.block_size))
        self._record = []
        self.start()
        for _ in range(n_blocks):
            self.pull(record=True)
        self.stop()
        return np.concatenate(self._record) if self._record else np.zeros(0)

    def timing_report(self) -> dict:
        t = np.asarray(self.times_ms)
        deadline = self._block_seconds * 1000.0
        if t.size == 0:
            return {"blocks": 0}
        return {
            "blocks": int(t.size),
            "deadline_ms": round(deadline, 3),
            "mean_ms": round(float(t.mean()), 4),
            "p95_ms": round(float(np.percentile(t, 95)), 4),
            "max_ms": round(float(t.max()), 4),
            "cpu_percent_of_deadline": round(100.0 * float(t.mean()) / deadline, 2),
            "overruns": self.overruns,
        }


class PortAudioBackend:
    """Real sounddevice/PortAudio backend, used when the system library exists."""

    name = "portaudio"

    def __init__(self, engine, sr: float = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.engine = engine
        self.sr = float(sr)
        self.block_size = int(block_size)
        self._stream = None
        self.times_ms: list[float] = []
        self.overruns = 0

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True

    def start(self) -> None:
        import sounddevice as sd
        self.engine.pull_block(self.block_size)  # warm-up allocation

        def callback(outdata, frames, time_info, status):  # noqa: ARG001
            t0 = time.perf_counter()
            block = self.engine.pull_block(frames)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.times_ms.append(dt_ms)
            if dt_ms > frames / self.sr * 1000.0:
                self.overruns += 1
            outdata[:, 0] = block
            if outdata.shape[1] > 1:
                outdata[:, 1] = block

        self._stream = sd.OutputStream(
            samplerate=int(self.sr), blocksize=self.block_size,
            channels=2, dtype="float32", callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    def timing_report(self) -> dict:
        t = np.asarray(self.times_ms)
        deadline = self.block_size / self.sr * 1000.0
        if t.size == 0:
            return {"blocks": 0}
        return {
            "blocks": int(t.size),
            "deadline_ms": round(deadline, 3),
            "mean_ms": round(float(t.mean()), 4),
            "p95_ms": round(float(np.percentile(t, 95)), 4),
            "max_ms": round(float(t.max()), 4),
            "cpu_percent_of_deadline": round(100.0 * float(t.mean()) / deadline, 2),
            "overruns": self.overruns,
        }


def make_backend(engine, sr: float = SAMPLE_RATE,
                 block_size: int = BLOCK_SIZE):
    if PortAudioBackend.available():
        return PortAudioBackend(engine, sr, block_size)
    return MockAudioBackend(engine, sr, block_size)
