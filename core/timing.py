from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class StageTimer:
    """Accumulates wall time per pipeline stage. Thread-safe (model calls run in a pool)."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + float(seconds)

    @contextmanager
    def time(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - t0)

    def wrap(self, stage: str, fn):
        def _timed(*args, **kwargs):
            with self.time(stage):
                return fn(*args, **kwargs)
        return _timed

    def per_frame_ms(self, frames: int) -> dict[str, float]:
        n = max(int(frames), 1)
        with self._lock:
            return {k: 1000.0 * v / n for k, v in sorted(self._totals.items(), key=lambda kv: -kv[1])}

    def report(self, frames: int, wall_seconds: float) -> str:
        wall_ms = 1000.0 * wall_seconds / max(int(frames), 1)
        lines = [f"[timing] wall {wall_ms:.0f} ms/frame; stage totals (model calls overlap in threads):"]
        for k, v in self.per_frame_ms(frames).items():
            lines.append(f"[timing]   {k:<22s} {v:8.1f} ms/frame  ({100.0 * v / max(wall_ms, 1e-9):5.1f}% of wall)")
        return "\n".join(lines)
