"""Per-stage timing.

The whole point of this project is a latency claim, so timing is built into the
pipeline rather than bolted on by the benchmark. Every stage reports separately —
an end-to-end number alone cannot tell us whether to optimise thresholding or CCL.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class StageTimer:
    """Accumulates wall-clock milliseconds per named stage.

    Re-entering the same stage name accumulates rather than overwrites, so a stage
    called once per region still reports its true total cost.
    """

    __slots__ = ("_ms",)

    def __init__(self) -> None:
        self._ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self._ms[name] = self._ms.get(name, 0.0) + (perf_counter() - start) * 1000.0

    def record(self, name: str, ms: float) -> None:
        self._ms[name] = self._ms.get(name, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return sum(self._ms.values())

    def as_dict(self) -> dict[str, float]:
        """Stage timings plus a ``total`` key, rounded to microsecond precision."""
        out = {k: round(v, 3) for k, v in self._ms.items()}
        out["total"] = round(self.total_ms, 3)
        return out
