import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DESResult:
    queries_total: int
    queries_failed: int
    downtime_pct: float
    lock_ticks: int


class DiscreteEventSimulator:
    def __init__(self, task_load_level: int, seed: int):
        self.queries_per_tick = task_load_level
        self.seed = seed
        self.tick_counter = 0

    def _step_seed(self) -> int:
        # Keep the simulator deterministic with the V2/V3 seed schedule.
        return self.seed + self.tick_counter * 31337

    def simulate_step(self, lock_ticks: int, failure_rate: float) -> DESResult:
        rng = random.Random(self._step_seed())
        total = self.queries_per_tick * max(1, lock_ticks)
        failed = sum(1 for _ in range(total) if rng.random() < failure_rate)
        self.tick_counter += 1
        return DESResult(
            queries_total=total,
            queries_failed=failed,
            downtime_pct=(failed / total) if total else 0.0,
            lock_ticks=lock_ticks,
        )
