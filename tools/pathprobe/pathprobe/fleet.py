"""Exact deadline coverage over recorded workflow graphs for one chosen task."""

from bisect import bisect_left
from collections.abc import Iterable

from .core import DurationIndex


def _nonnegative(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


class FleetBudget:
    """Index the largest permissible replacement duration in each recorded run.

    Durations and deadlines must use the same integer unit across all runs.
    A budget of None denotes a run that misses its deadline even when the
    chosen task is instantaneous. Every run contributes one count.
    """

    def __init__(self, budgets: Iterable[int | None]) -> None:
        feasible = []
        impossible = 0
        for budget in budgets:
            if budget is None:
                impossible += 1
            else:
                feasible.append(_nonnegative(budget, "budget"))
        feasible.sort()
        self._feasible = feasible
        self._impossible = impossible

    @classmethod
    def from_budgets(cls, budgets: Iterable[int | None]) -> "FleetBudget":
        """Consume previously computed per-run budgets, retaining the feasible values."""
        return cls(budgets)

    @classmethod
    def from_runs(cls, rows: Iterable[tuple[DurationIndex, str, int]]) -> "FleetBudget":
        """Consume (index, task name, deadline) rows for the same task.

        A missing task raises KeyError. A different task name raises ValueError.
        Input is consumed once and workflow graphs are not retained.
        """
        def budgets():
            chosen_task = None
            for index, task, deadline in rows:
                if not isinstance(task, str):
                    raise ValueError("task name must be a string")
                if chosen_task is not None and task != chosen_task:
                    raise ValueError("every run must refer to the same task name")
                chosen_task = task
                deadline = _nonnegative(deadline, "deadline")
                profile = index.profile(task)
                ceiling = deadline - (profile["through"] - profile["duration"])
                bypass = profile["bypass"]
                if ceiling < 0 or (bypass is not None and bypass > deadline):
                    yield None
                else:
                    yield ceiling

        return cls.from_budgets(budgets())

    @property
    def total_runs(self) -> int:
        return len(self._feasible) + self._impossible

    @property
    def impossible_runs(self) -> int:
        return self._impossible

    def met(self, duration: int) -> int:
        """Count recorded runs meeting their own deadline at this replacement duration."""
        value = _nonnegative(duration, "duration")
        return len(self._feasible) - bisect_left(self._feasible, value)

    def limit(self, required_runs: int) -> int | None:
        """Largest duration meeting at least required_runs deadlines, or None."""
        count = _nonnegative(required_runs, "required_runs")
        if count < 1 or count > self.total_runs:
            raise ValueError("required_runs must be between 1 and total_runs")
        if count > len(self._feasible):
            return None
        return self._feasible[-count]
