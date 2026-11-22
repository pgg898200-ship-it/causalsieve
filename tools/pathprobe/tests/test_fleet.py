"""Check cohort deadline queries against replay of every recorded DAG."""

from itertools import combinations
import random
import unittest

from pathprobe.core import DurationIndex
from pathprobe.fleet import FleetBudget


def replay(durations, parents, selected, replacement):
    # IDs follow dependency order in these generated runs.
    finished = []
    for i, weight in enumerate(durations):
        start = max((finished[parent] for parent in parents[i]), default=0)
        finished.append(start + (replacement if i == selected else weight))
    return max(finished)


class FleetBudgetTests(unittest.TestCase):
    def test_exact_deadline_equality_and_impossible_runs(self):
        graph = DurationIndex({"agent": 7, "join": 3, "other": 5}, [("agent", "join")])
        counts = FleetBudget.from_runs((graph, "agent", d) for d in (5, 7, 4, 5))
        self.assertEqual((counts.total_runs, counts.impossible_runs), (4, 1))
        self.assertEqual([counts.met(d) for d in (0, 2, 3, 4, 5)], [3, 3, 1, 1, 0])
        self.assertEqual([counts.limit(k) for k in range(1, 5)], [4, 2, 2, None])

    def test_through_path_misses_at_zero_and_missing_graph_rejections(self):
        graph = DurationIndex({"agent": 1, "join": 5}, [("agent", "join")])
        cohort = FleetBudget.from_runs([(graph, "agent", 4)])
        self.assertEqual(cohort.impossible_runs, 1)
        self.assertIsNone(cohort.limit(1))
        with self.assertRaises(KeyError):
            FleetBudget.from_runs([(graph, "missing", 10)])
        with self.assertRaises(ValueError):
            FleetBudget.from_runs([(graph, "agent", 4), (graph, "join", 4)])

    def test_empty_huge_and_zero(self):
        empty = FleetBudget.from_runs(iter(()))
        self.assertEqual(empty.total_runs, 0)
        self.assertEqual(empty.met(0), 0)
        with self.assertRaises(ValueError):
            empty.limit(1)
        huge = 10**300
        graph = DurationIndex({"": huge}, ())
        compiled = FleetBudget.from_runs([(graph, "", huge), (graph, "", 0)])
        loaded = FleetBudget.from_budgets(iter((huge, 0)))
        for fleet in (compiled, loaded):
            self.assertEqual(fleet.limit(1), huge)
            self.assertEqual(fleet.limit(2), 0)
            self.assertEqual(fleet.met(huge), 1)
            self.assertEqual(fleet.met(huge + 1), 0)

    def test_validates_all_counts_and_times(self):
        cohort = FleetBudget.from_budgets((None, None))
        graph = DurationIndex({"a": 1}, [])
        for bad in (True, -1, 0.1, "1"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    FleetBudget.from_budgets([bad])
                with self.assertRaises(ValueError):
                    FleetBudget.from_runs([(graph, "a", bad)])
                with self.assertRaises(ValueError):
                    cohort.met(bad)
                with self.assertRaises(ValueError):
                    cohort.limit(bad)
        for bad in (0, 3):
            with self.assertRaises(ValueError):
                cohort.limit(bad)

    def test_seeded_fleets_against_full_replay(self):
        rng = random.Random(251994)
        for _ in range(80):
            rows = []
            inputs = []
            for _ in range(rng.randint(5, 20)):
                n = rng.randint(1, 12)
                selected = rng.randrange(n)
                names = ["focus" if i == selected else f"task-{i}" for i in range(n)]
                weights = [rng.randrange(8) for _ in range(n)]
                parents = [[] for _ in range(n)]
                edges = []
                for parent, child in combinations(range(n), 2):
                    if rng.random() < 0.25:
                        parents[child].append(parent)
                        edges.append((names[parent], names[child]))
                deadline = rng.randrange(41)
                rows.append((DurationIndex(dict(zip(names, weights)), edges), "focus", deadline))
                inputs.append((weights, parents, selected, deadline))
            cohort = FleetBudget.from_runs(iter(rows))
            expected_counts = []
            # A duration above 40 exceeds every individual deadline in this fixture.
            for proposed in range(42):
                met = sum(replay(weights, parents, selected, proposed) <= deadline
                          for weights, parents, selected, deadline in inputs)
                expected_counts.append(met)
                self.assertEqual(cohort.met(proposed), met)
            self.assertEqual(cohort.impossible_runs, len(rows) - expected_counts[0])
            for k in range(1, len(rows) + 1):
                feasible = [d for d, count in enumerate(expected_counts) if count >= k]
                self.assertEqual(cohort.limit(k), max(feasible, default=None))


if __name__ == "__main__":
    unittest.main()
