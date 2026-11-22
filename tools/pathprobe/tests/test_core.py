"""Check the index against literal enumeration of original root-to-leaf paths."""

from itertools import combinations, product
import random
import unittest

from pathprobe import DurationIndex


def complete_paths(durations, edges):
    children = {task: [] for task in durations}
    roots = set(durations)
    for parent, child in edges:
        children[parent].append(child)
        roots.discard(child)
    pending = [(root,) for root in roots]
    while pending:
        path = pending.pop()
        if children[path[-1]]:
            pending.extend(path + (child,) for child in children[path[-1]])
        else:
            yield path


class DurationIndexTests(unittest.TestCase):
    def check_graph(self, durations, edges):
        index = DurationIndex(durations, iter(edges))
        paths = list(complete_paths(durations, edges))
        totals = [sum(durations[node] for node in path) for path in paths]
        self.assertEqual(index.makespan, max(totals, default=0))
        for task, duration in durations.items():
            profile = index.profile(task)
            with_task = [total for path, total in zip(paths, totals) if task in path]
            without = [total for path, total in zip(paths, totals) if task not in path]
            self.assertEqual(profile["duration"], duration)
            self.assertEqual(profile["through"], max(with_task))
            self.assertEqual(profile["bypass"], max(without, default=None))
            changes = {0, 1, duration, duration + 1, index.makespan + 1}
            # Probe both sides of the point where the controlling path changes.
            if without:
                crossover = duration + max(without) - max(with_task)
                changes.update(n for n in range(crossover - 1, crossover + 2) if n >= 0)
            for proposed in changes:
                expected = max(
                    sum(proposed if node == task else durations[node] for node in path)
                    for path in paths
                )
                self.assertEqual(index.predict(task, proposed), expected)
                if proposed == 0:
                    self.assertEqual(profile["max_saving"], index.makespan - expected)

    def test_all_four_node_ordered_dags_and_binary_durations(self):
        tasks = ("z", "b", "k", "a")
        possible = list(combinations(tasks, 2))
        for mask in range(1 << len(possible)):
            edges = [edge for i, edge in enumerate(possible) if mask & (1 << i)]
            for weights in product((0, 1), repeat=len(tasks)):
                with self.subTest(edges=edges, weights=weights):
                    self.check_graph(dict(zip(tasks, weights)), edges)

    def test_shuffled_random_dags(self):
        rng = random.Random(712488)
        for trial in range(200):
            tasks = [str(i) for i in range(rng.randrange(1, 11))]
            rng.shuffle(tasks)
            density = rng.random()
            edges = [e for e in combinations(tasks, 2) if rng.random() < density]
            rng.shuffle(edges)
            rng.shuffle(tasks)
            durations = {task: rng.choice((0, 0, 1, 4, 11, 10**90)) for task in tasks}
            with self.subTest(trial=trial):
                self.check_graph(durations, edges)

    def test_empty_and_one_node_graphs(self):
        self.check_graph({}, [])
        self.check_graph({"isolated": 0}, [])

    def test_cycles_and_unknown_endpoints(self):
        for edges in [[("a", "a")], [("a", "b"), ("b", "a")],
                      [("missing", "a")], [("b", "missing")]]:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                DurationIndex({"a": 1, "b": 2, "isolated": 10}, edges)

    def test_invalid_durations_and_unknown_queries(self):
        index = DurationIndex({"a": 1}, [])
        for duration in (-1, 0.5, "1", None, True):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    DurationIndex({"a": duration}, [])
                with self.assertRaises(ValueError):
                    index.predict("a", duration)
        with self.assertRaises(KeyError):
            index.predict("missing", 2)
        with self.assertRaises(KeyError):
            index.profile("missing")


if __name__ == "__main__":
    unittest.main()
