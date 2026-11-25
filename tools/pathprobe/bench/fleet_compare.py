"""Cold-build, fixed-task coverage on two archived GitHub Actions cohorts."""
import hashlib
import json
import platform
import statistics
from time import perf_counter_ns

from compare import ROOT, Recompute, host_details, source_hash
from pathprobe import DurationIndex, FleetBudget


def direct(rows, task, queries):
    runs = [(Recompute(row["durations"], row["edges"]), row["deadline"]) for row in rows]
    return [sum(graph.predict(task, duration) <= deadline for graph, deadline in runs)
            for duration in queries]


def two_replays(rows, task, queries):
    budgets = []
    for row in rows:
        graph = Recompute(row["durations"], row["edges"])
        deadline = row["deadline"]
        if graph.predict(task, 0) > deadline:
            budgets.append(None)
            continue
        # Every bypass finishes by the deadline. A duration one unit above
        # it forces a through-path to win, revealing its fixed overhead.
        probe = deadline + 1
        overhead = graph.predict(task, probe) - probe
        budgets.append(deadline - overhead)
    fleet = FleetBudget.from_budgets(budgets)
    return [fleet.met(duration) for duration in queries]


def all_tasks_index(rows, task, queries):
    fleet = FleetBudget.from_runs(
        (DurationIndex(row["durations"], row["edges"]), task, row["deadline"])
        for row in rows
    )
    return [fleet.met(duration) for duration in queries]


def benchmark(filename, task, single_query):
    fixture = ROOT / "examples" / "actions" / filename
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    stop = max(row["deadline"] for row in rows) + 1
    sweep = list(range(stop + 1))
    methods = [direct, two_replays, all_tasks_index]
    gold = direct(rows, task, sweep)
    for method in methods[1:]:
        if method(rows, task, sweep) != gold:
            raise AssertionError(f"{method.__name__}: coverage disagrees for {filename}")
    batches = []
    for queries in ([single_query], sweep):
        samples = {method.__name__: [] for method in methods}
        batch_gold = [gold[q] for q in queries]
        for trial in range(21):
            offset = trial % len(methods)
            for method in methods[offset:] + methods[:offset]:
                start = perf_counter_ns()
                got = method(rows, task, queries)
                elapsed = perf_counter_ns() - start
                if got != batch_gold:
                    raise AssertionError(method.__name__)
                samples[method.__name__].append(elapsed)
        batches.append({"queries": len(queries), "first": queries[0], "last": queries[-1],
                        "median_ns": {k: statistics.median(v) for k, v in samples.items()},
                        "trials_ns": samples})
    return {"fixture": filename, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "task": task, "runs": len(rows), "coverage_at_single_query": gold[single_query],
            "coverage_sha256": hashlib.sha256(json.dumps(gold).encode()).hexdigest(), "batches": batches}


if __name__ == "__main__":
    print(json.dumps({"python": platform.python_version(), **host_details(),
                      "source_sha256": {p: source_hash(p) for p in
                                        ("pathprobe/core.py", "pathprobe/fleet.py", "bench/compare.py",
                                         "bench/fleet_compare.py")},
                      "cohorts": [benchmark("python-sdk-ci.jsonl", "test (Python 3.10)", 289),
                                  benchmark("agents-sdk-releases.jsonl", "checks", 845)]}, indent=2))
