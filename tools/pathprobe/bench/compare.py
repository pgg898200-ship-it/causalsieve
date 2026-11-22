#!/usr/bin/env python3
"""Compare exact one-task DAG duration forecasts, charging all graph preparation."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import platform
import random
import statistics
import sys
from collections import deque
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pathprobe.core import DurationIndex


def host_details():
    cpuinfo = Path("/proc/cpuinfo")
    processor = platform.processor()
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                processor = line.split(":", 1)[1].strip()
                break
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": processor}


def source_hash(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Recompute:
    """Integer-indexed topological DP. Compile the graph once per query batch."""

    def __init__(self, durations, edges):
        names = list(durations)
        self.ids = {name: i for i, name in enumerate(names)}
        self.weights = [durations[name] for name in names]
        self.parents = [[] for _ in names]
        self.children = [[] for _ in names]
        for parent, child in edges:
            p, c = self.ids[parent], self.ids[child]
            self.parents[c].append(p)
            self.children[p].append(c)
        incoming = [len(row) for row in self.parents]
        ready = deque(i for i, degree in enumerate(incoming) if not degree)
        self.order = []
        while ready:
            node = ready.popleft()
            self.order.append(node)
            for child in self.children[node]:
                incoming[child] -= 1
                if not incoming[child]:
                    ready.append(child)
        if len(self.order) != len(names):
            raise ValueError("graph contains a cycle")

    def longest(self, changed, new_duration):
        finish = [0] * len(self.weights)
        makespan = 0
        for node in self.order:
            start = 0
            for parent in self.parents[node]:
                if finish[parent] > start:
                    start = finish[parent]
            duration = new_duration if node == changed else self.weights[node]
            end = start + duration
            finish[node] = end
            if end > makespan:
                makespan = end
        return makespan

    def predict(self, task, new_duration):
        return self.longest(self.ids[task], new_duration)


class CachedCriticalPath(Recompute):
    """Exact O(1) increases and off-path shrinks; cache affected-node shrink floors."""

    def __init__(self, durations, edges):
        super().__init__(durations, edges)
        n = len(self.weights)
        finish = [0] * n
        predecessor = [-1] * n
        for node in self.order:
            best = -1
            for p in self.parents[node]:
                if best == -1 or finish[p] > finish[best]:
                    best = p
            predecessor[node] = best
            finish[node] = self.weights[node] + (finish[best] if best >= 0 else 0)
        last = max(range(n), key=finish.__getitem__)
        self.makespan = finish[last]
        self.on_path = bytearray(n)
        while last >= 0:
            self.on_path[last] = 1
            last = predecessor[last]
        # "extra" excludes this node's duration and includes both path sides.
        tail = [0] * n
        self.extra = [0] * n
        for node in reversed(self.order):
            tail[node] = max((self.weights[c] + tail[c] for c in self.children[node]), default=0)
            self.extra[node] = finish[node] - self.weights[node] + tail[node]
        self.finish = finish
        self.tail = tail
        self.floor = {}

    def predict(self, task, new_duration):
        node = self.ids[task]
        via = self.extra[node] + new_duration
        if new_duration >= self.weights[node]:
            return max(self.makespan, via)
        if not self.on_path[node]:
            return self.makespan
        # At any allowed duration >= 0, the new via-path dominates its value at 0.
        if node not in self.floor:
            self.floor[node] = self.longest(node, 0)
        return max(self.floor[node], via)


class HeapSensitivity(CachedCriticalPath):
    """Independent eager sensitivity index using a topological sweep and max-heap."""

    def __init__(self, durations, edges):
        super().__init__(durations, edges)
        n = len(self.weights)
        before = [0] * n
        count = 0
        for node in self.order:
            before[node] = count
            count += self.on_path[node]
        bypass = [self.makespan] * n
        heap = []
        # Virtual source edges to each root can cross earlier critical tasks.
        for node in self.order:
            if not self.parents[node] and before[node]:
                heap.append((-(self.weights[node] + self.tail[node]), before[node]))
        heapq.heapify(heap)
        for node in self.order:
            point = before[node]
            if self.on_path[node]:
                while heap and heap[0][1] <= point:
                    heapq.heappop(heap)
                bypass[node] = -heap[0][0] if heap else -1
            start = point + self.on_path[node]
            prefix = self.finish[node]
            for child in self.children[node]:
                stop = before[child]
                if start < stop:
                    candidate = prefix + self.weights[child] + self.tail[child]
                    heapq.heappush(heap, (-candidate, stop))
            if not self.children[node] and start < count:
                heapq.heappush(heap, (-prefix, count))
        self.bypass = bypass

    def predict(self, task, new_duration):
        node = self.ids[task]
        return max(self.bypass[node], self.extra[node] + new_duration)


def synthetic(shape, n, seed):
    rng = random.Random(seed)
    durations = {str(i): rng.randrange(1, 101) for i in range(n)}
    if shape == "detours":
        # Series of diamonds p->a->j and p->b->j, with a direct p->j edge.
        edges = []
        previous = 0
        for start in range(1, n, 3):
            branches = list(range(start, min(start + 2, n)))
            edges.extend((previous, branch) for branch in branches)
            if start + 2 < n:
                join = start + 2
                edges.extend((branch, join) for branch in branches)
                edges.append((previous, join))
                previous = join
    else:
        # Wide layers with up to four distinct parents in the preceding layer.
        width = max(2, int(n**0.5))
        edges = []
        for child in range(width, n):
            layer = child // width
            parents = range((layer - 1) * width, layer * width)
            edges.extend((p, child) for p in rng.sample(parents, min(4, width)))
    return durations, [(str(a), str(b)) for a, b in edges]


def load_graph(path):
    data = json.loads(path.read_text())
    durations = data["durations"]
    if not isinstance(durations, dict) or not durations:
        raise ValueError("durations must be a nonempty JSON object")
    if any(type(d) is not int or d < 0 for d in durations.values()):
        raise ValueError("durations must contain nonnegative integers")
    edges = []
    for pair in data["edges"]:
        if not isinstance(pair, list) or len(pair) != 2 or any(k not in durations for k in pair):
            raise ValueError("edges must contain pairs of known task IDs")
        edges.append(tuple(pair))
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate edges")
    return durations, edges


def workload(durations, seed, edits):
    names = list(durations)
    rng = random.Random(seed)
    rng.shuffle(names)
    queries = []
    for i, task in enumerate(names):
        d = durations[task]
        increase = edits == "increase" or (edits == "mixed" and i % 2 == 1)
        queries.append((task, d + max(1, d // 2) if increase else d // 2))
    return queries


def benchmark(label, durations, edges, args):
    classes = {"full_dp": Recompute, "critical_path_lazy_cache": CachedCriticalPath, "heap_sensitivity": HeapSensitivity, "pathprobe": DurationIndex}
    all_queries = workload(durations, args.seed, args.edits)
    reference = Recompute(durations, edges)
    gold = [reference.predict(*q) for q in all_queries]
    count = len(durations)
    rows = []
    for batch in dict.fromkeys((1, min(10, count), count)):
        queries = all_queries[:batch]
        expected = gold[:batch]
        samples = {name: [] for name in classes}
        # Rotate method order. Every timed batch starts with a new instance.
        names = list(classes)
        for trial in range(args.repeats):
            offset = trial % len(names)
            for name in names[offset:] + names[:offset]:
                start = perf_counter_ns()
                index = classes[name](durations, edges)
                built = perf_counter_ns()
                observed = [index.predict(task, d) for task, d in queries]
                end = perf_counter_ns()
                if observed != expected:
                    i = next(i for i, pair in enumerate(zip(observed, expected)) if pair[0] != pair[1])
                    raise AssertionError(f"{label} {name} {queries[i]}: got {observed[i]}, expected {expected[i]}")
                samples[name].append((built - start, end - built, end - start))
                del index
        results = {}
        for name, values in samples.items():
            results[name] = {
                metric: statistics.median(row[col] for row in values)
                for col, metric in enumerate(("build_ns", "queries_ns", "total_ns"))
            }
        rows.append({
            "queries": batch,
            "changed_durations": sum(d != durations[task] for task, d in queries),
            "predictions_sha256": hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest(),
            "methods": results,
        })
    return {"graph": label, "nodes": count, "edges": len(edges), "makespan": reference.longest(-1, 0), "batches": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="JSON {durations: {id: int}, edges: [[a,b]]}; repeatable")
    parser.add_argument("--nodes", type=int, default=512, help="nodes per synthetic graph")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--edits", choices=("shorten", "increase", "mixed"), default="mixed")
    parser.add_argument("--output", type=Path, help="also save the stdout JSON to this path")
    args = parser.parse_args()
    if args.nodes < 1 or args.repeats < 1:
        parser.error("--nodes and --repeats must be positive")
    if args.input:
        graphs = ((p.name, *load_graph(p)) for p in args.input)
    else:
        graphs = ((name, *synthetic(name, args.nodes, args.seed)) for name in ("detours", "layers"))
    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        **host_details(),
        "core_sha256": source_hash("pathprobe/core.py"),
        "benchmark_sha256": source_hash("bench/compare.py"),
        "seed": args.seed,
        "repeats": args.repeats,
        "edits": args.edits,
        "timing": "Independent medians in ns. Total includes fresh graph/index construction and all scalar queries. Excludes file I/O, workload generation, correctness checks and destruction. One shuffled prefix per batch, each task queried at most once; mixed alternates shortening and increasing, starting with shortening.",
        "graphs": [benchmark(label, durations, edges, args) for label, durations, edges in graphs],
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
