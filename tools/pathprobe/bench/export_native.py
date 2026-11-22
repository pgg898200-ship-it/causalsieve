#!/usr/bin/env python3
"""Build the C++ benchmark and replay the same queries as compare.py."""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from compare import ROOT, host_details, load_graph, source_hash, synthetic, workload


def run_graph(binary, label, durations, edges, args):
    ids = {task: i for i, task in enumerate(durations)}
    queries = workload(durations, args.seed, args.edits)
    # These comparators use unchecked int64 arithmetic. Bound every possible
    # path, including its one edited task, before launching the timing process.
    max_extra = max(max(0, d - durations[u]) for u, d in queries)
    if sum(durations.values()) + max_extra > 2**63 - 1:
        raise ValueError("workload exceeds native benchmark's safe int64 bound")
    lines = [f"{len(durations)} {len(edges)} {args.repeats}", " ".join(map(str, durations.values()))]
    lines.extend(f"{ids[u]} {ids[v]}" for u, v in edges)
    lines.extend(f"{ids[u]} {d}" for u, d in queries)
    data = subprocess.check_output([binary], text=True, input="\n".join(lines) + "\n")
    return {"graph": label, **json.loads(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=[], help="pinned fixture in compare.py format")
    parser.add_argument("--nodes", type=int, default=2048, help="size of both additional synthetic graphs")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--edits", choices=("shorten", "increase", "mixed"), default="mixed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.nodes <= 0 or args.repeats <= 0:
        parser.error("--nodes and --repeats must be positive")
    compiler = os.environ.get("CXX", "c++")
    flags = ["-std=c++17", "-O3", "-DNDEBUG"]
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / "compare"
        subprocess.run([compiler, *flags, str(ROOT / "bench/native_compare.cpp"), "-o", str(binary)], check=True)
        graphs = [(shape, *synthetic(shape, args.nodes, args.seed)) for shape in ("detours", "layers")]
        graphs.extend((p.name, *load_graph(p)) for p in args.input)
        result = {
            "schema_version": 1,
            **host_details(),
            "header_sha256": source_hash("native/pathprobe.hpp"),
            "benchmark_sha256": source_hash("bench/native_compare.cpp"),
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": flags,
            "seed": args.seed,
            "repeats": args.repeats,
            "edits": args.edits,
            "timing": "Median wall nanoseconds and individual [build,query,total] trials. Process CPU also covers build + queries, including timer calls; interpret sub-microsecond differences cautiously. Each total covers graph construction and its full query batch. All methods receive the same integer IDs. Excludes file I/O, ID conversion, allocating the common output vector, verification, and destruction. Each method verified against full DP, trial order rotates. No CPU pinning. Timings cover an in-process kernel.",
            "graphs": [run_graph(binary, label, ds, es, args) for label, ds, es in graphs],
        }
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
