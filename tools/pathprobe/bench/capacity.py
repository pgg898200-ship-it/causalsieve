#!/usr/bin/env python3
"""Measure one million scalar counterfactuals on exact synthetic DAG families."""
import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

from compare import ROOT, host_details, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=1000002)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.nodes < 3 or args.nodes > 30000000 or args.nodes % 3 or args.repeats < 1:
        parser.error("use a multiple of three tasks in [3,30000000] and positive repeats")
    compiler = os.environ.get("CXX", "c++")
    flags = ["-std=c++17", "-O3", "-DNDEBUG"]
    methods = ["pathprobe", "heap_sensitivity"]
    result = {
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "flags": flags,
        **host_details(),
        "header_sha256": source_hash("native/pathprobe.hpp"),
        "benchmark_sha256": source_hash("bench/native_compare.cpp"),
        "seed": 421,
        "nodes": args.nodes,
        "repeats": args.repeats,
        "timing": "One fresh process per method and trial. Each task receives one shuffled scalar edit. Edits alternate halving and increasing duration by ~50%. Build and queries timed; input graph and output-vector allocation, correctness verification and destruction excluded. build_queries_cpu_ns also records process CPU time over the same construction + query window. Rotating trial order; unpinned CPU.",
        "memory": "Linux /proc/self/status process peak resident KiB, including executable, input graph, query and answer buffers, and transient construction. Input RSS sampled after generating fixtures. A reading of -1 means unavailable.",
        "oracle": "Every result equals its family's analytical makespan. Each family's 2,046-task fixture is first checked against full topological DP for every edit. The eager max-heap competitor has the same O(1) lookup contract.",
        "graphs": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "capacity"
        subprocess.run([compiler, *flags, str(ROOT / "bench/native_compare.cpp"), "-o", str(binary)], check=True)
        for shape in ("series_diamonds", "parallel_chains"):
            samples = {m: [] for m in methods}
            for m in methods:
                subprocess.check_output([binary,"--capacity",shape,"2046",m])
            for i in range(args.repeats):
                for m in (methods if i % 2 == 0 else reversed(methods)):
                    output = subprocess.check_output([binary,"--capacity",shape,str(args.nodes),m])
                    samples[m].append(json.loads(output))
            summary = {}
            for method, rows in samples.items():
                summary[method] = {key: statistics.median(x[key] for x in rows) for key in
                                   ("build_ns", "queries_ns", "total_ns", "build_queries_cpu_ns", "input_rss_kib", "process_peak_rss_kib")}
                summary[method]["trials"] = rows
            result["graphs"].append({"shape":shape, "edges":samples[methods[0]][0]["edges"], "methods":summary})
    out = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
