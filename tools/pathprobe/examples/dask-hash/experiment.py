"""Capture paired local SHA-256 workflows, or replay the checked-in observations."""
import argparse
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.dask_capture import capture
from pathprobe import DurationIndex


def digest(path, chunk):
    start = perf_counter_ns()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest(), start, perf_counter_ns()


def finish(left, right):
    # The processes share this host's monotonic clock.
    return [left[0], right[0]], [left[1:], right[1:]], perf_counter_ns()


def measure(client, paths, expected, workers, label, chunk):
    import dask
    from dask import delayed

    leaves = []
    keys = [f"candidate-{label}", f"sibling-{label}"]
    for key, path, size, worker in zip(keys, paths, (chunk, 65536), workers):
        with dask.annotate(workers=[worker]):
            leaves.append(delayed(digest)(path, size, dask_key_name=key))
    sink = f"finish-{label}"
    workflow = delayed(finish)(*leaves, dask_key_name=sink)
    (hashes, intervals, end), graph = capture(client, workflow)
    if hashes != expected or sorted(graph["edges"]) != sorted([[key, sink] for key in keys]):
        raise RuntimeError("digest or executed DAG changed")
    start = min(row[0] for row in intervals)
    return {"candidate": keys[0], "graph": graph,
            "leaf_intervals_ns": [[a - start, b - start] for a, b in intervals],
            "span_ns": end - start}


def collect():
    import dask
    import distributed
    from distributed import Client, LocalCluster

    result = {"python": platform.python_version(), "platform": platform.platform(),
              "dask": dask.__version__, "distributed": distributed.__version__, "pairs": []}
    with TemporaryDirectory() as temp:
        paths = [Path(temp) / name for name in ("candidate.bin", "sibling.bin")]
        block = bytes(range(256)) * 4096
        for path, mib in zip(paths, (32, 64)):
            with path.open("wb") as f:
                for _ in range(mib):
                    f.write(block)
        expected = [digest(path, 65536)[0] for path in paths]
        result["digests"] = expected
        for slots in (1, 2):
            with LocalCluster(n_workers=slots, processes=True, threads_per_worker=1,
                              host="127.0.0.1", dashboard_address=None) as cluster, Client(cluster) as client:
                addresses = sorted(client.scheduler_info()["workers"])
                workers = [addresses[0], addresses[-1]]
                # Exercise both reader sizes before timing; inputs stay in the OS cache.
                for mode, size in (("before", 64), ("after", 65536)):
                    measure(client, paths, expected, workers, f"warm-{slots}-{mode}", size)
                for trial in range(10):
                    pair = {"workers": slots, "trial": trial}
                    order = ["before", "after"] if trial % 2 == 0 else ["after", "before"]
                    for mode in order:
                        chunk = 64 if mode == "before" else 65536
                        pair[mode] = measure(client, paths, expected, workers,
                                             f"{slots}-{trial}-{mode}", chunk)
                    result["pairs"].append(pair)
    return result


def summarize(data):
    for workers in (1, 2):
        rows = []
        for pair in data["pairs"]:
            if pair["workers"] != workers:
                continue
            before, after = pair["before"], pair["after"]
            graph, task = before["graph"], before["candidate"]
            index = DurationIndex(graph["durations"], graph["edges"])
            replacement = after["graph"]["durations"][after["candidate"]]
            modeled = index.makespan - index.predict(task, replacement)
            full_delta = graph["durations"][task] - replacement
            observed = (before["span_ns"] - after["span_ns"]) / 1000
            rows.append((observed, modeled, full_delta, abs(observed - modeled),
                         abs(observed - full_delta)))
        metrics = ["observed_saving_ms", "dag_saving_ms", "full_delta_ms",
                   "dag_absolute_error_ms", "full_delta_absolute_error_ms"]
        print(workers, {k: round(statistics.median(col) / 1000, 2)
                        for k, col in zip(metrics, zip(*rows))})
        print("DAG closer:", sum(r[3] < r[4] for r in rows), "of", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, help="write a fresh capture to this JSON path")
    args = parser.parse_args()
    if args.capture:
        observations = collect()
        args.capture.write_text(json.dumps(observations, indent=2) + "\n")
    else:
        observations = json.loads(Path(__file__).with_name("observations.json").read_text())
    summarize(observations)
