"""Capture a fresh, static Dask Distributed workflow as PathProbe JSON.

Install the optional dependency with `python -m pip install "dask[distributed]"`.
From this checkout run:

    python examples/dask_capture.py | python -m pathprobe rank -

For your own single-output Dask collection, call capture(client, collection).
It returns (computed result, graph record). The record models worker compute
time in integer microseconds, using the scheduler's actual dependencies after
optimization. Use distinct Dask task keys for distinct work. Queueing,
transfers, worker limits and dynamic submissions inside tasks are outside this
model. Use a fresh graph whose full run fits Dask's task-stream buffer. Missing
events (including cache hits) and ambiguous attempts raise an error.
"""

import json
import math


def workflow_record(dependencies, events):
    """Join a complete root dependency closure with its worker compute events."""
    # Give tuple keys a readable label and retain ordinary task names.
    names = {key: key if isinstance(key, str) else repr(key) for key in dependencies}
    if len(set(names.values())) != len(names):
        raise ValueError("Dask keys collide after conversion to strings")
    durations = {}
    for event in events:
        key = event["key"]
        if key not in names:
            continue  # The task stream can include other clients' workflows.
        if event.get("status") != "OK" or key in durations:
            raise ValueError(f"expected one successful execution of {key!r}")
        compute = [item for item in event["startstops"] if item["action"] == "compute"]
        if len(compute) != 1:
            raise ValueError(f"expected one compute interval for {key!r}")
        seconds = compute[0]["stop"] - compute[0]["start"]
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid compute interval for {key!r}")
        durations[key] = round(seconds * 1_000_000)
    for key in dependencies:
        if key not in durations:
            raise ValueError(f"missing compute event for {key!r}; use a fresh, fully recorded graph")
    edges = sorted(
        [names[parent], names[child]]
        for child, parents in dependencies.items()
        for parent in parents
    )
    return {
        "time_unit": "microseconds",
        "durations": {names[key]: durations[key] for key in dependencies},
        "edges": edges,
    }


def capture(client, collection):
    """Compute one collection on a synchronous distributed.Client and capture it."""
    from distributed import get_task_stream

    def dependency_closure(root, dask_scheduler):
        # The output future keeps the optimized graph alive while we inspect it.
        pending = [root]
        graph = {}
        while pending:
            key = pending.pop()
            if key in graph:
                continue
            state = dask_scheduler.tasks[key]
            if state.has_lost_dependencies:
                raise ValueError(f"Dask forgot a dependency of {key!r}")
            parents = [parent.key for parent in state.dependencies]
            graph[key] = parents
            pending.extend(parents)
        return graph

    with get_task_stream(client) as stream:
        future = client.compute(collection, retries=0)
        value = future.result()
        dependencies = client.run_on_scheduler(dependency_closure, future.key)
    return value, workflow_record(dependencies, stream.data)


def demo_workflow():
    from dask import delayed

    documents = ("cache a search response", "plan a retrieval", "rank a search response")

    @delayed
    def tokenize(query):
        return query.split()

    @delayed
    def retrieve(terms, weight_first):
        scores = []
        for document in documents:
            words = document.split()
            matching = sum(term in words for term in terms)
            bonus = int(weight_first and words[0] == terms[0])
            scores.append(matching + bonus)
        return scores

    @delayed
    def merge(first, second):
        return max(range(len(documents)), key=lambda i: first[i] + second[i])

    plan = tokenize("rank search", dask_key_name="plan")
    first = retrieve(plan, False, dask_key_name="lexical")
    second = retrieve(plan, True, dask_key_name="prefix")
    return merge(first, second, dask_key_name="merge")


if __name__ == "__main__":
    from distributed import Client

    with Client(processes=False, n_workers=2, threads_per_worker=1, dashboard_address=None) as client:
        _, record = capture(client, demo_workflow())
    print(json.dumps(record))
