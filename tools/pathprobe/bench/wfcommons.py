"""Regenerate two compact DAG fixtures from SHA-256-pinned WfCommons executions."""
import argparse
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "e1600aa63731fb3fcdfeee2f1eee7b30a5036378"
BASE = f"https://raw.githubusercontent.com/wfcommons/WfInstances/{REVISION}"
SOURCES = {
    "cycles": (
        "pegasus/cycles/cycles-chameleon-1l-2c-9p-001.json",
        "821efe0ebde60334df09d5d5d43fa0dcd4ccaed2c33e59e585305b34bf21ee3c",
    ),
    "montage": (
        "pegasus/montage/montage-chameleon-dss-10d-001.json",
        "485fd0ce5c7cae88ec8ed5c22fb9197d51d49c7b3885ca119ddd57165716887d",
    ),
}


def project(data):
    workflow = json.loads(data, parse_float=Decimal)["workflow"]
    specification = workflow["specification"]["tasks"]
    # Stable numeric strings keep the compact fixtures independent of task-name length.
    ids = {task["id"]: str(i) for i, task in enumerate(specification)}
    execution = workflow["execution"]["tasks"]
    if (len(ids) != len(specification) or len(ids) != len(execution)
            or ids.keys() != {task["id"] for task in execution}):
        raise ValueError("expected one measured runtime per unique specified task")
    durations = {}
    for task in execution:
        seconds = Decimal(task["runtimeInSeconds"])
        if not seconds.is_finite() or seconds < 0:
            raise ValueError("expected a finite, nonnegative runtime")
        micros = (seconds * 1_000_000).to_integral_value(rounding=ROUND_HALF_EVEN)
        durations[ids[task["id"]]] = int(micros)
    edges = [
        [ids[parent], ids[task["id"]]]
        for task in specification
        for parent in task.get("parents", [])
    ]
    return {"durations": durations, "edges": edges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        help="read original JSONs by their upstream basename, offline")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (path, digest) in SOURCES.items():
        if args.source_dir:
            content = (args.source_dir / Path(path).name).read_bytes()
        else:
            with urlopen(f"{BASE}/{path}", timeout=30) as response:
                content = response.read()
        if sha256(content).hexdigest() != digest:
            raise ValueError(f"source checksum mismatch for {name}")
        graph = project(content)
        graph["source"] = {"url": f"{BASE}/{path}", "sha256": digest,
                           "unit": "microseconds", "license_file": "COPYING.LESSER"}
        destination = args.output_dir / f"{name}.json"
        destination.write_text(json.dumps(graph, separators=(",", ":")) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
