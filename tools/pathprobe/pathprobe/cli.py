"""Command-line access to single-run predictions and cohort deadline budgets."""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from .core import DurationIndex


def input_stream(name):
    return nullcontext(sys.stdin) if name == "-" else Path(name).open(encoding="utf-8")


def read_index(record):
    if not isinstance(record, dict) or not {"durations", "edges"}.issubset(record):
        raise ValueError("a workflow record needs durations and edges")
    return DurationIndex(record["durations"], record["edges"])


def evaluate(args):
    with input_stream(args.file) as stream:
        if args.command == "fleet":
            from .fleet import FleetBudget

            def runs():
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            index = read_index(record)
                            if args.task not in record["durations"]:
                                raise ValueError(f"missing task {args.task!r}")
                            deadline = record["deadline"]
                            if type(deadline) is not int or deadline < 0:
                                raise ValueError("deadline must be a nonnegative integer")
                        except (KeyError, ValueError, TypeError) as error:
                            raise ValueError(f"line {line_number}: {error}") from error
                        yield index, args.task, deadline

            fleet = FleetBudget.from_runs(runs())
            answer = dict(task=args.task, runs=fleet.total_runs, impossible=fleet.impossible_runs)
            if args.duration is not None:
                answer.update(duration=args.duration, met=fleet.met(args.duration))
            if args.required is not None:
                answer.update(required=args.required, limit=fleet.limit(args.required))
            return answer
        record = json.load(stream)
        index = read_index(record)
    if args.command == "predict":
        return {"task": args.task, "baseline": index.makespan, "replacement_duration": args.duration,
                "predicted": index.predict(args.task, args.duration), **index.profile(args.task)}
    profiles = (dict(task=name, **index.profile(name)) for name in record["durations"])
    ranked = sorted(profiles, key=lambda row: (-row["max_saving"], row["task"]))
    return {"baseline": index.makespan, "tasks": ranked[:args.top]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find the workflow latency savings available from one task.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="predict completion after one duration edit")
    predict.add_argument("file", help="graph JSON or - for stdin")
    predict.add_argument("task")
    predict.add_argument("duration", type=int)
    rank = commands.add_parser("rank", help="order tasks by the saving at zero duration")
    rank.add_argument("file", help="graph JSON or - for stdin")
    rank.add_argument("--top", type=int, default=10)
    fleet = commands.add_parser("fleet", help="query a replacement duration across historical workflow deadlines")
    fleet.add_argument("file", help="one graph per JSON line with an integer deadline, or - for stdin")
    fleet.add_argument("task")
    fleet.add_argument("--duration", type=int, help="count runs meeting deadline at this duration")
    fleet.add_argument("--required", type=int, help="find the largest duration meeting at least this many runs")
    args = parser.parse_args(argv)
    if args.command == "fleet" and args.duration is None and args.required is None:
        parser.error("fleet requires --duration, --required, or both")
    if args.command == "rank" and args.top < 0:
        parser.error("--top must be nonnegative")
    try:
        result = evaluate(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"pathprobe: {error}\n")
    print(json.dumps(result, allow_nan=False))
    return 0
