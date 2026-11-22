"""Rank metrics in a saved Prometheus query_range response or a synthetic incident."""

import argparse
import json
import random
from pathlib import Path

from causalsieve import analyze_prometheus


def demo_response():
    """Simulate a database delay that propagates to API latency one sample later."""
    rng = random.Random(2026)
    database = []
    api = []
    db_delay = api_delay = 0.0
    for tick in range(600):
        previous_db_delay = db_delay
        db_delay = 0.25 * db_delay + rng.gauss(0, 1)
        if tick >= 480:
            db_delay += 5.5
        api_delay = 0.65 * api_delay + 0.7 * previous_db_delay + rng.gauss(0, 0.35)
        timestamp = 1_700_000_000 + 5 * tick
        database.append([timestamp, f"{0.050 + 0.002 * db_delay:.6f}"])
        api.append([timestamp, f"{0.120 + 0.002 * api_delay:.6f}"])

    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": {"__name__": "latency_seconds", "service": "db"}, "values": database},
                {"metric": {"__name__": "latency_seconds", "service": "api"}, "values": api},
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path, nargs="?", help="Saved query_range JSON")
    args = parser.parse_args()
    payload = args.response.read_bytes() if args.response else demo_response()
    sieve, digest = analyze_prometheus(payload, top_k=3)
    try:
        print(json.dumps({
            "ticks": sieve.ticks,
            "series": sieve.dim,
            "candidates": [row.name for row in digest.root_causes],
        }, indent=2))
    finally:
        sieve.close()


if __name__ == "__main__":
    main()
