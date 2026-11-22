"""Reproduce SAPPHIRE CI durations, idealized savings and observed start gaps."""
import argparse
import concurrent.futures
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from urllib.request import Request, urlopen

REPO = "hydrosolutions/SAPPHIRE_flow"
API = f"https://api.github.com/repos/{REPO}"
WORKFLOW_BLOB = "129f2ffc61524666fd8c090d152f6fe91d115cc6"
SAMPLES = {
    45016316917: (610, "51caae70ede2055ae505a0b7ceda98382a4b3d73"),
    45010114942: (607, "bb15c9aead937c3bd06c55e99364aeb759cba3e8"),
    45002696542: (605, "f1085823a7542e90d7047d04ea4be4567402c669"),
    44997310687: (603, "390993157d04d7791eea096188e2c81fd15c3ea7"),
}
UNIT = ["unit (" + name + ")" for name in ("scripts", "services", "adapters-flows", "rest")]
INTEGRATION = ["integration (heavy)", "integration (rest)"]
EDGES = [(n, "unit-coverage") for n in UNIT] + [(n, "integration") for n in INTEGRATION]
JOBS = set(UNIT + INTEGRATION + ["integration", "unit-coverage", "lint", "wheel-only-guard",
                                "build-image-and-scan", "aquacast-release-image-proof"])
STEPS = {"Build Aquacast app image", "Trivy Aquacast image scan", "Qualify persistent private-profile installation",
         "Qualify disposable private-profile rehearsal", "Run integration tests"}
DATA = Path(__file__).with_name("runs.json")


def github(path):
    with urlopen(Request(API + path, headers={"User-Agent": "pathprobe-public-ci-study"}), timeout=45) as response:
        return json.load(response)


def download(run_id):
    pr, sha = SAMPLES[run_id]
    run = github(f"/actions/runs/{run_id}/attempts/1")
    if not (run["conclusion"] == "success" and run["run_attempt"] == 1 and
            run["head_sha"] == sha and run["event"] == "pull_request" and
            run["path"] == ".github/workflows/ci.yml"):
        raise ValueError(f"unexpected source run {run_id}")
    if github(f"/contents/.github/workflows/ci.yml?ref={sha}")["sha"] != WORKFLOW_BLOB:
        raise ValueError(f"job dependency definitions changed in {sha}")
    jobs = github(f"/actions/runs/{run_id}/attempts/1/jobs?per_page=100")
    if jobs["total_count"] != len(jobs["jobs"]):
        raise ValueError("incomplete jobs response")
    rows = {}
    for job in jobs["jobs"]:
        if job["name"] == "ci-attest" and job["conclusion"] == "skipped":
            continue
        if job["conclusion"] != "success" or job["name"] in rows:
            raise ValueError(f"unexpected job {job['name']!r}")
        rows[job["name"]] = {k: job[k] for k in ("id", "started_at", "completed_at")}
        rows[job["name"]]["steps"] = [
            {k: step[k] for k in ("name", "started_at", "completed_at")}
            for step in job["steps"] if step["conclusion"] == "success" and step["name"] in STEPS
        ]
    if set(rows) != JOBS:
        raise ValueError(f"unexpected jobs in run {run_id}")
    return {"run_id": run_id, "pr": pr, "head_sha": sha, "attempt": 1,
            "workflow_blob": WORKFLOW_BLOB, "created_at": run["created_at"], "jobs": rows}


def seconds(a, b):
    return int((datetime.fromisoformat(b.replace("Z", "+00:00"))
                - datetime.fromisoformat(a.replace("Z", "+00:00"))).total_seconds())


def model(run):
    from pathprobe import DurationIndex

    jobs = run["jobs"]
    durations = {job: seconds(row["started_at"], row["completed_at"]) for job, row in jobs.items()}
    index = DurationIndex(durations, EDGES)
    start_times = [j["started_at"] for j in jobs.values()]
    finish = max(j["completed_at"] for j in jobs.values())
    savings = {job: index.profile(job)["max_saving"] for job in jobs}
    lags = {}
    for job, row in jobs.items():
        predecessors = [a for a, b in EDGES if b == job]
        ready = max((jobs[p]["completed_at"] for p in predecessors), default=run["created_at"])
        lags[job] = seconds(ready, row["started_at"])
    proof = "aquacast-release-image-proof"
    image_step = next(s for s in jobs[proof]["steps"] if s["name"] == "Build Aquacast app image")
    return {"if_proof_120s_faster": index.makespan - index.predict(proof, durations[proof] - 120),
            "app_image_build": seconds(image_step["started_at"], image_step["completed_at"]),
            "proof_or_integration_floor": max(durations[n] for n in INTEGRATION) + durations["integration"],
            "run": run["run_id"], "pr": run["pr"], "creation_to_last_job": seconds(run["created_at"], finish),
            "first_start_to_last_job": seconds(min(start_times), finish), "modeled": index.makespan,
            "sum_job_seconds": sum(durations.values()), "max_saving": savings, "start_gaps": lags,
            "durations": durations}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="re-fetch the four pinned first attempts")
    options = p.parse_args()
    if options.refresh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(download, SAMPLES))
        DATA.write_text(json.dumps(runs, indent=2) + "\n")
    else:
        runs = json.loads(DATA.read_text())
    results = [model(run) for run in runs]
    print(json.dumps({"runs": results, "median": {
        key: median(row[key] for row in results) for key in
        ("creation_to_last_job", "first_start_to_last_job", "modeled", "sum_job_seconds")}}, indent=2))


if __name__ == "__main__":
    main()
