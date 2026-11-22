#!/usr/bin/env python3
"""Convert one successful GitHub Actions run attempt to a PathProbe duration graph."""

import argparse
from datetime import datetime, timezone
from graphlib import TopologicalSorter
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPRedirectHandler, Request


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("manifest contains a duplicate JSON key")
        result[key] = value
    return result


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("manifest must be a nonempty object keyed by exact job display name")
    for name, parents in manifest.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("expected a nonempty job display name")
        if parents is None:
            continue
        if not isinstance(parents, list):
            raise ValueError("expected a list of prerequisites or null for a skipped job")
        if any(not isinstance(p, str) or p not in manifest for p in parents):
            raise ValueError("each prerequisite must name a job in the manifest")
        if any(manifest[p] is None for p in parents):
            raise ValueError("an executed job cannot depend on a skipped job")
        if len(parents) != len(set(parents)):
            raise ValueError("a manifest entry repeats a prerequisite")
    executed = {n: p for n, p in manifest.items() if p is not None}
    if not executed:
        raise ValueError("manifest needs at least one executed job")
    # Fails on cycles even when all participating jobs have zero duration.
    try:
        TopologicalSorter(executed).prepare()
    except ValueError:
        raise ValueError("manifest must be acyclic") from None


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def github_get(path, token=None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "PathProbe-capture",
               "X-GitHub-Api-Version": "2022-11-28"}
    request = Request("https://api.github.com" + path, headers=headers)
    if token:
        if not token.isascii() or not token.isprintable() or any(c.isspace() for c in token):
            raise ValueError("invalid GITHUB_TOKEN format")
        request.add_unredirected_header("Authorization", "Bearer " + token)
    opener = build_opener(RejectRedirects())
    try:
        with opener.open(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(f"GitHub API returned HTTP {error.code}") from None


def timestamp(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("job timestamps must use UTC")
    t = datetime.fromisoformat(value[:-1] + "+00:00")
    if t.microsecond or t.utcoffset() != timezone.utc.utcoffset(t):
        raise ValueError("expected whole-second UTC job timestamps")
    return t


def capture(repo, run_id, attempt, manifest, get=github_get, deadline=None):
    if not isinstance(repo, str) or not re.fullmatch(r"[a-zA-Z0-9-]+/[a-zA-Z0-9_.-]+", repo):
        raise ValueError("repository must have OWNER/REPO format")
    if repo.split("/")[1] in (".", ".."):
        raise ValueError("invalid repository name")
    if any(type(n) is not int or n < 1 for n in (run_id, attempt)):
        raise ValueError("run ID and attempt must be positive integers")
    if deadline is not None and (type(deadline) is not int or deadline < 0):
        raise ValueError("deadline must be an integer number of seconds >= 0")
    validate_manifest(manifest)
    base = f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}"
    run = get(base)
    if not isinstance(run, dict) or (run.get("id"), run.get("run_attempt")) != (run_id, attempt):
        raise ValueError("run metadata does not match the requested attempt")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("run attempt must have completed successfully")
    repository = run.get("repository")
    if not isinstance(repository, dict) or str(repository.get("full_name", "")).casefold() != repo.casefold():
        raise ValueError("run metadata returned a different repository")
    sha = run.get("head_sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("run metadata lacks a valid commit SHA")

    durations, observed, starts, ends = {}, {}, {}, {}
    seen_ids, seen_names = set(), set()
    skipped = {}
    page = 1
    while True:
        batch = get(f"{base}/jobs?per_page=100&page={page}")
        if not isinstance(batch, dict) or not isinstance(batch.get("jobs"), list):
            raise ValueError("unexpected jobs response")
        if batch.get("total_count") != len(manifest):
            raise ValueError("manifest must cover every reported job, including matrix instances")
        jobs = batch["jobs"]
        if not jobs:
            raise ValueError("incomplete jobs response")
        for job in jobs:
            if not isinstance(job, dict) or (job.get("run_id"), job.get("run_attempt"), job.get("head_sha")) != (run_id, attempt, sha):
                raise ValueError("job metadata does not match run, attempt and commit")
            name, job_id = job.get("name"), job.get("id")
            if not isinstance(name, str) or name not in manifest or name in seen_names:
                raise ValueError("unknown or duplicate job display name")
            if type(job_id) is not int or job_id <= 0 or job_id in seen_ids:
                raise ValueError("invalid or duplicate job ID")
            seen_names.add(name)
            seen_ids.add(job_id)
            if manifest[name] is None:
                if job.get("status") != "completed" or job.get("conclusion") != "skipped":
                    raise ValueError("manifest expected a skipped job, but GitHub reports a different outcome")
                skipped[name] = job_id
                continue
            if job.get("status") != "completed" or job.get("conclusion") != "success":
                raise ValueError("all jobs must have completed successfully, including conditional jobs")
            start, end = timestamp(job.get("started_at")), timestamp(job.get("completed_at"))
            if end < start:
                raise ValueError("job completion precedes start")
            starts[name], ends[name] = start, end
            durations[name] = int((end - start).total_seconds())
            observed[name] = {k: job[k] for k in ("id", "started_at", "completed_at")}
        if len(seen_names) == len(manifest):
            break
        if len(jobs) != 100:
            raise ValueError("job listing ended before all jobs were returned")
        page += 1

    edges = []
    for child, parents in manifest.items():
        if parents is None:
            continue
        for parent in parents:
            if ends[parent] > starts[child]:
                raise ValueError("recorded job timestamps contradict a manifest dependency")
            edges.append([parent, child])
    graph = {"durations": durations, "edges": edges, "source": {
        "run": f"https://github.com/{repo}/actions/runs/{run_id}", "attempt": attempt,
        "commit": sha, "unit": "seconds", "jobs": observed,
        "dependencies": "operator-supplied manifest",
    }}
    if skipped:
        graph["source"]["skipped_jobs"] = skipped
    if deadline is not None:
        graph["deadline"] = deadline
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", metavar="OWNER/REPO")
    parser.add_argument("run_id", type=int)
    parser.add_argument("attempt", type=int)
    parser.add_argument("manifest", type=Path, help="JSON map of job display name to prerequisite names")
    parser.add_argument("--deadline", type=int, help="include a modeled completion deadline in seconds")
    parser.add_argument("--use-token", action="store_true", help="read GITHUB_TOKEN from the environment")
    args = parser.parse_args()
    try:
        token = os.environ.get("GITHUB_TOKEN") if args.use_token else None
        if args.use_token and not token:
            raise ValueError("set GITHUB_TOKEN before using --use-token")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = capture(args.repository, args.run_id, args.attempt, manifest,
                         get=lambda path: github_get(path, token), deadline=args.deadline)
    except (ValueError, OSError, HTTPException) as error:
        parser.exit(1, f"capture: {error}\n")
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
