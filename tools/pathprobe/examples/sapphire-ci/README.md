# Where SAPPHIRE can recover CI time

Four successful pull-request runs on November 22 took **21:02 to 29:20** from workflow creation to the last job completing. In all four, shortening any unit-test shard alone would have saved **zero** seconds under the declared dependency graph. The Aquacast release-image proof finished on the longest modeled path, tied with integration on one run. Its app-image build step took **7:13 to 11:55**. That build is the first step to profile for shorter PR turnaround. [Owner’s request, issue #584](https://github.com/hydrosolutions/SAPPHIRE_flow/issues/584)

| PR and first-attempt run | Observed CI | Modeled CI | App-image build step | Longest path avoiding image proof | Maximum saving from image proof | Saving if image proof were 2:00 faster |
|---|---:|---:|---:|---:|---:|---:|
| [#610 · 45016316917](https://github.com/hydrosolutions/SAPPHIRE_flow/actions/runs/45016316917/attempts/1) | 29:20 | 29:18 | 11:55 | 16:56 | 12:22 | 2:00 |
| [#607 · 45010114942](https://github.com/hydrosolutions/SAPPHIRE_flow/actions/runs/45010114942/attempts/1) | 22:01 | 21:58 | 7:25 | 20:05 | 1:53 | 1:53 |
| [#605 · 45002696542](https://github.com/hydrosolutions/SAPPHIRE_flow/actions/runs/45002696542/attempts/1) | 21:02 | 21:00 | 7:30 | 17:06 | 3:54 | 2:00 |
| [#603 · 44997310687](https://github.com/hydrosolutions/SAPPHIRE_flow/actions/runs/44997310687/attempts/1) | 22:31 | 22:26 | 7:13 | 22:26 | 0:00 | 0:00 |

Times are minutes:seconds. The median observed CI time was 22:16. The last three columns describe independent, single-job changes with all other durations held fixed. The integration (rest) shard and its required gate make up the path avoiding the image proof in all four runs.

## Which experiment to run

Record per-layer BuildKit timings for the Aquacast app image on repeated runs of one pinned revision with recorded cache states. Identify the stage responsible for the 4:42 spread between the fastest and slowest app-image builds in this sample. The pinned workflow sets per-SHA version and revision arguments, loads the image, verifies its identity and scans it. Check where a change in those arguments invalidates expensive layers before modifying the Dockerfile. Leave release identity, scanning and rehearsals enabled during measurement. The layer measurements can establish whether caching accounts for the spread. [Build and verification steps](https://github.com/hydrosolutions/SAPPHIRE_flow/blob/51caae70ede2055ae505a0b7ceda98382a4b3d73/.github/workflows/ci.yml#L1646-L1713)

For a sub-20-minute target on all four patterns, image work alone cannot suffice. On #607 the integration branch would still exceed 20 minutes by five seconds; on #603 it would exceed it by 2:26. The rest shard's actual integration-test step consumed 15:32 to 21:10, versus 13:52 to 14:39 for the **entire** heavy shard. Inspect per-test timings and fixtures for a repeatable rebalance within the existing two-shard layout. Account for setup time and validate both shards. Runner count and wall-time savings require a controlled before/after trial.

## Scope and audit trail

These are four recent merged, code-changing PRs with successful first attempts at different head revisions on one day. Each had the same CI workflow definition at its PR head, 12 successful jobs, and the expected skipped push-only attestation. The model expands all four unit shards into the coverage gate and both integration shards into the integration gate. Other executed jobs have no declared job dependencies. It assumes immediately available runners and uses each job's full elapsed time, including setup and teardown. Occasional cold/warm profiles, cancellations, failures, repeated attempts and the separate dependency-safety workflow would require a broader sample. We do not estimate a population p95 from four runs. [Pinned workflow](https://github.com/hydrosolutions/SAPPHIRE_flow/blob/51caae70ede2055ae505a0b7ceda98382a4b3d73/.github/workflows/ci.yml)

For a root job, subtract workflow creation from its start. For a dependent job, subtract the latest completed prerequisite from its start. Across the 48 executed jobs these observed start gaps were 2–3 seconds. These timestamps cannot separate scheduler, queue and provisioning components. The sum of all job durations was 84:53–95:00 per run, with median 90:01.5; billing requires the applicable runner rates and rounding rules. An operator can use this baseline to compare a proposed change on the same coverage, runner configuration and test data. Actual end-to-end improvement, cache-hit rates and billing impact remain unmeasured.

## Reproduction

Use a checkout of PathProbe on the Python path or install the package from that checkout with `python -m pip install .`. In this pilot directory run:

```sh
python analyze.py
```

The script reads the supplied `runs.json` offline. To re-fetch the same four immutable first-attempt selections, use `python analyze.py --refresh`. This reads GitHub's public Actions run and jobs endpoints, checks head SHAs and the CI file's Git blob SHA `129f2ffc61524666fd8c090d152f6fe91d115cc6`, and rewrites the normalized snapshot. Each row includes the run ID, PR number, head SHA, timestamped jobs, job IDs and selected step timestamps. No job logs or credentials are required. GitHub may expire historical Actions data.

Source endpoint templates use `https://api.github.com/repos/hydrosolutions/SAPPHIRE_flow` with `/actions/runs/{id}/attempts/1`, `/actions/runs/{id}/attempts/1/jobs?per_page=100` and `/contents/.github/workflows/ci.yml?ref={head_sha}`. All durations use GitHub's UTC second-resolution timestamps.
