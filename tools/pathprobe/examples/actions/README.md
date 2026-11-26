# How fast must release checks run to finish in 15 minutes?

Three OpenAI Agents SDK for Python releases ran the same four GitHub Actions jobs. Each run’s published workflow declares that `checks` must finish before `build` and `release-notes` can start, and that `build` must finish before `publish` can start. GitHub recorded successful completions and finish-to-start ordering for every listed edge on run attempt 1.

```sh
python -m pathprobe fleet examples/actions/agents-sdk-releases.jsonl checks --duration 845 --required 2
```

**849 seconds** is the largest replacement duration for `checks` that would meet a 900-second dependency-only budget on two of these three runs. A replacement taking 845 seconds would meet the budget on the October runs. This is a retrospective scenario using an illustrative 15-minute target. There is no implied project SLA, performance forecast for future runs, or use of PathProbe by OpenAI.

| Release run | Checks, s | Build, s | Publish, s | Release notes, s | Observed span, s | Dependency length, s | Largest checks budget, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Oct 2](https://github.com/openai/openai-agents-python/actions/runs/37026168456) | 997 | 18 | 33 | 7 | 1,353 | 1,048 | 849 |
| [Oct 15](https://github.com/openai/openai-agents-python/actions/runs/38670443034) | 1,032 | 18 | 28 | 11 | 1,087 | 1,078 | 854 |
| [Nov 18](https://github.com/openai/openai-agents-python/actions/runs/44207283148) | 942 | 27 | 35 | 7 | 1,014 | 1,004 | 838 |

On October 2, five minutes elapsed between completion of the build and the start of publication. The model includes the measured duration of each job and omits waiting *between* jobs. Observed span runs from the first job starting through the last completing. Each timestamp has one-second precision.

## Provenance and reconstruction

On 2026-11-22 we selected the three latest successful release runs of `openai/openai-agents-python` with this four-job graph. An older revision had a different release graph. The dataset contains factual timestamps, job identifiers and three dependency pairs, with no upstream workflow scripts or logs.

For each row the JSONL includes the run URL, attempt number, head commit, Git blob ID for `.github/workflows/publish.yml`, and start/completion times by job. Match records on job name. Each pinned workflow has four literal job identifiers, no matrix and no job-level name override. Compare UTC completion minus UTC start to the stored duration, in integer seconds. Inspect the job-level `needs` declarations in each workflow revision to verify the edges.

| Run | Pinned workflow | Job timestamp records |
| --- | --- | --- |
| 37026168456 | [c4a1d04](https://github.com/openai/openai-agents-python/blob/c4a1d047b22488234b2ba81056e1b9b8db2127a5/.github/workflows/publish.yml) | [attempt 1](https://api.github.com/repos/openai/openai-agents-python/actions/runs/37026168456/attempts/1/jobs) |
| 38670443034 | [17d994b](https://github.com/openai/openai-agents-python/blob/17d994bd3045dae71acda676a4ef4762ed8ecd94/.github/workflows/publish.yml) | [attempt 1](https://api.github.com/repos/openai/openai-agents-python/actions/runs/38670443034/attempts/1/jobs) |
| 44207283148 | [0a74177](https://github.com/openai/openai-agents-python/blob/0a74177ed23ffb0c2dba24eda3fb12fb58e3e09d/.github/workflows/publish.yml) | [attempt 1](https://api.github.com/repos/openai/openai-agents-python/actions/runs/44207283148/attempts/1/jobs) |

## Which single job can rescue a five-minute CI budget?

The second fixture contains 12 consecutive successful pushes selected from the `openai/openai-python` main-branch CI workflow on November 18–20, 2026. All 12 were first attempts and used the same workflow Git blob. Each executed ten jobs. The compatibility matrix only runs for scheduled or manually dispatched events, so its skipped placeholder contributes no measured work. The provider matrix expands to two named jobs (browser and computer), and the test matrix to two named jobs (Python 3.10 and 3.14). The independent Python support policy job runs alongside dependency lock validation. The eight remaining jobs explicitly depend on lock validation. Each observed child started at or after its parent completed.

```sh
python -m pathprobe fleet examples/actions/python-sdk-ci.jsonl 'test (Python 3.10)' --duration 289 --required 9
```

At 289 seconds for the Python 3.10 test, **9 of 12** replays meet an illustrative 300-second dependency budget. At 290 seconds, only **6 of 12** meet it. Two runs miss even with an instantaneous Python 3.10 test, because lock validation plus the parallel Python 3.14 test took 314 and 307 seconds. All other recorded job durations remain fixed in these counterfactuals. The cohort covers successful runs over three calendar days.

Each row contains its immutable commit, job IDs and UTC timestamps. To check the original records, open the run URL in the row or use `GET /repos/openai/openai-python/actions/runs/{run_id}/attempts/1/jobs?per_page=100` on the GitHub REST API. The SHA of the pinned `.github/workflows/ci.yml` blob is `69a810f97e3cb5249e313240a6f85e6f5bf01b2b` for all 12 commits. [Workflow at the oldest sampled commit](https://github.com/openai/openai-python/blob/642d6d4ac5ada8fc4c0a0211dc2ac883b176fdbe/.github/workflows/ci.yml) · [314-second competing path](https://github.com/openai/openai-python/actions/runs/44256794303) · [307-second competing path](https://github.com/openai/openai-python/actions/runs/44609511528)

## Did the 289-second budget hold on later pushes?

Yes for the same nine-of-twelve target. We held the job, five-minute modeled deadline and 289-second duration fixed before replaying a later cohort. Raising the duration by one second missed the target in both cohorts.

| Successful main pushes | Met at 289 s | Met at 290 s | Miss even at 0 s | Largest duration meeting 9 of 12 |
| --- | ---: | ---: | ---: | ---: |
| November 18–20, 2026 | 9 of 12 | 6 of 12 | 2 | 289 s |
| November 23–25, 2026 | 11 of 12 | 7 of 12 | 0 | 289 s |

The later cohort's [one miss at 289 seconds](https://github.com/openai/openai-python/actions/runs/45468857904) spent 13 seconds on dependency lock freshness before the test could start, giving a modeled finish of 302 seconds. In the actual, unedited executions, six of the twelve runs spanned at most 300 seconds from first job start to last job completion. The table replaces one job duration and omits waiting between jobs.

On November 26 we took the latest twelve of the fourteen [successful main-branch pushes returned for November 21–25 UTC](https://api.github.com/repos/openai/openai-python/actions/workflows/ci.yml/runs?branch=main&event=push&status=success&created=2026-11-21..2026-11-25&per_page=12). All were first attempts. Two used the [earlier workflow](https://github.com/openai/openai-python/blob/9d7f58e0a56eb2bf04984357d80f7409c0cf11eb/.github/workflows/ci.yml). The other ten used a [revised workflow](https://github.com/openai/openai-python/blob/66266569c6aad060ac1b184d2d593bb1e90f2de6/.github/workflows/ci.yml) that changed the examples job's commands and kept the same job dependencies. The [trace file](python-sdk-ci-nov23-25.jsonl) includes each run URL, commit, workflow blob, job IDs, UTC timestamps and skipped matrix placeholder. The sample covers successful pushes on one repository.

Replay the later cohort offline from the PathProbe directory, or fetch the timings for the one miss with the same dependency manifest.

```sh
python -m pathprobe fleet examples/actions/python-sdk-ci-nov23-25.jsonl 'test (Python 3.10)' --duration 289 --required 9
python examples/actions/capture.py openai/openai-python 45468857904 1 examples/actions/python-sdk-ci-needs.json --deadline 300
```
