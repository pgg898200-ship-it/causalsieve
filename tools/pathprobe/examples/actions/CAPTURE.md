# Replay your GitHub Actions run

Use Python 3.9 or newer to capture a completed, successful attempt as one PathProbe JSON record. This command reads public GitHub REST metadata and writes the result to stdout.

```sh
python examples/actions/capture.py openai/openai-agents-python 44207283148 1 \
  examples/actions/release-needs.json --deadline 900 > release.json
python -m pathprobe rank release.json --top 4
```

From the job-level `needs` in the workflow used by your run, create a JSON object keyed by exact GitHub job **display name**. Map a root job to `[]`, a dependent job to the names of all its prerequisites, and an expected skipped job to `null`. Include every job reported by this attempt. Expand each executed matrix instance to its displayed name. For a dependency on a matrix, include each required instance. Duplicate display names are ambiguous and the capture fails.

The manifest supplies the complete finish-to-start dependencies. Timestamp ordering can disprove a proposed edge but cannot prove that an edge exists or reveal an omitted one. Check the workflow revision before reusing a manifest. The tool rejects unsuccessful runs, failed jobs, unexpected skips, mismatched attempt/commit, cycles, and included jobs that depend on a skipped job. An expected skip must be reported by GitHub and is recorded by name and job ID in provenance.

```sh
python examples/actions/capture.py hydrosolutions/SAPPHIRE_flow 45016316917 1 \
  examples/actions/sapphire-needs.json > sapphire.json
python -m pathprobe rank sapphire.json --top 3
```

Durations are whole elapsed seconds inside each job. Queueing and waiting between jobs are excluded. The output includes run URL, attempt, commit, job identifiers, job timestamps and an optional modeled deadline. Redirect multiple successful captures into a JSONL file for `pathprobe fleet`; use the same units and a consistent job display name across the cohort. Capture emits one compact JSON line only after checking every job.

Anonymous requests are the default. For an account's accessible private workflows or authenticated quota, provide `--use-token` and set `GITHUB_TOKEN` through your environment. The token is sent only to `https://api.github.com`; redirects fail. Graph output includes job names and repository identity. Store private workflow outputs accordingly. No job logs, step payloads or credentials are included.

Run the offline checks with `python -m unittest discover -s examples/actions -p test_capture.py`.
