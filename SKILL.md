---
name: causalsieve
description: Run sub-millisecond online multivariate causal discovery, exogenous root-cause localization, counterfactual do-interventions, and 95-99.95% token-compressed lossless-recoverable digests on CSVs, DataFrames, Prometheus query_range responses, and live telemetry streams instead of dumping raw metrics into LLM context.
---

# CausalSieve Agent Skill

Use `causalsieve` (Python API, CLI, or MCP stdio server) whenever you need to diagnose anomalies, localize upstream root causes, inspect causal propagation graphs, or compress multivariate numeric time-series telemetry into a compact LLM context window without losing exact IEEE-754 FP64 data.

## When to Use

- A multivariate telemetry CSV, DataFrame, or Prometheus `query_range` window is too large to paste into context (500–100,000+ rows).
- Univariate Z-scores or raw magnitude alarms flag a loud downstream victim instead of the upstream exogenous root cause.
- You need to verify a root-cause hypothesis quantitatively using a counterfactual `do(channel += delta)` rollout before recommending remediation.
- Multiple agents are investigating the same incident stream concurrently and need immutable snapshots plus objective, append-only hypothesis scoring.
- You need exact IEEE-754 FP64 raw points only for a narrow incident window (`slice://<stream_id>/<channel>/<start>..<end>`).

## When NOT to Use

- Unstructured text logs, stack traces, or categorical event streams without numeric channels.
- Streams with fewer than `max(20, 3 * dim * lag)` ticks (insufficient degrees of freedom to estimate `VAR(p)` and innovation precision).
- Strongly periodic non-stationary signals where causal lags exceed `p = 16` without prior differencing or resampling.
- Contemporaneous causal orientation (`LiNGAM`) on purely Gaussian innovations (Gaussian contemporaneous shocks are observationally unorientable; undirected partial-correlation edges remain valid, while direction requires non-Gaussianity).

## Install

Zero runtime Python dependencies (stdlib + ctypes native shared library; optional zero-copy NumPy buffer support and duck-typed Pandas/Polars support):

```sh
pip install causalsieve          # prebuilt native wheel (Linux / macOS / Windows)
causalsieve info                 # verify version, active SIMD ISA, and bundled library path
```

Or run directly without a persistent environment using `uvx`:

```sh
uvx causalsieve info
uvx causalsieve rca telemetry.csv --lag 2 --top-k 5 --format md
```

To build from source:

```sh
pip install -e .                 # compiles csrc/causalsieve.cpp automatically via setup.py
```

## Standard Workflow: Digest -> Verify -> Recover Slice

Do **not** dump raw CSV rows or Prometheus JSON into LLM context. Follow this 3-step loop:

1. **Generate a token-compressed Causal Digest (~100–500 tokens, 95–99.95% smaller than raw telemetry):**
   ```sh
   causalsieve rca telemetry.csv --lag 2 --top-k 5 --format md
   ```
   Or cap strictly to a token budget:
   ```sh
   causalsieve digest telemetry.csv --tokens-budget 400
   ```
   Inspect the `Root Causes` table (`peak_exo_z` vs `raw_z` and `verdict`), the pruned `Causal Graph` (`lagged F`/`pR2` and `contemp pcorr`/`LiNGAM`), and the `PLA Knots` (`max_err<=...`).

2. **Verify the top root-cause candidate via counterfactual `do()` intervention:**
   ```sh
   causalsieve counterfactual telemetry.csv --channel db_cpu --delta 3.0 --horizon 10
   ```
   Confirm that intervening on the candidate propagates downstream to the observed victim channels.

3. **Recover 100% lossless exact FP64 values only for the narrow onset window:**
   ```sh
   causalsieve slice telemetry.csv --channel db_cpu --start 490 --end 530 --format json
   ```

## Drop-In Python Ecosystem Adapters

All adapters are duck-typed with zero hard external dependencies (`pandas`, `polars`, and `numpy` are optional):

### DataFrames & Dict-of-Lists (`analyze_dataframe`)

Works with Pandas DataFrames, Polars DataFrames, or `Mapping[str, Sequence[float]]`. Automatically skips timestamp and non-numeric columns, causally forward-fills isolated NaNs (zero look-ahead), and automatically activates Tier-2 screening when `D >= 128`:

```python
from causalsieve import analyze_dataframe

sieve, digest = analyze_dataframe(df, lag=2, forgetting=0.995, top_k=5)
print(digest.to_markdown())
```

### Prometheus `/api/v1/query_range` (`analyze_prometheus`)

Parses standard Prometheus matrix JSON (dict or raw JSON string), aligns timestamps causally across series with forward-fill, and formats clean `metric{label=val}` channel names:

```python
from causalsieve import analyze_prometheus

sieve, digest = analyze_prometheus(prom_json, lag=2, forgetting=0.995, top_k=5)
print(digest.to_markdown())
```

### OpenTelemetry OTLP/JSON (`analyze_otlp_metrics`) & NDJSON (`analyze_jsonl`)

Parse OpenTelemetry `ExportMetricsServiceRequest` JSON (`gauge`, `sum`, `histogram`) or streaming JSONL/NDJSON directly with zero third-party libraries:

```python
from causalsieve import analyze_otlp_metrics, analyze_jsonl

sieve, digest = analyze_otlp_metrics(otlp_json, lag=2, top_k=5)
sieve2, digest2 = analyze_jsonl("metrics.jsonl", lag=2, top_k=5)
```

### Microservice Root-Cause Localization (`rcaeval_rca`)

Drop-in function matching the `RCAEval` method interface (`<service>_<metric>` columns plus optional `"time"` column). Supports symmetric baseline/incident split via `inject_time` or fully blind mode when `inject_time=None`:

```python
from causalsieve import rcaeval_rca

res = rcaeval_rca(df, inject_time=1700000300, lag=2, forgetting=0.995)
print(res["ranks"][:5])          # ranked service names
print(res["scores"])             # {service: score}
print(res["digest"].to_markdown())

blind_res = rcaeval_rca(df, inject_time=None)  # zero knowledge of incident start
```

## MCP Server Configuration

Launch the zero-dependency JSON-RPC 2.0 MCP stdio server immediately via `uvx` (zero install) or locally via `causalsieve-mcp`:

```json
{
  "mcpServers": {
    "causalsieve": {
      "command": "uvx",
      "args": ["--from", "causalsieve", "causalsieve-mcp", "--allowed-root", "/path/to/telemetry"]
    }
  }
}
```

### MCP Tools

| Tool | Purpose | Key Required Args |
|---|---|---|
| `sieve_ingest_csv` | Stream a CSV file and return a 99%+ compressed Causal Digest + `stream_id` | `path` |
| `sieve_push_ticks` | Push live `(n_ticks, D)` rows into a stateful stream and get E-value / root alerts | `stream_id`, `rows` |
| `sieve_causal_graph` | Return pruned conditional Granger + contemporaneous partial-corr / LiNGAM edges | `stream_id` |
| `sieve_counterfactual` | Simulate `do(channel += delta)` for `horizon` steps to verify propagation | `stream_id`, `channel`, `delta` |
| `sieve_recover_slice` | Recover exact FP64 raw points by `handle` (`slice://...`) or `(stream_id, channel, start_tick, end_tick)` | `handle` or `(stream_id, channel, start_tick, end_tick)` |
| `sieve_swarm_propose` | Append & objectively score a root-cause hypothesis on the shared `CausalBlackboard` | `stream_id`, `agent_id`, `candidate_channel` |

## Multi-Agent Swarm Protocol

When multiple agents analyze the same stream:
1. One agent (or ingest loop) registers the stream via `sieve_ingest_csv` or `sieve_push_ticks` and shares only the `stream_id` and Causal Digest (~100–400 tokens).
2. Each specialist agent submits its candidate root cause via `sieve_swarm_propose(stream_id, agent_id, candidate_channel, delta, confidence, rationale)`.
3. Hypotheses are append-only (no peer overwrites) and scored strictly by engine exogenous attribution + counterfactual propagation fit + Brier/log proper scoring — not prose persuasion.
4. Read `consensus.winner_name`, `consensus.winner_probability`, and `consensus.agent_weights` from the returned JSON to decide next actions.

## CLI Quick Reference

```sh
causalsieve info [--json]
causalsieve rca <file.csv> [--lag 2] [--top-k 5] [--format md|json] [--hinge] [--max-err 0.05]
causalsieve digest <file.csv> [--tokens-budget 400] [--format md|json]
causalsieve slice <file.csv> --channel <name> --start <t0> --end <t1> [--format json|csv|table]
causalsieve counterfactual <file.csv> --channel <name> --delta <val> [--horizon 10] [--format md|json]
causalsieve watch [file|-] [--input-format auto|csv|jsonl] [--lag 2] [--alpha 0.01] [--format jsonl|md]
causalsieve mcp [--allowed-root <dir>] [--max-csv-bytes <bytes>]
```

## Known Limits

- `raw_ring_capacity` defaults to `8192` ticks per channel (auto-sized up to `65536` for `analyze_csv` and `analyze_dataframe`). Ticks evicted from the ring buffer cannot be recovered via `slice://`.
- Tier-1 full precision + Block-Schur Granger is designed for `D < 128` channels; for `D = 128..16,384+`, `analyze_csv`, `analyze_dataframe`, and `analyze_prometheus` automatically switch to Tier-2 `ScreenedSieve` (or pass `use_screened=False` / `--no-screened` to force Tier-1).
- `VAR(p)` assumes locally linear dynamics (or asymmetric saturation with `--hinge` / `nonlinear_hinge=True`); abrupt structural regime changes require forgetting (`--forgetting < 1.0`, default `0.995`) to adapt.
- Nonlinear non-monotone couplings (e.g., XOR or purely quadratic coupling without linear/hinge projection) are not captured by `VAR(p)` weights.
