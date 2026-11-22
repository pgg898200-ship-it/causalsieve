# CausalSieve

**Find the upstream root cause in live multivariate streams in microseconds — not minutes. Compress 130,000 raw telemetry tokens down to 65 with bit-exact FP64 recovery.**

Zero runtime Python dependencies. Zero GPU required. C++20 + **AVX2/FMA** (x86-64) & **NEON** (Apple Silicon / ARM64) SIMD core (`1.69 µs/tick`), C99 ABI, Python API + Pandas/Polars/Prometheus/RCAEval adapters, sub-30 ms CLI, JSON-RPC 2.0 MCP server, and drop-in Agent Skill (`SKILL.md` + `.claude-plugin`).

> **Why CausalSieve exists:** When service `A` breaks and cascades `A -> B -> C` with amplification, raw Z-scores and univariate detectors flag downstream victim `C`, batch causal tools (`statsmodels`, `tigramite`, `AERCA`) take seconds to minutes, and dumping raw CSV/Prometheus windows into LLM agents burns 50k–350k tokens while getting fooled by loud heteroscedastic decoys.
>
> **CausalSieve** solves all three in one pass: exact online conditional Granger (`< 3e-12` vs batch FP64 refits), innovation precision whitening, online non-Gaussian contemporaneous orientation, anytime-valid Ville E-process alarms, and a 95%–99.98% token-compressed Causal Digest with lossless `slice://` recovery.

---

## Headline Verified Results

Measured live on **1× Intel Xeon @ 2.30 GHz (`avx2+fma`)** against frozen binary `libcausalsieve.so` (`sha256: 69e960d0...`):

| Benchmark | Metric | CausalSieve | Best Incumbent | Edge |
|---|---|---:|---:|---:|
| **Streaming Throughput (`D=16, p=2`)** | Steady-state rate | **287,259 ticks/s** (`3.48 µs`) | `tigramite` PCMCI `3,293 ms` | **518× faster** (`98×` vs `statsmodels`) |
| **Full `32×32` Conditional Granger + Precision** | `T=500` ingest + graph | **12.12 ms** (`0.043 ms` query) | `statsmodels` `24,070 ms` / `tigramite` `15,670 ms` | **1,987×** vs `statsmodels`, **1,293×** vs `tigramite`, **24.8×** vs NumPy |
| **Full `64×64` Conditional Granger + Precision** | `T=500` ingest + graph | **38.45 ms** (`0.088 ms` query) | `statsmodels` `1,879 s`† / NumPy `613 ms` | **48,875×** vs `statsmodels`, **16.0×** vs NumPy |
| **Real K8s Outages (`RCAEval` OB+SS Sealed, n=40)** | Service AC@1 / AC@5 / MRR | **67.5% / 100% / 0.804** (`33.3 ms`) | `pairwise_granger_z` `60.0% / 90% / 0.741` (`162 ms`) | **#1 across all 5 metrics**, 40/40 Top-5, **99.95% fewer tokens** (`131.7k -> 65`) |
| **Real 64-Service K8s (`RCAEval` Train Ticket, D≈217, n=25)** | Service AC@1 / AC@3 / MRR | **48.0% / 84.0% / 0.669** (`375 ms`) | `pairwise_granger_z` `20.0% / 32% / 0.344` (`2,127 ms` uncapped) | **2.4×–12× AC@1**, **99.98% fewer tokens** (`346.7k -> 64`), blind also **48.0%** |
| **Blind Real K8s Outages (ZERO `inject_time`, n=40 OB+SS)** | Service AC@1 / MRR | **65.0% / 0.746** (`30.1 ms`) | `rolling_mad_z` `52.5% / 0.712` (given `inject_time`) | Beats every baseline **without knowing when the outage started** |
| **100-Episode Multi-Regime Synthetic RCA** | Overall AC@1 / MRR | **73.0% / 0.830** (`13.3 ms`) | `IsolationForest` `38.0% / 0.615` (`2,907 ms`) | **+35.0 pts AC@1**, **219× faster**, **99.88% fewer tokens** |
| **Tier-2 `D=4,096` Loud-Decoy Firehose (20% 3.5×-noise)** | Throughput & Top-1 Recall | **19,577 ticks/s** (`80.2M obs/s`), **100%** | Raw Max-Z / MAD-Z / Var: **0%** | **100% vs 0% AC@1** at 80M obs/s on 1 core |
| **Mathematical Exactness Oracle** | Parity vs batch FP64 SVD + refits | **`< 3e-12`** across 19/19 cells | Batch FP64 reference | Exact algebraic equivalence (`Freivalds < 1e-11`) |

†`statsmodels` at `D=64` linearly extrapolated from 200 timed pairs out of 4,032. Full tables, honest tradeoffs, and reproduction commands in [BENCHMARKS.md](BENCHMARKS.md).

---

## 3-Line Quickstart

### One-Command Install (Zero Python Runtime Dependencies)
```sh
pip install git+https://github.com/pgg898200-ship-it/causalsieve.git   # auto-compiles & bundles native AVX2/NEON C++20 core (zero runtime deps)
causalsieve info                                                       # verify active SIMD ISA (avx2+fma / neon / scalar)
```

### CLI: Compress a 130k-Token Incident CSV into a ~300-Token Causal Digest
```sh
causalsieve rca metrics.csv --lag 2 --top-k 5 --format md
causalsieve counterfactual metrics.csv --channel db_cpu --delta 3.0 --horizon 10
causalsieve slice metrics.csv --channel db_cpu --start 490 --end 530   # 0.0-error FP64 recovery
```

### Python API & Drop-In Ecosystem Adapters (Zero Hard Dependencies)
```python
from causalsieve import CausalSieve, build_digest, analyze_dataframe, analyze_prometheus, rcaeval_rca

# 1. Streaming / batch core (zero-copy if NumPy float64 C-contiguous passed)
with CausalSieve(dim=16, lag=2, channel_names=[f"svc_{i}" for i in range(16)]) as sieve:
    sieve.push_batch(rows)                          # 287k ticks/s on 1 Xeon core
    attr = sieve.root_attribution()                 # ranked exogenous root causes (not victims)
    granger = sieve.granger_matrix()                # exact Block-Schur D x D F-stats & partial R^2
    cf = sieve.counterfactual_do("svc_0", 4.0, 10)  # quantitative do-intervention propagation
    digest = build_digest(sieve, top_k=5)           # 95-99.98% token reduction + lossless handles

# 2. Drop-in adapters for Pandas / Polars DataFrames, Prometheus query_range JSON, & RCAEval
_, df_digest = analyze_dataframe(df)                # works with pandas, polars, or dict-of-lists
_, prom_digest = analyze_prometheus(prom_json)      # Prometheus /api/v1/query_range matrix JSON
rca_out = rcaeval_rca(metrics_df, inject_time=t0)   # exact RCAEval {"ranks": [...]} drop-in
```

### MCP Server & Agent Skill (Claude Code, Codex, Gemini CLI, Cursor)
Add to your MCP config (zero dependencies — runs over JSON-RPC 2.0 stdio):
```json
{
  "mcpServers": {
    "causalsieve": {
      "command": "causalsieve-mcp",
      "args": ["--allowed-root", "/path/to/telemetry"]
    }
  }
}
```
Exposes 6 tools backed by a thread-safe multi-agent `CausalBlackboard`:
- `sieve_ingest_csv` — stream a CSV and return a 95%–99.95% compressed Causal Digest + `stream_id`
- `sieve_push_ticks` — push live rows and receive instant exogenous root-cause & Ville E-value alerts
- `sieve_causal_graph` — return pruned lagged Granger + contemporaneous partial-corr / LiNGAM edges
- `sieve_counterfactual` — simulate `do(channel += delta)` to verify root-cause hypotheses
- `sieve_recover_slice` — recover 100% lossless IEEE-754 FP64 raw points via `slice://...` handle
- `sieve_swarm_propose` — append & objectively score multi-agent hypotheses (no overwrite churn, proper-scoring consensus)

Install the drop-in Agent Skill (`SKILL.md`, `llms.txt`, `llms-full.txt`) or Claude Code plugin directly:
```sh
npx skills add pgg898200-ship-it/causalsieve
# or in Claude Code:
claude plugin marketplace add pgg898200-ship-it/causalsieve
```

---

## Why It's Fast and Accurate (The Math Moat)

1. **Exact Block-Schur Conditional Granger (`O(D p^3 + D^2 p^2)` vs `O(D^4 p^3)`):**
   All $D$ target channels share one inverse lagged Gram matrix $P_t = (Z_t^\top W_t Z_t + \alpha I)^{-1} \in \mathbb{R}^{K \times K}$ updated in $O(K^2)$ AVX2+FMA Sherman-Morrison steps. By the exact Schur complement identity, removing source channel $i$'s lag block $S_i$ increases the exact weighted ridge objective for target $j$ by:
   $$\Delta J_{i \to j} \equiv B_{S_i, j}^\top \left(P_{S_i, S_i}\right)^{-1} B_{S_i, j}$$
   Zero reduced-model refits. Exact to `< 3e-12` against batch FP64 SVD refits.

2. **Precision-Whitened Exogenous Attribution + Online Non-Gaussian LiNGAM Orientation:**
   Temporal lags are stripped via causal one-step innovations $\hat{u}_t = \tilde{x}_t - B_{t-1}^\top z_t$. Contemporaneous propagation is stripped via the online innovation precision matrix $\Theta_{t-1} = (\Sigma_{u,t-1} + \gamma_{\text{eff}} I)^{-1}$:
   $$z^{\text{exo}}_{j,t} = \frac{(\Theta_{t-1} \hat{u}_t)_j}{\sqrt{\Theta_{jj,t-1}}}$$
   Under skewed or heavy-tailed shocks, online Hyvärinen-Smith pairwise cumulant/$\tanh$ ratios ($O(D^2)$ streaming moments) orient contemporaneous edges $i \to j$ within the same tick — overcoming Granger causality's classic no-instantaneous-effects limitation.

3. **Production Numerical & Statistical Armor:**
   - **Shifted-anchor normalization + flat-channel guard:** Immune to ERDDAP `1e10` DC offsets (`100%` AC@1) AND constant-during-warmup 4 KB Kubernetes memory-page blips (`0 NaNs`).
   - **Mallows-Huber shock shield:** Prevents sustained outages from poisoning baseline $B_t$ and $\Theta_t$.
   - **Self-healing Freivalds drift guard:** Cheap $O(K)$ residual probe triggers automatic exact Cholesky refresh (`Freivalds < 1e-11` over 50k+ ticks).
   - **Anytime-valid ONS betting E-process:** Finite-sample Hotelling-$T^2$ adjusted supermartingale obeying Ville's inequality ($\mathbb{P}_{H_0}(\exists t : E_t \ge 1/\alpha) \le \alpha$ — zero peeking inflation).
   - **Tier-2 `ScreenedSieve` (`D = 128..16,384+`):** $O(D)$ AVX2 scale-free robust pre-whitening kills heteroscedastic loud decoys at `80M obs/s` and dynamically promotes top-$M$ candidates into the exact Tier-1 core.
   - **Certified `L_inf` Swing-Filter PLA + Lossless FP64 Ring:** Guarantees every compressed trajectory knot interpolates within `max_abs_err` while every retained tick is recoverable bit-for-bit (`0.0` error).

---

## Real-World Kubernetes Benchmark (`RCAEval`, 40 Sealed Holdout Cases)

Evaluated strictly one-shot with **zero retuning** on 40 sealed held-out microservice incidents (`RE1` Online Boutique & Sock Shop, reps 2–5; reps 1 = 10 DEV cases; full SHA-256 provenance in [`benchmarks/results/rcaeval.json`](benchmarks/results/rcaeval.json)):

| Method | AC@1 | AC@3 | AC@5 | Avg@5 | MRR | Mean ms/case |
|---|---:|---:|---:|---:|---:|---:|
| **`causalsieve`** (symmetric baseline) | **67.5%** (27/40) | **92.5%** (37/40) | **100.0%** (40/40) | **0.880** | **0.804** | **33.3 ms** |
| **`causalsieve_blind`** (zero `inject_time`) | **65.0%** (26/40) | 75.0% (30/40) | 90.0% (36/40) | 0.780 | **0.746** | **30.1 ms** |
| `pairwise_granger_z` | 60.0% (24/40) | 87.5% (35/40) | 90.0% (36/40) | 0.805 | 0.741 | 162.0 ms |
| `rolling_mad_z` (Baro-style) | 52.5% (21/40) | 90.0% (36/40) | 92.5% (37/40) | 0.815 | 0.712 | 3.6 ms |
| `raw_max_z` (3-sigma) | 40.0% (16/40) | 87.5% (35/40) | 92.5% (37/40) | 0.755 | 0.624 | 1.1 ms |
| `sklearn_isolation_forest` | 35.0% (14/40) | 60.0% (24/40) | 72.5% (29/40) | 0.565 | 0.520 | 5,661.3 ms |

---

## Honest Tradeoffs & Known Limits

We report failures and crossovers alongside wins:
- **Small offline one-shot (`D <= 16`):** If you never need streaming updates or per-tick exogenous scoring, a single offline NumPy Cholesky refit (`5.92 ms`) matches CausalSieve's full 500-tick streaming pass (`6.36 ms`). CausalSieve pulls ahead sharply at `D >= 32` (`24.8×` vs NumPy, `1,987×` vs `statsmodels`) and on every live streaming workload.
- **Single-tick Python FFI overhead:** Calling `.push(x)` from Python 1,000 times adds ~11 ms of `ctypes` call overhead; use `.push_batch(X)` (or C ABI / native `causalsieve-bin`) for full `287k–592k ticks/s` speed.
- **Purely nonlinear / non-monotone couplings:** Default linear `VAR(p)` misses pure ReLU/threshold cascades (AC@1 `0.000` on our synthetic saturation test); enabling `nonlinear_hinge=True` recovers `0.450` AC@1 (`0.685` MRR) on saturation cascades, at the cost of doubling `K` (`2pD+1`) and slightly lower finite-sample linear accuracy (`0.850` vs `1.000`).
- **Gaussian contemporaneous shocks:** Contemporaneous direction (`LiNGAM`) requires non-Gaussian innovations (skewed or heavy-tailed); on purely Gaussian instantaneous shocks, the undirected precision skeleton remains valid while orientation is unidentifiable by theory.
- **Packet-loss faults on `RCAEval` (`loss`):** Across all methods, network packet-loss faults are the hardest class (`causalsieve` AC@1 `37.5%`, `pairwise_granger_z` `50.0%`; `causalsieve` reaches `100%` within Top-5). On Sock Shop (`RE1-SS`) alone, `pairwise_granger_z` edges out `causalsieve` (`85%` vs `75%` AC@1) while trailing by 25 points on Online Boutique (`35%` vs `60%`) and running 4.9× slower.

---

## Reproduce Everything

```sh
make test-asan                                    # C++20 unit tests under ASan + UBSan (0 findings)
pytest tests/ -v                                  # 48/48 oracle, causality, RCA, adapters, digest, MCP
pytest tests/breaker_audit.py -v                  # 29/29 adversarial security, provenance & zero-dep audit
python3 benchmarks/bench_speed.py                 # Speed vs statsmodels, tigramite, sklearn, river, NumPy
python3 benchmarks/bench_optimization_log.py      # 5-stage compounding optimization ladder
python3 benchmarks/bench_synthetic_rca.py         # 100-episode multi-regime RCA
python3 benchmarks/bench_sift_scale.py            # Tier-2 D=128..4096 loud-decoy firehose
python3 benchmarks/bench_rcaeval.py               # Real Kubernetes RCAEval (10 DEV + 40 sealed holdout)
python3 benchmarks/bench_rcaeval_trainticket.py   # Real 64-service Train Ticket (25 incidents, D~217)
```

Open [`demo/causalsieve_explorer.html`](demo/causalsieve_explorer.html) locally in any browser (zero external requests / zero CDN) — or via GitHub Pages at [pgg898200-ship-it.github.io/causalsieve](https://pgg898200-ship-it.github.io/causalsieve/) — for an interactive causal cascade simulator and benchmark dashboard.

---

## License

MIT — see [LICENSE](LICENSE).
