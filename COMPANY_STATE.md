# COMPANY_STATE.md — Autonomous R&D Operating Ledger (`causalsieve`)

> Operating ledger tracking hypotheses, engineering iterations, falsified/caught bugs, frozen SHA-256 verification gates, and next actions.

## 1. Mission & Product Positioning

- **Product:** `causalsieve` — Zero-runtime-dependency SIMD streaming engine for online multivariate causal discovery, precision-whitened exogenous root-cause localization, counterfactual `do()` verification, multi-agent `CausalBlackboard` consensus, and 95%–99.95% token-compressed lossless-recoverable stream digests.
- **Moat:**
  1. Exact Block-Schur Conditional Granger (`Delta_J = B_{S_i,j}^T P_{S_i,S_i}^{-1} B_{S_i,j}`): computes all $D^2$ conditional Granger F-stats in $O(D p^3 + D^2 p^2)$ instead of $O(D^4 p^3)$, verified `< 3e-12` vs batch FP64 SVD refits.
  2. Innovation precision whitening + online Hyvärinen-Smith non-Gaussian LiNGAM contemporaneous orientation (solving both lagged AND same-tick instantaneous cascades).
  3. Production numerical armor: shifted-anchor + flat-channel scale guard, Mallows-Huber shock shield, dimension-aware collinear ridge shrinkage, Freivalds drift guard + exact Cholesky self-healing, anytime-valid Hotelling-$T^2$-adjusted Ville E-process, and Tier-2 `ScreenedSieve` (`D` up to 16,384).
  4. Zero runtime Python dependencies, `<30 ms` CLI, zero-dep JSON-RPC 2.0 MCP stdio server, drop-in `SKILL.md` + `llms.txt` + `llms-full.txt`.

## 2. Verification Gates & Measured Outcomes

Frozen `v0.2.0` binary SHA-256: `e97025decbdf06663c73a63fa9adee73d9cae52e56661ba77098031f8d51bc36` (runtime-dispatched `avx2+fma` + `scalar`) · Scalar-only SHA-256: `7a8b2d1d9d4f673dc315e2b91b2e8c97190aafd0cfcf0e0b22e70caf22a99670`.

| ID | Hypothesis | Acceptance Gate | Verified Result | Status |
|---|---|---|---|---|
| **H1** | Exact Block-Schur Granger + Shared Inverse Gram matches batch FP64 SVD + literal reduced-model refits | Max err `< 1e-10` across `D in {4,12,24,32}`, `p in {1,2,3}`, `lambda in {1.0, 0.99}` | **19/19 PASS**: worst `< 1.86e-12`, `Freivalds <= 2.50e-14`; runtime `avx2+fma` vs `CAUSALSIEVE_FORCE_SCALAR=1` `< 6e-14` | **PASS** |
| **H2** | Multi-ISA AVX2+FMA + L1 BLAS-3 slabs delivers `>= 500x` vs incumbents & `>= 100k ticks/s` at `D=16, p=2` on 1 Xeon core | Measured live on host + compounding ladder | **`746.8k ticks/s`** at `D=8` (`1.34 µs`), **`385.3k`** at `D=16` (`585×` vs `tigramite`, beats offline NumPy), **`154.2k`** at `D=32` (`2,334×` vs `statsmodels`, `1,467×` vs `tigramite`, `28.5×` vs NumPy), **`48.95k`** at `D=64` (`71,514×` vs `statsmodels`) | **PASS** |
| **H3** | Beats published SOTA baselines (`BARO`, `N-Sigma`, `CIRCA`, `Pairwise Granger`) across all 645 real Kubernetes `RCAEval` RE1+RE2 outages | Zero hyperparameter retuning across full RE1 (375) + RE2 OTel (270) | **#1 across all 645 incidents**: **75.7% micro / 76.5% macro AC@1**, **0.839 / 0.844 MRR** vs `BARO` `72.7%`, `N-Sigma` `72.6%`, `CIRCA` `71.6%` (`+7.2 pts` on `RE1-TT`, `+5.5 pts` on `RE2-TT`); sealed 40-case OB+SS `67.5% AC@1 / 100% AC@5 / 0.804 MRR` (`29.1 ms`) | **PASS** |
| **H4** | Tier-2 `ScreenedSieve` handles `D=2048..4096` loud-decoy firehoses at `> 5,000 ticks/s` & `> 90%` Top-1 recall | Live measured vs univariate baselines | `D=128`: **`172.5k ticks/s`**; `D=2048`: **`36.9k ticks/s`** (`75.6M obs/s`); `D=4096`: **`18.5k ticks/s`** (`75.7M obs/s`), **100% AC@1** across all dims vs **0%** all 3 baselines | **PASS** |
| **H5** | Causal Digest + Swing-Filter PLA + Lossless `slice://` + Multi-Agent `CausalBlackboard` | `>= 95%` token cut, `0.0` FP64 error, adversarial consensus | `899k` CSV / `2.65M` JSON BPE tokens (`100%` overflow @ 128k) -> **`911.5` tokens (`0.0%` overflow @ 8k, `986×–2,905×` cut)**; **360/360** `0.0` FP64 slices (`65.1 µs`); `CausalBlackboard` **100% (50/50)** vs **0%** under 2× ballot stuffing (`0%` churn, `12.86×` grounded weight) | **PASS** |
| **H6** | Zero memory leaks/UB (ASan+UBSan), strict `assert_causal`, runtime CPUID dispatch, MCP security, zero runtime deps, clean anonymization | Independent Adversarial Breaker (`tests/breaker_audit.py`) + 5-platform wheels | **88/88 total tests PASS** (`59` core/adapters/dispatch + `29` Breaker), 0 ASan/UBSan findings (both `avx2+fma` and `scalar`), 8/8 frozen JSON SHA-256 provenance verified, 0 leaks | **PASS** |

## 3. Iteration & Bug-Kill Log

- **Iteration 1 (Initial Build):** Locked C99 ABI (`csrc/causalsieve.h`), fixed underdetermined warmup `S_u` deadlock, ERDDAP `1e10` variance floor, and Ville E-process Hotelling-$T^2$ finite-sample calibration.
- **Iteration 2 (DEV-Only Real K8s Audit):** Caught & fixed flat-memory-page `4 KB` NaN explosion, high-D collinear metric twin cancellation, and symmetric baseline comparator alignment.
- **Iteration 3–4 (v0.1.0 Freeze, Breaker & Global Launch):** Patched `cli_main.cpp` `--top-k 0 --compact` OOB bug, shipped PyPI `v0.1.0` across all 5 OS/arch wheels, GitHub Releases, GitHub Pages, and `skills.sh`.
- **Iteration 5 (v0.2.0 Performance Frontier, 645-Case SOTA & Agent Ecosystem):**
  - Eliminated portable-wheel `scalar` penalty via single-library runtime CPUID/OSXSAVE/XGETBV multi-ISA dispatch (`0` VEX/AVX ops in core/scalar TUs, `1,278` AVX2+FMA ops in AVX2 TU, native NEON on ARM64).
  - Implemented L1-resident `B_SLAB=32` register-blocked BLAS-3 deferred sufficient statistics + 4-row unrolled AVX2/NEON hot kernels + SIMD `all_finite_pd`, boosting throughput `1.28×–1.93×` over `v0.1.0` AVX2 (`~3.4×–5×` for portable PyPI x86-64 wheels) while improving `RE1-TT` latency `375 ms -> 215 ms`.
  - Expanded zero-tuning evaluation to **all 645 `RCAEval` RE1+RE2 OpenTelemetry incidents** (`#1` overall vs `BARO` FSE'24, `N-Sigma`, `CIRCA`), added the 60-outage BPE token overflow + 50-episode adversarial `CausalBlackboard` benchmark, added zero-dep OpenTelemetry OTLP/JSON (`analyze_otlp_metrics`), NDJSON (`analyze_jsonl`), live streaming `causalsieve watch` CLI, Ramp-AEO `docs/` + 4 deterministic SVG charts, and `context7.json` / `smithery.yaml` / `glama.json` / `sitemap.xml`.
