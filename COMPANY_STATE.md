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

Frozen binary SHA-256: `69e960d01e22f976a907e6de4f35ccadc048f07b8ac11f1dae481eaab5cdc6e6` (`avx2+fma`) · Scalar SHA-256: `a38bcb0d3ad6aa1c9f034e4bf5bd31ce20f5700df1ef37750d401b5c871984b9`.

| ID | Hypothesis | Acceptance Gate | Verified Result | Status |
|---|---|---|---|---|
| **H1** | Exact Block-Schur Granger + Shared Inverse Gram matches batch FP64 SVD + literal reduced-model refits | Max err `< 1e-10` across `D in {4,12,24,32}`, `p in {1,2,3}`, `lambda in {1.0, 0.99}` | **19/19 PASS**: `B=1.81e-12`, `F_rel=2.80e-12`, `R2=1.35e-13`, `Freivalds=9.64e-12`; AVX2 vs scalar `< 3e-15` | **PASS** |
| **H2** | AVX2+FMA delivers `>= 500x` vs incumbents & `>= 100k ticks/s` at `D=16, p=2` on 1 Xeon core | Measured live on host + compounding ladder | **`287,259 ticks/s`** at `D=16` (`518x` vs `tigramite`), **`101,490 ticks/s`** at `D=32` (`1,987x` vs `statsmodels`, `1,293x` vs `tigramite`, `24.8x` vs NumPy), **`48,875x`** vs `statsmodels` at `D=64` | **PASS** |
| **H3** | Beats all open-source baselines on 100-ep synthetic RCA AND zero-leakage sealed real Kubernetes `RCAEval` holdouts (`n=40`) | Strictly higher AC@1 / MRR on sealed holdouts with zero retuning | Synthetic: **73.0% AC@1 / 0.830 MRR** (+35 pts vs `IsolationForest` @ 219x speed). Real `RCAEval` 40-case sealed holdout: **#1 on every metric — 67.5% AC@1, 92.5% AC@3, 100.0% AC@5 (40/40), 0.880 Avg@5, 0.804 MRR** (`33.3 ms`); blind (no `inject_time`) **65.0% AC@1 / 0.746 MRR** still beats every baseline | **PASS** |
| **H4** | Tier-2 `ScreenedSieve` handles `D=2048..4096` loud-decoy firehoses at `> 5,000 ticks/s` & `> 90%` Top-1 recall | Live measured vs univariate baselines | `D=2048`: **`33,334 ticks/s`** (`68.3M obs/s`), **100% AC@1**. `D=4096`: **`19,577 ticks/s`** (`80.2M obs/s`), **100% AC@1** (vs **0%** for all 3 baselines) | **PASS** |
| **H5** | Causal Digest + Swing-Filter PLA + Lossless Slice Store achieves `>= 95%` token reduction & `0.0` FP64 slice error | Verified in `tests/test_digest_and_mcp.py` + real `RCAEval` | **97.7%–99.95%** token reduction (`131.7k -> 65` tokens on `RCAEval`), **0.0 bit-exact** FP64 recovery, exact `L_inf` PLA verified at every integer tick | **PASS** |
| **H6** | Zero memory leaks/UB (ASan+UBSan), strict `assert_causal`, ERDDAP `1e10`, MCP security, zero runtime deps, clean anonymization | Independent Adversarial Breaker (`tests/breaker_audit.py`) | **29/29 Breaker PASS** (**69/69 total**), 0 ASan/UBSan findings, bare `/usr/bin/python3` zero-dep verified, 0 leaks, 100% SHA-256 provenance | **PASS** |

## 3. Iteration & Bug-Kill Log

- **Iteration 1 (Initial Build):** Locked C99 ABI (`csrc/causalsieve.h`), launched 3 parallel workers (SIMD C++20 kernel, Python/Digest/MCP/Skill, Verification/Benchmarks). First pytest run hit **35/40 PASS (19/19 exact oracle PASS)** and caught 3 critical C++ bugs before benchmarking:
  - *Bug A:* Underdetermined warmup (`N < K` for `D >= 12`) overfitted `S_u -> 0`, deadlocking Huber shock shield at `w_t = 0.0`. Fixed with dof-ratio `S_u` shrinkage + predictive leverage + early anti-deadlock floor (`frac(w_t==0)` dropped `1.000 -> 0.0000–0.0080`).
  - *Bug B:* ERDDAP `1e10` DC-offset variance floor `1e-6 * |anchor|` crushed `O(1)` variance. Fixed to `max(1e-6, 1e-13 * |anchor|)` (`max_pr2_diff = 7.7e-8`).
  - *Bug C:* Ville E-process finite-sample Hotelling-$T^2$ tail caused 59.5% null FWER. Fixed with exact Hotelling-$T^2$/Student-$t$ dof adjustment + conservative tail budget (`FWER <= 0.05` across 200 null streams).
- **Iteration 2 (DEV-Only Real Kubernetes Telemetry Audit — Holdout Sealed):** Running `RCAEval` DEV (`n=10`) caught 3 genuine real-world production bugs:
  - *Real-Data Bug 1:* Flat-during-warmup `42 MB` Kubernetes RSS memory allocating one 4 KB page caused `4096 / 1e-6 = 4e9 sigma -> NaN` in 9 ticks. Fixed with relative flat-channel fallback scale `max(0.01, 0.01 * |anchor|)` + hard SM overflow guard (`w_t = 0` on `x_inf > 25 || mahal_sq > 1e5`).
  - *Real-Data Bug 2:* High-D collinear metric twins (`latency-50` vs `latency-90`) cancelled each other out under unshrunk `Theta` (`gamma=0.01`). Fixed with dimension-aware ridge shrinkage (`0.05 + 0.65*D/W_u`) + temporal innovation blending + noise-floored LiNGAM net outflow.
  - *Comparator Symmetry Fix:* Aligned `causalsieve` warmup to `case.inject_index` (matching all baselines) while preserving `causalsieve_blind` (`warmup=0, lookback=0`).
- **Iteration 3 (One-Shot Zero-Leakage Sealed Holdout + Full Freeze):** Froze binary SHA-256, passed **40/40 pytest** + **0 ASan/UBSan findings**, ran all 5 benchmarks + one-shot 40-case sealed `RCAEval` holdout (`#1` across all metrics: `67.5% AC@1`, `100% AC@5`, `0.804 MRR`).
- **Iteration 4 (Independent Adversarial Breaker + Final Hardening):** Hostile Breaker (`tests/breaker_audit.py`) caught 2 real defects on first run: (1) heap OOB `SIGSEGV` in `csrc/cli_main.cpp` when `--top-k 0 --compact` was passed on `D > 128` CSVs, and (2) 1 internal workspace name leak in `benchmarks/bench_sift_scale.py:3`. Patched both without altering frozen shared libraries; re-verified **69/69 PASS** (`40` core + `29` Breaker), **0 ASan/UBSan findings**, `ruff` 0, `mypy --strict` 0, and identical frozen binary SHA-256 across all 5 JSON artifacts.

## 4. Next Best Actions (Post-Release Growth Loop)

1. Publish standalone public repo + PyPI wheel (manylinux x86-64 AVX2 + macOS ARM64 NEON) + crates.io Rust wrapper over `csrc/causalsieve.h` + MCP Registry entry.
2. Add ARM64 NEON SIMD specialization alongside AVX2+FMA for Apple Silicon M4/M5 laptops.
3. Expand real-world benchmark coverage to `RCAEval` RE2 (multi-source metrics+traces) and `Train Ticket` (`D > 200`, exercising Tier-2 `ScreenedSieve` on live production traces).
4. Run Ramp-style AI-agent visibility queries across Claude, ChatGPT, Perplexity, and Gemini to measure organic tool recommendation pickup from `README.md`, `llms.txt`, and `SKILL.md`.
