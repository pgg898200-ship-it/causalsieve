# CausalSieve Benchmarks & Optimization Log (`v0.2.0`)

Every number below was measured live on the exact compiled `v0.2.0` runtime-dispatched binaries anchored by SHA-256 across all **8 frozen benchmark JSONs** in [`benchmarks/results/`](benchmarks/results/). Zero numbers are estimated except explicitly marked linear extrapolations (`extrapolated: true` in JSON).

## Provenance & Hardware

- **CPU:** Intel Xeon @ 2.30 GHz (2 logical cores, AVX2 + FMA, 32 KiB L1d, 256 KiB L2, Ubuntu 24.04 Linux)
- **Compiler:** `g++ 13.3.0` (`-std=c++20 -Wall -Wextra -Werror`, multi-TU runtime CPUID dispatch: portable core + scalar TU + `-mavx2 -mfma` AVX2 TU)
- **Active Runtime SIMD ISA:** `avx2+fma` (verified 0 VEX/AVX/FMA instructions in core/scalar TUs, 1,278 AVX2+FMA instructions in AVX2 TU; `CAUSALSIEVE_FORCE_SCALAR=1` parity `< 6e-14`)
- **Frozen `build/libcausalsieve.so` SHA-256:** `e97025decbdf06663c73a63fa9adee73d9cae52e56661ba77098031f8d51bc36`
- **Frozen `build/libcausalsieve_scalar.so` SHA-256:** `7a8b2d1d9d4f673dc315e2b91b2e8c97190aafd0cfcf0e0b22e70caf22a99670`
- **Incumbent Versions:** `Python 3.12.16`, `numpy 2.5.5`, `scipy 1.18.1`, `statsmodels 0.15.0`, `tigramite 5.2.10.1`, `scikit-learn 1.9.1`, `river 0.27.0`, `pandas 3.1.1`

---

## 1. Head-to-Head Wall-Clock Speed (`benchmarks/results/speed.json`)

Workload: `T = 500` multivariate ticks, VAR lag `p = 2`, full ingest + full `D×D` conditional Granger + `D×D` precision matrix.

| `D` | CausalSieve `v0.2.0` (ms) | Steady-State Rate | NumPy Batch Refits | `statsmodels` VAR+Granger | `tigramite` PCMCI | `sklearn` LW+IsoF | `river` Online |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | **3.10** | **746,820 ticks/s** (`1.34 µs`) | 2.43 ms (`0.78×`*) | 99.50 ms (**32.1×**) | 694.33 ms (**223.8×**) | 202.49 ms (**65.3×**) | 63.67 ms (**20.5×**) |
| **16** | **5.66** | **385,256 ticks/s** (`2.60 µs`) | 5.73 ms (**1.01×**) | 666.05 ms (**117.7×**) | 3,307.83 ms (**584.7×**) | 205.69 ms (**36.4×**) | 145.91 ms (**25.8×**) |
| **32** | **11.22** | **154,191 ticks/s** (`6.49 µs`) | 319.80 ms (**28.5×**) | 26,187.04 ms† (**2,333.9×**) | 16,461.03 ms (**1,467.1×**) | 220.97 ms (**19.7×**) | 463.41 ms (**41.3×**) |
| **64** | **30.57** | **48,950 ticks/s** (`20.43 µs`) | 641.05 ms (**21.0×**) | 2,185,817 ms† (**71,514×**) | >60 s (skipped) | 208.61 ms (**6.8×**) | 1,579.35 ms (**51.7×**) |

- \*At `D=8`, a single offline NumPy batch refit without streaming state takes `2.43 ms` vs CausalSieve's full streaming + `ctypes` pass (`3.10 ms`). By `D=16`, CausalSieve `v0.2.0` already beats offline NumPy (`5.66 ms` vs `5.73 ms`), reaching **28.5×** at `D=32` and **71,514×** vs `statsmodels` at `D=64`.
- †Extrapolated linearly from measured pairs (`413/992` at `D=32`, `200/4,032` at `D=64`, flagged `extrapolated: true` in `speed.json`).
- **v0.1.0 → v0.2.0 Core Speedup (C-ABI `p=2, T=2000`):** `D=8`: `595k → 761k` (`1.28×`), `D=16`: `292k → 397k` (`1.36×`), `D=32`: `101k → 159k` (`1.57×`), `D=64`: `26.6k → 51.4k` (**`1.93×`** vs v0.1 AVX2, **`3.34×–3.41×`** vs scalar; portable PyPI x86-64 wheels jump **~3.4×–5×** thanks to runtime CPUID AVX2+FMA dispatch).

---

## 2. Compounding Optimization Ladder (`benchmarks/results/optimization_log.json`)

Workload: `T = 1,000` ticks, `p = 2` + full `D×D` conditional Granger + precision matrix, with exact parity verified against the batch FP64 oracle (`< 1.86e-12`) and scalar vs AVX2+FMA (`max |delta F| = 6.13e-14` at `D=16`, `9.59e-14` at `D=32`).

| Stage | Description | `D=16` ms (Cum. Speedup) | `D=32` ms (Cum. Speedup) |
|---:|---|---:|---:|
| 1 | `statsmodels` VAR + conditional Granger baseline | 676.79 ms (`1.0×`) | 24,342.34 ms (`1.0×`) |
| 2 | Batch NumPy/SciPy ridge-VAR + exact reduced refits | 8.52 ms (`79.5×`) | 282.95 ms (`86.0×`) |
| 3 | CausalSieve C++20 scalar (`libcausalsieve_scalar.so`, fused `push_batch`) | 13.45 ms (`50.3×`) | 24.87 ms (`978.8×`) |
| 4 | CausalSieve C++20 `avx2+fma` single-tick Python `ctypes` loop | 21.49 ms (`31.5×`*) | 26.16 ms (`930.7×`) |
| 5 | CausalSieve C++20 `avx2+fma` v0.2.0 BLAS-3 slab + 4-row unrolled (`push_batch`) | **7.83 ms (`86.4×`)** | **14.67 ms (`1,659.5×`)** |

---

## 3. ALL 645 Real Kubernetes Outages (`RCAEval` RE1 + RE2 OpenTelemetry, `benchmarks/results/rcaeval_suite.json`)

Source: `https://huggingface.co/datasets/phamquiluan/RCAEval` — **100% of RE1 (375 cases) + 100% of RE2 OpenTelemetry (270 cases) = 645 real production microservice outages**, zero hyperparameter tuning (exact frozen defaults `lag=2, forgetting=0.995`), every parquet & `inject_time.txt` SHA-256 recorded. Mean digest compression across all 645 cases: **`297,241 -> 70.4` tokens (`99.976%` reduction)**.

### Overall Micro (n=645) & Macro (6 Datasets Equal Weight)

| Method | Micro AC@1 | Micro AC@3 | Micro AC@5 | Micro Avg@5 | Micro MRR | Macro AC@1 | Macro MRR | Mean ms/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`causalsieve`** (symmetric) | **0.757** | **0.910** | 0.952 | **0.882** | **0.839** | **0.765** | **0.844** | 160.9 ms |
| `baro_fse24` (Pham et al., FSE'24) | 0.727 | 0.901 | 0.958 | 0.874 | 0.824 | 0.736 | 0.829 | 7.4 ms |
| `nsigma_rcaeval` (RCAEval N-Sigma) | 0.726 | 0.905 | **0.961** | 0.876 | 0.823 | 0.737 | 0.830 | 3.6 ms |
| `circa_structural_rca` (KDD'22-style) | 0.716 | 0.862 | 0.916 | 0.847 | 0.808 | 0.727 | 0.816 | 44.6 ms |
| `pairwise_granger_z` | 0.535 | 0.784 | 0.864 | 0.745 | 0.681 | 0.549 | 0.689 | 210.3 ms |
| `causalsieve_blind` (zero `inject_time`) | 0.522 | 0.828 | 0.887 | 0.774 | 0.688 | 0.508 | 0.682 | 163.3 ms |
| `rolling_mad_z` | 0.457 | 0.781 | 0.879 | 0.729 | 0.642 | 0.459 | 0.642 | 11.8 ms |
| `causalsieve_tier2` (`ScreenedSieve M=32`, blind) | 0.276 | 0.459 | 0.591 | 0.445 | 0.422 | 0.243 | 0.395 | **4.0 ms** |

### Per-Dataset AC@1 / MRR Breakdown (All 6 Systems)

| Dataset | Cases | Metrics (`D`) | `causalsieve` AC@1 / MRR (ms) | `baro_fse24` AC@1 / MRR | `nsigma` AC@1 / MRR | `circa` AC@1 / MRR | `pairwise_granger` AC@1 / MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`RE1-OB`** | 125 | 49–59 | 0.736 / 0.834 (`29.9 ms`) | 0.704 / 0.804 | 0.752 / **0.840** | **0.768** / 0.835 | 0.576 / 0.722 |
| **`RE1-SS`** | 125 | 57–63 | 0.832 / 0.902 (`23.2 ms`) | **0.840 / 0.911** | 0.792 / 0.875 | 0.816 / 0.881 | 0.808 / 0.882 |
| **`RE1-TT`** (64 svcs) | 125 | 198–238 | **0.576 / 0.711** (`216.1 ms`) | 0.504 / 0.673 | 0.456 / 0.651 | 0.400 / 0.589 | 0.272 / 0.460 |
| **`RE2-OB`** (OTel) | 90 | 69–77 | **0.889 / 0.933** (`24.7 ms`) | 0.856 / 0.912 | 0.867 / 0.921 | 0.844 / 0.905 | 0.611 / 0.754 |
| **`RE2-SS`** (OTel) | 90 | 74–82 | 0.822 / 0.892 (`26.4 ms`) | 0.833 / 0.902 | **0.844 / 0.905** | 0.833 / 0.897 | 0.656 / 0.779 |
| **`RE2-TT`** (68 svcs, OTel) | 90 | 340–376 | **0.733 / 0.791** (`728.5 ms`) | 0.678 / 0.772 | 0.711 / 0.784 | 0.700 / 0.787 | 0.367 / 0.538 |
| **RE1 Total** | **375** | 49–238 | **0.715 / 0.816** (`89.7 ms`) | 0.683 / 0.796 | 0.667 / 0.789 | 0.661 / 0.768 | 0.552 / 0.688 |
| **RE2 Total** | **270** | 69–376 | **0.815 / 0.872** (`259.9 ms`) | 0.789 / 0.862 | 0.807 / 0.870 | 0.793 / 0.863 | 0.544 / 0.690 |

### Per-Fault Breakdown (All 645 Cases)

| Fault | n | `causalsieve` AC@1 / MRR | `baro_fse24` AC@1 / MRR | `nsigma` AC@1 / MRR | `circa` AC@1 / MRR |
|---|---:|---:|---:|---:|---:|
| **`mem`** | 120 | **0.958 / 0.978** | 0.900 / 0.942 | 0.900 / 0.943 | 0.917 / 0.950 |
| **`disk`** | 120 | **0.783 / 0.869** | 0.733 / 0.839 | 0.725 / 0.832 | 0.725 / 0.826 |
| **`socket`** | 45 | **0.778 / 0.844** | 0.711 / 0.813 | 0.733 / 0.824 | 0.733 / 0.817 |
| **`cpu`** | 120 | **0.758 / 0.852** | 0.733 / 0.831 | 0.708 / 0.818 | 0.692 / 0.806 |
| **`delay`** | 120 | 0.742 / 0.827 | 0.708 / 0.817 | **0.758 / 0.844** | 0.742 / 0.834 |
| **`loss`** | 120 | 0.533 / 0.670 | **0.567 / 0.686** | 0.533 / 0.671 | 0.500 / 0.644 |

---

## 4. Agent Context Compression & Multi-Agent Blackboard (`benchmarks/results/agent_tokens.json`)

### Part A: Context Overflow & Bit-Exact Recovery Across 60 Stratified Real Kubernetes Outages (10/dataset × 6 datasets)

| Context Format | Mean BPE Tokens | Overflow @ `8k` | Overflow @ `32k` | Overflow @ `128k` | Compression vs CSV | Fidelity |
|---|---:|---:|---:|---:|---:|---|
| Raw CSV | `899,003.8` | 100.0% | 100.0% | 100.0% | 1.0× | Lossless |
| Raw JSON Records | `2,647,465.0` | 100.0% | 100.0% | 100.0% | 0.34× | Lossless |
| 10× Downsampled CSV | `91,546.3` | 100.0% | 71.7% | 16.7% | 9.8× | **Lossy** (`57.1%` peak amplitude clipped) |
| **`CausalDigest` + `slice://`** | **`911.5`** | **`0.0%`** | **`0.0%`** | **`0.0%`** | **986.3×** (**2,905×** vs JSON) | **Bit-exact `0.0` FP64** (`360/360` slices, `173,642` values, median **`65.1 µs`**) |

- Swing-Filter PLA certification: **180/180** trajectories satisfied exact `L_inf` error guarantee (`0` violations).

### Part B: Multi-Agent `CausalBlackboard` Adversarial Consensus (50 Concurrent Episodes, 2 Grounded vs 2 Coordinated Ballot-Stuffing Agents)

| Coordination Method | Consensus AC@1 | Consensus AC@3 | MRR | Overwrite Churn |
|---|---:|---:|---:|---:|
| **`CausalBlackboard` (append-only + proper scoring)** | **1.000** (50/50) | **1.000** | **1.000** | **0.0%** (`0 / 900`) |
| Naive Majority Vote | 0.000 | 1.000 | 0.500 | N/A |
| Confidence-Weighted Vote | 0.000 | 1.000 | 0.500 | N/A |
| Mutable Last-Write-Wins Blackboard | 0.000 | 1.000 | 0.500 | **94.4%** (`850 / 900`) |

- Proper-scoring calibration: Grounded Brier loss **`0.0088`** vs Adversarial **`0.6851`** (gap `+0.6763`), yielding a **12.86×** grounded consensus weight advantage (`0.7897` vs `0.0614`) despite 2× adversarial ballot stuffing.

---

## 5. Original Sealed Holdout (`RE1` OB+SS, n=40, `rcaeval.json`) & Train Ticket (`RE1-TT`, n=25, `rcaeval_trainticket.json`)

- **RE1 OB+SS Sealed Holdout (`n = 40`, zero retuning):** `causalsieve` **67.5% AC@1 / 92.5% AC@3 / 100.0% AC@5 (40/40) / 0.880 Avg@5 / 0.804 MRR** in **`29.1 ms/case`** (down from `33.3 ms` in v0.1.0; `5.8×` faster than `pairwise_granger_z` `169.1 ms`, `197×` faster than `IsolationForest` `5,726 ms`). Blind (`causalsieve_blind`, zero `inject_time`): **65.0% AC@1 / 0.746 MRR** in **`25.9 ms/case`** (still #1 over every baseline given `inject_time`).
- **RE1 Train Ticket (`n = 25`, `D=198..238`, 64 services):** `causalsieve_tier1` **48.0% AC@1 / 84.0% AC@3 / 88.0% AC@5 / 0.669 MRR** in **`214.8 ms/case`** (`1.75×` faster than v0.1.0 `375.0 ms`; `103×` faster than `IsolationForest` `22,168 ms`; `10.7×` faster than uncapped pairwise Granger `2,307 ms`). Token reduction **`346,742 -> 63.8` (`99.982%`)**.

---

## 6. 100-Episode Synthetic RCA (`synthetic_rca.json`) & Tier-2 Firehose (`sift_scale.json`)

- **Synthetic 100 Episodes (5 regimes × 20 seeds):** `causalsieve` **0.730 AC@1 / 0.960 AC@3 / 0.830 MRR** in **`10.02 ms/ep`** (`259×` faster than `IsolationForest` `2,593 ms`, `0.380` AC@1; `1.000` AC@1 on Lagged Amplified, D=64 Decoys, and ERDDAP `1e10`). `causalsieve_hinge`: **`0.450` AC@1** on Nonlinear Saturation vs `0.000` for every other method.
- **Tier-2 `ScreenedSieve` Firehose (`D=128..4,096`, 20% 3.5×-noise decoys, 10 seeds each):**
  - `D=128`: **`172,466 ticks/s`** (`22.08M obs/s`), **100% AC@1** vs 0% all baselines
  - `D=512`: **`87,786 ticks/s`** (`44.95M obs/s`), **100% AC@1** vs 0% all baselines
  - `D=2,048`: **`36,924 ticks/s`** (`75.62M obs/s`), **100% AC@1** vs 0% all baselines
  - `D=4,096`: **`18,488 ticks/s`** (`75.73M obs/s`), **100% AC@1** vs 0% all baselines
