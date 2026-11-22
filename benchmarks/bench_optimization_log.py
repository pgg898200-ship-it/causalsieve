"""Compounding optimization ladder for CausalSieve (DeepGrove / Ankit methodology).

Measures the exact, step-by-step wall-clock progression on identical workloads (D=16 and D=32,
p=2, T=1,000 ticks + full D x D conditional Granger + precision matrix):
  Stage 1: statsmodels VAR(p) + conditional Granger F-tests (Python / SciPy baseline)
  Stage 2: Vectorized NumPy/SciPy batch ridge-VAR + reduced-model Cholesky refits
  Stage 3: CausalSieve C++20 scalar build (libcausalsieve_scalar.so, no AVX2/FMA)
  Stage 4: CausalSieve C++20 AVX2+FMA (libcausalsieve.so) single-tick Python loop
  Stage 5: CausalSieve C++20 AVX2+FMA (libcausalsieve.so) fused C-loop batch (push_batch)

Saves frozen compounding log to benchmarks/results/optimization_log.json.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from _cabi import DEFAULT_LIB_PATH, SCALAR_LIB_PATH, RawEngine  # noqa: E402
from baselines import collect_system_metadata, numpy_batch_ridge_var_granger  # noqa: E402

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "optimization_log.json"


def _gen_workload(D: int, T: int, lag: int = 2, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed + D)
    X = np.zeros((T + 50, D), dtype=np.float64)
    eps = rng.standard_normal((T + 50, D))
    for t in range(lag, T + 50):
        X[t] = 0.30 * X[t - 1] - 0.08 * X[t - 2] + eps[t]
        for k in range(0, D - 1):
            if k % 3 == 0:
                X[t, k + 1] += 0.35 * X[t - 1, k]
    return np.ascontiguousarray(X[50:])


def _bench(fn: Callable[[], Any], repeats: int = 5, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    vals: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(vals, dtype=np.float64)
    return {
        "median_ms": round(float(np.median(arr)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
    }


def _run_statsmodels(X: np.ndarray, lag: int) -> dict[str, Any]:
    from statsmodels.tsa.api import VAR

    D = X.shape[1]
    total_pairs = D * (D - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        res = VAR(X).fit(maxlags=lag)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        done = 0
        extrapolated = False
        for dst in range(D):
            for src in range(D):
                if src == dst:
                    continue
                res.test_causality(dst, [src], kind="f")
                done += 1
                if done >= 240 and (time.perf_counter() - t1) > 8.0:
                    extrapolated = True
                    break
            if extrapolated:
                break
        g_ms = (time.perf_counter() - t1) * 1000.0
    if extrapolated:
        g_ms = g_ms * (total_pairs / float(done))
    return {
        "median_ms": round(fit_ms + g_ms, 4),
        "var_fit_ms": round(fit_ms, 4),
        "granger_ms": round(g_ms, 4),
        "extrapolated": extrapolated,
    }


def _run_ladder_for_dim(D: int, T: int = 1000, lag: int = 2) -> dict[str, Any]:
    X = _gen_workload(D, T, lag=lag)
    stages: list[dict[str, Any]] = []

    # Stage 1: statsmodels baseline
    s1 = _run_statsmodels(X, lag)
    stages.append(
        {
            "stage": 1,
            "name": "statsmodels VAR + conditional Granger",
            "median_ms": s1["median_ms"],
            "ticks_per_sec": round(T / (s1["median_ms"] / 1000.0), 1),
            "details": s1,
        }
    )

    # Stage 2: Batch NumPy/SciPy ridge-VAR + literal reduced-model Cholesky refits
    s2 = _bench(
        lambda: numpy_batch_ridge_var_granger(X, lag=lag, forgetting=0.995),
        repeats=3 if D <= 16 else 2,
        warmup=1,
    )
    stages.append(
        {
            "stage": 2,
            "name": "Batch NumPy/SciPy ridge-VAR + reduced refits",
            "median_ms": s2["median_ms"],
            "ticks_per_sec": round(T / (s2["median_ms"] / 1000.0), 1),
            "details": s2,
        }
    )

    # Stage 3: CausalSieve C++20 scalar (no AVX2/FMA)
    if SCALAR_LIB_PATH.exists():
        def run_scalar_batch() -> None:
            with RawEngine(D, lag, forgetting=0.995, normalize_scale=1, lib_path=SCALAR_LIB_PATH) as eng:
                rc, _, _, _ = eng.push_batch(X, return_results=True)
                assert rc == 0
                eng.granger_matrix()
                eng.precision_matrix()

        with RawEngine(4, 1, lib_path=SCALAR_LIB_PATH) as p_sc:
            scalar_isa = p_sc.simd_isa
        s3 = _bench(run_scalar_batch, repeats=5, warmup=1)
        stages.append(
            {
                "stage": 3,
                "name": f"CausalSieve C++ scalar ({scalar_isa}, push_batch)",
                "median_ms": s3["median_ms"],
                "ticks_per_sec": round(T / (s3["median_ms"] / 1000.0), 1),
                "details": {**s3, "simd_isa": scalar_isa},
            }
        )
    else:
        stages.append(
            {
                "stage": 3,
                "name": "CausalSieve C++ scalar (missing libcausalsieve_scalar.so)",
                "median_ms": None,
                "ticks_per_sec": None,
                "details": {"missing_lib": str(SCALAR_LIB_PATH)},
            }
        )

    # Stage 4: CausalSieve C++20 AVX2+FMA single-tick Python call loop
    with RawEngine(4, 1, lib_path=DEFAULT_LIB_PATH) as p_avx:
        avx_isa = p_avx.simd_isa

    def run_avx_single() -> None:
        with RawEngine(D, lag, forgetting=0.995, normalize_scale=1, lib_path=DEFAULT_LIB_PATH) as eng:
            for t in range(T):
                rc, _, _, _ = eng.push(X[t], return_vectors=False)
                assert rc == 0
            eng.granger_matrix()
            eng.precision_matrix()

    s4 = _bench(run_avx_single, repeats=5, warmup=1)
    stages.append(
        {
            "stage": 4,
            "name": f"CausalSieve C++ {avx_isa} single-tick Python loop",
            "median_ms": s4["median_ms"],
            "ticks_per_sec": round(T / (s4["median_ms"] / 1000.0), 1),
            "details": {**s4, "simd_isa": avx_isa},
        }
    )

    # Stage 5: CausalSieve C++20 AVX2+FMA fused C-loop push_batch
    def run_avx_batch() -> None:
        with RawEngine(D, lag, forgetting=0.995, normalize_scale=1, lib_path=DEFAULT_LIB_PATH) as eng:
            rc, _, _, _ = eng.push_batch(X, return_results=True)
            assert rc == 0
            eng.granger_matrix()
            eng.precision_matrix()

    s5 = _bench(run_avx_batch, repeats=7, warmup=2)
    stages.append(
        {
            "stage": 5,
            "name": f"CausalSieve C++ {avx_isa} fused C-loop (push_batch)",
            "median_ms": s5["median_ms"],
            "ticks_per_sec": round(T / (s5["median_ms"] / 1000.0), 1),
            "details": {**s5, "simd_isa": avx_isa},
        }
    )

    # Annotate cumulative and incremental speedup ratios
    baseline_ms = stages[0]["median_ms"]
    prev_ms = baseline_ms
    for st in stages:
        ms = st["median_ms"]
        if ms is not None and ms > 0:
            st["cumulative_speedup_vs_stage1"] = round(baseline_ms / ms, 2)
            st["step_speedup_vs_prev"] = round(prev_ms / ms, 2) if prev_ms else None
            prev_ms = ms
        else:
            st["cumulative_speedup_vs_stage1"] = None
            st["step_speedup_vs_prev"] = None

    # Verify scalar vs AVX2 numerical parity if scalar build exists
    scalar_vs_avx_max_diff: float | None = None
    if SCALAR_LIB_PATH.exists():
        with RawEngine(D, lag, forgetting=0.995, normalize_scale=0, lib_path=SCALAR_LIB_PATH) as e_sc, RawEngine(
            D, lag, forgetting=0.995, normalize_scale=0, lib_path=DEFAULT_LIB_PATH
        ) as e_avx:
            e_sc.push_batch(X, return_results=False)
            e_avx.push_batch(X, return_results=False)
            f_sc, _ = e_sc.granger_matrix()
            f_avx, _ = e_avx.granger_matrix()
            scalar_vs_avx_max_diff = float(np.max(np.abs(f_sc - f_avx)))

    return {
        "D": D,
        "T": T,
        "lag": lag,
        "scalar_vs_avx_max_f_diff": scalar_vs_avx_max_diff,
        "stages": stages,
    }


def run_benchmark() -> dict[str, Any]:
    meta = collect_system_metadata()
    report: dict[str, Any] = {
        "benchmark": "compounding_optimization_log",
        "system": meta,
        "workloads": {},
    }

    print("=== CausalSieve Compounding Optimization Log ===")
    for D in (16, 32):
        res = _run_ladder_for_dim(D=D, T=1000, lag=2)
        report["workloads"][f"D{D}_p2_T1000"] = res
        print(f"\n--- Workload: D={D}, p=2, T=1000 ---")
        for st in res["stages"]:
            print(
                f"  Stage {st['stage']}: {st['name']:52s} | "
                f"{str(st['median_ms']):>9s} ms | "
                f"cum={str(st['cumulative_speedup_vs_stage1']):>8s}x | "
                f"step={str(st['step_speedup_vs_prev']):>6s}x"
            )
        if res["scalar_vs_avx_max_f_diff"] is not None:
            print(f"  Scalar vs AVX2 max |delta F|: {res['scalar_vs_avx_max_f_diff']:.3e}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved optimization log to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark()
