"""Head-to-head wall-clock speed benchmark: CausalSieve vs real open-source incumbents.

Compares CausalSieve (AVX2+FMA C++20 engine) against:
  - statsmodels (VAR.fit + conditional test_causality across D*(D-1) pairs)
  - tigramite (PCMCI with ParCorr)
  - scikit-learn (LedoitWolf precision + IsolationForest; plus GraphicalLassoCV where feasible)
  - river (online streaming HalfSpaceTrees + per-channel online LinearRegression)
  - pure NumPy/SciPy (exact batch ridge-VAR + literal reduced-model Cholesky refits + precision)

Measured across D in {8, 16, 32, 64}, p = 2, T = 500 ticks on this exact host.
Outputs frozen reproducible JSON to benchmarks/results/speed.json.
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

from _cabi import RawEngine  # noqa: E402
from baselines import collect_system_metadata, numpy_batch_ridge_var_granger  # noqa: E402

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "speed.json"


def _gen_var_stream(D: int, T: int, lag: int = 2, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed + D * 17)
    X = np.zeros((T + 50, D), dtype=np.float64)
    eps = rng.standard_normal((T + 50, D))
    for t in range(lag, T + 50):
        X[t] = 0.30 * X[t - 1] - 0.10 * X[t - 2] + eps[t]
        for k in range(0, D - 1, 2):
            X[t, k + 1] += 0.35 * X[t - 1, k]
    return np.ascontiguousarray(X[50:])


def _time_fn(fn: Callable[[], Any], repeats: int = 3, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    arr = np.array(samples_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "min_ms": float(np.min(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "samples_ms": [round(float(x), 4) for x in arr],
    }


def _bench_causalsieve_batch(X: np.ndarray, lag: int) -> Callable[[], None]:
    T, D = X.shape

    def run() -> None:
        with RawEngine(D, lag, forgetting=0.995, normalize_scale=1) as eng:
            rc, _, _, _ = eng.push_batch(X, return_results=True)
            assert rc == 0
            eng.granger_matrix()
            eng.precision_matrix()
            eng.root_attribution(64)

    return run


def _bench_causalsieve_streaming_ticks(X: np.ndarray, lag: int) -> Callable[[], None]:
    T, D = X.shape

    def run() -> None:
        with RawEngine(D, lag, forgetting=0.995, normalize_scale=1) as eng:
            for t in range(T):
                rc, _, _, _ = eng.push(X[t], return_vectors=False)
                assert rc == 0
            eng.granger_matrix()
            eng.precision_matrix()

    return run


def _bench_causalsieve_steady_state(D: int, lag: int, T_steady: int = 2000) -> dict[str, float]:
    X = _gen_var_stream(D, T_steady, lag=lag, seed=999)
    with RawEngine(D, lag, forgetting=0.995, normalize_scale=1) as eng:
        # Warmup 100 ticks
        eng.push_batch(X[:100], return_results=False)
        X_run = np.ascontiguousarray(X[100:])
        n_ticks = X_run.shape[0]
        timings = _time_fn(
            lambda: eng.push_batch(X_run, return_results=True),
            repeats=5,
            warmup=1,
        )
        us_per_tick = (timings["median_ms"] * 1000.0) / float(n_ticks)
        ticks_per_sec = float(n_ticks) / (timings["median_ms"] / 1000.0)

        # Also time standalone D x D Block-Schur Granger + Precision query
        q_timings = _time_fn(
            lambda: (eng.granger_matrix(), eng.precision_matrix()),
            repeats=7,
            warmup=2,
        )
        return {
            "n_ticks": n_ticks,
            "batch_median_ms": round(timings["median_ms"], 4),
            "us_per_tick": round(us_per_tick, 4),
            "ticks_per_sec": round(ticks_per_sec, 1),
            "graph_query_ms": round(q_timings["median_ms"], 4),
        }


def _bench_statsmodels_var_granger(X: np.ndarray, lag: int) -> dict[str, Any]:
    from statsmodels.tsa.api import VAR

    T, D = X.shape
    total_pairs = D * (D - 1)
    # Time full VAR.fit + all D*(D-1) conditional Granger tests (or timed exact subset if huge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        res = VAR(X).fit(maxlags=lag)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        # Run conditional Granger F-tests; cap wall clock at 12s per D and record exact pair count
        pairs_done = 0
        t_g0 = time.perf_counter()
        extrapolated = False
        for dst in range(D):
            for src in range(D):
                if src == dst:
                    continue
                res.test_causality(dst, [src], kind="f")
                pairs_done += 1
                if pairs_done >= 200 and (time.perf_counter() - t_g0) > 10.0:
                    extrapolated = True
                    break
            if extrapolated:
                break
        g_elapsed_ms = (time.perf_counter() - t_g0) * 1000.0

    if extrapolated:
        full_granger_ms = g_elapsed_ms * (total_pairs / float(pairs_done))
    else:
        full_granger_ms = g_elapsed_ms
    total_ms = fit_ms + full_granger_ms
    return {
        "median_ms": round(total_ms, 3),
        "var_fit_ms": round(fit_ms, 3),
        "granger_ms": round(full_granger_ms, 3),
        "pairs_measured": pairs_done,
        "total_pairs": total_pairs,
        "extrapolated": extrapolated,
    }


def _bench_tigramite_pcmci(X: np.ndarray, lag: int) -> dict[str, Any]:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    D = X.shape[1]
    if D > 32:
        # PCMCI at D=64 takes several minutes; skip with honest note rather than fabricating
        return {
            "median_ms": None,
            "skipped": True,
            "reason": "Skipped at D=64 (O(D^3) conditional independence search exceeds 60s budget)",
        }

    def run() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pp.DataFrame(X)
            pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(verbosity=0), verbosity=0)
            pcmci.run_pcmci(tau_max=lag, pc_alpha=0.2)

    repeats = 3 if D <= 16 else 1
    warmup = 1 if D <= 16 else 0
    res = _time_fn(run, repeats=repeats, warmup=warmup)
    res["skipped"] = False
    return res


def _bench_sklearn(X: np.ndarray) -> dict[str, Any]:
    from sklearn.covariance import GraphicalLassoCV, LedoitWolf
    from sklearn.ensemble import IsolationForest

    D = X.shape[1]

    def run_lw_iso() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LedoitWolf().fit(X)
            IsolationForest(n_estimators=100, random_state=0, n_jobs=1).fit(X)

    lw_iso = _time_fn(run_lw_iso, repeats=3, warmup=1)

    glasso_ms: float | None = None
    if D <= 32:
        def run_glasso() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                GraphicalLassoCV(cv=3, max_iter=100).fit(X)

        glasso_ms = round(_time_fn(run_glasso, repeats=2 if D <= 16 else 1, warmup=0)["median_ms"], 3)

    return {
        "median_ms": round(lw_iso["median_ms"], 3),
        "ledoit_wolf_plus_isoforest_ms": round(lw_iso["median_ms"], 3),
        "graphical_lasso_cv_ms": glasso_ms,
        "note": "No Granger causality; precision + anomaly detection only",
    }


def _bench_river_online(X: np.ndarray, lag: int) -> dict[str, Any]:
    from river import anomaly, linear_model

    T, D = X.shape

    def run() -> None:
        hst = anomaly.HalfSpaceTrees(n_trees=10, height=5, window_size=100, seed=0)
        models = [linear_model.LinearRegression() for _ in range(D)]
        for t in range(lag, T):
            cur = {f"x{j}": float(X[t, j]) for j in range(D)}
            hst.score_one(cur)
            hst.learn_one(cur)
            feats = {}
            for ell in range(1, lag + 1):
                for j in range(D):
                    feats[f"l{ell}_{j}"] = float(X[t - ell, j])
            for j in range(D):
                models[j].learn_one(feats, float(X[t, j]))

    repeats = 2 if D <= 16 else 1
    warmup = 1 if D <= 16 else 0
    res = _time_fn(run, repeats=repeats, warmup=warmup)
    res["note"] = "Online HST + per-channel SGD linear regression (no conditional Granger or precision)"
    return res


def run_benchmark() -> dict[str, Any]:
    dims = [8, 16, 32, 64]
    lag = 2
    T = 500
    meta = collect_system_metadata()

    # Record SIMD ISA directly from loaded libcausalsieve.so
    with RawEngine(4, 1) as probe:
        simd_isa = probe.simd_isa

    report: dict[str, Any] = {
        "benchmark": "head_to_head_speed",
        "config": {"dims": dims, "lag": lag, "T_window": T, "simd_isa": simd_isa},
        "system": meta,
        "results": {},
    }

    print(f"=== CausalSieve Head-to-Head Speed Benchmark (ISA={simd_isa}, T={T}, p={lag}) ===")
    for D in dims:
        print(f"\n--- D = {D} ---")
        X = _gen_var_stream(D, T, lag=lag)

        cs_batch = _time_fn(_bench_causalsieve_batch(X, lag), repeats=5, warmup=1)
        cs_single = _time_fn(_bench_causalsieve_streaming_ticks(X, lag), repeats=5, warmup=1)
        cs_steady = _bench_causalsieve_steady_state(D, lag, T_steady=2000)

        np_batch = _time_fn(
            lambda: numpy_batch_ridge_var_granger(X, lag=lag, forgetting=0.995),
            repeats=3 if D <= 32 else 1,
            warmup=1 if D <= 32 else 0,
        )
        sm_res = _bench_statsmodels_var_granger(X, lag)
        tig_res = _bench_tigramite_pcmci(X, lag)
        sk_res = _bench_sklearn(X)
        riv_res = _bench_river_online(X, lag)

        base_ms = cs_batch["median_ms"]
        speedups: dict[str, float | None] = {
            "vs_numpy_batch_refits": round(np_batch["median_ms"] / base_ms, 2),
            "vs_statsmodels_var_granger": round(sm_res["median_ms"] / base_ms, 2),
            "vs_tigramite_pcmci": (
                round(tig_res["median_ms"] / base_ms, 2) if tig_res.get("median_ms") else None
            ),
            "vs_sklearn_lw_iso": round(sk_res["median_ms"] / base_ms, 2),
            "vs_sklearn_glasso_cv": (
                round(sk_res["graphical_lasso_cv_ms"] / base_ms, 2)
                if sk_res.get("graphical_lasso_cv_ms")
                else None
            ),
            "vs_river_online": round(riv_res["median_ms"] / base_ms, 2),
        }

        entry = {
            "D": D,
            "causalsieve_batch_full_ms": round(cs_batch["median_ms"], 4),
            "causalsieve_single_tick_full_ms": round(cs_single["median_ms"], 4),
            "causalsieve_steady_state": cs_steady,
            "numpy_batch_refits_ms": round(np_batch["median_ms"], 3),
            "statsmodels_var_granger": sm_res,
            "tigramite_pcmci": tig_res,
            "sklearn": sk_res,
            "river_online": riv_res,
            "speedups_over_incumbents": speedups,
        }
        report["results"][str(D)] = entry

        print(
            f"  CausalSieve batch (500 ticks + D*D Granger + Precision): {base_ms:.3f} ms "
            f"| steady-state: {cs_steady['ticks_per_sec']:,.0f} ticks/s ({cs_steady['us_per_tick']:.2f} us/tick)"
        )
        print(
            f"  NumPy batch refits: {np_batch['median_ms']:.2f} ms ({speedups['vs_numpy_batch_refits']}x) "
            f"| statsmodels: {sm_res['median_ms']:.2f} ms ({speedups['vs_statsmodels_var_granger']}x)"
        )
        print(
            f"  tigramite PCMCI: {tig_res.get('median_ms')} ms ({speedups['vs_tigramite_pcmci']}x) "
            f"| sklearn: {sk_res['median_ms']:.2f} ms ({speedups['vs_sklearn_lw_iso']}x) "
            f"| river: {riv_res['median_ms']:.2f} ms ({speedups['vs_river_online']}x)"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved speed benchmark results to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark()
