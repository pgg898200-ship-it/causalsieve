"""High-dimensional Tier-2 ScreenedSieve scale & loud-decoy firehose benchmark (Hypothesis H4).

Directly tests the high-dimensional sparse-signal loud-decoy firehose regime across D in {128, 512, 2048, 4096}:
  - 20% of channels are heteroscedastic 3.5x-noise loud decoys with occasional benign spikes.
  - A sparse 3-hop amplified causal cascade (root -> mid -> leaf) is buried inside D channels.
  - Measures streaming throughput (ticks/sec and channel-observations/sec) AND Top-1 / Top-3 / MRR
    root recall of Tier-2 ScreenedSieve vs naive raw variance, Raw Max-Z, and Rolling-MAD Z.

Saves frozen results to benchmarks/results/sift_scale.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from _cabi import RawEngine, RawScreenedEngine  # noqa: E402
from baselines import (  # noqa: E402
    collect_system_metadata,
    eval_ranking,
    rank_raw_max_z,
    rank_rolling_mad_z,
    summarize_ranking_results,
)

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "sift_scale.json"


def _gen_firehose(
    D: int, T: int, shock_start: int, seed: int
) -> tuple[np.ndarray, int, list[int]]:
    rng = np.random.default_rng(700_000 + D * 31 + seed * 101)
    perm = rng.permutation(D)
    root, mid, leaf = int(perm[0]), int(perm[1]), int(perm[2])
    n_decoys = max(1, int(0.20 * D))
    decoys = perm[3 : 3 + n_decoys]

    noise_std = np.ones(D, dtype=np.float64)
    noise_std[decoys] = 3.5

    # Fast vectorized AR(1) background generation
    E = rng.standard_normal((T, D)) * noise_std[None, :]
    # Benign variance bursts on a few decoys during baseline & incident
    burst_decoys = decoys[: min(10, len(decoys))]
    for d_idx in burst_decoys:
        bt = rng.integers(50, T, size=4)
        E[bt, d_idx] *= 2.2

    X = np.empty((T, D), dtype=np.float64)
    X[0] = E[0]
    for t in range(1, T):
        X[t] = 0.25 * X[t - 1] + E[t]

    # Overwrite the 3 planted cascade channels with amplified causal chain root -> mid -> leaf
    e_r = rng.standard_normal(T)
    e_m = rng.standard_normal(T) * 0.35
    e_l = rng.standard_normal(T) * 0.30
    X[:, root] = 0.0
    X[:, mid] = 0.0
    X[:, leaf] = 0.0
    for t in range(1, T):
        u = 5.5 if t >= shock_start else 0.0
        X[t, root] = 0.25 * X[t - 1, root] + e_r[t] + u
        X[t, mid] = 0.65 * X[t - 1, mid] + 0.70 * X[t - 1, root] + e_m[t]
        X[t, leaf] = 0.70 * X[t - 1, leaf] + 0.72 * X[t - 1, mid] + e_l[t]

    return np.ascontiguousarray(X), root, [root, mid, leaf]


def _rank_naive_variance(X: np.ndarray, shock_start: int) -> np.ndarray:
    scores = np.var(X[shock_start:], axis=0)
    return np.argsort(-scores, kind="mergesort").astype(np.int32)


def run_benchmark(n_seeds: int = 10) -> dict[str, Any]:
    dims = [128, 512, 2048, 4096]
    T = 500
    shock_start = 400
    active_m = 24
    meta = collect_system_metadata()

    with RawEngine(4, 1) as probe:
        simd_isa = probe.simd_isa

    report: dict[str, Any] = {
        "benchmark": "tier2_sift_scale_firehose",
        "config": {
            "dims": dims,
            "T": T,
            "shock_start": shock_start,
            "decoy_fraction": 0.20,
            "decoy_noise_multiplier": 3.5,
            "active_m": active_m,
            "n_seeds": n_seeds,
            "simd_isa": simd_isa,
        },
        "system": meta,
        "results": {},
    }

    print(f"=== Tier-2 ScreenedSieve High-Dim Firehose Benchmark (ISA={simd_isa}, T={T}, seeds={n_seeds}) ===")
    for D in dims:
        methods = ["causalsieve_screened", "naive_variance", "raw_max_z", "rolling_mad_z"]
        rows: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
        times_ms: dict[str, list[float]] = {m: [] for m in methods}

        for seed in range(n_seeds):
            X, root, _ = _gen_firehose(D, T, shock_start, seed)

            # 1. Tier-2 ScreenedSieve
            t0 = time.perf_counter()
            with RawScreenedEngine(dim=D, active_m=active_m, lag=2, forgetting=0.995, ridge=1e-2) as sc:
                _, top_ch, _ = sc.push_batch(X, top_k=min(32, D))
            ms_cs = (time.perf_counter() - t0) * 1000.0
            rows["causalsieve_screened"].append(eval_ranking(top_ch.tolist(), root))
            times_ms["causalsieve_screened"].append(ms_cs)

            # 2. Naive variance
            t0 = time.perf_counter()
            r_var = _rank_naive_variance(X, shock_start)
            ms_var = (time.perf_counter() - t0) * 1000.0
            rows["naive_variance"].append(eval_ranking(r_var.tolist(), root))
            times_ms["naive_variance"].append(ms_var)

            # 3. Raw Max-Z
            t0 = time.perf_counter()
            _, r_z = rank_raw_max_z(X, shock_start)
            ms_z = (time.perf_counter() - t0) * 1000.0
            rows["raw_max_z"].append(eval_ranking(r_z.tolist(), root))
            times_ms["raw_max_z"].append(ms_z)

            # 4. Rolling-MAD Z
            t0 = time.perf_counter()
            _, r_mad = rank_rolling_mad_z(X, shock_start)
            ms_mad = (time.perf_counter() - t0) * 1000.0
            rows["rolling_mad_z"].append(eval_ranking(r_mad.tolist(), root))
            times_ms["rolling_mad_z"].append(ms_mad)

        med_cs_ms = float(np.median(times_ms["causalsieve_screened"]))
        ticks_per_sec = float(T) / (med_cs_ms / 1000.0)
        ch_obs_per_sec = float(D) * ticks_per_sec

        dim_entry = {
            "D": D,
            "causalsieve_throughput": {
                "median_ms": round(med_cs_ms, 3),
                "ticks_per_sec": round(ticks_per_sec, 1),
                "channel_obs_per_sec": round(ch_obs_per_sec, 1),
                "m_obs_per_sec": round(ch_obs_per_sec / 1e6, 3),
            },
            "methods": {
                m: {
                    **summarize_ranking_results(rows[m]),
                    "median_ms": round(float(np.median(times_ms[m])), 3),
                }
                for m in methods
            },
        }
        report["results"][str(D)] = dim_entry

        cs_s = dim_entry["methods"]["causalsieve_screened"]
        z_s = dim_entry["methods"]["raw_max_z"]
        mad_s = dim_entry["methods"]["rolling_mad_z"]
        var_s = dim_entry["methods"]["naive_variance"]
        print(
            f"  D={D:4d} | CausalSieve: {ticks_per_sec:8,.0f} ticks/s ({ch_obs_per_sec/1e6:6.2f}M obs/s) "
            f"| AC@1={cs_s['ac1']:.2f} AC@3={cs_s['ac3']:.2f} MRR={cs_s['mrr']:.3f} "
            f"| RawZ AC@1={z_s['ac1']:.2f} | MAD-Z AC@1={mad_s['ac1']:.2f} | Var AC@1={var_s['ac1']:.2f}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved high-dim sift-scale results to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark()
