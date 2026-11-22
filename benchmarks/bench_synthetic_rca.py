"""100-episode synthetic Root-Cause Analysis benchmark (5 challenging regimes x 20 seeds).

Regimes (20 independent randomized seeds each = 100 episodes total):
  1. lagged_amplified_cascade:        D=16, 4-node chain with queue-like downstream amplification.
  2. instantaneous_nongaussian:       D=12, contemporaneous non-Gaussian SEM (Exponential & Student-t4)
                                      where exogenous shock propagates synchronously at lag 0.
  3. nonlinear_saturation_cascade:    D=12, ReLU/hinge threshold saturation cascade.
  4. heteroscedastic_decoy_firehose:  D=64, 20 3.5x-noise loud decoy channels + hidden 3-hop cascade.
  5. erddap_1e10_dc_offset_cascade:   D=16, amplified cascade with +-1e10 DC offsets & 1e-5..1e5 scales.

Comparators (Amendment A10 symmetric protocol):
  - causalsieve (Huber shock-shielded, normalize_scale=1)
  - unshielded_online_rls (ablation: shock_threshold=0.0)
  - raw_max_z (classic 3-sigma Z-score)
  - rolling_mad_z (Baro-style robust median/MAD Z-score)
  - sklearn_isolation_forest (per-channel lagged IsolationForest)
  - pairwise_granger_z (bivariate pairwise Granger outflow + robust Z)

Reports AC@1, AC@3, MRR, mean latency (ms/episode), and token reduction.
Saves frozen results to benchmarks/results/synthetic_rca.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from _cabi import RawEngine  # noqa: E402
from baselines import (  # noqa: E402
    collect_system_metadata,
    estimate_raw_tokens,
    estimate_text_tokens,
    eval_ranking,
    rank_isolation_forest,
    rank_pairwise_granger_z,
    rank_raw_max_z,
    rank_rolling_mad_z,
    summarize_ranking_results,
)

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "synthetic_rca.json"


@dataclass
class Episode:
    regime: str
    seed: int
    X: np.ndarray
    true_root: int
    shock_start: int
    chain: list[int]


def _gen_episode(regime: str, seed: int) -> Episode:
    rng = np.random.default_rng(50_000 + hash((regime, seed)) % (2**31))
    # Use deterministic seed independent of PYTHONHASHSEED
    regime_idx = {
        "lagged_amplified_cascade": 1,
        "instantaneous_nongaussian": 2,
        "nonlinear_saturation_cascade": 3,
        "heteroscedastic_decoy_firehose": 4,
        "erddap_1e10_dc_offset_cascade": 5,
    }[regime]
    rng = np.random.default_rng(regime_idx * 100_000 + seed * 97 + 13)

    if regime in ("lagged_amplified_cascade", "erddap_1e10_dc_offset_cascade"):
        T, D, shock_start = 520, 16, 400
        perm = rng.permutation(D)
        r, m1, m2, leaf = [int(perm[i]) for i in range(4)]
        burn = 120
        X = np.zeros((T + burn, D), dtype=np.float64)
        eps = rng.standard_normal((T + burn, D))
        eps[:, m1] *= 0.35
        eps[:, m2] *= 0.35
        eps[:, leaf] *= 0.30
        for t in range(2, T + burn):
            X[t] = 0.28 * X[t - 1] - 0.08 * X[t - 2] + eps[t]
            u = 4.5 if (t - burn) >= shock_start else 0.0
            X[t, r] = 0.25 * X[t - 1, r] + eps[t, r] + u
            X[t, m1] = 0.65 * X[t - 1, m1] + 0.65 * X[t - 1, r] + eps[t, m1]
            X[t, m2] = 0.68 * X[t - 1, m2] + 0.65 * X[t - 1, m1] + eps[t, m2]
            X[t, leaf] = 0.72 * X[t - 1, leaf] + 0.70 * X[t - 1, m2] + eps[t, leaf]
        X = X[burn:]
        if regime == "erddap_1e10_dc_offset_cascade":
            # Add massive +-1e10 DC offsets and 10-order-of-magnitude scale disparity
            signs = rng.choice([-1.0, 1.0], size=D)
            offsets = signs * 1e10 * rng.uniform(0.5, 1.5, size=D)
            exponents = rng.uniform(-5.0, 5.0, size=D)
            scales = np.power(10.0, exponents)
            X = X * scales[None, :] + offsets[None, :]
        return Episode(regime, seed, np.ascontiguousarray(X), r, shock_start, [r, m1, m2, leaf])

    if regime == "instantaneous_nongaussian":
        T, D, shock_start = 600, 12, 520
        perm = rng.permutation(D)
        r, m1, leaf = [int(perm[i]) for i in range(3)]
        if seed % 2 == 0:
            E = rng.exponential(scale=1.0, size=(T, D)) - 1.0
        else:
            E = rng.standard_t(df=4.0, size=(T, D)) / np.sqrt(2.0)
        X = np.zeros((T, D), dtype=np.float64)
        for t in range(1, T):
            u = 6.5 if t >= shock_start else 0.0
            for j in range(D):
                if j not in (r, m1, leaf):
                    X[t, j] = 0.25 * X[t - 1, j] + E[t, j]
            X[t, r] = 0.20 * X[t - 1, r] + E[t, r] + u
            # Instantaneous lag-0 propagation with downstream amplification
            X[t, m1] = 0.20 * X[t - 1, m1] + 1.15 * X[t, r] + 0.35 * E[t, m1]
            X[t, leaf] = 0.20 * X[t - 1, leaf] + 1.25 * X[t, m1] + 0.30 * E[t, leaf]
        return Episode(regime, seed, np.ascontiguousarray(X), r, shock_start, [r, m1, leaf])

    if regime == "nonlinear_saturation_cascade":
        T, D, shock_start = 650, 12, 530
        perm = rng.permutation(D)
        r, m1, leaf = [int(perm[i]) for i in range(3)]
        burn = 100
        X = np.zeros((T + burn, D), dtype=np.float64)
        eps = rng.standard_normal((T + burn, D))
        for t in range(1, T + burn):
            X[t] = 0.25 * X[t - 1] + eps[t]
            u = 3.8 if (t - burn) >= shock_start else 0.0
            X[t, r] = 0.20 * X[t - 1, r] + eps[t, r] + u
            X[t, m1] = (
                0.45 * X[t - 1, m1]
                + 2.0 * max(0.0, X[t - 1, r] - 1.0)
                + 0.35 * eps[t, m1]
            )
            X[t, leaf] = (
                0.55 * X[t - 1, leaf]
                + 1.8 * max(0.0, X[t - 1, m1] - 1.0)
                + 0.30 * eps[t, leaf]
            )
        return Episode(regime, seed, np.ascontiguousarray(X[burn:]), r, shock_start, [r, m1, leaf])

    if regime == "heteroscedastic_decoy_firehose":
        T, D, shock_start = 520, 64, 400
        perm = rng.permutation(D)
        r, m1, leaf = [int(perm[i]) for i in range(3)]
        decoys = perm[3:23]
        noise_std = np.ones(D, dtype=np.float64)
        noise_std[decoys] = 3.5
        burn = 100
        X = np.zeros((T + burn, D), dtype=np.float64)
        eps = rng.standard_normal((T + burn, D)) * noise_std[None, :]
        # Benign heavy-tail jitter on decoys throughout the episode
        for d_idx in decoys:
            spike_ts = rng.integers(0, T + burn, size=10)
            eps[spike_ts, d_idx] *= 2.0
        eps[:, m1] = rng.standard_normal(T + burn) * 0.35
        eps[:, leaf] = rng.standard_normal(T + burn) * 0.30
        for t in range(2, T + burn):
            X[t] = 0.25 * X[t - 1] + eps[t]
            u = 4.8 if (t - burn) >= shock_start else 0.0
            X[t, r] = 0.25 * X[t - 1, r] + rng.standard_normal() + u
            X[t, m1] = 0.65 * X[t - 1, m1] + 0.68 * X[t - 1, r] + eps[t, m1]
            X[t, leaf] = 0.70 * X[t - 1, leaf] + 0.70 * X[t - 1, m1] + eps[t, leaf]
        return Episode(regime, seed, np.ascontiguousarray(X[burn:]), r, shock_start, [r, m1, leaf])

    raise ValueError(f"Unknown regime {regime}")


def _run_causalsieve_episode(
    ep: Episode, *, shock_threshold: float | None = None, nonlinear_hinge: int = 0
) -> tuple[np.ndarray, float, int]:
    T, D = ep.X.shape
    lookback = T - ep.shock_start + 12
    t0 = time.perf_counter()
    with RawEngine(
        D,
        lag=2,
        forgetting=0.995,
        shock_threshold=shock_threshold,
        nonlinear_hinge=nonlinear_hinge,
        normalize_scale=1,
    ) as eng:
        rc, _, _, _ = eng.push_batch(ep.X, return_results=True)
        assert rc == 0
        rca = eng.root_attribution(lookback_ticks=lookback)
        f_mat, _ = eng.granger_matrix()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Estimate Caveman Causal Digest tokens (compact top-5 roots + top-8 edges)
    ranked = rca["ranked_channels"]
    top5 = [(int(c), round(float(rca["root_scores"][c]), 2)) for c in ranked[:5]]
    flat_edges = np.argwhere(f_mat > 6.0)
    top_edges = sorted(
        [(int(i), int(j), round(float(f_mat[i, j]), 1)) for i, j in flat_edges if i != j],
        key=lambda x: -x[2],
    )[:8]
    digest_str = json.dumps({"T": T, "D": D, "roots": top5, "edges": top_edges}, separators=(",", ":"))
    digest_tokens = estimate_text_tokens(digest_str)
    return ranked, elapsed_ms, digest_tokens


def run_benchmark(n_seeds: int = 20) -> dict[str, Any]:
    regimes = [
        "lagged_amplified_cascade",
        "instantaneous_nongaussian",
        "nonlinear_saturation_cascade",
        "heteroscedastic_decoy_firehose",
        "erddap_1e10_dc_offset_cascade",
    ]
    methods = [
        "causalsieve",
        "causalsieve_hinge",
        "unshielded_online_rls",
        "raw_max_z",
        "rolling_mad_z",
        "sklearn_isolation_forest",
        "pairwise_granger_z",
    ]
    meta = collect_system_metadata()
    overall: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    overall_ms: dict[str, list[float]] = {m: [] for m in methods}
    raw_tokens_all: list[int] = []
    digest_tokens_all: list[int] = []
    per_regime_summary: dict[str, Any] = {}

    print(f"=== Synthetic RCA Benchmark: {len(regimes)} regimes x {n_seeds} seeds = {len(regimes)*n_seeds} episodes ===")
    for regime in regimes:
        reg_rows: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
        reg_ms: dict[str, list[float]] = {m: [] for m in methods}
        reg_raw_tok: list[int] = []
        reg_dig_tok: list[int] = []

        for seed in range(n_seeds):
            ep = _gen_episode(regime, seed)
            reg_raw_tok.append(estimate_raw_tokens(ep.X))
            raw_tokens_all.append(reg_raw_tok[-1])

            # 1. CausalSieve (default linear + production Huber shock shield)
            ranked, ms, dig_tok = _run_causalsieve_episode(ep, shock_threshold=None, nonlinear_hinge=0)
            reg_dig_tok.append(dig_tok)
            digest_tokens_all.append(dig_tok)
            m_eval = eval_ranking(ranked.tolist(), ep.true_root)
            reg_rows["causalsieve"].append(m_eval)
            overall["causalsieve"].append(m_eval)
            reg_ms["causalsieve"].append(ms)
            overall_ms["causalsieve"].append(ms)

            # 2. CausalSieve with nonlinear_hinge=1
            ranked_h, ms_h, _ = _run_causalsieve_episode(ep, shock_threshold=None, nonlinear_hinge=1)
            m_eval_h = eval_ranking(ranked_h.tolist(), ep.true_root)
            reg_rows["causalsieve_hinge"].append(m_eval_h)
            overall["causalsieve_hinge"].append(m_eval_h)
            reg_ms["causalsieve_hinge"].append(ms_h)
            overall_ms["causalsieve_hinge"].append(ms_h)

            # 3. Unshielded Online RLS ablation (shock_threshold=0.0)
            ranked_u, ms_u, _ = _run_causalsieve_episode(ep, shock_threshold=0.0, nonlinear_hinge=0)
            m_eval_u = eval_ranking(ranked_u.tolist(), ep.true_root)
            reg_rows["unshielded_online_rls"].append(m_eval_u)
            overall["unshielded_online_rls"].append(m_eval_u)
            reg_ms["unshielded_online_rls"].append(ms_u)
            overall_ms["unshielded_online_rls"].append(ms_u)

            # 4..7 Baselines
            for b_name, fn in [
                ("raw_max_z", lambda: rank_raw_max_z(ep.X, ep.shock_start)),
                ("rolling_mad_z", lambda: rank_rolling_mad_z(ep.X, ep.shock_start)),
                ("sklearn_isolation_forest", lambda: rank_isolation_forest(ep.X, ep.shock_start, lag=2, seed=seed)),
                ("pairwise_granger_z", lambda: rank_pairwise_granger_z(ep.X, ep.shock_start, lag=2)),
            ]:
                t0 = time.perf_counter()
                _, b_ranked = fn()
                b_ms = (time.perf_counter() - t0) * 1000.0
                b_eval = eval_ranking(b_ranked.tolist(), ep.true_root)
                reg_rows[b_name].append(b_eval)
                overall[b_name].append(b_eval)
                reg_ms[b_name].append(b_ms)
                overall_ms[b_name].append(b_ms)

        per_regime_summary[regime] = {
            "n_episodes": n_seeds,
            "mean_raw_tokens": round(float(np.mean(reg_raw_tok)), 1),
            "mean_digest_tokens": round(float(np.mean(reg_dig_tok)), 1),
            "token_reduction_pct": round(100.0 * (1.0 - np.mean(reg_dig_tok) / np.mean(reg_raw_tok)), 3),
            "methods": {
                m: {
                    **summarize_ranking_results(reg_rows[m]),
                    "mean_ms": round(float(np.mean(reg_ms[m])), 3),
                }
                for m in methods
            },
        }
        print(f"\n--- {regime} (n={n_seeds}) ---")
        for m in methods:
            s = per_regime_summary[regime]["methods"][m]
            print(
                f"  {m:26s} | AC@1={s['ac1']:.3f} | AC@3={s['ac3']:.3f} | "
                f"MRR={s['mrr']:.3f} | {s['mean_ms']:7.2f} ms"
            )

    overall_summary = {
        m: {
            **summarize_ranking_results(overall[m]),
            "mean_ms": round(float(np.mean(overall_ms[m])), 3),
        }
        for m in methods
    }
    report = {
        "benchmark": "synthetic_rca_100_episodes",
        "system": meta,
        "total_episodes": len(regimes) * n_seeds,
        "token_compression": {
            "mean_raw_tokens": round(float(np.mean(raw_tokens_all)), 1),
            "mean_digest_tokens": round(float(np.mean(digest_tokens_all)), 1),
            "reduction_pct": round(100.0 * (1.0 - np.mean(digest_tokens_all) / np.mean(raw_tokens_all)), 3),
        },
        "overall": overall_summary,
        "per_regime": per_regime_summary,
    }

    print("\n=== OVERALL ACROSS 100 EPISODES ===")
    for m in methods:
        s = overall_summary[m]
        print(f"  {m:26s} | AC@1={s['ac1']:.3f} | AC@3={s['ac3']:.3f} | MRR={s['mrr']:.3f} | {s['mean_ms']:7.2f} ms")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved synthetic RCA results to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark()
