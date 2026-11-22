"""Zero-leakage real Kubernetes microservice RCA benchmark on phamquiluan/RCAEval (HuggingFace).

Amendment A8 Sealed-Holdout Discipline:
  - 50 real Kubernetes incidents across Online Boutique (RE1-OB) and Sock Shop (RE1-SS),
    covering all 5 fault types (cpu, mem, disk, delay, loss) and all 10 target services:
      * DEV split (rep == 1):            10 real cases (used ONLY to verify parsing & pipeline)
      * SEALED HOLDOUT (reps 2, 3, 4, 5): 40 real cases (ZERO tuning, evaluated once frozen)
  - Amendment A10 Symmetric Comparators:
      * causalsieve (Huber shock-shielded, normalize_scale=1, lookback=faulty_window)
      * causalsieve_blind (ZERO knowledge of inject_time: lookback=0 autonomous onset detection)
      * raw_max_z (classic 3-sigma Z-score)
      * rolling_mad_z (Baro-style robust median/MAD Z-score)
      * sklearn_isolation_forest (per-channel lagged IsolationForest)
      * pairwise_granger_z (bivariate pairwise Granger outflow + robust Z)

Records exact case list, SHA-256 digests, per-system & per-fault breakdowns, latency, and
token compression ratio to benchmarks/results/rcaeval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from _cabi import RawEngine  # noqa: E402
from baselines import (  # noqa: E402
    aggregate_channel_scores_to_services,
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
from rcaeval_loader import RCAEvalCaseData, get_case_specs, load_case  # noqa: E402

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "rcaeval.json"


def _eval_case_causalsieve(
    case: RCAEvalCaseData, *, blind: bool = False
) -> tuple[list[str], float, int]:
    X = case.values
    T, D = X.shape
    lookback = 0 if blind else max(1, T - case.inject_index)
    warmup = 0 if blind else case.inject_index
    t0 = time.perf_counter()
    with RawEngine(
        D,
        lag=2,
        forgetting=0.995,
        warmup_ticks=warmup,
        raw_ring_capacity=max(8192, T + 16),
        normalize_scale=1,
    ) as eng:
        rc, _, _, _ = eng.push_batch(X, return_results=True)
        assert rc == 0
        rca = eng.root_attribution(lookback_ticks=lookback)
        f_mat, _ = eng.granger_matrix()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ranked_services, svc_scores = aggregate_channel_scores_to_services(
        rca["root_scores"], case.services, onset_ticks=rca["onset_tick"]
    )

    # Compact Caveman Causal Digest token count (top-5 services + top-8 metric causal edges)
    top5_svcs = [(s, round(svc_scores[s], 2)) for s in ranked_services[:5]]
    top_edges_idx = np.argwhere(f_mat > 8.0)
    top_edges = sorted(
        [
            (case.columns[i], case.columns[j], round(float(f_mat[i, j]), 1))
            for i, j in top_edges_idx
            if i != j
        ],
        key=lambda e: -e[2],
    )[:8]
    digest_str = json.dumps(
        {
            "case": case.spec.case_id,
            "T": T,
            "D": D,
            "top_services": top5_svcs,
            "top_causal_edges": top_edges,
        },
        separators=(",", ":"),
    )
    digest_tokens = estimate_text_tokens(digest_str)
    return ranked_services, elapsed_ms, digest_tokens


def _evaluate_split(split_name: str, specs: list[Any]) -> dict[str, Any]:
    methods = [
        "causalsieve",
        "causalsieve_blind",
        "raw_max_z",
        "rolling_mad_z",
        "sklearn_isolation_forest",
        "pairwise_granger_z",
    ]
    rows_by_method: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    ms_by_method: dict[str, list[float]] = {m: [] for m in methods}
    by_system: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {m: [] for m in methods}
    )
    by_fault: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {m: [] for m in methods}
    )
    raw_tokens: list[int] = []
    digest_tokens: list[int] = []
    case_records: list[dict[str, Any]] = []

    print(f"\n=== Evaluating RCAEval [{split_name.upper()}] (n={len(specs)} real Kubernetes cases) ===")
    for idx, spec in enumerate(specs, start=1):
        case = load_case(spec)
        raw_tok = estimate_raw_tokens(case.values, case.columns)
        raw_tokens.append(raw_tok)

        case_entry: dict[str, Any] = {
            "case_id": spec.case_id,
            "system": spec.system,
            "fault": spec.fault,
            "repetition": spec.repetition,
            "split": spec.split,
            "true_service": spec.root_cause_service,
            "n_timesteps": int(case.values.shape[0]),
            "n_metrics": int(case.values.shape[1]),
            "inject_time": case.inject_time,
            "inject_index": case.inject_index,
            "parquet_sha256": case.parquet_sha256,
            "inject_sha256": case.inject_sha256,
            "raw_tokens": raw_tok,
            "predictions": {},
        }

        # 1. CausalSieve (symmetric faulty window)
        cs_svcs, cs_ms, dig_tok = _eval_case_causalsieve(case, blind=False)
        digest_tokens.append(dig_tok)
        case_entry["digest_tokens"] = dig_tok

        # 2. CausalSieve blind (zero knowledge of inject_time)
        csb_svcs, csb_ms, _ = _eval_case_causalsieve(case, blind=True)

        # 3..6 Baselines (symmetric baseline=[:inject_index], incident=[inject_index:])
        method_outputs: list[tuple[str, list[str], float]] = [
            ("causalsieve", cs_svcs, cs_ms),
            ("causalsieve_blind", csb_svcs, csb_ms),
        ]

        for b_name, fn in [
            ("raw_max_z", lambda: rank_raw_max_z(case.values, case.inject_index)),
            ("rolling_mad_z", lambda: rank_rolling_mad_z(case.values, case.inject_index)),
            (
                "sklearn_isolation_forest",
                lambda: rank_isolation_forest(case.values, case.inject_index, lag=2, seed=idx),
            ),
            (
                "pairwise_granger_z",
                lambda: rank_pairwise_granger_z(case.values, case.inject_index, lag=2),
            ),
        ]:
            t0 = time.perf_counter()
            scores, _ = fn()
            b_ms = (time.perf_counter() - t0) * 1000.0
            b_svcs, _ = aggregate_channel_scores_to_services(scores, case.services)
            method_outputs.append((b_name, b_svcs, b_ms))

        for m_name, ranked_svcs, m_ms in method_outputs:
            ev = eval_ranking(ranked_svcs, spec.root_cause_service)
            rows_by_method[m_name].append(ev)
            ms_by_method[m_name].append(m_ms)
            by_system[spec.system][m_name].append(ev)
            by_fault[spec.fault][m_name].append(ev)
            case_entry["predictions"][m_name] = {
                "rank": int(ev["rank"]),
                "top5": ranked_svcs[:5],
                "ms": round(m_ms, 2),
            }

        case_records.append(case_entry)
        print(
            f"  [{idx:2d}/{len(specs)}] {spec.case_id:34s} (true={spec.root_cause_service:21s}) | "
            f"CS={case_entry['predictions']['causalsieve']['rank']:2d} ({cs_ms:5.1f}ms) | "
            f"CS-blind={case_entry['predictions']['causalsieve_blind']['rank']:2d} | "
            f"RawZ={case_entry['predictions']['raw_max_z']['rank']:2d} | "
            f"MAD={case_entry['predictions']['rolling_mad_z']['rank']:2d} | "
            f"IsoF={case_entry['predictions']['sklearn_isolation_forest']['rank']:2d} | "
            f"PWGr={case_entry['predictions']['pairwise_granger_z']['rank']:2d}"
        )

    summary = {
        m: {
            **summarize_ranking_results(rows_by_method[m]),
            "mean_ms": round(float(np.mean(ms_by_method[m])), 2),
            "median_ms": round(float(np.median(ms_by_method[m])), 2),
        }
        for m in methods
    }

    print(f"\n--- {split_name.upper()} SUMMARY (n={len(specs)}) ---")
    for m in methods:
        s = summary[m]
        print(
            f"  {m:26s} | AC@1={s['ac1']:.3f} | AC@3={s['ac3']:.3f} | AC@5={s['ac5']:.3f} | "
            f"Avg@5={s['avg5']:.3f} | MRR={s['mrr']:.3f} | {s['mean_ms']:7.1f} ms/case"
        )

    return {
        "n_cases": len(specs),
        "token_compression": {
            "mean_raw_tokens": round(float(np.mean(raw_tokens)), 1),
            "mean_digest_tokens": round(float(np.mean(digest_tokens)), 1),
            "reduction_pct": round(100.0 * (1.0 - np.mean(digest_tokens) / np.mean(raw_tokens)), 3),
        },
        "summary": summary,
        "by_system": {
            sys_name: {m: summarize_ranking_results(by_system[sys_name][m]) for m in methods}
            for sys_name in sorted(by_system.keys())
        },
        "by_fault": {
            fault_name: {m: summarize_ranking_results(by_fault[fault_name][m]) for m in methods}
            for fault_name in sorted(by_fault.keys())
        },
        "cases": case_records,
    }


def run_benchmark(split: str = "all") -> dict[str, Any]:
    meta = collect_system_metadata()
    with RawEngine(4, 1) as probe:
        simd_isa = probe.simd_isa

    report: dict[str, Any] = {
        "benchmark": "rcaeval_real_kubernetes_microservices",
        "dataset_source": "https://huggingface.co/datasets/phamquiluan/RCAEval",
        "protocol": {
            "amendment_a8": "Pre-registered canonical Latin square across RE1-OB and RE1-SS; DEV = rep 1 (10 cases), SEALED HOLDOUT = reps 2..5 (40 cases, zero tuning).",
            "amendment_a10": "Symmetric baseline/incident split at inject_time (warmup_ticks=inject_index, production defaults), symmetric max-over-metrics service aggregation, strictly causal imputation.",
            "simd_isa": simd_isa,
            "lib_sha256": meta["lib_sha256"],
        },
        "system": meta,
        "splits": {},
    }

    if split in ("dev", "all"):
        report["splits"]["dev"] = _evaluate_split("dev", get_case_specs("dev"))
    if split in ("holdout", "all"):
        report["splits"]["holdout"] = _evaluate_split("holdout", get_case_specs("holdout"))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved RCAEval benchmark results to {RESULTS_PATH}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-leakage RCAEval Kubernetes benchmark.")
    parser.add_argument(
        "--split",
        choices=["dev", "holdout", "all"],
        default="all",
        help="Which split to run (default: all).",
    )
    args = parser.parse_args()
    run_benchmark(split=args.split)


if __name__ == "__main__":
    main()
