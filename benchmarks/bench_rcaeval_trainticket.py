"""Zero-leakage high-dimensional real Kubernetes RCA benchmark on RCAEval Train Ticket (RE1-TT).

Pre-Registered Zero-Leakage Protocol:
  - Dataset: phamquiluan/RCAEval on HuggingFace, Train Ticket (RE1-TT, 41+ microservices,
    D = 200-400+ container/service metrics per incident).
  - Canonical 25-case evaluation set (selected deterministically BEFORE observing any data):
      * All 5 fault-injected target services (alphabetical):
          ts-auth-service, ts-order-service, ts-route-service,
          ts-train-service, ts-travel-service
      * All 5 fault types (canonical order): cpu, mem, disk, delay, loss
      * Repetition: rep == 2 (25 independent real Kubernetes incidents, ZERO tuning)
  - Every downloaded metrics.parquet and inject_time.txt is cached under
    benchmarks/data/rcaeval_tt/ with exact SHA-256 provenance digests.
  - Strictly causal imputation: forward-fill then 0.0 (NEVER backward-fill).
  - Symmetric comparators (production defaults, zero retuning):
      * causalsieve_tier2         (ScreenedSieve, active_m=32, lag=2, zero inject_time knowledge)
      * causalsieve_tier1         (full dense CausalSieve, lag=2, symmetric baseline split)
      * causalsieve_tier1_blind   (full dense CausalSieve, lag=2, zero inject_time knowledge)
      * raw_max_z                 (classic 3-sigma Z-score, symmetric baseline split)
      * rolling_mad_z             (Baro-style robust median/MAD Z-score, symmetric split)
      * pairwise_granger_z        (bivariate pairwise Granger outflow + robust Z; uses baselines.py
                                   top-48 MAD pre-screen when active channels > 48; uncapped O(D^2)
                                   cost also timed directly for transparency)
      * sklearn_isolation_forest  (per-channel lagged IsolationForest, symmetric baseline split)

Saves exact provenance, metrics, timings, and breakdowns to:
  benchmarks/results/rcaeval_trainticket.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from _cabi import RawEngine, RawScreenedEngine  # noqa: E402
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

CACHE_DIR = REPO_ROOT / "benchmarks" / "data" / "rcaeval_tt"
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "rcaeval_trainticket.json"
HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"

# Pre-registered canonical 5 services x 5 faults x rep 2 = 25 Train Ticket cases.
TARGET_SERVICES: tuple[str, ...] = (
    "ts-auth-service",
    "ts-order-service",
    "ts-route-service",
    "ts-train-service",
    "ts-travel-service",
)
FAULT_TYPES: tuple[str, ...] = ("cpu", "mem", "disk", "delay", "loss")
EVAL_REPETITION = 2


@dataclass(frozen=True)
class TrainTicketCaseSpec:
    case_id: str
    system: str
    root_cause_service: str
    fault: str
    repetition: int


@dataclass
class TrainTicketCaseData:
    spec: TrainTicketCaseSpec
    timestamps: np.ndarray        # (T,) int64 unix seconds
    values: np.ndarray            # (T, D) float64 strictly causally imputed
    columns: list[str]            # length D metric names
    services: list[str]           # length D service names parsed as col.split("_", 1)[0]
    n_unique_services: int
    inject_time: int
    inject_index: int
    parquet_sha256: str
    inject_sha256: str
    parquet_bytes: int


def get_trainticket_specs(repetition: int = EVAL_REPETITION) -> list[TrainTicketCaseSpec]:
    specs: list[TrainTicketCaseSpec] = []
    for svc in TARGET_SERVICES:
        for fault in FAULT_TYPES:
            specs.append(
                TrainTicketCaseSpec(
                    case_id=f"re1tt_{svc}_{fault}_{repetition}",
                    system="re1tt",
                    root_cause_service=svc,
                    fault=fault,
                    repetition=repetition,
                )
            )
    return specs


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_if_missing(rel_path: str, local_path: Path) -> Path:
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE_URL}/{rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "causalsieve-benchmark/1.0"})
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    tmp_path.write_bytes(data)
    tmp_path.replace(local_path)
    return local_path


def parse_service_name(metric_col: str) -> str:
    """Deterministic service-name parser verified on rep-1 schema (`<service>_<metric>`)."""
    return metric_col.split("_", 1)[0]


def load_trainticket_case(
    spec: TrainTicketCaseSpec, cache_dir: Path = CACHE_DIR
) -> TrainTicketCaseData:
    """Download (if needed), SHA-256 hash, and strictly causally load a Train Ticket case."""
    case_dir = cache_dir / spec.case_id
    parquet_path = _download_if_missing(
        f"{spec.case_id}/metrics.parquet", case_dir / "metrics.parquet"
    )
    inject_path = _download_if_missing(
        f"{spec.case_id}/inject_time.txt", case_dir / "inject_time.txt"
    )

    parquet_sha = _sha256_file(parquet_path)
    inject_sha = _sha256_file(inject_path)
    inject_time = int(inject_path.read_text(encoding="utf-8").strip())

    df = pd.read_parquet(parquet_path)
    if "time" not in df.columns:
        raise ValueError(f"{spec.case_id}: missing 'time' column")

    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    timestamps = df["time"].to_numpy(dtype=np.int64)
    metric_cols = [c for c in df.columns if c != "time"]

    # Strictly causal imputation: forward-fill only, then 0.0 for leading NaNs (never bfill).
    val_df = df[metric_cols].astype(np.float64)
    val_df = val_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    values = np.ascontiguousarray(val_df.to_numpy(dtype=np.float64))

    services = [parse_service_name(c) for c in metric_cols]
    n_unique_services = len(set(services))
    inject_index = int(np.searchsorted(timestamps, inject_time, side="left"))

    return TrainTicketCaseData(
        spec=spec,
        timestamps=timestamps,
        values=values,
        columns=metric_cols,
        services=services,
        n_unique_services=n_unique_services,
        inject_time=inject_time,
        inject_index=inject_index,
        parquet_sha256=parquet_sha,
        inject_sha256=inject_sha,
        parquet_bytes=parquet_path.stat().st_size,
    )


def _aggregate_tier2_to_services(
    top_ch: np.ndarray,
    top_sc: np.ndarray,
    services: list[str],
    active_m: int,
) -> tuple[list[str], dict[str, float]]:
    """Aggregate Tier-2 ScreenedSieve channel outputs to service ranking (max-over-metrics).

    The first ``m = min(active_m, D)`` channels returned by ``cs_screened_push_batch`` are
    the promoted active set in exact Tier-1 ``cs_root_attribution`` rank order. Remaining
    channels ``m..D-1`` carry their Tier-2 screening scores and are sorted deterministically
    by ``(-score, channel_index)``. Each service takes its earliest (best-ranked) metric.
    """
    n_ch = len(top_ch)
    m = min(int(active_m), n_ch)

    ordered: list[tuple[int, float]] = [
        (int(top_ch[i]), float(top_sc[i])) for i in range(m)
    ]
    tail = sorted(
        [(int(top_ch[i]), float(top_sc[i])) for i in range(m, n_ch)],
        key=lambda pair: (-(pair[1] if np.isfinite(pair[1]) else 0.0), pair[0]),
    )
    ordered.extend(tail)

    ranked_services: list[str] = []
    svc_scores: dict[str, float] = {}
    for ch, sc in ordered:
        svc = services[ch]
        if svc not in svc_scores:
            svc_scores[svc] = sc if np.isfinite(sc) else 0.0
            ranked_services.append(svc)
    return ranked_services, svc_scores


def _eval_case_tier2(
    case: TrainTicketCaseData, *, active_m: int = 32, lag: int = 2
) -> tuple[list[str], float, int, bool]:
    X = case.values
    T, D = X.shape
    eff_m = min(active_m, D)

    t0 = time.perf_counter()
    with RawScreenedEngine(
        dim=D,
        active_m=eff_m,
        lag=lag,
        forgetting=0.995,
        ridge=1e-2,
    ) as eng:
        _, top_ch, top_sc = eng.push_batch(X, top_k=D)
        active_ch, f_mat, _ = eng.active_graph()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ranked_services, svc_scores = _aggregate_tier2_to_services(
        top_ch, top_sc, case.services, active_m=eff_m
    )
    promoted_services = {case.services[int(c)] for c in active_ch}
    true_promoted = case.spec.root_cause_service in promoted_services

    top5_svcs = [(s, round(svc_scores[s], 2)) for s in ranked_services[:5]]
    top_edges_idx = np.argwhere(f_mat > 8.0) if f_mat.size > 0 else np.empty((0, 2), dtype=int)
    top_edges = sorted(
        [
            (
                case.columns[int(active_ch[i])],
                case.columns[int(active_ch[j])],
                round(float(f_mat[i, j]), 1),
            )
            for i, j in top_edges_idx
            if i != j
        ],
        key=lambda e: -e[2],
    )[:8]
    digest_str = json.dumps(
        {
            "case": case.spec.case_id,
            "tier": 2,
            "T": T,
            "D": D,
            "active_m": int(len(active_ch)),
            "top_services": top5_svcs,
            "top_causal_edges": top_edges,
        },
        separators=(",", ":"),
    )
    digest_tokens = estimate_text_tokens(digest_str)
    return ranked_services, elapsed_ms, digest_tokens, true_promoted


def _eval_case_tier1(
    case: TrainTicketCaseData, *, blind: bool = False, lag: int = 2
) -> tuple[list[str], float, int]:
    X = case.values
    T, D = X.shape
    lookback = 0 if blind else max(1, T - case.inject_index)
    warmup = 0 if blind else case.inject_index

    t0 = time.perf_counter()
    with RawEngine(
        D,
        lag=lag,
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
            "tier": 1,
            "T": T,
            "D": D,
            "top_services": top5_svcs,
            "top_causal_edges": top_edges,
        },
        separators=(",", ":"),
    )
    digest_tokens = estimate_text_tokens(digest_str)
    return ranked_services, elapsed_ms, digest_tokens


def _time_uncapped_pairwise_granger_extrapolation(
    case: TrainTicketCaseData, lag: int = 2, ridge: float = 1e-2, sample_pairs: int = 1000
) -> dict[str, Any]:
    """Measure exact per-pair bivariate Granger regression cost and extrapolate full O(D^2)."""
    X = case.values
    T, D = X.shape
    b_end = max(lag + 15, min(int(case.inject_index), T - 1))
    base = X[:b_end]
    mu = np.mean(base, axis=0)
    sd = np.std(base, axis=0)
    active = np.where(sd > 1e-12)[0]
    Da = len(active)
    total_pairs = Da * (Da - 1)
    if total_pairs == 0:
        return {"case_id": case.spec.case_id, "D": D, "active_D": Da, "total_pairs": 0}

    Xb = (base[:, active] - mu[active]) / sd[active]
    N = Xb.shape[0] - lag
    lags = [
        np.column_stack([Xb[lag - ell : Xb.shape[0] - ell, i] for ell in range(1, lag + 1)])
        for i in range(Da)
    ]
    ones = np.ones((N, 1), dtype=np.float64)
    eye_full = ridge * np.eye(2 * lag + 1)

    n_run = min(sample_pairs, total_pairs)
    t0 = time.perf_counter()
    done = 0
    for ii in range(Da):
        for jj in range(Da):
            if ii == jj:
                continue
            Zf = np.hstack([lags[jj], lags[ii], ones])
            y = Xb[lag:, jj]
            b = np.linalg.solve(Zf.T @ Zf + eye_full, Zf.T @ y)
            _ = y - Zf @ b
            done += 1
            if done >= n_run:
                break
        if done >= n_run:
            break
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    extrap_ms = elapsed_ms * (total_pairs / max(1, done))
    return {
        "case_id": case.spec.case_id,
        "D": int(D),
        "active_D": int(Da),
        "total_pairs": int(total_pairs),
        "timed_pairs": int(done),
        "timed_ms": round(elapsed_ms, 2),
        "extrapolated_full_o_d2_ms": round(extrap_ms, 1),
        "extrapolated": done < total_pairs,
    }


def _eval_from_rank(rank: int, n_services: int) -> dict[str, float]:
    ac1 = 1.0 if rank <= 1 else 0.0
    ac2 = 1.0 if rank <= 2 else 0.0
    ac3 = 1.0 if rank <= 3 else 0.0
    ac4 = 1.0 if rank <= 4 else 0.0
    ac5 = 1.0 if rank <= 5 else 0.0
    avg5 = (ac1 + ac2 + ac3 + ac4 + ac5) / 5.0
    mrr = 1.0 / float(rank) if rank <= n_services else 0.0
    return {
        "rank": float(rank),
        "ac1": ac1,
        "ac3": ac3,
        "ac5": ac5,
        "avg5": avg5,
        "mrr": mrr,
    }


def run_benchmark(
    *,
    skip_iforest: bool = False,
    checkpoint_path: Path | None = None,
    max_cases_per_run: int | None = None,
) -> dict[str, Any]:
    meta = collect_system_metadata()
    with RawEngine(4, 1) as probe:
        simd_isa = probe.simd_isa

    specs = get_trainticket_specs(EVAL_REPETITION)
    methods = [
        "causalsieve_tier2",
        "causalsieve_tier1",
        "causalsieve_tier1_blind",
        "raw_max_z",
        "rolling_mad_z",
        "pairwise_granger_z",
    ]
    if not skip_iforest:
        methods.append("sklearn_isolation_forest")

    cached_by_id: dict[str, dict[str, Any]] = {}
    uncapped_pw_timing: dict[str, Any] | None = None
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if ckpt.get("lib_sha256") == meta["lib_sha256"]:
            for rec in ckpt.get("cases", []):
                cached_by_id[rec["case_id"]] = rec
            uncapped_pw_timing = ckpt.get("pairwise_granger_uncapped_timing")

    print(
        f"=== RCAEval Train Ticket (RE1-TT) Zero-Leakage High-D Benchmark "
        f"(n={len(specs)}, rep={EVAL_REPETITION}, ISA={simd_isa}, cached={len(cached_by_id)}) ===",
        flush=True,
    )

    new_run = 0
    for idx, spec in enumerate(specs, start=1):
        if spec.case_id in cached_by_id:
            ce = cached_by_id[spec.case_id]
            p = ce["predictions"]
            isof_str = (
                f" | IsoF={p['sklearn_isolation_forest']['rank']:2d} "
                f"({p['sklearn_isolation_forest']['ms']:7.0f}ms)"
                if not skip_iforest and "sklearn_isolation_forest" in p
                else ""
            )
            print(
                f"  [{idx:2d}/{len(specs)}] {spec.case_id:34s} (cached) | "
                f"T2={p['causalsieve_tier2']['rank']:2d} | T1={p['causalsieve_tier1']['rank']:2d} | "
                f"RawZ={p['raw_max_z']['rank']:2d} | MAD={p['rolling_mad_z']['rank']:2d}{isof_str}",
                flush=True,
            )
            continue

        if max_cases_per_run is not None and new_run >= max_cases_per_run:
            print(
                f"Reached --max-cases-per-run={max_cases_per_run} ({len(cached_by_id)}/{len(specs)} done); "
                f"checkpoint saved to {checkpoint_path}.",
                flush=True,
            )
            return {"status": "partial", "completed": len(cached_by_id), "total": len(specs)}

        case = load_trainticket_case(spec)
        T, D = case.values.shape
        raw_tok = estimate_raw_tokens(case.values, case.columns)

        if uncapped_pw_timing is None:
            uncapped_pw_timing = _time_uncapped_pairwise_granger_extrapolation(case, lag=2)

        # 1. Tier-2 ScreenedSieve (active_m=32, lag=2, zero inject_time knowledge)
        t2_svcs, t2_ms, t2_tok, t2_promoted = _eval_case_tier2(case, active_m=32, lag=2)

        # 2. Tier-1 full dense CausalSieve (lag=2, symmetric baseline split)
        t1_svcs, t1_ms, t1_tok = _eval_case_tier1(case, blind=False, lag=2)

        # 3. Tier-1 full dense CausalSieve blind (lag=2, zero inject_time knowledge)
        t1b_svcs, t1b_ms, _ = _eval_case_tier1(case, blind=True, lag=2)

        method_outputs: list[tuple[str, list[str], float]] = [
            ("causalsieve_tier2", t2_svcs, t2_ms),
            ("causalsieve_tier1", t1_svcs, t1_ms),
            ("causalsieve_tier1_blind", t1b_svcs, t1b_ms),
        ]

        baseline_specs: list[tuple[str, Any]] = [
            ("raw_max_z", lambda c=case: rank_raw_max_z(c.values, c.inject_index)),
            ("rolling_mad_z", lambda c=case: rank_rolling_mad_z(c.values, c.inject_index)),
            (
                "pairwise_granger_z",
                lambda c=case: rank_pairwise_granger_z(c.values, c.inject_index, lag=2),
            ),
        ]
        if not skip_iforest:
            baseline_specs.append(
                (
                    "sklearn_isolation_forest",
                    lambda c=case, s=idx: rank_isolation_forest(
                        c.values, c.inject_index, lag=2, seed=s
                    ),
                )
            )

        for b_name, fn in baseline_specs:
            t0 = time.perf_counter()
            scores, _ = fn()
            b_ms = (time.perf_counter() - t0) * 1000.0
            b_svcs, _ = aggregate_channel_scores_to_services(scores, case.services)
            method_outputs.append((b_name, b_svcs, b_ms))

        case_entry: dict[str, Any] = {
            "case_id": spec.case_id,
            "system": spec.system,
            "fault": spec.fault,
            "repetition": spec.repetition,
            "true_service": spec.root_cause_service,
            "n_timesteps": int(T),
            "n_metrics": int(D),
            "n_services": int(case.n_unique_services),
            "inject_time": case.inject_time,
            "inject_index": case.inject_index,
            "parquet_sha256": case.parquet_sha256,
            "inject_sha256": case.inject_sha256,
            "parquet_bytes": case.parquet_bytes,
            "raw_tokens": raw_tok,
            "digest_tokens_tier2": t2_tok,
            "digest_tokens_tier1": t1_tok,
            "tier2_true_service_promoted": bool(t2_promoted),
            "predictions": {},
        }

        for m_name, ranked_svcs, m_ms in method_outputs:
            ev = eval_ranking(ranked_svcs, spec.root_cause_service)
            case_entry["predictions"][m_name] = {
                "rank": int(ev["rank"]),
                "top5": ranked_svcs[:5],
                "ms": round(m_ms, 2),
            }

        cached_by_id[spec.case_id] = case_entry
        new_run += 1

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            ordered_ckpt = [cached_by_id[s.case_id] for s in specs if s.case_id in cached_by_id]
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "lib_sha256": meta["lib_sha256"],
                        "pairwise_granger_uncapped_timing": uncapped_pw_timing,
                        "cases": ordered_ckpt,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        p = case_entry["predictions"]
        isof_str = (
            f" | IsoF={p['sklearn_isolation_forest']['rank']:2d} "
            f"({p['sklearn_isolation_forest']['ms']:7.0f}ms)"
            if not skip_iforest
            else ""
        )
        print(
            f"  [{idx:2d}/{len(specs)}] {spec.case_id:34s} (D={D:3d}, S={case.n_unique_services:2d}, T={T:4d}) | "
            f"T2={p['causalsieve_tier2']['rank']:2d} ({t2_ms:5.1f}ms, prom={'Y' if t2_promoted else 'N'}) | "
            f"T1={p['causalsieve_tier1']['rank']:2d} ({t1_ms:5.1f}ms) | "
            f"T1b={p['causalsieve_tier1_blind']['rank']:2d} | "
            f"RawZ={p['raw_max_z']['rank']:2d} | "
            f"MAD={p['rolling_mad_z']['rank']:2d} | "
            f"PWGr={p['pairwise_granger_z']['rank']:2d}{isof_str}",
            flush=True,
        )

    case_records = [cached_by_id[s.case_id] for s in specs]
    rows_by_method: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    ms_by_method: dict[str, list[float]] = {m: [] for m in methods}
    by_fault: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {m: [] for m in methods}
    )
    by_service: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {m: [] for m in methods}
    )

    dims_list = [int(c["n_metrics"]) for c in case_records]
    svcs_list = [int(c["n_services"]) for c in case_records]
    ticks_list = [int(c["n_timesteps"]) for c in case_records]
    raw_tokens = [int(c["raw_tokens"]) for c in case_records]
    digest_tokens_t2 = [int(c["digest_tokens_tier2"]) for c in case_records]
    digest_tokens_t1 = [int(c["digest_tokens_tier1"]) for c in case_records]
    tier2_promoted_flags = [bool(c["tier2_true_service_promoted"]) for c in case_records]

    for c in case_records:
        n_svc = int(c["n_services"])
        for m in methods:
            pred = c["predictions"][m]
            ev = _eval_from_rank(int(pred["rank"]), n_svc)
            rows_by_method[m].append(ev)
            ms_by_method[m].append(float(pred["ms"]))
            by_fault[c["fault"]][m].append(ev)
            by_service[c["true_service"]][m].append(ev)

    summary = {
        m: {
            **summarize_ranking_results(rows_by_method[m]),
            "mean_ms": round(float(np.mean(ms_by_method[m])), 2),
            "median_ms": round(float(np.median(ms_by_method[m])), 2),
        }
        for m in methods
    }

    print(f"\n--- TRAIN TICKET (RE1-TT) SUMMARY (n={len(specs)}, rep={EVAL_REPETITION}) ---", flush=True)
    print(
        f"  Dimensionality: D in [{min(dims_list)}, {max(dims_list)}] (mean={np.mean(dims_list):.1f}), "
        f"Services in [{min(svcs_list)}, {max(svcs_list)}] (mean={np.mean(svcs_list):.1f}), "
        f"T in [{min(ticks_list)}, {max(ticks_list)}] (mean={np.mean(ticks_list):.1f})",
        flush=True,
    )
    print(
        f"  Tier-2 active_m=32 true-service promotion rate: "
        f"{sum(tier2_promoted_flags)}/{len(tier2_promoted_flags)} "
        f"({100.0 * np.mean(tier2_promoted_flags):.1f}%)",
        flush=True,
    )
    for m in methods:
        s = summary[m]
        print(
            f"  {m:26s} | AC@1={s['ac1']:.3f} | AC@3={s['ac3']:.3f} | AC@5={s['ac5']:.3f} | "
            f"Avg@5={s['avg5']:.3f} | MRR={s['mrr']:.3f} | MeanRank={s['mean_rank']:5.2f} | "
            f"{s['mean_ms']:8.1f} ms/case",
            flush=True,
        )

    t2_red = round(100.0 * (1.0 - float(np.mean(digest_tokens_t2)) / float(np.mean(raw_tokens))), 3)
    t1_red = round(100.0 * (1.0 - float(np.mean(digest_tokens_t1)) / float(np.mean(raw_tokens))), 3)
    print(
        f"  Token compression (Tier-2): {np.mean(raw_tokens):.1f} -> {np.mean(digest_tokens_t2):.1f} "
        f"({t2_red:.3f}% reduction)",
        flush=True,
    )
    print(
        f"  Token compression (Tier-1): {np.mean(raw_tokens):.1f} -> {np.mean(digest_tokens_t1):.1f} "
        f"({t1_red:.3f}% reduction)",
        flush=True,
    )
    if uncapped_pw_timing:
        print(
            f"  Uncapped O(D^2) pairwise Granger probe ({uncapped_pw_timing['case_id']}, "
            f"active_D={uncapped_pw_timing['active_D']}, {uncapped_pw_timing['total_pairs']} pairs): "
            f"{uncapped_pw_timing['extrapolated_full_o_d2_ms']} ms/case (extrapolated from "
            f"{uncapped_pw_timing['timed_pairs']} pairs in {uncapped_pw_timing['timed_ms']} ms)",
            flush=True,
        )

    report: dict[str, Any] = {
        "benchmark": "rcaeval_trainticket_high_dimensional_kubernetes",
        "dataset_source": "https://huggingface.co/datasets/phamquiluan/RCAEval",
        "protocol": {
            "zero_leakage_holdout": (
                "Pre-registered canonical 25-case evaluation set from RE1-TT: "
                "all 5 target services x all 5 fault types (cpu, mem, disk, delay, loss), "
                "repetition 2, zero tuning against production defaults. "
                "Schema verified on rep-1 single case only."
            ),
            "symmetric_comparators": (
                "Identical strictly causal imputation (ffill then 0.0, never bfill) and "
                "max-over-metrics service aggregation across all methods. "
                "causalsieve_tier2 and causalsieve_tier1_blind receive ZERO inject_time knowledge; "
                "causalsieve_tier1, raw_max_z, rolling_mad_z, pairwise_granger_z, and "
                "sklearn_isolation_forest share the symmetric baseline[:inject_index] / "
                "incident[inject_index:] split. pairwise_granger_z uses baselines.py's built-in "
                "top-48 robust-MAD pre-screen when active channels > 48 (2,256 pairs/case)."
            ),
            "target_services": list(TARGET_SERVICES),
            "fault_types": list(FAULT_TYPES),
            "repetition": EVAL_REPETITION,
            "simd_isa": simd_isa,
            "lib_sha256": meta["lib_sha256"],
        },
        "system": meta,
        "dataset_stats": {
            "n_cases": len(specs),
            "n_metrics": {
                "min": int(min(dims_list)),
                "median": float(np.median(dims_list)),
                "mean": round(float(np.mean(dims_list)), 2),
                "max": int(max(dims_list)),
            },
            "n_services": {
                "min": int(min(svcs_list)),
                "median": float(np.median(svcs_list)),
                "mean": round(float(np.mean(svcs_list)), 2),
                "max": int(max(svcs_list)),
            },
            "n_timesteps": {
                "min": int(min(ticks_list)),
                "median": float(np.median(ticks_list)),
                "mean": round(float(np.mean(ticks_list)), 2),
                "max": int(max(ticks_list)),
            },
            "tier2_active_m": 32,
            "tier2_true_service_promoted_count": int(sum(tier2_promoted_flags)),
            "tier2_true_service_promoted_rate": round(float(np.mean(tier2_promoted_flags)), 4),
        },
        "token_compression": {
            "mean_raw_tokens": round(float(np.mean(raw_tokens)), 1),
            "tier2": {
                "mean_digest_tokens": round(float(np.mean(digest_tokens_t2)), 1),
                "reduction_pct": t2_red,
            },
            "tier1": {
                "mean_digest_tokens": round(float(np.mean(digest_tokens_t1)), 1),
                "reduction_pct": t1_red,
            },
        },
        "pairwise_granger_uncapped_timing": uncapped_pw_timing,
        "summary": summary,
        "by_fault": {
            fault_name: {m: summarize_ranking_results(by_fault[fault_name][m]) for m in methods}
            for fault_name in FAULT_TYPES
        },
        "by_service": {
            svc_name: {m: summarize_ranking_results(by_service[svc_name][m]) for m in methods}
            for svc_name in TARGET_SERVICES
        },
        "cases": case_records,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"\nSaved Train Ticket benchmark results to {RESULTS_PATH}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-leakage RCAEval Train Ticket (RE1-TT) high-D Kubernetes benchmark."
    )
    parser.add_argument(
        "--skip-iforest",
        action="store_true",
        help="Skip slow per-channel sklearn IsolationForest (for fast smoke checks only).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional JSON checkpoint file for resumable execution across runs.",
    )
    parser.add_argument(
        "--max-cases-per-run",
        type=int,
        default=None,
        help="Optional cap on new cases executed in this invocation (used with --checkpoint).",
    )
    args = parser.parse_args()
    run_benchmark(
        skip_iforest=args.skip_iforest,
        checkpoint_path=args.checkpoint,
        max_cases_per_run=args.max_cases_per_run,
    )


if __name__ == "__main__":
    main()
