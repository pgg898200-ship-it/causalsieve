"""Full-Suite Zero-Leakage RCAEval RE1 + RE2 OpenTelemetry Microservice RCA Benchmark.

Evaluates ALL 645 real Kubernetes microservice outage incidents from
`phamquiluan/RCAEval` (Hugging Face) across all 6 canonical datasets:
  - RE1 (375 cases): RE1-OB (125), RE1-SS (125), RE1-TT (125) — 5 fault types (cpu, mem, disk, delay, loss)
  - RE2 (270 cases): RE2-OB (90),  RE2-SS (90),  RE2-TT (90)  — 6 fault types (+ socket exhaustion)

Zero-Leakage & Symmetric-Comparator Discipline:
  - ZERO hyperparameter retuning: uses the exact frozen v0.1.0 CausalSieve defaults
    (`lag=2, forgetting=0.995, normalize_scale=1`).
  - Identical strictly causal missing-value imputation (`ffill` then `0.0`, NEVER `bfill`),
    identical `<service>_<metric>` parser, and identical max-over-metrics service aggregation
    across all methods.
  - Comparators evaluated on every single case:
      * `causalsieve`          — Full dense Tier-1 CausalSieve (symmetric `warmup_ticks=inject_index`, `lookback=T-inject_index`)
      * `causalsieve_blind`    — Full dense Tier-1 CausalSieve with ZERO `inject_time` knowledge (`warmup_ticks=0, lookback=0`)
      * `causalsieve_tier2`    — High-D streaming `ScreenedSieve` (`active_m=min(32, D), lag=2`, ZERO `inject_time` knowledge)
      * `baro_fse24`           — Faithful BARO (Pham et al., FSE 2024: normal-window median/IQR robust peak |z|)
      * `nsigma_rcaeval`       — Exact RCAEval N-Sigma baseline (normal-window mean/std peak |z|)
      * `circa_structural_rca` — CIRCA-style conditional linear regression residual RCA (Li et al., KDD 2022)
      * `rolling_mad_z`        — Robust median/MAD Z-score baseline
      * `pairwise_granger_z`   — Bivariate pairwise Granger outflow + robust Z baseline

Records exact `lib_sha256`, `lib_scalar_sha256`, `cases_parquet_sha256`, every case's
`parquet_sha256` & `inject_sha256`, micro/macro overall summaries, per-suite (`RE1`, `RE2`),
per-dataset (`RE1-OB..RE2-TT`), and per-fault (`cpu..socket`) breakdowns to:
  `benchmarks/results/rcaeval_suite.json`
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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
    baro_fse24,
    circa_structural_rca,
    collect_system_metadata,
    estimate_raw_tokens,
    estimate_text_tokens,
    eval_ranking,
    nsigma_rcaeval,
    rank_pairwise_granger_z,
    rank_rolling_mad_z,
    summarize_ranking_results,
)

HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"
CACHE_DIR = REPO_ROOT / "benchmarks" / "data" / "rcaeval_hf"
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "rcaeval_suite.json"

ALL_DATASETS: tuple[str, ...] = (
    "RE1-OB",
    "RE1-SS",
    "RE1-TT",
    "RE2-OB",
    "RE2-SS",
    "RE2-TT",
)
ALL_FAULTS: tuple[str, ...] = ("cpu", "mem", "disk", "delay", "loss", "socket")
METHODS: tuple[str, ...] = (
    "causalsieve",
    "causalsieve_blind",
    "causalsieve_tier2",
    "baro_fse24",
    "nsigma_rcaeval",
    "circa_structural_rca",
    "rolling_mad_z",
    "pairwise_granger_z",
)


@dataclass(frozen=True)
class SuiteCaseSpec:
    case_id: str
    dataset: str
    suite: str
    system: str
    root_cause_service: str
    fault: str
    repetition: int
    cases_parquet_inject_time: int


@dataclass
class SuiteCaseData:
    spec: SuiteCaseSpec
    timestamps: np.ndarray        # (T,) int64 unix seconds
    values: np.ndarray            # (T, D) float64 strictly causally imputed
    columns: list[str]            # length D metric names
    services: list[str]           # length D service names parsed as col.split("_", 1)[0]
    n_unique_services: int
    nan_fraction: float
    inject_time: int
    inject_index: int
    parquet_sha256: str
    inject_sha256: str
    parquet_bytes: int


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
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "causalsieve-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            tmp = local_path.with_suffix(local_path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(local_path)
            return local_path
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {rel_path}: {last_err}")


def load_suite_specs(
    datasets: tuple[str, ...] = ALL_DATASETS,
    cache_dir: Path = CACHE_DIR,
) -> tuple[list[SuiteCaseSpec], str]:
    """Load canonical RE1 + RE2 case specs from cached `cases.parquet`."""
    cases_pq = _download_if_missing("cases.parquet", cache_dir / "cases.parquet")
    cases_sha = _sha256_file(cases_pq)
    df = pd.read_parquet(cases_pq)
    ds_set = set(datasets)
    sub = df[df["dataset"].isin(ds_set)].copy()
    ds_order = {ds: i for i, ds in enumerate(ALL_DATASETS)}
    sub["_ds_ord"] = sub["dataset"].map(lambda d: ds_order.get(str(d), 999))
    sub = sub.sort_values(
        ["_ds_ord", "root_cause_service", "fault", "repetition", "case"], kind="mergesort"
    ).reset_index(drop=True)

    specs: list[SuiteCaseSpec] = []
    for _, row in sub.iterrows():
        specs.append(
            SuiteCaseSpec(
                case_id=str(row["case"]),
                dataset=str(row["dataset"]),
                suite=str(row["suite"]),
                system=str(row["system"]),
                root_cause_service=str(row["root_cause_service"]),
                fault=str(row["fault"]),
                repetition=int(row["repetition"]),
                cases_parquet_inject_time=int(row["inject_time"]),
            )
        )
    return specs, cases_sha


def ensure_cases_cached(
    specs: list[SuiteCaseSpec],
    cache_dir: Path = CACHE_DIR,
    workers: int = 16,
) -> None:
    """Download any missing `metrics.parquet` and `inject_time.txt` files in parallel."""
    missing = [
        s.case_id
        for s in specs
        if not (cache_dir / s.case_id / "metrics.parquet").is_file()
        or not (cache_dir / s.case_id / "inject_time.txt").is_file()
    ]
    if not missing:
        return
    print(f"Downloading {len(missing)} missing RCAEval cases with {workers} workers...", flush=True)

    def _fetch(cid: str) -> None:
        cdir = cache_dir / cid
        _download_if_missing(f"{cid}/metrics.parquet", cdir / "metrics.parquet")
        _download_if_missing(f"{cid}/inject_time.txt", cdir / "inject_time.txt")

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fetch, missing))


def parse_service_name(metric_col: str) -> str:
    """Canonical RCAEval `<service>_<metric>` parser."""
    return metric_col.split("_", 1)[0]


def load_suite_case(spec: SuiteCaseSpec, cache_dir: Path = CACHE_DIR) -> SuiteCaseData:
    """Hash-verify and strictly causally load a single RCAEval RE1/RE2 case."""
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

    raw_vals = df[metric_cols].astype(np.float64)
    nan_fraction = float(raw_vals.isna().mean().mean()) if metric_cols else 0.0

    # Strictly causal imputation: forward-fill only, then 0.0 for leading NaNs (never bfill)
    val_df = raw_vals.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    values = np.ascontiguousarray(val_df.to_numpy(dtype=np.float64))

    services = [parse_service_name(c) for c in metric_cols]
    n_unique_services = len(set(services))
    inject_index = int(np.searchsorted(timestamps, inject_time, side="left"))

    return SuiteCaseData(
        spec=spec,
        timestamps=timestamps,
        values=values,
        columns=metric_cols,
        services=services,
        n_unique_services=n_unique_services,
        nan_fraction=nan_fraction,
        inject_time=inject_time,
        inject_index=inject_index,
        parquet_sha256=parquet_sha,
        inject_sha256=inject_sha,
        parquet_bytes=parquet_path.stat().st_size,
    )


def _eval_case_causalsieve(
    case: SuiteCaseData, *, blind: bool = False
) -> tuple[list[str], float, int]:
    """Full dense Tier-1 CausalSieve with exact frozen v0.1.0 configuration."""
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
        assert rc == 0, f"{case.spec.case_id}: push_batch failed with rc={rc}"
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
            "T": T,
            "D": D,
            "top_services": top5_svcs,
            "top_causal_edges": top_edges,
        },
        separators=(",", ":"),
    )
    digest_tokens = estimate_text_tokens(digest_str)
    return ranked_services, elapsed_ms, digest_tokens


def _aggregate_tier2_to_services(
    top_ch: np.ndarray,
    top_sc: np.ndarray,
    services: list[str],
    active_m: int,
) -> tuple[list[str], dict[str, float]]:
    n_ch = len(top_ch)
    m = min(int(active_m), n_ch)
    ordered: list[tuple[int, float]] = [(int(top_ch[i]), float(top_sc[i])) for i in range(m)]
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
    case: SuiteCaseData, *, active_m: int = 32, lag: int = 2
) -> tuple[list[str], float, int]:
    """Tier-2 streaming ScreenedSieve (blind: zero inject_time knowledge)."""
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
    return ranked_services, elapsed_ms, digest_tokens


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


def _summarize_group(
    records: list[dict[str, Any]], methods: tuple[str, ...] = METHODS
) -> dict[str, dict[str, float]]:
    rows_by_m: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    ms_by_m: dict[str, list[float]] = {m: [] for m in methods}
    for rec in records:
        n_svc = int(rec["n_services"])
        preds = rec["predictions"]
        for m in methods:
            ev = _eval_from_rank(int(preds[m]["rank"]), n_svc)
            rows_by_m[m].append(ev)
            ms_by_m[m].append(float(preds[m]["ms"]))
    return {
        m: {
            **summarize_ranking_results(rows_by_m[m]),
            "mean_ms": round(float(np.mean(ms_by_m[m])), 2) if ms_by_m[m] else 0.0,
            "median_ms": round(float(np.median(ms_by_m[m])), 2) if ms_by_m[m] else 0.0,
        }
        for m in methods
    }


def _macro_average(
    per_ds_summaries: dict[str, dict[str, dict[str, float]]],
    datasets: list[str],
    methods: tuple[str, ...] = METHODS,
) -> dict[str, dict[str, float]]:
    active_ds = [d for d in datasets if d in per_ds_summaries]
    if not active_ds:
        return {}
    keys = ("ac1", "ac3", "ac5", "avg5", "mrr", "mean_rank", "mean_ms", "median_ms")
    out: dict[str, dict[str, float]] = {}
    for m in methods:
        entry: dict[str, float] = {"n_datasets": float(len(active_ds))}
        for k in keys:
            entry[k] = round(float(np.mean([per_ds_summaries[d][m][k] for d in active_ds])), 4)
        out[m] = entry
    return out


def run_benchmark(
    *,
    datasets: tuple[str, ...] = ALL_DATASETS,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
) -> dict[str, Any]:
    meta = collect_system_metadata()
    with RawEngine(4, 1) as probe:
        simd_isa = probe.simd_isa

    specs, cases_sha = load_suite_specs(datasets=datasets)
    ensure_cases_cached(specs)

    cached_by_id: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if ckpt.get("lib_sha256") == meta["lib_sha256"]:
            for rec in ckpt.get("cases", []):
                cached_by_id[rec["case_id"]] = rec

    print(
        f"=== Full RCAEval RE1 + RE2 Suite Benchmark "
        f"(n={len(specs)}, datasets={list(datasets)}, ISA={simd_isa}, cached={len(cached_by_id)}) ===",
        flush=True,
    )

    t_suite0 = time.perf_counter()
    done_new = 0
    for idx, spec in enumerate(specs, start=1):
        if spec.case_id in cached_by_id:
            continue

        case = load_suite_case(spec)
        T, D = case.values.shape
        raw_tok = estimate_raw_tokens(case.values, case.columns)

        # 1. CausalSieve full dense Tier-1 (symmetric baseline/incident split)
        cs_svcs, cs_ms, dig_tok = _eval_case_causalsieve(case, blind=False)

        # 2. CausalSieve full dense Tier-1 blind (ZERO inject_time knowledge)
        csb_svcs, csb_ms, _ = _eval_case_causalsieve(case, blind=True)

        # 3. CausalSieve Tier-2 ScreenedSieve (blind, sub-10ms streaming)
        t2_svcs, t2_ms, t2_tok = _eval_case_tier2(case, active_m=32, lag=2)

        method_outputs: list[tuple[str, list[str], float]] = [
            ("causalsieve", cs_svcs, cs_ms),
            ("causalsieve_blind", csb_svcs, csb_ms),
            ("causalsieve_tier2", t2_svcs, t2_ms),
        ]

        baseline_specs: list[tuple[str, Any]] = [
            ("baro_fse24", lambda c=case: baro_fse24(c.values, c.inject_index, c.columns)),
            ("nsigma_rcaeval", lambda c=case: nsigma_rcaeval(c.values, c.inject_index, c.columns)),
            (
                "circa_structural_rca",
                lambda c=case: circa_structural_rca(c.values, c.inject_index, c.columns),
            ),
            ("rolling_mad_z", lambda c=case: rank_rolling_mad_z(c.values, c.inject_index)),
            (
                "pairwise_granger_z",
                lambda c=case: rank_pairwise_granger_z(c.values, c.inject_index, lag=2),
            ),
        ]

        for b_name, fn in baseline_specs:
            tb0 = time.perf_counter()
            scores, _ = fn()
            b_ms = (time.perf_counter() - tb0) * 1000.0
            b_svcs, _ = aggregate_channel_scores_to_services(scores, case.services)
            method_outputs.append((b_name, b_svcs, b_ms))

        case_entry: dict[str, Any] = {
            "case_id": spec.case_id,
            "dataset": spec.dataset,
            "suite": spec.suite,
            "system": spec.system,
            "fault": spec.fault,
            "repetition": spec.repetition,
            "true_service": spec.root_cause_service,
            "n_timesteps": int(T),
            "n_metrics": int(D),
            "n_services": int(case.n_unique_services),
            "nan_fraction": round(case.nan_fraction, 6),
            "inject_time": case.inject_time,
            "inject_index": case.inject_index,
            "parquet_sha256": case.parquet_sha256,
            "inject_sha256": case.inject_sha256,
            "parquet_bytes": case.parquet_bytes,
            "raw_tokens": raw_tok,
            "digest_tokens": dig_tok,
            "digest_tokens_tier2": t2_tok,
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
        done_new += 1

        if done_new % 25 == 0 or idx == len(specs) or (idx > 1 and specs[idx - 2].dataset != spec.dataset):
            p = case_entry["predictions"]
            el = time.perf_counter() - t_suite0
            print(
                f"  [{idx:3d}/{len(specs)}] {spec.dataset:6s} {spec.case_id:34s} (D={D:3d},T={T:4d}) | "
                f"CS={p['causalsieve']['rank']:2d} ({cs_ms:6.1f}ms) | "
                f"Blind={p['causalsieve_blind']['rank']:2d} | "
                f"BARO={p['baro_fse24']['rank']:2d} | "
                f"NSig={p['nsigma_rcaeval']['rank']:2d} | "
                f"CIRCA={p['circa_structural_rca']['rank']:2d} | "
                f"PWGr={p['pairwise_granger_z']['rank']:2d} | {el:5.1f}s",
                flush=True,
            )

        if checkpoint_path is not None and (done_new % checkpoint_every == 0):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            ordered_ckpt = [cached_by_id[s.case_id] for s in specs if s.case_id in cached_by_id]
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "lib_sha256": meta["lib_sha256"],
                        "cases": ordered_ckpt,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    case_records = [cached_by_id[s.case_id] for s in specs]

    # Groupings
    records_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_suite_fault: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in case_records:
        records_by_dataset[rec["dataset"]].append(rec)
        records_by_suite[rec["suite"]].append(rec)
        records_by_fault[rec["fault"]].append(rec)
        records_by_suite_fault[rec["suite"]][rec["fault"]].append(rec)

    overall_micro = _summarize_group(case_records)

    by_dataset_out: dict[str, Any] = {}
    per_ds_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for ds in datasets:
        recs = records_by_dataset.get(ds, [])
        if not recs:
            continue
        ds_sum = _summarize_group(recs)
        per_ds_summaries[ds] = ds_sum
        dims = [int(r["n_metrics"]) for r in recs]
        svcs = [int(r["n_services"]) for r in recs]
        ticks = [int(r["n_timesteps"]) for r in recs]
        raw_t = [int(r["raw_tokens"]) for r in recs]
        dig_t = [int(r["digest_tokens"]) for r in recs]
        by_dataset_out[ds] = {
            "n_cases": len(recs),
            "dataset_stats": {
                "n_metrics": {
                    "min": min(dims),
                    "median": float(np.median(dims)),
                    "mean": round(float(np.mean(dims)), 2),
                    "max": max(dims),
                },
                "n_services": {
                    "min": min(svcs),
                    "median": float(np.median(svcs)),
                    "mean": round(float(np.mean(svcs)), 2),
                    "max": max(svcs),
                },
                "n_timesteps": {
                    "min": min(ticks),
                    "median": float(np.median(ticks)),
                    "mean": round(float(np.mean(ticks)), 2),
                    "max": max(ticks),
                },
                "nan_cases": int(sum(1 for r in recs if r["nan_fraction"] > 0.0)),
            },
            "token_compression": {
                "mean_raw_tokens": round(float(np.mean(raw_t)), 1),
                "mean_digest_tokens": round(float(np.mean(dig_t)), 1),
                "reduction_pct": round(
                    100.0 * (1.0 - float(np.mean(dig_t)) / max(1.0, float(np.mean(raw_t)))), 3
                ),
            },
            "summary": ds_sum,
        }

    overall_macro = _macro_average(per_ds_summaries, list(datasets))

    by_suite_out: dict[str, Any] = {}
    for suite_name in ("RE1", "RE2"):
        recs = records_by_suite.get(suite_name, [])
        if not recs:
            continue
        suite_ds = [d for d in datasets if d.startswith(suite_name + "-")]
        by_suite_out[suite_name] = {
            "n_cases": len(recs),
            "datasets": suite_ds,
            "micro": _summarize_group(recs),
            "macro": _macro_average(per_ds_summaries, suite_ds),
            "by_fault": {
                f: _summarize_group(records_by_suite_fault[suite_name][f])
                for f in ALL_FAULTS
                if records_by_suite_fault[suite_name].get(f)
            },
        }

    by_fault_out = {
        f: _summarize_group(records_by_fault[f])
        for f in ALL_FAULTS
        if records_by_fault.get(f)
    }

    all_raw_tok = [int(r["raw_tokens"]) for r in case_records]
    all_dig_tok = [int(r["digest_tokens"]) for r in case_records]
    all_t2_tok = [int(r["digest_tokens_tier2"]) for r in case_records]

    token_compression = {
        "mean_raw_tokens": round(float(np.mean(all_raw_tok)), 1),
        "mean_digest_tokens": round(float(np.mean(all_dig_tok)), 1),
        "reduction_pct": round(
            100.0 * (1.0 - float(np.mean(all_dig_tok)) / max(1.0, float(np.mean(all_raw_tok)))), 3
        ),
        "mean_digest_tokens_tier2": round(float(np.mean(all_t2_tok)), 1),
        "reduction_tier2_pct": round(
            100.0 * (1.0 - float(np.mean(all_t2_tok)) / max(1.0, float(np.mean(all_raw_tok)))), 3
        ),
    }

    report: dict[str, Any] = {
        "benchmark": "rcaeval_full_re1_re2_suite",
        "dataset_source": "https://huggingface.co/datasets/phamquiluan/RCAEval",
        "protocol": {
            "zero_leakage": (
                "Full exhaustive evaluation across all canonical RE1 (375 cases) and RE2 (270 cases) "
                "real Kubernetes/OpenTelemetry incidents (total 645 cases) using frozen v0.1.0 defaults "
                "(lag=2, forgetting=0.995, normalize_scale=1), zero retuning, zero test-case peeking."
            ),
            "symmetric_comparators": (
                "Identical strictly causal imputation (ffill then 0.0, never bfill), identical "
                "<service>_<metric> parser, and symmetric max-over-metrics service aggregation across "
                "all methods. causalsieve_blind and causalsieve_tier2 receive ZERO inject_time knowledge; "
                "causalsieve, baro_fse24, nsigma_rcaeval, circa_structural_rca, rolling_mad_z, and "
                "pairwise_granger_z share the exact pre/post inject_index split."
            ),
            "datasets": list(datasets),
            "total_cases": len(case_records),
            "cases_parquet_sha256": cases_sha,
            "simd_isa": simd_isa,
            "lib_sha256": meta["lib_sha256"],
            "lib_scalar_sha256": meta["lib_scalar_sha256"],
        },
        "system": meta,
        "token_compression": token_compression,
        "overall_micro": overall_micro,
        "overall_macro": overall_macro,
        "by_suite": by_suite_out,
        "by_dataset": by_dataset_out,
        "by_fault": by_fault_out,
        "cases": case_records,
    }

    # Print clean summary tables
    print(f"\n=== OVERALL MICRO SUMMARY (n={len(case_records)} cases) ===", flush=True)
    for m in METHODS:
        s = overall_micro[m]
        print(
            f"  {m:22s} | AC@1={s['ac1']:.3f} | AC@3={s['ac3']:.3f} | AC@5={s['ac5']:.3f} | "
            f"Avg@5={s['avg5']:.3f} | MRR={s['mrr']:.3f} | {s['mean_ms']:7.1f} ms/case",
            flush=True,
        )

    print("\n=== OVERALL MACRO SUMMARY (equal weight across 6 datasets) ===", flush=True)
    for m in METHODS:
        s = overall_macro[m]
        print(
            f"  {m:22s} | AC@1={s['ac1']:.3f} | AC@3={s['ac3']:.3f} | AC@5={s['ac5']:.3f} | "
            f"Avg@5={s['avg5']:.3f} | MRR={s['mrr']:.3f} | {s['mean_ms']:7.1f} ms/case",
            flush=True,
        )

    print("\n=== PER-DATASET AC@1 / MRR (mean ms) ===", flush=True)
    for ds in datasets:
        if ds not in by_dataset_out:
            continue
        dsum = by_dataset_out[ds]["summary"]
        n_c = by_dataset_out[ds]["n_cases"]
        parts = [
            f"{m}={dsum[m]['ac1']:.3f}/{dsum[m]['mrr']:.3f}"
            for m in (
                "causalsieve",
                "causalsieve_blind",
                "baro_fse24",
                "nsigma_rcaeval",
                "circa_structural_rca",
                "rolling_mad_z",
                "pairwise_granger_z",
            )
        ]
        print(f"  {ds:6s} (n={n_c:3d}): " + " | ".join(parts), flush=True)

    print(
        f"\n  Token compression: {token_compression['mean_raw_tokens']:.1f} -> "
        f"{token_compression['mean_digest_tokens']:.1f} tokens "
        f"({token_compression['reduction_pct']:.3f}% reduction)",
        flush=True,
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"\nSaved full RCAEval suite results to benchmarks/results/{RESULTS_PATH.name}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full zero-leakage RCAEval RE1 + RE2 OpenTelemetry benchmark suite."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(ALL_DATASETS),
        default=list(ALL_DATASETS),
        help="Subset of datasets to evaluate (default: all 6 RE1+RE2 datasets).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional JSON checkpoint file for resumable execution (auto-invalidated on lib_sha256 change).",
    )
    args = parser.parse_args()
    run_benchmark(datasets=tuple(args.datasets), checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
