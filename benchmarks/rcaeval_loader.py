"""Deterministic, zero-leakage loader for real Kubernetes failure cases from phamquiluan/RCAEval.

Amendment A8 Sealed-Holdout Protocol:
- Pre-registered canonical Latin-square selection across Online Boutique (RE1-OB) and
  Sock Shop (RE1-SS): 10 (system, service, fault) cells covering ALL 5 target services
  and ALL 5 fault types (cpu, mem, disk, delay, loss) in each system, across all 5
  independent fault-injection repetitions (total = 50 real Kubernetes microservice incidents).
- Deterministic split by repetition index BEFORE looking at data:
    * DEV split:            repetition == 1 (10 cases: 5 RE1-OB + 5 RE1-SS)
    * SEALED HOLDOUT split: repetition in {2, 3, 4, 5} (40 cases: 20 RE1-OB + 20 RE1-SS)
- Every downloaded parquet & inject_time.txt is cached locally under benchmarks/data/rcaeval/
  and recorded with its exact SHA-256 digest for bit-for-bit reproducibility.
- Strictly causal NaN handling (forward-fill then 0.0; NEVER backward-fill).
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "benchmarks" / "data" / "rcaeval"
HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"

# Canonical alphabetical Latin square: every target service and every fault type appears equally.
LATIN_SQUARE_CELLS: list[tuple[str, str, str]] = [
    # Online Boutique (RE1-OB): 5 services x 5 fault types
    ("re1ob", "adservice", "cpu"),
    ("re1ob", "cartservice", "mem"),
    ("re1ob", "checkoutservice", "disk"),
    ("re1ob", "currencyservice", "delay"),
    ("re1ob", "productcatalogservice", "loss"),
    # Sock Shop (RE1-SS): 5 services x 5 fault types
    ("re1ss", "carts", "cpu"),
    ("re1ss", "catalogue", "mem"),
    ("re1ss", "orders", "disk"),
    ("re1ss", "payment", "delay"),
    ("re1ss", "user", "loss"),
]

DEV_REPS = (1,)
HOLDOUT_REPS = (2, 3, 4, 5)


@dataclass(frozen=True)
class RCAEvalCaseSpec:
    case_id: str
    system: str
    root_cause_service: str
    fault: str
    repetition: int
    split: Literal["dev", "holdout"]


@dataclass
class RCAEvalCaseData:
    spec: RCAEvalCaseSpec
    timestamps: np.ndarray        # (T,) int64 unix seconds
    values: np.ndarray            # (T, D) float64 strictly causally imputed
    columns: list[str]            # length D metric names
    services: list[str]           # length D service names parsed as col.split("_")[0]
    inject_time: int
    inject_index: int             # first index where timestamps >= inject_time
    parquet_sha256: str
    inject_sha256: str
    parquet_bytes: int


def get_case_specs(split: Literal["dev", "holdout", "all"] = "all") -> list[RCAEvalCaseSpec]:
    specs: list[RCAEvalCaseSpec] = []
    reps: tuple[int, ...]
    if split == "dev":
        reps = DEV_REPS
    elif split == "holdout":
        reps = HOLDOUT_REPS
    else:
        reps = DEV_REPS + HOLDOUT_REPS

    for system, service, fault in LATIN_SQUARE_CELLS:
        for rep in reps:
            case_id = f"{system}_{service}_{fault}_{rep}"
            case_split: Literal["dev", "holdout"] = "dev" if rep in DEV_REPS else "holdout"
            specs.append(
                RCAEvalCaseSpec(
                    case_id=case_id,
                    system=system,
                    root_cause_service=service,
                    fault=fault,
                    repetition=rep,
                    split=case_split,
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
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    tmp_path.write_bytes(data)
    tmp_path.replace(local_path)
    return local_path


def parse_service_name(metric_col: str) -> str:
    """Deterministic service-name parser verified on DEV rep-1 schema (`<service>_<metric>`)."""
    return metric_col.split("_", 1)[0]


def load_case(spec: RCAEvalCaseSpec, cache_dir: Path = CACHE_DIR) -> RCAEvalCaseData:
    """Download (if needed), hash-verify, and strictly causally load a single RCAEval case."""
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

    # Strictly causal missing-value imputation: forward-fill only, then 0.0 for leading NaNs.
    val_df = df[metric_cols].astype(np.float64)
    val_df = val_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    values = np.ascontiguousarray(val_df.to_numpy(dtype=np.float64))

    services = [parse_service_name(c) for c in metric_cols]
    inject_index = int(np.searchsorted(timestamps, inject_time, side="left"))

    return RCAEvalCaseData(
        spec=spec,
        timestamps=timestamps,
        values=values,
        columns=metric_cols,
        services=services,
        inject_time=inject_time,
        inject_index=inject_index,
        parquet_sha256=parquet_sha,
        inject_sha256=inject_sha,
        parquet_bytes=parquet_path.stat().st_size,
    )
