"""Symmetric, fair open-source comparators and evaluation metrics for CausalSieve benchmarks.

Amendment A10 Symmetric-Comparator Discipline:
- Every baseline receives the exact same strictly causal observations (never peeks past t).
- Every RCA baseline uses the exact same service aggregation rule and metric definitions.
- Every timing harness reports warm median, p95, throughput, and exact hardware metadata.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from scipy import linalg
from sklearn.ensemble import IsolationForest


def collect_system_metadata() -> dict[str, Any]:
    """Capture exact host hardware, OS, compiler, and package versions for reproducibility."""
    cpu_model = platform.processor() or "unknown"
    isa_flags: list[str] = []
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name") and cpu_model in ("unknown", "x86_64", ""):
                    cpu_model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not isa_flags:
                    flags = set(line.split(":", 1)[1].strip().split())
                    isa_flags = sorted(
                        fl for fl in ("sse4_2", "avx", "avx2", "fma", "avx512f") if fl in flags
                    )
    except OSError:
        pass

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "statsmodels", "sklearn", "pandas", "river", "pyarrow"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            versions[pkg] = "unavailable"
    try:
        import tigramite  # noqa: F401
        import importlib.metadata as ilm

        versions["tigramite"] = ilm.version("tigramite")
    except Exception:
        versions["tigramite"] = "unavailable"

    try:
        cxx = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    except Exception:
        cxx = "unknown"

    import hashlib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    def _sha256_or_none(p: Path) -> str | None:
        if not p.exists():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    lib_sha256 = _sha256_or_none(repo_root / "build" / "libcausalsieve.so")
    lib_scalar_sha256 = _sha256_or_none(repo_root / "build" / "libcausalsieve_scalar.so")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count_logical": int(os_cpu_count()),
        "simd_flags": isa_flags,
        "compiler": cxx,
        "lib_sha256": lib_sha256,
        "lib_scalar_sha256": lib_scalar_sha256,
        "versions": versions,
    }


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def eval_ranking(ranked_items: Sequence[Any], true_item: Any) -> dict[str, float]:
    """Compute standard RCA metrics: AC@1, AC@3, AC@5, Avg@5, MRR, and 1-indexed rank."""
    rank = len(ranked_items) + 1
    for idx, item in enumerate(ranked_items):
        if item == true_item:
            rank = idx + 1
            break
    ac1 = 1.0 if rank <= 1 else 0.0
    ac2 = 1.0 if rank <= 2 else 0.0
    ac3 = 1.0 if rank <= 3 else 0.0
    ac4 = 1.0 if rank <= 4 else 0.0
    ac5 = 1.0 if rank <= 5 else 0.0
    avg5 = (ac1 + ac2 + ac3 + ac4 + ac5) / 5.0
    mrr = 1.0 / float(rank) if rank <= len(ranked_items) else 0.0
    return {
        "rank": float(rank),
        "ac1": ac1,
        "ac3": ac3,
        "ac5": ac5,
        "avg5": avg5,
        "mrr": mrr,
    }


def summarize_ranking_results(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"n": 0, "ac1": 0.0, "ac3": 0.0, "ac5": 0.0, "avg5": 0.0, "mrr": 0.0}
    return {
        "n": len(rows),
        "ac1": float(np.mean([r["ac1"] for r in rows])),
        "ac3": float(np.mean([r["ac3"] for r in rows])),
        "ac5": float(np.mean([r["ac5"] for r in rows])),
        "avg5": float(np.mean([r["avg5"] for r in rows])),
        "mrr": float(np.mean([r["mrr"] for r in rows])),
        "mean_rank": float(np.mean([r["rank"] for r in rows])),
    }


def aggregate_channel_scores_to_services(
    channel_scores: np.ndarray,
    services: Sequence[str],
    *,
    onset_ticks: np.ndarray | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Aggregate per-metric anomaly/root scores to service level via max-over-metrics.

    Symmetric across all methods: each service receives the maximum score among its metrics
    (with deterministic tie-breaking by earliest onset tick if provided, then service name).
    """
    svc_best: dict[str, float] = {}
    svc_onset: dict[str, int] = {}
    for ch, svc in enumerate(services):
        sc = float(channel_scores[ch])
        if not np.isfinite(sc):
            sc = 0.0
        if svc not in svc_best or sc > svc_best[svc]:
            svc_best[svc] = sc
            if onset_ticks is not None:
                svc_onset[svc] = int(onset_ticks[ch])
    ranked = sorted(
        svc_best.keys(),
        key=lambda s: (-svc_best[s], svc_onset.get(s, 10**9), s),
    )
    return ranked, svc_best


def estimate_raw_tokens(X: np.ndarray, columns: Sequence[str] | None = None) -> int:
    """Transparent ~4 chars/token estimate for raw CSV telemetry representation."""
    T, D = X.shape
    header_len = sum(len(c) + 1 for c in columns) if columns else D * 6
    # Typical %.4g formatted CSV cell is ~7 chars including comma
    body_chars = T * D * 7
    return max(1, int((header_len + body_chars) / 4))


def estimate_text_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


# =============================================================================
# Exact Batch NumPy / SciPy Ridge-VAR(p) + Literal Reduced-Model Granger Refits
# =============================================================================


def numpy_batch_ridge_var_granger(
    X: np.ndarray,
    lag: int = 2,
    forgetting: float = 1.0,
    ridge: float = 1e-2,
    precision_ridge: float = 1e-2,
    include_intercept: bool = True,
    intercept_last: bool = True,
    anchor_shift: bool = True,
) -> dict[str, np.ndarray]:
    """Exact batch FP64 ridge-VAR(p) + literal reduced-model Granger refits + precision matrix.

    Used both as the clean mathematical reference baseline in speed/optimization benchmarks
    and as an independent NumPy/SciPy solver.
    """
    X = np.asarray(X, dtype=np.float64)
    T, D = X.shape
    if T <= lag + 2:
        raise ValueError(f"Need T > lag + 2, got T={T}, lag={lag}")

    Xs = X - X[0:1, :] if anchor_shift else X.copy()
    N = T - lag
    K = lag * D + (1 if include_intercept else 0)

    # Build design matrix Z (N x K) and targets Y (N x D)
    Z = np.empty((N, K), dtype=np.float64)
    col_offset = 0 if (not include_intercept or intercept_last) else 1
    if include_intercept and not intercept_last:
        Z[:, 0] = 1.0
    for ell in range(1, lag + 1):
        Z[:, col_offset + (ell - 1) * D : col_offset + ell * D] = Xs[lag - ell : T - ell, :]
    if include_intercept and intercept_last:
        Z[:, -1] = 1.0
    Y = Xs[lag:, :]

    # Exponential forgetting weights w_k = lambda^{N - 1 - k}
    if forgetting == 1.0:
        W = np.ones(N, dtype=np.float64)
    else:
        exponents = np.arange(N - 1, -1, -1, dtype=np.float64)
        W = np.power(forgetting, exponents)
    W_eff = float(np.sum(W))

    ZW = Z * W[:, None]
    G = Z.T @ ZW
    G_reg = G + ridge * np.eye(K, dtype=np.float64)
    C_zy = ZW.T @ Y

    c_fac, lower = linalg.cho_factor(G_reg, overwrite_a=False, check_finite=False)
    B = linalg.cho_solve((c_fac, lower), C_zy, check_finite=False)

    # Full-model ridge objective J_full[j] = sum_k w_k (y_{kj} - z_k^T b_j)^2 + ridge * ||b_j||^2
    resid = Y - Z @ B
    rss_full = np.sum(W[:, None] * (resid ** 2), axis=0)
    j_full = rss_full + ridge * np.sum(B ** 2, axis=0)

    # Literal reduced-model refits: for each source i, drop all lag columns belonging to i
    # and re-solve the reduced ridge regression from scratch!
    delta_j = np.zeros((D, D), dtype=np.float64)
    f_stat = np.zeros((D, D), dtype=np.float64)
    partial_r2 = np.zeros((D, D), dtype=np.float64)
    dof = max(1.0, W_eff - float(K))

    all_cols = np.arange(K)
    for src in range(D):
        drop_cols = [col_offset + ell * D + src for ell in range(lag)]
        keep_mask = np.ones(K, dtype=bool)
        keep_mask[drop_cols] = False
        keep_idx = all_cols[keep_mask]

        G_red = G_reg[np.ix_(keep_idx, keep_idx)]
        C_red = C_zy[keep_idx, :]
        cf_r, low_r = linalg.cho_factor(G_red, overwrite_a=False, check_finite=False)
        B_red = linalg.cho_solve((cf_r, low_r), C_red, check_finite=False)

        Z_red = Z[:, keep_idx]
        resid_red = Y - Z_red @ B_red
        j_red = np.sum(W[:, None] * (resid_red ** 2), axis=0) + ridge * np.sum(B_red ** 2, axis=0)

        dj = np.maximum(0.0, j_red - j_full)
        delta_j[src, :] = dj
        denom_f = np.maximum(j_full, 1e-30)
        f_stat[src, :] = (dj / float(lag)) / (denom_f / dof)
        partial_r2[src, :] = dj / np.maximum(j_full + dj, 1e-30)

    # Exact innovation precision matrix from posterior residuals
    cov_u = (resid.T @ (W[:, None] * resid)) / max(W_eff, 1.0) + precision_ridge * np.eye(D)
    theta = linalg.inv(cov_u)
    diag_t = np.sqrt(np.maximum(np.diag(theta), 1e-30))
    pcorr = -theta / np.outer(diag_t, diag_t)
    np.fill_diagonal(pcorr, 1.0)

    return {
        "B": B,
        "delta_j": delta_j,
        "f_stat": f_stat,
        "partial_r2": partial_r2,
        "theta": theta,
        "pcorr": pcorr,
        "residuals": resid,
        "W_eff": np.array([W_eff], dtype=np.float64),
    }


# =============================================================================
# Symmetric RCA Baselines
# =============================================================================


def _causal_baseline_stats(X: np.ndarray, baseline_end: int) -> tuple[np.ndarray, np.ndarray]:
    base = X[:baseline_end]
    mu = np.mean(base, axis=0)
    sd = np.std(base, axis=0)
    return mu, sd


def rank_raw_max_z(X: np.ndarray, baseline_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Classic univariate 3-sigma Z-score baseline (fit strictly on X[:baseline_end])."""
    X = np.asarray(X, dtype=np.float64)
    T, D = X.shape
    b_end = max(10, min(int(baseline_end), T - 1))
    mu, sd = _causal_baseline_stats(X, b_end)
    fault_seg = X[b_end:]
    scores = np.zeros(D, dtype=np.float64)
    for j in range(D):
        dev = np.max(np.abs(fault_seg[:, j] - mu[j]))
        if sd[j] > 1e-12:
            scores[j] = dev / sd[j]
        elif dev > 1e-12:
            # Constant-zero channel that suddenly spiked during fault
            scores[j] = 1e6 * dev
        else:
            scores[j] = 0.0
    ranked = np.argsort(-scores, kind="mergesort").astype(np.int32)
    return scores, ranked


def rank_rolling_mad_z(X: np.ndarray, baseline_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Baro-style robust median + 1.4826 * MAD anomaly scorer (fit strictly on X[:baseline_end])."""
    X = np.asarray(X, dtype=np.float64)
    T, D = X.shape
    b_end = max(10, min(int(baseline_end), T - 1))
    base = X[:b_end]
    fault_seg = X[b_end:]

    med = np.median(base, axis=0)
    mad = np.median(np.abs(base - med[None, :]), axis=0)
    iqr = np.percentile(base, 75, axis=0) - np.percentile(base, 25, axis=0)
    sd = np.std(base, axis=0)

    scores = np.zeros(D, dtype=np.float64)
    for j in range(D):
        scale = 1.4826 * mad[j]
        if scale <= 1e-12:
            scale = iqr[j] / 1.349
        if scale <= 1e-12:
            scale = sd[j]
        dev = np.max(np.abs(fault_seg[:, j] - med[j]))
        if scale > 1e-12:
            scores[j] = dev / scale
        elif dev > 1e-12:
            scores[j] = 1e6 * dev
        else:
            scores[j] = 0.0
    ranked = np.argsort(-scores, kind="mergesort").astype(np.int32)
    return scores, ranked


def rank_isolation_forest(
    X: np.ndarray, baseline_end: int, lag: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel lagged sklearn IsolationForest trained strictly on X[:baseline_end]."""
    X = np.asarray(X, dtype=np.float64)
    T, D = X.shape
    b_end = max(lag + 10, min(int(baseline_end), T - 1))
    mu, sd = _causal_baseline_stats(X, b_end)

    scores = np.zeros(D, dtype=np.float64)
    # Build lag window [x_j(t), x_j(t-1), ..., x_j(t-lag)] per channel
    for j in range(D):
        col = X[:, j]
        # Shift by baseline mean first so 1e10 offsets don't break float32 sklearn trees
        col_c = col - mu[j]
        if sd[j] > 1e-12:
            col_n = col_c / sd[j]
        else:
            if np.max(np.abs(col_c[b_end:])) > 1e-12:
                scores[j] = 10.0
            continue

        feats = np.column_stack([col_n[lag - ell : T - ell] for ell in range(lag + 1)])
        train_end = b_end - lag
        if train_end < 8 or train_end >= feats.shape[0]:
            continue
        X_tr = feats[:train_end]
        X_te = feats[train_end:]
        # Subsample training rows if very long to keep runtime reasonable & deterministic
        if X_tr.shape[0] > 400:
            step = max(1, X_tr.shape[0] // 400)
            X_tr = X_tr[::step]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso = IsolationForest(
                n_estimators=50,
                max_samples=min(128, X_tr.shape[0]),
                random_state=seed + j,
                n_jobs=1,
            )
            iso.fit(X_tr)
            # score_samples is higher for normal points, more negative for anomalies
            anom = -iso.score_samples(X_te)
            # Combine peak anomaly and top-5% tail severity
            scores[j] = float(np.percentile(anom, 98))

    ranked = np.argsort(-scores, kind="mergesort").astype(np.int32)
    return scores, ranked


def rank_pairwise_granger_z(
    X: np.ndarray, baseline_end: int, lag: int = 2, ridge: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise bivariate Granger outflow + robust peak Z baseline (strictly causal).

    Fits bivariate ridge ARX(p) for each pair (src -> dst) on standardized baseline ticks
    and weights each channel's fault anomaly score by its causal outflow ratio.
    """
    X = np.asarray(X, dtype=np.float64)
    T, D = X.shape
    b_end = max(lag + 15, min(int(baseline_end), T - 1))

    z_scores, _ = rank_rolling_mad_z(X, b_end)
    base = X[:b_end]
    mu = np.mean(base, axis=0)
    sd = np.std(base, axis=0)
    active = np.where(sd > 1e-12)[0]
    if len(active) < 2:
        ranked = np.argsort(-z_scores, kind="mergesort").astype(np.int32)
        return z_scores, ranked

    # Standardize active channels strictly on baseline window
    Xb = (base[:, active] - mu[active]) / sd[active]
    N = Xb.shape[0] - lag
    Da = len(active)
    # Precompute lag blocks per active channel: L[i] is (N, lag)
    lags = [
        np.column_stack([Xb[lag - ell : Xb.shape[0] - ell, i] for ell in range(1, lag + 1)])
        for i in range(Da)
    ]
    ones = np.ones((N, 1), dtype=np.float64)

    # Restricted AR(p) RSS for every active target dst
    rss_ar = np.empty(Da, dtype=np.float64)
    for dst in range(Da):
        Zr = np.hstack([lags[dst], ones])
        y = Xb[lag:, dst]
        G = Zr.T @ Zr + ridge * np.eye(lag + 1)
        b = np.linalg.solve(G, Zr.T @ y)
        e = y - Zr @ b
        rss_ar[dst] = float(np.dot(e, e)) + 1e-12

    # Pairwise Granger F-statistic matrix among active channels (cap to top-48 by z if huge D)
    if Da > 48:
        cand_local = np.argsort(-z_scores[active])[:48]
    else:
        cand_local = np.arange(Da)

    outflow = np.zeros(D, dtype=np.float64)
    inflow = np.zeros(D, dtype=np.float64)
    eye_full = ridge * np.eye(2 * lag + 1)
    for ii in cand_local:
        for jj in cand_local:
            if ii == jj:
                continue
            Zf = np.hstack([lags[jj], lags[ii], ones])
            y = Xb[lag:, jj]
            b = np.linalg.solve(Zf.T @ Zf + eye_full, Zf.T @ y)
            e = y - Zf @ b
            rss_f = float(np.dot(e, e)) + 1e-12
            if rss_ar[jj] > rss_f:
                f_val = ((rss_ar[jj] - rss_f) / lag) / (rss_f / max(1.0, N - 2 * lag - 1))
                if f_val > 3.0:
                    w = np.log1p(f_val)
                    outflow[active[ii]] += w
                    inflow[active[jj]] += w

    # Combine robust Z with causal net-outflow bonus (log-scaled so scale-invariant)
    net_flow = np.maximum(0.0, outflow - 0.5 * inflow)
    flow_mult = 1.0 + 0.5 * np.log1p(net_flow)
    combined = np.log1p(np.maximum(0.0, z_scores)) * flow_mult
    ranked = np.argsort(-combined, kind="mergesort").astype(np.int32)
    return combined, ranked
