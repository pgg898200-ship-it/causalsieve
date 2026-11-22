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


# =============================================================================
# Published RCAEval SOTA Baselines (Zero-Handicap Faithful Implementations)
# =============================================================================


def _deterministic_rank(
    scores: np.ndarray, col_names: Sequence[str] | None = None
) -> np.ndarray:
    """Rank channel indices by descending score with deterministic tie-breaking."""
    n_cols = len(scores)
    if col_names is not None and len(col_names) == n_cols:
        order = sorted(
            range(n_cols),
            key=lambda j: (
                -(float(scores[j]) if np.isfinite(scores[j]) else 0.0),
                str(col_names[j]),
                j,
            ),
        )
        return np.asarray(order, dtype=np.int32)
    clean = np.where(np.isfinite(scores), scores, 0.0)
    return np.argsort(-clean, kind="mergesort").astype(np.int32)


def baro_fse24(
    data: np.ndarray,
    inject_idx: int,
    col_names: Sequence[str] | None = None,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Faithful BARO (Pham et al., FSE 2024) robust median/IQR anomaly scorer.

    Strictly causal normal/anomalous split matching official RCAEval `baro`:
      - Normal window: `X[:b_end]`, anomalous window: `X[b_end:]`.
      - Constant-column protection matching official RCAEval `drop_constant` on
        both normal and anomalous windows: columns constant in either segment
        receive score 0.0 so zero-variance artifacts cannot dominate rankings.
      - Robust scaling: `median` and `IQR = Q75 - Q25` computed strictly on the
        normal window. For active non-constant columns whose normal IQR is zero
        (e.g., stepped memory or sparse latency metrics where Q25 == Q75 but
        std > 0), falls back to Gaussian-equivalent `1.349 * std` for strict
        unit invariance (bytes vs MB), bounded below by `eps`.
      - Channel score: peak absolute robust z-score on the anomalous window.
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"baro_fse24 expects a 2D array (T, D), got shape {X.shape}")
    T, D = X.shape
    b_end = max(10, min(int(inject_idx), T - 1))
    normal = X[:b_end]
    anomal = X[b_end:]

    med = np.median(normal, axis=0)
    q75 = np.percentile(normal, 75.0, axis=0)
    q25 = np.percentile(normal, 25.0, axis=0)
    iqr = q75 - q25
    sd = np.std(normal, axis=0)
    ptp_norm = np.ptp(normal, axis=0)
    ptp_anom = np.ptp(anomal, axis=0)

    scores = np.zeros(D, dtype=np.float64)
    for j in range(D):
        # Official BARO drop_constant(normal) intersect drop_constant(anomal)
        if not np.isfinite(ptp_norm[j]) or not np.isfinite(ptp_anom[j]):
            continue
        if ptp_norm[j] <= 1e-12 or ptp_anom[j] <= 1e-12:
            continue
        scale = float(iqr[j])
        if scale <= eps:
            # Non-constant normal column with degenerate quartiles: use scale-invariant std
            scale = max(1.349 * float(sd[j]), eps)
        z = np.abs(anomal[:, j] - med[j]) / scale
        peak = float(np.max(z))
        scores[j] = peak if np.isfinite(peak) else 0.0

    ranked = _deterministic_rank(scores, col_names)
    return scores, ranked


def nsigma_rcaeval(
    data: np.ndarray,
    inject_idx: int,
    col_names: Sequence[str] | None = None,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact RCAEval N-Sigma baseline (`mean/std` on normal window, peak |z| on anomalous).

    Matches official RCAEval `nsigma` constant-column protection (`drop_constant`
    on normal and anomalous segments) and population `StandardScaler` normalization
    fit strictly on `X[:inject_idx]`.
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"nsigma_rcaeval expects a 2D array (T, D), got shape {X.shape}")
    T, D = X.shape
    b_end = max(10, min(int(inject_idx), T - 1))
    normal = X[:b_end]
    anomal = X[b_end:]

    mu = np.mean(normal, axis=0)
    sd = np.std(normal, axis=0)
    ptp_norm = np.ptp(normal, axis=0)
    ptp_anom = np.ptp(anomal, axis=0)

    scores = np.zeros(D, dtype=np.float64)
    for j in range(D):
        if not np.isfinite(ptp_norm[j]) or not np.isfinite(ptp_anom[j]):
            continue
        if ptp_norm[j] <= eps or ptp_anom[j] <= eps or sd[j] <= eps:
            continue
        z = np.abs(anomal[:, j] - mu[j]) / sd[j]
        peak = float(np.max(z))
        scores[j] = peak if np.isfinite(peak) else 0.0

    ranked = _deterministic_rank(scores, col_names)
    return scores, ranked


def circa_structural_rca(
    data: np.ndarray,
    inject_idx: int,
    col_names: Sequence[str] | None = None,
    *,
    ridge: float = 0.1,
    group_by_service: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """CIRCA-style structural regression residual RCA (Li et al., KDD 2022).

    Fits conditional ridge linear regression of each metric on peer metrics
    strictly on the normal pre-fault window `X[:inject_idx]`, then scores each
    metric by the peak absolute standardized regression-residual shift on
    `X[inject_idx:]` (matching CIRCA's ANMRegressor / RHT residual z-score).

    Uses the exact block-precision-matrix identity:
      For regularized normal-window correlation matrix `G = (Z^T Z)/N + ridge * I`
      and `Omega = G^{-1}`, the exact ridge regression residuals of any block `B`
      on all peer columns `-B` equal `Z @ (Omega[:, B] @ Omega[B, B]^{-1})`.
    When `col_names` encodes `<service>_<metric>` and `group_by_service=True`,
    same-service sibling metrics (e.g., `latency-50` and `latency-90`) are grouped
    into a block so they are regressed on all cross-service peers without
    mutually cancelling each other's fault signal; when all columns have unique
    service prefixes (or `col_names is None`), each block has size 1 (exact
    nodewise peer regression).
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"circa_structural_rca expects a 2D array (T, D), got shape {X.shape}")
    T, D = X.shape
    b_end = max(10, min(int(inject_idx), T - 1))
    normal = X[:b_end]
    anomal = X[b_end:]

    mu = np.mean(normal, axis=0)
    sd = np.std(normal, axis=0)
    ptp_norm = np.ptp(normal, axis=0)
    ptp_anom = np.ptp(anomal, axis=0)

    active = np.where(
        np.isfinite(ptp_norm)
        & np.isfinite(ptp_anom)
        & (ptp_norm > eps)
        & (ptp_anom > eps)
        & (sd > eps)
    )[0]

    scores = np.zeros(D, dtype=np.float64)
    Da = len(active)
    if Da == 0:
        return scores, _deterministic_rank(scores, col_names)
    if Da == 1:
        j = int(active[0])
        scores[j] = float(np.max(np.abs(anomal[:, j] - mu[j]) / sd[j]))
        return scores, _deterministic_rank(scores, col_names)

    # Standardize active columns strictly on the normal window
    Z_norm = (normal[:, active] - mu[active]) / sd[active]
    Z_anom = (anomal[:, active] - mu[active]) / sd[active]
    N = float(Z_norm.shape[0])

    # Regularized normal-window Gram matrix and exact precision matrix Omega
    reg = max(float(ridge), 1e-6)
    G = (Z_norm.T @ Z_norm) / max(N, 1.0) + reg * np.eye(Da, dtype=np.float64)
    c_fac, lower = linalg.cho_factor(G, overwrite_a=False, check_finite=False)
    Omega = linalg.cho_solve((c_fac, lower), np.eye(Da, dtype=np.float64), check_finite=False)

    # Partition active columns into service blocks (or singletons)
    blocks: list[np.ndarray] = []
    if group_by_service and col_names is not None and len(col_names) == D:
        svc_to_local: dict[str, list[int]] = {}
        for local_i, orig_j in enumerate(active):
            svc = str(col_names[int(orig_j)]).split("_", 1)[0]
            svc_to_local.setdefault(svc, []).append(local_i)
        if len(svc_to_local) > 1:
            blocks = [np.asarray(idxs, dtype=np.intp) for idxs in svc_to_local.values()]
        else:
            blocks = [np.asarray([i], dtype=np.intp) for i in range(Da)]
    else:
        blocks = [np.asarray([i], dtype=np.intp) for i in range(Da)]

    # Build exact conditional-regression residual projection matrix W (Da x Da)
    W = np.empty((Da, Da), dtype=np.float64)
    for blk in blocks:
        if len(blk) == 1:
            idx = int(blk[0])
            diag_val = max(float(Omega[idx, idx]), 1e-12)
            W[:, idx] = Omega[:, idx] / diag_val
        elif len(blk) == Da:
            W[:, blk] = np.eye(Da, dtype=np.float64)[:, blk]
        else:
            Omega_bb = Omega[np.ix_(blk, blk)]
            Omega_all_b = Omega[:, blk]
            # Solve Omega_bb @ M^T = Omega_all_b^T -> W[:, blk] = Omega_all_b @ Omega_bb^{-1}
            try:
                W[:, blk] = linalg.solve(
                    Omega_bb, Omega_all_b.T, assume_a="pos", check_finite=False
                ).T
            except linalg.LinAlgError:
                W[:, blk] = (linalg.pinv(Omega_bb) @ Omega_all_b.T).T

    # Regression residuals strictly from normal-fitted coefficients
    err_norm = Z_norm @ W
    err_anom = Z_anom @ W

    # CIRCA RHT standardized residual shift score on anomalous window
    err_mu = np.mean(err_norm, axis=0)
    err_sd = np.std(err_norm, axis=0)
    safe_sd = np.maximum(err_sd, 1e-6)
    z_anom = np.abs((err_anom - err_mu[None, :]) / safe_sd[None, :])
    peak_scores = np.max(z_anom, axis=0)
    peak_scores = np.where(np.isfinite(peak_scores), peak_scores, 0.0)
    scores[active] = peak_scores

    ranked = _deterministic_rank(scores, col_names)
    return scores, ranked

