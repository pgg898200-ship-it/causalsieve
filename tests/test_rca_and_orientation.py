"""Causal root-cause attribution, non-Gaussian LiNGAM orientation, nonlinear hinge, counterfactual IRF, and Tier-2 screening tests.

Covers Hypotheses H3 and H4 across 5 distinct causal regimes:
  1. Lagged amplified cascade (0 -> 1 -> 2 -> 3) where downstream victim 3 has >= 2x raw Z-score
     (Raw Z fails; CausalSieve ranks channel 0 #1).
  2. Instantaneous non-Gaussian SEM (0 -> 1 -> 2) with (a) skewed Exponential innovations and
     (b) symmetric heavy-tailed Student-t(df=4) innovations: verifies Hyvarinen-Smith orientation
     R[0,1] > 0, R[1,2] > 0 AND precision-whitened exogenous shock isolation on channel 0.
  3. Nonlinear saturation cascade x_1(t) = 0.3 x_1(t-1) + 2.5 * max(0, x_0(t-1) - 1.0) + eps
     with nonlinear_hinge=1.
  4. Exact counterfactual do-intervention trajectory matching analytic VAR(p) IRF to < 1e-11.
  5. 512-channel Tier-2 ScreenedSieve firehose with 100 heteroscedastic 3x-noise loud decoys +
     hidden causal cascade (42 -> 107 -> 213).
"""

from __future__ import annotations

import numpy as np
import pytest

from _cabi import RawEngine, RawScreenedEngine


def _simulate_amplified_cascade(
    rng: np.random.Generator,
    T: int = 600,
    D: int = 16,
    shock_start: int = 480,
    shock_mag: float = 4.5,
) -> np.ndarray:
    """16-channel lagged chain 0 -> 1 -> 2 -> 3 where queue-like persistence amplifies raw Z on ch 3."""
    burn = 150
    X = np.zeros((T + burn, D), dtype=np.float64)
    eps = rng.standard_normal((T + burn, D))

    # Channel 0..3 noise stds: downstream has tight baseline noise + high self-persistence so
    # a steady step shock at ch 0 accumulates into a 3x-5x larger raw Z-score at ch 3!
    eps[:, 1] *= 0.35
    eps[:, 2] *= 0.35
    eps[:, 3] *= 0.30

    for t in range(2, T + burn):
        # Independent background channels 4..D-1
        for j in range(4, D):
            X[t, j] = 0.30 * X[t - 1, j] - 0.10 * X[t - 2, j] + eps[t, j]

        u0 = shock_mag if (t - burn) >= shock_start else 0.0
        X[t, 0] = 0.25 * X[t - 1, 0] + eps[t, 0] + u0
        X[t, 1] = 0.65 * X[t - 1, 1] + 0.65 * X[t - 1, 0] + eps[t, 1]
        X[t, 2] = 0.68 * X[t - 1, 2] + 0.65 * X[t - 1, 1] + eps[t, 2]
        X[t, 3] = 0.72 * X[t - 1, 3] + 0.70 * X[t - 1, 2] + eps[t, 3]

    return np.ascontiguousarray(X[burn:])


def test_lagged_amplified_cascade_rca() -> None:
    """CausalSieve ranks true root 0 #1 on an amplified cascade where Raw Z-score fails (picks victim 3)."""
    rng = np.random.default_rng(30301)
    T, D, shock_start = 600, 16, 480
    X = _simulate_amplified_cascade(rng, T=T, D=D, shock_start=shock_start, shock_mag=4.5)

    # Verify Raw Z-score truly fails on this amplified cascade (victim 3 has largest raw Z)
    base_mu = X[:shock_start].mean(axis=0)
    base_sd = X[:shock_start].std(axis=0)
    raw_peak_z = np.max(np.abs(X[shock_start:] - base_mu) / base_sd, axis=0)
    raw_top = int(np.argmax(raw_peak_z))
    assert raw_top != 0, f"Test fixture sanity check failed: raw Z picked {raw_top} (z={raw_peak_z[:4]})"
    assert raw_peak_z[3] > 2.0 * raw_peak_z[0], (
        f"Expected victim 3 raw Z > 2x root 0 raw Z, got z[0]={raw_peak_z[0]:.2f}, z[3]={raw_peak_z[3]:.2f}"
    )

    with RawEngine(D, lag=2, forgetting=0.995, normalize_scale=1, shock_threshold=4.0) as eng:
        rc, _, _, _ = eng.push_batch(X, return_results=True)
        assert rc == 0

        f_mat, pr2_mat = eng.granger_matrix()
        # Causal chain 0 -> 1 -> 2 -> 3 must be cleanly discovered
        assert pr2_mat[0, 1] > 0.15 and f_mat[0, 1] > f_mat[1, 0]
        assert pr2_mat[1, 2] > 0.15 and f_mat[1, 2] > f_mat[2, 1]
        assert pr2_mat[2, 3] > 0.15 and f_mat[2, 3] > f_mat[3, 2]

        rca = eng.root_attribution(lookback_ticks=T - shock_start + 10)
        assert int(rca["ranked_channels"][0]) == 0, (
            f"CausalSieve failed to rank root 0 #1: ranked={rca['ranked_channels'][:5]}, "
            f"scores={rca['root_scores'][:5]}, exo_z={rca['peak_exo_z'][:5]}"
        )


@pytest.mark.parametrize("dist_name", ["exponential", "student_t4"])
def test_instantaneous_nongaussian_lingam_orientation_and_whitening(dist_name: str) -> None:
    """Contemporaneous SEM 0 -> 1 -> 2 under skewed Exponential & symmetric heavy-tailed Student-t(4)."""
    rng = np.random.default_rng(30302 if dist_name == "exponential" else 30303)
    T, D = 2500, 6

    if dist_name == "exponential":
        # Zero-mean unit-variance skewed exponential innovations
        E = rng.exponential(scale=1.0, size=(T, D)) - 1.0
    else:
        # Zero-mean symmetric heavy-tailed Student-t(df=4) innovations normalized to unit variance
        E = rng.standard_t(df=4.0, size=(T, D)) / np.sqrt(2.0)

    # Contemporaneous SEM: x_0 = e_0, x_1 = 0.85 * x_0 + 0.55 * e_1, x_2 = 0.85 * x_1 + 0.55 * e_2
    X = np.zeros((T, D), dtype=np.float64)
    X[:, 0] = E[:, 0]
    X[:, 1] = 0.85 * X[:, 0] + 0.55 * E[:, 1]
    X[:, 2] = 0.85 * X[:, 1] + 0.55 * E[:, 2]
    X[:, 3:] = E[:, 3:]

    # Inject a pure instantaneous exogenous shock on channel 0 at the final tick
    # which propagates instantaneously to channels 1 and 2 on the SAME tick!
    shock = 10.0
    X_shock = X.copy()
    X_shock[-1, :] = 0.0
    X_shock[-1, 0] = shock
    X_shock[-1, 1] = 0.85 * shock
    X_shock[-1, 2] = 0.85 * 0.85 * shock

    with RawEngine(D, lag=1, forgetting=1.0, ridge=1e-2, precision_ridge=1e-2, normalize_scale=1) as eng:
        rc, _, exo_mat, _ = eng.push_batch(X_shock, return_results=True, return_matrices=True)
        assert rc == 0

        R = eng.contemporaneous_orientation()
        assert np.all(np.isfinite(R))
        # Must recover contemporaneous causal arrows 0 -> 1 and 1 -> 2
        assert R[0, 1] > 0.0, f"[{dist_name}] Expected R[0,1] > 0, got R[0,1]={R[0,1]:.6f}"
        assert R[1, 2] > 0.0, f"[{dist_name}] Expected R[1,2] > 0, got R[1,2]={R[1,2]:.6f}"
        assert R[1, 0] < 0.0 and R[2, 1] < 0.0, f"[{dist_name}] Expected skew-symmetric orientation"

        # Precision whitening must isolate the exogenous shock on channel 0 at the shock tick
        # (even though x_1 and x_2 jumped synchronously on the exact same tick!)
        last_exo = np.abs(exo_mat[-1])
        assert int(np.argmax(last_exo)) == 0, (
            f"[{dist_name}] Precision whitening failed to isolate root 0: |exo_z|={last_exo}"
        )
        assert last_exo[0] > 2.5 * last_exo[1] and last_exo[0] > 2.5 * last_exo[2], (
            f"[{dist_name}] Downstream victims not suppressed by whitening: |exo_z|={last_exo}"
        )


def test_nonlinear_saturation_cascade_hinge() -> None:
    """Threshold saturation cascade x_1(t) = 0.3 x_1(t-1) + 2.5 * max(0, x_0(t-1) - 1.0) + eps."""
    rng = np.random.default_rng(30304)
    T, D = 1200, 6
    burn = 100
    X = np.zeros((T + burn, D), dtype=np.float64)
    eps = rng.standard_normal((T + burn, D))

    for t in range(1, T + burn):
        X[t, 0] = 0.20 * X[t - 1, 0] + eps[t, 0]
        X[t, 1] = 0.30 * X[t - 1, 1] + 2.5 * max(0.0, X[t - 1, 0] - 1.0) + 0.5 * eps[t, 1]
        for j in range(2, D):
            X[t, j] = 0.25 * X[t - 1, j] + eps[t, j]
    X = np.ascontiguousarray(X[burn:])

    with RawEngine(
        D, lag=2, forgetting=1.0, ridge=1e-2, nonlinear_hinge=1, normalize_scale=1
    ) as eng_hinge, RawEngine(
        D, lag=2, forgetting=1.0, ridge=1e-2, nonlinear_hinge=0, normalize_scale=1
    ) as eng_lin:
        assert eng_hinge.push_batch(X, return_results=False)[0] == 0
        assert eng_lin.push_batch(X, return_results=False)[0] == 0

        f_h, pr2_h = eng_hinge.granger_matrix()
        f_l, pr2_l = eng_lin.granger_matrix()

        # Hinge basis expands feature dimension beyond linear VAR(p)
        assert eng_hinge.feature_dim > eng_lin.feature_dim

        # Must cleanly discover 0 -> 1 with strong directional asymmetry and higher explained variance
        assert pr2_h[0, 1] > 0.40, f"Hinge missed nonlinear edge 0->1: pr2={pr2_h[0, 1]:.4f}"
        assert f_h[0, 1] > 50.0 and f_h[0, 1] > 10.0 * f_h[1, 0], (
            f"Weak hinge F-stat: F[0,1]={f_h[0,1]:.2f}, F[1,0]={f_h[1,0]:.2f}"
        )
        assert pr2_h[0, 1] > pr2_l[0, 1] + 0.05, (
            f"Expected nonlinear_hinge to improve partial R^2 on threshold cascade: "
            f"hinge={pr2_h[0,1]:.4f} vs linear={pr2_l[0,1]:.4f}"
        )


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_counterfactual_do_matches_analytic_irf(lag: int) -> None:
    """counterfactual_do(0, +5.0, horizon=8) matches analytic VAR(p) IRF of B_t to < 1e-11."""
    rng = np.random.default_rng(30305 + lag)
    T, D, horizon = 350, 6, 8
    X = np.zeros((T, D), dtype=np.float64)
    eps = rng.standard_normal((T, D))
    for t in range(lag, T):
        X[t] = 0.35 * X[t - 1] + eps[t]
        X[t, 1] += 0.55 * X[t - 1, 0]
        X[t, 2] += 0.45 * X[t - 1, 1]
        if lag >= 2:
            X[t, 2] -= 0.25 * X[t - 2, 0]
        if lag >= 3:
            X[t, 3] += 0.30 * X[t - 3, 2]

    with RawEngine(D, lag=lag, forgetting=1.0, ridge=1e-2, normalize_scale=0, shock_threshold=0.0) as eng:
        assert eng.push_batch(X, return_results=False)[0] == 0
        B, _ = eng.inspect_state()
        traj = eng.counterfactual_do(clamp_channel=0, delta_value=5.0, horizon=horizon)

    # Extract lag weight blocks B_ell (D x D) from B (K x D):
    # Feature vector z_t = [1.0, x_{t-1}, ..., x_{t-p}] -> intercept at row 0, lags start at row 1.
    assert B.shape[0] == lag * D + 1, f"Unexpected B shape {B.shape} for D={D}, lag={lag}"

    def _compute_irf(offset: int) -> np.ndarray:
        irf = np.zeros((horizon, D), dtype=np.float64)
        irf[0, 0] = 5.0
        for h in range(1, horizon):
            acc = np.zeros(D, dtype=np.float64)
            for ell in range(1, min(lag, h) + 1):
                B_ell = B[offset + (ell - 1) * D : offset + ell * D, :]
                acc += irf[h - ell] @ B_ell
            irf[h] = acc
        return irf

    irf_ref = _compute_irf(1)
    err = float(np.max(np.abs(traj - irf_ref)))
    assert err < 1e-11, f"Counterfactual do-intervention IRF mismatch (lag={lag}): max_err={err:.3e}"


def test_tier2_screened_sieve_512ch_loud_decoy_firehose() -> None:
    """512-channel stream with 100 3x-noise loud decoys + hidden cascade 42 -> 107 -> 213."""
    rng = np.random.default_rng(30309)
    T, D = 600, 512
    shock_start = 480
    root, mid, leaf = 42, 107, 213

    # 100 heteroscedastic loud decoy channels with 3x higher baseline noise (and zero true fault!)
    noise_std = np.ones(D, dtype=np.float64)
    decoy_idx = np.arange(300, 400)
    noise_std[decoy_idx] = 3.5

    X = rng.standard_normal((T, D)) * noise_std[None, :]
    # Add stationary AR(1) background persistence
    for t in range(1, T):
        X[t] += 0.25 * X[t - 1]
    # Plant causal cascade 42 -> 107 -> 213
    X[:, root] = 0.0
    X[:, mid] = 0.0
    X[:, leaf] = 0.0
    e_r = rng.standard_normal(T)
    e_m = rng.standard_normal(T) * 0.4
    e_l = rng.standard_normal(T) * 0.35
    for t in range(1, T):
        u = 5.5 if t >= shock_start else 0.0
        X[t, root] = 0.25 * X[t - 1, root] + e_r[t] + u
        X[t, mid] = 0.65 * X[t - 1, mid] + 0.70 * X[t - 1, root] + e_m[t]
        X[t, leaf] = 0.70 * X[t - 1, leaf] + 0.75 * X[t - 1, mid] + e_l[t]
    X = np.ascontiguousarray(X)

    # Verify naive raw variance/max-deviation fails due to 3.5x loud decoys and amplified leaf 213
    naive_var_top3 = set(np.argsort(-np.var(X, axis=0))[:3].tolist())
    assert root not in naive_var_top3, f"Expected naive variance to fail, got {naive_var_top3}"
    base_mu = X[:shock_start].mean(axis=0)
    base_sd = X[:shock_start].std(axis=0)
    raw_z_top = int(np.argmax(np.max(np.abs(X[shock_start:] - base_mu) / base_sd, axis=0)))
    assert raw_z_top != root, f"Expected raw Z-score to pick amplified victim, got {raw_z_top}"

    with RawScreenedEngine(dim=D, active_m=24, lag=2, forgetting=0.995, ridge=1e-2) as sc:
        last_z, top_ch, top_sc = sc.push_batch(X, top_k=10)
        assert np.all(np.isfinite(last_z))
        assert np.all(np.isfinite(top_sc))

        top3 = top_ch[:3].tolist()
        assert root in top3, (
            f"Tier-2 ScreenedSieve failed to rank root {root} in Top-3: top10={top_ch.tolist()}, "
            f"scores={top_sc[:5].tolist()}"
        )
        active_ch, g_f, pcorr = sc.active_graph()
        assert len(active_ch) > 0
        assert root in active_ch.tolist(), f"Root {root} was not promoted to Tier-1 active set"
        assert np.all(np.isfinite(g_f)) and np.all(np.isfinite(pcorr))
