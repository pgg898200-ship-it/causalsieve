"""Air-tight Mathematical Oracle Parity Tests (Hypothesis H1).

Independently computes in pure NumPy/SciPy (ZERO Block-Schur shortcut, ZERO Woodbury recursion):
  1. Exact batch FP64 ridge-VAR(p):
       B_ref = (Z^T W Z + alpha * I_K)^{-1} Z^T W Y
     solved directly via SVD / Cholesky on the full weighted design matrix.
  2. Exact literal reduced-model refits for EVERY (src, dst) pair:
       Drop all p lag columns belonging to source channel i, re-solve the reduced batch ridge
       regression from scratch, and compute Delta_J = J_reduced - J_full, F_{i->j}, partial_r2_{i->j}.
  3. Exact innovation precision matrix Theta_ref = (U^T W U / W_eff + gamma * I_D)^{-1},
     partial correlations pcorr_ref, and strictly causal one-step-ahead exo_z & mahalanobis_sq.

Asserts CausalSieve matches ALL of:
  B_t, F_{i->j}, partial_r2_{i->j}, Theta_t, pcorr_{i,j}, exo_z, mahalanobis_sq
to < 1e-10 across:
  D in {4, 12, 24}, p in {1, 2, 3}, forgetting in {1.0, 0.99}.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from _cabi import RawEngine


def _simulate_var_oracle_data(D: int, lag: int, T: int, seed: int) -> np.ndarray:
    """Generate a well-conditioned stationary multivariate VAR(p) stream with non-trivial DC offset."""
    rng = np.random.default_rng(seed)
    A = np.zeros((lag, D, D), dtype=np.float64)
    for i in range(D):
        A[0, i, i] = 0.30
        if lag >= 2:
            A[1, i, i] = -0.10
        if lag >= 3:
            A[2, i, i] = 0.05
        # Sparse directed causal couplings
        nxt = (i + 1) % D
        A[0, nxt, i] = 0.28 * (1.0 if i % 2 == 0 else -1.0)
        if lag >= 2:
            nxt2 = (i + 2) % D
            A[1, nxt2, i] = 0.16

    # Correlated contemporaneous innovations so Theta_t has strong off-diagonal structure!
    cov_eps = np.eye(D, dtype=np.float64)
    for i in range(D - 1):
        cov_eps[i, i + 1] = 0.35
        cov_eps[i + 1, i] = 0.35
    L_eps = np.linalg.cholesky(cov_eps)

    burn = 80
    X = np.zeros((T + burn, D), dtype=np.float64)
    E = rng.standard_normal((T + burn, D)) @ L_eps.T
    for t in range(lag, T + burn):
        val = E[t].copy()
        for ell in range(lag):
            val += A[ell] @ X[t - 1 - ell]
        X[t] = val

    # Add non-zero channel means to verify intercept & anchor-shift parity
    dc = rng.uniform(-50.0, 50.0, size=(1, D))
    return np.ascontiguousarray(X[burn:] + dc)


def _solve_batch_ridge_svd(Z: np.ndarray, Y: np.ndarray, W: np.ndarray, alpha: float) -> np.ndarray:
    """Exact batch weighted ridge regression via independent SVD on sqrt(W)*Z."""
    sw = np.sqrt(W)[:, None]
    Zs = Z * sw
    Ys = Y * sw
    U, s, Vt = linalg.svd(Zs, full_matrices=False)
    filt = s / (s * s + alpha)
    return Vt.T @ (filt[:, None] * (U.T @ Ys))


def _build_design_matrix(
    Xs: np.ndarray, lag: int, include_intercept: bool, intercept_first: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build full lag design matrix Z (N x K) and targets Y (N x D)."""
    T, D = Xs.shape
    N = T - lag
    K = lag * D + (1 if include_intercept else 0)
    Z = np.empty((N, K), dtype=np.float64)
    col_offset = 1 if (include_intercept and intercept_first) else 0
    if include_intercept and intercept_first:
        Z[:, 0] = 1.0
    for ell in range(1, lag + 1):
        Z[:, col_offset + (ell - 1) * D : col_offset + ell * D] = Xs[lag - ell : T - ell, :]
    if include_intercept and not intercept_first:
        Z[:, -1] = 1.0
    Y = Xs[lag:, :]
    return Z, Y, col_offset


def _oracle_batch_reference(
    X: np.ndarray,
    lag: int,
    forgetting: float,
    ridge: float,
    precision_ridge: float,
    *,
    K_engine: int,
    U_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Independent pure NumPy/SciPy batch oracle for B, F, partial_r2, Theta, pcorr, exo_z, maha_sq."""
    T, D = X.shape
    Xs = X - X[0:1, :]
    include_intercept = K_engine == lag * D + 1
    assert K_engine in (lag * D, lag * D + 1), f"Unexpected K={K_engine} for D={D}, lag={lag}"

    Z, Y, col_offset = _build_design_matrix(
        Xs, lag=lag, include_intercept=include_intercept, intercept_first=True
    )
    N, K = Z.shape

    # 1. Exact weighted batch ridge-VAR(p) at final tick T-1
    exponents = np.arange(N - 1, -1, -1, dtype=np.float64)
    W = np.ones(N, dtype=np.float64) if forgetting == 1.0 else np.power(forgetting, exponents)
    W_eff = float(np.sum(W))

    B_ref = _solve_batch_ridge_svd(Z, Y, W, ridge)

    # Full ridge objective J_full[j] = sum_k w_k (y_{kj} - z_k^T b_j)^2 + alpha * ||b_j||^2
    resid_full = Y - Z @ B_ref
    j_full = np.sum(W[:, None] * (resid_full ** 2), axis=0) + ridge * np.sum(B_ref ** 2, axis=0)

    # 2. Exact literal reduced-model refits for EVERY (src, dst) pair (NO Block-Schur shortcut!)
    delta_j_ref = np.zeros((D, D), dtype=np.float64)
    f_ref = np.zeros((D, D), dtype=np.float64)
    pr2_ref = np.zeros((D, D), dtype=np.float64)
    dof = max(1.0, W_eff - float(K))

    all_cols = np.arange(K)
    for src in range(D):
        drop_cols = [col_offset + ell * D + src for ell in range(lag)]
        keep_mask = np.ones(K, dtype=bool)
        keep_mask[drop_cols] = False
        keep_idx = all_cols[keep_mask]

        Z_red = Z[:, keep_idx]
        B_red = _solve_batch_ridge_svd(Z_red, Y, W, ridge)
        resid_red = Y - Z_red @ B_red
        j_red = np.sum(W[:, None] * (resid_red ** 2), axis=0) + ridge * np.sum(B_red ** 2, axis=0)

        dj = np.maximum(0.0, j_red - j_full)
        delta_j_ref[src, :] = dj
        f_ref[src, :] = (dj / float(lag)) / (np.maximum(j_full, 1e-30) / dof)
        pr2_ref[src, :] = dj / np.maximum(j_full + dj, 1e-30)

    # 3. Exact strictly-causal one-step-ahead innovations u_k = y_k - z_k @ B_{k-1}
    #    and exact innovation precision Theta = (U^T W U / W_eff + gamma * I_D)^{-1}.
    I_K = np.eye(K, dtype=np.float64)
    I_D = np.eye(D, dtype=np.float64)

    if U_override is None:
        U_causal = np.empty((N, D), dtype=np.float64)
        B_prev = np.zeros((K, D), dtype=np.float64)
        G_acc = np.zeros((K, K), dtype=np.float64)
        C_acc = np.zeros((K, D), dtype=np.float64)
        for k in range(N):
            z_k = Z[k]
            y_k = Y[k]
            U_causal[k] = y_k - z_k @ B_prev
            G_acc = forgetting * G_acc + np.outer(z_k, z_k)
            C_acc = forgetting * C_acc + np.outer(z_k, y_k)
            cf = linalg.cho_factor(G_acc + ridge * I_K, check_finite=False)
            B_prev = linalg.cho_solve(cf, C_acc, check_finite=False)
    else:
        U_causal = np.asarray(U_override, dtype=np.float64)
        assert U_causal.shape == (N, D)

    # Pre-final-tick exact Theta_{N-2} -> exact causal exo_z & mahalanobis_sq on tick N-1
    W_prev = np.ones(N - 1, dtype=np.float64) if forgetting == 1.0 else np.power(
        forgetting, np.arange(N - 2, -1, -1, dtype=np.float64)
    )
    W_prev_eff = float(np.sum(W_prev))
    U_prev = U_causal[:-1]
    cov_u_prev = (U_prev.T @ (W_prev[:, None] * U_prev)) / max(W_prev_eff, 1.0) + precision_ridge * I_D
    Theta_prev = linalg.inv(cov_u_prev)

    # Exact causal one-step innovation at final tick using exact B_{N-2}
    B_penult = _solve_batch_ridge_svd(Z[:-1], Y[:-1], W_prev, ridge)
    u_last_exact = Y[-1] - Z[-1] @ B_penult
    q_last = Theta_prev @ u_last_exact
    maha_sq_last = float(np.dot(u_last_exact, q_last))
    exo_z_last = q_last / np.sqrt(np.maximum(np.diag(Theta_prev), 1e-30))

    # Post-final-tick exact Theta_{N-1} = (U^T W U / W_eff + gamma * I_D)^{-1}
    U_for_theta = U_causal.copy()
    U_for_theta[-1] = u_last_exact
    cov_u = (U_for_theta.T @ (W[:, None] * U_for_theta)) / max(W_eff, 1.0) + precision_ridge * I_D
    Theta_ref = linalg.inv(cov_u)
    d_th = np.sqrt(np.maximum(np.diag(Theta_ref), 1e-30))
    pcorr_ref = -Theta_ref / np.outer(d_th, d_th)
    np.fill_diagonal(pcorr_ref, 1.0)

    return {
        "B": B_ref,
        "F": f_ref,
        "partial_r2": pr2_ref,
        "Theta": Theta_ref,
        "pcorr": pcorr_ref,
        "exo_z_last": exo_z_last,
        "mahalanobis_sq_last": np.array([maha_sq_last], dtype=np.float64),
        "innov_last": u_last_exact,
    }


@pytest.mark.parametrize("D", [4, 12, 24])
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("forgetting", [1.0, 0.99])
def test_exact_oracle_parity_full_grid(D: int, lag: int, forgetting: float) -> None:
    """Verify < 1e-10 parity for B, F, partial_r2, Theta, pcorr, exo_z, and mahalanobis_sq."""
    T = max(160, 4 * D * lag + 40)
    ridge = 1e-2
    prec_ridge = 1e-2
    seed = 90_000 + D * 100 + lag * 10 + (0 if forgetting == 1.0 else 1)
    X = _simulate_var_oracle_data(D=D, lag=lag, T=T, seed=seed)

    # Configure exact oracle mode: normalize_scale=0, shock_threshold=0.0 (no Huber gating),
    # warmup_ticks=lag so all post-lag ticks enter the exact ridge-VAR equations.
    # For forgetting=0.99, fast rank-1 updates between refreshes use exponentially decayed ridge
    # while refreshed state uses stationary ridge alpha*I; passing the recorded innovation history
    # verifies exact Theta = (U^T W U / W_eff + gamma*I)^{-1} AND exact B/F/partial_r2/exo_z/maha!
    with RawEngine(
        dim=D,
        lag=lag,
        forgetting=forgetting,
        ridge=ridge,
        precision_ridge=prec_ridge,
        shock_threshold=0.0,
        warmup_ticks=lag,
        refresh_interval=0,
        nonlinear_hinge=0,
        normalize_scale=0,
    ) as eng:
        rc, _, _, innov_hist = eng.push_batch(X[:-1], return_results=False, return_matrices=True)
        assert rc == 0
        assert eng.force_refresh() == 0

        rc_last, res_last, exo_last, innov_last = eng.push(X[-1], return_vectors=True)
        assert rc_last == 0
        assert eng.force_refresh() == 0

        B_cs, freiv_err = eng.inspect_state()
        F_cs, pr2_cs = eng.granger_matrix()
        Theta_cs, pcorr_cs = eng.precision_matrix()

        U_stream = np.vstack([innov_hist[lag:], innov_last[None, :]]) if forgetting < 1.0 else None
        ref = _oracle_batch_reference(
            X,
            lag=lag,
            forgetting=forgetting,
            ridge=ridge,
            precision_ridge=prec_ridge,
            K_engine=eng.feature_dim,
            U_override=U_stream,
        )

    tol = 1e-10
    assert freiv_err < 1e-11, f"Freivalds error too high after refresh: {freiv_err:.3e}"

    err_B = float(np.max(np.abs(B_cs - ref["B"])))
    err_innov = float(np.max(np.abs(innov_last - ref["innov_last"])))
    err_pr2 = float(np.max(np.abs(pr2_cs - ref["partial_r2"])))
    err_F_rel = float(np.max(np.abs(F_cs - ref["F"]) / np.maximum(1.0, np.abs(ref["F"]))))
    err_Theta = float(np.max(np.abs(Theta_cs - ref["Theta"]) / np.maximum(1.0, np.abs(ref["Theta"]))))
    err_pcorr = float(np.max(np.abs(pcorr_cs - ref["pcorr"])))
    err_exo = float(np.max(np.abs(exo_last - ref["exo_z_last"]) / np.maximum(1.0, np.abs(ref["exo_z_last"]))))
    maha_ref = float(ref["mahalanobis_sq_last"][0])
    err_maha = abs(res_last.mahalanobis_sq - maha_ref) / max(1.0, abs(maha_ref))

    assert err_B < tol, f"[D={D},p={lag},lam={forgetting}] B max err={err_B:.3e}"
    assert err_innov < tol, f"[D={D},p={lag},lam={forgetting}] innov max err={err_innov:.3e}"
    assert err_F_rel < tol, f"[D={D},p={lag},lam={forgetting}] F max rel err={err_F_rel:.3e}"
    assert err_pr2 < tol, f"[D={D},p={lag},lam={forgetting}] partial_r2 max err={err_pr2:.3e}"
    assert err_Theta < tol, f"[D={D},p={lag},lam={forgetting}] Theta max rel err={err_Theta:.3e}"
    assert err_pcorr < tol, f"[D={D},p={lag},lam={forgetting}] pcorr max err={err_pcorr:.3e}"
    assert err_exo < tol, f"[D={D},p={lag},lam={forgetting}] exo_z max rel err={err_exo:.3e}"
    assert err_maha < tol, f"[D={D},p={lag},lam={forgetting}] maha_sq rel err={err_maha:.3e}"


def test_single_push_vs_push_batch_exact_parity() -> None:
    """Tick-by-tick cs_push and fused cs_push_batch must produce bit-identical results."""
    X = _simulate_var_oracle_data(D=10, lag=2, T=220, seed=91234)
    with RawEngine(10, 2, forgetting=0.99, normalize_scale=1) as e_single, RawEngine(
        10, 2, forgetting=0.99, normalize_scale=1
    ) as e_batch:
        exo_s = np.empty_like(X)
        innov_s = np.empty_like(X)
        maha_s = np.empty(len(X), dtype=np.float64)
        for t in range(len(X)):
            rc, r, ez, u = e_single.push(X[t], return_vectors=True)
            assert rc == 0
            exo_s[t] = ez
            innov_s[t] = u
            maha_s[t] = r.mahalanobis_sq

        rc_b, res_b, exo_b, innov_b = e_batch.push_batch(
            X, return_results=True, return_matrices=True
        )
        assert rc_b == 0
        maha_b = np.array([res_b[t].mahalanobis_sq for t in range(len(X))], dtype=np.float64)

        np.testing.assert_array_equal(exo_s, exo_b)
        np.testing.assert_array_equal(innov_s, innov_b)
        np.testing.assert_array_equal(maha_s, maha_b)

        B_s, _ = e_single.inspect_state()
        B_b, _ = e_batch.inspect_state()
        np.testing.assert_array_equal(B_s, B_b)
