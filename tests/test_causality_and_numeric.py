"""Strict causality, numerical torture, shock-shield, E-process, and drift-guard tests.

Covers Hypothesis H6 and core statistical/numerical invariants of the CausalSieve C99 core:
  1. Bit-exact strict causality (future corruption +-1e9 has 0-ULP effect on t <= probe).
  2. ERDDAP 1e10 DC-offset and 15-order-of-magnitude mixed-scale invariance.
  3. Atomic NaN / Inf rejection in both cs_push and cs_push_batch (zero state mutation).
  4. Rank-deficient collinearity & constant-zero-then-spike channel safety (Kubernetes error_rate).
  5. Huber Mahalanobis shock-shield immunity vs unshielded baseline corruption.
  6. Anytime-valid Ville E-process Type-I error calibration across 200 null streams x 1,000 ticks.
  7. 50,000-tick forgetting-factor self-healing Freivalds drift guard (< 1e-7 relative residual).
"""

from __future__ import annotations

import numpy as np
import pytest

from _cabi import RawEngine


def _simulate_stable_var(
    rng: np.random.Generator,
    T: int,
    D: int,
    lag: int = 2,
    chain_edges: list[tuple[int, int, float]] | None = None,
    noise_scale: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Generate a stationary multivariate VAR(p) process with optional planted causal edges."""
    A = np.zeros((lag, D, D), dtype=np.float64)
    # Mild stable self-persistence on lag 1 and lag 2
    for i in range(D):
        A[0, i, i] = 0.35 + 0.1 * ((i % 3) - 1)
        if lag >= 2:
            A[1, i, i] = -0.12
    if chain_edges:
        for src, dst, w in chain_edges:
            A[0, dst, src] = w
            if lag >= 2:
                A[1, dst, src] = 0.25 * w

    # Ensure strict spectral stability
    comp = np.zeros((lag * D, lag * D), dtype=np.float64)
    for ell in range(lag):
        comp[:D, ell * D : (ell + 1) * D] = A[ell]
    if lag > 1:
        comp[D:, : (lag - 1) * D] = np.eye((lag - 1) * D)
    rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
    if rho >= 0.85:
        scale = 0.80 / rho
        for ell in range(lag):
            A[ell] *= scale ** (ell + 1)

    burn = 100
    X = np.zeros((T + burn, D), dtype=np.float64)
    ns = np.broadcast_to(np.asarray(noise_scale, dtype=np.float64), (D,))
    eps = rng.standard_normal((T + burn, D)) * ns[None, :]
    for t in range(lag, T + burn):
        val = eps[t].copy()
        for ell in range(lag):
            val += A[ell] @ X[t - 1 - ell]
        X[t] = val
    return np.ascontiguousarray(X[burn:])


@pytest.mark.parametrize("normalize_scale", [0, 1])
@pytest.mark.parametrize("use_batch", [False, True])
def test_strict_assert_causal_bit_identical(normalize_scale: int, use_batch: bool) -> None:
    """Corrupting ALL future ticks t > probe with +-1e9 must leave t <= probe bit-identical."""
    rng = np.random.default_rng(202601)
    T, D, lag = 320, 10, 2
    X_clean = _simulate_stable_var(rng, T, D, lag=lag, chain_edges=[(0, 1, 0.5), (1, 2, 0.5)])

    for probe in (0, 1, 5, 40, 150, 260):
        X_corrupt = X_clean.copy()
        n_future = T - (probe + 1)
        signs = np.where((np.arange(n_future) % 2) == 0, 1.0, -1.0)[:, None]
        X_corrupt[probe + 1 :] = signs * 1e9

        with RawEngine(D, lag, normalize_scale=normalize_scale) as e_clean, RawEngine(
            D, lag, normalize_scale=normalize_scale
        ) as e_corrupt:
            if use_batch:
                rc1, r1, z1, u1 = e_clean.push_batch(X_clean, return_results=True, return_matrices=True)
                rc2, r2, z2, u2 = e_corrupt.push_batch(
                    X_corrupt, return_results=True, return_matrices=True
                )
                assert rc1 == 0 and rc2 == 0
                np.testing.assert_array_equal(z1[: probe + 1], z2[: probe + 1])
                np.testing.assert_array_equal(u1[: probe + 1], u2[: probe + 1])
                for t in range(probe + 1):
                    assert r1[t].tick == r2[t].tick
                    assert r1[t].mahalanobis_sq == r2[t].mahalanobis_sq
                    assert r1[t].e_value == r2[t].e_value
                    assert r1[t].shock_weight == r2[t].shock_weight
                    assert r1[t].max_exo_z == r2[t].max_exo_z
                    assert r1[t].top_root_channel == r2[t].top_root_channel
                    assert r1[t].alarm == r2[t].alarm
            else:
                for t in range(T):
                    rc1, r1, z1, u1 = e_clean.push(X_clean[t])
                    rc2, r2, z2, u2 = e_corrupt.push(X_corrupt[t])
                    assert rc1 == 0 and rc2 == 0
                    if t <= probe:
                        np.testing.assert_array_equal(z1, z2)
                        np.testing.assert_array_equal(u1, u2)
                        assert r1.mahalanobis_sq == r2.mahalanobis_sq
                        assert r1.e_value == r2.e_value
                        assert r1.shock_weight == r2.shock_weight
                        assert r1.max_exo_z == r2.max_exo_z
                        assert r1.top_root_channel == r2.top_root_channel
                        assert r1.alarm == r2.alarm


def test_erddap_1e10_dc_offset_and_mixed_scale_torture() -> None:
    """+1e10 DC offset and mixed 1e10..1e-5 channel scales preserve causal graph & root cause."""
    rng = np.random.default_rng(202602)
    T, D, lag = 550, 12, 2
    edges = [(0, 1, 0.65), (1, 2, 0.65), (2, 3, 0.65)]
    X_base = _simulate_stable_var(rng, T, D, lag=lag, chain_edges=edges)

    # Inject a clear root-cause shock on channel 0 starting at t=450 and propagating naturally
    X = X_base.copy()
    for t in range(450, T):
        X[t, 0] += 8.0
        if t >= 451:
            X[t, 1] += 0.65 * (X[t - 1, 0] - X_base[t - 1, 0])
        if t >= 452:
            X[t, 2] += 0.65 * (X[t - 1, 1] - X_base[t - 1, 1])
        if t >= 453:
            X[t, 3] += 0.65 * (X[t - 1, 2] - X_base[t - 1, 2])

    # Case A: uniform +1e10 DC offset across all channels
    X_dc = X + 1e10

    # Case B: mixed extreme DC offsets AND mixed scales spanning 15 orders of magnitude + a zero channel
    offsets = np.array(
        [1e10, -1e10, 5e9, 1e8, 0.0, 1e10, -3e9, 1e6, 0.0, 1e10, 1e4, 0.0], dtype=np.float64
    )
    scales = np.array(
        [1e5, 1e-5, 1e3, 1e-4, 1.0, 1e6, 1e-5, 1e2, 1e-3, 1.0, 1e4, 0.0], dtype=np.float64
    )
    X_mixed = X * scales[None, :] + offsets[None, :]

    with RawEngine(D, lag, normalize_scale=1, forgetting=0.995) as e_ref, RawEngine(
        D, lag, normalize_scale=1, forgetting=0.995
    ) as e_dc, RawEngine(D, lag, normalize_scale=1, forgetting=0.995) as e_mix:
        assert e_ref.push_batch(X, return_results=False)[0] == 0
        assert e_dc.push_batch(X_dc, return_results=False)[0] == 0
        assert e_mix.push_batch(X_mixed, return_results=False)[0] == 0

        f_ref, pr2_ref = e_ref.granger_matrix()
        f_dc, pr2_dc = e_dc.granger_matrix()
        f_mix, pr2_mix = e_mix.granger_matrix()

        # Uniform +1e10 DC offset must match baseline Granger discovery tightly
        # (float64 loses ~10 of 16 mantissa digits when adding 1e10 to O(1) values, leaving ~1e-6 precision)
        assert np.all(np.isfinite(f_dc))
        assert np.max(np.abs(pr2_dc - pr2_ref)) < 1e-4, (
            f"DC-offset partial R^2 diverged: {np.max(np.abs(pr2_dc - pr2_ref))}"
        )

        # Mixed scale + DC offset (channels 0..10 nonzero, channel 11 constant zero)
        assert np.all(np.isfinite(f_mix))
        assert np.all(np.isfinite(pr2_mix))
        for src, dst, _ in edges:
            assert pr2_dc[src, dst] > 0.10, f"DC-offset missed edge {src}->{dst}: {pr2_dc[src, dst]}"
            assert pr2_mix[src, dst] > 0.10, f"Mixed-scale missed edge {src}->{dst}: {pr2_mix[src, dst]}"

        # Root-cause localization must rank channel 0 #1 in ALL THREE regimes
        rca_ref = e_ref.root_attribution(lookback_ticks=120)
        rca_dc = e_dc.root_attribution(lookback_ticks=120)
        rca_mix = e_mix.root_attribution(lookback_ticks=120)

        assert int(rca_ref["ranked_channels"][0]) == 0
        assert int(rca_dc["ranked_channels"][0]) == 0
        assert int(rca_mix["ranked_channels"][0]) == 0


def test_nan_and_inf_atomic_rejection() -> None:
    """Single and batch pushes containing NaN or +-Inf must return -1 with ZERO state mutation."""
    rng = np.random.default_rng(202603)
    D, lag = 6, 2
    X_warm = _simulate_stable_var(rng, 120, D, lag=lag)
    X_after = _simulate_stable_var(rng, 25, D, lag=lag)

    with RawEngine(D, lag, normalize_scale=1) as eng, RawEngine(D, lag, normalize_scale=1) as twin:
        assert eng.push_batch(X_warm, return_results=False)[0] == 0
        assert twin.push_batch(X_warm, return_results=False)[0] == 0

        tick_before = eng.tick_count

        # 1. Single-tick NaN, +Inf, -Inf rejection (no state-refreshing queries before twin compare)
        for bad_val in (np.nan, np.inf, -np.inf):
            bad_row = X_after[0].copy()
            bad_row[3] = bad_val
            rc, _, _, _ = eng.push(bad_row)
            assert rc == -1, f"Expected -1 for bad_val={bad_val}, got {rc}"
            assert eng.tick_count == tick_before

        # 2. Batch all-or-nothing rejection when corruption is deep inside the batch
        for bad_row_idx, bad_val in ((0, np.nan), (12, np.inf), (19, -np.inf)):
            bad_batch = X_after[:20].copy()
            bad_batch[bad_row_idx, 4] = bad_val
            rc, _, _, _ = eng.push_batch(bad_batch, return_results=True, return_matrices=True)
            assert rc == -1, f"Expected batch -1 at row {bad_row_idx} val={bad_val}, got {rc}"
            assert eng.tick_count == tick_before

        # 3. Subsequent clean ticks must be BIT-IDENTICAL to untouched twin (proving ZERO hidden mutation!)
        rc1, r1, z1, u1 = eng.push_batch(X_after, return_results=True, return_matrices=True)
        rc2, r2, z2, u2 = twin.push_batch(X_after, return_results=True, return_matrices=True)
        assert rc1 == 0 and rc2 == 0
        np.testing.assert_array_equal(z1, z2)
        np.testing.assert_array_equal(u1, u2)
        for t in range(len(X_after)):
            assert r1[t].mahalanobis_sq == r2[t].mahalanobis_sq
            assert r1[t].e_value == r2[t].e_value

        B1, _ = eng.inspect_state()
        B2, _ = twin.inspect_state()
        f1, pr1 = eng.granger_matrix()
        f2, pr2 = twin.granger_matrix()
        th1, pc1 = eng.precision_matrix()
        th2, pc2 = twin.precision_matrix()
        np.testing.assert_array_equal(B1, B2)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(pr1, pr2)
        np.testing.assert_array_equal(th1, th2)
        np.testing.assert_array_equal(pc1, pc2)


def test_collinear_and_constant_zero_sudden_spike() -> None:
    """Duplicate channels and constant-zero channels that suddenly spike (error_rate) stay finite and detected."""
    rng = np.random.default_rng(202604)
    T, D, lag = 400, 10, 2
    X = _simulate_stable_var(rng, T, D, lag=lag, chain_edges=[(0, 1, 0.5)])

    # Exact duplicate collinearity: ch2 == ch1, ch3 == -2.0 * ch1 + 7.5
    X[:, 2] = X[:, 1]
    X[:, 3] = -2.0 * X[:, 1] + 7.5
    # Constant-zero channel (Kubernetes error_rate) that stays 0 until t=300 then spikes!
    X[:, 5] = 0.0
    X[300:340, 5] = 5.0
    # Permanently constant nonzero & permanently constant zero channels
    X[:, 6] = 42.0
    X[:, 7] = 0.0

    with RawEngine(D, lag, normalize_scale=1, forgetting=0.995) as eng:
        rc, res, exo_mat, innov_mat = eng.push_batch(X, return_results=True, return_matrices=True)
        assert rc == 0
        assert np.all(np.isfinite(exo_mat)), "Non-finite exo_z on collinear/constant channels"
        assert np.all(np.isfinite(innov_mat)), "Non-finite innovations on collinear/constant channels"

        maha = np.array([res[t].mahalanobis_sq for t in range(T)])
        evals = np.array([res[t].e_value for t in range(T)])
        assert np.all(np.isfinite(maha)) and np.all(maha >= 0.0)
        assert np.all(np.isfinite(evals)) and np.all(evals >= 0.0)

        B, freiv = eng.inspect_state()
        f_mat, pr2_mat = eng.granger_matrix()
        theta, pcorr = eng.precision_matrix()
        rca = eng.root_attribution(lookback_ticks=110)

        for name, arr in [
            ("B", B),
            ("F", f_mat),
            ("partial_r2", pr2_mat),
            ("theta", theta),
            ("pcorr", pcorr),
            ("root_scores", rca["root_scores"]),
        ]:
            assert np.all(np.isfinite(arr)), f"Non-finite values in {name}"
        assert np.isfinite(freiv) and freiv < 1e-6

        # Instant detection at t=300 when constant-zero channel 5 suddenly spikes!
        spike_exo = np.abs(exo_mat[300, 5])
        assert spike_exo > 4.0, f"Expected strong instant exo_z on sudden-spike channel 5, got {spike_exo}"
        assert int(rca["ranked_channels"][0]) == 5, (
            f"Expected sudden-spike channel 5 ranked #1, got {rca['ranked_channels'][:3]}"
        )


def test_shock_shield_immunity_vs_unshielded() -> None:
    """Huber shock shield prevents a 50-sigma 100-tick sustained outage from poisoning B_t and Theta_t."""
    rng = np.random.default_rng(202605)
    D, lag = 8, 2
    T_clean = 500
    T_outage = 100
    X_clean = _simulate_stable_var(rng, T_clean, D, lag=lag, chain_edges=[(0, 1, 0.55), (1, 2, 0.5)])

    # Sustained 50-sigma correlated outage on channels 0 and 1
    X_outage = _simulate_stable_var(rng, T_outage, D, lag=lag)
    t_grid = np.arange(T_outage, dtype=np.float64)
    X_outage[:, 0] += 50.0 + 10.0 * np.sin(0.3 * t_grid)
    X_outage[:, 1] -= 50.0 + 10.0 * np.cos(0.3 * t_grid)

    with RawEngine(
        D, lag, forgetting=0.99, shock_threshold=4.0, normalize_scale=0
    ) as shielded, RawEngine(
        D, lag, forgetting=0.99, shock_threshold=0.0, normalize_scale=0
    ) as unshielded:
        assert shielded.push_batch(X_clean, return_results=False)[0] == 0
        assert unshielded.push_batch(X_clean, return_results=False)[0] == 0

        B_base_s, _ = shielded.inspect_state()
        Th_base_s, _ = shielded.precision_matrix()
        B_base_u, _ = unshielded.inspect_state()
        Th_base_u, _ = unshielded.precision_matrix()

        # Feed the exact same 50-sigma sustained outage to both
        rc_s, res_s, _, _ = shielded.push_batch(X_outage, return_results=True)
        rc_u, res_u, _, _ = unshielded.push_batch(X_outage, return_results=True)
        assert rc_s == 0 and rc_u == 0

        B_post_s, _ = shielded.inspect_state()
        Th_post_s, _ = shielded.precision_matrix()
        B_post_u, _ = unshielded.inspect_state()
        Th_post_u, _ = unshielded.precision_matrix()

        drift_B_shielded = float(np.linalg.norm(B_post_s - B_base_s))
        drift_B_unshielded = float(np.linalg.norm(B_post_u - B_base_u))
        drift_Th_shielded = float(np.linalg.norm(Th_post_s - Th_base_s))
        drift_Th_unshielded = float(np.linalg.norm(Th_post_u - Th_base_u))

        # Shock shield must actively downweight outage ticks
        mean_w_shielded = float(np.mean([res_s[t].shock_weight for t in range(T_outage)]))
        mean_w_unshielded = float(np.mean([res_u[t].shock_weight for t in range(T_outage)]))
        assert mean_w_shielded < 0.25, f"Expected heavy shock downweighting, got mean w={mean_w_shielded}"
        assert mean_w_unshielded > 0.99

        # Shielded baseline B and Theta must suffer drastically less corruption than unshielded
        assert drift_B_shielded < 0.35 * drift_B_unshielded, (
            f"Shock shield failed to protect B: shielded={drift_B_shielded:.4f}, "
            f"unshielded={drift_B_unshielded:.4f}"
        )
        assert drift_Th_shielded < 0.35 * drift_Th_unshielded, (
            f"Shock shield failed to protect Theta: shielded={drift_Th_shielded:.4f}, "
            f"unshielded={drift_Th_unshielded:.4f}"
        )


def test_anytime_valid_ville_e_process_calibration() -> None:
    """Across 200 null Gaussian VAR streams of 1,000 ticks at e_alpha=0.05, FWER <= 0.05."""
    n_streams = 200
    T = 1000
    D = 6
    lag = 2
    alpha = 0.05
    threshold = 1.0 / alpha

    false_alarms = 0
    max_e_values: list[float] = []

    for seed in range(n_streams):
        rng = np.random.default_rng(10_000 + seed)
        X = _simulate_stable_var(rng, T, D, lag=lag, chain_edges=[(0, 1, 0.3), (2, 3, -0.25)])
        with RawEngine(D, lag, forgetting=0.995, e_alpha=alpha, normalize_scale=1) as eng:
            rc, res, _, _ = eng.push_batch(X, return_results=True)
            assert rc == 0
            e_vals = [res[t].e_value for t in range(T)]
            peak_e = max(e_vals)
            max_e_values.append(peak_e)
            if peak_e >= threshold:
                false_alarms += 1

    fwer = false_alarms / float(n_streams)
    assert fwer <= alpha, (
        f"Ville E-process FWER violated: {false_alarms}/{n_streams} = {fwer:.4f} > {alpha} "
        f"(median peak E={np.median(max_e_values):.3f}, max peak E={np.max(max_e_values):.3f})"
    )


def test_self_healing_drift_guard_50k_ticks() -> None:
    """50,000 ticks with forgetting=0.995 must maintain Freivalds inverse residual < 1e-7."""
    rng = np.random.default_rng(202607)
    T, D, lag = 50_000, 8, 2
    X = _simulate_stable_var(
        rng, T, D, lag=lag, chain_edges=[(0, 1, 0.55), (1, 2, -0.4), (3, 4, 0.5)]
    )
    # Add occasional benign regime variations to stress rank-1 updates
    X[15_000:15_020, 2] += 4.0
    X[35_000:35_020, 5] -= 4.0

    with RawEngine(D, lag, forgetting=0.995, normalize_scale=0, refresh_interval=0) as eng:
        rc, res, _, _ = eng.push_batch(X, return_results=True)
        assert rc == 0
        assert eng.tick_count == T

        B, freiv_err = eng.inspect_state()
        assert np.all(np.isfinite(B))
        assert np.isfinite(freiv_err)
        assert freiv_err < 1e-7, f"Freivalds relative inverse residual drifted to {freiv_err:.3e}"
        assert res[T - 1].refresh_count >= 1, "Expected drift guard to execute >= 1 Cholesky refresh"
