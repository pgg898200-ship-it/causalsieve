#ifndef CAUSALSIEVE_H
#define CAUSALSIEVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSALSIEVE_ABI_VERSION 1

typedef struct cs_engine cs_engine_t;
typedef struct cs_screened cs_screened_t;

typedef struct cs_config {
    uint32_t dim;                /* D: number of input channels (>= 1) */
    uint32_t lag;                /* p: VAR lag order (>= 1, <= 16) */
    double forgetting;           /* lambda in (0, 1], default 0.995 (1.0 = exact expanding) */
    double ridge;                /* alpha > 0, ridge on lagged Gram, default 1e-2 */
    double precision_ridge;      /* gamma > 0, ridge on innovation covariance, default 1e-2 */
    double shock_threshold;      /* Huber Mahalanobis gate in sqrt(D) units; 0 disables shield */
    double e_alpha;              /* Anytime-valid Ville false-alarm level, default 0.01 */
    uint32_t warmup_ticks;       /* Ticks before scoring/shock-gating; 0 = auto */
    uint32_t refresh_interval;   /* Freivalds drift check / Cholesky refresh cadence; 0 = auto */
    uint32_t nonlinear_hinge;    /* 0 = linear VAR(p); 1 = add asymmetric hinge max(0, x-1) basis */
    uint32_t raw_ring_capacity;  /* Per-channel lossless FP64 history capacity; 0 = 8192 */
    uint32_t normalize_scale;    /* 0 = anchor-shift only (exact oracle); 1 = causal baseline scale */
} cs_config_t;

typedef struct cs_tick_result {
    uint64_t tick;               /* 0-indexed arrival tick */
    double mahalanobis_sq;       /* u_t^T Theta_{t-1} u_t */
    double e_value;              /* Anytime-valid nonnegative supermartingale wealth E_t */
    double shock_weight;         /* Huber update weight w_t in [0, 1] */
    double max_exo_z;            /* max_j |z_exo[j]| */
    int32_t top_root_channel;    /* argmax_j |z_exo[j]| (-1 during warmup) */
    uint32_t alarm;              /* 1 if anytime-valid alarm or severe shock active */
    uint32_t refresh_count;      /* Cumulative drift-guard Cholesky refreshes */
} cs_tick_result_t;

/* Populate sensible production defaults for a D-channel stream. */
void cs_config_default(cs_config_t* cfg, uint32_t dim, uint32_t lag);

/* Lifecycle */
cs_engine_t* cs_create(const cs_config_t* cfg);
void cs_destroy(cs_engine_t* eng);
void cs_reset(cs_engine_t* eng);

/* Metadata */
const char* cs_simd_isa(void);
const char* cs_last_error(const cs_engine_t* eng);
uint64_t cs_tick_count(const cs_engine_t* eng);
uint32_t cs_dim(const cs_engine_t* eng);
uint32_t cs_lag(const cs_engine_t* eng);
uint32_t cs_feature_dim(const cs_engine_t* eng);

/*
 * Single-tick strictly causal ingest.
 * x: length D finite doubles. If any entry is NaN/Inf, returns -1 with ZERO state mutation.
 * out: optional tick summary (may be NULL).
 * out_exo_z: optional length D precision-whitened exogenous z-scores (may be NULL).
 * out_innov: optional length D one-step-ahead innovations u_t (may be NULL).
 */
int cs_push(
    cs_engine_t* eng,
    const double* x,
    cs_tick_result_t* out,
    double* out_exo_z,
    double* out_innov
);

/*
 * High-throughput batch ingest over n_ticks consecutive rows of length D (row-major).
 * Pre-validates all inputs before mutating state (all-or-nothing atomicity).
 * out_results: optional array of length n_ticks (may be NULL).
 * out_exo_z_matrix: optional n_ticks x D row-major array (may be NULL).
 * out_innov_matrix: optional n_ticks x D row-major array (may be NULL).
 */
int cs_push_batch(
    cs_engine_t* eng,
    const double* X,
    size_t n_ticks,
    cs_tick_result_t* out_results,
    double* out_exo_z_matrix,
    double* out_innov_matrix
);

/*
 * Exact Block-Schur Conditional Granger Causality across all D x D pairs (row-major [src * D + dst]).
 * Computes exact Delta_J = B_{S_i, j}^T P_{S_i, S_i}^{-1} B_{S_i, j} without refitting reduced models.
 * out_f_stat: optional D x D F-statistics (may be NULL).
 * out_partial_r2: optional D x D partial R^2 in [0, 1) (may be NULL).
 */
int cs_granger_matrix(
    cs_engine_t* eng,
    double* out_f_stat,
    double* out_partial_r2
);

/*
 * Exact online innovation precision matrix Theta_t (D x D) and partial correlations (D x D).
 * Diagonal of out_pcorr is 1.0; off-diagonal is -Theta_ij / sqrt(Theta_ii * Theta_jj).
 */
int cs_precision_matrix(
    cs_engine_t* eng,
    double* out_theta,
    double* out_pcorr
);

/*
 * Online Hyvarinen-Smith non-Gaussian contemporaneous causal orientation R_{i -> j} (D x D).
 * Positive R[i*D + j] indicates i -> j contemporaneous direction under skewed/heavy-tailed shocks.
 */
int cs_contemporaneous_orientation(
    cs_engine_t* eng,
    double* out_lingam_r
);

/*
 * Causal Root-Cause Attribution over the last `lookback_ticks` ticks (0 = since incident start or 64).
 * Combines precision-whitened exogenous shock intensity, onset precedence, and causal graph outflow.
 * All output arrays have length D (any may be NULL).
 * out_ranked_channels receives channel indices sorted from most likely root cause to least.
 */
int cs_root_attribution(
    cs_engine_t* eng,
    uint32_t lookback_ticks,
    double* out_root_scores,
    double* out_peak_exo_z,
    double* out_raw_z,
    int64_t* out_onset_tick,
    int32_t* out_ranked_channels
);

/*
 * Exact h-step counterfactual impulse/do-intervention propagation through current VAR(p) B_t.
 * Injects shock `delta_value` (in normalized units if normalize_scale=1, else raw units)
 * on `clamp_channel` at h=0 and propagates through lags 1..horizon-1.
 * out_delta_trajectory: horizon x D row-major array.
 */
int cs_counterfactual_do(
    cs_engine_t* eng,
    int32_t clamp_channel,
    double delta_value,
    uint32_t horizon,
    double* out_delta_trajectory
);

/*
 * Lossless exact FP64 slice recovery from the per-channel raw ring buffer.
 * Copies ticks in [start_tick, end_tick) currently retained in the ring into out_values.
 * out_values must have space for at least (end_tick - start_tick) doubles.
 * Sets *out_count to number of values written, *out_actual_start to first tick returned.
 */
int cs_get_raw_slice(
    cs_engine_t* eng,
    uint32_t channel,
    uint64_t start_tick,
    uint64_t end_tick,
    double* out_values,
    uint64_t* out_count,
    uint64_t* out_actual_start
);

/*
 * Exact L_inf-error-bounded Swing-Filter piecewise linear compression over retained ticks
 * in [start_tick, end_tick). Every linearly interpolated point is guaranteed within max_abs_err
 * of the exact FP64 raw value.
 */
int cs_compress_channel_pla(
    cs_engine_t* eng,
    uint32_t channel,
    uint64_t start_tick,
    uint64_t end_tick,
    double max_abs_err,
    uint64_t* out_knot_ticks,
    double* out_knot_vals,
    size_t max_knots,
    size_t* out_n_knots
);

/*
 * Inspect internal VAR weights B (K x D row-major) and verify Freivalds relative inverse residual
 * ||G_reg * (P * v) - v||_2 / ||v||_2 against exact sufficient statistics.
 */
int cs_inspect_state(
    cs_engine_t* eng,
    double* out_B,
    double* out_freivalds_rel_err
);

/*
 * Force an exact Cholesky refresh from sufficient statistics (useful for diagnostics/testing).
 */
int cs_force_refresh(cs_engine_t* eng);

/* =========================================================================
 * Tier-2 High-Dimensional Screened Causal Engine (D = 128 .. 16,384+)
 * O(D) AVX2 scale-free robust pre-whitening + Top-M promotion into Tier-1.
 * ========================================================================= */
cs_screened_t* cs_screened_create(
    uint32_t dim,
    uint32_t active_m,
    uint32_t lag,
    double forgetting,
    double ridge
);
void cs_screened_destroy(cs_screened_t* s);

int cs_screened_push_batch(
    cs_screened_t* s,
    const double* X,
    size_t n_ticks,
    double* out_last_z_scores,      /* optional length D scale-free pre-whitened z-scores at last tick */
    int32_t* out_top_k_channels,    /* optional length top_k ranked root-cause channel indices */
    double* out_top_k_scores,       /* optional length top_k root-cause scores */
    uint32_t top_k
);

int cs_screened_active_graph(
    cs_screened_t* s,
    int32_t* out_active_channels,   /* length active_m */
    double* out_granger_f,          /* active_m x active_m */
    double* out_pcorr,              /* active_m x active_m */
    uint32_t* out_active_count
);

#ifdef __cplusplus
}
#endif

#endif /* CAUSALSIEVE_H */
