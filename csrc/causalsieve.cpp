#include "causalsieve_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#if defined(CAUSALSIEVE_HAS_AVX2_TU) && !defined(CAUSALSIEVE_FORCE_SCALAR)
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__amd64__)
#include <cpuid.h>
#endif
#endif

using namespace causalsieve;

namespace causalsieve::kernels {

namespace {

#if !defined(CAUSALSIEVE_FORCE_SCALAR)
bool env_is_truthy(const char* raw) noexcept {
    if (!raw) return false;
    const char* p = raw;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    const char* end = p + std::strlen(p);
    while (end > p && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) --end;
    const size_t len = static_cast<size_t>(end - p);
    if (len == 1 && p[0] == '1') return true;
    auto eq_ci = [](const char* s, size_t n, const char* lit) noexcept -> bool {
        for (size_t i = 0; i < n; ++i) {
            char c = s[i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != lit[i]) return false;
        }
        return lit[n] == '\0';
    };
    if (len == 4 && eq_ci(p, 4, "true")) return true;
    if (len == 3 && eq_ci(p, 3, "yes")) return true;
    if (len == 2 && eq_ci(p, 2, "on")) return true;
    return false;
}

bool env_force_scalar() noexcept {
#if defined(_MSC_VER)
    char* val = nullptr;
    size_t len = 0;
    if (_dupenv_s(&val, &len, "CAUSALSIEVE_FORCE_SCALAR") != 0 || !val) {
        return false;
    }
    const bool res = env_is_truthy(val);
    std::free(val);
    return res;
#else
    return env_is_truthy(std::getenv("CAUSALSIEVE_FORCE_SCALAR"));
#endif
}
#endif

#if defined(CAUSALSIEVE_HAS_AVX2_TU) && !defined(CAUSALSIEVE_FORCE_SCALAR)
bool cpu_supports_avx2_fma() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    int info[4] = {0, 0, 0, 0};
    __cpuidex(info, 0, 0);
    if (info[0] < 7) return false;
    __cpuidex(info, 1, 0);
    const uint32_t ecx1 = static_cast<uint32_t>(info[2]);
    const bool has_osxsave = (ecx1 & (1u << 27)) != 0;
    const bool has_avx     = (ecx1 & (1u << 28)) != 0;
    const bool has_fma     = (ecx1 & (1u << 12)) != 0;
    if (!(has_osxsave && has_avx && has_fma)) return false;
    const uint64_t xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ULL) != 0x6ULL) return false;
    __cpuidex(info, 7, 0);
    const uint32_t ebx7 = static_cast<uint32_t>(info[1]);
    return (ebx7 & (1u << 5)) != 0;
#elif defined(__x86_64__) || defined(__amd64__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    const bool has_osxsave = (ecx & (1u << 27)) != 0;
    const bool has_avx     = (ecx & (1u << 28)) != 0;
    const bool has_fma     = (ecx & (1u << 12)) != 0;
    if (!(has_osxsave && has_avx && has_fma)) return false;
    uint32_t xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0u));
    (void)xcr0_hi;
    if ((xcr0_lo & 0x6u) != 0x6u) return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1u << 5)) != 0;
#else
    return false;
#endif
}
#endif

const KernelOps* resolve_active_ops() noexcept {
#if defined(CAUSALSIEVE_FORCE_SCALAR)
    return &get_scalar_ops();
#else
    if (env_force_scalar()) {
        return &get_scalar_ops();
    }
#if defined(CAUSALSIEVE_HAS_AVX2_TU)
    if (cpu_supports_avx2_fma()) {
        return &get_avx2_ops();
    }
    return &get_scalar_ops();
#elif defined(__aarch64__) || defined(_M_ARM64)
    return &get_neon_ops();
#else
    return &get_scalar_ops();
#endif
#endif
}

} /* namespace */

const KernelOps& active_ops() noexcept {
    static const KernelOps* const kActive = resolve_active_ops();
    return *kActive;
}

} /* namespace causalsieve::kernels */

namespace {

inline uint64_t splitmix64(uint64_t& x) noexcept {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline bool validate_finite(const double* ptr, size_t n) noexcept {
    return causalsieve::kernels::active_ops().all_finite_pd(ptr, n);
}

} /* namespace */

/* =========================================================================
 * cs_engine Member Implementations
 * ========================================================================= */
void cs_engine::init_buffers() {
    kops = &causalsieve::kernels::active_ops();
    D_pad = pad4(D);
    K_pad = pad4(K);

    anchor.resize(D_pad);
    welford_mean.resize(D_pad);
    welford_m2.resize(D_pad);
    mean0.resize(D_pad);
    base_var.resize(D_pad);
    inv_scale.resize(D_pad);
    floor_std.resize(D_pad);
    passive_mean.resize(D_pad);
    passive_m2.resize(D_pad);

    lag_hist.resize(static_cast<size_t>(p) * D_pad);

    P_tilde.resize(static_cast<size_t>(K) * K_pad);
    B.resize(static_cast<size_t>(K) * D_pad);
    G_tilde.resize(static_cast<size_t>(K) * K_pad);
    C_tilde.resize(static_cast<size_t>(K) * D_pad);
    S_yy_tilde.resize(D_pad);
    J_rec.resize(D_pad);
    J_exact.resize(D_pad);

    S_u_tilde.resize(static_cast<size_t>(D) * D_pad);
    S_u_diag.resize(D_pad);
    Theta_tilde.resize(static_cast<size_t>(D) * D_pad);

    u_mean.resize(D_pad);
    u_var.resize(D_pad);
    M_u2v_tilde.resize(static_cast<size_t>(D) * D_pad);
    M_utanhv_tilde.resize(static_cast<size_t>(D) * D_pad);
    M_uv_tilde.resize(static_cast<size_t>(D) * D_pad);

    var_z_slab.resize(kSlabBatchSize * K_pad);
    var_x_slab.resize(kSlabBatchSize * D_pad);
    var_w_slab.resize(kSlabBatchSize);
    su_u_slab.resize(kSlabBatchSize * D_pad);
    su_w_slab.resize(kSlabBatchSize);
    lin_s_slab.resize(kSlabBatchSize * D_pad);
    lin_s2_slab.resize(kSlabBatchSize * D_pad);
    lin_ts_slab.resize(kSlabBatchSize * D_pad);
    lin_w_slab.resize(kSlabBatchSize);

    raw_ring.resize(static_cast<size_t>(ring_cap) * D);
    exo_z_ring.resize(static_cast<size_t>(ring_cap) * D);
    raw_z_ring.resize(static_cast<size_t>(ring_cap) * D);
    uq_ring.resize(static_cast<size_t>(ring_cap) * D);
    innov_z_ring.resize(static_cast<size_t>(ring_cap) * D);

    dx_buf.resize(D_pad);
    x_work_buf.resize(D_pad);
    raw_z_buf.resize(D_pad);
    z_buf.resize(K_pad);
    y_hat_buf.resize(D_pad);
    u_buf.resize(D_pad);
    q_tilde_buf.resize(D_pad);
    q_buf.resize(D_pad);
    exo_z_buf.resize(D_pad);
    Pz_tilde_buf.resize(K_pad);
    g_buf.resize(K_pad);
    s_buf.resize(D_pad);
    s2_buf.resize(D_pad);
    ts_buf.resize(D_pad);

    chol_L_K.resize(static_cast<size_t>(K) * K_pad);
    chol_U_K.resize(static_cast<size_t>(K) * K_pad);
    chol_Y_KD.resize(static_cast<size_t>(K) * D_pad);
    chol_L_D.resize(static_cast<size_t>(D) * D_pad);
    chol_U_D.resize(static_cast<size_t>(D) * D_pad);
    probe_v.resize(K_pad);
    probe_Pv.resize(K_pad);
    probe_GPv.resize(K_pad);

    z_bonf = linalg::normal_two_sided_quantile(0.005 / static_cast<double>(D));
    z_inf_gate = linalg::normal_two_sided_quantile(0.0005 / static_cast<double>(std::max<uint32_t>(1u, K - 1u)));
}

void cs_engine::reset_state() {
    tick = 0;
    refresh_count = 0;
    ticks_since_refresh = 0;
    var_dirty_since_refresh = false;
    theta_dirty = false;
    baseline_frozen = false;
    has_refreshed_j = false;

    log_e_value = 0.0;
    ons_bet = 0.15;
    ons_A = 1.0;
    incident_start_tick = UINT64_MAX;
    consecutive_normal = 0;

    p_scale = 1.0;
    g_scale = 1.0;
    su_scale = 1.0;
    theta_scale = 1.0;
    lin_scale = 1.0;

    W_eff_tilde = 0.0;
    W_u_tilde = 0.0;
    lin_W_tilde = 0.0;
    n_var_updates = 0;
    n_u_updates = 0;
    lingam_warm_count = 0;
    welford_n = 0;
    passive_welford_n = 0;
    rng_state = 0x9e3779b97f4a7c15ULL;
    lag_head = 0;

    anchor.zero();
    welford_mean.zero();
    welford_m2.zero();
    mean0.zero();
    base_var.zero();
    inv_scale.zero();
    floor_std.zero();
    passive_mean.zero();
    passive_m2.zero();
    lag_hist.zero();

    P_tilde.zero();
    const double inv_ridge = 1.0 / cfg.ridge;
    for (uint32_t i = 0; i < K; ++i) {
        P_tilde[static_cast<size_t>(i) * K_pad + i] = inv_ridge;
    }

    B.zero();
    G_tilde.zero();
    C_tilde.zero();
    S_yy_tilde.zero();
    J_rec.zero();
    J_exact.zero();

    S_u_tilde.zero();
    S_u_diag.zero();
    Theta_tilde.zero();
    const double inv_pridge = 1.0 / cfg.precision_ridge;
    for (uint32_t j = 0; j < D; ++j) {
        Theta_tilde[static_cast<size_t>(j) * D_pad + j] = inv_pridge;
        inv_scale[j] = 1.0;
        base_var[j] = 1.0;
        floor_std[j] = 1e-6;
    }

    u_mean.zero();
    u_var.zero();
    M_u2v_tilde.zero();
    M_utanhv_tilde.zero();
    M_uv_tilde.zero();

    var_slab_len = 0;
    su_slab_len = 0;
    lin_slab_len = 0;
    var_z_slab.zero();
    var_x_slab.zero();
    var_w_slab.zero();
    su_u_slab.zero();
    su_w_slab.zero();
    lin_s_slab.zero();
    lin_s2_slab.zero();
    lin_ts_slab.zero();
    lin_w_slab.zero();

    raw_ring.zero();
    exo_z_ring.zero();
    raw_z_ring.zero();
    uq_ring.zero();
    innov_z_ring.zero();

    dx_buf.zero();
    x_work_buf.zero();
    raw_z_buf.zero();
    z_buf.zero();
    y_hat_buf.zero();
    u_buf.zero();
    q_tilde_buf.zero();
    q_buf.zero();
    exo_z_buf.zero();
    Pz_tilde_buf.zero();
    g_buf.zero();
    s_buf.zero();
    s2_buf.zero();
    ts_buf.zero();
    last_error.clear();
}

void cs_engine::push_lag_hist(const double* xw) noexcept {
    double* dst = lag_hist.data() + static_cast<size_t>(lag_head) * D_pad;
    if (cfg.normalize_scale == 1) {
        for (uint32_t j = 0; j < D; ++j) {
            const double v = xw[j];
            const double av = std::fabs(v);
            if (av <= 100.0) {
                dst[j] = v;
            } else {
                dst[j] = (v > 0.0 ? 1.0 : -1.0) * (100.0 + 20.0 * std::tanh((av - 100.0) * 0.02));
            }
        }
        for (size_t j = D; j < D_pad; ++j) dst[j] = 0.0;
    } else {
        std::memcpy(dst, xw, D_pad * sizeof(double));
    }
    lag_head = (lag_head + 1u) % p;
}

void cs_engine::build_feature_z(double* out_z) const noexcept {
    out_z[0] = 1.0;
    const uint32_t hinge_offset = 1u + p * D;
    for (uint32_t l = 0; l < p; ++l) {
        /* lag_head points to oldest slot; most recent is (lag_head + p - 1) % p */
        const uint32_t slot = (lag_head + p - 1u - l) % p;
        const double* src = lag_hist.data() + static_cast<size_t>(slot) * D_pad;
        double* dst_lin = out_z + 1u + static_cast<size_t>(l) * D;
        std::memcpy(dst_lin, src, D * sizeof(double));
        if (cfg.nonlinear_hinge) {
            double* dst_hinge = out_z + hinge_offset + static_cast<size_t>(l) * D;
            for (uint32_t c = 0; c < D; ++c) {
                const double v = src[c];
                dst_hinge[c] = (v > 1.0) ? (v - 1.0) : 0.0;
            }
        }
    }
    for (size_t i = K; i < K_pad; ++i) out_z[i] = 0.0;
}

void cs_engine::flush_var_slab() noexcept {
    if (var_slab_len == 0) return;
    kops->sym_rankk_update_upper(
        G_tilde.data(), var_z_slab.data(), var_w_slab.data(), var_slab_len, K, K_pad);
    kops->gemm_update_rect(
        C_tilde.data(), var_z_slab.data(), var_x_slab.data(), var_w_slab.data(), var_slab_len, K, K_pad, D_pad);
    var_slab_len = 0;
}

void cs_engine::flush_su_slab() noexcept {
    if (su_slab_len == 0) return;
    kops->sym_rankk_update_upper(
        S_u_tilde.data(), su_u_slab.data(), su_w_slab.data(), su_slab_len, D, D_pad);
    su_slab_len = 0;
}

void cs_engine::flush_lingam_slab() noexcept {
    if (lin_slab_len == 0) return;
    kops->lingam_batch_update(
        M_u2v_tilde.data(), M_utanhv_tilde.data(), M_uv_tilde.data(),
        lin_s_slab.data(), lin_s2_slab.data(), lin_ts_slab.data(),
        lin_w_slab.data(), lin_slab_len, D, D_pad);
    lin_slab_len = 0;
}

void cs_engine::flush_all_slabs() noexcept {
    flush_var_slab();
    flush_su_slab();
    flush_lingam_slab();
}

void cs_engine::rescale_var_stats() noexcept {
    flush_var_slab();
    if (g_scale != 1.0) {
        const double s = g_scale;
        for (uint32_t i = 0; i < K; ++i) {
            double* row = G_tilde.data() + static_cast<size_t>(i) * K_pad;
            for (size_t j = (i & ~size_t(3u)); j < K_pad; ++j) row[j] *= s;
        }
        const size_t kc = static_cast<size_t>(K) * D_pad;
        for (size_t i = 0; i < kc; ++i) C_tilde[i] *= s;
        for (uint32_t j = 0; j < D; ++j) S_yy_tilde[j] *= s;
        W_eff_tilde *= s;
        g_scale = 1.0;
    }
    if (p_scale != 1.0) {
        const double s = p_scale;
        for (uint32_t i = 0; i < K; ++i) {
            double* row = P_tilde.data() + static_cast<size_t>(i) * K_pad;
            for (size_t j = (i & ~size_t(3u)); j < K_pad; ++j) row[j] *= s;
        }
        p_scale = 1.0;
    }
}

void cs_engine::rescale_su_stats() noexcept {
    flush_su_slab();
    if (su_scale != 1.0) {
        const double s = su_scale;
        for (uint32_t i = 0; i < D; ++i) {
            double* row = S_u_tilde.data() + static_cast<size_t>(i) * D_pad;
            for (size_t j = (i & ~size_t(3u)); j < D_pad; ++j) row[j] *= s;
            S_u_diag[i] = row[i];
        }
        W_u_tilde *= s;
        su_scale = 1.0;
    } else {
        for (uint32_t i = 0; i < D; ++i) {
            S_u_diag[i] = S_u_tilde[static_cast<size_t>(i) * D_pad + i];
        }
    }
}

void cs_engine::rescale_lingam_stats() noexcept {
    flush_lingam_slab();
    if (lin_scale != 1.0) {
        const double s = lin_scale;
        const size_t dd = static_cast<size_t>(D) * D_pad;
        for (size_t i = 0; i < dd; ++i) {
            M_u2v_tilde[i] *= s;
            M_utanhv_tilde[i] *= s;
            M_uv_tilde[i] *= s;
        }
        lin_W_tilde *= s;
        lin_scale = 1.0;
    }
}

void cs_engine::update_lingam_moments(const double* u) noexcept {
    ++lingam_warm_count;
    const double alpha_u = (cfg.forgetting < 1.0)
        ? std::max(1.0 - cfg.forgetting, 1.0 / static_cast<double>(lingam_warm_count))
        : (1.0 / static_cast<double>(lingam_warm_count));

    for (uint32_t j = 0; j < D; ++j) {
        const double du = u[j] - u_mean[j];
        u_mean[j] += alpha_u * du;
        const double du2 = u[j] - u_mean[j];
        if (lingam_warm_count == 1) {
            u_var[j] = du * du;
        } else {
            u_var[j] = (1.0 - alpha_u) * u_var[j] + alpha_u * du * du2;
        }
    }

    if (lingam_warm_count < 10) return;

    lin_scale *= cfg.forgetting;
    if (lin_scale < 1e-100) {
        rescale_lingam_stats();
    }
    const double w_lin = 1.0 / lin_scale;
    lin_W_tilde += w_lin;

    const size_t row_off = lin_slab_len * D_pad;
    double* s_row  = lin_s_slab.data()  + row_off;
    double* s2_row = lin_s2_slab.data() + row_off;
    double* ts_row = lin_ts_slab.data() + row_off;
    for (uint32_t j = 0; j < D; ++j) {
        const double sd = std::sqrt(std::max(1e-18, u_var[j]));
        const double sv = std::clamp((u[j] - u_mean[j]) / sd, -8.0, 8.0);
        s_row[j]  = sv;
        s2_row[j] = sv * sv;
        ts_row[j] = std::tanh(sv);
    }
    for (size_t j = D; j < D_pad; ++j) {
        s_row[j]  = 0.0;
        s2_row[j] = 0.0;
        ts_row[j] = 0.0;
    }
    lin_w_slab[lin_slab_len] = w_lin;
    if (++lin_slab_len == kSlabBatchSize) {
        flush_lingam_slab();
    }
}

bool cs_engine::refresh_var_cholesky() noexcept {
    rescale_var_stats();

    const double r_var = eff_var_ridge();
    if (!kops->cholesky_factor_scaled_upper(
            G_tilde.data(), 1.0, r_var, chol_L_K.data(), K, K_pad)) {
        return false;
    }

    kops->cholesky_invert_to_upper(
        chol_L_K.data(), P_tilde.data(), chol_U_K.data(), K, K_pad);
    p_scale = 1.0;

    kops->cholesky_solve_multi(
        chol_L_K.data(), C_tilde.data(), 1.0, B.data(), chol_Y_KD.data(), K, D, K_pad, D_pad);

    for (uint32_t j = 0; j < D; ++j) {
        double y_sq = 0.0;
        for (uint32_t i = 0; i < K; ++i) {
            const double yij = chol_Y_KD[static_cast<size_t>(i) * D_pad + j];
            y_sq += yij * yij;
        }
        const double j_val = std::max(1e-30, S_yy_tilde[j] - y_sq);
        J_exact[j] = j_val;
        J_rec[j] = j_val;
    }

    has_refreshed_j = true;
    var_dirty_since_refresh = false;
    return true;
}

bool cs_engine::refresh_theta_cholesky() noexcept {
    rescale_su_stats();

    const double inv_w = (W_u_tilde > 1.0) ? (1.0 / W_u_tilde) : 1.0;
    const double r_prec = eff_precision_ridge();
    if (!kops->cholesky_factor_scaled_upper(
            S_u_tilde.data(), inv_w, r_prec, chol_L_D.data(), D, D_pad)) {
        return false;
    }

    kops->cholesky_invert_to_upper(
        chol_L_D.data(), Theta_tilde.data(), chol_U_D.data(), D, D_pad);
    theta_scale = 1.0;
    theta_dirty = false;
    return true;
}

double cs_engine::compute_freivalds_err() noexcept {
    flush_var_slab();

    for (uint32_t i = 0; i < K; ++i) {
        probe_v[i] = (splitmix64(rng_state) & 1ULL) ? 1.0 : -1.0;
    }
    for (size_t i = K; i < K_pad; ++i) probe_v[i] = 0.0;

    kops->sym_matvec_upper(P_tilde.data(), probe_v.data(), probe_Pv.data(), K, K_pad);
    kops->scale_vec(probe_Pv.data(), p_scale, probe_Pv.data(), K_pad);
    kops->sym_matvec_upper(G_tilde.data(), probe_Pv.data(), probe_GPv.data(), K, K_pad);

    const double r_var = eff_var_ridge();
    double num_sq = 0.0;
    for (uint32_t i = 0; i < K; ++i) {
        const double lhs = g_scale * probe_GPv[i] + r_var * probe_Pv[i];
        const double diff = lhs - probe_v[i];
        num_sq += diff * diff;
    }
    return std::sqrt(num_sq / static_cast<double>(K));
}

void cs_engine::finalize_warmup_replay() noexcept {
    for (uint32_t j = 0; j < D; ++j) {
        mean0[j] = welford_mean[j];
        const double v0 = (welford_n > 1)
            ? (welford_m2[j] / static_cast<double>(welford_n - 1))
            : 0.0;
        const double raw_std = std::sqrt(std::max(0.0, v0));
        const double const_eps = std::max(1e-15, 2.0 * std::numeric_limits<double>::epsilon() * std::fabs(anchor[j]));
        double s0 = 0.0;
        if (raw_std > const_eps) {
            /* Real nonzero variance during warmup (preserves ERDDAP 1e10 + 1e-5 mixed scale exactly) */
            s0 = raw_std;
            floor_std[j] = std::min(s0, std::max(1e-12, const_eps));
        } else {
            /* Truly flat constant-during-warmup metric (e.g. RSS memory or zero error_rate):
             * 1% relative scale prevents 4 KB memory-page blips from exploding to 1e10 sigma! */
            s0 = std::max(0.01, 0.01 * std::fabs(anchor[j]));
            floor_std[j] = s0;
        }
        base_var[j] = s0 * s0;
        inv_scale[j] = 1.0 / s0;
    }
    baseline_frozen = true;

    const uint64_t avail = std::min<uint64_t>(tick, ring_cap);
    const uint64_t start_s = tick - avail;
    const uint64_t end_s = tick;

    /* Pass 1: accumulate exact normalized sufficient statistics on strictly past warmup ticks */
    lag_hist.zero();
    lag_head = 0;
    uint64_t n_samples = 0;
    std::vector<double> diag_var_x(D, 0.0);
    for (uint64_t s = start_s; s < end_s; ++s) {
        const double* raw_s = raw_ring.data() + static_cast<size_t>(s % ring_cap) * D;
        for (uint32_t j = 0; j < D; ++j) {
            const double xw = (raw_s[j] - anchor[j] - mean0[j]) * inv_scale[j];
            x_work_buf[j] = xw;
            diag_var_x[j] += xw * xw;
        }
        for (size_t j = D; j < D_pad; ++j) x_work_buf[j] = 0.0;

        if (s - start_s >= p) {
            build_feature_z(z_buf.data());
            g_scale *= cfg.forgetting;
            if (g_scale < 1e-100) rescale_var_stats();
            const double w_tilde = 1.0 / g_scale;
            W_eff_tilde += w_tilde;
            ++n_samples;
            ++n_var_updates;
            std::memcpy(var_z_slab.data() + var_slab_len * K_pad, z_buf.data(), K_pad * sizeof(double));
            std::memcpy(var_x_slab.data() + var_slab_len * D_pad, x_work_buf.data(), D_pad * sizeof(double));
            var_w_slab[var_slab_len] = w_tilde;
            if (++var_slab_len == kSlabBatchSize) {
                flush_var_slab();
            }
            for (uint32_t j = 0; j < D; ++j) {
                S_yy_tilde[j] += w_tilde * x_work_buf[j] * x_work_buf[j];
            }
        }
        push_lag_hist(x_work_buf.data());
    }
    flush_var_slab();
    if (avail > 0) {
        const double inv_av = 1.0 / static_cast<double>(avail);
        for (uint32_t j = 0; j < D; ++j) {
            diag_var_x[j] = std::max(1e-2, diag_var_x[j] * inv_av);
        }
    } else {
        for (uint32_t j = 0; j < D; ++j) diag_var_x[j] = 1.0;
    }

    if (W_eff_tilde > 0.0) {
        refresh_var_cholesky();

        /* Degree-of-freedom shrinkage prevents S_u -> 0 when warmup n_samples <= K */
        const double dof_ratio = (n_samples > K)
            ? std::clamp(static_cast<double>(n_samples - K) / static_cast<double>(K), 0.0, 1.0)
            : 0.0;

        /* Pass 2: PRESS/LOO residuals + dof_ratio shrinkage for well-conditioned initial Theta */
        lag_hist.zero();
        lag_head = 0;
        for (uint64_t s = start_s; s < end_s; ++s) {
            const double* raw_s = raw_ring.data() + static_cast<size_t>(s % ring_cap) * D;
            for (uint32_t j = 0; j < D; ++j) {
                x_work_buf[j] = (raw_s[j] - anchor[j] - mean0[j]) * inv_scale[j];
            }
            for (size_t j = D; j < D_pad; ++j) x_work_buf[j] = 0.0;

            if (s - start_s >= p) {
                su_scale *= cfg.forgetting;
                if (su_scale < 1e-100) rescale_su_stats();
                const double wu_tilde = 1.0 / su_scale;
                W_u_tilde += wu_tilde;
                ++n_u_updates;

                if (dof_ratio > 0.0) {
                    build_feature_z(z_buf.data());
                    kops->matvec_BT_z(B.data(), z_buf.data(), y_hat_buf.data(), K, D, D_pad);
                    kops->sub_vec(x_work_buf.data(), y_hat_buf.data(), u_buf.data(), D_pad);

                    kops->sym_matvec_upper(P_tilde.data(), z_buf.data(), Pz_tilde_buf.data(), K, K_pad);
                    const double lev = std::clamp(p_scale * kops->dot_pd(z_buf.data(), Pz_tilde_buf.data(), K_pad), 0.0, 0.90);
                    const double inv_1mh = 1.0 / std::max(0.10, 1.0 - lev);
                    for (uint32_t j = 0; j < D; ++j) {
                        u_buf[j] = std::clamp(u_buf[j] * inv_1mh, -5.0, 5.0);
                    }
                    for (size_t j = D; j < D_pad; ++j) u_buf[j] = 0.0;

                    std::memcpy(su_u_slab.data() + su_slab_len * D_pad, u_buf.data(), D_pad * sizeof(double));
                    su_w_slab[su_slab_len] = wu_tilde;
                    if (++su_slab_len == kSlabBatchSize) {
                        flush_su_slab();
                    }
                    update_lingam_moments(u_buf.data());
                }
            }
            push_lag_hist(x_work_buf.data());
        }

        rescale_su_stats();
        if (dof_ratio < 1.0) {
            for (uint32_t i = 0; i < D; ++i) {
                double* row = S_u_tilde.data() + static_cast<size_t>(i) * D_pad;
                for (size_t j = (i & ~size_t(3u)); j < D_pad; ++j) {
                    row[j] *= dof_ratio;
                }
                row[i] += (1.0 - dof_ratio) * W_u_tilde * diag_var_x[i];
                S_u_diag[i] = row[i];
            }
        }
        refresh_theta_cholesky();
    }

    ticks_since_refresh = 0;
    var_dirty_since_refresh = false;
    theta_dirty = false;
}

int cs_engine::push_single_validated(const double* x,
                                     cs_tick_result_t* out,
                                     double* out_exo_z,
                                     double* out_innov) noexcept {
    const uint64_t t = tick;
    const size_t ring_slot = static_cast<size_t>(t % ring_cap);

    /* Immutable tick-0 shifted anchor */
    if (t == 0) {
        for (uint32_t j = 0; j < D; ++j) {
            anchor[j] = x[j];
            floor_std[j] = std::max(1e-6, 1e-13 * std::fabs(x[j]));
        }
    }

    /* Lossless exact raw storage */
    std::memcpy(raw_ring.data() + ring_slot * D, x, D * sizeof(double));

    for (uint32_t j = 0; j < D; ++j) {
        dx_buf[j] = x[j] - anchor[j];
    }
    for (size_t j = D; j < D_pad; ++j) dx_buf[j] = 0.0;

    /* Production mode warmup phase (accumulate Welford on dx; defer VAR until warmup_ticks) */
    if (cfg.normalize_scale == 1 && t < warmup_ticks) {
        ++welford_n;
        const double inv_n = 1.0 / static_cast<double>(welford_n);
        for (uint32_t j = 0; j < D; ++j) {
            const double d1 = dx_buf[j] - welford_mean[j];
            welford_mean[j] += d1 * inv_n;
            const double d2 = dx_buf[j] - welford_mean[j];
            welford_m2[j] += d1 * d2;
        }
        std::memset(exo_z_ring.data() + ring_slot * D, 0, D * sizeof(double));
        std::memset(raw_z_ring.data() + ring_slot * D, 0, D * sizeof(double));
        std::memset(uq_ring.data() + ring_slot * D, 0, D * sizeof(double));
        std::memset(innov_z_ring.data() + ring_slot * D, 0, D * sizeof(double));
        if (out_exo_z) std::memset(out_exo_z, 0, D * sizeof(double));
        if (out_innov) std::memset(out_innov, 0, D * sizeof(double));
        if (out) {
            out->tick = t;
            out->mahalanobis_sq = 0.0;
            out->e_value = 1.0;
            out->shock_weight = 1.0;
            out->max_exo_z = 0.0;
            out->top_root_channel = -1;
            out->alarm = 0;
            out->refresh_count = refresh_count;
        }
        ++tick;
        return 0;
    }

    /* Strictly causal warmup freeze + past-only replay at t == warmup_ticks */
    if (cfg.normalize_scale == 1 && !baseline_frozen) {
        finalize_warmup_replay();
    }

    /* Transform current tick using strictly t-1 baseline parameters */
    if (cfg.normalize_scale == 1) {
        for (uint32_t j = 0; j < D; ++j) {
            const double v = (dx_buf[j] - mean0[j]) * inv_scale[j];
            x_work_buf[j] = v;
            raw_z_buf[j] = v;
        }
    } else {
        for (uint32_t j = 0; j < D; ++j) {
            x_work_buf[j] = dx_buf[j];
            if (passive_welford_n >= 2) {
                const double pv = passive_m2[j] / static_cast<double>(passive_welford_n - 1);
                const double ps = std::max(std::sqrt(std::max(0.0, pv)), floor_std[j]);
                raw_z_buf[j] = (dx_buf[j] - passive_mean[j]) / ps;
            } else {
                raw_z_buf[j] = 0.0;
            }
        }
    }
    for (size_t j = D; j < D_pad; ++j) {
        x_work_buf[j] = 0.0;
        raw_z_buf[j] = 0.0;
    }

    /* Before we have p lagged observations, record history only */
    if (t < p) {
        push_lag_hist(x_work_buf.data());
        if (cfg.normalize_scale == 0) {
            ++passive_welford_n;
            const double inv_n = 1.0 / static_cast<double>(passive_welford_n);
            for (uint32_t j = 0; j < D; ++j) {
                const double d1 = dx_buf[j] - passive_mean[j];
                passive_mean[j] += d1 * inv_n;
                const double d2 = dx_buf[j] - passive_mean[j];
                passive_m2[j] += d1 * d2;
            }
        }
        std::memset(exo_z_ring.data() + ring_slot * D, 0, D * sizeof(double));
        std::memcpy(raw_z_ring.data() + ring_slot * D, raw_z_buf.data(), D * sizeof(double));
        std::memset(uq_ring.data() + ring_slot * D, 0, D * sizeof(double));
        std::memset(innov_z_ring.data() + ring_slot * D, 0, D * sizeof(double));
        if (out_exo_z) std::memset(out_exo_z, 0, D * sizeof(double));
        if (out_innov) std::memset(out_innov, 0, D * sizeof(double));
        if (out) {
            out->tick = t;
            out->mahalanobis_sq = 0.0;
            out->e_value = 1.0;
            out->shock_weight = 1.0;
            out->max_exo_z = 0.0;
            out->top_root_channel = -1;
            out->alarm = 0;
            out->refresh_count = refresh_count;
        }
        ++tick;
        return 0;
    }

    /* =====================================================================
     * STEP 1: Strictly Causal Scoring Using t-1 Parameters (B, Theta)
     * ===================================================================== */
    const auto& ops = *kops;
    build_feature_z(z_buf.data());

    /* y_hat = B_{t-1}^T * z_t, u_t = x_work_t - y_hat */
    ops.matvec_BT_z(B.data(), z_buf.data(), y_hat_buf.data(), K, D, D_pad);
    ops.sub_vec(x_work_buf.data(), y_hat_buf.data(), u_buf.data(), D_pad);
    for (size_t j = D; j < D_pad; ++j) u_buf[j] = 0.0;

    /* Precision whitening: q = Theta_{t-1} * u_t */
    ops.sym_matvec_upper(Theta_tilde.data(), u_buf.data(), q_tilde_buf.data(), D, D_pad);
    ops.scale_vec(q_tilde_buf.data(), theta_scale, q_buf.data(), D_pad);
    double mahal_sq = ops.dot_pd(u_buf.data(), q_buf.data(), D_pad);
    if (!std::isfinite(mahal_sq) || mahal_sq < 0.0) {
        mahal_sq = std::isfinite(mahal_sq) ? 0.0 : 1e300;
    }

    const double W_u_cur_pre = su_scale * W_u_tilde;
    const double inv_wu_pre = (W_u_cur_pre > 1.0) ? (1.0 / W_u_cur_pre) : 1.0;
    const double su_fac_pre = su_scale * inv_wu_pre;
    const double r_prec_pre = eff_precision_ridge();

    double max_exo_z = 0.0;
    double x_inf = 0.0;
    double u_inf = 0.0;
    int32_t arg_max = 0;
    double* uq_slot = uq_ring.data() + ring_slot * D;
    double* iz_slot = innov_z_ring.data() + ring_slot * D;
    for (uint32_t j = 0; j < D; ++j) {
        const double theta_jj = std::max(1e-30, theta_scale * Theta_tilde[static_cast<size_t>(j) * D_pad + j]);
        double ez = q_buf[j] / std::sqrt(theta_jj);
        if (!std::isfinite(ez)) ez = 0.0;
        exo_z_buf[j] = ez;
        double uqj = u_buf[j] * q_buf[j];
        if (!std::isfinite(uqj)) uqj = 0.0;
        uq_slot[j] = uqj;

        const double var_uj = std::max(1e-12, su_fac_pre * S_u_diag[j] + r_prec_pre);
        double izj = std::fabs(u_buf[j]) / std::sqrt(var_uj);
        if (!std::isfinite(izj)) izj = 0.0;
        iz_slot[j] = izj;

        const double aez = std::fabs(ez);
        if (aez > max_exo_z) {
            max_exo_z = aez;
            arg_max = static_cast<int32_t>(j);
        }
        const double ax = std::fabs(x_work_buf[j]);
        if (ax > x_inf) x_inf = ax;
        const double au = std::fabs(u_buf[j]);
        if (au > u_inf) u_inf = au;
    }

    std::memcpy(exo_z_ring.data() + ring_slot * D, exo_z_buf.data(), D * sizeof(double));
    std::memcpy(raw_z_ring.data() + ring_slot * D, raw_z_buf.data(), D * sizeof(double));
    if (out_exo_z) std::memcpy(out_exo_z, exo_z_buf.data(), D * sizeof(double));
    if (out_innov) std::memcpy(out_innov, u_buf.data(), D * sizeof(double));

    /* Precompute strictly t-1 leverage P_{t-1} * z_t (reused directly in Step 3a for zero extra flops!) */
    ops.sym_matvec_upper(P_tilde.data(), z_buf.data(), Pz_tilde_buf.data(), K, K_pad);
    const double zPz_tilde = ops.dot_pd(z_buf.data(), Pz_tilde_buf.data(), K_pad);
    const double zPz = std::max(0.0, p_scale * zPz_tilde);

    /* =====================================================================
     * STEP 2: Huber Shock Shield & Anytime-Valid ONS Betting E-Process
     * ===================================================================== */
    double w_t = 1.0;
    uint32_t alarm = 0;
    int32_t top_root = -1;
    double e_val = std::exp(log_e_value);
    const double lam = cfg.forgetting;

    if (t >= warmup_ticks) {
        top_root = arg_max;

        /* Finite-sample predictive leverage (1 + z^T P z):
         * Cap zPz at normal manifold mean K/W_eff once determined so sustained faults
         * in z_t cannot self-suppress their own anomaly score or defeat the shock shield. */
        const double W_eff_cur = g_scale * W_eff_tilde;
        const double W_u_cur = su_scale * W_u_tilde;
        const double mean_lev = static_cast<double>(K) / std::max(1.0, W_eff_cur);
        const double zPz_eff = (n_var_updates >= static_cast<uint64_t>(K))
            ? std::min(zPz, mean_lev)
            : zPz;
        const double pred_lev = 1.0 + zPz_eff;

        /* D-dim Hotelling-T^2 adjustment for mahalanobis_sq; 1D Student-t dof adjustment for exo_z */
        const double dof_u = std::max(static_cast<double>(D) + 4.0, W_u_cur - static_cast<double>(K));
        const double hotelling_fac = std::clamp((dof_u - static_cast<double>(D) - 1.0) / dof_u, 0.25, 1.0);
        const double m_sq_adj = (mahal_sq / pred_lev) * hotelling_fac;

        const double dof_1d = std::max(6.0, W_u_cur - static_cast<double>(K));
        const double t_scale_1d = std::sqrt((dof_1d - 2.0) / dof_1d);
        const double exo_z_adj = (max_exo_z / std::sqrt(pred_lev)) * t_scale_1d;

        /* Anytime-valid ONS betting supermartingale (only bets once baseline covariance is determined) */
        const double W_u_max = (lam < 1.0) ? (1.0 / (1.0 - lam)) : 1e30;
        const double w_u_gate = std::min(std::max(40.0, 2.0 * static_cast<double>(K)), 0.75 * W_u_max);
        const uint64_t n_u_gate = std::max<uint64_t>(40u, static_cast<uint64_t>(K));

        if (W_u_cur >= w_u_gate && n_u_updates >= n_u_gate) {
            const double df = static_cast<double>(D);
            const double c_wh = 2.0 / (9.0 * df);
            const double z_wh_adj = (std::cbrt(m_sq_adj / df) - (1.0 - c_wh)) / std::sqrt(c_wh);
            const double bonf_e = z_bonf + 0.50;
            const double ind = (z_wh_adj > 3.5 || exo_z_adj > bonf_e) ? 1.0 : 0.0;
            const double g_t = (ind - 0.01) / 0.99;

            const double wealth_mult = std::max(1e-12, 1.0 + ons_bet * g_t);
            log_e_value = std::clamp(log_e_value + std::log(wealth_mult), -690.0, 690.0);
            e_val = std::exp(log_e_value);

            /* Update predictable betting fraction AFTER scoring tick t */
            const double z_grad = -g_t / wealth_mult;
            ons_A += z_grad * z_grad;
            constexpr double kOnsStep = 2.0 / (2.0 - 1.0986122886681096914);
            ons_bet = std::clamp(ons_bet - kOnsStep * (z_grad / ons_A), 0.0, 0.75);
        }

        const double sev_thresh = std::max(4.5, cfg.shock_threshold * 1.5);
        if (e_val >= (1.0 / cfg.e_alpha) || exo_z_adj > sev_thresh) {
            alarm = 1;
            if (incident_start_tick == UINT64_MAX) {
                incident_start_tick = t;
            }
            consecutive_normal = 0;
        } else if (incident_start_tick != UINT64_MAX) {
            if (++consecutive_normal >= 64) {
                incident_start_tick = UINT64_MAX;
                consecutive_normal = 0;
            }
        }

        /* Mallows-Huber bounded-influence shock shield with underdetermined anti-deadlock guard */
        if (cfg.shock_threshold > 0.0) {
            const double m_ratio = std::sqrt(m_sq_adj / std::max(1.0, static_cast<double>(D)));
            const double r_mahal = m_ratio / cfg.shock_threshold;
            const double r_exo = exo_z_adj / z_bonf;
            const double r = std::max(r_mahal, r_exo);
            const uint64_t det_min = static_cast<uint64_t>(
                std::max<double>(static_cast<double>(warmup_ticks), 1.5 * static_cast<double>(K)));

            /* Coordinate-wise Mallows regressor gate: if lagged state z_t itself already contains
             * a >z_inf_gate fault outlier, never let RLS fit baseline B/Theta on corrupted regressors! */
            double z_inf = 0.0;
            if (cfg.normalize_scale == 1 && n_var_updates >= det_min) {
                const uint32_t lin_end = 1u + p * D;
                for (uint32_t i = 1; i < lin_end; ++i) {
                    const double az = std::fabs(z_buf[i]);
                    if (az > z_inf) z_inf = az;
                }
            }

            if (n_var_updates < det_min) {
                /* Underdetermined warmup anti-deadlock: never lock out learning before K params are determined */
                w_t = (r <= 1.0) ? 1.0 : std::max(0.25, 1.0 / (r * r));
            } else if (r > 1.15 || z_inf > z_inf_gate || (e_val >= (1.0 / cfg.e_alpha) && r > 0.80)) {
                /* Hard quarantine during severe shocks, corrupted fault regressors, or Ville-confirmed incidents */
                w_t = 0.0;
            } else if (r <= 1.0) {
                w_t = 1.0;
            } else {
                const double inv_r2 = 1.0 / (r * r);
                w_t = inv_r2 * inv_r2;
            }
        }

        /* Hard Sherman-Morrison Overflow Guard (production normalize_scale == 1 only):
         * Unconditionally overrides early 0.25 floor so catastrophic spikes never enter P/G/S_u/Theta! */
        if (cfg.normalize_scale == 1 && (x_inf > 25.0 || u_inf > 25.0 || mahal_sq > 1e5)) {
            w_t = 0.0;
        }
    }

    /* =====================================================================
     * STEP 3: Strictly Post-Scoring State Updates (P, B, G, C, S_u, Theta)
     * ===================================================================== */
    if (w_t > 0.0) {
        /* 3a. Shared Inverse Gram P_t & VAR weights B_t rank-1 Sherman-Morrison (reuses Pz_tilde & zPz) */
        const double denom = lam + w_t * zPz;

        if (denom > 1e-18 && std::isfinite(denom)) {
            const double gain_coef = (w_t * p_scale) / denom;
            ops.scale_vec(Pz_tilde_buf.data(), gain_coef, g_buf.data(), K_pad);

            /* B += g * u^T */
            ops.outer_update_rect(B.data(), g_buf.data(), u_buf.data(), 1.0, K, D_pad);

            /* P_tilde -= gain_coef * Pz_tilde * Pz_tilde^T; p_scale /= lam */
            ops.sym_rank1_update_upper(P_tilde.data(), Pz_tilde_buf.data(), -gain_coef, K, K_pad);
            p_scale /= lam;

            /* Strictly nonnegative recursive ridge objective RSS_j */
            const double rss_coef = (w_t * lam) / denom;
            for (uint32_t j = 0; j < D; ++j) {
                J_rec[j] = lam * J_rec[j] + rss_coef * u_buf[j] * u_buf[j];
            }
        }

        /* 3b. Exact exponentially-weighted sufficient statistics (deferred L1 micro-batch slab) */
        g_scale *= lam;
        if (g_scale < 1e-100 || p_scale > 1e100) {
            rescale_var_stats();
        }
        const double w_tilde = w_t / g_scale;
        W_eff_tilde += w_tilde;
        ++n_var_updates;
        std::memcpy(var_z_slab.data() + var_slab_len * K_pad, z_buf.data(), K_pad * sizeof(double));
        std::memcpy(var_x_slab.data() + var_slab_len * D_pad, x_work_buf.data(), D_pad * sizeof(double));
        var_w_slab[var_slab_len] = w_tilde;
        if (++var_slab_len == kSlabBatchSize) {
            flush_var_slab();
        }
        for (uint32_t j = 0; j < D; ++j) {
            S_yy_tilde[j] += w_tilde * x_work_buf[j] * x_work_buf[j];
        }
        var_dirty_since_refresh = true;

        /* Self-heal P immediately if any diagonal becomes non-finite or non-positive */
        bool need_p_heal = !std::isfinite(p_scale);
        if (!need_p_heal) {
            for (uint32_t i = 0; i < K; ++i) {
                const double pii = P_tilde[static_cast<size_t>(i) * K_pad + i];
                if (!std::isfinite(pii) || pii <= 1e-15) {
                    need_p_heal = true;
                    break;
                }
            }
        }
        if (need_p_heal) {
            refresh_var_cholesky();
            ++refresh_count;
        }

        /* 3c. Innovation covariance S_u (deferred slab + live diagonal) & precision Theta rank-1 update */
        const double W_u_old = su_scale * W_u_tilde;
        su_scale *= lam;
        if (su_scale < 1e-100) {
            rescale_su_stats();
        }
        const double wu_tilde = w_t / su_scale;
        W_u_tilde += wu_tilde;
        ++n_u_updates;
        for (uint32_t j = 0; j < D; ++j) {
            S_u_diag[j] += wu_tilde * u_buf[j] * u_buf[j];
        }
        std::memcpy(su_u_slab.data() + su_slab_len * D_pad, u_buf.data(), D_pad * sizeof(double));
        su_w_slab[su_slab_len] = wu_tilde;
        if (++su_slab_len == kSlabBatchSize) {
            flush_su_slab();
        }
        const double W_u_new = su_scale * W_u_tilde;

        if (W_u_old < 1.0 && W_u_new > 1.0) {
            refresh_theta_cholesky();
        } else {
            const double decay_th = (W_u_old >= 1.0) ? ((lam * W_u_old) / W_u_new) : lam;
            const double coef_th  = (W_u_old >= 1.0) ? (w_t / W_u_new) : w_t;
            const double denom_th = decay_th + coef_th * mahal_sq;
            if (denom_th > 1e-15 && decay_th > 1e-15 && std::isfinite(denom_th)) {
                const double alpha_th = (coef_th * theta_scale) / denom_th;
                ops.sym_rank1_update_upper(Theta_tilde.data(), q_tilde_buf.data(), -alpha_th, D, D_pad);
                theta_scale /= decay_th;
                theta_dirty = true;

                bool need_th_heal = (!std::isfinite(theta_scale) || theta_scale > 1e100);
                if (!need_th_heal) {
                    for (uint32_t j = 0; j < D; ++j) {
                        const double tjj = Theta_tilde[static_cast<size_t>(j) * D_pad + j];
                        if (!std::isfinite(tjj) || tjj <= 1e-15) {
                            need_th_heal = true;
                            break;
                        }
                    }
                }
                if (need_th_heal) {
                    refresh_theta_cholesky();
                }
            } else {
                refresh_theta_cholesky();
            }
        }

        /* 3d. Online Hyvarinen-Smith non-Gaussian moments on normal (non-shock) ticks */
        if (w_t >= 0.99) {
            update_lingam_moments(u_buf.data());
        }
    }

    /* Gentle baseline tracking during calm non-alarm forgetting operation */
    if (cfg.normalize_scale == 1 && alarm == 0 && w_t == 1.0 && lam < 1.0) {
        const double a_b = std::min(0.001, 0.1 * (1.0 - lam));
        for (uint32_t j = 0; j < D; ++j) {
            const double diff = dx_buf[j] - mean0[j];
            mean0[j] += a_b * diff;
            base_var[j] = (1.0 - a_b) * base_var[j] + a_b * diff * diff;
            const double s = std::max(std::sqrt(std::max(0.0, base_var[j])), floor_std[j]);
            inv_scale[j] = 1.0 / s;
        }
    } else if (cfg.normalize_scale == 0 && w_t >= 0.99) {
        ++passive_welford_n;
        const double inv_n = 1.0 / static_cast<double>(passive_welford_n);
        for (uint32_t j = 0; j < D; ++j) {
            const double d1 = dx_buf[j] - passive_mean[j];
            passive_mean[j] += d1 * inv_n;
            const double d2 = dx_buf[j] - passive_mean[j];
            passive_m2[j] += d1 * d2;
        }
    }

    push_lag_hist(x_work_buf.data());

    /* Periodic Freivalds drift monitor & exact Cholesky self-healing */
    if (++ticks_since_refresh >= refresh_interval) {
        ticks_since_refresh = 0;
        const double rel_err = compute_freivalds_err();
        if (rel_err > 1e-7) {
            if (refresh_var_cholesky()) {
                refresh_theta_cholesky();
                ++refresh_count;
            }
        } else if (theta_dirty) {
            refresh_theta_cholesky();
        }
    }

    if (out) {
        out->tick = t;
        out->mahalanobis_sq = mahal_sq;
        out->e_value = e_val;
        out->shock_weight = w_t;
        out->max_exo_z = max_exo_z;
        out->top_root_channel = top_root;
        out->alarm = alarm;
        out->refresh_count = refresh_count;
    }

    ++tick;
    return 0;
}

/* =========================================================================
 * Tier-2 Screened Engine Member Implementations
 * ========================================================================= */
void cs_screened::freeze_warmup_stats() noexcept {
    const uint64_t avail = std::min<uint64_t>(tick, ring_cap);
    const uint64_t start_s = tick - avail;
    const uint64_t end_s = tick;
    const double n_w = static_cast<double>(std::max<uint64_t>(1, tick));

    for (uint32_t j = 0; j < D; ++j) {
        mean[j] = w_mean[j];
        const double v0 = (n_w > 1.0) ? (w_m2[j] / (n_w - 1.0)) : 0.0;
        const double raw_std = std::sqrt(std::max(0.0, v0));
        const double const_eps = std::max(1e-15, 2.0 * std::numeric_limits<double>::epsilon() * std::fabs(anchor[j]));
        double s0 = 0.0;
        if (raw_std > const_eps) {
            s0 = raw_std;
            floor_std[j] = std::min(s0, std::max(1e-12, const_eps));
        } else {
            s0 = std::max(0.01, 0.01 * std::fabs(anchor[j]));
            floor_std[j] = s0;
        }
        var[j] = s0 * s0;
        inv_std[j] = 1.0 / s0;
        ar_cov[j] = 0.0;
        ar_var[j] = 0.0;
        ar_res_var[j] = 0.0;
        prev_norm[j] = 0.0;
        phi[j] = 0.0;
        inv_res_std[j] = 1.0;
        energy[j] = 0.0;
        recent_peak[j] = 0.0;
    }

    /* Replay warmup ticks to calibrate exact per-channel AR(1) phi and residual variance */
    if (avail >= 3) {
        const double* r0 = raw_ring.data() + static_cast<size_t>(start_s % ring_cap) * D;
        for (uint32_t j = 0; j < D; ++j) {
            prev_norm[j] = (r0[j] - anchor[j] - mean[j]) * inv_std[j];
        }
        const double inv_cnt = 1.0 / static_cast<double>(avail - 1);
        for (uint64_t s = start_s + 1; s < end_s; ++s) {
            const double* rs = raw_ring.data() + static_cast<size_t>(s % ring_cap) * D;
            for (uint32_t j = 0; j < D; ++j) {
                const double xn = (rs[j] - anchor[j] - mean[j]) * inv_std[j];
                ar_cov[j] += xn * prev_norm[j] * inv_cnt;
                ar_var[j] += prev_norm[j] * prev_norm[j] * inv_cnt;
                prev_norm[j] = xn;
            }
        }
        for (uint32_t j = 0; j < D; ++j) {
            phi[j] = std::clamp(ar_cov[j] / std::max(ar_var[j], 1e-6), -0.98, 0.98);
            prev_norm[j] = (r0[j] - anchor[j] - mean[j]) * inv_std[j];
        }
        for (uint64_t s = start_s + 1; s < end_s; ++s) {
            const double* rs = raw_ring.data() + static_cast<size_t>(s % ring_cap) * D;
            for (uint32_t j = 0; j < D; ++j) {
                const double xn = (rs[j] - anchor[j] - mean[j]) * inv_std[j];
                const double res = xn - phi[j] * prev_norm[j];
                ar_res_var[j] += res * res * inv_cnt;
                prev_norm[j] = xn;
            }
        }
        for (uint32_t j = 0; j < D; ++j) {
            ar_res_var[j] = std::max(ar_res_var[j], 1e-4);
            inv_res_std[j] = 1.0 / std::sqrt(ar_res_var[j]);
            prev_norm[j] = std::clamp(prev_norm[j], -6.0, 6.0);
        }
    }

    baseline_frozen = true;
}

void cs_screened::sync_tier1() noexcept {
    if (tick == 0) return;

    const uint32_t M = active_m;
    for (uint32_t j = 0; j < D; ++j) {
        const double pk_ex = std::max(0.0, recent_peak[j] - 3.0);
        scratch_scores[j] = energy[j] + 0.5 * pk_ex * pk_ex;
        scratch_order[j] = static_cast<int32_t>(j);
    }

    std::partial_sort(
        scratch_order.begin(),
        scratch_order.begin() + M,
        scratch_order.end(),
        [&](int32_t a, int32_t b) {
            if (scratch_scores[a] != scratch_scores[b]) {
                return scratch_scores[a] > scratch_scores[b];
            }
            return a < b;
        });

    bool same_active = tier1_valid && (active_channels.size() == M);
    if (same_active) {
        /* Compare as unordered set or exact ordered list; exact list keeps channel mapping fixed */
        for (uint32_t m = 0; m < M; ++m) {
            if (active_channels[m] != scratch_order[m]) {
                same_active = false;
                break;
            }
        }
    }

    if (same_active && tick >= tier1_synced_tick && (tick - tier1_synced_tick) < (ring_cap / 2)) {
        for (uint64_t t = tier1_synced_tick; t < tick; ++t) {
            const double* row = raw_ring.data() + static_cast<size_t>(t % ring_cap) * D;
            for (uint32_t m = 0; m < M; ++m) {
                scratch_sub_x[m] = row[active_channels[m]];
            }
            tier1->push_single_validated(scratch_sub_x.data(), nullptr, nullptr, nullptr);
        }
        tier1_synced_tick = tick;
        return;
    }

    /* Active channel set changed or first sync: replay retained ring window cleanly */
    for (uint32_t m = 0; m < M; ++m) {
        active_channels[m] = scratch_order[m];
    }

    const uint64_t avail = std::min<uint64_t>(tick, ring_cap);
    const uint64_t start_t = tick - avail;

    /* Detect first high-z spike in the replay window so Tier-1 warmup covers clean pre-incident baseline */
    const double z_spike_thresh = std::max(4.8, linalg::normal_two_sided_quantile(0.001 / static_cast<double>(D)));
    uint64_t first_spike_off = avail;
    for (uint64_t off = 0; off < avail; ++off) {
        const uint64_t t = start_t + off;
        if (t >= warmup_ticks && max_z_ring[static_cast<size_t>(t % ring_cap)] >= z_spike_thresh) {
            first_spike_off = off;
            break;
        }
    }

    const uint32_t min_warm = static_cast<uint32_t>(std::min<uint64_t>(avail / 2, std::max<uint32_t>(20u, lag + 4u)));
    const uint32_t max_warm = static_cast<uint32_t>(
        std::min<uint64_t>(avail > 8 ? avail - 8 : avail, 256u));
    uint32_t t1_warm = std::min<uint32_t>(max_warm, std::max<uint32_t>(64u, 2u * M + lag));
    if (first_spike_off < avail && first_spike_off >= min_warm + 2) {
        t1_warm = static_cast<uint32_t>(std::min<uint64_t>(max_warm, first_spike_off - 2));
    } else if (avail < t1_warm + 4) {
        t1_warm = min_warm;
    }

    tier1->warmup_ticks = std::max<uint32_t>(lag + 2u, t1_warm);
    tier1->reset_state();

    for (uint64_t t = start_t; t < tick; ++t) {
        const double* row = raw_ring.data() + static_cast<size_t>(t % ring_cap) * D;
        for (uint32_t m = 0; m < M; ++m) {
            scratch_sub_x[m] = row[active_channels[m]];
        }
        tier1->push_single_validated(scratch_sub_x.data(), nullptr, nullptr, nullptr);
    }

    tier1_synced_tick = tick;
    tier1_valid = true;
}

/* =========================================================================
 * C ABI Exported Functions
 * ========================================================================= */
extern "C" {

void cs_config_default(cs_config_t* cfg, uint32_t dim, uint32_t lag) {
    if (!cfg) return;
    cfg->dim = dim;
    cfg->lag = lag;
    cfg->forgetting = 0.995;
    cfg->ridge = 1e-2;
    cfg->precision_ridge = 1e-2;
    cfg->shock_threshold = (dim > 0)
        ? std::max(2.0, std::min(3.5, 1.5 + 3.0 / std::sqrt(static_cast<double>(dim))))
        : 2.5;
    cfg->e_alpha = 0.01;
    cfg->warmup_ticks = 0;
    cfg->refresh_interval = 0;
    cfg->nonlinear_hinge = 0;
    cfg->raw_ring_capacity = 8192;
    cfg->normalize_scale = 1;
}

cs_engine_t* cs_create(const cs_config_t* cfg) {
    if (!cfg) return nullptr;
    if (cfg->dim == 0 || cfg->dim > 4096) return nullptr;
    if (cfg->lag == 0 || cfg->lag > 16) return nullptr;
    if (!std::isfinite(cfg->forgetting) || cfg->forgetting <= 0.0 || cfg->forgetting > 1.0) return nullptr;
    if (!std::isfinite(cfg->ridge) || cfg->ridge <= 0.0) return nullptr;
    if (!std::isfinite(cfg->precision_ridge) || cfg->precision_ridge <= 0.0) return nullptr;
    if (!std::isfinite(cfg->shock_threshold) || cfg->shock_threshold < 0.0) return nullptr;
    if (!std::isfinite(cfg->e_alpha) || cfg->e_alpha <= 0.0 || cfg->e_alpha >= 1.0) return nullptr;
    if (cfg->nonlinear_hinge > 1 || cfg->normalize_scale > 1) return nullptr;

    const uint64_t mult = cfg->nonlinear_hinge ? 2ULL : 1ULL;
    const uint64_t K64 = 1ULL + mult * static_cast<uint64_t>(cfg->dim) * static_cast<uint64_t>(cfg->lag);
    if (K64 > 65536ULL) return nullptr;

    cs_engine_t* eng = nullptr;
    try {
        eng = new cs_engine();
        eng->cfg = *cfg;
        eng->D = cfg->dim;
        eng->p = cfg->lag;
        eng->K = static_cast<uint32_t>(K64);
        eng->warmup_ticks = (cfg->warmup_ticks > 0)
            ? cfg->warmup_ticks
            : std::min<uint32_t>(80u, std::max<uint32_t>(24u, std::max<uint32_t>(cfg->lag + 2u, 2u * cfg->dim + cfg->lag)));
        eng->refresh_interval = (cfg->refresh_interval > 0)
            ? cfg->refresh_interval
            : ((cfg->forgetting < 1.0) ? 128u : 512u);
        eng->ring_cap = (cfg->raw_ring_capacity > 0) ? cfg->raw_ring_capacity : 8192u;
        eng->init_buffers();
        eng->reset_state();
    } catch (...) {
        delete eng;
        return nullptr;
    }
    return eng;
}

void cs_destroy(cs_engine_t* eng) {
    delete eng;
}

void cs_reset(cs_engine_t* eng) {
    if (!eng) return;
    eng->reset_state();
}

const char* cs_simd_isa(void) {
    return causalsieve::kernels::active_ops().isa_name;
}

const char* cs_last_error(const cs_engine_t* eng) {
    if (!eng) return "null engine";
    return eng->last_error.c_str();
}

uint64_t cs_tick_count(const cs_engine_t* eng) {
    return eng ? eng->tick : 0;
}

uint32_t cs_dim(const cs_engine_t* eng) {
    return eng ? eng->D : 0;
}

uint32_t cs_lag(const cs_engine_t* eng) {
    return eng ? eng->p : 0;
}

uint32_t cs_feature_dim(const cs_engine_t* eng) {
    return eng ? eng->K : 0;
}

int cs_push(cs_engine_t* eng,
            const double* x,
            cs_tick_result_t* out,
            double* out_exo_z,
            double* out_innov) {
    if (!eng) return -1;
    if (!x) {
        eng->last_error = "null input vector x";
        return -1;
    }
    if (!validate_finite(x, eng->D)) {
        eng->last_error = "non-finite input value (NaN/Inf); state unchanged";
        return -1;
    }
    eng->last_error.clear();
    return eng->push_single_validated(x, out, out_exo_z, out_innov);
}

int cs_push_batch(cs_engine_t* eng,
                  const double* X,
                  size_t n_ticks,
                  cs_tick_result_t* out_results,
                  double* out_exo_z_matrix,
                  double* out_innov_matrix) {
    if (!eng) return -1;
    if (n_ticks == 0) {
        eng->last_error.clear();
        return 0;
    }
    if (!X) {
        eng->last_error = "null batch input matrix X";
        return -1;
    }
    /* All-or-nothing atomic pre-validation */
    const size_t total = n_ticks * static_cast<size_t>(eng->D);
    if (!validate_finite(X, total)) {
        eng->last_error = "non-finite input value in batch (NaN/Inf); state unchanged";
        return -1;
    }
    eng->last_error.clear();

    const size_t D = eng->D;
    for (size_t t = 0; t < n_ticks; ++t) {
        cs_tick_result_t* r_ptr = out_results ? (out_results + t) : nullptr;
        double* ez_ptr = out_exo_z_matrix ? (out_exo_z_matrix + t * D) : nullptr;
        double* in_ptr = out_innov_matrix ? (out_innov_matrix + t * D) : nullptr;
        eng->push_single_validated(X + t * D, r_ptr, ez_ptr, in_ptr);
    }
    return 0;
}

int cs_granger_matrix(cs_engine_t* eng,
                      double* out_f_stat,
                      double* out_partial_r2) {
    if (!eng) return -1;
    const uint32_t D = eng->D;
    const size_t DD = static_cast<size_t>(D) * D;

    if (eng->W_eff_tilde <= 0.0) {
        if (out_f_stat) std::memset(out_f_stat, 0, DD * sizeof(double));
        if (out_partial_r2) std::memset(out_partial_r2, 0, DD * sizeof(double));
        return 0;
    }

    /* Guarantee exact (G_t + ridge*I)^{-1} on forgetting streams, adaptive ridge, or numerical drift */
    eng->flush_var_slab();
    if (eng->var_dirty_since_refresh) {
        if (eng->cfg.forgetting < 1.0 || eng->cfg.normalize_scale == 1 || eng->compute_freivalds_err() > 1e-9) {
            eng->refresh_var_cholesky();
        }
    }

    const double W_eff = eng->g_scale * eng->W_eff_tilde;
    const double df_eff = std::max(1.0, W_eff - static_cast<double>(eng->K));
    const double* J_ptr = (eng->has_refreshed_j && !eng->var_dirty_since_refresh)
        ? eng->J_exact.data()
        : eng->J_rec.data();

    const uint32_t p = eng->p;
    const uint32_t m = eng->cfg.nonlinear_hinge ? (2u * p) : p;
    const size_t K_pad = eng->K_pad;
    const size_t D_pad = eng->D_pad;
    const double ps = eng->p_scale;

    uint32_t S_idx[32];
    double P_sub[32 * 32];
    double L_sub[32 * 32];
    double v_sub[32];

    for (uint32_t src = 0; src < D; ++src) {
        /* Indices for source channel `src` in ascending order */
        for (uint32_t l = 0; l < p; ++l) {
            S_idx[l] = 1u + l * D + src;
        }
        if (eng->cfg.nonlinear_hinge) {
            for (uint32_t l = 0; l < p; ++l) {
                S_idx[p + l] = 1u + p * D + l * D + src;
            }
        }

        if (m == 1) {
            const uint32_t s0 = S_idx[0];
            const double p00 = std::max(1e-30, ps * eng->P_tilde[static_cast<size_t>(s0) * K_pad + s0]);
            const double inv_p00 = 1.0 / p00;
            const double* B_row = eng->B.data() + static_cast<size_t>(s0) * D_pad;
            for (uint32_t dst = 0; dst < D; ++dst) {
                const double b = B_row[dst];
                const double delta = b * b * inv_p00;
                const double rss = std::max(1e-30, J_ptr[dst]);
                if (out_f_stat) out_f_stat[static_cast<size_t>(src) * D + dst] = (delta / 1.0) / (rss / df_eff);
                if (out_partial_r2) out_partial_r2[static_cast<size_t>(src) * D + dst] = delta / (rss + delta);
            }
        } else {
            /* Extract upper triangle of principal submatrix P_{S_i, S_i} */
            for (uint32_t a = 0; a < m; ++a) {
                const size_t sa = S_idx[a];
                for (uint32_t b = a; b < m; ++b) {
                    const size_t sb = S_idx[b];
                    P_sub[a * m + b] = ps * eng->P_tilde[sa * K_pad + sb];
                }
            }
            linalg::cholesky_small(P_sub, L_sub, m);

            for (uint32_t dst = 0; dst < D; ++dst) {
                /* Forward solve L_sub * v = B_{S_i, dst} -> delta = ||v||_2^2 >= 0 */
                double delta = 0.0;
                for (uint32_t a = 0; a < m; ++a) {
                    double sum = eng->B[static_cast<size_t>(S_idx[a]) * D_pad + dst];
                    for (uint32_t k = 0; k < a; ++k) {
                        sum -= L_sub[a * m + k] * v_sub[k];
                    }
                    const double va = sum / L_sub[a * m + a];
                    v_sub[a] = va;
                    delta += va * va;
                }
                const double rss = std::max(1e-30, J_ptr[dst]);
                if (out_f_stat) {
                    out_f_stat[static_cast<size_t>(src) * D + dst] =
                        (delta / static_cast<double>(m)) / (rss / df_eff);
                }
                if (out_partial_r2) {
                    out_partial_r2[static_cast<size_t>(src) * D + dst] = delta / (rss + delta);
                }
            }
        }
    }
    return 0;
}

int cs_precision_matrix(cs_engine_t* eng,
                        double* out_theta,
                        double* out_pcorr) {
    if (!eng) return -1;
    eng->flush_su_slab();
    if (eng->theta_dirty) {
        eng->refresh_theta_cholesky();
    }

    const uint32_t D = eng->D;
    const size_t D_pad = eng->D_pad;
    const double ts = eng->theta_scale;

    if (out_theta) {
        for (uint32_t i = 0; i < D; ++i) {
            out_theta[static_cast<size_t>(i) * D + i] = ts * eng->Theta_tilde[static_cast<size_t>(i) * D_pad + i];
            for (uint32_t j = i + 1; j < D; ++j) {
                const double v = ts * eng->Theta_tilde[static_cast<size_t>(i) * D_pad + j];
                out_theta[static_cast<size_t>(i) * D + j] = v;
                out_theta[static_cast<size_t>(j) * D + i] = v;
            }
        }
    }

    if (out_pcorr) {
        for (uint32_t i = 0; i < D; ++i) {
            out_pcorr[static_cast<size_t>(i) * D + i] = 1.0;
            const double tii = std::max(1e-30, ts * eng->Theta_tilde[static_cast<size_t>(i) * D_pad + i]);
            for (uint32_t j = i + 1; j < D; ++j) {
                const double tjj = std::max(1e-30, ts * eng->Theta_tilde[static_cast<size_t>(j) * D_pad + j]);
                const double tij = ts * eng->Theta_tilde[static_cast<size_t>(i) * D_pad + j];
                const double pc = std::clamp(-tij / std::sqrt(tii * tjj), -1.0, 1.0);
                out_pcorr[static_cast<size_t>(i) * D + j] = pc;
                out_pcorr[static_cast<size_t>(j) * D + i] = pc;
            }
        }
    }
    return 0;
}

int cs_contemporaneous_orientation(cs_engine_t* eng, double* out_lingam_r) {
    if (!eng || !out_lingam_r) return -1;
    eng->flush_lingam_slab();
    const uint32_t D = eng->D;
    const size_t D_pad = eng->D_pad;

    if (eng->lin_W_tilde <= 0.0) {
        std::memset(out_lingam_r, 0, static_cast<size_t>(D) * D * sizeof(double));
        return 0;
    }

    const double inv_W = 1.0 / eng->lin_W_tilde;

    for (uint32_t i = 0; i < D; ++i) {
        out_lingam_r[static_cast<size_t>(i) * D + i] = 0.0;
        const double vi = std::max(1e-18, eng->M_uv_tilde[static_cast<size_t>(i) * D_pad + i] * inv_W);
        for (uint32_t j = i + 1; j < D; ++j) {
            const double vj = std::max(1e-18, eng->M_uv_tilde[static_cast<size_t>(j) * D_pad + j] * inv_W);
            const double cov_ij = eng->M_uv_tilde[static_cast<size_t>(i) * D_pad + j] * inv_W;
            const double rho_ij = std::clamp(cov_ij / std::sqrt(vi * vj), -1.0, 1.0);

            const double m_u2v_ij = eng->M_u2v_tilde[static_cast<size_t>(i) * D_pad + j] * inv_W;
            const double m_u2v_ji = eng->M_u2v_tilde[static_cast<size_t>(j) * D_pad + i] * inv_W;
            const double m_ut_ij  = eng->M_utanhv_tilde[static_cast<size_t>(i) * D_pad + j] * inv_W;
            const double m_ut_ji  = eng->M_utanhv_tilde[static_cast<size_t>(j) * D_pad + i] * inv_W;

            /* Skewness asymmetry (sign-invariant in rho and skewness) + Hyvarinen-Smith super-Gaussian tanh */
            const double r_skew = std::fabs(rho_ij) * (std::fabs(m_u2v_ij) - std::fabs(m_u2v_ji));
            const double r_tanh = rho_ij * (m_ut_ij - m_ut_ji);
            const double R_ij = r_skew + r_tanh;

            out_lingam_r[static_cast<size_t>(i) * D + j] = R_ij;
            out_lingam_r[static_cast<size_t>(j) * D + i] = -R_ij;
        }
    }
    return 0;
}

int cs_root_attribution(cs_engine_t* eng,
                        uint32_t lookback_ticks,
                        double* out_root_scores,
                        double* out_peak_exo_z,
                        double* out_raw_z,
                        int64_t* out_onset_tick,
                        int32_t* out_ranked_channels) {
    if (!eng) return -1;
    eng->flush_all_slabs();
    const uint32_t D = eng->D;
    const uint64_t t_end = eng->tick;
    const uint64_t avail = std::min<uint64_t>(t_end, eng->ring_cap);

    std::vector<double> peak_exo(D, 0.0);
    std::vector<double> peak_raw(D, 0.0);
    std::vector<double> excess_exo(D, 0.0);
    std::vector<int64_t> onset(D, -1);
    std::vector<double> root_scores(D, 0.0);
    std::vector<int32_t> ranked(D);
    std::iota(ranked.begin(), ranked.end(), 0);

    if (avail > 0) {
        uint64_t lb = 0;
        if (lookback_ticks == 0) {
            if (eng->incident_start_tick != UINT64_MAX && eng->incident_start_tick < t_end) {
                lb = std::min<uint64_t>(avail, std::min<uint64_t>(512, std::max<uint64_t>(64, t_end - eng->incident_start_tick + 8)));
            } else {
                lb = std::min<uint64_t>(avail, 64);
            }
        } else {
            lb = std::min<uint64_t>(avail, lookback_ticks);
        }

        uint64_t t_start = t_end - lb;
        if (t_end > eng->warmup_ticks) {
            t_start = std::max<uint64_t>(t_start, eng->warmup_ticks);
        }

        std::vector<double> peak_mz(D, 0.0);
        std::vector<double> peak_iz(D, 0.0);
        std::vector<double> sum_uq(D, 0.0);
        double global_peak_mz = 0.0;
        const uint64_t win_len = (t_end > t_start) ? (t_end - t_start) : 1;

        for (uint64_t t = t_start; t < t_end; ++t) {
            const size_t slot = static_cast<size_t>(t % eng->ring_cap) * D;
            const double* ez_row = eng->exo_z_ring.data() + slot;
            const double* rz_row = eng->raw_z_ring.data() + slot;
            const double* uq_row = eng->uq_ring.data() + slot;
            const double* iz_row = eng->innov_z_ring.data() + slot;
            for (uint32_t j = 0; j < D; ++j) {
                const double aez = std::fabs(ez_row[j]);
                const double arz = std::fabs(rz_row[j]);
                double uq  = uq_row[j];
                double mz  = (uq > 0.0) ? std::sqrt(uq) : 0.0;
                double iz  = iz_row[j];
                if (eng->cfg.normalize_scale == 1) {
                    /* Prevent hallucinated missing-cascade predictions on calm channels from exceeding channel deviation */
                    const double dev_cap = std::max(2.0, 2.5 * arz);
                    if (mz > dev_cap) mz = dev_cap;
                    if (uq > dev_cap * dev_cap) uq = dev_cap * dev_cap;
                    if (iz > dev_cap) iz = dev_cap;
                }
                if (aez > peak_exo[j]) peak_exo[j] = aez;
                if (arz > peak_raw[j]) peak_raw[j] = arz;
                if (mz  > peak_mz[j])  peak_mz[j]  = mz;
                if (iz  > peak_iz[j])  peak_iz[j]  = iz;
                if (aez > 2.5) excess_exo[j] += (aez - 2.5);
                sum_uq[j] += uq;
                if (mz > global_peak_mz) global_peak_mz = mz;
            }
        }

        /* Per-channel adaptive exogenous onset tick (only significant incident channels set earliest_onset) */
        const double sig_channel_floor = std::max(3.5, 0.25 * global_peak_mz);
        int64_t earliest_onset = -1;
        for (uint64_t t = t_start; t < t_end; ++t) {
            const size_t slot = static_cast<size_t>(t % eng->ring_cap) * D;
            const double* uq_row = eng->uq_ring.data() + slot;
            const double* rz_row = eng->raw_z_ring.data() + slot;
            for (uint32_t j = 0; j < D; ++j) {
                if (onset[j] >= 0 || peak_mz[j] < 3.2) continue;
                const double ch_thresh = std::max(3.2, std::min(6.0, 0.25 * peak_mz[j]));
                double mz = (uq_row[j] > 0.0) ? std::sqrt(uq_row[j]) : 0.0;
                if (eng->cfg.normalize_scale == 1) {
                    const double dev_cap = std::max(2.0, 2.5 * std::fabs(rz_row[j]));
                    if (mz > dev_cap) mz = dev_cap;
                }
                if (mz >= ch_thresh) {
                    onset[j] = static_cast<int64_t>(t);
                    if (peak_mz[j] >= sig_channel_floor) {
                        if (earliest_onset < 0 || onset[j] < earliest_onset) {
                            earliest_onset = onset[j];
                        }
                    }
                }
            }
        }

        /* Directed causal graph net outflow: conditional Granger + |pcorr|-weighted LiNGAM */
        std::vector<double> partial_r2(static_cast<size_t>(D) * D, 0.0);
        std::vector<double> lingam_r(static_cast<size_t>(D) * D, 0.0);
        std::vector<double> pcorr(static_cast<size_t>(D) * D, 0.0);
        cs_granger_matrix(eng, nullptr, partial_r2.data());
        cs_contemporaneous_orientation(eng, lingam_r.data());
        cs_precision_matrix(eng, nullptr, pcorr.data());

        const double W_eff = eng->g_scale * eng->W_eff_tilde;
        const double df_eff = std::max(1.0, W_eff - static_cast<double>(eng->K));
        const double m = static_cast<double>(eng->cfg.nonlinear_hinge ? (2u * eng->p) : eng->p);
        const double r2_floor = std::min(0.05, 2.0 * m / df_eff);
        const double W_lin = eng->lin_scale * eng->lin_W_tilde;
        const double lingam_floor = 0.75 / std::sqrt(std::max(25.0, W_lin));

        std::vector<double> anom_w(D, 0.0);
        for (uint32_t k = 0; k < D; ++k) {
            const double mx = std::max(peak_mz[k], peak_raw[k]);
            anom_w[k] = std::clamp((mx - 2.5) / 5.0, 0.0, 1.0);
        }

        for (uint32_t j = 0; j < D; ++j) {
            double outflow = 0.0;
            double inflow = 0.0;
            for (uint32_t k = 0; k < D; ++k) {
                if (k == j || anom_w[k] <= 0.05) continue;
                const double g_jk = std::max(0.0, partial_r2[static_cast<size_t>(j) * D + k] - r2_floor);
                const double g_kj = std::max(0.0, partial_r2[static_cast<size_t>(k) * D + j] - r2_floor);
                const double pc_abs = std::fabs(pcorr[static_cast<size_t>(j) * D + k]);
                const double r_jk = std::max(0.0, lingam_r[static_cast<size_t>(j) * D + k] - lingam_floor);
                const double r_kj = std::max(0.0, lingam_r[static_cast<size_t>(k) * D + j] - lingam_floor);
                const double l_jk = 1.5 * pc_abs * std::tanh(8.0 * r_jk);
                const double l_kj = 1.5 * pc_abs * std::tanh(8.0 * r_kj);
                outflow += anom_w[k] * (g_jk + l_jk);
                inflow  += anom_w[k] * (g_kj + l_kj);
            }
            const double net_outflow = outflow - inflow;

            /* Exact DAG Mahalanobis energy u_j*(Theta u)_j + damped temporal innovation z (for collinear twin metrics) */
            const double net_energy = std::max(0.0, sum_uq[j] - static_cast<double>(win_len));
            double dag_intensity = peak_mz[j] + 0.5 * std::sqrt(net_energy);
            if (eng->cfg.normalize_scale == 1) {
                dag_intensity += 0.35 * std::log1p(peak_iz[j]);
            }

            double onset_mult = 1.0;
            if (onset[j] >= 0 && earliest_onset >= 0) {
                const double delay = std::max(0.0, static_cast<double>(onset[j] - earliest_onset));
                onset_mult = 1.0 + 0.55 * std::exp(-0.35 * delay);
            }

            const double graph_mult = std::exp(0.55 * std::tanh(net_outflow));
            const double act_gate = (eng->cfg.normalize_scale == 1)
                ? std::clamp((peak_raw[j] - 1.5) / 2.5, 0.05, 1.0)
                : 1.0;
            root_scores[j] = dag_intensity * act_gate * onset_mult * graph_mult;
        }

        std::sort(ranked.begin(), ranked.end(), [&](int32_t a, int32_t b) {
            if (root_scores[a] != root_scores[b]) return root_scores[a] > root_scores[b];
            if (peak_mz[a] != peak_mz[b]) return peak_mz[a] > peak_mz[b];
            return a < b;
        });
    }

    if (out_root_scores) std::memcpy(out_root_scores, root_scores.data(), D * sizeof(double));
    if (out_peak_exo_z) std::memcpy(out_peak_exo_z, peak_exo.data(), D * sizeof(double));
    if (out_raw_z) std::memcpy(out_raw_z, peak_raw.data(), D * sizeof(double));
    if (out_onset_tick) std::memcpy(out_onset_tick, onset.data(), D * sizeof(int64_t));
    if (out_ranked_channels) std::memcpy(out_ranked_channels, ranked.data(), D * sizeof(int32_t));
    return 0;
}

int cs_counterfactual_do(cs_engine_t* eng,
                         int32_t clamp_channel,
                         double delta_value,
                         uint32_t horizon,
                         double* out_delta_trajectory) {
    if (!eng || !out_delta_trajectory) return -1;
    if (clamp_channel < 0 || clamp_channel >= static_cast<int32_t>(eng->D)) return -1;
    if (!std::isfinite(delta_value) || horizon == 0) return -1;

    if (eng->cfg.forgetting < 1.0 && eng->var_dirty_since_refresh && eng->W_eff_tilde > 0.0) {
        eng->refresh_var_cholesky();
    }

    const uint32_t D = eng->D;
    const uint32_t p = eng->p;
    const size_t D_pad = eng->D_pad;

    std::memset(out_delta_trajectory, 0, static_cast<size_t>(horizon) * D * sizeof(double));
    out_delta_trajectory[clamp_channel] = delta_value;

    for (uint32_t h = 1; h < horizon; ++h) {
        double* dst_row = out_delta_trajectory + static_cast<size_t>(h) * D;
        for (uint32_t l = 0; l < p; ++l) {
            if (h < l + 1u) break;
            const double* prev = out_delta_trajectory + static_cast<size_t>(h - 1u - l) * D;
            const size_t base_lin = 1u + static_cast<size_t>(l) * D;
            for (uint32_t c = 0; c < D; ++c) {
                const double pc = prev[c];
                if (pc == 0.0) continue;
                const double* B_row = eng->B.data() + (base_lin + c) * D_pad;
                for (uint32_t j = 0; j < D; ++j) {
                    dst_row[j] += pc * B_row[j];
                }
            }
            if (eng->cfg.nonlinear_hinge) {
                const size_t base_h = 1u + static_cast<size_t>(p) * D + static_cast<size_t>(l) * D;
                for (uint32_t c = 0; c < D; ++c) {
                    const double hc = (prev[c] > 1.0) ? (prev[c] - 1.0) : 0.0;
                    if (hc == 0.0) continue;
                    const double* B_row = eng->B.data() + (base_h + c) * D_pad;
                    for (uint32_t j = 0; j < D; ++j) {
                        dst_row[j] += hc * B_row[j];
                    }
                }
            }
        }
    }
    return 0;
}

int cs_get_raw_slice(cs_engine_t* eng,
                     uint32_t channel,
                     uint64_t start_tick,
                     uint64_t end_tick,
                     double* out_values,
                     uint64_t* out_count,
                     uint64_t* out_actual_start) {
    if (!eng || !out_count || !out_actual_start) return -1;
    if (channel >= eng->D) {
        eng->last_error = "channel out of range";
        return -1;
    }

    if (start_tick >= end_tick || eng->tick == 0) {
        *out_count = 0;
        *out_actual_start = start_tick;
        return 0;
    }

    const uint64_t ring_end = eng->tick;
    const uint64_t ring_start = (eng->tick > eng->ring_cap) ? (eng->tick - eng->ring_cap) : 0;
    const uint64_t a = std::max(start_tick, ring_start);
    const uint64_t b = std::min(end_tick, ring_end);

    if (a >= b) {
        *out_count = 0;
        *out_actual_start = a;
        return 0;
    }
    if (!out_values) {
        eng->last_error = "null out_values buffer";
        return -1;
    }

    const uint32_t D = eng->D;
    const uint32_t cap = eng->ring_cap;
    const double* ring = eng->raw_ring.data();
    for (uint64_t t = a; t < b; ++t) {
        out_values[t - a] = ring[static_cast<size_t>(t % cap) * D + channel];
    }
    *out_actual_start = a;
    *out_count = b - a;
    return 0;
}

int cs_compress_channel_pla(cs_engine_t* eng,
                            uint32_t channel,
                            uint64_t start_tick,
                            uint64_t end_tick,
                            double max_abs_err,
                            uint64_t* out_knot_ticks,
                            double* out_knot_vals,
                            size_t max_knots,
                            size_t* out_n_knots) {
    if (!eng || !out_knot_ticks || !out_knot_vals || !out_n_knots) return -1;
    if (channel >= eng->D || !std::isfinite(max_abs_err) || max_abs_err < 0.0) return -1;

    if (start_tick >= end_tick || eng->tick == 0) {
        *out_n_knots = 0;
        return 0;
    }

    const uint64_t ring_end = eng->tick;
    const uint64_t ring_start = (eng->tick > eng->ring_cap) ? (eng->tick - eng->ring_cap) : 0;
    const uint64_t a = std::max(start_tick, ring_start);
    const uint64_t b = std::min(end_tick, ring_end);
    if (a >= b) {
        *out_n_knots = 0;
        return 0;
    }

    const uint32_t D = eng->D;
    const uint32_t cap = eng->ring_cap;
    const double* ring = eng->raw_ring.data();
    auto val_at = [&](uint64_t t) noexcept -> double {
        return ring[static_cast<size_t>(t % cap) * D + channel];
    };

    const uint64_t n = b - a;
    if (n == 1) {
        if (max_knots < 1) return -1;
        out_knot_ticks[0] = a;
        out_knot_vals[0] = val_at(a);
        *out_n_knots = 1;
        return 0;
    }

    if (max_knots < 2) return -1;

    /* Exact O(N) Swing-Filter (Elmeleegy et al., VLDB 2009) with strict FP64 L_inf guard */
    size_t n_knots = 0;
    uint64_t t_seg = a;
    double y_seg = val_at(a);
    out_knot_ticks[n_knots] = t_seg;
    out_knot_vals[n_knots] = y_seg;
    ++n_knots;

    double u_min = -std::numeric_limits<double>::infinity();
    double u_max =  std::numeric_limits<double>::infinity();

    constexpr double kEpsMach = std::numeric_limits<double>::epsilon();
    auto safe_eps = [&](double y, double y0) noexcept -> double {
        if (max_abs_err <= 0.0) return 0.0;
        const double mag = std::max(1.0, std::max(std::fabs(y), std::fabs(y0)));
        const double guard = 16.0 * kEpsMach * mag;
        const double shrunk = max_abs_err * (1.0 - 1e-11) - guard;
        return (shrunk > 0.0) ? shrunk : 0.0;
    };

    for (uint64_t t = a + 1; t < b; ++t) {
        const double y = val_at(t);
        const double dt = static_cast<double>(t - t_seg);
        const double eps = safe_eps(y, y_seg);
        const double lo_k = (y - eps - y_seg) / dt;
        const double hi_k = (y + eps - y_seg) / dt;
        const double new_min = std::max(u_min, lo_k);
        const double new_max = std::min(u_max, hi_k);

        if (new_min <= new_max) {
            u_min = new_min;
            u_max = new_max;
        } else {
            /* Close previous segment at t - 1 */
            const uint64_t t_prev = t - 1;
            double y_end = 0.0;
            if (t_prev == t_seg + 1) {
                y_end = val_at(t_prev);
            } else {
                const double slope = 0.5 * (u_min + u_max);
                y_end = y_seg + slope * static_cast<double>(t_prev - t_seg);
            }

            if (n_knots >= max_knots) {
                *out_n_knots = n_knots;
                return -1;
            }
            out_knot_ticks[n_knots] = t_prev;
            out_knot_vals[n_knots] = y_end;
            ++n_knots;

            /* Re-anchor new connected segment at (t_prev, y_end) and process tick t */
            t_seg = t_prev;
            y_seg = y_end;
            const double eps2 = safe_eps(y, y_seg);
            u_min = (y - eps2 - y_seg); /* dt == 1.0 */
            u_max = (y + eps2 - y_seg);
        }
    }

    /* Emit final knot at b - 1 */
    const uint64_t t_last = b - 1;
    double y_last = 0.0;
    if (t_last == t_seg + 1) {
        y_last = val_at(t_last);
    } else {
        const double slope = 0.5 * (u_min + u_max);
        y_last = y_seg + slope * static_cast<double>(t_last - t_seg);
    }

    if (n_knots >= max_knots) {
        *out_n_knots = n_knots;
        return -1;
    }
    out_knot_ticks[n_knots] = t_last;
    out_knot_vals[n_knots] = y_last;
    ++n_knots;

    *out_n_knots = n_knots;
    return 0;
}

int cs_inspect_state(cs_engine_t* eng,
                     double* out_B,
                     double* out_freivalds_rel_err) {
    if (!eng) return -1;
    eng->flush_var_slab();

    if ((eng->cfg.forgetting < 1.0 || eng->cfg.normalize_scale == 1) && eng->var_dirty_since_refresh && eng->W_eff_tilde > 0.0) {
        eng->refresh_var_cholesky();
    }

    if (out_B) {
        const uint32_t K = eng->K;
        const uint32_t D = eng->D;
        const size_t D_pad = eng->D_pad;
        for (uint32_t i = 0; i < K; ++i) {
            std::memcpy(out_B + static_cast<size_t>(i) * D,
                        eng->B.data() + static_cast<size_t>(i) * D_pad,
                        D * sizeof(double));
        }
    }

    if (out_freivalds_rel_err) {
        *out_freivalds_rel_err = eng->compute_freivalds_err();
    }
    return 0;
}

int cs_force_refresh(cs_engine_t* eng) {
    if (!eng) return -1;
    eng->flush_all_slabs();
    const bool ok_var = eng->refresh_var_cholesky();
    const bool ok_th = eng->refresh_theta_cholesky();
    eng->ticks_since_refresh = 0;
    ++eng->refresh_count;
    return (ok_var && ok_th) ? 0 : -1;
}

/* =========================================================================
 * Tier-2 High-Dimensional Screened Engine C ABI
 * ========================================================================= */
cs_screened_t* cs_screened_create(uint32_t dim,
                                  uint32_t active_m,
                                  uint32_t lag,
                                  double forgetting,
                                  double ridge) {
    if (dim == 0 || active_m == 0 || lag == 0 || lag > 16) return nullptr;
    if (!std::isfinite(forgetting) || forgetting <= 0.0 || forgetting > 1.0) return nullptr;
    if (!std::isfinite(ridge) || ridge <= 0.0) return nullptr;

    cs_screened_t* s = nullptr;
    try {
        s = new cs_screened();
        s->kops = &causalsieve::kernels::active_ops();
        s->D = dim;
        s->active_m = std::min(active_m, dim);
        s->lag = lag;
        s->D_pad = pad4(dim);
        s->forgetting = forgetting;
        s->ridge = ridge;
        s->warmup_ticks = 40;
        s->ring_cap = 512;

        s->anchor.resize(s->D_pad);
        s->floor_std.resize(s->D_pad);
        s->w_mean.resize(s->D_pad);
        s->w_m2.resize(s->D_pad);
        s->mean.resize(s->D_pad);
        s->var.resize(s->D_pad);
        s->inv_std.resize(s->D_pad);
        s->prev_norm.resize(s->D_pad);
        s->ar_cov.resize(s->D_pad);
        s->ar_var.resize(s->D_pad);
        s->ar_res_var.resize(s->D_pad);
        s->phi.resize(s->D_pad);
        s->inv_res_std.resize(s->D_pad);
        s->energy.resize(s->D_pad);
        s->recent_peak.resize(s->D_pad);
        s->last_z.resize(s->D_pad);

        for (uint32_t j = 0; j < s->D; ++j) {
            s->floor_std[j] = 1e-6;
            s->inv_std[j] = 1.0;
            s->inv_res_std[j] = 1.0;
            s->var[j] = 1.0;
            s->ar_res_var[j] = 1.0;
        }

        s->raw_ring.resize(static_cast<size_t>(s->ring_cap) * s->D);
        s->max_z_ring.resize(s->ring_cap);

        s->active_channels.resize(s->active_m, 0);
        s->scratch_order.resize(s->D, 0);
        s->scratch_scores.resize(s->D, 0.0);
        s->scratch_sub_x.resize(s->active_m, 0.0);
        s->t1_root_scores.resize(s->active_m, 0.0);
        s->t1_ranked.resize(s->active_m, 0);

        cs_config_t t1_cfg{};
        cs_config_default(&t1_cfg, s->active_m, lag);
        t1_cfg.forgetting = forgetting;
        t1_cfg.ridge = ridge;
        t1_cfg.precision_ridge = std::max(1e-2, ridge);
        t1_cfg.raw_ring_capacity = s->ring_cap;
        t1_cfg.normalize_scale = 1;
        s->tier1 = cs_create(&t1_cfg);
        if (!s->tier1) {
            delete s;
            return nullptr;
        }
    } catch (...) {
        delete s;
        return nullptr;
    }
    return s;
}

void cs_screened_destroy(cs_screened_t* s) {
    delete s;
}

int cs_screened_push_batch(cs_screened_t* s,
                           const double* X,
                           size_t n_ticks,
                           double* out_last_z_scores,
                           int32_t* out_top_k_channels,
                           double* out_top_k_scores,
                           uint32_t top_k) {
    if (!s) return -1;
    if (n_ticks == 0) return 0;
    if (!X) return -1;

    const size_t D = s->D;
    if (!validate_finite(X, n_ticks * D)) return -1;

    const double ema_alpha = (s->forgetting < 1.0)
        ? std::min(0.01, 1.0 - s->forgetting)
        : 0.002;
    const double energy_decay = 0.94;
    const double peak_decay = 0.97;

    for (size_t r = 0; r < n_ticks; ++r) {
        const double* row = X + r * D;
        const uint64_t t = s->tick;
        const size_t slot = static_cast<size_t>(t % s->ring_cap);

        std::memcpy(s->raw_ring.data() + slot * D, row, D * sizeof(double));

        if (t == 0) {
            std::memcpy(s->anchor.data(), row, D * sizeof(double));
        }

        if (t < s->warmup_ticks) {
            const double inv_n = 1.0 / static_cast<double>(t + 1);
            for (size_t j = 0; j < D; ++j) {
                const double dx = row[j] - s->anchor[j];
                const double d1 = dx - s->w_mean[j];
                s->w_mean[j] += d1 * inv_n;
                const double d2 = dx - s->w_mean[j];
                s->w_m2[j] += d1 * d2;
                s->last_z[j] = 0.0;
            }
            s->max_z_ring[slot] = 0.0;
            ++s->tick;
            continue;
        }

        if (!s->baseline_frozen) {
            s->freeze_warmup_stats();
        }

        const double max_z = s->kops->screened_step_postwarmup(
            D, s->D_pad,
            row,
            s->anchor.data(),
            s->floor_std.data(),
            s->mean.data(),
            s->var.data(),
            s->inv_std.data(),
            s->prev_norm.data(),
            s->ar_cov.data(),
            s->ar_var.data(),
            s->ar_res_var.data(),
            s->phi.data(),
            s->inv_res_std.data(),
            s->energy.data(),
            s->recent_peak.data(),
            s->last_z.data(),
            ema_alpha,
            energy_decay,
            peak_decay
        );
        s->max_z_ring[slot] = max_z;
        ++s->tick;
    }

    if (out_last_z_scores) {
        std::memcpy(out_last_z_scores, s->last_z.data(), D * sizeof(double));
    }

    if ((out_top_k_channels || out_top_k_scores) && top_k > 0) {
        s->sync_tier1();

        const uint32_t k_out = std::min(top_k, s->D);
        const uint32_t M = s->active_m;
        if (s->tier1_valid && s->tier1->tick > s->tier1->warmup_ticks) {
            cs_root_attribution(
                s->tier1, 0,
                s->t1_root_scores.data(), nullptr, nullptr, nullptr,
                s->t1_ranked.data()
            );

            const uint32_t k_t1 = std::min(k_out, M);
            for (uint32_t i = 0; i < k_t1; ++i) {
                const int32_t local_idx = s->t1_ranked[i];
                if (out_top_k_channels) out_top_k_channels[i] = s->active_channels[local_idx];
                if (out_top_k_scores)   out_top_k_scores[i] = s->t1_root_scores[local_idx];
            }
            for (uint32_t i = k_t1; i < k_out; ++i) {
                const int32_t ch = s->scratch_order[i];
                if (out_top_k_channels) out_top_k_channels[i] = ch;
                if (out_top_k_scores)   out_top_k_scores[i] = s->scratch_scores[ch];
            }
        } else {
            for (uint32_t i = 0; i < k_out; ++i) {
                const int32_t ch = s->scratch_order[i];
                if (out_top_k_channels) out_top_k_channels[i] = ch;
                if (out_top_k_scores)   out_top_k_scores[i] = s->scratch_scores[ch];
            }
        }
    }

    return 0;
}

int cs_screened_active_graph(cs_screened_t* s,
                             int32_t* out_active_channels,
                             double* out_granger_f,
                             double* out_pcorr,
                             uint32_t* out_active_count) {
    if (!s) return -1;
    s->sync_tier1();

    const uint32_t M = s->active_m;
    if (out_active_count) *out_active_count = s->tier1_valid ? M : 0;
    if (!s->tier1_valid) {
        if (out_granger_f) std::memset(out_granger_f, 0, static_cast<size_t>(M) * M * sizeof(double));
        if (out_pcorr)     std::memset(out_pcorr, 0, static_cast<size_t>(M) * M * sizeof(double));
        return 0;
    }

    if (out_active_channels) {
        std::memcpy(out_active_channels, s->active_channels.data(), M * sizeof(int32_t));
    }
    if (out_granger_f) {
        cs_granger_matrix(s->tier1, out_granger_f, nullptr);
    }
    if (out_pcorr) {
        cs_precision_matrix(s->tier1, nullptr, out_pcorr);
    }
    return 0;
}

} /* extern "C" */
