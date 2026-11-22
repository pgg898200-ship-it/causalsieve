#ifndef CAUSALSIEVE_CORE_HPP
#define CAUSALSIEVE_CORE_HPP

#include "causalsieve.h"
#include "causalsieve_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace causalsieve {

constexpr size_t kCacheLine = 64;

inline constexpr size_t pad4(size_t n) noexcept {
    return (n + 3u) & ~size_t(3u);
}

/* =========================================================================
 * 64-byte aligned RAII buffer (zero hot-path allocations, ASan/UBSan clean)
 * ========================================================================= */
template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(size_t count) {
        resize(count);
    }

    ~AlignedBuffer() {
        release();
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void resize(size_t count) {
        release();
        if (count == 0) return;
        const size_t bytes = count * sizeof(T);
        void* p = ::operator new(bytes, std::align_val_t{kCacheLine});
        data_ = static_cast<T*>(p);
        size_ = count;
        std::memset(data_, 0, bytes);
    }

    void zero() noexcept {
        if (data_ && size_ > 0) {
            std::memset(data_, 0, size_ * sizeof(T));
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }

    T& operator[](size_t i) noexcept { return data_[i]; }
    const T& operator[](size_t i) const noexcept { return data_[i]; }

private:
    void release() noexcept {
        if (data_) {
            ::operator delete(data_, std::align_val_t{kCacheLine});
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data_ = nullptr;
    size_t size_ = 0;
};

/* =========================================================================
 * Small Scalar Linear Algebra & Quantile Helpers
 * ========================================================================= */
namespace linalg {

/* Exact small m x m Cholesky (m <= 32) on upper-triangle input A_sub (row-major m x m). */
inline bool cholesky_small(const double* A_sub, double* L_sub, size_t m) noexcept {
    std::memset(L_sub, 0, m * m * sizeof(double));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < i; ++j) {
            double sum = A_sub[j * m + i];
            for (size_t k = 0; k < j; ++k) {
                sum -= L_sub[i * m + k] * L_sub[j * m + k];
            }
            L_sub[i * m + j] = sum / L_sub[j * m + j];
        }
        double diag = A_sub[i * m + i];
        for (size_t k = 0; k < i; ++k) {
            diag -= L_sub[i * m + k] * L_sub[i * m + k];
        }
        if (!(diag > 1e-30) || !std::isfinite(diag)) {
            diag = 1e-30;
        }
        L_sub[i * m + i] = std::sqrt(diag);
    }
    return true;
}

/* Compute exact z_bonf such that erfc(z / sqrt(2)) <= p_tail (i.e. P(|Z| > z) <= p_tail). */
inline double normal_two_sided_quantile(double p_tail) noexcept {
    double lo = 0.0, hi = 12.0;
    for (int iter = 0; iter < 60; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double p = std::erfc(mid * 0.7071067811865475244);
        if (p > p_tail) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return hi;
}

} /* namespace linalg */

} /* namespace causalsieve */

/* =========================================================================
 * Tier-1 Core Streaming Causal Engine (cs_engine)
 * ========================================================================= */
struct cs_engine {
    cs_config_t cfg{};
    const causalsieve::kernels::KernelOps* kops = nullptr;
    uint32_t D = 0;
    uint32_t p = 0;
    uint32_t K = 0;
    size_t D_pad = 0;
    size_t K_pad = 0;
    uint32_t warmup_ticks = 0;
    uint32_t refresh_interval = 0;
    uint32_t ring_cap = 0;

    uint64_t tick = 0;
    uint32_t refresh_count = 0;
    uint32_t ticks_since_refresh = 0;
    bool var_dirty_since_refresh = false;
    bool theta_dirty = false;
    bool baseline_frozen = false;
    bool has_refreshed_j = false;

    /* Anytime-valid ONS betting E-process state */
    double log_e_value = 0.0;
    double ons_bet = 0.1;
    double ons_A = 1.0;
    double z_bonf = 3.5;
    double z_inf_gate = 4.5;
    uint64_t incident_start_tick = UINT64_MAX;
    uint32_t consecutive_normal = 0;

    /* Lazy scale factors for exponential forgetting */
    double p_scale = 1.0;
    double g_scale = 1.0;
    double su_scale = 1.0;
    double theta_scale = 1.0;
    double lin_scale = 1.0;

    double W_eff_tilde = 0.0;
    double W_u_tilde = 0.0;
    double lin_W_tilde = 0.0;
    uint64_t n_var_updates = 0;
    uint64_t n_u_updates = 0;
    uint64_t lingam_warm_count = 0;
    uint64_t welford_n = 0;
    uint64_t passive_welford_n = 0;
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

    /* Shifted-anchor & baseline normalization buffers (size D_pad) */
    causalsieve::AlignedBuffer<double> anchor;
    causalsieve::AlignedBuffer<double> welford_mean;
    causalsieve::AlignedBuffer<double> welford_m2;
    causalsieve::AlignedBuffer<double> mean0;
    causalsieve::AlignedBuffer<double> base_var;
    causalsieve::AlignedBuffer<double> inv_scale;
    causalsieve::AlignedBuffer<double> floor_std;
    causalsieve::AlignedBuffer<double> passive_mean;
    causalsieve::AlignedBuffer<double> passive_m2;

    /* Lag history ring buffer (p x D_pad) */
    causalsieve::AlignedBuffer<double> lag_hist;
    uint32_t lag_head = 0;

    /* Core VAR(p) & Shared Inverse Gram state */
    causalsieve::AlignedBuffer<double> P_tilde;   /* K x K_pad (upper triangle) */
    causalsieve::AlignedBuffer<double> B;         /* K x D_pad */
    causalsieve::AlignedBuffer<double> G_tilde;   /* K x K_pad (upper triangle) */
    causalsieve::AlignedBuffer<double> C_tilde;   /* K x D_pad */
    causalsieve::AlignedBuffer<double> S_yy_tilde;/* D_pad */
    causalsieve::AlignedBuffer<double> J_rec;     /* D_pad (nonnegative recursive objective) */
    causalsieve::AlignedBuffer<double> J_exact;   /* D_pad (exact refreshed objective) */

    /* Innovation covariance & precision state */
    causalsieve::AlignedBuffer<double> S_u_tilde;   /* D x D_pad (upper triangle) */
    causalsieve::AlignedBuffer<double> S_u_diag;    /* D_pad (exact live diagonal of S_u_tilde) */
    causalsieve::AlignedBuffer<double> Theta_tilde; /* D x D_pad (upper triangle) */

    /* Online Hyvarinen-Smith LiNGAM state */
    causalsieve::AlignedBuffer<double> u_mean;
    causalsieve::AlignedBuffer<double> u_var;
    causalsieve::AlignedBuffer<double> M_u2v_tilde;    /* D x D_pad */
    causalsieve::AlignedBuffer<double> M_utanhv_tilde; /* D x D_pad */
    causalsieve::AlignedBuffer<double> M_uv_tilde;     /* D x D_pad (upper triangle) */

    /* Deferred L1-resident micro-batch accumulators (max kSlabBatchSize = 32) */
    size_t var_slab_len = 0;
    causalsieve::AlignedBuffer<double> var_z_slab;   /* B_SLAB x K_pad */
    causalsieve::AlignedBuffer<double> var_x_slab;   /* B_SLAB x D_pad */
    causalsieve::AlignedBuffer<double> var_w_slab;   /* B_SLAB */

    size_t su_slab_len = 0;
    causalsieve::AlignedBuffer<double> su_u_slab;    /* B_SLAB x D_pad */
    causalsieve::AlignedBuffer<double> su_w_slab;    /* B_SLAB */

    size_t lin_slab_len = 0;
    causalsieve::AlignedBuffer<double> lin_s_slab;   /* B_SLAB x D_pad */
    causalsieve::AlignedBuffer<double> lin_s2_slab;  /* B_SLAB x D_pad */
    causalsieve::AlignedBuffer<double> lin_ts_slab;  /* B_SLAB x D_pad */
    causalsieve::AlignedBuffer<double> lin_w_slab;   /* B_SLAB */

    /* Tick-major lossless raw ring & attribution rings (ring_cap x D) */
    causalsieve::AlignedBuffer<double> raw_ring;
    causalsieve::AlignedBuffer<double> exo_z_ring;
    causalsieve::AlignedBuffer<double> raw_z_ring;
    causalsieve::AlignedBuffer<double> uq_ring;      /* Exact DAG Mahalanobis energy u_j * (Theta u)_j */
    causalsieve::AlignedBuffer<double> innov_z_ring; /* Marginal temporal innovation z-scores |u_j| / sqrt(Sigma_u[j,j]) */

    /* Pre-allocated scratch buffers (ZERO hot allocations!) */
    causalsieve::AlignedBuffer<double> dx_buf;
    causalsieve::AlignedBuffer<double> x_work_buf;
    causalsieve::AlignedBuffer<double> raw_z_buf;
    causalsieve::AlignedBuffer<double> z_buf;
    causalsieve::AlignedBuffer<double> y_hat_buf;
    causalsieve::AlignedBuffer<double> u_buf;
    causalsieve::AlignedBuffer<double> q_tilde_buf;
    causalsieve::AlignedBuffer<double> q_buf;
    causalsieve::AlignedBuffer<double> exo_z_buf;
    causalsieve::AlignedBuffer<double> Pz_tilde_buf;
    causalsieve::AlignedBuffer<double> g_buf;
    causalsieve::AlignedBuffer<double> s_buf;
    causalsieve::AlignedBuffer<double> s2_buf;
    causalsieve::AlignedBuffer<double> ts_buf;

    /* Cholesky / Freivalds / Attribution scratch buffers */
    causalsieve::AlignedBuffer<double> chol_L_K;   /* K x K_pad */
    causalsieve::AlignedBuffer<double> chol_U_K;   /* K x K_pad */
    causalsieve::AlignedBuffer<double> chol_Y_KD;  /* K x D_pad */
    causalsieve::AlignedBuffer<double> chol_L_D;   /* D x D_pad */
    causalsieve::AlignedBuffer<double> chol_U_D;   /* D x D_pad */
    causalsieve::AlignedBuffer<double> probe_v;    /* K_pad */
    causalsieve::AlignedBuffer<double> probe_Pv;   /* K_pad */
    causalsieve::AlignedBuffer<double> probe_GPv;  /* K_pad */

    std::string last_error;

    void init_buffers();
    void reset_state();
    void push_lag_hist(const double* xw) noexcept;
    void build_feature_z(double* out_z) const noexcept;

    void flush_var_slab() noexcept;
    void flush_su_slab() noexcept;
    void flush_lingam_slab() noexcept;
    void flush_all_slabs() noexcept;

    void rescale_var_stats() noexcept;
    void rescale_su_stats() noexcept;
    void rescale_lingam_stats() noexcept;
    void update_lingam_moments(const double* u) noexcept;

    double eff_var_ridge() const noexcept {
        if (cfg.normalize_scale == 0) return cfg.ridge;
        const double W_eff = std::max(1.0, g_scale * W_eff_tilde);
        return std::max(cfg.ridge, 0.25 * static_cast<double>(K) / W_eff);
    }

    double eff_precision_ridge() const noexcept {
        if (cfg.normalize_scale == 0) return cfg.precision_ridge;
        const double W_u = std::max(1.0, su_scale * W_u_tilde);
        return std::max(cfg.precision_ridge, 0.05 + 0.65 * static_cast<double>(D) / W_u);
    }

    bool refresh_var_cholesky() noexcept;
    bool refresh_theta_cholesky() noexcept;
    double compute_freivalds_err() noexcept;
    void finalize_warmup_replay() noexcept;
    int push_single_validated(const double* x,
                              cs_tick_result_t* out,
                              double* out_exo_z,
                              double* out_innov) noexcept;
};

/* =========================================================================
 * Tier-2 High-Dimensional Screened Causal Engine (cs_screened)
 * ========================================================================= */
struct cs_screened {
    const causalsieve::kernels::KernelOps* kops = nullptr;
    uint32_t D = 0;
    uint32_t active_m = 0;
    uint32_t lag = 0;
    size_t D_pad = 0;
    double forgetting = 0.995;
    double ridge = 1e-2;
    uint32_t warmup_ticks = 40;
    uint32_t ring_cap = 512;

    uint64_t tick = 0;
    uint64_t tier1_synced_tick = 0;
    bool tier1_valid = false;
    bool baseline_frozen = false;

    /* Per-channel O(D) aligned state */
    causalsieve::AlignedBuffer<double> anchor;
    causalsieve::AlignedBuffer<double> floor_std;
    causalsieve::AlignedBuffer<double> w_mean;
    causalsieve::AlignedBuffer<double> w_m2;
    causalsieve::AlignedBuffer<double> mean;
    causalsieve::AlignedBuffer<double> var;
    causalsieve::AlignedBuffer<double> inv_std;
    causalsieve::AlignedBuffer<double> prev_norm;
    causalsieve::AlignedBuffer<double> ar_cov;
    causalsieve::AlignedBuffer<double> ar_var;
    causalsieve::AlignedBuffer<double> ar_res_var;
    causalsieve::AlignedBuffer<double> phi;
    causalsieve::AlignedBuffer<double> inv_res_std;
    causalsieve::AlignedBuffer<double> energy;
    causalsieve::AlignedBuffer<double> recent_peak;
    causalsieve::AlignedBuffer<double> last_z;

    /* Tick-major ring buffer for Top-M replay (ring_cap x D) */
    causalsieve::AlignedBuffer<double> raw_ring;
    causalsieve::AlignedBuffer<double> max_z_ring;

    /* Promoted Top-M active subgraph */
    std::vector<int32_t> active_channels;
    std::vector<int32_t> scratch_order;
    std::vector<double> scratch_scores;
    std::vector<double> scratch_sub_x;
    std::vector<double> t1_root_scores;
    std::vector<int32_t> t1_ranked;

    cs_engine_t* tier1 = nullptr;

    ~cs_screened() {
        if (tier1) {
            cs_destroy(tier1);
            tier1 = nullptr;
        }
    }

    void freeze_warmup_stats() noexcept;
    void sync_tier1() noexcept;
};

#endif /* CAUSALSIEVE_CORE_HPP */
