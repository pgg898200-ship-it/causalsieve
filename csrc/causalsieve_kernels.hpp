#ifndef CAUSALSIEVE_KERNELS_HPP
#define CAUSALSIEVE_KERNELS_HPP

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define CAUSALSIEVE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define CAUSALSIEVE_RESTRICT __restrict__
#else
#define CAUSALSIEVE_RESTRICT
#endif

namespace causalsieve {

constexpr size_t kSlabBatchSize = 32;

namespace kernels {

/* =========================================================================
 * Runtime-Dispatched SIMD Kernel Operations Table
 *
 * Compiled into separate translation units on x86_64:
 *   - causalsieve_kernels_scalar.cpp : portable baseline x86-64 (+ ARM64 NEON)
 *   - causalsieve_kernels_avx2.cpp   : -mavx2 -mfma / /arch:AVX2
 * Selected once at process startup via hardware CPUID + OSXSAVE + XGETBV
 * (with CAUSALSIEVE_FORCE_SCALAR=1 runtime override for exact parity testing).
 * ========================================================================= */
struct KernelOps {
    const char* isa_name;

    double (*dot_pd)(
        const double* CAUSALSIEVE_RESTRICT a,
        const double* CAUSALSIEVE_RESTRICT b,
        size_t n_pad) noexcept;

    void (*sym_matvec_upper)(
        const double* CAUSALSIEVE_RESTRICT A,
        const double* CAUSALSIEVE_RESTRICT x,
        double* CAUSALSIEVE_RESTRICT y,
        size_t n,
        size_t n_pad) noexcept;

    void (*sym_rank1_update_upper)(
        double* CAUSALSIEVE_RESTRICT A,
        const double* CAUSALSIEVE_RESTRICT v,
        double alpha,
        size_t n,
        size_t n_pad) noexcept;

    void (*matvec_BT_z)(
        const double* CAUSALSIEVE_RESTRICT B,
        const double* CAUSALSIEVE_RESTRICT z,
        double* CAUSALSIEVE_RESTRICT y,
        size_t K,
        size_t D,
        size_t D_pad) noexcept;

    void (*outer_update_rect)(
        double* CAUSALSIEVE_RESTRICT M,
        const double* CAUSALSIEVE_RESTRICT a,
        const double* CAUSALSIEVE_RESTRICT b,
        double alpha,
        size_t rows,
        size_t cols_pad) noexcept;

    void (*scale_vec)(
        const double* CAUSALSIEVE_RESTRICT src,
        double s,
        double* CAUSALSIEVE_RESTRICT dst,
        size_t n_pad) noexcept;

    void (*sub_vec)(
        const double* CAUSALSIEVE_RESTRICT a,
        const double* CAUSALSIEVE_RESTRICT b,
        double* CAUSALSIEVE_RESTRICT out,
        size_t n_pad) noexcept;

    bool (*all_finite_pd)(
        const double* ptr,
        size_t n) noexcept;

    /* Deferred micro-batched L1-resident register-blocked BLAS-3 kernels */
    void (*sym_rankk_update_upper)(
        double* CAUSALSIEVE_RESTRICT A,
        const double* CAUSALSIEVE_RESTRICT V,
        const double* CAUSALSIEVE_RESTRICT w,
        size_t b,
        size_t n,
        size_t n_pad) noexcept;

    void (*gemm_update_rect)(
        double* CAUSALSIEVE_RESTRICT M,
        const double* CAUSALSIEVE_RESTRICT A,
        const double* CAUSALSIEVE_RESTRICT B_mat,
        const double* CAUSALSIEVE_RESTRICT w,
        size_t b,
        size_t rows,
        size_t rows_pad,
        size_t cols_pad) noexcept;

    void (*lingam_batch_update)(
        double* CAUSALSIEVE_RESTRICT M_u2v,
        double* CAUSALSIEVE_RESTRICT M_utanhv,
        double* CAUSALSIEVE_RESTRICT M_uv,
        const double* CAUSALSIEVE_RESTRICT S,
        const double* CAUSALSIEVE_RESTRICT S2,
        const double* CAUSALSIEVE_RESTRICT TS,
        const double* CAUSALSIEVE_RESTRICT w,
        size_t b,
        size_t D,
        size_t D_pad) noexcept;

    /* Dense Cholesky factorization / inversion / multi-RHS solve */
    bool (*cholesky_factor_scaled_upper)(
        const double* CAUSALSIEVE_RESTRICT A_upper,
        double scale,
        double diag_ridge,
        double* CAUSALSIEVE_RESTRICT L,
        size_t n,
        size_t n_pad) noexcept;

    void (*cholesky_invert_to_upper)(
        const double* CAUSALSIEVE_RESTRICT L,
        double* CAUSALSIEVE_RESTRICT P_upper,
        double* CAUSALSIEVE_RESTRICT tmp_U,
        size_t n,
        size_t n_pad) noexcept;

    void (*cholesky_solve_multi)(
        const double* CAUSALSIEVE_RESTRICT L,
        const double* CAUSALSIEVE_RESTRICT C_tilde,
        double c_scale,
        double* CAUSALSIEVE_RESTRICT B,
        double* CAUSALSIEVE_RESTRICT Y,
        size_t K,
        size_t D,
        size_t K_pad,
        size_t D_pad) noexcept;

    /* Tier-2 O(D) screened pre-whitening kernel */
    double (*screened_step_postwarmup)(
        size_t D,
        size_t D_pad,
        const double* CAUSALSIEVE_RESTRICT x,
        const double* CAUSALSIEVE_RESTRICT anchor,
        const double* CAUSALSIEVE_RESTRICT floor_std,
        double* CAUSALSIEVE_RESTRICT mean,
        double* CAUSALSIEVE_RESTRICT var,
        double* CAUSALSIEVE_RESTRICT inv_std,
        double* CAUSALSIEVE_RESTRICT prev_norm,
        double* CAUSALSIEVE_RESTRICT ar_cov,
        double* CAUSALSIEVE_RESTRICT ar_var,
        double* CAUSALSIEVE_RESTRICT ar_res_var,
        double* CAUSALSIEVE_RESTRICT phi,
        double* CAUSALSIEVE_RESTRICT inv_res_std,
        double* CAUSALSIEVE_RESTRICT energy,
        double* CAUSALSIEVE_RESTRICT recent_peak,
        double* CAUSALSIEVE_RESTRICT out_z,
        double ema_alpha,
        double energy_decay,
        double peak_decay) noexcept;
};

const KernelOps& get_scalar_ops() noexcept;

#if defined(CAUSALSIEVE_HAS_AVX2_TU) || defined(CAUSALSIEVE_BUILD_AVX2)
const KernelOps& get_avx2_ops() noexcept;
#endif

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(CAUSALSIEVE_FORCE_SCALAR)
const KernelOps& get_neon_ops() noexcept;
#endif

/* Thread-safe once-initialized runtime ISA dispatch table. */
const KernelOps& active_ops() noexcept;

} /* namespace kernels */
} /* namespace causalsieve */

#endif /* CAUSALSIEVE_KERNELS_HPP */
