#include "causalsieve_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(CAUSALSIEVE_FORCE_SCALAR)
#define CAUSALSIEVE_COMPILE_NEON_IMPL 1
#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#else
#define CAUSALSIEVE_COMPILE_NEON_IMPL 0
#endif

namespace causalsieve::kernels {

/* =========================================================================
 * Portable Scalar Reference Kernels (zero SIMD instructions, 100% portable)
 * ========================================================================= */
namespace scalar_impl {

double dot_pd(const double* CAUSALSIEVE_RESTRICT a,
              const double* CAUSALSIEVE_RESTRICT b,
              size_t n_pad) noexcept {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 4 <= n_pad; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n_pad; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

void sym_matvec_upper(const double* CAUSALSIEVE_RESTRICT A,
                      const double* CAUSALSIEVE_RESTRICT x,
                      double* CAUSALSIEVE_RESTRICT y,
                      size_t n,
                      size_t n_pad) noexcept {
    std::memset(y, 0, n_pad * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        const double* CAUSALSIEVE_RESTRICT row = A + i * n_pad;
        const double xi = x[i];
        double sum = row[i] * xi;
        for (size_t j = i + 1; j < n; ++j) {
            const double aij = row[j];
            sum += aij * x[j];
            y[j] += aij * xi;
        }
        y[i] += sum;
    }
}

void sym_rank1_update_upper(double* CAUSALSIEVE_RESTRICT A,
                            const double* CAUSALSIEVE_RESTRICT v,
                            double alpha,
                            size_t n,
                            size_t n_pad) noexcept {
    for (size_t i = 0; i < n; ++i) {
        double* CAUSALSIEVE_RESTRICT row = A + i * n_pad;
        const double s = alpha * v[i];
        for (size_t j = i; j < n; ++j) {
            row[j] += s * v[j];
        }
    }
}

void matvec_BT_z(const double* CAUSALSIEVE_RESTRICT B,
                 const double* CAUSALSIEVE_RESTRICT z,
                 double* CAUSALSIEVE_RESTRICT y,
                 size_t K,
                 size_t D,
                 size_t D_pad) noexcept {
    std::memset(y, 0, D_pad * sizeof(double));
    for (size_t i = 0; i < K; ++i) {
        const double* CAUSALSIEVE_RESTRICT row = B + i * D_pad;
        const double zi = z[i];
        for (size_t j = 0; j < D; ++j) {
            y[j] += zi * row[j];
        }
    }
}

void outer_update_rect(double* CAUSALSIEVE_RESTRICT M,
                       const double* CAUSALSIEVE_RESTRICT a,
                       const double* CAUSALSIEVE_RESTRICT b,
                       double alpha,
                       size_t rows,
                       size_t cols_pad) noexcept {
    for (size_t i = 0; i < rows; ++i) {
        double* CAUSALSIEVE_RESTRICT r0 = M + i * cols_pad;
        const double ai = alpha * a[i];
        for (size_t j = 0; j < cols_pad; ++j) {
            r0[j] += ai * b[j];
        }
    }
}

void scale_vec(const double* CAUSALSIEVE_RESTRICT src,
               double s,
               double* CAUSALSIEVE_RESTRICT dst,
               size_t n_pad) noexcept {
    for (size_t i = 0; i < n_pad; ++i) dst[i] = src[i] * s;
}

void sub_vec(const double* CAUSALSIEVE_RESTRICT a,
             const double* CAUSALSIEVE_RESTRICT b,
             double* CAUSALSIEVE_RESTRICT out,
             size_t n_pad) noexcept {
    for (size_t i = 0; i < n_pad; ++i) out[i] = a[i] - b[i];
}

bool all_finite_pd(const double* ptr, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(ptr[i])) return false;
    }
    return true;
}

void sym_rankk_update_upper(double* CAUSALSIEVE_RESTRICT A,
                            const double* CAUSALSIEVE_RESTRICT V,
                            const double* CAUSALSIEVE_RESTRICT w,
                            size_t b,
                            size_t n,
                            size_t n_pad) noexcept {
    if (b == 0 || n == 0) return;
    for (size_t r = 0; r < b; ++r) {
        const double* CAUSALSIEVE_RESTRICT vr = V + r * n_pad;
        const double wr = w[r];
        for (size_t i = 0; i < n; ++i) {
            double* CAUSALSIEVE_RESTRICT row = A + i * n_pad;
            const double s = wr * vr[i];
            for (size_t j = i; j < n; ++j) {
                row[j] += s * vr[j];
            }
        }
    }
}

void gemm_update_rect(double* CAUSALSIEVE_RESTRICT M,
                      const double* CAUSALSIEVE_RESTRICT A,
                      const double* CAUSALSIEVE_RESTRICT B_mat,
                      const double* CAUSALSIEVE_RESTRICT w,
                      size_t b,
                      size_t rows,
                      size_t rows_pad,
                      size_t cols_pad) noexcept {
    if (b == 0 || rows == 0 || cols_pad == 0) return;
    for (size_t r = 0; r < b; ++r) {
        const double* CAUSALSIEVE_RESTRICT ar = A + r * rows_pad;
        const double* CAUSALSIEVE_RESTRICT br = B_mat + r * cols_pad;
        const double wr = w[r];
        for (size_t i = 0; i < rows; ++i) {
            double* CAUSALSIEVE_RESTRICT row = M + i * cols_pad;
            const double s = wr * ar[i];
            for (size_t j = 0; j < cols_pad; ++j) {
                row[j] += s * br[j];
            }
        }
    }
}

void lingam_batch_update(double* CAUSALSIEVE_RESTRICT M_u2v,
                         double* CAUSALSIEVE_RESTRICT M_utanhv,
                         double* CAUSALSIEVE_RESTRICT M_uv,
                         const double* CAUSALSIEVE_RESTRICT S,
                         const double* CAUSALSIEVE_RESTRICT S2,
                         const double* CAUSALSIEVE_RESTRICT TS,
                         const double* CAUSALSIEVE_RESTRICT w,
                         size_t b,
                         size_t D,
                         size_t D_pad) noexcept {
    if (b == 0 || D == 0) return;
    for (size_t r = 0; r < b; ++r) {
        const double* CAUSALSIEVE_RESTRICT sr  = S  + r * D_pad;
        const double* CAUSALSIEVE_RESTRICT s2r = S2 + r * D_pad;
        const double* CAUSALSIEVE_RESTRICT tsr = TS + r * D_pad;
        const double wr = w[r];
        for (size_t i = 0; i < D; ++i) {
            double* CAUSALSIEVE_RESTRICT r_u2v = M_u2v    + i * D_pad;
            double* CAUSALSIEVE_RESTRICT r_ut  = M_utanhv + i * D_pad;
            double* CAUSALSIEVE_RESTRICT r_uv  = M_uv     + i * D_pad;
            const double ws2 = wr * s2r[i];
            const double ws  = wr * sr[i];
            for (size_t j = 0; j < D; ++j) {
                r_u2v[j] += ws2 * sr[j];
                r_ut[j]  += ws  * tsr[j];
            }
            for (size_t j = i; j < D; ++j) {
                r_uv[j] += ws * sr[j];
            }
        }
    }
}

bool cholesky_factor_scaled_upper(const double* CAUSALSIEVE_RESTRICT A_upper,
                                  double scale,
                                  double diag_ridge,
                                  double* CAUSALSIEVE_RESTRICT L,
                                  size_t n,
                                  size_t n_pad) noexcept {
    std::memset(L, 0, n * n_pad * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double* CAUSALSIEVE_RESTRICT Li = L + i * n_pad;
        for (size_t j = 0; j < i; ++j) {
            const double* CAUSALSIEVE_RESTRICT Lj = L + j * n_pad;
            double sum = scale * A_upper[j * n_pad + i];
            for (size_t k = 0; k < j; ++k) {
                sum -= Li[k] * Lj[k];
            }
            Li[j] = sum / Lj[j];
        }
        double diag = scale * A_upper[i * n_pad + i] + diag_ridge;
        for (size_t k = 0; k < i; ++k) {
            diag -= Li[k] * Li[k];
        }
        if (!(diag > 0.0) || !std::isfinite(diag)) {
            if (diag > -1e-12) {
                diag = 1e-15;
            } else {
                return false;
            }
        }
        Li[i] = std::sqrt(diag);
    }
    return true;
}

void cholesky_invert_to_upper(const double* CAUSALSIEVE_RESTRICT L,
                              double* CAUSALSIEVE_RESTRICT P_upper,
                              double* CAUSALSIEVE_RESTRICT tmp_U,
                              size_t n,
                              size_t n_pad) noexcept {
    std::memset(tmp_U, 0, n * n_pad * sizeof(double));
    std::memset(P_upper, 0, n * n_pad * sizeof(double));

    for (size_t j = 0; j < n; ++j) {
        double* CAUSALSIEVE_RESTRICT Uj = tmp_U + j * n_pad;
        Uj[j] = 1.0 / L[j * n_pad + j];
        for (size_t i = j + 1; i < n; ++i) {
            const double* CAUSALSIEVE_RESTRICT Li = L + i * n_pad;
            double sum = 0.0;
            for (size_t k = j; k < i; ++k) {
                sum += Li[k] * Uj[k];
            }
            Uj[i] = -sum / Li[i];
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const double* CAUSALSIEVE_RESTRICT Ui = tmp_U + i * n_pad;
        double* CAUSALSIEVE_RESTRICT Pi = P_upper + i * n_pad;
        for (size_t j = i; j < n; ++j) {
            const double* CAUSALSIEVE_RESTRICT Uj = tmp_U + j * n_pad;
            double sum = 0.0;
            for (size_t k = j; k < n; ++k) {
                sum += Ui[k] * Uj[k];
            }
            Pi[j] = sum;
        }
    }
}

void cholesky_solve_multi(const double* CAUSALSIEVE_RESTRICT L,
                          const double* CAUSALSIEVE_RESTRICT C_tilde,
                          double c_scale,
                          double* CAUSALSIEVE_RESTRICT B,
                          double* CAUSALSIEVE_RESTRICT Y,
                          size_t K,
                          size_t D,
                          size_t K_pad,
                          size_t D_pad) noexcept {
    for (size_t i = 0; i < K; ++i) {
        const double* CAUSALSIEVE_RESTRICT Li = L + i * K_pad;
        const double* CAUSALSIEVE_RESTRICT Ci = C_tilde + i * D_pad;
        double* CAUSALSIEVE_RESTRICT Yi = Y + i * D_pad;
        for (size_t j = 0; j < D; ++j) {
            Yi[j] = c_scale * Ci[j];
        }
        for (size_t k = 0; k < i; ++k) {
            const double lik = Li[k];
            const double* CAUSALSIEVE_RESTRICT Yk = Y + k * D_pad;
            for (size_t j = 0; j < D; ++j) {
                Yi[j] -= lik * Yk[j];
            }
        }
        const double inv_d = 1.0 / Li[i];
        for (size_t j = 0; j < D; ++j) {
            Yi[j] *= inv_d;
        }
        for (size_t j = D; j < D_pad; ++j) Yi[j] = 0.0;
    }

    for (size_t ii = K; ii > 0; --ii) {
        const size_t i = ii - 1;
        double* CAUSALSIEVE_RESTRICT Bi = B + i * D_pad;
        const double* CAUSALSIEVE_RESTRICT Yi = Y + i * D_pad;
        for (size_t j = 0; j < D; ++j) {
            Bi[j] = Yi[j];
        }
        for (size_t k = i + 1; k < K; ++k) {
            const double lki = L[k * K_pad + i];
            const double* CAUSALSIEVE_RESTRICT Bk = B + k * D_pad;
            for (size_t j = 0; j < D; ++j) {
                Bi[j] -= lki * Bk[j];
            }
        }
        const double inv_d = 1.0 / L[i * K_pad + i];
        for (size_t j = 0; j < D; ++j) {
            Bi[j] *= inv_d;
        }
        for (size_t j = D; j < D_pad; ++j) Bi[j] = 0.0;
    }
}

double screened_step_postwarmup(
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
    double peak_decay
) noexcept {
    (void)D_pad;
    double max_abs_z = 0.0;
    for (size_t i = 0; i < D; ++i) {
        const double dx = x[i] - anchor[i];
        const double diff = dx - mean[i];
        const double xnorm = diff * inv_std[i];
        const double res = xnorm - phi[i] * prev_norm[i];
        const double z = res * inv_res_std[i];
        out_z[i] = z;

        const double az = std::fabs(z);
        if (az > max_abs_z) max_abs_z = az;

        const double ex = (az > 2.5) ? (az - 2.5) : 0.0;
        energy[i] = energy_decay * energy[i] + ex * ex;
        recent_peak[i] = std::max(peak_decay * recent_peak[i], az);

        if (az <= 3.5 && std::fabs(xnorm) <= 3.5) {
            mean[i] += ema_alpha * diff;
            var[i] = (1.0 - ema_alpha) * var[i] + ema_alpha * diff * diff;
            const double s = std::max(std::sqrt(std::max(0.0, var[i])), floor_std[i]);
            inv_std[i] = 1.0 / s;

            ar_cov[i] = (1.0 - ema_alpha) * ar_cov[i] + ema_alpha * xnorm * prev_norm[i];
            ar_var[i] = (1.0 - ema_alpha) * ar_var[i] + ema_alpha * prev_norm[i] * prev_norm[i];
            ar_res_var[i] = (1.0 - ema_alpha) * ar_res_var[i] + ema_alpha * res * res;
            phi[i] = std::clamp(ar_cov[i] / std::max(ar_var[i], 1e-6), -0.98, 0.98);
            inv_res_std[i] = 1.0 / std::sqrt(std::max(ar_res_var[i], 1e-4));
        }
        prev_norm[i] = std::clamp(xnorm, -6.0, 6.0);
    }
    return max_abs_z;
}

} /* namespace scalar_impl */

const KernelOps& get_scalar_ops() noexcept {
    static const KernelOps kOps{
        "scalar",
        &scalar_impl::dot_pd,
        &scalar_impl::sym_matvec_upper,
        &scalar_impl::sym_rank1_update_upper,
        &scalar_impl::matvec_BT_z,
        &scalar_impl::outer_update_rect,
        &scalar_impl::scale_vec,
        &scalar_impl::sub_vec,
        &scalar_impl::all_finite_pd,
        &scalar_impl::sym_rankk_update_upper,
        &scalar_impl::gemm_update_rect,
        &scalar_impl::lingam_batch_update,
        &scalar_impl::cholesky_factor_scaled_upper,
        &scalar_impl::cholesky_invert_to_upper,
        &scalar_impl::cholesky_solve_multi,
        &scalar_impl::screened_step_postwarmup,
    };
    return kOps;
}

/* =========================================================================
 * ARM64 NEON Kernels (zero-overhead baseline SIMD on AArch64 / Apple Silicon)
 * ========================================================================= */
#if CAUSALSIEVE_COMPILE_NEON_IMPL
namespace neon_impl {

double dot_pd(const double* CAUSALSIEVE_RESTRICT a,
              const double* CAUSALSIEVE_RESTRICT b,
              size_t n_pad) noexcept {
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);
    size_t i = 0;
    for (; i + 8 <= n_pad; i += 8) {
        acc0 = vfmaq_f64(acc0, vld1q_f64(a + i),     vld1q_f64(b + i));
        acc1 = vfmaq_f64(acc1, vld1q_f64(a + i + 2), vld1q_f64(b + i + 2));
        acc2 = vfmaq_f64(acc2, vld1q_f64(a + i + 4), vld1q_f64(b + i + 4));
        acc3 = vfmaq_f64(acc3, vld1q_f64(a + i + 6), vld1q_f64(b + i + 6));
    }
    for (; i + 2 <= n_pad; i += 2) {
        acc0 = vfmaq_f64(acc0, vld1q_f64(a + i), vld1q_f64(b + i));
    }
    acc0 = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
    return vaddvq_f64(acc0);
}

/* 4-row unrolled upper-triangle symmetric matvec (NEON). */
void sym_matvec_upper(const double* CAUSALSIEVE_RESTRICT A,
                      const double* CAUSALSIEVE_RESTRICT x,
                      double* CAUSALSIEVE_RESTRICT y,
                      size_t n,
                      size_t n_pad) noexcept {
    std::memset(y, 0, n_pad * sizeof(double));

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const double* CAUSALSIEVE_RESTRICT r0 = A + (i + 0) * n_pad;
        const double* CAUSALSIEVE_RESTRICT r1 = A + (i + 1) * n_pad;
        const double* CAUSALSIEVE_RESTRICT r2 = A + (i + 2) * n_pad;
        const double* CAUSALSIEVE_RESTRICT r3 = A + (i + 3) * n_pad;

        const double x0 = x[i + 0], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
        const double a01 = r0[i + 1], a02 = r0[i + 2], a03 = r0[i + 3];
        const double a12 = r1[i + 2], a13 = r1[i + 3];
        const double a23 = r2[i + 3];

        const double s0 = r0[i + 0] * x0 + a01 * x1 + a02 * x2 + a03 * x3;
        const double s1 = a01 * x0 + r1[i + 1] * x1 + a12 * x2 + a13 * x3;
        const double s2 = a02 * x0 + a12 * x1 + r2[i + 2] * x2 + a23 * x3;
        const double s3 = a03 * x0 + a13 * x1 + a23 * x2 + r3[i + 3] * x3;

        const float64x2_t vx0 = vdupq_n_f64(x0);
        const float64x2_t vx1 = vdupq_n_f64(x1);
        const float64x2_t vx2 = vdupq_n_f64(x2);
        const float64x2_t vx3 = vdupq_n_f64(x3);

        float64x2_t acc0 = vdupq_n_f64(0.0);
        float64x2_t acc1 = vdupq_n_f64(0.0);
        float64x2_t acc2 = vdupq_n_f64(0.0);
        float64x2_t acc3 = vdupq_n_f64(0.0);

        size_t j = i + 4;
        for (; j + 4 <= n_pad; j += 4) {
            const float64x2_t xj0 = vld1q_f64(x + j);
            const float64x2_t xj1 = vld1q_f64(x + j + 2);
            float64x2_t yj0 = vld1q_f64(y + j);
            float64x2_t yj1 = vld1q_f64(y + j + 2);

            const float64x2_t a0_0 = vld1q_f64(r0 + j);
            const float64x2_t a0_1 = vld1q_f64(r0 + j + 2);
            acc0 = vfmaq_f64(acc0, a0_0, xj0);
            acc0 = vfmaq_f64(acc0, a0_1, xj1);
            yj0  = vfmaq_f64(yj0,  a0_0, vx0);
            yj1  = vfmaq_f64(yj1,  a0_1, vx0);

            const float64x2_t a1_0 = vld1q_f64(r1 + j);
            const float64x2_t a1_1 = vld1q_f64(r1 + j + 2);
            acc1 = vfmaq_f64(acc1, a1_0, xj0);
            acc1 = vfmaq_f64(acc1, a1_1, xj1);
            yj0  = vfmaq_f64(yj0,  a1_0, vx1);
            yj1  = vfmaq_f64(yj1,  a1_1, vx1);

            const float64x2_t a2_0 = vld1q_f64(r2 + j);
            const float64x2_t a2_1 = vld1q_f64(r2 + j + 2);
            acc2 = vfmaq_f64(acc2, a2_0, xj0);
            acc2 = vfmaq_f64(acc2, a2_1, xj1);
            yj0  = vfmaq_f64(yj0,  a2_0, vx2);
            yj1  = vfmaq_f64(yj1,  a2_1, vx2);

            const float64x2_t a3_0 = vld1q_f64(r3 + j);
            const float64x2_t a3_1 = vld1q_f64(r3 + j + 2);
            acc3 = vfmaq_f64(acc3, a3_0, xj0);
            acc3 = vfmaq_f64(acc3, a3_1, xj1);
            yj0  = vfmaq_f64(yj0,  a3_0, vx3);
            yj1  = vfmaq_f64(yj1,  a3_1, vx3);

            vst1q_f64(y + j,     yj0);
            vst1q_f64(y + j + 2, yj1);
        }
        for (; j + 2 <= n_pad; j += 2) {
            const float64x2_t xj0 = vld1q_f64(x + j);
            float64x2_t yj0 = vld1q_f64(y + j);

            const float64x2_t a0_0 = vld1q_f64(r0 + j);
            acc0 = vfmaq_f64(acc0, a0_0, xj0);
            yj0  = vfmaq_f64(yj0,  a0_0, vx0);

            const float64x2_t a1_0 = vld1q_f64(r1 + j);
            acc1 = vfmaq_f64(acc1, a1_0, xj0);
            yj0  = vfmaq_f64(yj0,  a1_0, vx1);

            const float64x2_t a2_0 = vld1q_f64(r2 + j);
            acc2 = vfmaq_f64(acc2, a2_0, xj0);
            yj0  = vfmaq_f64(yj0,  a2_0, vx2);

            const float64x2_t a3_0 = vld1q_f64(r3 + j);
            acc3 = vfmaq_f64(acc3, a3_0, xj0);
            yj0  = vfmaq_f64(yj0,  a3_0, vx3);

            vst1q_f64(y + j, yj0);
        }

        y[i + 0] += s0 + vaddvq_f64(acc0);
        y[i + 1] += s1 + vaddvq_f64(acc1);
        y[i + 2] += s2 + vaddvq_f64(acc2);
        y[i + 3] += s3 + vaddvq_f64(acc3);
    }

    for (; i < n; ++i) {
        const double* CAUSALSIEVE_RESTRICT row = A + i * n_pad;
        const double xi = x[i];
        double sum = row[i] * xi;
        for (size_t j = i + 1; j < n; ++j) {
            const double aij = row[j];
            sum += aij * x[j];
            y[j] += aij * xi;
        }
        y[i] += sum;
    }
    for (size_t k = n; k < n_pad; ++k) y[k] = 0.0;
}

/* 4-row unrolled upper-triangle symmetric rank-1 update (NEON). */
void sym_rank1_update_upper(double* CAUSALSIEVE_RESTRICT A,
                            const double* CAUSALSIEVE_RESTRICT v,
                            double alpha,
                            size_t n,
                            size_t n_pad) noexcept {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        double* CAUSALSIEVE_RESTRICT r0 = A + (i + 0) * n_pad;
        double* CAUSALSIEVE_RESTRICT r1 = A + (i + 1) * n_pad;
        double* CAUSALSIEVE_RESTRICT r2 = A + (i + 2) * n_pad;
        double* CAUSALSIEVE_RESTRICT r3 = A + (i + 3) * n_pad;

        const float64x2_t vs0 = vdupq_n_f64(alpha * v[i + 0]);
        const float64x2_t vs1 = vdupq_n_f64(alpha * v[i + 1]);
        const float64x2_t vs2 = vdupq_n_f64(alpha * v[i + 2]);
        const float64x2_t vs3 = vdupq_n_f64(alpha * v[i + 3]);

        size_t j = i;
        for (; j + 4 <= n_pad; j += 4) {
            const float64x2_t v0 = vld1q_f64(v + j);
            const float64x2_t v1 = vld1q_f64(v + j + 2);
            vst1q_f64(r0 + j,     vfmaq_f64(vld1q_f64(r0 + j),     vs0, v0));
            vst1q_f64(r0 + j + 2, vfmaq_f64(vld1q_f64(r0 + j + 2), vs0, v1));
            vst1q_f64(r1 + j,     vfmaq_f64(vld1q_f64(r1 + j),     vs1, v0));
            vst1q_f64(r1 + j + 2, vfmaq_f64(vld1q_f64(r1 + j + 2), vs1, v1));
            vst1q_f64(r2 + j,     vfmaq_f64(vld1q_f64(r2 + j),     vs2, v0));
            vst1q_f64(r2 + j + 2, vfmaq_f64(vld1q_f64(r2 + j + 2), vs2, v1));
            vst1q_f64(r3 + j,     vfmaq_f64(vld1q_f64(r3 + j),     vs3, v0));
            vst1q_f64(r3 + j + 2, vfmaq_f64(vld1q_f64(r3 + j + 2), vs3, v1));
        }
        for (; j + 2 <= n_pad; j += 2) {
            const float64x2_t v0 = vld1q_f64(v + j);
            vst1q_f64(r0 + j, vfmaq_f64(vld1q_f64(r0 + j), vs0, v0));
            vst1q_f64(r1 + j, vfmaq_f64(vld1q_f64(r1 + j), vs1, v0));
            vst1q_f64(r2 + j, vfmaq_f64(vld1q_f64(r2 + j), vs2, v0));
            vst1q_f64(r3 + j, vfmaq_f64(vld1q_f64(r3 + j), vs3, v0));
        }
    }
    if (i < n) {
        const size_t j0 = i;
        const float64x2_t v0 = vld1q_f64(v + j0);
        const float64x2_t v1 = vld1q_f64(v + j0 + 2);
        for (; i < n; ++i) {
            double* CAUSALSIEVE_RESTRICT row = A + i * n_pad;
            const float64x2_t vs = vdupq_n_f64(alpha * v[i]);
            vst1q_f64(row + j0,     vfmaq_f64(vld1q_f64(row + j0),     vs, v0));
            vst1q_f64(row + j0 + 2, vfmaq_f64(vld1q_f64(row + j0 + 2), vs, v1));
        }
    }
}

void matvec_BT_z(const double* CAUSALSIEVE_RESTRICT B,
                 const double* CAUSALSIEVE_RESTRICT z,
                 double* CAUSALSIEVE_RESTRICT y,
                 size_t K,
                 size_t D,
                 size_t D_pad) noexcept {
    size_t j = 0;
    for (; j + 8 <= D_pad; j += 8) {
        float64x2_t y0 = vdupq_n_f64(0.0);
        float64x2_t y1 = vdupq_n_f64(0.0);
        float64x2_t y2 = vdupq_n_f64(0.0);
        float64x2_t y3 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < K; ++i) {
            const double* CAUSALSIEVE_RESTRICT row = B + i * D_pad + j;
            const float64x2_t vz = vdupq_n_f64(z[i]);
            y0 = vfmaq_f64(y0, vz, vld1q_f64(row));
            y1 = vfmaq_f64(y1, vz, vld1q_f64(row + 2));
            y2 = vfmaq_f64(y2, vz, vld1q_f64(row + 4));
            y3 = vfmaq_f64(y3, vz, vld1q_f64(row + 6));
        }
        vst1q_f64(y + j,     y0);
        vst1q_f64(y + j + 2, y1);
        vst1q_f64(y + j + 4, y2);
        vst1q_f64(y + j + 6, y3);
    }
    for (; j + 4 <= D_pad; j += 4) {
        float64x2_t y0 = vdupq_n_f64(0.0);
        float64x2_t y1 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < K; ++i) {
            const double* CAUSALSIEVE_RESTRICT row = B + i * D_pad + j;
            const float64x2_t vz = vdupq_n_f64(z[i]);
            y0 = vfmaq_f64(y0, vz, vld1q_f64(row));
            y1 = vfmaq_f64(y1, vz, vld1q_f64(row + 2));
        }
        vst1q_f64(y + j,     y0);
        vst1q_f64(y + j + 2, y1);
    }
    for (; j + 2 <= D_pad; j += 2) {
        float64x2_t y0 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < K; ++i) {
            const float64x2_t vz = vdupq_n_f64(z[i]);
            y0 = vfmaq_f64(y0, vz, vld1q_f64(B + i * D_pad + j));
        }
        vst1q_f64(y + j, y0);
    }
    for (size_t k = D; k < D_pad; ++k) y[k] = 0.0;
}

/* 4-row unrolled rectangular outer update (NEON). */
void outer_update_rect(double* CAUSALSIEVE_RESTRICT M,
                       const double* CAUSALSIEVE_RESTRICT a,
                       const double* CAUSALSIEVE_RESTRICT b,
                       double alpha,
                       size_t rows,
                       size_t cols_pad) noexcept {
    size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        double* CAUSALSIEVE_RESTRICT r0 = M + (i + 0) * cols_pad;
        double* CAUSALSIEVE_RESTRICT r1 = M + (i + 1) * cols_pad;
        double* CAUSALSIEVE_RESTRICT r2 = M + (i + 2) * cols_pad;
        double* CAUSALSIEVE_RESTRICT r3 = M + (i + 3) * cols_pad;
        const float64x2_t va0 = vdupq_n_f64(alpha * a[i + 0]);
        const float64x2_t va1 = vdupq_n_f64(alpha * a[i + 1]);
        const float64x2_t va2 = vdupq_n_f64(alpha * a[i + 2]);
        const float64x2_t va3 = vdupq_n_f64(alpha * a[i + 3]);

        size_t j = 0;
        for (; j + 4 <= cols_pad; j += 4) {
            const float64x2_t b0 = vld1q_f64(b + j);
            const float64x2_t b1 = vld1q_f64(b + j + 2);
            vst1q_f64(r0 + j,     vfmaq_f64(vld1q_f64(r0 + j),     va0, b0));
            vst1q_f64(r0 + j + 2, vfmaq_f64(vld1q_f64(r0 + j + 2), va0, b1));
            vst1q_f64(r1 + j,     vfmaq_f64(vld1q_f64(r1 + j),     va1, b0));
            vst1q_f64(r1 + j + 2, vfmaq_f64(vld1q_f64(r1 + j + 2), va1, b1));
            vst1q_f64(r2 + j,     vfmaq_f64(vld1q_f64(r2 + j),     va2, b0));
            vst1q_f64(r2 + j + 2, vfmaq_f64(vld1q_f64(r2 + j + 2), va2, b1));
            vst1q_f64(r3 + j,     vfmaq_f64(vld1q_f64(r3 + j),     va3, b0));
            vst1q_f64(r3 + j + 2, vfmaq_f64(vld1q_f64(r3 + j + 2), va3, b1));
        }
        for (; j + 2 <= cols_pad; j += 2) {
            const float64x2_t b0 = vld1q_f64(b + j);
            vst1q_f64(r0 + j, vfmaq_f64(vld1q_f64(r0 + j), va0, b0));
            vst1q_f64(r1 + j, vfmaq_f64(vld1q_f64(r1 + j), va1, b0));
            vst1q_f64(r2 + j, vfmaq_f64(vld1q_f64(r2 + j), va2, b0));
            vst1q_f64(r3 + j, vfmaq_f64(vld1q_f64(r3 + j), va3, b0));
        }
    }
    for (; i < rows; ++i) {
        double* CAUSALSIEVE_RESTRICT r0 = M + i * cols_pad;
        const float64x2_t va0 = vdupq_n_f64(alpha * a[i]);
        for (size_t j = 0; j + 2 <= cols_pad; j += 2) {
            const float64x2_t b0 = vld1q_f64(b + j);
            vst1q_f64(r0 + j, vfmaq_f64(vld1q_f64(r0 + j), va0, b0));
        }
    }
}

void scale_vec(const double* CAUSALSIEVE_RESTRICT src,
               double s,
               double* CAUSALSIEVE_RESTRICT dst,
               size_t n_pad) noexcept {
    const float64x2_t vs = vdupq_n_f64(s);
    size_t i = 0;
    for (; i + 8 <= n_pad; i += 8) {
        vst1q_f64(dst + i,     vmulq_f64(vld1q_f64(src + i),     vs));
        vst1q_f64(dst + i + 2, vmulq_f64(vld1q_f64(src + i + 2), vs));
        vst1q_f64(dst + i + 4, vmulq_f64(vld1q_f64(src + i + 4), vs));
        vst1q_f64(dst + i + 6, vmulq_f64(vld1q_f64(src + i + 6), vs));
    }
    for (; i + 2 <= n_pad; i += 2) {
        vst1q_f64(dst + i, vmulq_f64(vld1q_f64(src + i), vs));
    }
}

void sub_vec(const double* CAUSALSIEVE_RESTRICT a,
             const double* CAUSALSIEVE_RESTRICT b,
             double* CAUSALSIEVE_RESTRICT out,
             size_t n_pad) noexcept {
    size_t i = 0;
    for (; i + 8 <= n_pad; i += 8) {
        vst1q_f64(out + i,     vsubq_f64(vld1q_f64(a + i),     vld1q_f64(b + i)));
        vst1q_f64(out + i + 2, vsubq_f64(vld1q_f64(a + i + 2), vld1q_f64(b + i + 2)));
        vst1q_f64(out + i + 4, vsubq_f64(vld1q_f64(a + i + 4), vld1q_f64(b + i + 4)));
        vst1q_f64(out + i + 6, vsubq_f64(vld1q_f64(a + i + 6), vld1q_f64(b + i + 6)));
    }
    for (; i + 2 <= n_pad; i += 2) {
        vst1q_f64(out + i, vsubq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
    }
}

bool all_finite_pd(const double* ptr, size_t n) noexcept {
    const uint64x2_t exp_mask = vdupq_n_u64(0x7FF0000000000000ULL);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint64x2_t v0 = vandq_u64(vreinterpretq_u64_f64(vld1q_f64(ptr + i)),     exp_mask);
        const uint64x2_t v1 = vandq_u64(vreinterpretq_u64_f64(vld1q_f64(ptr + i + 2)), exp_mask);
        const uint64x2_t v2 = vandq_u64(vreinterpretq_u64_f64(vld1q_f64(ptr + i + 4)), exp_mask);
        const uint64x2_t v3 = vandq_u64(vreinterpretq_u64_f64(vld1q_f64(ptr + i + 6)), exp_mask);
        const uint64x2_t bad = vorrq_u64(
            vorrq_u64(vceqq_u64(v0, exp_mask), vceqq_u64(v1, exp_mask)),
            vorrq_u64(vceqq_u64(v2, exp_mask), vceqq_u64(v3, exp_mask)));
        if ((vgetq_lane_u64(bad, 0) | vgetq_lane_u64(bad, 1)) != 0) return false;
    }
    for (; i + 2 <= n; i += 2) {
        const uint64x2_t v0 = vandq_u64(vreinterpretq_u64_f64(vld1q_f64(ptr + i)), exp_mask);
        const uint64x2_t bad = vceqq_u64(v0, exp_mask);
        if ((vgetq_lane_u64(bad, 0) | vgetq_lane_u64(bad, 1)) != 0) return false;
    }
    for (; i < n; ++i) {
        if (!std::isfinite(ptr[i])) return false;
    }
    return true;
}

/* L1-resident 4x4 register-blocked NEON micro-kernel using vfmaq_laneq_f64. */
template <size_t MR>
inline void microkernel_mr_x_4_neon(double* CAUSALSIEVE_RESTRICT C,
                                    size_t ldc,
                                    const double (*CAUSALSIEVE_RESTRICT Ap)[4],
                                    const double (*CAUSALSIEVE_RESTRICT Bp)[4],
                                    size_t rb) noexcept {
    float64x2_t c00 = vld1q_f64(C + 0 * ldc + 0);
    float64x2_t c01 = vld1q_f64(C + 0 * ldc + 2);
    float64x2_t c10 = vdupq_n_f64(0.0), c11 = vdupq_n_f64(0.0);
    float64x2_t c20 = vdupq_n_f64(0.0), c21 = vdupq_n_f64(0.0);
    float64x2_t c30 = vdupq_n_f64(0.0), c31 = vdupq_n_f64(0.0);

    if constexpr (MR >= 2) {
        c10 = vld1q_f64(C + 1 * ldc + 0);
        c11 = vld1q_f64(C + 1 * ldc + 2);
    }
    if constexpr (MR >= 3) {
        c20 = vld1q_f64(C + 2 * ldc + 0);
        c21 = vld1q_f64(C + 2 * ldc + 2);
    }
    if constexpr (MR >= 4) {
        c30 = vld1q_f64(C + 3 * ldc + 0);
        c31 = vld1q_f64(C + 3 * ldc + 2);
    }

    for (size_t r = 0; r < rb; ++r) {
        const float64x2_t b0 = vld1q_f64(Bp[r] + 0);
        const float64x2_t b1 = vld1q_f64(Bp[r] + 2);
        const float64x2_t a01 = vld1q_f64(Ap[r] + 0);

        c00 = vfmaq_laneq_f64(c00, b0, a01, 0);
        c01 = vfmaq_laneq_f64(c01, b1, a01, 0);
        if constexpr (MR >= 2) {
            c10 = vfmaq_laneq_f64(c10, b0, a01, 1);
            c11 = vfmaq_laneq_f64(c11, b1, a01, 1);
        }
        if constexpr (MR >= 3) {
            const float64x2_t a23 = vld1q_f64(Ap[r] + 2);
            c20 = vfmaq_laneq_f64(c20, b0, a23, 0);
            c21 = vfmaq_laneq_f64(c21, b1, a23, 0);
            if constexpr (MR >= 4) {
                c30 = vfmaq_laneq_f64(c30, b0, a23, 1);
                c31 = vfmaq_laneq_f64(c31, b1, a23, 1);
            }
        }
    }

    vst1q_f64(C + 0 * ldc + 0, c00);
    vst1q_f64(C + 0 * ldc + 2, c01);
    if constexpr (MR >= 2) {
        vst1q_f64(C + 1 * ldc + 0, c10);
        vst1q_f64(C + 1 * ldc + 2, c11);
    }
    if constexpr (MR >= 3) {
        vst1q_f64(C + 2 * ldc + 0, c20);
        vst1q_f64(C + 2 * ldc + 2, c21);
    }
    if constexpr (MR >= 4) {
        vst1q_f64(C + 3 * ldc + 0, c30);
        vst1q_f64(C + 3 * ldc + 2, c31);
    }
}

inline void dispatch_mr_x_4_neon(size_t rem,
                                 double* CAUSALSIEVE_RESTRICT C,
                                 size_t ldc,
                                 const double (*CAUSALSIEVE_RESTRICT Ap)[4],
                                 const double (*CAUSALSIEVE_RESTRICT Bp)[4],
                                 size_t rb) noexcept {
    switch (rem) {
        case 4: microkernel_mr_x_4_neon<4>(C, ldc, Ap, Bp, rb); break;
        case 3: microkernel_mr_x_4_neon<3>(C, ldc, Ap, Bp, rb); break;
        case 2: microkernel_mr_x_4_neon<2>(C, ldc, Ap, Bp, rb); break;
        default: microkernel_mr_x_4_neon<1>(C, ldc, Ap, Bp, rb); break;
    }
}

void sym_rankk_update_upper(double* CAUSALSIEVE_RESTRICT A,
                            const double* CAUSALSIEVE_RESTRICT V,
                            const double* CAUSALSIEVE_RESTRICT w,
                            size_t b,
                            size_t n,
                            size_t n_pad) noexcept {
    if (b == 0 || n == 0) return;

    constexpr size_t kMc = 32;
    constexpr size_t kKc = kSlabBatchSize;
    alignas(64) double A_tile[kMc / 4][kKc][4];
    alignas(64) double B_tile[kKc][4];

    for (size_t r0 = 0; r0 < b; r0 += kKc) {
        const size_t rb = std::min(kKc, b - r0);
        const double* Vr = V + r0 * n_pad;
        const double* wr = w + r0;

        for (size_t i0 = 0; i0 < n; i0 += kMc) {
            const size_t m_blk = std::min(kMc, n - i0);
            for (size_t im = 0; im < m_blk; im += 4) {
                double (*ap)[4] = A_tile[im >> 2];
                const size_t col = i0 + im;
                for (size_t r = 0; r < rb; ++r) {
                    const float64x2_t vw = vdupq_n_f64(wr[r]);
                    const double* src = Vr + r * n_pad + col;
                    vst1q_f64(ap[r] + 0, vmulq_f64(vw, vld1q_f64(src + 0)));
                    vst1q_f64(ap[r] + 2, vmulq_f64(vw, vld1q_f64(src + 2)));
                }
            }

            for (size_t j = i0; j + 4 <= n_pad; j += 4) {
                for (size_t r = 0; r < rb; ++r) {
                    const double* src = Vr + r * n_pad + j;
                    vst1q_f64(B_tile[r] + 0, vld1q_f64(src + 0));
                    vst1q_f64(B_tile[r] + 2, vld1q_f64(src + 2));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    if (row > j) break;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_4_neon(rem, A + row * n_pad + j, n_pad, A_tile[im >> 2], B_tile, rb);
                }
            }
        }
    }
}

void gemm_update_rect(double* CAUSALSIEVE_RESTRICT M,
                      const double* CAUSALSIEVE_RESTRICT A,
                      const double* CAUSALSIEVE_RESTRICT B_mat,
                      const double* CAUSALSIEVE_RESTRICT w,
                      size_t b,
                      size_t rows,
                      size_t rows_pad,
                      size_t cols_pad) noexcept {
    if (b == 0 || rows == 0 || cols_pad == 0) return;

    constexpr size_t kMc = 32;
    constexpr size_t kKc = kSlabBatchSize;
    alignas(64) double A_tile[kMc / 4][kKc][4];
    alignas(64) double B_tile[kKc][4];

    for (size_t r0 = 0; r0 < b; r0 += kKc) {
        const size_t rb = std::min(kKc, b - r0);
        const double* Ar = A + r0 * rows_pad;
        const double* Br = B_mat + r0 * cols_pad;
        const double* wr = w + r0;

        for (size_t i0 = 0; i0 < rows; i0 += kMc) {
            const size_t m_blk = std::min(kMc, rows - i0);
            for (size_t im = 0; im < m_blk; im += 4) {
                double (*ap)[4] = A_tile[im >> 2];
                const size_t col = i0 + im;
                for (size_t r = 0; r < rb; ++r) {
                    const float64x2_t vw = vdupq_n_f64(wr[r]);
                    const double* src = Ar + r * rows_pad + col;
                    vst1q_f64(ap[r] + 0, vmulq_f64(vw, vld1q_f64(src + 0)));
                    vst1q_f64(ap[r] + 2, vmulq_f64(vw, vld1q_f64(src + 2)));
                }
            }

            for (size_t j = 0; j + 4 <= cols_pad; j += 4) {
                for (size_t r = 0; r < rb; ++r) {
                    const double* src = Br + r * cols_pad + j;
                    vst1q_f64(B_tile[r] + 0, vld1q_f64(src + 0));
                    vst1q_f64(B_tile[r] + 2, vld1q_f64(src + 2));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_4_neon(rem, M + row * cols_pad + j, cols_pad, A_tile[im >> 2], B_tile, rb);
                }
            }
        }
    }
}

void lingam_batch_update(double* CAUSALSIEVE_RESTRICT M_u2v,
                         double* CAUSALSIEVE_RESTRICT M_utanhv,
                         double* CAUSALSIEVE_RESTRICT M_uv,
                         const double* CAUSALSIEVE_RESTRICT S,
                         const double* CAUSALSIEVE_RESTRICT S2,
                         const double* CAUSALSIEVE_RESTRICT TS,
                         const double* CAUSALSIEVE_RESTRICT w,
                         size_t b,
                         size_t D,
                         size_t D_pad) noexcept {
    if (b == 0 || D == 0) return;

    constexpr size_t kMc = 32;
    constexpr size_t kKc = kSlabBatchSize;
    alignas(64) double S2_w_tile[kMc / 4][kKc][4];
    alignas(64) double S_w_tile[kMc / 4][kKc][4];
    alignas(64) double S_col[kKc][4];
    alignas(64) double TS_col[kKc][4];

    for (size_t r0 = 0; r0 < b; r0 += kKc) {
        const size_t rb = std::min(kKc, b - r0);
        const double* Sr  = S  + r0 * D_pad;
        const double* S2r = S2 + r0 * D_pad;
        const double* TSr = TS + r0 * D_pad;
        const double* wr  = w  + r0;

        for (size_t i0 = 0; i0 < D; i0 += kMc) {
            const size_t m_blk = std::min(kMc, D - i0);
            for (size_t im = 0; im < m_blk; im += 4) {
                double (*s2p)[4] = S2_w_tile[im >> 2];
                double (*sp)[4]  = S_w_tile[im >> 2];
                const size_t col = i0 + im;
                for (size_t r = 0; r < rb; ++r) {
                    const float64x2_t vw = vdupq_n_f64(wr[r]);
                    const double* s2_src = S2r + r * D_pad + col;
                    const double* s_src  = Sr  + r * D_pad + col;
                    vst1q_f64(s2p[r] + 0, vmulq_f64(vw, vld1q_f64(s2_src + 0)));
                    vst1q_f64(s2p[r] + 2, vmulq_f64(vw, vld1q_f64(s2_src + 2)));
                    vst1q_f64(sp[r]  + 0, vmulq_f64(vw, vld1q_f64(s_src  + 0)));
                    vst1q_f64(sp[r]  + 2, vmulq_f64(vw, vld1q_f64(s_src  + 2)));
                }
            }

            for (size_t j = 0; j + 4 <= D_pad; j += 4) {
                for (size_t r = 0; r < rb; ++r) {
                    const double* s_src  = Sr  + r * D_pad + j;
                    const double* ts_src = TSr + r * D_pad + j;
                    vst1q_f64(S_col[r]  + 0, vld1q_f64(s_src  + 0));
                    vst1q_f64(S_col[r]  + 2, vld1q_f64(s_src  + 2));
                    vst1q_f64(TS_col[r] + 0, vld1q_f64(ts_src + 0));
                    vst1q_f64(TS_col[r] + 2, vld1q_f64(ts_src + 2));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_4_neon(rem, M_u2v    + row * D_pad + j, D_pad, S2_w_tile[im >> 2], S_col,  rb);
                    dispatch_mr_x_4_neon(rem, M_utanhv + row * D_pad + j, D_pad, S_w_tile[im >> 2],  TS_col, rb);
                    if (row <= j) {
                        dispatch_mr_x_4_neon(rem, M_uv + row * D_pad + j, D_pad, S_w_tile[im >> 2], S_col, rb);
                    }
                }
            }
        }
    }
}

bool cholesky_factor_scaled_upper(const double* CAUSALSIEVE_RESTRICT A_upper,
                                  double scale,
                                  double diag_ridge,
                                  double* CAUSALSIEVE_RESTRICT L,
                                  size_t n,
                                  size_t n_pad) noexcept {
    std::memset(L, 0, n * n_pad * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double* CAUSALSIEVE_RESTRICT Li = L + i * n_pad;
        for (size_t j = 0; j < i; ++j) {
            const double* CAUSALSIEVE_RESTRICT Lj = L + j * n_pad;
            float64x2_t acc0 = vdupq_n_f64(0.0);
            float64x2_t acc1 = vdupq_n_f64(0.0);
            size_t k = 0;
            for (; k + 4 <= j; k += 4) {
                acc0 = vfmaq_f64(acc0, vld1q_f64(Li + k),     vld1q_f64(Lj + k));
                acc1 = vfmaq_f64(acc1, vld1q_f64(Li + k + 2), vld1q_f64(Lj + k + 2));
            }
            for (; k + 2 <= j; k += 2) {
                acc0 = vfmaq_f64(acc0, vld1q_f64(Li + k), vld1q_f64(Lj + k));
            }
            double dot = vaddvq_f64(vaddq_f64(acc0, acc1));
            for (; k < j; ++k) dot += Li[k] * Lj[k];
            Li[j] = (scale * A_upper[j * n_pad + i] - dot) / Lj[j];
        }
        float64x2_t d0 = vdupq_n_f64(0.0);
        float64x2_t d1 = vdupq_n_f64(0.0);
        size_t k = 0;
        for (; k + 4 <= i; k += 4) {
            const float64x2_t v0 = vld1q_f64(Li + k);
            const float64x2_t v1 = vld1q_f64(Li + k + 2);
            d0 = vfmaq_f64(d0, v0, v0);
            d1 = vfmaq_f64(d1, v1, v1);
        }
        for (; k + 2 <= i; k += 2) {
            const float64x2_t v0 = vld1q_f64(Li + k);
            d0 = vfmaq_f64(d0, v0, v0);
        }
        double norm2 = vaddvq_f64(vaddq_f64(d0, d1));
        for (; k < i; ++k) norm2 += Li[k] * Li[k];
        double diag = scale * A_upper[i * n_pad + i] + diag_ridge - norm2;
        if (!(diag > 0.0) || !std::isfinite(diag)) {
            if (diag > -1e-12) {
                diag = 1e-15;
            } else {
                return false;
            }
        }
        Li[i] = std::sqrt(diag);
    }
    return true;
}

void cholesky_invert_to_upper(const double* CAUSALSIEVE_RESTRICT L,
                              double* CAUSALSIEVE_RESTRICT P_upper,
                              double* CAUSALSIEVE_RESTRICT tmp_U,
                              size_t n,
                              size_t n_pad) noexcept {
    std::memset(tmp_U, 0, n * n_pad * sizeof(double));
    std::memset(P_upper, 0, n * n_pad * sizeof(double));

    for (size_t j = 0; j < n; ++j) {
        double* CAUSALSIEVE_RESTRICT Uj = tmp_U + j * n_pad;
        Uj[j] = 1.0 / L[j * n_pad + j];
        for (size_t i = j + 1; i < n; ++i) {
            const double* CAUSALSIEVE_RESTRICT Li = L + i * n_pad;
            float64x2_t acc0 = vdupq_n_f64(0.0);
            float64x2_t acc1 = vdupq_n_f64(0.0);
            size_t k = j;
            for (; k + 4 <= i; k += 4) {
                acc0 = vfmaq_f64(acc0, vld1q_f64(Li + k),     vld1q_f64(Uj + k));
                acc1 = vfmaq_f64(acc1, vld1q_f64(Li + k + 2), vld1q_f64(Uj + k + 2));
            }
            for (; k + 2 <= i; k += 2) {
                acc0 = vfmaq_f64(acc0, vld1q_f64(Li + k), vld1q_f64(Uj + k));
            }
            double sum = vaddvq_f64(vaddq_f64(acc0, acc1));
            for (; k < i; ++k) sum += Li[k] * Uj[k];
            Uj[i] = -sum / Li[i];
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const double* CAUSALSIEVE_RESTRICT Ui = tmp_U + i * n_pad;
        double* CAUSALSIEVE_RESTRICT Pi = P_upper + i * n_pad;
        for (size_t j = i; j < n; ++j) {
            const double* CAUSALSIEVE_RESTRICT Uj = tmp_U + j * n_pad;
            float64x2_t acc0 = vdupq_n_f64(0.0);
            float64x2_t acc1 = vdupq_n_f64(0.0);
            size_t k = j;
            for (; k + 4 <= n; k += 4) {
                acc0 = vfmaq_f64(acc0, vld1q_f64(Ui + k),     vld1q_f64(Uj + k));
                acc1 = vfmaq_f64(acc1, vld1q_f64(Ui + k + 2), vld1q_f64(Uj + k + 2));
            }
            for (; k + 2 <= n; k += 2) {
                acc0 = vfmaq_f64(acc0, vld1q_f64(Ui + k), vld1q_f64(Uj + k));
            }
            double sum = vaddvq_f64(vaddq_f64(acc0, acc1));
            for (; k < n; ++k) sum += Ui[k] * Uj[k];
            Pi[j] = sum;
        }
    }
}

void cholesky_solve_multi(const double* CAUSALSIEVE_RESTRICT L,
                          const double* CAUSALSIEVE_RESTRICT C_tilde,
                          double c_scale,
                          double* CAUSALSIEVE_RESTRICT B,
                          double* CAUSALSIEVE_RESTRICT Y,
                          size_t K,
                          size_t D,
                          size_t K_pad,
                          size_t D_pad) noexcept {
    const float64x2_t vc = vdupq_n_f64(c_scale);

    for (size_t i = 0; i < K; ++i) {
        const double* CAUSALSIEVE_RESTRICT Li = L + i * K_pad;
        const double* CAUSALSIEVE_RESTRICT Ci = C_tilde + i * D_pad;
        double* CAUSALSIEVE_RESTRICT Yi = Y + i * D_pad;
        for (size_t j = 0; j + 2 <= D_pad; j += 2) {
            vst1q_f64(Yi + j, vmulq_f64(vc, vld1q_f64(Ci + j)));
        }
        size_t k = 0;
        for (; k + 4 <= i; k += 4) {
            const float64x2_t l0 = vdupq_n_f64(Li[k + 0]);
            const float64x2_t l1 = vdupq_n_f64(Li[k + 1]);
            const float64x2_t l2 = vdupq_n_f64(Li[k + 2]);
            const float64x2_t l3 = vdupq_n_f64(Li[k + 3]);
            const double* CAUSALSIEVE_RESTRICT Y0 = Y + (k + 0) * D_pad;
            const double* CAUSALSIEVE_RESTRICT Y1 = Y + (k + 1) * D_pad;
            const double* CAUSALSIEVE_RESTRICT Y2 = Y + (k + 2) * D_pad;
            const double* CAUSALSIEVE_RESTRICT Y3 = Y + (k + 3) * D_pad;
            for (size_t j = 0; j + 2 <= D_pad; j += 2) {
                float64x2_t yj = vld1q_f64(Yi + j);
                yj = vfmsq_f64(yj, l0, vld1q_f64(Y0 + j));
                yj = vfmsq_f64(yj, l1, vld1q_f64(Y1 + j));
                yj = vfmsq_f64(yj, l2, vld1q_f64(Y2 + j));
                yj = vfmsq_f64(yj, l3, vld1q_f64(Y3 + j));
                vst1q_f64(Yi + j, yj);
            }
        }
        for (; k < i; ++k) {
            const float64x2_t lk = vdupq_n_f64(Li[k]);
            const double* CAUSALSIEVE_RESTRICT Yk = Y + k * D_pad;
            for (size_t j = 0; j + 2 <= D_pad; j += 2) {
                vst1q_f64(Yi + j, vfmsq_f64(vld1q_f64(Yi + j), lk, vld1q_f64(Yk + j)));
            }
        }
        const float64x2_t vinv = vdupq_n_f64(1.0 / Li[i]);
        for (size_t j = 0; j + 2 <= D_pad; j += 2) {
            vst1q_f64(Yi + j, vmulq_f64(vld1q_f64(Yi + j), vinv));
        }
        for (size_t j = D; j < D_pad; ++j) Yi[j] = 0.0;
    }

    for (size_t ii = K; ii > 0; --ii) {
        const size_t i = ii - 1;
        double* CAUSALSIEVE_RESTRICT Bi = B + i * D_pad;
        const double* CAUSALSIEVE_RESTRICT Yi = Y + i * D_pad;
        std::memcpy(Bi, Yi, D_pad * sizeof(double));
        size_t k = i + 1;
        for (; k + 4 <= K; k += 4) {
            const float64x2_t l0 = vdupq_n_f64(L[(k + 0) * K_pad + i]);
            const float64x2_t l1 = vdupq_n_f64(L[(k + 1) * K_pad + i]);
            const float64x2_t l2 = vdupq_n_f64(L[(k + 2) * K_pad + i]);
            const float64x2_t l3 = vdupq_n_f64(L[(k + 3) * K_pad + i]);
            const double* CAUSALSIEVE_RESTRICT B0 = B + (k + 0) * D_pad;
            const double* CAUSALSIEVE_RESTRICT B1 = B + (k + 1) * D_pad;
            const double* CAUSALSIEVE_RESTRICT B2 = B + (k + 2) * D_pad;
            const double* CAUSALSIEVE_RESTRICT B3 = B + (k + 3) * D_pad;
            for (size_t j = 0; j + 2 <= D_pad; j += 2) {
                float64x2_t bj = vld1q_f64(Bi + j);
                bj = vfmsq_f64(bj, l0, vld1q_f64(B0 + j));
                bj = vfmsq_f64(bj, l1, vld1q_f64(B1 + j));
                bj = vfmsq_f64(bj, l2, vld1q_f64(B2 + j));
                bj = vfmsq_f64(bj, l3, vld1q_f64(B3 + j));
                vst1q_f64(Bi + j, bj);
            }
        }
        for (; k < K; ++k) {
            const float64x2_t lk = vdupq_n_f64(L[k * K_pad + i]);
            const double* CAUSALSIEVE_RESTRICT Bk = B + k * D_pad;
            for (size_t j = 0; j + 2 <= D_pad; j += 2) {
                vst1q_f64(Bi + j, vfmsq_f64(vld1q_f64(Bi + j), lk, vld1q_f64(Bk + j)));
            }
        }
        const float64x2_t vinv = vdupq_n_f64(1.0 / L[i * K_pad + i]);
        for (size_t j = 0; j + 2 <= D_pad; j += 2) {
            vst1q_f64(Bi + j, vmulq_f64(vld1q_f64(Bi + j), vinv));
        }
        for (size_t j = D; j < D_pad; ++j) Bi[j] = 0.0;
    }
}

double screened_step_postwarmup(
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
    double peak_decay
) noexcept {
    (void)D_pad;
    const float64x2_t v_ema = vdupq_n_f64(ema_alpha);
    const float64x2_t v_1mema = vdupq_n_f64(1.0 - ema_alpha);
    const float64x2_t v_edecay = vdupq_n_f64(energy_decay);
    const float64x2_t v_pdecay = vdupq_n_f64(peak_decay);
    const float64x2_t v_gate = vdupq_n_f64(3.5);
    const float64x2_t v_ethresh = vdupq_n_f64(2.5);
    const float64x2_t v_zero = vdupq_n_f64(0.0);
    const float64x2_t v_one = vdupq_n_f64(1.0);
    const float64x2_t v_1e6 = vdupq_n_f64(1e-6);
    const float64x2_t v_1e4 = vdupq_n_f64(1e-4);
    const float64x2_t v_pmax = vdupq_n_f64(0.98);
    const float64x2_t v_pmin = vdupq_n_f64(-0.98);
    const float64x2_t v_clamp_norm = vdupq_n_f64(6.0);
    const float64x2_t v_nclamp_norm = vdupq_n_f64(-6.0);
    float64x2_t vmax = vdupq_n_f64(0.0);

    size_t i = 0;
    for (; i + 2 <= D; i += 2) {
        const float64x2_t vx = vld1q_f64(x + i);
        const float64x2_t vanch = vld1q_f64(anchor + i);
        const float64x2_t vdx = vsubq_f64(vx, vanch);

        float64x2_t vmean = vld1q_f64(mean + i);
        const float64x2_t vinv_std = vld1q_f64(inv_std + i);
        const float64x2_t vdiff = vsubq_f64(vdx, vmean);
        const float64x2_t vxnorm = vmulq_f64(vdiff, vinv_std);

        const float64x2_t vprev = vld1q_f64(prev_norm + i);
        const float64x2_t vphi = vld1q_f64(phi + i);
        const float64x2_t vres = vfmsq_f64(vxnorm, vphi, vprev);

        const float64x2_t vinv_rstd = vld1q_f64(inv_res_std + i);
        const float64x2_t vz = vmulq_f64(vres, vinv_rstd);
        vst1q_f64(out_z + i, vz);

        const float64x2_t vaz = vabsq_f64(vz);
        vmax = vmaxq_f64(vmax, vaz);

        const float64x2_t vex = vmaxq_f64(v_zero, vsubq_f64(vaz, v_ethresh));
        float64x2_t ven = vld1q_f64(energy + i);
        ven = vfmaq_f64(vmulq_f64(vex, vex), v_edecay, ven);
        vst1q_f64(energy + i, ven);

        float64x2_t vpk = vld1q_f64(recent_peak + i);
        vpk = vmaxq_f64(vmulq_f64(v_pdecay, vpk), vaz);
        vst1q_f64(recent_peak + i, vpk);

        const float64x2_t vaxnorm = vabsq_f64(vxnorm);
        const uint64x2_t mask_z = vcleq_f64(vaz, v_gate);
        const uint64x2_t mask_x = vcleq_f64(vaxnorm, v_gate);
        const uint64x2_t mask = vandq_u64(mask_z, mask_x);

        const float64x2_t new_mean = vfmaq_f64(vmean, v_ema, vdiff);
        float64x2_t vvar = vld1q_f64(var + i);
        const float64x2_t new_var = vfmaq_f64(vmulq_f64(v_1mema, vvar),
                                              v_ema, vmulq_f64(vdiff, vdiff));
        vmean = vbslq_f64(mask, new_mean, vmean);
        vvar  = vbslq_f64(mask, new_var, vvar);
        vst1q_f64(mean + i, vmean);
        vst1q_f64(var + i, vvar);

        const float64x2_t vfloor_s = vld1q_f64(floor_std + i);
        const float64x2_t vstd = vmaxq_f64(vsqrtq_f64(vmaxq_f64(v_zero, vvar)), vfloor_s);
        vst1q_f64(inv_std + i, vdivq_f64(v_one, vstd));

        float64x2_t vacov = vld1q_f64(ar_cov + i);
        float64x2_t vavar = vld1q_f64(ar_var + i);
        float64x2_t vrvar = vld1q_f64(ar_res_var + i);
        const float64x2_t new_acov = vfmaq_f64(vmulq_f64(v_1mema, vacov),
                                               v_ema, vmulq_f64(vxnorm, vprev));
        const float64x2_t new_avar = vfmaq_f64(vmulq_f64(v_1mema, vavar),
                                               v_ema, vmulq_f64(vprev, vprev));
        const float64x2_t new_rvar = vfmaq_f64(vmulq_f64(v_1mema, vrvar),
                                               v_ema, vmulq_f64(vres, vres));
        vacov = vbslq_f64(mask, new_acov, vacov);
        vavar = vbslq_f64(mask, new_avar, vavar);
        vrvar = vbslq_f64(mask, new_rvar, vrvar);
        vst1q_f64(ar_cov + i, vacov);
        vst1q_f64(ar_var + i, vavar);
        vst1q_f64(ar_res_var + i, vrvar);

        float64x2_t new_phi = vdivq_f64(vacov, vmaxq_f64(vavar, v_1e6));
        new_phi = vmaxq_f64(v_pmin, vminq_f64(v_pmax, new_phi));
        vst1q_f64(phi + i, new_phi);
        const float64x2_t new_rstd = vsqrtq_f64(vmaxq_f64(vrvar, v_1e4));
        vst1q_f64(inv_res_std + i, vdivq_f64(v_one, new_rstd));

        const float64x2_t vprev_next = vmaxq_f64(v_nclamp_norm, vminq_f64(v_clamp_norm, vxnorm));
        vst1q_f64(prev_norm + i, vprev_next);
    }

    alignas(16) double max_lanes[2];
    vst1q_f64(max_lanes, vmax);
    double max_abs_z = std::max(max_lanes[0], max_lanes[1]);

    for (; i < D; ++i) {
        const double dx = x[i] - anchor[i];
        const double diff = dx - mean[i];
        const double xnorm = diff * inv_std[i];
        const double res = xnorm - phi[i] * prev_norm[i];
        const double z = res * inv_res_std[i];
        out_z[i] = z;

        const double az = std::fabs(z);
        if (az > max_abs_z) max_abs_z = az;

        const double ex = (az > 2.5) ? (az - 2.5) : 0.0;
        energy[i] = energy_decay * energy[i] + ex * ex;
        recent_peak[i] = std::max(peak_decay * recent_peak[i], az);

        if (az <= 3.5 && std::fabs(xnorm) <= 3.5) {
            mean[i] += ema_alpha * diff;
            var[i] = (1.0 - ema_alpha) * var[i] + ema_alpha * diff * diff;
            const double s = std::max(std::sqrt(std::max(0.0, var[i])), floor_std[i]);
            inv_std[i] = 1.0 / s;

            ar_cov[i] = (1.0 - ema_alpha) * ar_cov[i] + ema_alpha * xnorm * prev_norm[i];
            ar_var[i] = (1.0 - ema_alpha) * ar_var[i] + ema_alpha * prev_norm[i] * prev_norm[i];
            ar_res_var[i] = (1.0 - ema_alpha) * ar_res_var[i] + ema_alpha * res * res;
            phi[i] = std::clamp(ar_cov[i] / std::max(ar_var[i], 1e-6), -0.98, 0.98);
            inv_res_std[i] = 1.0 / std::sqrt(std::max(ar_res_var[i], 1e-4));
        }
        prev_norm[i] = std::clamp(xnorm, -6.0, 6.0);
    }
    return max_abs_z;
}

} /* namespace neon_impl */

const KernelOps& get_neon_ops() noexcept {
    static const KernelOps kOps{
        "neon",
        &neon_impl::dot_pd,
        &neon_impl::sym_matvec_upper,
        &neon_impl::sym_rank1_update_upper,
        &neon_impl::matvec_BT_z,
        &neon_impl::outer_update_rect,
        &neon_impl::scale_vec,
        &neon_impl::sub_vec,
        &neon_impl::all_finite_pd,
        &neon_impl::sym_rankk_update_upper,
        &neon_impl::gemm_update_rect,
        &neon_impl::lingam_batch_update,
        &neon_impl::cholesky_factor_scaled_upper,
        &neon_impl::cholesky_invert_to_upper,
        &neon_impl::cholesky_solve_multi,
        &neon_impl::screened_step_postwarmup,
    };
    return kOps;
}
#endif /* CAUSALSIEVE_COMPILE_NEON_IMPL */

} /* namespace causalsieve::kernels */
