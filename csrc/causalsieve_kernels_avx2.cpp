#include "causalsieve_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(CAUSALSIEVE_BUILD_AVX2)
#if !(defined(__AVX2__) && defined(__FMA__))
#error "causalsieve_kernels_avx2.cpp built with CAUSALSIEVE_BUILD_AVX2 requires -mavx2 -mfma (or /arch:AVX2 /D__FMA__=1)"
#endif
#define CAUSALSIEVE_COMPILE_AVX2_IMPL 1
#elif defined(__AVX2__) && defined(__FMA__) && !defined(CAUSALSIEVE_FORCE_SCALAR)
#define CAUSALSIEVE_COMPILE_AVX2_IMPL 1
#else
#define CAUSALSIEVE_COMPILE_AVX2_IMPL 0
#endif

#if CAUSALSIEVE_COMPILE_AVX2_IMPL
#include <immintrin.h>

namespace causalsieve::kernels {
namespace avx2_impl {

inline double hsum256_pd(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_unpackhi_pd(lo, lo);
    return _mm_cvtsd_f64(_mm_add_sd(lo, shuf));
}

double dot_pd(const double* CAUSALSIEVE_RESTRICT a,
              const double* CAUSALSIEVE_RESTRICT b,
              size_t n_pad) noexcept {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 16 <= n_pad; i += 16) {
        acc0 = _mm256_fmadd_pd(_mm256_load_pd(a + i),      _mm256_load_pd(b + i),      acc0);
        acc1 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 4),  _mm256_load_pd(b + i + 4),  acc1);
        acc2 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 8),  _mm256_load_pd(b + i + 8),  acc2);
        acc3 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 12), _mm256_load_pd(b + i + 12), acc3);
    }
    for (; i + 4 <= n_pad; i += 4) {
        acc0 = _mm256_fmadd_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i), acc0);
    }
    acc0 = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    return hsum256_pd(acc0);
}

/*
 * 4-row register-blocked symmetric matrix-vector product y = A * x
 * reading ONLY the upper triangle of A (n x n_pad, row-major).
 * Loads x[j..j+7] and loads/stores y[j..j+7] 4x less often than 1-row!
 */
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

        const double x0 = x[i + 0];
        const double x1 = x[i + 1];
        const double x2 = x[i + 2];
        const double x3 = x[i + 3];

        /* Exact 4x4 symmetric diagonal block */
        const double a01 = r0[i + 1], a02 = r0[i + 2], a03 = r0[i + 3];
        const double a12 = r1[i + 2], a13 = r1[i + 3];
        const double a23 = r2[i + 3];

        const double s0 = r0[i + 0] * x0 + a01 * x1 + a02 * x2 + a03 * x3;
        const double s1 = a01 * x0 + r1[i + 1] * x1 + a12 * x2 + a13 * x3;
        const double s2 = a02 * x0 + a12 * x1 + r2[i + 2] * x2 + a23 * x3;
        const double s3 = a03 * x0 + a13 * x1 + a23 * x2 + r3[i + 3] * x3;

        const __m256d vx0 = _mm256_set1_pd(x0);
        const __m256d vx1 = _mm256_set1_pd(x1);
        const __m256d vx2 = _mm256_set1_pd(x2);
        const __m256d vx3 = _mm256_set1_pd(x3);

        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();

        size_t j = i + 4;
        for (; j + 4 <= n_pad; j += 4) {
            const __m256d xj = _mm256_load_pd(x + j);
            __m256d yj = _mm256_load_pd(y + j);

            const __m256d a0 = _mm256_load_pd(r0 + j);
            const __m256d a1 = _mm256_load_pd(r1 + j);
            const __m256d a2 = _mm256_load_pd(r2 + j);
            const __m256d a3 = _mm256_load_pd(r3 + j);

            acc0 = _mm256_fmadd_pd(a0, xj, acc0);
            acc1 = _mm256_fmadd_pd(a1, xj, acc1);
            acc2 = _mm256_fmadd_pd(a2, xj, acc2);
            acc3 = _mm256_fmadd_pd(a3, xj, acc3);

            __m256d t0 = _mm256_mul_pd(a1, vx1);
            yj = _mm256_fmadd_pd(a0, vx0, yj);
            t0 = _mm256_fmadd_pd(a3, vx3, t0);
            yj = _mm256_fmadd_pd(a2, vx2, yj);
            _mm256_store_pd(y + j, _mm256_add_pd(yj, t0));
        }

        /* 4-way vectorized horizontal reduction into y[i..i+3] */
        const __m256d h01  = _mm256_hadd_pd(acc0, acc1);
        const __m256d h23  = _mm256_hadd_pd(acc2, acc3);
        const __m256d lo   = _mm256_permute2f128_pd(h01, h23, 0x20);
        const __m256d hi   = _mm256_permute2f128_pd(h01, h23, 0x31);
        const __m256d hsum = _mm256_add_pd(lo, hi);
        const __m256d svec = _mm256_set_pd(s3, s2, s1, s0);
        const __m256d yi   = _mm256_load_pd(y + i);
        _mm256_store_pd(y + i, _mm256_add_pd(yi, _mm256_add_pd(svec, hsum)));
    }

    /* Remaining 0..3 rows lie entirely inside the final 4x4 diagonal corner */
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

/*
 * 4-row register-blocked symmetric rank-1 update of upper triangle:
 * A += alpha * v * v^T. All 4 rows start at aligned j = i.
 */
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

        const __m256d vs0 = _mm256_set1_pd(alpha * v[i + 0]);
        const __m256d vs1 = _mm256_set1_pd(alpha * v[i + 1]);
        const __m256d vs2 = _mm256_set1_pd(alpha * v[i + 2]);
        const __m256d vs3 = _mm256_set1_pd(alpha * v[i + 3]);

        size_t j = i;
        for (; j + 8 <= n_pad; j += 8) {
            const __m256d v0 = _mm256_load_pd(v + j);
            const __m256d v1 = _mm256_load_pd(v + j + 4);
            _mm256_store_pd(r0 + j,     _mm256_fmadd_pd(vs0, v0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r0 + j + 4, _mm256_fmadd_pd(vs0, v1, _mm256_load_pd(r0 + j + 4)));
            _mm256_store_pd(r1 + j,     _mm256_fmadd_pd(vs1, v0, _mm256_load_pd(r1 + j)));
            _mm256_store_pd(r1 + j + 4, _mm256_fmadd_pd(vs1, v1, _mm256_load_pd(r1 + j + 4)));
            _mm256_store_pd(r2 + j,     _mm256_fmadd_pd(vs2, v0, _mm256_load_pd(r2 + j)));
            _mm256_store_pd(r2 + j + 4, _mm256_fmadd_pd(vs2, v1, _mm256_load_pd(r2 + j + 4)));
            _mm256_store_pd(r3 + j,     _mm256_fmadd_pd(vs3, v0, _mm256_load_pd(r3 + j)));
            _mm256_store_pd(r3 + j + 4, _mm256_fmadd_pd(vs3, v1, _mm256_load_pd(r3 + j + 4)));
        }
        for (; j + 4 <= n_pad; j += 4) {
            const __m256d v0 = _mm256_load_pd(v + j);
            _mm256_store_pd(r0 + j, _mm256_fmadd_pd(vs0, v0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r1 + j, _mm256_fmadd_pd(vs1, v0, _mm256_load_pd(r1 + j)));
            _mm256_store_pd(r2 + j, _mm256_fmadd_pd(vs2, v0, _mm256_load_pd(r2 + j)));
            _mm256_store_pd(r3 + j, _mm256_fmadd_pd(vs3, v0, _mm256_load_pd(r3 + j)));
        }
    }
    if (i < n) {
        const size_t j0 = i; /* aligned n_pad - 4 */
        const __m256d v0 = _mm256_load_pd(v + j0);
        for (; i < n; ++i) {
            double* CAUSALSIEVE_RESTRICT row = A + i * n_pad;
            const __m256d vs = _mm256_set1_pd(alpha * v[i]);
            _mm256_store_pd(row + j0, _mm256_fmadd_pd(vs, v0, _mm256_load_pd(row + j0)));
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
    for (; j + 16 <= D_pad; j += 16) {
        __m256d y0 = _mm256_setzero_pd();
        __m256d y1 = _mm256_setzero_pd();
        __m256d y2 = _mm256_setzero_pd();
        __m256d y3 = _mm256_setzero_pd();
        for (size_t i = 0; i < K; ++i) {
            const double* CAUSALSIEVE_RESTRICT row = B + i * D_pad + j;
            const __m256d vz = _mm256_set1_pd(z[i]);
            y0 = _mm256_fmadd_pd(vz, _mm256_load_pd(row),      y0);
            y1 = _mm256_fmadd_pd(vz, _mm256_load_pd(row + 4),  y1);
            y2 = _mm256_fmadd_pd(vz, _mm256_load_pd(row + 8),  y2);
            y3 = _mm256_fmadd_pd(vz, _mm256_load_pd(row + 12), y3);
        }
        _mm256_store_pd(y + j,      y0);
        _mm256_store_pd(y + j + 4,  y1);
        _mm256_store_pd(y + j + 8,  y2);
        _mm256_store_pd(y + j + 12, y3);
    }
    for (; j + 8 <= D_pad; j += 8) {
        __m256d y0a = _mm256_setzero_pd();
        __m256d y1a = _mm256_setzero_pd();
        __m256d y0b = _mm256_setzero_pd();
        __m256d y1b = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 2 <= K; i += 2) {
            const double* CAUSALSIEVE_RESTRICT r0 = B + i * D_pad + j;
            const double* CAUSALSIEVE_RESTRICT r1 = B + (i + 1) * D_pad + j;
            const __m256d vz0 = _mm256_set1_pd(z[i]);
            const __m256d vz1 = _mm256_set1_pd(z[i + 1]);
            y0a = _mm256_fmadd_pd(vz0, _mm256_load_pd(r0),     y0a);
            y1a = _mm256_fmadd_pd(vz0, _mm256_load_pd(r0 + 4), y1a);
            y0b = _mm256_fmadd_pd(vz1, _mm256_load_pd(r1),     y0b);
            y1b = _mm256_fmadd_pd(vz1, _mm256_load_pd(r1 + 4), y1b);
        }
        for (; i < K; ++i) {
            const double* CAUSALSIEVE_RESTRICT r0 = B + i * D_pad + j;
            const __m256d vz0 = _mm256_set1_pd(z[i]);
            y0a = _mm256_fmadd_pd(vz0, _mm256_load_pd(r0),     y0a);
            y1a = _mm256_fmadd_pd(vz0, _mm256_load_pd(r0 + 4), y1a);
        }
        _mm256_store_pd(y + j,     _mm256_add_pd(y0a, y0b));
        _mm256_store_pd(y + j + 4, _mm256_add_pd(y1a, y1b));
    }
    for (; j + 4 <= D_pad; j += 4) {
        __m256d y0 = _mm256_setzero_pd();
        __m256d y1 = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 2 <= K; i += 2) {
            y0 = _mm256_fmadd_pd(_mm256_set1_pd(z[i]),     _mm256_load_pd(B + i * D_pad + j),       y0);
            y1 = _mm256_fmadd_pd(_mm256_set1_pd(z[i + 1]), _mm256_load_pd(B + (i + 1) * D_pad + j), y1);
        }
        for (; i < K; ++i) {
            y0 = _mm256_fmadd_pd(_mm256_set1_pd(z[i]), _mm256_load_pd(B + i * D_pad + j), y0);
        }
        _mm256_store_pd(y + j, _mm256_add_pd(y0, y1));
    }
    for (size_t k = D; k < D_pad; ++k) y[k] = 0.0;
}

/*
 * 4-row register-blocked rectangular rank-1 outer update M += alpha * a * b^T.
 * Loads b[j..j+7] 4x less often and saturates dual AVX2 FMA ports.
 */
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

        const __m256d va0 = _mm256_set1_pd(alpha * a[i + 0]);
        const __m256d va1 = _mm256_set1_pd(alpha * a[i + 1]);
        const __m256d va2 = _mm256_set1_pd(alpha * a[i + 2]);
        const __m256d va3 = _mm256_set1_pd(alpha * a[i + 3]);

        size_t j = 0;
        for (; j + 8 <= cols_pad; j += 8) {
            const __m256d b0 = _mm256_load_pd(b + j);
            const __m256d b1 = _mm256_load_pd(b + j + 4);
            _mm256_store_pd(r0 + j,     _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r0 + j + 4, _mm256_fmadd_pd(va0, b1, _mm256_load_pd(r0 + j + 4)));
            _mm256_store_pd(r1 + j,     _mm256_fmadd_pd(va1, b0, _mm256_load_pd(r1 + j)));
            _mm256_store_pd(r1 + j + 4, _mm256_fmadd_pd(va1, b1, _mm256_load_pd(r1 + j + 4)));
            _mm256_store_pd(r2 + j,     _mm256_fmadd_pd(va2, b0, _mm256_load_pd(r2 + j)));
            _mm256_store_pd(r2 + j + 4, _mm256_fmadd_pd(va2, b1, _mm256_load_pd(r2 + j + 4)));
            _mm256_store_pd(r3 + j,     _mm256_fmadd_pd(va3, b0, _mm256_load_pd(r3 + j)));
            _mm256_store_pd(r3 + j + 4, _mm256_fmadd_pd(va3, b1, _mm256_load_pd(r3 + j + 4)));
        }
        for (; j + 4 <= cols_pad; j += 4) {
            const __m256d b0 = _mm256_load_pd(b + j);
            _mm256_store_pd(r0 + j, _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r1 + j, _mm256_fmadd_pd(va1, b0, _mm256_load_pd(r1 + j)));
            _mm256_store_pd(r2 + j, _mm256_fmadd_pd(va2, b0, _mm256_load_pd(r2 + j)));
            _mm256_store_pd(r3 + j, _mm256_fmadd_pd(va3, b0, _mm256_load_pd(r3 + j)));
        }
    }
    for (; i < rows; ++i) {
        double* CAUSALSIEVE_RESTRICT r0 = M + i * cols_pad;
        const __m256d va0 = _mm256_set1_pd(alpha * a[i]);
        size_t j = 0;
        for (; j + 8 <= cols_pad; j += 8) {
            const __m256d b0 = _mm256_load_pd(b + j);
            const __m256d b1 = _mm256_load_pd(b + j + 4);
            _mm256_store_pd(r0 + j,     _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r0 + j + 4, _mm256_fmadd_pd(va0, b1, _mm256_load_pd(r0 + j + 4)));
        }
        for (; j + 4 <= cols_pad; j += 4) {
            const __m256d b0 = _mm256_load_pd(b + j);
            _mm256_store_pd(r0 + j, _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
        }
    }
}

void scale_vec(const double* CAUSALSIEVE_RESTRICT src,
               double s,
               double* CAUSALSIEVE_RESTRICT dst,
               size_t n_pad) noexcept {
    const __m256d vs = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 16 <= n_pad; i += 16) {
        _mm256_store_pd(dst + i,      _mm256_mul_pd(_mm256_load_pd(src + i),      vs));
        _mm256_store_pd(dst + i + 4,  _mm256_mul_pd(_mm256_load_pd(src + i + 4),  vs));
        _mm256_store_pd(dst + i + 8,  _mm256_mul_pd(_mm256_load_pd(src + i + 8),  vs));
        _mm256_store_pd(dst + i + 12, _mm256_mul_pd(_mm256_load_pd(src + i + 12), vs));
    }
    for (; i + 4 <= n_pad; i += 4) {
        _mm256_store_pd(dst + i, _mm256_mul_pd(_mm256_load_pd(src + i), vs));
    }
}

void sub_vec(const double* CAUSALSIEVE_RESTRICT a,
             const double* CAUSALSIEVE_RESTRICT b,
             double* CAUSALSIEVE_RESTRICT out,
             size_t n_pad) noexcept {
    size_t i = 0;
    for (; i + 16 <= n_pad; i += 16) {
        _mm256_store_pd(out + i,      _mm256_sub_pd(_mm256_load_pd(a + i),      _mm256_load_pd(b + i)));
        _mm256_store_pd(out + i + 4,  _mm256_sub_pd(_mm256_load_pd(a + i + 4),  _mm256_load_pd(b + i + 4)));
        _mm256_store_pd(out + i + 8,  _mm256_sub_pd(_mm256_load_pd(a + i + 8),  _mm256_load_pd(b + i + 8)));
        _mm256_store_pd(out + i + 12, _mm256_sub_pd(_mm256_load_pd(a + i + 12), _mm256_load_pd(b + i + 12)));
    }
    for (; i + 4 <= n_pad; i += 4) {
        _mm256_store_pd(out + i, _mm256_sub_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
    }
}

/* Exact IEEE-754 bitwise NaN/Inf scan: 16 doubles per iteration, zero OOB reads. */
bool all_finite_pd(const double* ptr, size_t n) noexcept {
    const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000LL);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i v0 = _mm256_and_si256(_mm256_castpd_si256(_mm256_loadu_pd(ptr + i)),      exp_mask);
        const __m256i v1 = _mm256_and_si256(_mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 4)),  exp_mask);
        const __m256i v2 = _mm256_and_si256(_mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 8)),  exp_mask);
        const __m256i v3 = _mm256_and_si256(_mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 12)), exp_mask);
        const __m256i bad = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi64(v0, exp_mask), _mm256_cmpeq_epi64(v1, exp_mask)),
            _mm256_or_si256(_mm256_cmpeq_epi64(v2, exp_mask), _mm256_cmpeq_epi64(v3, exp_mask)));
        if (!_mm256_testz_si256(bad, bad)) return false;
    }
    for (; i + 4 <= n; i += 4) {
        const __m256i v0 = _mm256_and_si256(_mm256_castpd_si256(_mm256_loadu_pd(ptr + i)), exp_mask);
        const __m256i bad = _mm256_cmpeq_epi64(v0, exp_mask);
        if (!_mm256_testz_si256(bad, bad)) return false;
    }
    for (; i < n; ++i) {
        if (!std::isfinite(ptr[i])) return false;
    }
    return true;
}

/* =========================================================================
 * L1-Resident Register-Blocked 4x8 / 4x4 BLAS-3 Micro-Kernels (AVX2+FMA)
 * ========================================================================= */
template <size_t MR>
inline void microkernel_mr_x_8(double* CAUSALSIEVE_RESTRICT C,
                               size_t ldc,
                               const double (*CAUSALSIEVE_RESTRICT Ap)[4],
                               const double (*CAUSALSIEVE_RESTRICT Bp)[8],
                               size_t rb) noexcept {
    __m256d c00 = _mm256_load_pd(C + 0 * ldc + 0);
    __m256d c01 = _mm256_load_pd(C + 0 * ldc + 4);
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();

    if constexpr (MR >= 2) {
        c10 = _mm256_load_pd(C + 1 * ldc + 0);
        c11 = _mm256_load_pd(C + 1 * ldc + 4);
    }
    if constexpr (MR >= 3) {
        c20 = _mm256_load_pd(C + 2 * ldc + 0);
        c21 = _mm256_load_pd(C + 2 * ldc + 4);
    }
    if constexpr (MR >= 4) {
        c30 = _mm256_load_pd(C + 3 * ldc + 0);
        c31 = _mm256_load_pd(C + 3 * ldc + 4);
    }

    for (size_t r = 0; r < rb; ++r) {
        const __m256d b0 = _mm256_load_pd(Bp[r] + 0);
        const __m256d b1 = _mm256_load_pd(Bp[r] + 4);

        const __m256d a0 = _mm256_broadcast_sd(Ap[r] + 0);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);

        if constexpr (MR >= 2) {
            const __m256d a1 = _mm256_broadcast_sd(Ap[r] + 1);
            c10 = _mm256_fmadd_pd(a1, b0, c10);
            c11 = _mm256_fmadd_pd(a1, b1, c11);
        }
        if constexpr (MR >= 3) {
            const __m256d a2 = _mm256_broadcast_sd(Ap[r] + 2);
            c20 = _mm256_fmadd_pd(a2, b0, c20);
            c21 = _mm256_fmadd_pd(a2, b1, c21);
        }
        if constexpr (MR >= 4) {
            const __m256d a3 = _mm256_broadcast_sd(Ap[r] + 3);
            c30 = _mm256_fmadd_pd(a3, b0, c30);
            c31 = _mm256_fmadd_pd(a3, b1, c31);
        }
    }

    _mm256_store_pd(C + 0 * ldc + 0, c00);
    _mm256_store_pd(C + 0 * ldc + 4, c01);
    if constexpr (MR >= 2) {
        _mm256_store_pd(C + 1 * ldc + 0, c10);
        _mm256_store_pd(C + 1 * ldc + 4, c11);
    }
    if constexpr (MR >= 3) {
        _mm256_store_pd(C + 2 * ldc + 0, c20);
        _mm256_store_pd(C + 2 * ldc + 4, c21);
    }
    if constexpr (MR >= 4) {
        _mm256_store_pd(C + 3 * ldc + 0, c30);
        _mm256_store_pd(C + 3 * ldc + 4, c31);
    }
}

template <size_t MR>
inline void microkernel_mr_x_4(double* CAUSALSIEVE_RESTRICT C,
                               size_t ldc,
                               const double (*CAUSALSIEVE_RESTRICT Ap)[4],
                               const double* CAUSALSIEVE_RESTRICT Bp,
                               size_t b_stride,
                               size_t rb) noexcept {
    __m256d c0 = _mm256_load_pd(C + 0 * ldc);
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    if constexpr (MR >= 2) c1 = _mm256_load_pd(C + 1 * ldc);
    if constexpr (MR >= 3) c2 = _mm256_load_pd(C + 2 * ldc);
    if constexpr (MR >= 4) c3 = _mm256_load_pd(C + 3 * ldc);

    for (size_t r = 0; r < rb; ++r) {
        const __m256d b0 = _mm256_load_pd(Bp + r * b_stride);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(Ap[r] + 0), b0, c0);
        if constexpr (MR >= 2) c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(Ap[r] + 1), b0, c1);
        if constexpr (MR >= 3) c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(Ap[r] + 2), b0, c2);
        if constexpr (MR >= 4) c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(Ap[r] + 3), b0, c3);
    }

    _mm256_store_pd(C + 0 * ldc, c0);
    if constexpr (MR >= 2) _mm256_store_pd(C + 1 * ldc, c1);
    if constexpr (MR >= 3) _mm256_store_pd(C + 2 * ldc, c2);
    if constexpr (MR >= 4) _mm256_store_pd(C + 3 * ldc, c3);
}

inline void dispatch_mr_x_8(size_t rem,
                            double* CAUSALSIEVE_RESTRICT C,
                            size_t ldc,
                            const double (*CAUSALSIEVE_RESTRICT Ap)[4],
                            const double (*CAUSALSIEVE_RESTRICT Bp)[8],
                            size_t rb) noexcept {
    switch (rem) {
        case 4: microkernel_mr_x_8<4>(C, ldc, Ap, Bp, rb); break;
        case 3: microkernel_mr_x_8<3>(C, ldc, Ap, Bp, rb); break;
        case 2: microkernel_mr_x_8<2>(C, ldc, Ap, Bp, rb); break;
        default: microkernel_mr_x_8<1>(C, ldc, Ap, Bp, rb); break;
    }
}

inline void dispatch_mr_x_4(size_t rem,
                            double* CAUSALSIEVE_RESTRICT C,
                            size_t ldc,
                            const double (*CAUSALSIEVE_RESTRICT Ap)[4],
                            const double* CAUSALSIEVE_RESTRICT Bp,
                            size_t b_stride,
                            size_t rb) noexcept {
    switch (rem) {
        case 4: microkernel_mr_x_4<4>(C, ldc, Ap, Bp, b_stride, rb); break;
        case 3: microkernel_mr_x_4<3>(C, ldc, Ap, Bp, b_stride, rb); break;
        case 2: microkernel_mr_x_4<2>(C, ldc, Ap, Bp, b_stride, rb); break;
        default: microkernel_mr_x_4<1>(C, ldc, Ap, Bp, b_stride, rb); break;
    }
}

/*
 * Symmetric rank-k update of upper triangle:
 *   A += sum_{r=0..b-1} w[r] * V[r] * V[r]^T
 * V is b x n_pad (row-major, padding lanes zeroed).
 * L1-resident MC=32, KC=32, MR=4, NR=8 register-blocked SYRK.
 */
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
    alignas(64) double B_tile[kKc][8];

    for (size_t r0 = 0; r0 < b; r0 += kKc) {
        const size_t rb = std::min(kKc, b - r0);
        const double* Vr = V + r0 * n_pad;
        const double* wr = w + r0;

        for (size_t i0 = 0; i0 < n; i0 += kMc) {
            const size_t m_blk = std::min(kMc, n - i0);

            /* Pack w[r] * V[r, i0..i0+m_blk-1] into 8 KB L1-resident A_tile */
            for (size_t im = 0; im < m_blk; im += 4) {
                double (*ap)[4] = A_tile[im >> 2];
                const size_t col = i0 + im;
                for (size_t r = 0; r < rb; ++r) {
                    const __m256d vw = _mm256_set1_pd(wr[r]);
                    const __m256d vv = _mm256_load_pd(Vr + r * n_pad + col);
                    _mm256_store_pd(ap[r], _mm256_mul_pd(vw, vv));
                }
            }

            /* Column panels of width NR=8 starting at j = i0 (multiple of 32) */
            size_t j = i0;
            for (; j + 8 <= n_pad; j += 8) {
                for (size_t r = 0; r < rb; ++r) {
                    const double* src = Vr + r * n_pad + j;
                    _mm256_store_pd(B_tile[r] + 0, _mm256_load_pd(src + 0));
                    _mm256_store_pd(B_tile[r] + 4, _mm256_load_pd(src + 4));
                }

                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    if (row <= j) {
                        dispatch_mr_x_8(rem, A + row * n_pad + j, n_pad, A_tile[im >> 2], B_tile, rb);
                    } else if (row == j + 4) {
                        dispatch_mr_x_4(rem, A + row * n_pad + (j + 4), n_pad, A_tile[im >> 2], &B_tile[0][4], 8, rb);
                        break;
                    } else {
                        break;
                    }
                }
            }

            /* Final NR=4 tail column panel when n_pad % 8 == 4 */
            if (j + 4 <= n_pad) {
                for (size_t r = 0; r < rb; ++r) {
                    _mm256_store_pd(B_tile[r] + 0, _mm256_load_pd(Vr + r * n_pad + j));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    if (row > j) break;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_4(rem, A + row * n_pad + j, n_pad, A_tile[im >> 2], &B_tile[0][0], 8, rb);
                }
            }
        }
    }
}

/*
 * Rectangular weighted GEMM update:
 *   M += sum_{r=0..b-1} w[r] * A[r] * B_mat[r]^T
 * M is rows x cols_pad, A is b x rows_pad, B_mat is b x cols_pad.
 */
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
    alignas(64) double B_tile[kKc][8];

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
                    const __m256d vw = _mm256_set1_pd(wr[r]);
                    const __m256d va = _mm256_load_pd(Ar + r * rows_pad + col);
                    _mm256_store_pd(ap[r], _mm256_mul_pd(vw, va));
                }
            }

            size_t j = 0;
            for (; j + 8 <= cols_pad; j += 8) {
                for (size_t r = 0; r < rb; ++r) {
                    const double* src = Br + r * cols_pad + j;
                    _mm256_store_pd(B_tile[r] + 0, _mm256_load_pd(src + 0));
                    _mm256_store_pd(B_tile[r] + 4, _mm256_load_pd(src + 4));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_8(rem, M + row * cols_pad + j, cols_pad, A_tile[im >> 2], B_tile, rb);
                }
            }
            if (j + 4 <= cols_pad) {
                for (size_t r = 0; r < rb; ++r) {
                    _mm256_store_pd(B_tile[r] + 0, _mm256_load_pd(Br + r * cols_pad + j));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_4(rem, M + row * cols_pad + j, cols_pad, A_tile[im >> 2], &B_tile[0][0], 8, rb);
                }
            }
        }
    }
}

/*
 * Fused single-pass L1-resident blocked update for all 3 LiNGAM moment matrices:
 *   M_u2v    += sum_r w[r] * S2[r] * S[r]^T   (full D x D_pad)
 *   M_utanhv += sum_r w[r] * S[r]  * TS[r]^T  (full D x D_pad)
 *   M_uv     += sum_r w[r] * S[r]  * S[r]^T   (upper triangle D x D_pad)
 * Reuses packed S_w and S_col across multiple output matrices inside L1d!
 */
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
    alignas(64) double S_col[kKc][8];
    alignas(64) double TS_col[kKc][8];

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
                    const __m256d vw = _mm256_set1_pd(wr[r]);
                    _mm256_store_pd(s2p[r], _mm256_mul_pd(vw, _mm256_load_pd(S2r + r * D_pad + col)));
                    _mm256_store_pd(sp[r],  _mm256_mul_pd(vw, _mm256_load_pd(Sr  + r * D_pad + col)));
                }
            }

            size_t j = 0;
            for (; j + 8 <= D_pad; j += 8) {
                for (size_t r = 0; r < rb; ++r) {
                    const double* s_src  = Sr  + r * D_pad + j;
                    const double* ts_src = TSr + r * D_pad + j;
                    _mm256_store_pd(S_col[r]  + 0, _mm256_load_pd(s_src  + 0));
                    _mm256_store_pd(S_col[r]  + 4, _mm256_load_pd(s_src  + 4));
                    _mm256_store_pd(TS_col[r] + 0, _mm256_load_pd(ts_src + 0));
                    _mm256_store_pd(TS_col[r] + 4, _mm256_load_pd(ts_src + 4));
                }

                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_8(rem, M_u2v    + row * D_pad + j, D_pad, S2_w_tile[im >> 2], S_col,  rb);
                    dispatch_mr_x_8(rem, M_utanhv + row * D_pad + j, D_pad, S_w_tile[im >> 2],  TS_col, rb);
                    if (row <= j) {
                        dispatch_mr_x_8(rem, M_uv + row * D_pad + j, D_pad, S_w_tile[im >> 2], S_col, rb);
                    } else if (row == j + 4) {
                        dispatch_mr_x_4(rem, M_uv + row * D_pad + (j + 4), D_pad, S_w_tile[im >> 2], &S_col[0][4], 8, rb);
                    }
                }
            }

            if (j + 4 <= D_pad) {
                for (size_t r = 0; r < rb; ++r) {
                    _mm256_store_pd(S_col[r]  + 0, _mm256_load_pd(Sr  + r * D_pad + j));
                    _mm256_store_pd(TS_col[r] + 0, _mm256_load_pd(TSr + r * D_pad + j));
                }
                for (size_t im = 0; im < m_blk; im += 4) {
                    const size_t row = i0 + im;
                    const size_t rem = std::min<size_t>(4, m_blk - im);
                    dispatch_mr_x_4(rem, M_u2v    + row * D_pad + j, D_pad, S2_w_tile[im >> 2], &S_col[0][0],  8, rb);
                    dispatch_mr_x_4(rem, M_utanhv + row * D_pad + j, D_pad, S_w_tile[im >> 2],  &TS_col[0][0], 8, rb);
                    if (row <= j) {
                        dispatch_mr_x_4(rem, M_uv + row * D_pad + j, D_pad, S_w_tile[im >> 2], &S_col[0][0], 8, rb);
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * AVX2+FMA Register-Blocked Cholesky Factorization, Inversion, Multi-RHS Solve
 * ========================================================================= */
inline __m256d hsum4x256_pd(__m256d a0, __m256d a1, __m256d a2, __m256d a3) noexcept {
    const __m256d h01 = _mm256_hadd_pd(a0, a1);
    const __m256d h23 = _mm256_hadd_pd(a2, a3);
    const __m256d lo  = _mm256_permute2f128_pd(h01, h23, 0x20);
    const __m256d hi  = _mm256_permute2f128_pd(h01, h23, 0x31);
    return _mm256_add_pd(lo, hi);
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
        size_t j = 0;
        for (; j + 4 <= i; j += 4) {
            const double* CAUSALSIEVE_RESTRICT L0 = L + (j + 0) * n_pad;
            const double* CAUSALSIEVE_RESTRICT L1 = L + (j + 1) * n_pad;
            const double* CAUSALSIEVE_RESTRICT L2 = L + (j + 2) * n_pad;
            const double* CAUSALSIEVE_RESTRICT L3 = L + (j + 3) * n_pad;

            __m256d d0 = _mm256_setzero_pd();
            __m256d d1 = _mm256_setzero_pd();
            __m256d d2 = _mm256_setzero_pd();
            __m256d d3 = _mm256_setzero_pd();
            for (size_t k = 0; k < j; k += 4) {
                const __m256d vi = _mm256_load_pd(Li + k);
                d0 = _mm256_fmadd_pd(vi, _mm256_load_pd(L0 + k), d0);
                d1 = _mm256_fmadd_pd(vi, _mm256_load_pd(L1 + k), d1);
                d2 = _mm256_fmadd_pd(vi, _mm256_load_pd(L2 + k), d2);
                d3 = _mm256_fmadd_pd(vi, _mm256_load_pd(L3 + k), d3);
            }
            alignas(32) double dots[4];
            _mm256_store_pd(dots, hsum4x256_pd(d0, d1, d2, d3));

            const double l0 = (scale * A_upper[(j + 0) * n_pad + i] - dots[0]) / L0[j + 0];
            Li[j + 0] = l0;
            const double l1 = (scale * A_upper[(j + 1) * n_pad + i] - dots[1] - l0 * L1[j + 0]) / L1[j + 1];
            Li[j + 1] = l1;
            const double l2 = (scale * A_upper[(j + 2) * n_pad + i] - dots[2] - l0 * L2[j + 0] - l1 * L2[j + 1]) / L2[j + 2];
            Li[j + 2] = l2;
            const double l3 = (scale * A_upper[(j + 3) * n_pad + i] - dots[3] - l0 * L3[j + 0] - l1 * L3[j + 1] - l2 * L3[j + 2]) / L3[j + 3];
            Li[j + 3] = l3;
        }
        for (; j < i; ++j) {
            const double* CAUSALSIEVE_RESTRICT Lj = L + j * n_pad;
            __m256d acc0 = _mm256_setzero_pd();
            size_t k = 0;
            for (; k + 4 <= j; k += 4) {
                acc0 = _mm256_fmadd_pd(_mm256_load_pd(Li + k), _mm256_load_pd(Lj + k), acc0);
            }
            double dot = hsum256_pd(acc0);
            for (; k < j; ++k) dot += Li[k] * Lj[k];
            Li[j] = (scale * A_upper[j * n_pad + i] - dot) / Lj[j];
        }

        __m256d n0 = _mm256_setzero_pd();
        __m256d n1 = _mm256_setzero_pd();
        size_t k = 0;
        for (; k + 8 <= i; k += 8) {
            const __m256d v0 = _mm256_load_pd(Li + k);
            const __m256d v1 = _mm256_load_pd(Li + k + 4);
            n0 = _mm256_fmadd_pd(v0, v0, n0);
            n1 = _mm256_fmadd_pd(v1, v1, n1);
        }
        for (; k + 4 <= i; k += 4) {
            const __m256d v0 = _mm256_load_pd(Li + k);
            n0 = _mm256_fmadd_pd(v0, v0, n0);
        }
        double norm2 = hsum256_pd(_mm256_add_pd(n0, n1));
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

    /* Part 1: Compute U = L^{-T} (4 rows of U simultaneously -> 4x fewer loads of L) */
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        double* CAUSALSIEVE_RESTRICT U0 = tmp_U + (j + 0) * n_pad;
        double* CAUSALSIEVE_RESTRICT U1 = tmp_U + (j + 1) * n_pad;
        double* CAUSALSIEVE_RESTRICT U2 = tmp_U + (j + 2) * n_pad;
        double* CAUSALSIEVE_RESTRICT U3 = tmp_U + (j + 3) * n_pad;

        const double* CAUSALSIEVE_RESTRICT Lj0 = L + (j + 0) * n_pad;
        const double* CAUSALSIEVE_RESTRICT Lj1 = L + (j + 1) * n_pad;
        const double* CAUSALSIEVE_RESTRICT Lj2 = L + (j + 2) * n_pad;
        const double* CAUSALSIEVE_RESTRICT Lj3 = L + (j + 3) * n_pad;

        U0[j + 0] = 1.0 / Lj0[j + 0];
        U0[j + 1] = -(Lj1[j + 0] * U0[j + 0]) / Lj1[j + 1];
        U1[j + 1] = 1.0 / Lj1[j + 1];
        U0[j + 2] = -(Lj2[j + 0] * U0[j + 0] + Lj2[j + 1] * U0[j + 1]) / Lj2[j + 2];
        U1[j + 2] = -(Lj2[j + 1] * U1[j + 1]) / Lj2[j + 2];
        U2[j + 2] = 1.0 / Lj2[j + 2];
        U0[j + 3] = -(Lj3[j + 0] * U0[j + 0] + Lj3[j + 1] * U0[j + 1] + Lj3[j + 2] * U0[j + 2]) / Lj3[j + 3];
        U1[j + 3] = -(Lj3[j + 1] * U1[j + 1] + Lj3[j + 2] * U1[j + 2]) / Lj3[j + 3];
        U2[j + 3] = -(Lj3[j + 2] * U2[j + 2]) / Lj3[j + 3];
        U3[j + 3] = 1.0 / Lj3[j + 3];

        for (size_t i = j + 4; i < n; ++i) {
            const double* CAUSALSIEVE_RESTRICT Li = L + i * n_pad;
            __m256d s0 = _mm256_setzero_pd();
            __m256d s1 = _mm256_setzero_pd();
            __m256d s2 = _mm256_setzero_pd();
            __m256d s3 = _mm256_setzero_pd();
            size_t k = j;
            for (; k + 4 <= i; k += 4) {
                const __m256d vL = _mm256_load_pd(Li + k);
                s0 = _mm256_fmadd_pd(vL, _mm256_load_pd(U0 + k), s0);
                s1 = _mm256_fmadd_pd(vL, _mm256_load_pd(U1 + k), s1);
                s2 = _mm256_fmadd_pd(vL, _mm256_load_pd(U2 + k), s2);
                s3 = _mm256_fmadd_pd(vL, _mm256_load_pd(U3 + k), s3);
            }
            alignas(32) double sums[4];
            _mm256_store_pd(sums, hsum4x256_pd(s0, s1, s2, s3));
            for (; k < i; ++k) {
                const double lik = Li[k];
                sums[0] += lik * U0[k];
                sums[1] += lik * U1[k];
                sums[2] += lik * U2[k];
                sums[3] += lik * U3[k];
            }
            const double neg_inv = -1.0 / Li[i];
            U0[i] = sums[0] * neg_inv;
            U1[i] = sums[1] * neg_inv;
            U2[i] = sums[2] * neg_inv;
            U3[i] = sums[3] * neg_inv;
        }
    }
    for (; j < n; ++j) {
        double* CAUSALSIEVE_RESTRICT Uj = tmp_U + j * n_pad;
        Uj[j] = 1.0 / L[j * n_pad + j];
        for (size_t i = j + 1; i < n; ++i) {
            const double* CAUSALSIEVE_RESTRICT Li = L + i * n_pad;
            double sum = 0.0;
            for (size_t k = j; k < i; ++k) sum += Li[k] * Uj[k];
            Uj[i] = -sum / Li[i];
        }
    }

    /* Part 2: Upper triangle P = U * U^T (2x4 register-blocked, 100% 32B aligned!) */
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const double* CAUSALSIEVE_RESTRICT Ui0 = tmp_U + (i + 0) * n_pad;
        const double* CAUSALSIEVE_RESTRICT Ui1 = tmp_U + (i + 1) * n_pad;
        double* CAUSALSIEVE_RESTRICT Pi0 = P_upper + (i + 0) * n_pad;
        double* CAUSALSIEVE_RESTRICT Pi1 = P_upper + (i + 1) * n_pad;

        const size_t j_aligned = (i + 4u) & ~size_t(3u);
        const size_t j_head = std::min(j_aligned, n);
        for (size_t col = i; col < j_head; ++col) {
            const double* CAUSALSIEVE_RESTRICT Uj = tmp_U + col * n_pad;
            __m256d a0 = _mm256_setzero_pd();
            __m256d a1 = _mm256_setzero_pd();
            for (size_t k = (col & ~size_t(3u)); k < n_pad; k += 4) {
                const __m256d vj = _mm256_load_pd(Uj + k);
                a0 = _mm256_fmadd_pd(_mm256_load_pd(Ui0 + k), vj, a0);
                if (col >= i + 1) {
                    a1 = _mm256_fmadd_pd(_mm256_load_pd(Ui1 + k), vj, a1);
                }
            }
            Pi0[col] = hsum256_pd(a0);
            if (col >= i + 1) Pi1[col] = hsum256_pd(a1);
        }

        size_t col = j_aligned;
        for (; col + 4 <= n; col += 4) {
            const double* CAUSALSIEVE_RESTRICT Uj0 = tmp_U + (col + 0) * n_pad;
            const double* CAUSALSIEVE_RESTRICT Uj1 = tmp_U + (col + 1) * n_pad;
            const double* CAUSALSIEVE_RESTRICT Uj2 = tmp_U + (col + 2) * n_pad;
            const double* CAUSALSIEVE_RESTRICT Uj3 = tmp_U + (col + 3) * n_pad;

            __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
            __m256d c02 = _mm256_setzero_pd(), c03 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
            __m256d c12 = _mm256_setzero_pd(), c13 = _mm256_setzero_pd();

            for (size_t k = col; k < n_pad; k += 4) {
                const __m256d u0 = _mm256_load_pd(Ui0 + k);
                const __m256d u1 = _mm256_load_pd(Ui1 + k);
                const __m256d v0 = _mm256_load_pd(Uj0 + k);
                const __m256d v1 = _mm256_load_pd(Uj1 + k);
                const __m256d v2 = _mm256_load_pd(Uj2 + k);
                const __m256d v3 = _mm256_load_pd(Uj3 + k);
                c00 = _mm256_fmadd_pd(u0, v0, c00);
                c10 = _mm256_fmadd_pd(u1, v0, c10);
                c01 = _mm256_fmadd_pd(u0, v1, c01);
                c11 = _mm256_fmadd_pd(u1, v1, c11);
                c02 = _mm256_fmadd_pd(u0, v2, c02);
                c12 = _mm256_fmadd_pd(u1, v2, c12);
                c03 = _mm256_fmadd_pd(u0, v3, c03);
                c13 = _mm256_fmadd_pd(u1, v3, c13);
            }
            _mm256_store_pd(Pi0 + col, hsum4x256_pd(c00, c01, c02, c03));
            _mm256_store_pd(Pi1 + col, hsum4x256_pd(c10, c11, c12, c13));
        }
        for (; col < n; ++col) {
            const double* CAUSALSIEVE_RESTRICT Uj = tmp_U + col * n_pad;
            __m256d a0 = _mm256_setzero_pd();
            __m256d a1 = _mm256_setzero_pd();
            for (size_t k = (col & ~size_t(3u)); k < n_pad; k += 4) {
                const __m256d vj = _mm256_load_pd(Uj + k);
                a0 = _mm256_fmadd_pd(_mm256_load_pd(Ui0 + k), vj, a0);
                a1 = _mm256_fmadd_pd(_mm256_load_pd(Ui1 + k), vj, a1);
            }
            Pi0[col] = hsum256_pd(a0);
            Pi1[col] = hsum256_pd(a1);
        }
    }
    if (i < n) {
        const double* CAUSALSIEVE_RESTRICT Ui = tmp_U + i * n_pad;
        double* CAUSALSIEVE_RESTRICT Pi = P_upper + i * n_pad;
        for (size_t col = i; col < n; ++col) {
            const double* CAUSALSIEVE_RESTRICT Uj = tmp_U + col * n_pad;
            double sum = 0.0;
            for (size_t k = col; k < n; ++k) sum += Ui[k] * Uj[k];
            Pi[col] = sum;
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
    const __m256d vc = _mm256_set1_pd(c_scale);

    /* Forward solve: L * Y = c_scale * C_tilde (register-resident 16-wide tiles -> zero Yi spill!) */
    for (size_t i = 0; i < K; ++i) {
        const double* CAUSALSIEVE_RESTRICT Li = L + i * K_pad;
        const double* CAUSALSIEVE_RESTRICT Ci = C_tilde + i * D_pad;
        double* CAUSALSIEVE_RESTRICT Yi = Y + i * D_pad;
        const __m256d vinv = _mm256_set1_pd(1.0 / Li[i]);

        size_t j = 0;
        for (; j + 16 <= D_pad; j += 16) {
            __m256d y0 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j));
            __m256d y1 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j + 4));
            __m256d y2 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j + 8));
            __m256d y3 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j + 12));
            for (size_t k = 0; k < i; ++k) {
                const __m256d lk = _mm256_set1_pd(Li[k]);
                const double* CAUSALSIEVE_RESTRICT Yk = Y + k * D_pad + j;
                y0 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Yk),      y0);
                y1 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Yk + 4),  y1);
                y2 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Yk + 8),  y2);
                y3 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Yk + 12), y3);
            }
            _mm256_store_pd(Yi + j,      _mm256_mul_pd(y0, vinv));
            _mm256_store_pd(Yi + j + 4,  _mm256_mul_pd(y1, vinv));
            _mm256_store_pd(Yi + j + 8,  _mm256_mul_pd(y2, vinv));
            _mm256_store_pd(Yi + j + 12, _mm256_mul_pd(y3, vinv));
        }
        for (; j + 8 <= D_pad; j += 8) {
            __m256d y0 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j));
            __m256d y1 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j + 4));
            for (size_t k = 0; k < i; ++k) {
                const __m256d lk = _mm256_set1_pd(Li[k]);
                const double* CAUSALSIEVE_RESTRICT Yk = Y + k * D_pad + j;
                y0 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Yk),     y0);
                y1 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Yk + 4), y1);
            }
            _mm256_store_pd(Yi + j,     _mm256_mul_pd(y0, vinv));
            _mm256_store_pd(Yi + j + 4, _mm256_mul_pd(y1, vinv));
        }
        for (; j + 4 <= D_pad; j += 4) {
            __m256d y0 = _mm256_mul_pd(vc, _mm256_load_pd(Ci + j));
            for (size_t k = 0; k < i; ++k) {
                y0 = _mm256_fnmadd_pd(_mm256_set1_pd(Li[k]), _mm256_load_pd(Y + k * D_pad + j), y0);
            }
            _mm256_store_pd(Yi + j, _mm256_mul_pd(y0, vinv));
        }
        for (size_t p = D; p < D_pad; ++p) Yi[p] = 0.0;
    }

    /* Backward solve: L^T * B = Y (register-resident 16-wide tiles -> zero Bi spill!) */
    for (size_t ii = K; ii > 0; --ii) {
        const size_t i = ii - 1;
        double* CAUSALSIEVE_RESTRICT Bi = B + i * D_pad;
        const double* CAUSALSIEVE_RESTRICT Yi = Y + i * D_pad;
        const __m256d vinv = _mm256_set1_pd(1.0 / L[i * K_pad + i]);

        size_t j = 0;
        for (; j + 16 <= D_pad; j += 16) {
            __m256d b0 = _mm256_load_pd(Yi + j);
            __m256d b1 = _mm256_load_pd(Yi + j + 4);
            __m256d b2 = _mm256_load_pd(Yi + j + 8);
            __m256d b3 = _mm256_load_pd(Yi + j + 12);
            for (size_t k = i + 1; k < K; ++k) {
                const __m256d lk = _mm256_set1_pd(L[k * K_pad + i]);
                const double* CAUSALSIEVE_RESTRICT Bk = B + k * D_pad + j;
                b0 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Bk),      b0);
                b1 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Bk + 4),  b1);
                b2 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Bk + 8),  b2);
                b3 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Bk + 12), b3);
            }
            _mm256_store_pd(Bi + j,      _mm256_mul_pd(b0, vinv));
            _mm256_store_pd(Bi + j + 4,  _mm256_mul_pd(b1, vinv));
            _mm256_store_pd(Bi + j + 8,  _mm256_mul_pd(b2, vinv));
            _mm256_store_pd(Bi + j + 12, _mm256_mul_pd(b3, vinv));
        }
        for (; j + 8 <= D_pad; j += 8) {
            __m256d b0 = _mm256_load_pd(Yi + j);
            __m256d b1 = _mm256_load_pd(Yi + j + 4);
            for (size_t k = i + 1; k < K; ++k) {
                const __m256d lk = _mm256_set1_pd(L[k * K_pad + i]);
                const double* CAUSALSIEVE_RESTRICT Bk = B + k * D_pad + j;
                b0 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Bk),     b0);
                b1 = _mm256_fnmadd_pd(lk, _mm256_load_pd(Bk + 4), b1);
            }
            _mm256_store_pd(Bi + j,     _mm256_mul_pd(b0, vinv));
            _mm256_store_pd(Bi + j + 4, _mm256_mul_pd(b1, vinv));
        }
        for (; j + 4 <= D_pad; j += 4) {
            __m256d b0 = _mm256_load_pd(Yi + j);
            for (size_t k = i + 1; k < K; ++k) {
                b0 = _mm256_fnmadd_pd(_mm256_set1_pd(L[k * K_pad + i]), _mm256_load_pd(B + k * D_pad + j), b0);
            }
            _mm256_store_pd(Bi + j, _mm256_mul_pd(b0, vinv));
        }
        for (size_t p = D; p < D_pad; ++p) Bi[p] = 0.0;
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
    const __m256d v_ema = _mm256_set1_pd(ema_alpha);
    const __m256d v_1mema = _mm256_set1_pd(1.0 - ema_alpha);
    const __m256d v_edecay = _mm256_set1_pd(energy_decay);
    const __m256d v_pdecay = _mm256_set1_pd(peak_decay);
    const __m256d v_signmask = _mm256_set1_pd(-0.0);
    const __m256d v_gate = _mm256_set1_pd(3.5);
    const __m256d v_ethresh = _mm256_set1_pd(2.5);
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_1e6 = _mm256_set1_pd(1e-6);
    const __m256d v_1e4 = _mm256_set1_pd(1e-4);
    const __m256d v_pmax = _mm256_set1_pd(0.98);
    const __m256d v_pmin = _mm256_set1_pd(-0.98);
    const __m256d v_clamp_norm = _mm256_set1_pd(6.0);
    const __m256d v_nclamp_norm = _mm256_set1_pd(-6.0);
    __m256d vmax = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= D; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x + i);
        const __m256d vanch = _mm256_load_pd(anchor + i);
        const __m256d vdx = _mm256_sub_pd(vx, vanch);

        __m256d vmean = _mm256_load_pd(mean + i);
        const __m256d vinv_std = _mm256_load_pd(inv_std + i);
        const __m256d vdiff = _mm256_sub_pd(vdx, vmean);
        const __m256d vxnorm = _mm256_mul_pd(vdiff, vinv_std);

        const __m256d vprev = _mm256_load_pd(prev_norm + i);
        const __m256d vphi = _mm256_load_pd(phi + i);
        const __m256d vres = _mm256_fnmadd_pd(vphi, vprev, vxnorm);

        const __m256d vinv_rstd = _mm256_load_pd(inv_res_std + i);
        const __m256d vz = _mm256_mul_pd(vres, vinv_rstd);
        _mm256_store_pd(out_z + i, vz);

        const __m256d vaz = _mm256_andnot_pd(v_signmask, vz);
        vmax = _mm256_max_pd(vmax, vaz);

        const __m256d vex = _mm256_max_pd(v_zero, _mm256_sub_pd(vaz, v_ethresh));
        __m256d ven = _mm256_load_pd(energy + i);
        ven = _mm256_fmadd_pd(v_edecay, ven, _mm256_mul_pd(vex, vex));
        _mm256_store_pd(energy + i, ven);

        __m256d vpk = _mm256_load_pd(recent_peak + i);
        vpk = _mm256_max_pd(_mm256_mul_pd(v_pdecay, vpk), vaz);
        _mm256_store_pd(recent_peak + i, vpk);

        const __m256d vaxnorm = _mm256_andnot_pd(v_signmask, vxnorm);
        const __m256d mask_z = _mm256_cmp_pd(vaz, v_gate, _CMP_LE_OQ);
        const __m256d mask_x = _mm256_cmp_pd(vaxnorm, v_gate, _CMP_LE_OQ);
        const __m256d mask = _mm256_and_pd(mask_z, mask_x);

        const __m256d new_mean = _mm256_fmadd_pd(v_ema, vdiff, vmean);
        __m256d vvar = _mm256_load_pd(var + i);
        const __m256d new_var = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vdiff, vdiff),
                                                _mm256_mul_pd(v_1mema, vvar));
        vmean = _mm256_blendv_pd(vmean, new_mean, mask);
        vvar  = _mm256_blendv_pd(vvar, new_var, mask);
        _mm256_store_pd(mean + i, vmean);
        _mm256_store_pd(var + i, vvar);

        const __m256d vfloor_s = _mm256_load_pd(floor_std + i);
        const __m256d vstd = _mm256_max_pd(_mm256_sqrt_pd(_mm256_max_pd(v_zero, vvar)), vfloor_s);
        _mm256_store_pd(inv_std + i, _mm256_div_pd(v_one, vstd));

        __m256d vacov = _mm256_load_pd(ar_cov + i);
        __m256d vavar = _mm256_load_pd(ar_var + i);
        __m256d vrvar = _mm256_load_pd(ar_res_var + i);
        const __m256d new_acov = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vxnorm, vprev),
                                                 _mm256_mul_pd(v_1mema, vacov));
        const __m256d new_avar = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vprev, vprev),
                                                 _mm256_mul_pd(v_1mema, vavar));
        const __m256d new_rvar = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vres, vres),
                                                 _mm256_mul_pd(v_1mema, vrvar));
        vacov = _mm256_blendv_pd(vacov, new_acov, mask);
        vavar = _mm256_blendv_pd(vavar, new_avar, mask);
        vrvar = _mm256_blendv_pd(vrvar, new_rvar, mask);
        _mm256_store_pd(ar_cov + i, vacov);
        _mm256_store_pd(ar_var + i, vavar);
        _mm256_store_pd(ar_res_var + i, vrvar);

        __m256d new_phi = _mm256_div_pd(vacov, _mm256_max_pd(vavar, v_1e6));
        new_phi = _mm256_max_pd(v_pmin, _mm256_min_pd(v_pmax, new_phi));
        _mm256_store_pd(phi + i, new_phi);
        const __m256d new_rstd = _mm256_sqrt_pd(_mm256_max_pd(vrvar, v_1e4));
        _mm256_store_pd(inv_res_std + i, _mm256_div_pd(v_one, new_rstd));

        const __m256d vprev_next = _mm256_max_pd(v_nclamp_norm, _mm256_min_pd(v_clamp_norm, vxnorm));
        _mm256_store_pd(prev_norm + i, vprev_next);
    }

    alignas(32) double max_lanes[4];
    _mm256_store_pd(max_lanes, vmax);
    double max_abs_z = std::max(std::max(max_lanes[0], max_lanes[1]),
                                std::max(max_lanes[2], max_lanes[3]));

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

} /* namespace avx2_impl */

const KernelOps& get_avx2_ops() noexcept {
    static const KernelOps kOps{
        "avx2+fma",
        &avx2_impl::dot_pd,
        &avx2_impl::sym_matvec_upper,
        &avx2_impl::sym_rank1_update_upper,
        &avx2_impl::matvec_BT_z,
        &avx2_impl::outer_update_rect,
        &avx2_impl::scale_vec,
        &avx2_impl::sub_vec,
        &avx2_impl::all_finite_pd,
        &avx2_impl::sym_rankk_update_upper,
        &avx2_impl::gemm_update_rect,
        &avx2_impl::lingam_batch_update,
        &avx2_impl::cholesky_factor_scaled_upper,
        &avx2_impl::cholesky_invert_to_upper,
        &avx2_impl::cholesky_solve_multi,
        &avx2_impl::screened_step_postwarmup,
    };
    return kOps;
}

} /* namespace causalsieve::kernels */

#else

namespace causalsieve::kernels {

const KernelOps& get_avx2_ops() noexcept {
    return get_scalar_ops();
}

} /* namespace causalsieve::kernels */

#endif /* CAUSALSIEVE_COMPILE_AVX2_IMPL */
