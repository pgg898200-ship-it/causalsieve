#ifndef CAUSALSIEVE_CORE_HPP
#define CAUSALSIEVE_CORE_HPP

#include "causalsieve.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__) && !defined(CAUSALSIEVE_FORCE_SCALAR)
#define CAUSALSIEVE_USE_AVX2 1
#define CAUSALSIEVE_USE_NEON 0
#include <immintrin.h>
#elif defined(__aarch64__) && !defined(CAUSALSIEVE_FORCE_SCALAR)
#define CAUSALSIEVE_USE_AVX2 0
#define CAUSALSIEVE_USE_NEON 1
#include <arm_neon.h>
#else
#define CAUSALSIEVE_USE_AVX2 0
#define CAUSALSIEVE_USE_NEON 0
#endif

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
 * Hand-Tuned AVX2+FMA / ARM64 NEON SIMD Kernels (with Scalar Fallback)
 * ========================================================================= */
namespace simd {

#if CAUSALSIEVE_USE_AVX2
inline double hsum256_pd(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_unpackhi_pd(lo, lo);
    return _mm_cvtsd_f64(_mm_add_sd(lo, shuf));
}
#endif

/* Dot product over padded 4-aligned vectors (padding lanes must be 0.0). */
inline double dot_pd(const double* __restrict__ a,
                     const double* __restrict__ b,
                     size_t n_pad) noexcept {
#if CAUSALSIEVE_USE_AVX2
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
#elif CAUSALSIEVE_USE_NEON
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
#else
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
#endif
}

/*
 * Symmetric matrix-vector product y = A * x where ONLY the upper triangle
 * of A (n x n_pad, row-major) is maintained.
 * x[n..n_pad-1] must be 0.0. y[n..n_pad-1] is zeroed on exit.
 * Single pass over upper triangle cuts L1/L2 memory traffic in half!
 */
inline void sym_matvec_upper(const double* __restrict__ A,
                             const double* __restrict__ x,
                             double* __restrict__ y,
                             size_t n,
                             size_t n_pad) noexcept {
    std::memset(y, 0, n_pad * sizeof(double));
#if CAUSALSIEVE_USE_AVX2
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict__ row = A + i * n_pad;
        const double xi = x[i];
        const size_t j_aligned = (i + 4u) & ~size_t(3u);
        const size_t j_head_end = std::min(j_aligned, n);

        double acc_scalar = row[i] * xi;
        for (size_t j = i + 1; j < j_head_end; ++j) {
            const double aij = row[j];
            acc_scalar += aij * x[j];
            y[j] += aij * xi;
        }

        __m256d vxi = _mm256_set1_pd(xi);
        __m256d vacc0 = _mm256_setzero_pd();
        __m256d vacc1 = _mm256_setzero_pd();
        size_t j = j_aligned;
        for (; j + 8 <= n_pad; j += 8) {
            __m256d a0 = _mm256_load_pd(row + j);
            __m256d a1 = _mm256_load_pd(row + j + 4);
            __m256d x0 = _mm256_load_pd(x + j);
            __m256d x1 = _mm256_load_pd(x + j + 4);
            vacc0 = _mm256_fmadd_pd(a0, x0, vacc0);
            vacc1 = _mm256_fmadd_pd(a1, x1, vacc1);
            __m256d y0 = _mm256_load_pd(y + j);
            __m256d y1 = _mm256_load_pd(y + j + 4);
            _mm256_store_pd(y + j,     _mm256_fmadd_pd(a0, vxi, y0));
            _mm256_store_pd(y + j + 4, _mm256_fmadd_pd(a1, vxi, y1));
        }
        for (; j + 4 <= n_pad; j += 4) {
            __m256d a0 = _mm256_load_pd(row + j);
            __m256d x0 = _mm256_load_pd(x + j);
            vacc0 = _mm256_fmadd_pd(a0, x0, vacc0);
            __m256d y0 = _mm256_load_pd(y + j);
            _mm256_store_pd(y + j, _mm256_fmadd_pd(a0, vxi, y0));
        }
        y[i] += acc_scalar + hsum256_pd(_mm256_add_pd(vacc0, vacc1));
    }
    for (size_t i = n; i < n_pad; ++i) y[i] = 0.0;
#elif CAUSALSIEVE_USE_NEON
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict__ row = A + i * n_pad;
        const double xi = x[i];
        const size_t j_aligned = (i + 2u) & ~size_t(1u);
        const size_t j_head_end = std::min(j_aligned, n);

        double acc_scalar = row[i] * xi;
        for (size_t j = i + 1; j < j_head_end; ++j) {
            const double aij = row[j];
            acc_scalar += aij * x[j];
            y[j] += aij * xi;
        }

        const float64x2_t vxi = vdupq_n_f64(xi);
        float64x2_t vacc0 = vdupq_n_f64(0.0);
        float64x2_t vacc1 = vdupq_n_f64(0.0);
        size_t j = j_aligned;
        for (; j + 4 <= n_pad; j += 4) {
            const float64x2_t a0 = vld1q_f64(row + j);
            const float64x2_t a1 = vld1q_f64(row + j + 2);
            const float64x2_t x0 = vld1q_f64(x + j);
            const float64x2_t x1 = vld1q_f64(x + j + 2);
            vacc0 = vfmaq_f64(vacc0, a0, x0);
            vacc1 = vfmaq_f64(vacc1, a1, x1);
            const float64x2_t y0 = vld1q_f64(y + j);
            const float64x2_t y1 = vld1q_f64(y + j + 2);
            vst1q_f64(y + j,     vfmaq_f64(y0, a0, vxi));
            vst1q_f64(y + j + 2, vfmaq_f64(y1, a1, vxi));
        }
        for (; j + 2 <= n_pad; j += 2) {
            const float64x2_t a0 = vld1q_f64(row + j);
            const float64x2_t x0 = vld1q_f64(x + j);
            vacc0 = vfmaq_f64(vacc0, a0, x0);
            const float64x2_t y0 = vld1q_f64(y + j);
            vst1q_f64(y + j, vfmaq_f64(y0, a0, vxi));
        }
        y[i] += acc_scalar + vaddvq_f64(vaddq_f64(vacc0, vacc1));
    }
    for (size_t i = n; i < n_pad; ++i) y[i] = 0.0;
#else
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict__ row = A + i * n_pad;
        const double xi = x[i];
        double sum = row[i] * xi;
        for (size_t j = i + 1; j < n; ++j) {
            const double aij = row[j];
            sum += aij * x[j];
            y[j] += aij * xi;
        }
        y[i] += sum;
    }
#endif
}

/*
 * Symmetric rank-1 update of upper triangle: A += alpha * v * v^T.
 * Updates from 4-aligned start j0 = (i & ~3) through n_pad for 100% aligned
 * 256-bit loads/stores. v[n..n_pad-1] must be 0.0.
 */
inline void sym_rank1_update_upper(double* __restrict__ A,
                                   const double* __restrict__ v,
                                   double alpha,
                                   size_t n,
                                   size_t n_pad) noexcept {
#if CAUSALSIEVE_USE_AVX2
    for (size_t i = 0; i < n; ++i) {
        double* __restrict__ row = A + i * n_pad;
        const __m256d vs = _mm256_set1_pd(alpha * v[i]);
        size_t j = i & ~size_t(3u);
        for (; j + 8 <= n_pad; j += 8) {
            __m256d a0 = _mm256_load_pd(row + j);
            __m256d a1 = _mm256_load_pd(row + j + 4);
            __m256d v0 = _mm256_load_pd(v + j);
            __m256d v1 = _mm256_load_pd(v + j + 4);
            _mm256_store_pd(row + j,     _mm256_fmadd_pd(vs, v0, a0));
            _mm256_store_pd(row + j + 4, _mm256_fmadd_pd(vs, v1, a1));
        }
        for (; j + 4 <= n_pad; j += 4) {
            __m256d a0 = _mm256_load_pd(row + j);
            __m256d v0 = _mm256_load_pd(v + j);
            _mm256_store_pd(row + j, _mm256_fmadd_pd(vs, v0, a0));
        }
    }
#elif CAUSALSIEVE_USE_NEON
    for (size_t i = 0; i < n; ++i) {
        double* __restrict__ row = A + i * n_pad;
        const float64x2_t vs = vdupq_n_f64(alpha * v[i]);
        size_t j = i & ~size_t(1u);
        for (; j + 4 <= n_pad; j += 4) {
            const float64x2_t a0 = vld1q_f64(row + j);
            const float64x2_t a1 = vld1q_f64(row + j + 2);
            const float64x2_t v0 = vld1q_f64(v + j);
            const float64x2_t v1 = vld1q_f64(v + j + 2);
            vst1q_f64(row + j,     vfmaq_f64(a0, vs, v0));
            vst1q_f64(row + j + 2, vfmaq_f64(a1, vs, v1));
        }
        for (; j + 2 <= n_pad; j += 2) {
            const float64x2_t a0 = vld1q_f64(row + j);
            const float64x2_t v0 = vld1q_f64(v + j);
            vst1q_f64(row + j, vfmaq_f64(a0, vs, v0));
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        double* __restrict__ row = A + i * n_pad;
        const double s = alpha * v[i];
        for (size_t j = i; j < n; ++j) {
            row[j] += s * v[j];
        }
    }
#endif
}

/*
 * Transposed matrix-vector product y = B^T * z where B is K x D_pad (row-major).
 * Uses register blocking over 16 doubles (4 YMM registers) to eliminate inner stores.
 */
inline void matvec_BT_z(const double* __restrict__ B,
                        const double* __restrict__ z,
                        double* __restrict__ y,
                        size_t K,
                        size_t D,
                        size_t D_pad) noexcept {
#if CAUSALSIEVE_USE_AVX2
    size_t j = 0;
    for (; j + 16 <= D_pad; j += 16) {
        __m256d y0 = _mm256_setzero_pd();
        __m256d y1 = _mm256_setzero_pd();
        __m256d y2 = _mm256_setzero_pd();
        __m256d y3 = _mm256_setzero_pd();
        for (size_t i = 0; i < K; ++i) {
            const double* __restrict__ row = B + i * D_pad + j;
            __m256d vz = _mm256_set1_pd(z[i]);
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
    for (; j + 4 <= D_pad; j += 4) {
        __m256d y0 = _mm256_setzero_pd();
        for (size_t i = 0; i < K; ++i) {
            __m256d vz = _mm256_set1_pd(z[i]);
            y0 = _mm256_fmadd_pd(vz, _mm256_load_pd(B + i * D_pad + j), y0);
        }
        _mm256_store_pd(y + j, y0);
    }
    for (size_t k = D; k < D_pad; ++k) y[k] = 0.0;
#elif CAUSALSIEVE_USE_NEON
    size_t j = 0;
    for (; j + 8 <= D_pad; j += 8) {
        float64x2_t y0 = vdupq_n_f64(0.0);
        float64x2_t y1 = vdupq_n_f64(0.0);
        float64x2_t y2 = vdupq_n_f64(0.0);
        float64x2_t y3 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < K; ++i) {
            const double* __restrict__ row = B + i * D_pad + j;
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
    for (; j + 2 <= D_pad; j += 2) {
        float64x2_t y0 = vdupq_n_f64(0.0);
        for (size_t i = 0; i < K; ++i) {
            const float64x2_t vz = vdupq_n_f64(z[i]);
            y0 = vfmaq_f64(y0, vz, vld1q_f64(B + i * D_pad + j));
        }
        vst1q_f64(y + j, y0);
    }
    for (size_t k = D; k < D_pad; ++k) y[k] = 0.0;
#else
    std::memset(y, 0, D_pad * sizeof(double));
    for (size_t i = 0; i < K; ++i) {
        const double* __restrict__ row = B + i * D_pad;
        const double zi = z[i];
        for (size_t j = 0; j < D; ++j) {
            y[j] += zi * row[j];
        }
    }
#endif
}

/*
 * Rectangular rank-1 outer update M += alpha * a * b^T (M is rows x cols_pad).
 * 2-row unrolled so b is loaded half as often.
 */
inline void outer_update_rect(double* __restrict__ M,
                              const double* __restrict__ a,
                              const double* __restrict__ b,
                              double alpha,
                              size_t rows,
                              size_t cols_pad) noexcept {
#if CAUSALSIEVE_USE_AVX2
    size_t i = 0;
    for (; i + 2 <= rows; i += 2) {
        double* __restrict__ r0 = M + i * cols_pad;
        double* __restrict__ r1 = M + (i + 1) * cols_pad;
        const __m256d va0 = _mm256_set1_pd(alpha * a[i]);
        const __m256d va1 = _mm256_set1_pd(alpha * a[i + 1]);
        size_t j = 0;
        for (; j + 8 <= cols_pad; j += 8) {
            __m256d b0 = _mm256_load_pd(b + j);
            __m256d b1 = _mm256_load_pd(b + j + 4);
            _mm256_store_pd(r0 + j,     _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r0 + j + 4, _mm256_fmadd_pd(va0, b1, _mm256_load_pd(r0 + j + 4)));
            _mm256_store_pd(r1 + j,     _mm256_fmadd_pd(va1, b0, _mm256_load_pd(r1 + j)));
            _mm256_store_pd(r1 + j + 4, _mm256_fmadd_pd(va1, b1, _mm256_load_pd(r1 + j + 4)));
        }
        for (; j + 4 <= cols_pad; j += 4) {
            __m256d b0 = _mm256_load_pd(b + j);
            _mm256_store_pd(r0 + j, _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
            _mm256_store_pd(r1 + j, _mm256_fmadd_pd(va1, b0, _mm256_load_pd(r1 + j)));
        }
    }
    for (; i < rows; ++i) {
        double* __restrict__ r0 = M + i * cols_pad;
        const __m256d va0 = _mm256_set1_pd(alpha * a[i]);
        for (size_t j = 0; j + 4 <= cols_pad; j += 4) {
            __m256d b0 = _mm256_load_pd(b + j);
            _mm256_store_pd(r0 + j, _mm256_fmadd_pd(va0, b0, _mm256_load_pd(r0 + j)));
        }
    }
#elif CAUSALSIEVE_USE_NEON
    size_t i = 0;
    for (; i + 2 <= rows; i += 2) {
        double* __restrict__ r0 = M + i * cols_pad;
        double* __restrict__ r1 = M + (i + 1) * cols_pad;
        const float64x2_t va0 = vdupq_n_f64(alpha * a[i]);
        const float64x2_t va1 = vdupq_n_f64(alpha * a[i + 1]);
        size_t j = 0;
        for (; j + 4 <= cols_pad; j += 4) {
            const float64x2_t b0 = vld1q_f64(b + j);
            const float64x2_t b1 = vld1q_f64(b + j + 2);
            vst1q_f64(r0 + j,     vfmaq_f64(vld1q_f64(r0 + j),     va0, b0));
            vst1q_f64(r0 + j + 2, vfmaq_f64(vld1q_f64(r0 + j + 2), va0, b1));
            vst1q_f64(r1 + j,     vfmaq_f64(vld1q_f64(r1 + j),     va1, b0));
            vst1q_f64(r1 + j + 2, vfmaq_f64(vld1q_f64(r1 + j + 2), va1, b1));
        }
        for (; j + 2 <= cols_pad; j += 2) {
            const float64x2_t b0 = vld1q_f64(b + j);
            vst1q_f64(r0 + j, vfmaq_f64(vld1q_f64(r0 + j), va0, b0));
            vst1q_f64(r1 + j, vfmaq_f64(vld1q_f64(r1 + j), va1, b0));
        }
    }
    for (; i < rows; ++i) {
        double* __restrict__ r0 = M + i * cols_pad;
        const float64x2_t va0 = vdupq_n_f64(alpha * a[i]);
        for (size_t j = 0; j + 2 <= cols_pad; j += 2) {
            const float64x2_t b0 = vld1q_f64(b + j);
            vst1q_f64(r0 + j, vfmaq_f64(vld1q_f64(r0 + j), va0, b0));
        }
    }
#else
    for (size_t i = 0; i < rows; ++i) {
        double* __restrict__ r0 = M + i * cols_pad;
        const double ai = alpha * a[i];
        for (size_t j = 0; j < cols_pad; ++j) {
            r0[j] += ai * b[j];
        }
    }
#endif
}

inline void scale_vec(const double* __restrict__ src,
                      double s,
                      double* __restrict__ dst,
                      size_t n_pad) noexcept {
#if CAUSALSIEVE_USE_AVX2
    const __m256d vs = _mm256_set1_pd(s);
    for (size_t i = 0; i + 4 <= n_pad; i += 4) {
        _mm256_store_pd(dst + i, _mm256_mul_pd(_mm256_load_pd(src + i), vs));
    }
#elif CAUSALSIEVE_USE_NEON
    const float64x2_t vs = vdupq_n_f64(s);
    for (size_t i = 0; i + 2 <= n_pad; i += 2) {
        vst1q_f64(dst + i, vmulq_f64(vld1q_f64(src + i), vs));
    }
#else
    for (size_t i = 0; i < n_pad; ++i) dst[i] = src[i] * s;
#endif
}

inline void sub_vec(const double* __restrict__ a,
                    const double* __restrict__ b,
                    double* __restrict__ out,
                    size_t n_pad) noexcept {
#if CAUSALSIEVE_USE_AVX2
    for (size_t i = 0; i + 4 <= n_pad; i += 4) {
        _mm256_store_pd(out + i, _mm256_sub_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
    }
#elif CAUSALSIEVE_USE_NEON
    for (size_t i = 0; i + 2 <= n_pad; i += 2) {
        vst1q_f64(out + i, vsubq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
    }
#else
    for (size_t i = 0; i < n_pad; ++i) out[i] = a[i] - b[i];
#endif
}

/* Tier-2 O(D) AVX2+FMA causal robust pre-whitening + AR(1) residual + anomaly energy.
 * Returns max |z_j| across the D channels at this tick.
 */
inline double screened_step_postwarmup(
    size_t D,
    size_t D_pad,
    const double* __restrict__ x,
    const double* __restrict__ anchor,
    const double* __restrict__ floor_std,
    double* __restrict__ mean,
    double* __restrict__ var,
    double* __restrict__ inv_std,
    double* __restrict__ prev_norm,
    double* __restrict__ ar_cov,
    double* __restrict__ ar_var,
    double* __restrict__ ar_res_var,
    double* __restrict__ phi,
    double* __restrict__ inv_res_std,
    double* __restrict__ energy,
    double* __restrict__ recent_peak,
    double* __restrict__ out_z,
    double ema_alpha,
    double energy_decay,
    double peak_decay
) noexcept {
    (void)D_pad;
    double max_abs_z = 0.0;
#if CAUSALSIEVE_USE_AVX2
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
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vanch = _mm256_load_pd(anchor + i);
        __m256d vdx = _mm256_sub_pd(vx, vanch);

        __m256d vmean = _mm256_load_pd(mean + i);
        __m256d vinv_std = _mm256_load_pd(inv_std + i);
        __m256d vdiff = _mm256_sub_pd(vdx, vmean);
        __m256d vxnorm = _mm256_mul_pd(vdiff, vinv_std);

        __m256d vprev = _mm256_load_pd(prev_norm + i);
        __m256d vphi = _mm256_load_pd(phi + i);
        __m256d vres = _mm256_fnmadd_pd(vphi, vprev, vxnorm); /* xnorm - phi * prev */

        __m256d vinv_rstd = _mm256_load_pd(inv_res_std + i);
        __m256d vz = _mm256_mul_pd(vres, vinv_rstd);
        _mm256_store_pd(out_z + i, vz);

        __m256d vaz = _mm256_andnot_pd(v_signmask, vz);
        vmax = _mm256_max_pd(vmax, vaz);

        /* Excess energy & peak tracking */
        __m256d vex = _mm256_max_pd(v_zero, _mm256_sub_pd(vaz, v_ethresh));
        __m256d ven = _mm256_load_pd(energy + i);
        ven = _mm256_fmadd_pd(v_edecay, ven, _mm256_mul_pd(vex, vex));
        _mm256_store_pd(energy + i, ven);

        __m256d vpk = _mm256_load_pd(recent_peak + i);
        vpk = _mm256_max_pd(_mm256_mul_pd(v_pdecay, vpk), vaz);
        _mm256_store_pd(recent_peak + i, vpk);

        /* Masked Huber update: only adapt baseline & AR(1) when |z| <= 3.5 AND |xnorm| <= 3.5 */
        __m256d vaxnorm = _mm256_andnot_pd(v_signmask, vxnorm);
        __m256d mask_z = _mm256_cmp_pd(vaz, v_gate, _CMP_LE_OQ);
        __m256d mask_x = _mm256_cmp_pd(vaxnorm, v_gate, _CMP_LE_OQ);
        __m256d mask = _mm256_and_pd(mask_z, mask_x);

        /* Update mean & var gently where mask is true */
        __m256d new_mean = _mm256_fmadd_pd(v_ema, vdiff, vmean);
        __m256d vvar = _mm256_load_pd(var + i);
        __m256d new_var = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vdiff, vdiff),
                                          _mm256_mul_pd(v_1mema, vvar));
        vmean = _mm256_blendv_pd(vmean, new_mean, mask);
        vvar  = _mm256_blendv_pd(vvar, new_var, mask);
        _mm256_store_pd(mean + i, vmean);
        _mm256_store_pd(var + i, vvar);

        __m256d vfloor_s = _mm256_load_pd(floor_std + i);
        __m256d vstd = _mm256_max_pd(_mm256_sqrt_pd(_mm256_max_pd(v_zero, vvar)), vfloor_s);
        _mm256_store_pd(inv_std + i, _mm256_div_pd(v_one, vstd));

        /* Update AR(1) stats where mask is true */
        __m256d vacov = _mm256_load_pd(ar_cov + i);
        __m256d vavar = _mm256_load_pd(ar_var + i);
        __m256d vrvar = _mm256_load_pd(ar_res_var + i);
        __m256d new_acov = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vxnorm, vprev),
                                           _mm256_mul_pd(v_1mema, vacov));
        __m256d new_avar = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vprev, vprev),
                                           _mm256_mul_pd(v_1mema, vavar));
        __m256d new_rvar = _mm256_fmadd_pd(v_ema, _mm256_mul_pd(vres, vres),
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
        __m256d new_rstd = _mm256_sqrt_pd(_mm256_max_pd(vrvar, v_1e4));
        _mm256_store_pd(inv_res_std + i, _mm256_div_pd(v_one, new_rstd));

        /* Store clamped prev_norm so extreme impulse doesn't create echo artifact next tick */
        __m256d vprev_next = _mm256_max_pd(v_nclamp_norm, _mm256_min_pd(v_clamp_norm, vxnorm));
        _mm256_store_pd(prev_norm + i, vprev_next);
    }

    alignas(32) double max_lanes[4];
    _mm256_store_pd(max_lanes, vmax);
    max_abs_z = std::max(std::max(max_lanes[0], max_lanes[1]),
                         std::max(max_lanes[2], max_lanes[3]));

    /* Scalar tail for D % 4 != 0 */
    for (; i < D; ++i) {
#elif CAUSALSIEVE_USE_NEON
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
        const float64x2_t vres = vfmsq_f64(vxnorm, vphi, vprev); /* xnorm - phi * prev */

        const float64x2_t vinv_rstd = vld1q_f64(inv_res_std + i);
        const float64x2_t vz = vmulq_f64(vres, vinv_rstd);
        vst1q_f64(out_z + i, vz);

        const float64x2_t vaz = vabsq_f64(vz);
        vmax = vmaxq_f64(vmax, vaz);

        /* Excess energy & peak tracking */
        const float64x2_t vex = vmaxq_f64(v_zero, vsubq_f64(vaz, v_ethresh));
        float64x2_t ven = vld1q_f64(energy + i);
        ven = vfmaq_f64(vmulq_f64(vex, vex), v_edecay, ven);
        vst1q_f64(energy + i, ven);

        float64x2_t vpk = vld1q_f64(recent_peak + i);
        vpk = vmaxq_f64(vmulq_f64(v_pdecay, vpk), vaz);
        vst1q_f64(recent_peak + i, vpk);

        /* Masked Huber update: only adapt baseline & AR(1) when |z| <= 3.5 AND |xnorm| <= 3.5 */
        const float64x2_t vaxnorm = vabsq_f64(vxnorm);
        const uint64x2_t mask_z = vcleq_f64(vaz, v_gate);
        const uint64x2_t mask_x = vcleq_f64(vaxnorm, v_gate);
        const uint64x2_t mask = vandq_u64(mask_z, mask_x);

        /* Update mean & var gently where mask is true */
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

        /* Update AR(1) stats where mask is true */
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

        /* Store clamped prev_norm so extreme impulse doesn't create echo artifact next tick */
        const float64x2_t vprev_next = vmaxq_f64(v_nclamp_norm, vminq_f64(v_clamp_norm, vxnorm));
        vst1q_f64(prev_norm + i, vprev_next);
    }

    alignas(16) double max_lanes[2];
    vst1q_f64(max_lanes, vmax);
    max_abs_z = std::max(max_lanes[0], max_lanes[1]);

    /* Scalar tail for D % 2 != 0 */
    for (; i < D; ++i) {
#else
    (void)D_pad;
    for (size_t i = 0; i < D; ++i) {
#endif
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

} /* namespace simd */

/* =========================================================================
 * Exact Dense Linear Algebra (Cholesky Factorization, Inversion, Solves)
 * ========================================================================= */
namespace linalg {

/*
 * Exact Cholesky factorization A = L * L^T where A is symmetric SPD stored in
 * the upper triangle of A_upper (n x n_pad, scaled by scale + diag_ridge * I).
 * Output L is lower-triangular (n x n_pad, zero above diagonal).
 */
inline bool cholesky_factor_scaled_upper(const double* __restrict__ A_upper,
                                         double scale,
                                         double diag_ridge,
                                         double* __restrict__ L,
                                         size_t n,
                                         size_t n_pad) noexcept {
    std::memset(L, 0, n * n_pad * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double* __restrict__ Li = L + i * n_pad;
        for (size_t j = 0; j < i; ++j) {
            const double* __restrict__ Lj = L + j * n_pad;
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

/*
 * Compute P = L^{-T} L^{-1} into upper triangle of P_upper (n x n_pad),
 * using tmp_U (n x n_pad) to hold U = L^{-T} (upper triangular) so P = U * U^T
 * with contiguous row dot products.
 */
inline void cholesky_invert_to_upper(const double* __restrict__ L,
                                     double* __restrict__ P_upper,
                                     double* __restrict__ tmp_U,
                                     size_t n,
                                     size_t n_pad) noexcept {
    std::memset(tmp_U, 0, n * n_pad * sizeof(double));
    std::memset(P_upper, 0, n * n_pad * sizeof(double));

    /* Compute U = L^{-T}: upper triangular where U[j, i] = (L^{-1})[i, j] for j <= i */
    for (size_t j = 0; j < n; ++j) {
        double* __restrict__ Uj = tmp_U + j * n_pad;
        Uj[j] = 1.0 / L[j * n_pad + j];
        for (size_t i = j + 1; i < n; ++i) {
            const double* __restrict__ Li = L + i * n_pad;
            double sum = 0.0;
            for (size_t k = j; k < i; ++k) {
                sum += Li[k] * Uj[k];
            }
            Uj[i] = -sum / Li[i];
        }
    }

    /* P[i, j] = sum_{k=j}^{n-1} U[i, k] * U[j, k] for i <= j */
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict__ Ui = tmp_U + i * n_pad;
        double* __restrict__ Pi = P_upper + i * n_pad;
        for (size_t j = i; j < n; ++j) {
            const double* __restrict__ Uj = tmp_U + j * n_pad;
            double sum = 0.0;
            for (size_t k = j; k < n; ++k) {
                sum += Ui[k] * Uj[k];
            }
            Pi[j] = sum;
        }
    }
}

/*
 * Solve (L L^T) B = c_scale * C_tilde for B (K x D_pad), storing intermediate
 * Y = L^{-1} (c_scale * C_tilde) in tmp_Y (K x D_pad).
 * Notice ||Y[:, j]||_2^2 == C_j^T B_j exact!
 */
inline void cholesky_solve_multi(const double* __restrict__ L,
                                 const double* __restrict__ C_tilde,
                                 double c_scale,
                                 double* __restrict__ B,
                                 double* __restrict__ Y,
                                 size_t K,
                                 size_t D,
                                 size_t K_pad,
                                 size_t D_pad) noexcept {
    /* Forward solve: L * Y = c_scale * C_tilde */
    for (size_t i = 0; i < K; ++i) {
        const double* __restrict__ Li = L + i * K_pad;
        const double* __restrict__ Ci = C_tilde + i * D_pad;
        double* __restrict__ Yi = Y + i * D_pad;
        for (size_t j = 0; j < D; ++j) {
            Yi[j] = c_scale * Ci[j];
        }
        for (size_t k = 0; k < i; ++k) {
            const double lik = Li[k];
            const double* __restrict__ Yk = Y + k * D_pad;
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

    /* Backward solve: L^T * B = Y */
    for (size_t ii = K; ii > 0; --ii) {
        const size_t i = ii - 1;
        double* __restrict__ Bi = B + i * D_pad;
        const double* __restrict__ Yi = Y + i * D_pad;
        for (size_t j = 0; j < D; ++j) {
            Bi[j] = Yi[j];
        }
        for (size_t k = i + 1; k < K; ++k) {
            const double lki = L[k * K_pad + i];
            const double* __restrict__ Bk = B + k * D_pad;
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
    causalsieve::AlignedBuffer<double> S_u_tilde;     /* D x D_pad (upper triangle) */
    causalsieve::AlignedBuffer<double> Theta_tilde;   /* D x D_pad (upper triangle) */

    /* Online Hyvarinen-Smith LiNGAM state */
    causalsieve::AlignedBuffer<double> u_mean;
    causalsieve::AlignedBuffer<double> u_var;
    causalsieve::AlignedBuffer<double> M_u2v_tilde;    /* D x D_pad */
    causalsieve::AlignedBuffer<double> M_utanhv_tilde; /* D x D_pad */
    causalsieve::AlignedBuffer<double> M_uv_tilde;     /* D x D_pad (upper triangle) */

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
