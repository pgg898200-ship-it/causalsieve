#include "causalsieve.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                                                   \
    do {                                                                                             \
        double _va = (a), _vb = (b);                                                                 \
        double _diff = std::fabs(_va - _vb);                                                         \
        if (!std::isfinite(_diff) || _diff > (tol)) {                                                \
            std::fprintf(stderr, "[FAIL] %s:%d: %s (got %.12g, exp %.12g, diff %.3e > %.3e)\n",      \
                         __FILE__, __LINE__, msg, _va, _vb, _diff, (double)(tol));                   \
            ++g_failures;                                                                            \
        }                                                                                            \
    } while (0)

/* Independent dense SPD linear solver (Cholesky) for exact batch FP64 oracle */
bool oracle_solve_spd(const std::vector<double>& A_in,
                      const std::vector<double>& B_in,
                      size_t n,
                      size_t nrhs,
                      std::vector<double>& X_out) {
    std::vector<double> L(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            double s = A_in[i * n + j];
            for (size_t k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            L[i * n + j] = s / L[j * n + j];
        }
        double d = A_in[i * n + i];
        for (size_t k = 0; k < i; ++k) d -= L[i * n + k] * L[i * n + k];
        if (d <= 0.0) return false;
        L[i * n + i] = std::sqrt(d);
    }

    std::vector<double> Y(n * nrhs, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t r = 0; r < nrhs; ++r) {
            double s = B_in[i * nrhs + r];
            for (size_t k = 0; k < i; ++k) s -= L[i * n + k] * Y[k * nrhs + r];
            Y[i * nrhs + r] = s / L[i * n + i];
        }
    }
    X_out.assign(n * nrhs, 0.0);
    for (size_t ii = n; ii > 0; --ii) {
        size_t i = ii - 1;
        for (size_t r = 0; r < nrhs; ++r) {
            double s = Y[i * nrhs + r];
            for (size_t k = i + 1; k < n; ++k) s -= L[k * n + i] * X_out[k * nrhs + r];
            X_out[i * nrhs + r] = s / L[i * n + i];
        }
    }
    return true;
}

/* Brute-force batch ridge-VAR + explicit reduced-model refits for all D source channels */
void oracle_batch_granger(const std::vector<double>& X,
                          size_t T,
                          uint32_t D,
                          uint32_t p,
                          double lam,
                          double ridge,
                          uint32_t hinge,
                          std::vector<double>& B_out,
                          std::vector<double>& F_out,
                          std::vector<double>& R2_out) {
    const uint32_t m = hinge ? (2u * p) : p;
    const uint32_t K = 1u + m * D;

    /* Shift by tick-0 anchor (matches engine anchor shift) */
    std::vector<double> dx(T * D, 0.0);
    for (size_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < D; ++j) dx[t * D + j] = X[t * D + j] - X[j];
    }

    std::vector<double> G(static_cast<size_t>(K) * K, 0.0);
    std::vector<double> C(static_cast<size_t>(K) * D, 0.0);
    std::vector<double> Syy(D, 0.0);
    double W_eff = 0.0;

    std::vector<double> z(K, 0.0);
    for (size_t t = p; t < T; ++t) {
        const double w = std::pow(lam, static_cast<double>(T - 1 - t));
        W_eff += w;
        z[0] = 1.0;
        for (uint32_t l = 0; l < p; ++l) {
            for (uint32_t c = 0; c < D; ++c) {
                const double v = dx[(t - 1 - l) * D + c];
                z[1 + l * D + c] = v;
                if (hinge) z[1 + p * D + l * D + c] = (v > 1.0) ? (v - 1.0) : 0.0;
            }
        }
        for (uint32_t i = 0; i < K; ++i) {
            for (uint32_t j = 0; j < K; ++j) G[i * K + j] += w * z[i] * z[j];
            for (uint32_t j = 0; j < D; ++j) C[i * D + j] += w * z[i] * dx[t * D + j];
        }
        for (uint32_t j = 0; j < D; ++j) Syy[j] += w * dx[t * D + j] * dx[t * D + j];
    }

    std::vector<double> G_reg = G;
    for (uint32_t i = 0; i < K; ++i) G_reg[i * K + i] += ridge;

    oracle_solve_spd(G_reg, C, K, D, B_out);

    std::vector<double> J_full(D, 0.0);
    for (uint32_t j = 0; j < D; ++j) {
        double cb = 0.0;
        for (uint32_t i = 0; i < K; ++i) cb += C[i * D + j] * B_out[i * D + j];
        J_full[j] = std::max(1e-30, Syy[j] - cb);
    }

    const double df_eff = std::max(1.0, W_eff - static_cast<double>(K));
    F_out.assign(static_cast<size_t>(D) * D, 0.0);
    R2_out.assign(static_cast<size_t>(D) * D, 0.0);

    for (uint32_t src = 0; src < D; ++src) {
        std::vector<uint32_t> keep;
        keep.reserve(K - m);
        for (uint32_t idx = 0; idx < K; ++idx) {
            bool drop = false;
            for (uint32_t l = 0; l < p; ++l) {
                if (idx == 1u + l * D + src) drop = true;
                if (hinge && idx == 1u + p * D + l * D + src) drop = true;
            }
            if (!drop) keep.push_back(idx);
        }
        const size_t Kr = keep.size();
        std::vector<double> Gr(Kr * Kr, 0.0);
        std::vector<double> Cr(Kr * D, 0.0);
        for (size_t a = 0; a < Kr; ++a) {
            for (size_t b = 0; b < Kr; ++b) Gr[a * Kr + b] = G_reg[keep[a] * K + keep[b]];
            for (uint32_t j = 0; j < D; ++j) Cr[a * D + j] = C[keep[a] * D + j];
        }
        std::vector<double> Br;
        oracle_solve_spd(Gr, Cr, Kr, D, Br);

        for (uint32_t dst = 0; dst < D; ++dst) {
            double cb = 0.0;
            for (size_t a = 0; a < Kr; ++a) cb += Cr[a * D + dst] * Br[a * D + dst];
            const double J_red = std::max(1e-30, Syy[dst] - cb);
            const double delta = std::max(0.0, J_red - J_full[dst]);
            F_out[src * D + dst] = (delta / static_cast<double>(m)) / (J_full[dst] / df_eff);
            R2_out[src * D + dst] = delta / (J_full[dst] + delta);
        }
    }
}

void test_atomic_validation_and_causality() {
    std::printf("[TEST] Atomic validation & strict causality...\n");
    cs_config_t cfg{};
    cs_config_default(&cfg, 6, 2);
    cs_engine_t* eng = cs_create(&cfg);
    CHECK(eng != nullptr, "cs_create failed");

    std::mt19937_64 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> X(80 * 6);
    for (double& v : X) v = nd(rng);

    CHECK(cs_push_batch(eng, X.data(), 40, nullptr, nullptr, nullptr) == 0, "valid batch push");
    const uint64_t ticks_before = cs_tick_count(eng);
    std::vector<double> B_before(cs_feature_dim(eng) * 6);
    cs_inspect_state(eng, B_before.data(), nullptr);

    /* Single NaN/Inf rejection with zero mutation */
    std::vector<double> bad = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0, 5.0, 6.0};
    CHECK(cs_push(eng, bad.data(), nullptr, nullptr, nullptr) == -1, "NaN push must fail");
    CHECK(cs_tick_count(eng) == ticks_before, "NaN push must not increment tick");
    CHECK(std::strlen(cs_last_error(eng)) > 0, "NaN push must set last_error");

    bad[2] = std::numeric_limits<double>::infinity();
    CHECK(cs_push(eng, bad.data(), nullptr, nullptr, nullptr) == -1, "Inf push must fail");
    CHECK(cs_tick_count(eng) == ticks_before, "Inf push must not increment tick");

    /* Batch atomic rejection: NaN on row 35 of 40 must mutate ZERO rows */
    std::vector<double> bad_batch(X.begin() + 40 * 6, X.end());
    bad_batch[35 * 6 + 4] = -std::numeric_limits<double>::infinity();
    CHECK(cs_push_batch(eng, bad_batch.data(), 40, nullptr, nullptr, nullptr) == -1, "batch Inf must fail");
    CHECK(cs_tick_count(eng) == ticks_before, "batch Inf must preserve tick_count");

    std::vector<double> B_after(cs_feature_dim(eng) * 6);
    cs_inspect_state(eng, B_after.data(), nullptr);
    CHECK(std::memcmp(B_before.data(), B_after.data(), B_before.size() * sizeof(double)) == 0,
          "batch Inf must perform ZERO state mutation");

    /* Strict causality (future corruption immunity) + push vs push_batch bit-exactness */
    cs_engine_t* e1 = cs_create(&cfg);
    cs_engine_t* e2 = cs_create(&cfg);
    cs_engine_t* e3 = cs_create(&cfg);
    std::vector<double> X_corrupt = X;
    for (size_t t = 55; t < 80; ++t) {
        for (size_t j = 0; j < 6; ++j) X_corrupt[t * 6 + j] += 1e7;
    }

    std::vector<cs_tick_result_t> r1(80), r2(80), r3(80);
    std::vector<double> ez1(80 * 6), ez2(80 * 6), ez3(80 * 6);
    std::vector<double> in1(80 * 6), in2(80 * 6), in3(80 * 6);

    CHECK(cs_push_batch(e1, X.data(), 80, r1.data(), ez1.data(), in1.data()) == 0, "e1 batch");
    CHECK(cs_push_batch(e2, X_corrupt.data(), 80, r2.data(), ez2.data(), in2.data()) == 0, "e2 batch");
    for (size_t t = 0; t < 80; ++t) {
        CHECK(cs_push(e3, X.data() + t * 6, &r3[t], ez3.data() + t * 6, in3.data() + t * 6) == 0, "e3 single");
    }

    for (size_t t = 0; t < 55; ++t) {
        CHECK(std::memcmp(&r1[t], &r2[t], sizeof(cs_tick_result_t)) == 0, "strict causality result bit-exact");
        CHECK(std::memcmp(ez1.data() + t * 6, ez2.data() + t * 6, 6 * sizeof(double)) == 0, "strict causality exo_z bit-exact");
        CHECK(std::memcmp(in1.data() + t * 6, in2.data() + t * 6, 6 * sizeof(double)) == 0, "strict causality innov bit-exact");
    }
    CHECK(std::memcmp(r1.data(), r3.data(), 80 * sizeof(cs_tick_result_t)) == 0, "push vs push_batch result bit-exact");
    CHECK(std::memcmp(ez1.data(), ez3.data(), ez1.size() * sizeof(double)) == 0, "push vs push_batch exo_z bit-exact");

    cs_destroy(eng);
    cs_destroy(e1);
    cs_destroy(e2);
    cs_destroy(e3);
}

void test_h1_exact_oracle() {
    std::printf("[TEST] H1 Exact Block-Schur Granger vs FP64 batch reduced-model refits...\n");
    const uint32_t dims[] = {4, 16, 32};
    const uint32_t lags[] = {1, 2, 3};
    const double lams[] = {1.0, 0.99};

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd(0.0, 1.0);

    double worst_B_err = 0.0;
    double worst_F_rel = 0.0;
    double worst_R2_err = 0.0;
    double worst_freiv = 0.0;

    for (uint32_t D : dims) {
        for (uint32_t p : lags) {
            for (double lam : lams) {
                for (uint32_t hinge = 0; hinge <= 1; ++hinge) {
                    if (D == 32 && p == 3 && hinge == 1) continue; /* keep test runtime fast under ASan */
                    const size_t T = (D <= 16) ? 160 : 220;
                    std::vector<double> X(T * D, 0.0);
                    for (size_t t = 0; t < T; ++t) {
                        for (uint32_t j = 0; j < D; ++j) {
                            double v = nd(rng) * 0.6;
                            if (t >= 1) {
                                v += 0.35 * X[(t - 1) * D + j];
                                if (j > 0) v += 0.45 * X[(t - 1) * D + (j - 1)];
                            }
                            if (t >= 2 && j > 1) {
                                v -= 0.2 * X[(t - 2) * D + (j - 2)];
                            }
                            X[t * D + j] = v;
                        }
                    }

                    cs_config_t cfg{};
                    cs_config_default(&cfg, D, p);
                    cfg.forgetting = lam;
                    cfg.ridge = 1e-2;
                    cfg.precision_ridge = 1e-2;
                    cfg.shock_threshold = 0.0;
                    cfg.normalize_scale = 0;
                    cfg.nonlinear_hinge = hinge;
                    cfg.warmup_ticks = 10;

                    cs_engine_t* eng = cs_create(&cfg);
                    CHECK(eng != nullptr, "cs_create H1");
                    CHECK(cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr) == 0, "push H1");

                    const uint32_t K = cs_feature_dim(eng);
                    std::vector<double> B_cs(static_cast<size_t>(K) * D);
                    std::vector<double> F_cs(static_cast<size_t>(D) * D);
                    std::vector<double> R2_cs(static_cast<size_t>(D) * D);
                    double freiv = 1.0;

                    CHECK(cs_granger_matrix(eng, F_cs.data(), R2_cs.data()) == 0, "granger H1");
                    CHECK(cs_inspect_state(eng, B_cs.data(), &freiv) == 0, "inspect H1");

                    std::vector<double> B_ref, F_ref, R2_ref;
                    oracle_batch_granger(X, T, D, p, lam, cfg.ridge, hinge, B_ref, F_ref, R2_ref);

                    for (size_t i = 0; i < B_cs.size(); ++i) {
                        worst_B_err = std::max(worst_B_err, std::fabs(B_cs[i] - B_ref[i]));
                    }
                    for (size_t i = 0; i < F_cs.size(); ++i) {
                        const double f_rel = std::fabs(F_cs[i] - F_ref[i]) / std::max(1.0, std::fabs(F_ref[i]));
                        worst_F_rel = std::max(worst_F_rel, f_rel);
                        worst_R2_err = std::max(worst_R2_err, std::fabs(R2_cs[i] - R2_ref[i]));
                    }
                    worst_freiv = std::max(worst_freiv, freiv);
                    cs_destroy(eng);
                }
            }
        }
    }

    std::printf("  H1 worst errs: B=%.3e, F_rel=%.3e, R2=%.3e, Freivalds=%.3e\n",
                worst_B_err, worst_F_rel, worst_R2_err, worst_freiv);
    CHECK(worst_B_err < 1e-10, "H1 B exactness < 1e-10");
    CHECK(worst_F_rel < 1e-10, "H1 Granger F exactness < 1e-10");
    CHECK(worst_R2_err < 1e-10, "H1 partial R2 exactness < 1e-10");
    CHECK(worst_freiv < 1e-10, "H1 Freivalds inverse err < 1e-10");
}

void test_erddap_dc_offset_and_zero_channel() {
    std::printf("[TEST] ERDDAP 1e10 DC-offset immunity & constant-zero channel safety...\n");
    const uint32_t D = 6, p = 2;
    const size_t T = 200;
    std::mt19937_64 rng(777);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<double> X0(T * D, 0.0), X_big(T * D, 0.0);
    for (size_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < D; ++j) {
            double v = nd(rng);
            if (t >= 1 && j == 1) v += 0.6 * X0[(t - 1) * D + 0];
            if (j == 5) v = (t < 150) ? 0.0 : 0.05; /* constant-zero error_rate channel */
            X0[t * D + j] = v;
            X_big[t * D + j] = v + ((j == 5) ? 0.0 : 1e10 * (j + 1));
        }
    }

    cs_config_t cfg{};
    cs_config_default(&cfg, D, p);
    cfg.normalize_scale = 1;
    cfg.warmup_ticks = 50;

    cs_engine_t* e0 = cs_create(&cfg);
    cs_engine_t* eb = cs_create(&cfg);
    std::vector<cs_tick_result_t> r0(T), rb(T);
    CHECK(cs_push_batch(e0, X0.data(), T, r0.data(), nullptr, nullptr) == 0, "X0 push");
    CHECK(cs_push_batch(eb, X_big.data(), T, rb.data(), nullptr, nullptr) == 0, "X_big push");

    for (size_t t = 0; t < T; ++t) {
        CHECK(std::isfinite(rb[t].mahalanobis_sq) && std::isfinite(rb[t].e_value) && std::isfinite(rb[t].max_exo_z),
              "1e10 offset produces finite scores");
    }
    /* At t=150 constant-zero channel 5 jumps to 0.05 -> must trigger alarm & top_root=5 */
    CHECK(rb[150].alarm == 1 && rb[150].top_root_channel == 5, "constant-zero channel spike detected cleanly");

    std::vector<double> R2_0(D * D), R2_b(D * D);
    cs_granger_matrix(e0, nullptr, R2_0.data());
    cs_granger_matrix(eb, nullptr, R2_b.data());
    double max_r2_diff = 0.0;
    for (size_t i = 0; i < D * D; ++i) max_r2_diff = std::max(max_r2_diff, std::fabs(R2_0[i] - R2_b[i]));
    std::printf("  ERDDAP 1e10 offset max partial_r2 diff = %.3e\n", max_r2_diff);
    CHECK(max_r2_diff < 1e-4, "ERDDAP 1e10 DC-offset invariant");

    cs_destroy(e0);
    cs_destroy(eb);
}

void test_precision_lingam_and_rca() {
    std::printf("[TEST] Precision whitening, Hyvarinen-Smith LiNGAM, shock shield & RCA...\n");
    const uint32_t D = 6, p = 1;

    /* 1. Skewed Exponential instantaneous cascade: 0 -> 1 -> 2, and negative edge 0 -> 3 */
    {
        std::mt19937_64 rng(9001);
        std::exponential_distribution<double> expd(1.0);
        const size_t T = 2500;
        std::vector<double> X(T * D, 0.0);
        for (size_t t = 0; t < T; ++t) {
            const double e0 = expd(rng) - 1.0;
            const double e1 = expd(rng) - 1.0;
            const double e2 = expd(rng) - 1.0;
            const double e3 = expd(rng) - 1.0;
            const double e4 = expd(rng) - 1.0;
            const double e5 = expd(rng) - 1.0;
            const double x0 = e0;
            const double x1 = 0.75 * x0 + 0.66 * e1;
            const double x2 = 0.75 * x1 + 0.66 * e2;
            const double x3 = -0.70 * x0 + 0.71 * e3;
            X[t * D + 0] = x0;
            X[t * D + 1] = x1;
            X[t * D + 2] = x2;
            X[t * D + 3] = x3;
            X[t * D + 4] = e4;
            X[t * D + 5] = e5;
        }
        cs_config_t cfg{};
        cs_config_default(&cfg, D, p);
        cfg.forgetting = 1.0;
        cfg.shock_threshold = 0.0;
        cfg.warmup_ticks = 50;
        cs_engine_t* eng = cs_create(&cfg);
        cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr);

        std::vector<double> R(D * D, 0.0);
        cs_contemporaneous_orientation(eng, R.data());
        std::printf("  LiNGAM (Exponential): R[0->1]=%.4f, R[1->2]=%.4f, R[0->3]=%.4f\n",
                    R[0 * D + 1], R[1 * D + 2], R[0 * D + 3]);
        CHECK(R[0 * D + 1] > 0.05, "Exp LiNGAM 0->1 > 0");
        CHECK(R[1 * D + 2] > 0.05, "Exp LiNGAM 1->2 > 0");
        CHECK(R[0 * D + 3] > 0.05, "Exp LiNGAM negative-coef 0->3 > 0");
        for (uint32_t i = 0; i < D; ++i) {
            CHECK(R[i * D + i] == 0.0, "LiNGAM zero diag");
            for (uint32_t j = 0; j < D; ++j) {
                CHECK_NEAR(R[i * D + j], -R[j * D + i], 1e-15, "LiNGAM antisymmetry");
            }
        }
        cs_destroy(eng);
    }

    /* 2. Symmetric Heavy-Tailed Student-t4 instantaneous cascade: 0 -> 1 -> 2 (ZERO skewness!)
     * Uses STL-independent Box-Muller normal + chi^2(4) driven directly by mt19937_64 averaged
     * over 4 independent streams so draws are identical across GNU libstdc++, Apple libc++, and
     * MSVC STL while eliminating single-sample heavy-tail variance in r_skew. */
    {
        double mean_01 = 0.0, mean_12 = 0.0, mean_02 = 0.0;
        constexpr int N_STREAMS = 8;
        const size_t T = 6000;
        for (int s_idx = 0; s_idx < N_STREAMS; ++s_idx) {
            std::mt19937_64 rng(9002ULL + static_cast<uint64_t>(s_idx));
            auto draw_u01 = [&rng]() -> double {
                return (static_cast<double>(rng() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
            };
            auto draw_norm = [&draw_u01]() -> double {
                const double u1 = std::max(1e-15, draw_u01());
                const double u2 = draw_u01();
                return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.28318530717958647692 * u2);
            };
            auto draw_t4 = [&draw_norm]() -> double {
                const double z = draw_norm();
                double chi2 = 0.0;
                for (int k = 0; k < 4; ++k) {
                    const double g = draw_norm();
                    chi2 += g * g;
                }
                return z / std::sqrt(chi2 * 0.25);
            };
            std::vector<double> X(T * D, 0.0);
            for (size_t t = 0; t < T; ++t) {
                const double e0 = draw_t4();
                const double e1 = draw_t4();
                const double e2 = draw_t4();
                const double x0 = e0;
                const double x1 = 0.70 * x0 + 0.71 * e1;
                const double x2 = 0.70 * x1 + 0.71 * e2;
                X[t * D + 0] = x0;
                X[t * D + 1] = x1;
                X[t * D + 2] = x2;
                X[t * D + 3] = draw_t4();
                X[t * D + 4] = draw_t4();
                X[t * D + 5] = draw_t4();
            }
            cs_config_t cfg{};
            cs_config_default(&cfg, D, p);
            cfg.forgetting = 1.0;
            cfg.shock_threshold = 0.0;
            cfg.warmup_ticks = 50;
            cs_engine_t* eng = cs_create(&cfg);
            cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr);

            std::vector<double> R(D * D, 0.0);
            cs_contemporaneous_orientation(eng, R.data());
            mean_01 += R[0 * D + 1] / static_cast<double>(N_STREAMS);
            mean_12 += R[1 * D + 2] / static_cast<double>(N_STREAMS);
            mean_02 += R[0 * D + 2] / static_cast<double>(N_STREAMS);
            cs_destroy(eng);
        }
        std::printf("  LiNGAM (Symmetric Student-t4, 8-stream mean): R[0->1]=%.4f, R[1->2]=%.4f, R[0->2]=%.4f\n",
                    mean_01, mean_12, mean_02);
        CHECK(mean_01 > 0.01, "Student-t LiNGAM 0->1 > 0");
        CHECK(mean_12 > 0.005, "Student-t LiNGAM 1->2 > 0");
        CHECK(mean_02 > 0.01, "Student-t LiNGAM 0->2 > 0");
    }

    /* 3. Lagged cascade with 3x downstream victim amplification + Huber shock shield non-poisoning */
    {
        std::mt19937_64 rng(9003);
        std::normal_distribution<double> nd(0.0, 0.5);
        const size_t T_clean = 300;
        const size_t T_fault = 40;
        const size_t T = T_clean + T_fault;
        std::vector<double> X(T * D, 0.0);
        for (size_t t = 0; t < T; ++t) {
            double e0 = nd(rng);
            if (t >= T_clean) e0 += 6.0; /* Root cause shock on ch 0 */
            X[t * D + 0] = (t > 0 ? 0.3 * X[(t - 1) * D + 0] : 0.0) + e0;
            /* Downstream ch 1 amplifies ch 0 by 2.5x! Raw Z-score will pick ch 1 or ch 2! */
            X[t * D + 1] = (t > 0 ? 2.5 * X[(t - 1) * D + 0] : 0.0) + nd(rng);
            X[t * D + 2] = (t > 0 ? 1.5 * X[(t - 1) * D + 1] : 0.0) + nd(rng);
            for (uint32_t j = 3; j < D; ++j) X[t * D + j] = nd(rng) * 2.0;
        }

        cs_config_t cfg{};
        cs_config_default(&cfg, D, 2);
        cfg.warmup_ticks = 60;
        cs_engine_t* eng = cs_create(&cfg);
        std::vector<cs_tick_result_t> res(T);
        cs_push_batch(eng, X.data(), T_clean, res.data(), nullptr, nullptr);

        std::vector<double> B_clean(cs_feature_dim(eng) * D);
        cs_inspect_state(eng, B_clean.data(), nullptr);

        /* Now inject huge 100-sigma shock for 10 ticks to verify shock shield w_t == 0 */
        cs_engine_t* eng_shield = cs_create(&cfg);
        cs_push_batch(eng_shield, X.data(), T_clean, nullptr, nullptr, nullptr);
        std::vector<double> B_pre(cs_feature_dim(eng_shield) * D);
        cs_inspect_state(eng_shield, B_pre.data(), nullptr);
        for (int s = 0; s < 10; ++s) {
            double shock_x[6] = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            cs_tick_result_t sr{};
            cs_push(eng_shield, shock_x, &sr, nullptr, nullptr);
            CHECK(sr.shock_weight == 0.0, "extreme 100-sigma shock must clamp w_t == 0.0");
            CHECK(sr.alarm == 1, "extreme shock must alarm");
        }
        std::vector<double> B_post(cs_feature_dim(eng_shield) * D);
        cs_inspect_state(eng_shield, B_post.data(), nullptr);
        double b_drift = 0.0;
        for (size_t i = 0; i < B_pre.size(); ++i) b_drift = std::max(b_drift, std::fabs(B_post[i] - B_pre[i]));
        CHECK(b_drift < 1e-12, "shock shield guarantees ZERO baseline B poisoning");
        cs_destroy(eng_shield);

        /* Push actual cascade fault into eng and verify Root Cause Attribution ranks ch 0 #1 */
        cs_push_batch(eng, X.data() + T_clean * D, T_fault, res.data() + T_clean, nullptr, nullptr);
        std::vector<double> rscores(D), pexo(D), praw(D);
        std::vector<int64_t> onset(D);
        std::vector<int32_t> ranked(D);
        cs_root_attribution(eng, 0, rscores.data(), pexo.data(), praw.data(), onset.data(), ranked.data());
        std::printf("  Cascade RCA: ranked[0]=%d (score=%.2f, exo=%.2f, raw=%.2f), victim1(exo=%.2f, raw=%.2f)\n",
                    ranked[0], rscores[ranked[0]], pexo[ranked[0]], praw[ranked[0]], pexo[1], praw[1]);
        CHECK(ranked[0] == 0, "RCA must rank true root 0 above amplified victims 1 and 2");
        CHECK(res.back().e_value > 100.0, "ONS E-process must cross 1/alpha during fault");

        /* Counterfactual do-intervention on ch 0 propagates to ch 1 at h=1 and ch 2 at h=2 */
        std::vector<double> traj(5 * D, 0.0);
        CHECK(cs_counterfactual_do(eng, 0, 1.0, 5, traj.data()) == 0, "counterfactual_do");
        CHECK_NEAR(traj[0 * D + 0], 1.0, 1e-12, "do-intervention h=0 ch 0");
        CHECK(traj[1 * D + 1] > 0.5, "do-intervention propagates ch 0 -> ch 1 at h=1");
        CHECK(traj[2 * D + 2] > 0.5, "do-intervention propagates ch 0 -> ch 2 at h=2");

        cs_destroy(eng);
    }

    /* 4. Instantaneous middle-node shock P(0) -> A(1) -> B(2): shock hits A(1), silent parent P(0) */
    {
        std::mt19937_64 rng(9004);
        std::normal_distribution<double> nd(0.0, 1.0);
        const size_t T_clean = 300, T_fault = 20, T = T_clean + T_fault;
        std::vector<double> X(T * D, 0.0);
        for (size_t t = 0; t < T; ++t) {
            const double p_val = nd(rng);
            const double shock_a = (t >= T_clean) ? 10.0 : 0.0;
            const double a_val = 0.8 * p_val + 0.5 * nd(rng) + shock_a;
            const double b_val = 1.5 * a_val + 0.5 * nd(rng);
            X[t * D + 0] = p_val;
            X[t * D + 1] = a_val;
            X[t * D + 2] = b_val;
            for (uint32_t j = 3; j < D; ++j) X[t * D + j] = nd(rng);
        }
        cs_config_t cfg{};
        cs_config_default(&cfg, D, 1);
        cfg.warmup_ticks = 60;
        cs_engine_t* eng = cs_create(&cfg);
        cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr);
        std::vector<int32_t> ranked(D);
        std::vector<double> pexo(D), praw(D);
        cs_root_attribution(eng, 30, nullptr, pexo.data(), praw.data(), nullptr, ranked.data());
        std::printf("  Instantaneous mid-node RCA (0->1->2, shock@1): ranked[0]=%d (exo: P=%.2f A=%.2f B=%.2f)\n",
                    ranked[0], pexo[0], pexo[1], pexo[2]);
        CHECK(ranked[0] == 1, "Mid-node instantaneous shock must rank A(1) #1 over silent parent P(0) and victim B(2)");
        cs_destroy(eng);
    }
}

void test_ons_eprocess_null_calibration() {
    std::printf("[TEST] Anytime-valid ONS E-process Ville null bound (100 null streams)...\n");
    const uint32_t D = 8, p = 1;
    const size_t T = 300;
    const int n_trials = 100;
    int false_e_alarms = 0;

    std::mt19937_64 rng(54321);
    std::normal_distribution<double> nd(0.0, 1.0);

    for (int trial = 0; trial < n_trials; ++trial) {
        cs_config_t cfg{};
        cs_config_default(&cfg, D, p);
        cfg.e_alpha = 0.01;
        cfg.warmup_ticks = 60;
        cs_engine_t* eng = cs_create(&cfg);
        std::vector<double> X(T * D);
        for (double& v : X) v = nd(rng);
        std::vector<cs_tick_result_t> res(T);
        cs_push_batch(eng, X.data(), T, res.data(), nullptr, nullptr);
        bool crossed = false;
        for (size_t t = 60; t < T; ++t) {
            if (res[t].e_value >= 100.0) crossed = true;
        }
        if (crossed) ++false_e_alarms;
        cs_destroy(eng);
    }
    std::printf("  ONS E-process null crossings >= 100: %d / %d (expected <= 1..3)\n", false_e_alarms, n_trials);
    CHECK(false_e_alarms <= 5, "ONS E-process satisfies anytime-valid Ville bound");
}

void test_lossless_ring_and_swing_pla() {
    std::printf("[TEST] Lossless raw ring wrap-around & exact L_inf Swing-Filter PLA...\n");
    cs_config_t cfg{};
    cs_config_default(&cfg, 4, 1);
    cfg.raw_ring_capacity = 256;
    cs_engine_t* eng = cs_create(&cfg);

    const size_t T = 600;
    std::vector<double> X(T * 4, 0.0);
    std::mt19937_64 rng(31415);
    std::normal_distribution<double> nd(0.0, 0.3);
    for (size_t t = 0; t < T; ++t) {
        X[t * 4 + 0] = std::sin(0.07 * static_cast<double>(t)) * 10.0 + nd(rng);
        X[t * 4 + 1] = (t < 450) ? 5.0 : 25.0; /* step fault */
        X[t * 4 + 2] = 1e10 + 0.02 * static_cast<double>(t) + std::cos(0.1 * static_cast<double>(t));
        X[t * 4 + 3] = 42.0; /* constant */
    }
    CHECK(cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr) == 0, "push PLA stream");

    /* Verify ring wrap-around retains exact [344, 600) with 0.0 error */
    std::vector<double> slice(T, 0.0);
    uint64_t cnt = 0, actual_start = 0;
    CHECK(cs_get_raw_slice(eng, 0, 0, T, slice.data(), &cnt, &actual_start) == 0, "get_raw_slice");
    CHECK(actual_start == T - 256 && cnt == 256, "ring wrap-around bounds exact");
    for (uint64_t i = 0; i < cnt; ++i) {
        CHECK(slice[i] == X[(actual_start + i) * 4 + 0], "lossless slice bit-exact FP64");
    }

    /* Fuzz Swing-Filter L_inf guarantee across all 4 channels and multiple error tolerances */
    const double errs[] = {0.0, 1e-6, 1e-2, 0.5, 2.0};
    std::vector<uint64_t> kt(512);
    std::vector<double> kv(512);
    for (uint32_t ch = 0; ch < 4; ++ch) {
        cs_get_raw_slice(eng, ch, actual_start, T, slice.data(), &cnt, &actual_start);
        for (double eps : errs) {
            size_t nk = 0;
            CHECK(cs_compress_channel_pla(eng, ch, actual_start, T, eps,
                                          kt.data(), kv.data(), kt.size(), &nk) == 0,
                  "compress_channel_pla");
            CHECK(nk >= 2 && kt[0] == actual_start && kt[nk - 1] == T - 1, "PLA endpoint ticks");

            /* Reconstruct EVERY single integer tick and verify |y - y_hat| <= eps */
            size_t seg = 0;
            double max_err_seen = 0.0;
            for (uint64_t i = 0; i < cnt; ++i) {
                const uint64_t tick_i = actual_start + i;
                while (seg + 1 < nk && kt[seg + 1] < tick_i) ++seg;
                const double t0 = static_cast<double>(kt[seg]);
                const double t1 = static_cast<double>(kt[seg + 1]);
                const double v0 = kv[seg];
                const double v1 = kv[seg + 1];
                const double yhat = (t1 > t0) ? (v0 + (v1 - v0) * ((static_cast<double>(tick_i) - t0) / (t1 - t0))) : v0;
                const double e = std::fabs(slice[i] - yhat);
                if (e > max_err_seen) max_err_seen = e;
                CHECK(e <= eps + 1e-14 * std::max(1.0, std::fabs(slice[i])),
                      "Swing-Filter L_inf guarantee violated!");
            }
            if (ch == 3 && eps >= 1e-6) {
                CHECK(nk == 2, "constant line must compress to 2 knots");
            }
        }
    }
    cs_destroy(eng);
}

void test_tier2_screened_loud_decoys() {
    std::printf("[TEST] H4 Tier-2 Screened Engine (D=2048 loud-decoy firehose)...\n");
    const uint32_t D = 2048;
    const uint32_t M = 32;
    const uint32_t p = 2;
    const size_t T_clean = 220;
    const size_t T_fault = 40;
    const size_t T = T_clean + T_fault;

    const int trials = 5;
    int top1_hits = 0;

    for (int tr = 0; tr < trials; ++tr) {
        std::mt19937_64 rng(20000 + tr);
        std::normal_distribution<double> nd(0.0, 1.0);
        const uint32_t root = 17 + tr * 100;
        const uint32_t v1 = root + 1;
        const uint32_t v2 = root + 2;

        std::vector<double> X(T * D, 0.0);
        for (size_t t = 0; t < T; ++t) {
            for (uint32_t j = 0; j < D; ++j) {
                /* 3x-5x loud heteroscedastic AR(1) decoys everywhere! */
                const double noise_scale = (j % 3 == 0) ? 4.0 : ((j % 3 == 1) ? 2.5 : 1.0);
                const double phi_j = (j % 5 == 0) ? 0.85 : 0.2;
                double val = (t > 0 ? phi_j * X[(t - 1) * D + j] : 0.0) + noise_scale * nd(rng);
                X[t * D + j] = val;
            }
            /* Causal cluster: root -> v1 (2.5x amplified) -> v2 */
            double r_shock = (t >= T_clean) ? 7.0 : 0.0;
            X[t * D + root] = (t > 0 ? 0.3 * X[(t - 1) * D + root] : 0.0) + nd(rng) + r_shock;
            X[t * D + v1]   = (t > 0 ? 2.2 * X[(t - 1) * D + root] : 0.0) + nd(rng);
            X[t * D + v2]   = (t > 0 ? 1.5 * X[(t - 1) * D + v1] : 0.0) + nd(rng);
        }

        cs_screened_t* sc = cs_screened_create(D, M, p, 0.995, 1e-2);
        CHECK(sc != nullptr, "cs_screened_create");
        std::vector<int32_t> top_ch(5, -1);
        std::vector<double> top_sc(5, 0.0);
        CHECK(cs_screened_push_batch(sc, X.data(), T, nullptr, top_ch.data(), top_sc.data(), 5) == 0,
              "cs_screened_push_batch");

        if (top_ch[0] == static_cast<int32_t>(root)) ++top1_hits;

        std::vector<int32_t> act_ch(M);
        std::vector<double> gf(M * M), pc(M * M);
        uint32_t act_cnt = 0;
        CHECK(cs_screened_active_graph(sc, act_ch.data(), gf.data(), pc.data(), &act_cnt) == 0, "active_graph");
        CHECK(act_cnt == M, "active_count == M");

        cs_screened_destroy(sc);
    }
    std::printf("  H4 Tier-2 D=2048 loud-decoy Top-1 recall: %d / %d\n", top1_hits, trials);
    CHECK(top1_hits >= trials - 1, "Tier-2 loud-decoy Top-1 recall >= 80% (expect 100%)");
}

void test_scalar_vs_avx2_so_parity(const char* avx_so, const char* scalar_so) {
    std::printf("[TEST] Dynamic Scalar vs AVX2+FMA exact parity (%s vs %s)...\n", avx_so, scalar_so);
    void* h1 = dlopen(avx_so, RTLD_NOW | RTLD_LOCAL);
    void* h2 = dlopen(scalar_so, RTLD_NOW | RTLD_LOCAL);
    CHECK(h1 != nullptr && h2 != nullptr, "dlopen both .so libs");
    if (!h1 || !h2) return;

    auto load_fn = [](void* h, const char* name) { return dlsym(h, name); };
    using fn_isa = const char* (*)(void);
    using fn_def = void (*)(cs_config_t*, uint32_t, uint32_t);
    using fn_create = cs_engine_t* (*)(const cs_config_t*);
    using fn_destroy = void (*)(cs_engine_t*);
    using fn_batch = int (*)(cs_engine_t*, const double*, size_t, cs_tick_result_t*, double*, double*);
    using fn_granger = int (*)(cs_engine_t*, double*, double*);
    using fn_prec = int (*)(cs_engine_t*, double*, double*);
    using fn_lingam = int (*)(cs_engine_t*, double*);
    using fn_inspect = int (*)(cs_engine_t*, double*, double*);

    auto isa1 = reinterpret_cast<fn_isa>(load_fn(h1, "cs_simd_isa"));
    auto isa2 = reinterpret_cast<fn_isa>(load_fn(h2, "cs_simd_isa"));
    std::printf("  Loaded ISAs: [%s] vs [%s]\n", isa1(), isa2());
    CHECK(std::strcmp(isa2(), "scalar") == 0, "scalar .so reports scalar ISA");

    const uint32_t D = 17, p = 2; /* intentionally non-multiple of 4 to stress SIMD tails! */
    const size_t T = 300;
    std::mt19937_64 rng(2026);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> X(T * D, 0.0);
    for (size_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < D; ++j) {
            double v = nd(rng);
            if (t > 0 && j > 0) v += 0.4 * X[(t - 1) * D + (j - 1)];
            if (t == 250 && j == 3) v += 8.0;
            X[t * D + j] = v;
        }
    }

    cs_config_t cfg{};
    reinterpret_cast<fn_def>(load_fn(h1, "cs_config_default"))(&cfg, D, p);
    cfg.forgetting = 0.99;
    cfg.warmup_ticks = 50;

    cs_engine_t* e1 = reinterpret_cast<fn_create>(load_fn(h1, "cs_create"))(&cfg);
    cs_engine_t* e2 = reinterpret_cast<fn_create>(load_fn(h2, "cs_create"))(&cfg);

    std::vector<cs_tick_result_t> r1(T), r2(T);
    std::vector<double> ez1(T * D), ez2(T * D);
    reinterpret_cast<fn_batch>(load_fn(h1, "cs_push_batch"))(e1, X.data(), T, r1.data(), ez1.data(), nullptr);
    reinterpret_cast<fn_batch>(load_fn(h2, "cs_push_batch"))(e2, X.data(), T, r2.data(), ez2.data(), nullptr);

    const uint32_t K = 1 + D * p;
    std::vector<double> B1(K * D), B2(K * D);
    std::vector<double> F1(D * D), F2(D * D), R21(D * D), R22(D * D);
    std::vector<double> Th1(D * D), Th2(D * D), L1(D * D), L2(D * D);

    reinterpret_cast<fn_granger>(load_fn(h1, "cs_granger_matrix"))(e1, F1.data(), R21.data());
    reinterpret_cast<fn_granger>(load_fn(h2, "cs_granger_matrix"))(e2, F2.data(), R22.data());
    reinterpret_cast<fn_prec>(load_fn(h1, "cs_precision_matrix"))(e1, Th1.data(), nullptr);
    reinterpret_cast<fn_prec>(load_fn(h2, "cs_precision_matrix"))(e2, Th2.data(), nullptr);
    reinterpret_cast<fn_lingam>(load_fn(h1, "cs_contemporaneous_orientation"))(e1, L1.data());
    reinterpret_cast<fn_lingam>(load_fn(h2, "cs_contemporaneous_orientation"))(e2, L2.data());
    reinterpret_cast<fn_inspect>(load_fn(h1, "cs_inspect_state"))(e1, B1.data(), nullptr);
    reinterpret_cast<fn_inspect>(load_fn(h2, "cs_inspect_state"))(e2, B2.data(), nullptr);

    double d_ez = 0.0, d_B = 0.0, d_F = 0.0, d_R2 = 0.0, d_Th = 0.0, d_L = 0.0;
    for (size_t i = 0; i < ez1.size(); ++i) d_ez = std::max(d_ez, std::fabs(ez1[i] - ez2[i]));
    for (size_t i = 0; i < B1.size(); ++i) d_B = std::max(d_B, std::fabs(B1[i] - B2[i]));
    for (size_t i = 0; i < F1.size(); ++i) {
        d_F = std::max(d_F, std::fabs(F1[i] - F2[i]) / std::max(1.0, std::fabs(F2[i])));
        d_R2 = std::max(d_R2, std::fabs(R21[i] - R22[i]));
        d_Th = std::max(d_Th, std::fabs(Th1[i] - Th2[i]) / std::max(1.0, std::fabs(Th2[i])));
        d_L = std::max(d_L, std::fabs(L1[i] - L2[i]));
    }
    std::printf("  Parity diffs (D=17 non-mult-of-4): exo_z=%.3e, B=%.3e, F_rel=%.3e, R2=%.3e, Theta_rel=%.3e, LiNGAM=%.3e\n",
                d_ez, d_B, d_F, d_R2, d_Th, d_L);
    CHECK(d_ez < 1e-11, "Scalar vs AVX2 exo_z parity < 1e-11");
    CHECK(d_B  < 1e-12, "Scalar vs AVX2 B parity < 1e-12");
    CHECK(d_F  < 1e-12, "Scalar vs AVX2 F parity < 1e-12");
    CHECK(d_R2 < 1e-12, "Scalar vs AVX2 R2 parity < 1e-12");
    CHECK(d_Th < 1e-12, "Scalar vs AVX2 Theta parity < 1e-12");
    CHECK(d_L  < 1e-12, "Scalar vs AVX2 LiNGAM parity < 1e-12");

    reinterpret_cast<fn_destroy>(load_fn(h1, "cs_destroy"))(e1);
    reinterpret_cast<fn_destroy>(load_fn(h2, "cs_destroy"))(e2);
    dlclose(h1);
    dlclose(h2);
}

void run_microbenchmarks() {
    std::printf("[BENCH] Single-core throughput (%s):\n", cs_simd_isa());
    const uint32_t dims[] = {16, 32, 64};
    std::mt19937_64 rng(7777);
    std::normal_distribution<double> nd(0.0, 1.0);

    for (uint32_t D : dims) {
        const uint32_t p = 2;
        const size_t T = (D == 16) ? 20000 : ((D == 32) ? 10000 : 4000);
        std::vector<double> X(T * D);
        for (double& v : X) v = nd(rng);

        cs_config_t cfg{};
        cs_config_default(&cfg, D, p);
        cfg.warmup_ticks = 50;
        cs_engine_t* eng = cs_create(&cfg);

        /* Warmup cache */
        cs_push_batch(eng, X.data(), 200, nullptr, nullptr, nullptr);
        cs_reset(eng);

        const auto t0 = std::chrono::steady_clock::now();
        cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double tps = static_cast<double>(T) / sec;
        const double ns_per_tick = sec * 1e9 / static_cast<double>(T);
        std::printf("  Tier-1 D=%2u p=%u (K=%3u): %10.1f ticks/s  (%8.1f ns/tick)\n",
                    D, p, cs_feature_dim(eng), tps, ns_per_tick);
        cs_destroy(eng);
    }

    /* Tier-2 high-dim benchmark at D=2048 and D=4096 */
    for (uint32_t D : {2048u, 4096u}) {
        const size_t T = 1000;
        std::vector<double> X(T * D);
        for (double& v : X) v = nd(rng);
        cs_screened_t* sc = cs_screened_create(D, 32, 2, 0.995, 1e-2);
        std::vector<int32_t> top_ch(5);
        std::vector<double> top_sc(5);
        const auto t0 = std::chrono::steady_clock::now();
        cs_screened_push_batch(sc, X.data(), T, nullptr, top_ch.data(), top_sc.data(), 5);
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::printf("  Tier-2 D=%4u M=32 p=2 : %10.1f ticks/s  (%8.1f us/tick)\n",
                    D, static_cast<double>(T) / sec, sec * 1e6 / static_cast<double>(T));
        cs_screened_destroy(sc);
    }
}

void test_underdetermined_no_deadlock() {
    std::printf("[TEST] Underdetermined warmup anti-deadlock (D=12,16,64 lag=2 i.i.d. N(0,1))...\n");
    std::mt19937_64 rng(88888);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (uint32_t D : {12u, 16u, 64u}) {
        const uint32_t p = 2;
        const size_t T = 600;
        std::vector<double> X(T * D);
        for (double& v : X) v = nd(rng);

        cs_config_t cfg{};
        cs_config_default(&cfg, D, p);
        cfg.normalize_scale = 1;
        cs_engine_t* eng = cs_create(&cfg);
        CHECK(eng != nullptr, "cs_create deadlock test");

        std::vector<cs_tick_result_t> res(T);
        CHECK(cs_push_batch(eng, X.data(), T, res.data(), nullptr, nullptr) == 0, "push deadlock test");

        size_t post_warm = 0;
        size_t zero_w = 0;
        for (size_t t = 0; t < T; ++t) {
            if (res[t].top_root_channel >= 0) {
                ++post_warm;
                if (res[t].shock_weight == 0.0) ++zero_w;
            }
        }
        const double frac_zero = (post_warm > 0) ? (static_cast<double>(zero_w) / static_cast<double>(post_warm)) : 1.0;
        std::printf("  D=%2u lag=%u: post_warm=%zu, frac(w_t==0)=%.4f, final_E=%.4f\n",
                    D, p, post_warm, frac_zero, res.back().e_value);
        CHECK(frac_zero < 0.01, "frac(w_t == 0) on i.i.d. N(0,1) must be < 0.01 (zero deadlock)");
        cs_destroy(eng);
    }
}

void test_flat_mem_collinear_and_instantaneous_ac1() {
    std::printf("[TEST] Universal Fixes 1-3: flat 42MB RSS page step, D=60 collinear twins, instantaneous non-Gaussian AC@1...\n");

    /* 1. Flat 42 MB RSS memory stepping by +4096 bytes + D=60 collinear twins */
    {
        const uint32_t D = 60, p = 2;
        const size_t T = 360;
        std::mt19937_64 rng(60601);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<double> X(T * D, 0.0);
        for (size_t t = 0; t < T; ++t) {
            for (uint32_t j = 0; j < D; ++j) X[t * D + j] = nd(rng);
            /* Flat 42 MB RSS memory metric stepping by 1 page (+4096 bytes) right after warmup */
            X[t * D + 10] = (t < 100) ? 4.2e7 : (4.2e7 + 4096.0);
            /* Flat zero error metric jumping moderately */
            X[t * D + 11] = 0.0;
            /* Collinear latency-50 / latency-90 twins on true root pod (ch 2 & ch 3, rho ~ 0.995) */
            const double shock = (t >= 300) ? 10.0 : 0.0;
            X[t * D + 2] = (t > 0 ? 0.3 * X[(t - 1) * D + 2] : 0.0) + nd(rng) + shock;
            X[t * D + 3] = X[t * D + 2] + 0.08 * nd(rng);
            /* Downstream victim ch 4 */
            X[t * D + 4] = (t > 0 ? 1.8 * X[(t - 1) * D + 2] : 0.0) + 0.4 * nd(rng);
        }
        cs_config_t cfg{};
        cs_config_default(&cfg, D, p);
        cs_engine_t* eng = cs_create(&cfg);
        std::vector<cs_tick_result_t> res(T);
        std::vector<double> ez(T * D);
        CHECK(cs_push_batch(eng, X.data(), T, res.data(), ez.data(), nullptr) == 0, "D=60 flat/twin push");
        for (size_t t = 0; t < T; ++t) {
            CHECK(std::isfinite(res[t].mahalanobis_sq) && std::isfinite(res[t].e_value) && std::isfinite(res[t].max_exo_z),
                  "D=60 zero NaN/Inf");
        }
        CHECK(std::fabs(ez[100 * D + 10]) < 1.0, "42MB RSS +4KB page step must stay < 1.0 sigma");

        std::vector<int32_t> ranked(D);
        cs_root_attribution(eng, 0, nullptr, nullptr, nullptr, nullptr, ranked.data());
        std::printf("  D=60 collinear twins RCA top3=[%d, %d, %d], mem_page_ez=%.4f\n",
                    ranked[0], ranked[1], ranked[2], ez[100 * D + 10]);
        CHECK(ranked[0] == 2 || ranked[0] == 3, "Collinear twin root pod (ch 2 or 3) ranked #1");
        cs_destroy(eng);
    }

    /* 2. 20-trial Instantaneous Non-Gaussian Cascade (0 -> 1 -> 2, shock@0) AC@1 */
    {
        const int n_trials = 20;
        int ac1 = 0;
        for (int tr = 0; tr < n_trials; ++tr) {
            const uint32_t D = 8, p = 1;
            const size_t T_clean = 350, T_fault = 30, T = T_clean + T_fault;
            std::mt19937_64 rng(70000 + tr);
            std::exponential_distribution<double> expd(1.0);
            std::vector<double> X(T * D, 0.0);
            for (size_t t = 0; t < T; ++t) {
                const double s = (t >= T_clean) ? 7.0 : 0.0;
                const double x0 = (expd(rng) - 1.0) + s;
                const double x1 = 0.80 * x0 + 0.60 * (expd(rng) - 1.0);
                const double x2 = 0.80 * x1 + 0.60 * (expd(rng) - 1.0);
                X[t * D + 0] = x0;
                X[t * D + 1] = x1;
                X[t * D + 2] = x2;
                for (uint32_t j = 3; j < D; ++j) X[t * D + j] = expd(rng) - 1.0;
            }
            cs_config_t cfg{};
            cs_config_default(&cfg, D, p);
            cfg.warmup_ticks = 60;
            cs_engine_t* eng = cs_create(&cfg);
            cs_push_batch(eng, X.data(), T, nullptr, nullptr, nullptr);
            std::vector<int32_t> ranked(D);
            cs_root_attribution(eng, 40, nullptr, nullptr, nullptr, nullptr, ranked.data());
            if (ranked[0] == 0) ++ac1;
            cs_destroy(eng);
        }
        std::printf("  Instantaneous Non-Gaussian Cascade AC@1: %d / %d (%.1f%%)\n",
                    ac1, n_trials, 100.0 * ac1 / n_trials);
        CHECK(ac1 >= 17, "Instantaneous Non-Gaussian Cascade AC@1 >= 85%");
    }
}

} /* namespace */

int main(int argc, char** argv) {
    bool do_bench = false;
    const char* avx_so = nullptr;
    const char* scalar_so = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
        } else if (std::strcmp(argv[i], "--parity-so") == 0 && i + 2 < argc) {
            avx_so = argv[++i];
            scalar_so = argv[++i];
        }
    }

    std::printf("=== causalsieve C++20 Core Test Suite (ISA: %s) ===\n", cs_simd_isa());
    test_atomic_validation_and_causality();
    test_h1_exact_oracle();
    test_erddap_dc_offset_and_zero_channel();
    test_underdetermined_no_deadlock();
    test_flat_mem_collinear_and_instantaneous_ac1();
    test_precision_lingam_and_rca();
    test_ons_eprocess_null_calibration();
    test_lossless_ring_and_swing_pla();
    test_tier2_screened_loud_decoys();

    if (avx_so && scalar_so) {
        test_scalar_vs_avx2_so_parity(avx_so, scalar_so);
    }
    if (do_bench) {
        run_microbenchmarks();
    }

    if (g_failures == 0) {
        std::printf("=== ALL TESTS PASSED (0 failures) ===\n");
        return 0;
    } else {
        std::fprintf(stderr, "=== FAILED with %d failure(s) ===\n", g_failures);
        return 1;
    }
}
