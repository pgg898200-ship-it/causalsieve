#include "causalsieve.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options] [input.csv|-]\n"
        "Zero-dependency AVX2/FMA streaming causal RCA engine.\n\n"
        "Options:\n"
        "  --lag P               VAR lag order (1..16, default 2)\n"
        "  --forgetting LAM      Exponential forgetting factor in (0,1] (default 0.995)\n"
        "  --ridge R             Ridge penalty on lagged Gram (default 1e-2)\n"
        "  --precision-ridge PR  Ridge penalty on innovation covariance (default 1e-2)\n"
        "  --shock-threshold S   Huber Mahalanobis gate; 0 disables shield (default auto)\n"
        "  --warmup W            Warmup ticks (0 = auto)\n"
        "  --hinge               Enable asymmetric hinge basis max(0, x-1)\n"
        "  --no-normalize        Disable scale normalization (anchor-shift only)\n"
        "  --lookback L          Root-cause lookback window (0 = auto)\n"
        "  --top-k K             Number of top root causes to report (default 5)\n"
        "  --active-m M          Active subgraph size for high-dim D>128 (default 32)\n"
        "  --no-header           First row is numeric data\n"
        "  --drop-first-col      Drop column 0 (timestamp/index)\n"
        "  --compact             Output compact one-line digest instead of JSON\n"
        "  -h, --help            Show this help\n",
        prog);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string trim_token(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n' || s[a] == '"')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n' || s[b - 1] == '"')) --b;
    return s.substr(a, b - a);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            tokens.push_back(trim_token(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    tokens.push_back(trim_token(cur));
    return tokens;
}

bool try_parse_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return (end != s.c_str() && *end == '\0' && std::isfinite(out));
}

bool is_time_col_name(const std::string& name) {
    std::string low = name;
    for (char& c : low) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return (low == "time" || low == "timestamp" || low == "ts" || low == "date" ||
            low == "datetime" || low == "tick" || low == "step" || low == "index" || low == "t");
}

struct EdgeItem {
    uint32_t src;
    uint32_t dst;
    double score;
    double aux;
};

} /* namespace */

int main(int argc, char** argv) {
    uint32_t lag = 2;
    double forgetting = 0.995;
    double ridge = 1e-2;
    double precision_ridge = 1e-2;
    double shock_threshold = -1.0; /* <0 means auto */
    uint32_t warmup = 0;
    uint32_t hinge = 0;
    uint32_t normalize = 1;
    uint32_t lookback = 0;
    uint32_t top_k = 5;
    uint32_t active_m = 32;
    bool no_header = false;
    bool drop_first = false;
    bool compact = false;
    std::string input_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--lag" && i + 1 < argc) {
            lag = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--forgetting" && i + 1 < argc) {
            forgetting = std::strtod(argv[++i], nullptr);
        } else if (arg == "--ridge" && i + 1 < argc) {
            ridge = std::strtod(argv[++i], nullptr);
        } else if (arg == "--precision-ridge" && i + 1 < argc) {
            precision_ridge = std::strtod(argv[++i], nullptr);
        } else if (arg == "--shock-threshold" && i + 1 < argc) {
            shock_threshold = std::strtod(argv[++i], nullptr);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--hinge") {
            hinge = 1;
        } else if (arg == "--no-normalize") {
            normalize = 0;
        } else if (arg == "--lookback" && i + 1 < argc) {
            lookback = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--active-m" && i + 1 < argc) {
            active_m = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--no-header") {
            no_header = true;
        } else if (arg == "--drop-first-col") {
            drop_first = true;
        } else if (arg == "--compact") {
            compact = true;
        } else if (arg == "-" || arg[0] != '-') {
            input_path = arg;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return 2;
        }
    }

    std::ifstream file_in;
    std::istream* in = &std::cin;
    if (!input_path.empty() && input_path != "-") {
        file_in.open(input_path);
        if (!file_in) {
            std::fprintf(stderr, "Failed to open input file: %s\n", input_path.c_str());
            return 1;
        }
        in = &file_in;
    }

    std::vector<std::string> col_names;
    std::vector<double> data_flat;
    size_t n_ticks = 0;
    size_t raw_cols = 0;
    size_t col_start = drop_first ? 1 : 0;
    bool initialized_schema = false;

    std::string line;
    while (std::getline(*in, line)) {
        if (line.empty()) continue;
        bool all_ws = true;
        for (char c : line) {
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') { all_ws = false; break; }
        }
        if (all_ws) continue;

        auto toks = split_csv_line(line);
        if (toks.empty()) continue;

        if (!initialized_schema) {
            raw_cols = toks.size();
            bool has_non_numeric = false;
            for (size_t c = col_start; c < toks.size(); ++c) {
                double dummy = 0.0;
                if (!try_parse_double(toks[c], dummy)) {
                    has_non_numeric = true;
                    break;
                }
            }

            if (!no_header && has_non_numeric) {
                if (!drop_first && toks.size() > 1 && is_time_col_name(toks[0])) {
                    col_start = 1;
                }
                for (size_t c = col_start; c < toks.size(); ++c) {
                    col_names.push_back(toks[c].empty() ? ("ch_" + std::to_string(c - col_start)) : toks[c]);
                }
                initialized_schema = true;
                continue;
            } else {
                /* Auto-detect non-numeric timestamp col 0 even without header */
                double d0 = 0.0;
                if (!drop_first && toks.size() > 1 && !try_parse_double(toks[0], d0)) {
                    col_start = 1;
                }
                for (size_t c = col_start; c < toks.size(); ++c) {
                    col_names.push_back("ch_" + std::to_string(c - col_start));
                }
                initialized_schema = true;
            }
        }

        if (toks.size() < raw_cols) continue;
        const size_t D_row = col_names.size();
        bool row_ok = true;
        const size_t base = data_flat.size();
        data_flat.resize(base + D_row);
        for (size_t c = 0; c < D_row; ++c) {
            double v = 0.0;
            if (!try_parse_double(toks[col_start + c], v)) {
                row_ok = false;
                break;
            }
            data_flat[base + c] = v;
        }
        if (!row_ok) {
            data_flat.resize(base);
            continue;
        }
        ++n_ticks;
    }

    const uint32_t D = static_cast<uint32_t>(col_names.size());
    if (D == 0 || n_ticks == 0) {
        std::fprintf(stderr, "No valid numeric CSV rows found.\n");
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();

    if (D > 128) {
        /* Tier-2 Screened Engine for high-dimensional streams */
        cs_screened_t* sc = cs_screened_create(D, std::min(active_m, D), lag, forgetting, ridge);
        if (!sc) {
            std::fprintf(stderr, "Failed to create Tier-2 screened engine.\n");
            return 1;
        }
        const uint32_t k_rep = std::min(std::max<uint32_t>(1u, top_k), D);
        std::vector<double> last_z(D, 0.0);
        std::vector<int32_t> top_ch(k_rep, 0);
        std::vector<double> top_sc(k_rep, 0.0);
        if (cs_screened_push_batch(sc, data_flat.data(), n_ticks, last_z.data(), top_ch.data(), top_sc.data(), k_rep) != 0) {
            std::fprintf(stderr, "cs_screened_push_batch failed.\n");
            cs_screened_destroy(sc);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double tps = (ms > 0.0) ? (static_cast<double>(n_ticks) * 1000.0 / ms) : 0.0;

        if (compact) {
            if (!top_ch.empty() && top_ch[0] >= 0 && static_cast<size_t>(top_ch[0]) < col_names.size()) {
                std::printf("ISA=%s MODE=screened D=%u T=%zu TPS=%.0f TOP_ROOT=%s(%.2f)\n",
                            cs_simd_isa(), D, n_ticks, tps,
                            col_names[static_cast<size_t>(top_ch[0])].c_str(), top_sc[0]);
            } else {
                std::printf("ISA=%s MODE=screened D=%u T=%zu TPS=%.0f TOP_ROOT=none\n",
                            cs_simd_isa(), D, n_ticks, tps);
            }
        } else {
            std::printf("{\n  \"simd_isa\": \"%s\",\n  \"mode\": \"tier2_screened\",\n  \"dim\": %u,\n  \"ticks\": %zu,\n  \"elapsed_ms\": %.3f,\n  \"ticks_per_sec\": %.1f,\n  \"top_root_causes\": [\n",
                        cs_simd_isa(), D, n_ticks, ms, tps);
            for (uint32_t i = 0; i < k_rep; ++i) {
                const int32_t ch = top_ch[i];
                std::printf("    {\"rank\": %u, \"channel_idx\": %d, \"name\": \"%s\", \"root_score\": %.4f, \"last_z\": %.4f}%s\n",
                            i + 1, ch, json_escape(col_names[ch]).c_str(), top_sc[i], last_z[ch],
                            (i + 1 < k_rep) ? "," : "");
            }
            std::printf("  ]\n}\n");
        }
        cs_screened_destroy(sc);
        return 0;
    }

    /* Tier-1 Exact Streaming Causal Engine */
    cs_config_t cfg{};
    cs_config_default(&cfg, D, lag);
    cfg.forgetting = forgetting;
    cfg.ridge = ridge;
    cfg.precision_ridge = precision_ridge;
    if (shock_threshold >= 0.0) cfg.shock_threshold = shock_threshold;
    cfg.warmup_ticks = warmup;
    cfg.nonlinear_hinge = hinge;
    cfg.normalize_scale = normalize;
    if (n_ticks > cfg.raw_ring_capacity && n_ticks <= 65536) {
        cfg.raw_ring_capacity = static_cast<uint32_t>(n_ticks);
    }

    cs_engine_t* eng = cs_create(&cfg);
    if (!eng) {
        std::fprintf(stderr, "Failed to create Tier-1 cs_engine.\n");
        return 1;
    }

    std::vector<cs_tick_result_t> results(n_ticks);
    if (cs_push_batch(eng, data_flat.data(), n_ticks, results.data(), nullptr, nullptr) != 0) {
        std::fprintf(stderr, "cs_push_batch failed: %s\n", cs_last_error(eng));
        cs_destroy(eng);
        return 1;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double tps = (ms > 0.0) ? (static_cast<double>(n_ticks) * 1000.0 / ms) : 0.0;

    uint64_t alarm_count = 0;
    int64_t first_alarm_tick = -1;
    for (size_t i = 0; i < n_ticks; ++i) {
        if (results[i].alarm) {
            ++alarm_count;
            if (first_alarm_tick < 0) first_alarm_tick = static_cast<int64_t>(results[i].tick);
        }
    }

    std::vector<double> root_scores(D, 0.0);
    std::vector<double> peak_exo(D, 0.0);
    std::vector<double> raw_z(D, 0.0);
    std::vector<int64_t> onset(D, -1);
    std::vector<int32_t> ranked(D, 0);
    cs_root_attribution(eng, lookback, root_scores.data(), peak_exo.data(), raw_z.data(), onset.data(), ranked.data());

    std::vector<double> f_mat(static_cast<size_t>(D) * D, 0.0);
    std::vector<double> r2_mat(static_cast<size_t>(D) * D, 0.0);
    std::vector<double> pcorr(static_cast<size_t>(D) * D, 0.0);
    std::vector<double> lingam(static_cast<size_t>(D) * D, 0.0);
    cs_granger_matrix(eng, f_mat.data(), r2_mat.data());
    cs_precision_matrix(eng, nullptr, pcorr.data());
    cs_contemporaneous_orientation(eng, lingam.data());

    std::vector<EdgeItem> granger_edges;
    std::vector<EdgeItem> lingam_edges;
    for (uint32_t i = 0; i < D; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            if (i == j) continue;
            const double r2 = r2_mat[static_cast<size_t>(i) * D + j];
            if (r2 > 0.05) {
                granger_edges.push_back({i, j, r2, f_mat[static_cast<size_t>(i) * D + j]});
            }
            const double lr = lingam[static_cast<size_t>(i) * D + j];
            if (lr > 0.02 && std::fabs(pcorr[static_cast<size_t>(i) * D + j]) > 0.08) {
                lingam_edges.push_back({i, j, lr, pcorr[static_cast<size_t>(i) * D + j]});
            }
        }
    }
    std::sort(granger_edges.begin(), granger_edges.end(), [](const EdgeItem& a, const EdgeItem& b) {
        return a.score > b.score;
    });
    std::sort(lingam_edges.begin(), lingam_edges.end(), [](const EdgeItem& a, const EdgeItem& b) {
        return a.score > b.score;
    });

    /* Swing-Filter PLA compression ratio on top root channel */
    const uint32_t top_ch = static_cast<uint32_t>(ranked[0]);
    std::vector<double> raw_slice(n_ticks, 0.0);
    uint64_t slice_n = 0, slice_start = 0;
    cs_get_raw_slice(eng, top_ch, 0, n_ticks, raw_slice.data(), &slice_n, &slice_start);
    double vmin = 0.0, vmax = 0.0;
    if (slice_n > 0) {
        vmin = vmax = raw_slice[0];
        for (uint64_t i = 1; i < slice_n; ++i) {
            vmin = std::min(vmin, raw_slice[i]);
            vmax = std::max(vmax, raw_slice[i]);
        }
    }
    const double pla_eps = std::max(1e-6, 0.01 * (vmax - vmin));
    std::vector<uint64_t> knot_t(slice_n + 2, 0);
    std::vector<double> knot_v(slice_n + 2, 0.0);
    size_t n_knots = 0;
    if (slice_n > 0) {
        cs_compress_channel_pla(eng, top_ch, slice_start, slice_start + slice_n,
                                pla_eps, knot_t.data(), knot_v.data(), knot_t.size(), &n_knots);
    }

    const uint32_t k_rep = std::min(top_k, D);
    const cs_tick_result_t& last_res = results.back();

    if (compact) {
        std::printf("ISA=%s D=%u p=%u T=%zu TPS=%.0f ALARMS=%llu FIRST_ALARM=%lld E=%.2f ROOT1=%s(score=%.2f,exo_z=%.2f,raw_z=%.2f,onset=%lld) PLA=%zu/%llu\n",
                    cs_simd_isa(), D, lag, n_ticks, tps,
                    static_cast<unsigned long long>(alarm_count),
                    static_cast<long long>(first_alarm_tick),
                    last_res.e_value,
                    col_names[ranked[0]].c_str(),
                    root_scores[ranked[0]], peak_exo[ranked[0]], raw_z[ranked[0]],
                    static_cast<long long>(onset[ranked[0]]),
                    n_knots, static_cast<unsigned long long>(slice_n));
    } else {
        std::printf("{\n");
        std::printf("  \"simd_isa\": \"%s\",\n", cs_simd_isa());
        std::printf("  \"dim\": %u,\n", D);
        std::printf("  \"lag\": %u,\n", lag);
        std::printf("  \"feature_dim\": %u,\n", cs_feature_dim(eng));
        std::printf("  \"ticks\": %zu,\n", n_ticks);
        std::printf("  \"elapsed_ms\": %.3f,\n", ms);
        std::printf("  \"ticks_per_sec\": %.1f,\n", tps);
        std::printf("  \"alarms\": %llu,\n", static_cast<unsigned long long>(alarm_count));
        std::printf("  \"first_alarm_tick\": %lld,\n", static_cast<long long>(first_alarm_tick));
        std::printf("  \"final_e_value\": %.6g,\n", last_res.e_value);
        std::printf("  \"refresh_count\": %u,\n", last_res.refresh_count);
        std::printf("  \"top_root_causes\": [\n");
        for (uint32_t i = 0; i < k_rep; ++i) {
            const int32_t ch = ranked[i];
            std::printf("    {\"rank\": %u, \"channel_idx\": %d, \"name\": \"%s\", \"root_score\": %.4f, \"peak_exo_z\": %.4f, \"raw_z\": %.4f, \"onset_tick\": %lld}%s\n",
                        i + 1, ch, json_escape(col_names[ch]).c_str(),
                        root_scores[ch], peak_exo[ch], raw_z[ch],
                        static_cast<long long>(onset[ch]),
                        (i + 1 < k_rep) ? "," : "");
        }
        std::printf("  ],\n");

        const size_t max_edges = 15;
        const size_t ng = std::min(max_edges, granger_edges.size());
        std::printf("  \"top_granger_edges\": [\n");
        for (size_t i = 0; i < ng; ++i) {
            const auto& e = granger_edges[i];
            std::printf("    {\"src\": \"%s\", \"dst\": \"%s\", \"partial_r2\": %.4f, \"f_stat\": %.3f}%s\n",
                        json_escape(col_names[e.src]).c_str(),
                        json_escape(col_names[e.dst]).c_str(),
                        e.score, e.aux, (i + 1 < ng) ? "," : "");
        }
        std::printf("  ],\n");

        const size_t nl = std::min(max_edges, lingam_edges.size());
        std::printf("  \"top_contemporaneous_edges\": [\n");
        for (size_t i = 0; i < nl; ++i) {
            const auto& e = lingam_edges[i];
            std::printf("    {\"src\": \"%s\", \"dst\": \"%s\", \"lingam_r\": %.4f, \"pcorr\": %.4f}%s\n",
                        json_escape(col_names[e.src]).c_str(),
                        json_escape(col_names[e.dst]).c_str(),
                        e.score, e.aux, (i + 1 < nl) ? "," : "");
        }
        std::printf("  ],\n");

        const double ratio = (n_knots > 0) ? (static_cast<double>(slice_n) / static_cast<double>(n_knots)) : 1.0;
        std::printf("  \"pla_summary\": {\"channel\": \"%s\", \"raw_points\": %llu, \"knots\": %zu, \"max_abs_err\": %.6g, \"compression_ratio\": %.2f}\n",
                    json_escape(col_names[top_ch]).c_str(),
                    static_cast<unsigned long long>(slice_n),
                    n_knots, pla_eps, ratio);
        std::printf("}\n");
    }

    cs_destroy(eng);
    return 0;
}
