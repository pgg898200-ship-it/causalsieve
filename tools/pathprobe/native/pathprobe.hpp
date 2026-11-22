#ifndef PATHPROBE_DURATION_INDEX_HPP
#define PATHPROBE_DURATION_INDEX_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pathprobe {

struct Profile {
    std::int64_t duration;
    std::int64_t through;
    std::optional<std::int64_t> bypass;
    std::int64_t max_saving;
};

// Task IDs are [0, durations.size()). Edges point from prerequisite to dependent.
// All roots start at zero; every sink must finish. Concurrent const queries are safe.
class DurationIndex {
public:
    using Time = std::int64_t;
    using Id = std::size_t;
    using Edge = std::pair<Id, Id>;

    DurationIndex(const std::vector<Time>& durations, const std::vector<Edge>& edges)
        : weights_(durations), bounds_(durations.size()) {
        const Id n = durations.size();
        for (Time value : durations) {
            validate_duration(value);
        }
        const Id source = n, sink = n + 1, count = n + 2;
        std::vector<Id> degree(count), indegree(count);
        for (const auto& edge : edges) {
            if (edge.first >= n || edge.second >= n) {
                throw std::out_of_range("edge references an unknown task");
            }
            ++degree[edge.first];
            ++indegree[edge.second];
        }
        if (n == 0) {
            return;
        }
        for (Id u = 0; u < n; ++u) {
            if (indegree[u] == 0) {
                ++degree[source];
            }
            if (degree[u] == 0) {
                ++degree[u];
                ++indegree[sink];
            }
        }

        // Flat adjacency keeps per-vertex allocations out of graph traversal.
        std::vector<Id> offsets(count + 1);
        for (Id u = 0; u < count; ++u) {
            offsets[u + 1] = offsets[u] + degree[u];
        }
        std::vector<Id> children(offsets.back());
        {
            // Reuse the outgoing degree buffer after finalizing CSR offsets.
            auto& cursor = degree;
            std::copy_n(offsets.begin(), count, cursor.begin());
            for (const auto& edge : edges) {
                children[cursor[edge.first]++] = edge.second;
            }
            for (Id u = 0; u < n; ++u) {
                if (indegree[u] == 0) {
                    children[cursor[source]++] = u;
                    ++indegree[u];
                }
                if (cursor[u] == offsets[u]) {
                    children[cursor[u]++] = sink;
                }
            }
        }

        auto weight = [&](Id u) -> Time { return u < n ? weights_[u] : 0; };
        std::vector<Time> prefix(count, -1), suffix(count);
        auto& predecessor = degree;
        prefix[source] = 0;
        // Finish the cycle check even if a traversed prefix exceeds int64_t.
        bool overflow = false;
        std::vector<Id> order;
        order.reserve(count);
        order.push_back(source);
        for (Id step = 0; step < order.size(); ++step) {
            const Id u = order[step];
            Time finish = prefix[u];
            if (weight(u) > std::numeric_limits<Time>::max() - finish) {
                overflow = true;
            } else {
                finish += weight(u);
            }
            prefix[u] = finish;
            for (Id k = offsets[u]; k < offsets[u + 1]; ++k) {
                const Id v = children[k];
                if (finish > prefix[v]) {
                    prefix[v] = finish;
                    predecessor[v] = u;
                }
                if (--indegree[v] == 0) {
                    order.push_back(v);
                }
            }
        }
        if (order.size() != count) {
            throw std::invalid_argument("dependency graph contains a cycle");
        }
        if (overflow) {
            throw std::overflow_error("path duration exceeds int64_t");
        }
        makespan_ = prefix[sink];
        // With nonnegative weights, each suffix and each valid detour is
        // bounded by the verified makespan. Only predictions can now overflow.
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const Id u = *it;
            Time longest_child = 0;
            for (Id k = offsets[u]; k < offsets[u + 1]; ++k) {
                longest_child = std::max(longest_child, suffix[children[k]]);
            }
            suffix[u] = weight(u) + longest_child;
            if (u < n) {
                bounds_[u].other = (prefix[u] - weights_[u]) + longest_child;
            }
        }

        // One chosen longest path already bypasses every task outside it.
        std::vector<unsigned char> on_path(count, 0);
        for (Id u = predecessor[sink]; u != source; u = predecessor[u]) {
            on_path[u] = 1;
        }
        Id path_length = 0;
        auto& before = indegree;
        for (Id u : order) {
            before[u] = path_length;
            path_length += on_path[u];
        }

        // In topological order, all intervals beginning before the current
        // task have arrived. In a small table, scan endpoints that still span
        // this task. Bound the table size to keep per-task scans inexpensive.
        constexpr Id max_scan_length = 48;
        if (path_length <= max_scan_length) {
            std::vector<Time> ending(path_length + 1, -1);
            for (Id u : order) {
                const Id pos = before[u];
                if (on_path[u]) {
                    bounds_[u].bypass = *std::max_element(ending.begin() + pos + 1, ending.end());
                } else if (u < n) {
                    bounds_[u].bypass = makespan_;
                }
                for (Id k = offsets[u]; k < offsets[u + 1]; ++k) {
                    const Id v = children[k], stop = before[v];
                    if (stop > pos + on_path[u]) {
                        ending[stop] = std::max(ending[stop], prefix[u] + suffix[v]);
                    }
                }
            }
            return;
        }

        Id base = 1;
        while (base < path_length) {
            base *= 2;
        }
        auto& selected = predecessor;
        for (Id u = 0; u < n; ++u) {
            bounds_[u].bypass = on_path[u] ? -1 : makespan_;
            if (on_path[u]) {
                selected[before[u]] = u;
            }
        }
        std::vector<Time> best;
        for (Id u : order) {
            const Id start = before[u] + on_path[u];
            for (Id k = offsets[u]; k < offsets[u + 1]; ++k) {
                const Id v = children[k];
                if (start >= before[v]) {
                    continue;
                }
                // This edge skips the chosen path tasks in [start, before[v]).
                const Time length = prefix[u] + suffix[v];
                if (before[v] == start + 1) {
                    auto& single = bounds_[selected[start]].bypass;
                    single = std::max(single, length);
                    continue;
                }
                if (best.empty()) {
                    best.assign(2 * base, -1);
                }
                Id left = base + start, right = base + before[v];
                while (left < right) {
                    if (left % 2 != 0) {
                        best[left] = std::max(best[left], length);
                        ++left;
                    }
                    if (right % 2 != 0) {
                        --right;
                        best[right] = std::max(best[right], length);
                    }
                    left /= 2;
                    right /= 2;
                }
            }
        }
        if (best.empty()) {
            return;
        }
        for (Id i = 1; i < base; ++i) {
            best[2 * i] = std::max(best[2 * i], best[i]);
            best[2 * i + 1] = std::max(best[2 * i + 1], best[i]);
        }
        for (Id i = 0; i < path_length; ++i) {
            auto& bypass = bounds_[selected[i]].bypass;
            bypass = std::max(bypass, best[base + i]);
        }
    }

    Time makespan() const noexcept { return makespan_; }

    Time predict(Id task, Time new_duration) const {
        const auto& bound = bounds_.at(task);
        validate_duration(new_duration);
        return std::max(bound.bypass, add(bound.other, new_duration));
    }

    Profile profile(Id task) const {
        const Time duration = weights_.at(task);
        const Time through = bounds_[task].other + duration, bypass = bounds_[task].bypass;
        return {duration, through, bypass < 0 ? std::nullopt : std::optional<Time>(bypass),
                makespan_ - std::max(bypass, through - duration)};
    }

private:
    static void validate_duration(Time value) {
        if (value < 0) {
            throw std::invalid_argument("duration must be nonnegative");
        }
    }

    static Time add(Time a, Time b) {
        // Both operands are nonnegative; never form an overflowing signed sum.
        if (b > std::numeric_limits<Time>::max() - a) {
            throw std::overflow_error("path duration exceeds int64_t");
        }
        return a + b;
    }

    // A query touches one pair. The original weight is needed only for profiles.
    struct Bounds {
        Time other;
        Time bypass;
    };
    std::vector<Time> weights_;
    std::vector<Bounds> bounds_;
    Time makespan_ = 0;
};

}  // namespace pathprobe
#endif
