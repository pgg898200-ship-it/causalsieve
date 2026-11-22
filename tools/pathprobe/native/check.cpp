#include "pathprobe.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>

using Index = pathprobe::DurationIndex;
using Time = Index::Time;
using Id = Index::Id;
using Edge = Index::Edge;

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Exception, class Function>
static void expect_throw(Function run) {
    try {
        run();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected exception");
}

// Independent full forward dynamic program over the original DAG. Track paths
// that include the queried vertex separately from those that avoid it. Only
// original roots and sinks are endpoints, even when a task has duration zero.
struct Oracle {
    std::vector<Id> order;
    std::vector<std::vector<Id>> parents;
    std::vector<bool> sinks;

    Oracle(const std::vector<Id>& supplied_order, const std::vector<Edge>& edges)
        : order(supplied_order), parents(order.size()), sinks(order.size(), true) {
        for (auto [u, v] : edges) {
            parents[v].push_back(u);
            sinks[u] = false;
        }
    }

    std::pair<Time, Time> solve(const std::vector<Time>& durations, Id task,
                                Time replacement) const {
        struct State {
            Time via = -1;
            Time around = -1;
        };
        std::vector<State> state(order.size());
        Time best_via = -1, best_around = -1;
        for (Id u : order) {
            Time from_via = -1, from_around = parents[u].empty() ? 0 : -1;
            for (Id v : parents[u]) {
                from_via = std::max(from_via, state[v].via);
                from_around = std::max(from_around, state[v].around);
            }
            if (u == task) {
                state[u].via = std::max(from_via, from_around) + replacement;
            } else {
                if (from_via >= 0) {
                    state[u].via = from_via + durations[u];
                }
                if (from_around >= 0) {
                    state[u].around = from_around + durations[u];
                }
            }
            if (sinks[u]) {
                best_via = std::max(best_via, state[u].via);
                best_around = std::max(best_around, state[u].around);
            }
        }
        return {best_via, best_around};
    }
};

static void boundaries() {
    using Limits = std::numeric_limits<Time>;
    Index empty({}, {});
    require(empty.makespan() == 0, "empty graph");
    expect_throw<std::out_of_range>([&] { empty.profile(0); });
    expect_throw<std::out_of_range>([&] { empty.predict(0, 1); });
    expect_throw<std::out_of_range>([] { Index bad({}, {{0, 0}}); });
    expect_throw<std::invalid_argument>([] { Index bad({-1}, {}); });
    expect_throw<std::invalid_argument>([] { Index bad({1, 2, 3}, {{1, 2}, {2, 1}}); });
    expect_throw<std::invalid_argument>([] { Index bad({1}, {{0, 0}}); });
    expect_throw<std::overflow_error>([] { Index bad({Limits::max(), 1}, {{0, 1}}); });

    Index frontier({Limits::max() - 1, 1}, {{0, 1}, {0, 1}});
    require(frontier.makespan() == Limits::max(), "maximum representable path");
    require(frontier.profile(0).through == Limits::max(), "through near overflow");
    require(!frontier.profile(0).bypass.has_value(), "mandatory node");
    require(frontier.predict(1, 0) == Limits::max() - 1, "near maximum prediction");
    expect_throw<std::overflow_error>([&] { frontier.predict(1, 2); });
    expect_throw<std::invalid_argument>([&] { frontier.predict(1, -1); });

    Index ties({1, 1, 0, 7}, {{0, 2}, {1, 2}});
    require(ties.makespan() == 7 && ties.predict(0, 8) == 8, "disconnected roots");
    require(ties.profile(3).bypass == 1, "disconnected bypass");
    Index zero_tie({0, 0}, {});
    require(zero_tie.profile(0).bypass == 0, "zero-length bypass is present");
    require(zero_tie.profile(1).max_saving == 0, "zero-length drag");

    // Exercise interval handling at and beyond the short-path crossover.
    for (Id length : {48, 49, 86}) {
        std::vector<Time> weights(length, 3);
        std::vector<Edge> edges;
        for (Id i = 1; i < length; ++i) {
            edges.emplace_back(i - 1, i);
        }
        edges.emplace_back(3, 5);
        edges.emplace_back(13, 20);
        edges.emplace_back(18, 28);
        edges.emplace_back(1, length - 1);
        std::vector<Id> order(length);
        std::iota(order.begin(), order.end(), 0);
        Index index(weights, edges);
        Oracle oracle(order, edges);
        for (Id i = 0; i < length; ++i) {
            const auto [through, bypass] = oracle.solve(weights, i, weights[i]);
            const auto p = index.profile(i);
            require(p.through == through && p.bypass.value_or(-1) == bypass,
                    "mixed singleton and multi-task detours");
        }
    }

    // Linear depth substantially beyond typical native recursion limits.
    const Id n = 100000;
    std::vector<Edge> chain;
    chain.reserve(n - 1);
    for (Id i = 1; i < n; ++i) {
        chain.emplace_back(i - 1, i);
    }
    Index deep(std::vector<Time>(n, 1), chain);
    require(deep.predict(n / 2, 10) == static_cast<Time>(n + 9), "deep chain");
}

int main() {
    try {
        boundaries();
        std::mt19937 rng(21453);
        std::size_t predictions = 0;
        for (Id n = 1; n <= 25; ++n) {
            for (int trial = 0; trial < 80; ++trial) {
                std::vector<Id> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                std::vector<Edge> edges;
                std::vector<Time> durations(n);
                for (Id i = 0; i < n; ++i) {
                    durations[i] = rng() % 8;
                    for (Id j = 0; j < i; ++j) {
                        if (rng() % 7 == 0) {
                            edges.emplace_back(order[j], order[i]);
                            if (rng() % 6 == 0) {
                                edges.push_back(edges.back());
                            }
                        }
                    }
                }
                std::shuffle(edges.begin(), edges.end(), rng);
                Index index(durations, edges);
                Oracle oracle(order, edges);
                for (Id t = 0; t < n; ++t) {
                    auto [through, bypass] = oracle.solve(durations, t, durations[t]);
                    auto p = index.profile(t);
                    require(index.makespan() == std::max(through, bypass), "makespan");
                    require(p.duration == durations[t] && p.through == through, "through");
                    require(p.bypass.value_or(-1) == bypass, "bypass");
                    for (Time proposed : {0, 1, 4, 40}) {
                        auto [via, around] = oracle.solve(durations, t, proposed);
                        Time expected = std::max(via, around);
                        require(index.predict(t, proposed) == expected, "prediction");
                        if (proposed == 0) {
                            require(p.max_saving == index.makespan() - expected, "saving");
                        }
                        ++predictions;
                    }
                }
            }
        }
        std::cout << predictions << " predictions matched full DAG replay on 2000 graphs\n";
    } catch (const std::exception& e) {
        std::cerr << "native checks failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
