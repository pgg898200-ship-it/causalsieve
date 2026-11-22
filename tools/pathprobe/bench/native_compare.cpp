// Driver for export_native.py. Input is the same validated integer graph and
// scalar query list for each method. Only construction and predictions are timed.
#include "../native/pathprobe.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Id = std::size_t;
using Time = std::int64_t;
using Edge = std::pair<Id, Id>;
using Query = std::pair<Id, Time>;
using Vec = std::vector<Time>;
using Clock = std::chrono::steady_clock;
using Trial = std::array<Time, 3>;
using Methods = std::array<std::vector<Trial>, 4>;

struct FullDP {
    Vec weights;
    std::vector<Id> offsets, children, order, indegree;
    Vec scratch;

    FullDP(const Vec& w, const std::vector<Edge>& edges)
        : weights(w), offsets(w.size()+1), indegree(w.size()), scratch(w.size()) {
        for (auto [u,v] : edges) {
            ++offsets[u+1];
            ++indegree[v];
        }
        for (Id i=1; i<offsets.size(); ++i) offsets[i] += offsets[i-1];
        auto cursor = offsets;
        children.resize(edges.size());
        for (auto [u,v] : edges) children[cursor[u]++] = v;
        auto pending = indegree;
        order.reserve(w.size());
        for (Id u=0; u<w.size(); ++u) if (!pending[u]) order.push_back(u);
        for (Id i=0; i<order.size(); ++i) {
            Id u = order[i];
            for (Id j=offsets[u]; j<offsets[u+1]; ++j) {
                Id v = children[j];
                if (!--pending[v]) order.push_back(v);
            }
        }
        if (order.size() != w.size()) throw std::runtime_error("cyclic graph");
    }

    Time predict(Id edited, Time duration) {
        std::fill(scratch.begin(), scratch.end(), 0);
        Time best = 0;
        for (Id u : order) {
            Time finish = scratch[u] + (u == edited ? duration : weights[u]);
            best = std::max(best, finish);
            for (Id k=offsets[u]; k<offsets[u+1]; ++k) {
                Id v = children[k];
                scratch[v] = std::max(scratch[v], finish);
            }
        }
        return best;
    }
};

struct LazyDP : FullDP {
    Vec finish, tail, extra, floors;
    std::vector<char> path;
    Time span;

    LazyDP(const Vec& w, const std::vector<Edge>& edges)
        : FullDP(w,edges), finish(w.size()), tail(w.size()), extra(w.size()),
          floors(w.size(), -1), path(w.size()) {
        const Id none = w.size();
        std::vector<Id> previous(w.size(), none);
        for (Id u : order) {
            finish[u] += w[u];
            for (Id k=offsets[u]; k<offsets[u+1]; ++k) {
                Id v = children[k];
                if (finish[u] > finish[v] || previous[v] == none) {
                    finish[v] = finish[u];
                    previous[v] = u;
                }
            }
        }
        Id tip = std::max_element(finish.begin(), finish.end()) - finish.begin();
        span = finish[tip];
        for (; tip != none; tip = previous[tip]) path[tip] = 1;
        for (auto it=order.rbegin(); it!=order.rend(); ++it) {
            Id u = *it;
            for (Id k=offsets[u]; k<offsets[u+1]; ++k) {
                Id v = children[k];
                tail[u] = std::max(tail[u], w[v] + tail[v]);
            }
            extra[u] = (finish[u] - w[u]) + tail[u];
        }
    }

    Time predict(Id u, Time d) {
        Time via = extra[u] + d;
        if (d >= weights[u]) return std::max(span, via);
        if (!path[u]) return span;
        if (floors[u] < 0) floors[u] = FullDP::predict(u, 0);
        return std::max(floors[u], via);
    }
};

struct HeapSensitivity : LazyDP {
    Vec bypass;
    HeapSensitivity(const Vec& w, const std::vector<Edge>& edges)
        : LazyDP(w, edges), bypass(w.size(), span) {
        std::vector<Id> before(w.size());
        Id count = 0;
        for (Id u : order) {
            before[u] = count;
            count += path[u];
        }
        std::priority_queue<std::pair<Time, Id>> active;
        for (Id u : order) {
            if (!indegree[u] && before[u]) active.emplace(w[u] + tail[u], before[u]);
        }
        for (Id u : order) {
            Id p = before[u];
            if (path[u]) {
                while (!active.empty() && active.top().second <= p) active.pop();
                bypass[u] = active.empty() ? -1 : active.top().first;
            }
            Id start = p + path[u];
            for (Id k=offsets[u]; k<offsets[u+1]; ++k) {
                Id v = children[k];
                if (start < before[v]) active.emplace(finish[u] + w[v] + tail[v], before[v]);
            }
            if (offsets[u] == offsets[u+1] && start < count) active.emplace(finish[u], count);
        }
    }
    Time predict(Id u, Time d) const { return std::max(bypass[u], extra[u] + d); }
};

Time ns(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}

Time process_ns() {
#ifdef CLOCK_PROCESS_CPUTIME_ID
    struct timespec t{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)==0)
        return static_cast<Time>(t.tv_sec)*1000000000 + t.tv_nsec;
#endif
    return static_cast<Time>(std::clock()*(1e9/CLOCKS_PER_SEC));
}

template<class Method>
Trial sample(const Vec& w, const std::vector<Edge>& es, const std::vector<Query>& qs,
             Id batch, const Vec& gold, Time* elapsed_cpu_ns=nullptr) {
    Vec outputs(batch);
    auto cpu_start = elapsed_cpu_ns ? process_ns() : 0;
    auto a = Clock::now();
    Method method(w,es);
    auto b = Clock::now();
    for (Id i=0; i<batch; ++i) outputs[i] = method.predict(qs[i].first, qs[i].second);
    auto c = Clock::now();
    if (elapsed_cpu_ns) *elapsed_cpu_ns = process_ns()-cpu_start;
    if (!std::equal(outputs.begin(), outputs.end(), gold.begin())) {
        throw std::runtime_error("prediction mismatch");
    }
    return {ns(a,b), ns(b,c), ns(a,c)};
}

void print_stats(const std::vector<Trial>& trials, const Vec& cpu_trials) {
    std::cout << "{";
    const char* names[] = {"build_ns","queries_ns","total_ns"};
    for (Id j=0; j<3; ++j) {
        Vec times;
        for (auto t : trials) times.push_back(t[j]);
        std::sort(times.begin(), times.end());
        Id n = times.size();
        Time low = times[(n-1)/2], high = times[n/2];
        std::cout << "\"" << names[j] << "\":" << low + (high-low)/2 << ",";
    }
    Vec sorted_cpu(cpu_trials);
    std::sort(sorted_cpu.begin(), sorted_cpu.end());
    Id mid = sorted_cpu.size()/2;
    Time lo = sorted_cpu[(sorted_cpu.size()-1)/2];
    std::cout << "\"build_queries_cpu_ns\":" << lo+(sorted_cpu[mid]-lo)/2 << ",";
    std::cout << "\"build_queries_cpu_trials_ns\":[";
    for (Id i=0; i<cpu_trials.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << cpu_trials[i];
    }
    std::cout << "],\"trials_ns\":[";
    for (Id i=0; i<trials.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << "[" << trials[i][0] << "," << trials[i][1] << "," << trials[i][2] << "]";
    }
    std::cout << "]}";
}

// Linux kernel RSS figures include the process, input, oracle and results.
Time proc_kib(const std::string& name) {
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input,line)) {
        if (!line.compare(0, name.size(), name)) return std::stoll(line.substr(name.size()));
    }
    return -1;
}

struct CapacityGraph {
    Vec weights, gold;
    std::vector<Edge> edges;
    std::vector<Query> queries;

    CapacityGraph(Id n, const std::string& shape) : weights(n), gold(n) {
        std::mt19937 engine(421);
        for (Time& w : weights) w = 1 + engine()%100;
        std::vector<Id> permutation(n);
        std::iota(permutation.begin(),permutation.end(),0);
        // Explicit Fisher-Yates for identical queries across STL versions.
        for (Id k=n; k>1; --k) std::swap(permutation[k-1],permutation[engine()%k]);
        queries.reserve(n);
        for (Id i=0; i<n; ++i) {
            Id u=permutation[i];
            Time d=weights[u];
            queries.emplace_back(u, i%2 ? d + std::max<Time>(1,d/2) : d/2);
        }
        if (shape=="series_diamonds") {
            // Each stage contributes max(left,right) + join; stages are serial.
            Time span = 0;
            edges.reserve(n/3*4);
            for (Id u=0; u<n; u+=3) {
                if (u>0) {
                    edges.emplace_back(u-1,u);
                    edges.emplace_back(u-1,u+1);
                }
                edges.emplace_back(u,u+2);
                edges.emplace_back(u+1,u+2);
                span += std::max(weights[u], weights[u+1])+weights[u+2];
            }
            for (Id i=0; i<n; ++i) {
                auto [u,d]=queries[i];
                Id start=u-u%3;
                if (u%3==2) gold[i] = span-weights[u]+d;
                else {
                    Time other=weights[start+1-u%3];
                    gold[i]=span-std::max(other,weights[u])+std::max(other,d);
                }
            }
        } else if (shape=="parallel_chains") {
            // Width is fixed. Each chain contributes a sum; all chains must finish.
            Id width=std::min<Id>(256,n);
            Vec sums(width);
            edges.reserve(n-width);
            for (Id i=0; i<n; ++i) {
                sums[i%width]+=weights[i];
                if (i>=width) edges.emplace_back(i-width,i);
            }
            Id best=std::max_element(sums.begin(),sums.end())-sums.begin();
            Time second=0;
            for (Id c=0; c<width; ++c) if (c!=best) second=std::max(second,sums[c]);
            for (Id i=0; i<n; ++i) {
                auto [u,d]=queries[i];
                Id c=u%width;
                gold[i]=std::max(sums[c]-weights[u]+d,c==best ? second : sums[best]);
            }
        } else throw std::runtime_error("unknown capacity shape");
    }
};

int capacity(const std::string& shape, Id n, const std::string& method) {
    if (n<3 || n>30000000 || n%3) throw std::runtime_error("tasks must be a multiple of 3 in [3,30000000]");
    CapacityGraph graph(n,shape);
    if (n<=4096) {
        FullDP reference(graph.weights,graph.edges);
        for (Id i=0; i<n; ++i) {
            auto [u,d]=graph.queries[i];
            if (reference.predict(u,d)!=graph.gold[i]) throw std::runtime_error("analytical oracle mismatch");
        }
    }
    Time input_rss=proc_kib("VmRSS:");
    Trial times{};
    Time cpu_ns = 0;
    if (method=="pathprobe")
        times=sample<pathprobe::DurationIndex>(graph.weights,graph.edges,graph.queries,n,graph.gold,&cpu_ns);
    else if (method=="heap_sensitivity")
        times=sample<HeapSensitivity>(graph.weights,graph.edges,graph.queries,n,graph.gold,&cpu_ns);
    else throw std::runtime_error("unknown benchmark method");
    std::cout << "{\"nodes\":" << n << ",\"edges\":" << graph.edges.size()
              << ",\"build_ns\":" << times[0] << ",\"queries_ns\":" << times[1]
              << ",\"total_ns\":" << times[2] << ",\"build_queries_cpu_ns\":" << cpu_ns << ",\"input_rss_kib\":" << input_rss
              << ",\"process_peak_rss_kib\":" << proc_kib("VmHWM:") << "}\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc==5 && std::string(argv[1])=="--capacity")
        return capacity(argv[2],std::stoull(argv[3]),argv[4]);
    if (argc!=1) return 2;
    Id n, m, repeats;
    if (!(std::cin >> n >> m >> repeats) || n == 0 || repeats == 0) return 2;
    Vec w(n);
    for (Time& value : w) if (!(std::cin >> value) || value < 0) return 2;
    std::vector<Edge> es(m);
    for (auto& [u,v] : es) if (!(std::cin >> u >> v) || u >= n || v >= n) return 2;
    std::vector<Query> qs(n);
    for (auto& [u,d] : qs) if (!(std::cin >> u >> d) || u >= n || d < 0) return 2;
    // export_native.py bounds all path sums for the benchmark-only comparators.
    FullDP oracle(w,es);
    Vec gold;
    for (auto [u,d] : qs) gold.push_back(oracle.predict(u,d));
    std::vector<Id> batches{1, std::min<Id>(10,n), n};
    batches.erase(std::unique(batches.begin(),batches.end()),batches.end());
    const char* names[] = {"full_dp","critical_path_lazy_cache","heap_sensitivity","pathprobe"};
    std::cout << "{\"nodes\":" << n << ",\"edges\":" << m
              << ",\"makespan\":" << oracle.predict(n,0) << ",\"batches\":[";
    for (Id h=0; h<batches.size(); ++h) {
        if (h) std::cout << ",";
        Id batch = batches[h];
        Methods results;
        std::array<Vec,4> cpu_results;
        for (Id r=0; r<repeats; ++r) {
            for (Id k=0; k<results.size(); ++k) {
                Id j=(k+r)%results.size();
                Trial t;
                Time cpu;
                if (j==0) t = sample<FullDP>(w,es,qs,batch,gold,&cpu);
                else if (j==1) t = sample<LazyDP>(w,es,qs,batch,gold,&cpu);
                else if (j==2) t = sample<HeapSensitivity>(w,es,qs,batch,gold,&cpu);
                else t = sample<pathprobe::DurationIndex>(w,es,qs,batch,gold,&cpu);
                results[j].push_back(t);
                cpu_results[j].push_back(cpu);
            }
        }
        std::uint64_t hash = 14695981039346656037ULL;
        for (Id i=0; i<batch; ++i) {
            auto value = static_cast<std::uint64_t>(gold[i]);
            for (int byte=0; byte<8; ++byte, value>>=8) {
                hash ^= value & 255;
                hash *= 1099511628211ULL;
            }
        }
        std::cout << "{\"queries\":" << batch << ",\"predictions_fnv1a64_le\":\"" << std::hex
                  << hash << std::dec << "\",\"methods\":{";
        for (Id j=0; j<results.size(); ++j) {
            if (j) std::cout << ",";
            std::cout << "\"" << names[j] << "\":";
            print_stats(results[j], cpu_results[j]);
        }
        std::cout << "}}";
    }
    std::cout << "]}\n";
}
