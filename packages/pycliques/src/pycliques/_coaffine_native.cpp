// Shared C++17 engine for coaffine endomorphisms and octahedral retractions.
// Protocol v1: integer vertex positions, verified again by Python.
// A bounded search reports LIMIT_REACHED, never a negative theorem.
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int CAPACITY = 2048;
using Mask = std::bitset<CAPACITY>;
using Clock = std::chrono::steady_clock;
int n = 0, min_dimension = 3;
std::int64_t max_states = 20000, states = 0, revisions = 0, assignments = 0;
double seconds_limit = 3600.0;
Clock::time_point started;
bool limited = false, fiber_order = false;
Mask full_vertices;
std::vector<Mask> closed;
std::vector<Mask> complement_open;
std::vector<Mask> adjacency;
std::vector<std::pair<int, int>> edges;
std::vector<std::pair<int, int>> answer_pairs;
std::vector<int> answer_projection, answer_endomorphism;

bool timed_out() {
    return std::chrono::duration<double>(Clock::now() - started).count() >= seconds_limit;
}

// Portable fixed-width bitset iteration. GNU libstdc++ provides a fast
// extension; other C++17 implementations use the standard test operation.
int next_bit(const Mask& bits, int previous = -1) {
#ifdef __GLIBCXX__
    return static_cast<int>(previous < 0 ? bits._Find_first() : bits._Find_next(previous));
#else
    for (int i = previous + 1; i < n; ++i)
        if (bits.test(i)) return i;
    return CAPACITY;
#endif
}

// A value excluded from every possible image of v cannot be the image
// of any neighbor of v. Store the transpose of the domains to apply
// these deletions to all affected sources at once.
Mask forbidden(const Mask& domain) {
    Mask excluded = full_vertices;
    for (int v = next_bit(domain); v < n; v = next_bit(domain, v)) {
        excluded &= complement_open[v];
        if (excluded.none()) break;
    }
    return excluded;
}

struct State {
    std::vector<Mask> domains, columns, processed;
};

bool propagate(State& state, std::vector<int>& queue) {
    Mask queued;
    for (int u : queue) queued.set(u);
    while (!queue.empty()) {
        if (timed_out()) {
            limited = true;
            return false;
        }
        const int v = queue.back();
        queue.pop_back();
        queued.reset(v);
        if (state.domains[v].none()) return false;
        const Mask bad = forbidden(state.domains[v]);
        const Mask fresh = bad & ~state.processed[v];
        state.processed[v] = bad;
        for (int t = next_bit(fresh); t < n; t = next_bit(fresh, t)) {
            const Mask removed = state.columns[t] & adjacency[v];
            state.columns[t] &= ~removed;
            for (int u = next_bit(removed); u < n; u = next_bit(removed, u)) {
                ++revisions;
                state.domains[u].reset(t);
                if (state.domains[u].none()) return false;
                if (!queued.test(u)) {
                    queued.set(u);
                    queue.push_back(u);
                }
            }
        }
    }
    return true;
}

void assign(State& state, int u, int t) {
    Mask others = state.domains[u];
    others.reset(t);
    for (int v = next_bit(others); v < n; v = next_bit(others, v))
        state.columns[v].reset(u);
    state.domains[u].reset();
    state.domains[u].set(t);
}

bool positive(const State& state) {
    std::vector<int> f(n, -1), projection(n, -1);
    for (int u = 0; u < n; ++u) {
        if (state.domains[u].count() != 1) throw std::logic_error("non-singleton final domain");
        for (int v = 0; v < n; ++v) {
            if (state.domains[u].test(v)) {
                f[u] = v;
                break;
            }
        }
        if (f[u] < 0 || closed[u].test(f[u])) throw std::logic_error("not coaffine");
    }
    for (const auto& [u, v] : edges) {
        if (!closed[f[u]].test(f[v])) throw std::logic_error("not a weak endomorphism");
    }
    if (min_dimension == 0) {
        answer_endomorphism = std::move(f);
        return true;
    }
    std::set<int> image;
    for (int u = 0; u < n; ++u) {
        projection[u] = f[f[u]];
        image.insert(projection[u]);
    }
    if (image.size() < static_cast<std::size_t>(2LL * min_dimension) || image.size() % 2) return false;
    for (const int u : image) {
        if (projection[u] != u) return false;
    }
    std::set<int> remaining(image);
    std::vector<std::pair<int, int>> pairs;
    while (!remaining.empty()) {
        const int u = *remaining.begin();
        int antipode = -1;
        for (const int v : image) {
            if (u != v && !closed[u].test(v)) {
                if (antipode != -1) return false;
                antipode = v;
            }
        }
        if (antipode < 0 || remaining.count(antipode) == 0) return false;
        int reverse_count = 0;
        for (const int w : image) {
            if (w != antipode && !closed[antipode].test(w)) {
                if (w != u) return false;
                ++reverse_count;
            }
        }
        if (reverse_count != 1) return false;
        pairs.emplace_back(u, antipode);
        remaining.erase(u);
        remaining.erase(antipode);
    }
    answer_pairs = std::move(pairs);
    answer_projection = std::move(projection);
    answer_endomorphism = std::move(f);
    return true;
}

bool dfs(const State& state) {
    if (states >= max_states || timed_out()) {
        limited = true;
        return false;
    }
    ++states;
    int pivot = -1, min_size = n + 1, max_degree = -1;
    for (int u = 0; u < n; ++u) {
        const int size = static_cast<int>(state.domains[u].count());
        if (size == 0) throw std::logic_error("empty domain in dfs");
        if (size > 1 && (size < min_size ||
                         (size == min_size && static_cast<int>(adjacency[u].count()) > max_degree))) {
            pivot = u;
            min_size = size;
            max_degree = static_cast<int>(adjacency[u].count());
        }
    }
    if (pivot < 0) return positive(state);
    std::vector<int> choices;
    for (int image = next_bit(state.domains[pivot]); image < n;
         image = next_bit(state.domains[pivot], image)) {
        choices.push_back(image);
    }
    if (fiber_order) {
        // Keep the research prototype's ordering, including std::sort's
        // implementation-defined ordering of equal-size fibers.
        std::sort(choices.begin(), choices.end(), [&](int a, int b) {
            return state.columns[a].count() > state.columns[b].count();
        });
    }
    for (int image : choices) {
        if (timed_out()) {
            limited = true;
            return false;
        }
        ++assignments;
        State next = state;
        assign(next, pivot, image);
        std::vector<int> queue{pivot};
        if (propagate(next, queue) && dfs(next)) return true;
        if (limited) return false;
    }
    return false;
}

void print_result(bool found) {
    std::cout << "RESULT " << (found ? "FOUND" : (limited ? "LIMIT_REACHED" : "EXHAUSTED"))
              << ' ' << states << ' ' << revisions << ' ' << assignments << '\n';
    if (!found) return;
    if (min_dimension == 0) {
        std::cout << "ENDOMORPHISM";
        for (const int i : answer_endomorphism) std::cout << ' ' << i;
        std::cout << '\n';
        return;
    }
    std::cout << "PAIRS";
    for (const auto& [a, b] : answer_pairs) std::cout << ' ' << a << ',' << b;
    std::cout << '\n' << "PROJECTION";
    for (const int i : answer_projection) std::cout << ' ' << i;
    std::cout << '\n' << "ENDOMORPHISM";
    for (const int i : answer_endomorphism) std::cout << ' ' << i;
    std::cout << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::invalid_argument("usage: solver input.txt max_states time_seconds min_dimension");
        max_states = std::stoll(argv[2]);
        seconds_limit = std::stod(argv[3]);
        min_dimension = std::stoi(argv[4]);
        if (argc == 6) {
            const std::string order = argv[5];
            if (order != "fiber" && order != "label") throw std::invalid_argument("invalid value order");
            fiber_order = order == "fiber";
        }
        if (max_states < 1 || seconds_limit <= 0 || (min_dimension != 0 && min_dimension < 3)) {
            throw std::invalid_argument("invalid bound");
        }
        std::ifstream input(argv[1]);
        int m = 0;
        if (!(input >> n >> m) || n < 0 || n > CAPACITY || m < 0 ||
            m > n * (n - 1) / 2) throw std::invalid_argument("invalid graph size");
        closed.assign(n, {});
        adjacency.assign(n, {});
        for (int u = 0; u < n; ++u) closed[u].set(u);
        std::set<std::pair<int, int>> seen;
        for (int k = 0; k < m; ++k) {
            int u = -1, v = -1;
            if (!(input >> u >> v) || u < 0 || v < 0 || u >= n || v >= n || u == v) {
                throw std::invalid_argument("invalid edge");
            }
            if (u > v) std::swap(u, v);
            if (!seen.emplace(u, v).second) throw std::invalid_argument("duplicate edge");
            closed[u].set(v);
            closed[v].set(u);
            adjacency[u].set(v);
            adjacency[v].set(u);
            edges.emplace_back(u, v);
        }
        started = Clock::now();
        if (static_cast<std::int64_t>(n) < 2LL * min_dimension) {
            print_result(false);
            return 0;
        }
        for (int u = 0; u < n; ++u) full_vertices.set(u);
        complement_open.reserve(n);
        for (int u = 0; u < n; ++u) {
            complement_open.push_back(full_vertices & ~closed[u]);
        }
        State initial{complement_open, complement_open, std::vector<Mask>(n)};
        std::vector<int> queue;
        for (int u = 0; u < n; ++u) {
            if (initial.domains[u].none()) {
                print_result(false);
                return 0;
            }
            queue.push_back(u);
        }
        bool found = false;
        if (propagate(initial, queue)) found = dfs(initial);
        print_result(found);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "pycliques native solver: " << e.what() << '\n';
        return 2;
    }
}
