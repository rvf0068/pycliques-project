// Standalone C++17 acceleration for pycliques octahedral-coaffine search.
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
constexpr int CAPACITY = 512;
using Mask = std::bitset<CAPACITY>;
using Clock = std::chrono::steady_clock;
int n = 0, min_dimension = 3;
std::int64_t max_states = 20000, states = 0, revisions = 0, assignments = 0;
double seconds_limit = 3600.0;
Clock::time_point started;
bool limited = false;
std::vector<Mask> closed;
std::vector<std::vector<int>> neighbors;
std::vector<std::pair<int, int>> edges;
std::vector<std::pair<int, int>> answer_pairs;
std::vector<int> answer_projection, answer_endomorphism;

bool timed_out() {
    return std::chrono::duration<double>(Clock::now() - started).count() >= seconds_limit;
}

Mask support(Mask domain) {
    Mask result;
    for (int v = 0; v < n; ++v) {
        if (domain.test(v)) result |= closed[v];
    }
    return result;
}

struct State {
    std::vector<Mask> domains;
    std::vector<Mask> supports;
};

bool propagate(State& state, std::deque<std::pair<int, int>>& queue) {
    std::uint64_t steps = 0;
    while (!queue.empty()) {
        if ((++steps % 4096) == 0 && timed_out()) {
            limited = true;
            return false;
        }
        const auto [u, v] = queue.front();
        queue.pop_front();
        const Mask revised = state.domains[u] & state.supports[v];
        if (revised == state.domains[u]) continue;
        ++revisions;
        if (revised.none()) return false;
        state.domains[u] = revised;
        state.supports[u] = support(revised);
        for (const int w : neighbors[u]) {
            if (w != v) queue.emplace_back(w, u);
        }
    }
    return true;
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
    std::set<int> image;
    for (int u = 0; u < n; ++u) {
        projection[u] = f[f[u]];
        image.insert(projection[u]);
    }
    if (image.size() < static_cast<std::size_t>(2 * min_dimension) || image.size() % 2) return false;
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
                         (size == min_size && static_cast<int>(neighbors[u].size()) > max_degree))) {
            pivot = u;
            min_size = size;
            max_degree = static_cast<int>(neighbors[u].size());
        }
    }
    if (pivot < 0) return positive(state);
    for (int image = 0; image < n; ++image) {
        if (!state.domains[pivot].test(image)) continue;
        ++assignments;
        State next = state;
        next.domains[pivot].reset();
        next.domains[pivot].set(image);
        next.supports[pivot] = closed[image];
        std::deque<std::pair<int, int>> queue;
        for (const int w : neighbors[pivot]) queue.emplace_back(w, pivot);
        if (propagate(next, queue) && dfs(next)) return true;
        if (limited) return false;
    }
    return false;
}

void print_result(bool found) {
    std::cout << "RESULT " << (found ? "FOUND" : (limited ? "LIMIT_REACHED" : "EXHAUSTED"))
              << ' ' << states << ' ' << revisions << ' ' << assignments << '\n';
    if (!found) return;
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
        if (argc != 5) throw std::invalid_argument("usage: solver input.txt max_states time_seconds min_dimension");
        max_states = std::stoll(argv[2]);
        seconds_limit = std::stod(argv[3]);
        min_dimension = std::stoi(argv[4]);
        if (max_states < 1 || seconds_limit <= 0 || min_dimension < 3) {
            throw std::invalid_argument("invalid bound");
        }
        std::ifstream input(argv[1]);
        int m = 0;
        if (!(input >> n >> m) || n < 1 || n > CAPACITY || m < 0 ||
            m > n * (n - 1) / 2) throw std::invalid_argument("invalid graph size");
        closed.assign(n, {});
        neighbors.assign(n, {});
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
            neighbors[u].push_back(v);
            neighbors[v].push_back(u);
            edges.emplace_back(u, v);
        }
        started = Clock::now();
        if (n < 2 * min_dimension) {
            print_result(false);
            return 0;
        }
        Mask all;
        for (int u = 0; u < n; ++u) all.set(u);
        State initial;
        for (int u = 0; u < n; ++u) {
            const Mask domain = all & ~closed[u];
            if (domain.none()) {
                print_result(false);
                return 0;
            }
            initial.domains.push_back(domain);
            initial.supports.push_back(support(domain));
        }
        std::deque<std::pair<int, int>> queue;
        for (const auto& [u, v] : edges) {
            queue.emplace_back(u, v);
            queue.emplace_back(v, u);
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
