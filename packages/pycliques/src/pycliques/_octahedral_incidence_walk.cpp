// Adaptive weighted cut walk adapted from blind_octahedral_retractions.org.
// GNU libstdc++ bit scanning; only validated compact supports enter this worker.
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
using Word = uint64_t;
using VertexSet = bitset<1280>;
struct Cut {
  int a, b;
  Word g;
  VertexSet cover;
};
bool adaptive;
int n, ground_bytes;
VertexSet full;
Word ground;
vector<Word> supports;
vector<VertexSet> disjoint;
vector<array<VertexSet, 256>> atom_table;
VertexSet weight_bits[5];
int miss_weight[1280]{};
int weighted_count(const VertexSet &x) {
  int r = x.count();
  for (int j = 0; j < 5; j++)
    r += (x & weight_bits[j]).count() * (1 << j);
  return r;
}
mt19937 rng;
int draw(int k) { return rng() % k; }
VertexSet supports_meeting(Word bits) {
  VertexSet ans;
  for (int i = 0; i < ground_bytes; i++) {
    ans |= atom_table[i][bits & 255];
    bits >>= 8;
  }
  return ans;
}
VertexSet coverage(Word p) {
  return full & ~(supports_meeting(p) & supports_meeting(ground ^ p));
}
struct Edge {
  int t, back, cap;
};
// Integral maximum flow. The final unsuccessful BFS gives the source side.
struct Flow {
  vector<vector<Edge>> g;
  vector<int> level, cur;
  explicit Flow(int size) : g(size), level(size), cur(size) {}
  void add(int a, int b, int c) {
    int j = g[b].size(), i = g[a].size();
    g[a].push_back({b, j, c});
    g[b].push_back({a, i, 0});
  }
  bool bfs(int s, int t) {
    fill(level.begin(), level.end(), -1);
    vector<int> q = {s};
    level[s] = 0;
    for (size_t k = 0; k < q.size(); k++) {
      int x = q[k];
      for (const auto &e : g[x])
        if (e.cap && level[e.t] < 0) {
          level[e.t] = level[x] + 1;
          q.push_back(e.t);
        }
    }
    return level[t] >= 0;
  }
  int dfs(int x, int t, int amount) {
    if (x == t)
      return amount;
    for (int &i = cur[x]; i < (int)g[x].size(); ++i) {
      Edge &e = g[x][i];
      if (e.cap && level[e.t] == level[x] + 1) {
        int f = dfs(e.t, t, min(amount, e.cap));
        if (f) {
          e.cap -= f;
          g[e.t][e.back].cap += f;
          return f;
        }
      }
    }
    return 0;
  }
  void solve(int s, int t) {
    while (bfs(s, t)) {
      fill(cur.begin(), cur.end(), 0);
      while (dfs(s, t, 1000000000)) {
      }
    }
  }
};
// Options have finite reward arcs. All containment implications are hard arcs.
Word best_anchored_cut(int a, int b, const VertexSet &need,
                       const VertexSet &covered_once, int variation, int force = -1) {
  const int bgn = 2 + ground_bytes * 8;
  int size = bgn + disjoint[a].count() + disjoint[b].count();
  Flow fl(size);
  int id = bgn;
  const int inf = 100000000;
  for (Word y = supports[a]; y; y &= y - 1)
    fl.add(0, 2 + __builtin_ctzll(y), inf);
  for (Word y = supports[b]; y; y &= y - 1)
    fl.add(2 + __builtin_ctzll(y), 1, inf);
  for (int dir = 0; dir < 2; dir++) {
    const auto &hs = dir ? disjoint[a] : disjoint[b];
    for (int x = hs._Find_first(); x < n; x = hs._Find_next(x)) {
      int w = need[x]           ? (adaptive ? 1300 * (1 + miss_weight[x]) : 3000) +
                                      (variation ? draw(160) : 0)
              : covered_once[x] ? 10 + (variation ? draw(10) : 0)
                                : 1;
      if (x == force)
        w = 410000;
      int t = id++;
      if (!dir)
        fl.add(0, t, w);
      else
        fl.add(t, 1, w);
      for (Word y = supports[x]; y; y &= y - 1) {
        int v = 2 + __builtin_ctzll(y);
        if (!dir)
          fl.add(t, v, inf);
        else
          fl.add(v, t, inf);
      }
    }
  }
  fl.solve(0, 1);
  Word ans = 0;
  for (int j = 0; j < ground_bytes * 8; j++)
    if (fl.level[2 + j] >= 0)
      ans |= Word(1) << j;
  return ans & ground;
}
// Index supports meeting byte masks, for constant-width coverage queries.
void load() {
  ifstream ff("supports.txt");
  Word x;
  while (ff >> x)
    supports.push_back(x);
  n = supports.size();
  if (n == 0 || n > 1280 ||
      find(supports.begin(), supports.end(), 0) != supports.end())
    throw runtime_error("Expected 1..1280 nonempty supports");
  ground = 0;
  for (Word y : supports)
    ground |= y;
  ground_bytes = (64 - __builtin_clzll(ground) + 7) / 8;
  atom_table.resize(ground_bytes);
  for (int v = 0; v < n; v++) {
    full.set(v);
    for (Word m = supports[v]; m; m &= m - 1) {
      int k = __builtin_ctzll(m);
      atom_table[k / 8][1 << (k % 8)].set(v);
    }
  }
  for (auto &t : atom_table)
    for (int j = 1; j < 256; j++)
      if (j & (j - 1))
        t[j] = t[j & (j - 1)] | t[j & -j];
  disjoint.resize(n);
  for (int v = 0; v < n; v++)
    disjoint[v] = full & ~supports_meeting(supports[v]);
}
// Keep uncovered supports, singly covered supports and eligible new endpoints.
struct Totals {
  VertexSet need, covered_once, allowed;
};
Totals cut_totals(const vector<Cut> &state) {
  VertexSet covered, two, pool = full;
  for (auto &p : state) {
    two |= covered & p.cover;
    covered |= p.cover;
    pool &= ~(disjoint[p.a] | disjoint[p.b]);
  }
  return {full ^ covered, covered & ~two, pool};
}
int score(const VertexSet &prop, const VertexSet &need,
          const VertexSet &covered_once) {
  return weighted_count(prop & need) * 1200 + (prop & covered_once).count() * 5 +
         prop.count() / 4;
}
bool add_pair(vector<Cut> &st, int olda, int oldb, int variation) {
  auto [need, covered_once, allow] = cut_totals(st);
  vector<pair<int, int>> edges;
  for (int a = allow._Find_first(); a < n; a = allow._Find_next(a)) {
    VertexSet opts = disjoint[a] & allow;
    for (int b = opts._Find_next(a); b < n; b = opts._Find_next(b))
      edges.emplace_back(a, b);
  }
  if (edges.empty())
    return false;
  shuffle(edges.begin(), edges.end(), rng);
  // Rank candidate anchors by optimistic coverage and the best cuts represented by
  // single supports.
  vector<pair<int, int>> rank;
  rank.reserve(edges.size());
  for (int i = 0; i < (int)edges.size(); i++) {
    auto [a, b] = edges[i];
    int c = score(disjoint[a] | disjoint[b], need, covered_once);
    if ((a == olda && b == oldb) || (a == oldb && b == olda))
      c += 10000000;
    rank.emplace_back(c + draw(900 + variation * 300), i);
  }
  int limit = min<int>(edges.size(), 45);
  partial_sort(rank.begin(), rank.begin() + limit, rank.end(), greater<>());
  struct Candidate {
    int val, a, b;
    Word g;
  };
  vector<Candidate> cand;
  for (int j = 0; j < limit; j++) {
    auto [a, b] = edges[rank[j].second];
    Word chosen = supports[a];
    int best = score(coverage(chosen), need, covered_once);
    auto test = [&](Word g) {
      int s = score(coverage(g), need, covered_once);
      if (s > best) {
        best = s;
        chosen = g;
      }
    };
    test(ground ^ supports[b]);
    for (int d = 0; d < 2; d++) {
      VertexSet opts = (d ? disjoint[a] : disjoint[b]) & need;
      vector<int> vs;
      for (int x = opts._Find_first(); x < n; x = opts._Find_next(x))
        vs.push_back(x);
      shuffle(vs.begin(), vs.end(), rng);
      int num = min<int>(5, vs.size());
      for (int i = 0; i < num; i++) {
        Word g = (d ? supports[b] : supports[a]) | supports[vs[i]];
        test(d ? (ground ^ g) : g);
      }
    }
    cand.push_back({best + draw(95 + variation * 180), a, b, chosen});
  }
  sort(cand.begin(), cand.end(), [](auto &x, auto &y) { return x.val > y.val; });
  int top = min<int>(cand.size(), 2 + (variation == 0));
  Candidate chosen = cand[0];
  int finalScore = -1;
  for (int k = 0; k < top; k++) {
    auto z = cand[k];
    Word g = best_anchored_cut(z.a, z.b, need, covered_once, variation);
    int v = score(coverage(g), need, covered_once);
    int v2 = score(coverage(z.g), need, covered_once);
    if (v >= v2)
      z.g = g;
    else
      v = v2;
    int adjust = v + draw(70 + variation * 225);
    if (adjust > finalScore) {
      finalScore = adjust;
      chosen = z;
    }
  }
  st.push_back({chosen.a, chosen.b, chosen.g, coverage(chosen.g)});
  return true;
}
int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  adaptive = string(argv[1]) == "weighted";
  rng.seed(stoul(argv[2]));
  int steps = stoi(argv[3]);
  int min_dimension = stoi(argv[4]);
  load();
  if (any_of(disjoint.begin(), disjoint.end(),
             [](const VertexSet &row) { return row.none(); }))
    return 0;
  vector<vector<pair<int, int>>> init;
  ifstream stream("starts.txt");
  int sz;
  while (stream >> sz) {
    vector<pair<int, int>> e;
    for (int i = 0; i < sz; i++) {
      int a, b;
      stream >> a >> b;
      e.emplace_back(a, b);
    }
    init.push_back(e);
  }
  vector<Cut> best, local;
  int fewest_missed = n + 1, local_cost = n + 1, stalled_steps = 0;
  {
    ifstream loadState("prior.txt");
    int e, k;
    if (loadState >> e >> k) {
      for (int j = 0; j < k; j++) {
        int a, b;
        Word g;
        loadState >> a >> b >> g;
        best.push_back({a, b, g, coverage(g)});
      }
      fewest_missed = cut_totals(best).need.count();
    }
  }
  int horizon = 0;

  for (int it = 0; it < steps; it++) {
    if (it == 0 || stalled_steps > (adaptive ? 270 : 70) ||
        it % (adaptive ? 1200 : 157) == 0) {
      if (best.size() && draw(5) < 4) {
        local = best;
        local_cost = fewest_missed;
      } else {
        local.clear();
        if (!init.empty() && draw(3)) {
          for (auto [a, b] : init[draw(init.size())]) {
            Word g = draw(2) ? supports[a] : ground ^ supports[b];
            local.push_back({a, b, g, coverage(g)});
          }
        }
        while (add_pair(local, -1, -1, 2)) {
        }
        local_cost = cut_totals(local).need.count();
      }
      stalled_steps = 0;
    }
    auto prev = cut_totals(local);
    if (adaptive && ((it % 23 == 2 && stalled_steps > 12) || it % 59 == 11)) {
      for (int x = prev.need._Find_first(); x < n; x = prev.need._Find_next(x)) {
        if (miss_weight[x] < 15)
          miss_weight[x]++;
      }
      if (++horizon == 29) {
        horizon = 0;
        for (int x = 0; x < n; x++)
          miss_weight[x] /= 2;
      }
      for (int j = 0; j < 5; j++) {
        weight_bits[j].reset();
        for (int x = 0; x < n; x++)
          if (miss_weight[x] >> j & 1)
            weight_bits[j].set(x);
      }
    }
    if (adaptive)
      local_cost = weighted_count(prev.need);
    auto trial = local;
    int qty = 1;
    if (draw(100) < 48)
      qty = 2 + draw(3);
    if (draw(100) < 15)
      qty = 4 + draw(4);
    vector<Cut> removed;
    // Try a new pair compatible with a repeatedly uncovered support.
    if (adaptive && !trial.empty() && trial.size() < 64 && prev.need.any() &&
        draw(100) < 57) {
      vector<int> holes;
      for (int x = prev.need._Find_first(); x < n; x = prev.need._Find_next(x))
        holes.push_back(x);
      int x = holes[draw(holes.size())];
      vector<int> cand;
      for (int a = disjoint[x]._Find_first(); a < n; a = disjoint[x]._Find_next(a))
        cand.push_back(a);
      shuffle(cand.begin(), cand.end(), rng);
      int val = 1e9, pa = -1, pb = -1;
      vector<Word> blockers(n, 0);
      for (int j = 0; j < (int)trial.size(); j++) {
        const auto &p = trial[j];
        VertexSet h = disjoint[p.a] | disjoint[p.b];
        for (int v = h._Find_first(); v < n; v = h._Find_next(v))
          blockers[v] |= Word(1) << j;
      }
      int stop = min<int>(cand.size(), 5 + draw(18));
      Word saved = 0;
      int intensity = 35 + draw(150);
      for (int t = 0; t < stop; t++) {
        int a = cand[t];
        for (int b = disjoint[a]._Find_first(); b < n;
             b = disjoint[a]._Find_next(b)) {
          Word bits = blockers[a] | blockers[b];
          int local = __builtin_popcountll(bits) * intensity + draw(160);
          if (a == x || b == x)
            local -= 15;
          if (local < val) {
            val = local;
            pa = a;
            pb = b;
            saved = bits;
          }
        }
      }
      if (draw(100) < 22)
        saved |= Word(1) << draw(trial.size());
      vector<Cut> hold;
      for (int j = 0; j < (int)trial.size(); j++) {
        if (saved >> j & 1)
          removed.push_back(trial[j]);
        else
          hold.push_back(trial[j]);
      }
      trial = hold;
      auto here = cut_totals(trial);
      Word g = best_anchored_cut(pa, pb, here.need, here.covered_once, 2, x);
      trial.push_back({pa, pb, g, coverage(g)});
      shuffle(removed.begin(), removed.end(), rng);
    } else {
      if (qty > (int)trial.size())
        qty = trial.size();
      shuffle(trial.begin(), trial.end(), rng);
      removed.assign(trial.end() - qty, trial.end());
      trial.resize(trial.size() - qty);
    }

    for (auto p : removed)
      if (!add_pair(trial, p.a, p.b, 1 + draw(3)))
        break;
    while (add_pair(trial, -1, -1, 1)) {
    }
    VertexSet defects = cut_totals(trial).need;
    int energy = weighted_count(defects), miss = defects.count();
    if (miss < fewest_missed && (miss != 0 || (int)trial.size() >= min_dimension)) {
      fewest_missed = miss;
      best = trial;
      ofstream f("best.tmp");
      f << miss << " " << trial.size() << " " << (it + 1) << "\n";
      for (auto p : trial)
        f << p.a << " " << p.b << " " << p.g << "\n";
      f.close();
      if (rename("best.tmp", "best.txt") != 0)
        throw runtime_error("Cannot publish checkpoint");
      cout << "best " << miss << " " << trial.size() << " " << it << endl;
      if (!miss)
        return 0;
    }
    if (energy < local_cost)
      stalled_steps = 0;
    else
      stalled_steps++;
    if (energy <= local_cost ||
        (energy <= local_cost + (adaptive ? 3 : 1) &&
         draw(100) < (adaptive ? 7 : 5)) ||
        (energy <= local_cost + (adaptive ? 8 : 3) && draw(100) < 2)) {
      local = trial;
      local_cost = energy;
    }
  }
}
