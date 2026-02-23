#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

inline void generate_values_problem_distribution(XorShift64& rng, std::array<int, CELL_MAX>& out_value) {
    std::array<double, CELL_MAX> v{};

    const double a = rng.next_double01() * 3.0;  // [0, 3)
    for (int i = 0; i < CELL_MAX; i++) {
        const double x = 0.5 + 0.5 * rng.next_double01();  // [0.5, 1.0)
        v[i] = std::pow(x, a);
    }

    const int k = rng.next_int(0, 2);
    for (int it = 0; it < k; it++) {
        const int cx = rng.next_int(0, N - 1);
        const int cy = rng.next_int(0, N - 1);
        const double b = 1.0 + 3.0 * rng.next_double01();  // [1,4)
        const int mode = rng.next_int(0, 4);
        const double r = 1.0 + 4.0 * rng.next_double01();  // [1,5)
        const double r2 = r * r;

        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                const int dx = x - cx;
                const int dy = y - cy;
                const double d2 = static_cast<double>(dx * dx + dy * dy);
                const double man = static_cast<double>(std::abs(dx) + std::abs(dy));
                double add = 0.0;
                if (mode == 0) {
                    add = b * std::exp(-d2 / (2.0 * r2));
                } else if (mode == 1) {
                    add = b / (1.0 + std::sqrt(d2) / r);
                } else if (mode == 2) {
                    if (d2 <= r2 + 1e-12)
                        add = b / 4.0;
                } else if (mode == 3) {
                    add = b / (1.0 + man / r);
                } else {
                    if (man <= r + 1e-12)
                        add = b / 4.0;
                }
                v[cell_index(x, y)] += add;
            }
        }
    }

    double s = 0.0;
    for (int i = 0; i < CELL_MAX; i++)
        s += v[i];
    if (!(s > 0.0))
        s = 1.0;

    constexpr int TARGET_SUM = 1000 * N * N;  // 100000
    int sum_int = 0;
    for (int i = 0; i < CELL_MAX; i++) {
        const double scaled = v[i] * static_cast<double>(TARGET_SUM) / s;
        const int vi = static_cast<int>(std::ceil(scaled - 1e-12));
        out_value[i] = std::max(1, vi);
        sum_int += out_value[i];
    }

    // Reduce until sum == TARGET_SUM.
    while (sum_int > TARGET_SUM) {
        std::array<int, CELL_MAX> cand{};
        int cand_cnt = 0;
        for (int i = 0; i < CELL_MAX; i++) {
            if (out_value[i] >= 2)
                cand[cand_cnt++] = i;
        }
        if (cand_cnt == 0)
            break;
        const int pick = cand[rng.next_int(0, cand_cnt - 1)];
        out_value[pick]--;
        sum_int--;
    }
}

inline void generate_random_case(
    std::uint64_t seed,
    State& out_state,
    std::array<OpponentParam, M_MAX>& out_opponent_param) {
    XorShift64 rng(seed);

    const int m = rng.next_int(2, M_MAX);
    const int u_max = rng.next_int(1, 5);
    const int t_max = 100;

    out_state.m = m;
    out_state.u_max = u_max;
    out_state.t_max = t_max;

    generate_values_problem_distribution(rng, out_state.value);

    // choose distinct start positions uniformly
    std::array<int, CELL_MAX> cells{};
    for (int i = 0; i < CELL_MAX; i++)
        cells[i] = i;
    for (int i = CELL_MAX - 1; i >= 1; i--) {
        const int j = rng.next_int(0, i);
        std::swap(cells[i], cells[j]);
    }

    out_state.owner.fill(-1);
    out_state.level.fill(0);
    for (int p = 0; p < m; p++) {
        const int idx = cells[p];
        const int x = idx / N;
        const int y = idx % N;
        out_state.ex[p] = static_cast<std::uint8_t>(x);
        out_state.ey[p] = static_cast<std::uint8_t>(y);
        out_state.owner[idx] = static_cast<std::int8_t>(p);
        out_state.level[idx] = 1;
    }

    out_opponent_param.fill(OpponentParam{});
    for (int p = 1; p < m; p++) {
        auto randdouble = [&](double lo, double hi) {
            return lo + (hi - lo) * rng.next_double01();
        };
        out_opponent_param[p].wa = randdouble(0.3, 1.0);
        out_opponent_param[p].wb = randdouble(0.3, 1.0);
        out_opponent_param[p].wc = randdouble(0.3, 1.0);
        out_opponent_param[p].wd = randdouble(0.3, 1.0);
        out_opponent_param[p].eps = randdouble(0.1, 0.5);
    }
}

inline std::uint64_t compute_case_seed_for_pf(const State& st) {
    std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
    seed ^= static_cast<std::uint64_t>(st.m) * 10007ULL;
    seed ^= static_cast<std::uint64_t>(st.t_max) * 101ULL;
    seed ^= static_cast<std::uint64_t>(st.u_max) * 1009ULL;
    for (int i = 0; i < CELL_MAX; i++)
        seed ^= static_cast<std::uint64_t>(st.value[i]) * 1315423911ULL;
    for (int p = 0; p < st.m; p++)
        seed ^= static_cast<std::uint64_t>(st.ex[p] * 31 + st.ey[p]) * 2654435761ULL;
    return seed;
}

}  // namespace ahc061::exp002
