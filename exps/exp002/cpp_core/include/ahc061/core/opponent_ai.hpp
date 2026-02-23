#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

inline void compute_move_dist_ai_like_from_moves(
    const State& st,
    int p,
    const OpponentParam& param,
    const int* moves,
    int cnt,
    float* out_dist) {  // [CELL_MAX]
    std::fill(out_dist, out_dist + CELL_MAX, 0.0f);
    if (cnt <= 0)
        return;
    if (cnt == 1) {
        out_dist[moves[0]] = 1.0f;
        return;
    }

    double eps = param.eps;
    if (eps < 0.0)
        eps = 0.0;
    if (eps > 1.0)
        eps = 1.0;

    const double base = eps / static_cast<double>(cnt);
    for (int i = 0; i < cnt; i++) {
        const int idx = moves[i];
        out_dist[idx] = static_cast<float>(base);
    }

    double best_a = -1e100;
    std::array<int, CELL_MAX> best_move_idx{};
    int best_cnt = 0;
    for (int i = 0; i < cnt; i++) {
        const int idx = moves[i];
        const int o = static_cast<int>(st.owner[idx]);
        const int lv = static_cast<int>(st.level[idx]);

        double a = 0.0;
        if (o == -1) {
            a = static_cast<double>(st.value[idx]) * param.wa;
        } else if (o == p) {
            if (lv < st.u_max)
                a = static_cast<double>(st.value[idx]) * param.wb;
        } else {
            if (lv == 1)
                a = static_cast<double>(st.value[idx]) * param.wc;
            else
                a = static_cast<double>(st.value[idx]) * param.wd;
        }

        if (a > best_a + 1e-12) {
            best_a = a;
            best_cnt = 0;
            best_move_idx[best_cnt++] = idx;
        } else if (std::abs(a - best_a) <= 1e-12) {
            best_move_idx[best_cnt++] = idx;
        }
    }

    if (best_cnt <= 0) {
        const double prob = 1.0 / static_cast<double>(cnt);
        for (int i = 0; i < cnt; i++) {
            out_dist[moves[i]] = static_cast<float>(prob);
        }
        return;
    }

    const double add = (1.0 - eps) / static_cast<double>(best_cnt);
    for (int i = 0; i < best_cnt; i++) {
        const int idx = best_move_idx[i];
        out_dist[idx] += static_cast<float>(add);
    }

    double sum = 0.0;
    for (int i = 0; i < cnt; i++) {
        sum += static_cast<double>(out_dist[moves[i]]);
    }
    if (!(sum > 0.0)) {
        const double prob = 1.0 / static_cast<double>(cnt);
        for (int i = 0; i < cnt; i++) {
            out_dist[moves[i]] = static_cast<float>(prob);
        }
        return;
    }
    const double inv_sum = 1.0 / sum;
    for (int i = 0; i < cnt; i++) {
        const int idx = moves[i];
        out_dist[idx] = static_cast<float>(static_cast<double>(out_dist[idx]) * inv_sum);
    }
}

inline int select_move_ai_like_from_moves(
    const State& st,
    int p,
    const OpponentParam& param,
    double r1,
    double r2,
    const int* moves,
    int cnt) {
    if (cnt <= 1)
        return moves[0];

    auto pick_index = [&](int k) -> int {
        if (k <= 1)
            return 0;
        if (r2 < 0.0)
            r2 = 0.0;
        if (r2 >= 1.0)
            r2 = std::nextafter(1.0, 0.0);
        int idx = static_cast<int>(r2 * static_cast<double>(k));
        if (idx < 0)
            idx = 0;
        if (idx >= k)
            idx = k - 1;
        return idx;
    };

    if (r1 < param.eps) {
        const int idx = pick_index(cnt);
        return moves[idx];
    }

    double best_a = -1e100;
    std::array<int, CELL_MAX> best_ids{};
    int best_cnt = 0;
    for (int i = 0; i < cnt; i++) {
        const int idx = moves[i];
        const int o = static_cast<int>(st.owner[idx]);
        const int lv = static_cast<int>(st.level[idx]);

        double a = 0.0;
        if (o == -1) {
            a = static_cast<double>(st.value[idx]) * param.wa;
        } else if (o == p) {
            if (lv < st.u_max)
                a = static_cast<double>(st.value[idx]) * param.wb;
        } else {
            if (lv == 1)
                a = static_cast<double>(st.value[idx]) * param.wc;
            else
                a = static_cast<double>(st.value[idx]) * param.wd;
        }

        if (a > best_a + 1e-12) {
            best_a = a;
            best_cnt = 0;
            best_ids[best_cnt++] = i;
        } else if (std::abs(a - best_a) <= 1e-12) {
            best_ids[best_cnt++] = i;
        }
    }

    const int pick = best_ids[pick_index(best_cnt)];
    return moves[pick];
}

inline int select_move_ai_like(
    const State& st,
    int p,
    const OpponentParam& param,
    double r1,
    double r2) {
    std::array<int, CELL_MAX> moves{};
    const int cnt = enumerate_legal_moves(st, p, moves);
    return select_move_ai_like_from_moves(st, p, param, r1, r2, moves.data(), cnt);
}

}  // namespace ahc061::exp002
