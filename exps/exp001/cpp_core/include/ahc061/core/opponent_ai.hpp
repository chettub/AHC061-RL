#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp001 {

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

}  // namespace ahc061::exp001
