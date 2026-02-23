#pragma once

#include <array>

#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

struct MoveSummary {
    int n = 0;                  // |B|
    bool all_zero = false;      // greedy が一様になる例外ケース
    std::array<int, 4> vmax{};  // 各カテゴリの V 最大（無ければ 0）
    std::array<int, 4> cnt{};   // vmax を達成する手数
    bool action_in_b = false;   // 観測行動が B に含まれるか
    int action_cat = -2;        // 0..3, -1: 自領土(L=U), -2: 不明/非合法
    int action_value = 0;       // V_a
};

inline int categorize_move_for_ai(const State& st, int p, int idx) {
    int o = static_cast<int>(st.owner[idx]);
    if (o == -1)
        return 0;
    if (o == p)
        return (static_cast<int>(st.level[idx]) < st.u_max) ? 1 : -1;
    return (static_cast<int>(st.level[idx]) == 1) ? 2 : 3;
}

inline MoveSummary summarize_ai_observation_from_moves(
    const State& st_start,
    int p,
    int action_cell,
    const int* moves,
    int cnt) {
    MoveSummary s{};
    s.vmax.fill(0);
    s.cnt.fill(0);
    s.n = cnt;
    bool nz = false;
    for (int i = 0; i < cnt; i++) {
        int idx = moves[i];
        int c = categorize_move_for_ai(st_start, p, idx);
        if (c >= 0) {
            nz = true;
            int v = st_start.value[idx];
            if (v > s.vmax[c]) {
                s.vmax[c] = v;
                s.cnt[c] = 1;
            } else if (v == s.vmax[c]) {
                s.cnt[c]++;
            }
        }
        if (idx == action_cell) {
            s.action_in_b = true;
            s.action_cat = c;
            s.action_value = st_start.value[idx];
        }
    }
    s.all_zero = !nz;
    if (!s.action_in_b) {
        s.action_cat = -2;
        s.action_value = 0;
    }
    return s;
}

}  // namespace ahc061::exp002
