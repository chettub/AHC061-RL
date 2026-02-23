#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

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
    double e = param.eps;
    if (e < 0.0)
        e = 0.0;
    if (e > 1.0)
        e = 1.0;
    double b = e / static_cast<double>(cnt);
    for (int i = 0; i < cnt; i++) {
        int idx = moves[i];
        out_dist[idx] = static_cast<float>(b);
    }
    double ba = -1e100;
    std::array<int, CELL_MAX> bi{};
    int bc = 0;
    for (int i = 0; i < cnt; i++) {
        int idx = moves[i];
        int o = static_cast<int>(st.owner[idx]);
        int lv = static_cast<int>(st.level[idx]);
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
        if (a > ba + 1e-12) {
            ba = a;
            bc = 0;
            bi[bc++] = idx;
        } else if (std::abs(a - ba) <= 1e-12) {
            bi[bc++] = idx;
        }
    }
    if (bc <= 0) {
        double p0 = 1.0 / static_cast<double>(cnt);
        for (int i = 0; i < cnt; i++) {
            out_dist[moves[i]] = static_cast<float>(p0);
        }
        return;
    }
    double ad = (1.0 - e) / static_cast<double>(bc);
    for (int i = 0; i < bc; i++) {
        int idx = bi[i];
        out_dist[idx] += static_cast<float>(ad);
    }
    double s = 0.0;
    for (int i = 0; i < cnt; i++) {
        s += static_cast<double>(out_dist[moves[i]]);
    }
    if (!(s > 0.0)) {
        double p0 = 1.0 / static_cast<double>(cnt);
        for (int i = 0; i < cnt; i++) {
            out_dist[moves[i]] = static_cast<float>(p0);
        }
        return;
    }
    double is = 1.0 / s;
    for (int i = 0; i < cnt; i++) {
        int idx = moves[i];
        out_dist[idx] = static_cast<float>(static_cast<double>(out_dist[idx]) * is);
    }
}

}  // namespace ahc061::exp002
