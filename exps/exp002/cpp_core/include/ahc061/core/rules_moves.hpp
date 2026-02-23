#pragma once

#include <array>
#include <cstdint>

#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

inline void compute_component_mask(const State& st, int p, std::array<std::uint8_t, CELL_MAX>& comp) {
    comp.fill(0);
    int st0 = cell_index(static_cast<int>(st.ex[p]), static_cast<int>(st.ey[p]));
    comp[st0] = 1;
    std::array<int, CELL_MAX> q{};
    int h = 0, t = 0;
    q[t++] = st0;
    while (h < t) {
        int v = q[h++];
        for (int dir = 0; dir < 4; dir++) {
            int ni = NEIGH_RDLU[v][dir];
            if (ni < 0)
                continue;
            if (comp[ni])
                continue;
            if (st.owner[ni] != p)
                continue;
            comp[ni] = 1;
            q[t++] = ni;
        }
    }
}

inline int enumerate_legal_moves(
    const State& st,
    int p,
    std::array<int, CELL_MAX>& out_moves,
    std::array<std::uint8_t, CELL_MAX>* out_comp_mask = nullptr,
    std::array<std::uint8_t, CELL_MAX>* out_reach_mask = nullptr) {
    int st0 = cell_index(static_cast<int>(st.ex[p]), static_cast<int>(st.ey[p]));
    std::array<std::uint8_t, CELL_MAX> comp{};
    comp.fill(0);
    comp[st0] = 1;
    if (st.owner[st0] != p) {
        compute_component_mask(st, p, comp);
    }
    std::array<std::uint8_t, CELL_MAX> occupied_by_other{};
    occupied_by_other.fill(0);
    for (int i = 0; i < st.m; i++) {
        if (i == p)
            continue;
        int idx = cell_index(static_cast<int>(st.ex[i]), static_cast<int>(st.ey[i]));
        occupied_by_other[idx] = 1;
    }
    std::array<std::uint8_t, CELL_MAX> visited{};
    visited.fill(0);
    std::array<int, CELL_MAX> q{};
    int h = 0, t = 0;
    visited[st0] = 1;
    q[t++] = st0;
    std::array<std::uint8_t, CELL_MAX> reach{};
    reach.fill(0);
    int c = 0;
    while (h < t) {
        int idx = q[h++];
        if (st.owner[idx] == p)
            comp[idx] = 1;
        if (!occupied_by_other[idx]) {
            reach[idx] = 1;
            out_moves[c++] = idx;
        }
        if (st.owner[idx] != p)
            continue;
        for (int dir = 0; dir < 4; dir++) {
            int ni = NEIGH_RDLU[idx][dir];
            if (ni < 0)
                continue;
            if (visited[ni])
                continue;
            visited[ni] = 1;
            q[t++] = ni;
        }
    }
    if (c == 0) {
        reach.fill(0);
        reach[st0] = 1;
        out_moves[c++] = st0;
    }
    if (out_reach_mask)
        *out_reach_mask = reach;
    if (out_comp_mask)
        *out_comp_mask = comp;
    return c;
}

}  // namespace ahc061::exp002
