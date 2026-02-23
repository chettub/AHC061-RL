#pragma once

#include <array>
#include <cstdint>

#include "ahc061/core/state.hpp"

namespace ahc061::exp001 {

inline void compute_component_mask(const State& st, int p, std::array<std::uint8_t, CELL_MAX>& comp) {
    comp.fill(0);
    const int start = cell_index(static_cast<int>(st.ex[p]), static_cast<int>(st.ey[p]));
    comp[start] = 1;

    std::array<int, CELL_MAX> q{};
    int head = 0;
    int tail = 0;
    q[tail++] = start;
    while (head < tail) {
        const int v = q[head++];
        for (int dir = 0; dir < 4; dir++) {
            const int ni = NEIGH_RDLU[v][dir];
            if (ni < 0)
                continue;
            if (comp[ni])
                continue;
            if (st.owner[ni] != p)
                continue;
            comp[ni] = 1;
            q[tail++] = ni;
        }
    }
}

inline int enumerate_legal_moves(
    const State& st,
    int p,
    std::array<int, CELL_MAX>& out_moves,
    std::array<std::uint8_t, CELL_MAX>* out_comp_mask = nullptr,
    std::array<std::uint8_t, CELL_MAX>* out_reach_mask = nullptr) {
    // Match official tool ordering: BFS that expands only from own territory,
    // pushing neighbors in (R, D, L, U) order, and outputs only non-occupied cells.
    const int start = cell_index(static_cast<int>(st.ex[p]), static_cast<int>(st.ey[p]));

    std::array<std::uint8_t, CELL_MAX> comp{};
    comp.fill(0);
    comp[start] = 1;
    if (st.owner[start] != p) {
        compute_component_mask(st, p, comp);
    }

    std::array<std::uint8_t, CELL_MAX> occupied_by_other{};
    occupied_by_other.fill(0);
    for (int i = 0; i < st.m; i++) {
        if (i == p)
            continue;
        const int idx = cell_index(static_cast<int>(st.ex[i]), static_cast<int>(st.ey[i]));
        occupied_by_other[idx] = 1;
    }

    std::array<std::uint8_t, CELL_MAX> visited{};
    visited.fill(0);
    std::array<int, CELL_MAX> q{};
    int head = 0;
    int tail = 0;

    visited[start] = 1;
    q[tail++] = start;

    std::array<std::uint8_t, CELL_MAX> reach{};
    reach.fill(0);

    int cnt = 0;
    while (head < tail) {
        const int idx = q[head++];

        if (st.owner[idx] == p)
            comp[idx] = 1;

        if (!occupied_by_other[idx]) {
            reach[idx] = 1;
            out_moves[cnt++] = idx;
        }

        if (st.owner[idx] != p)
            continue;

        for (int dir = 0; dir < 4; dir++) {
            const int ni = NEIGH_RDLU[idx][dir];
            if (ni < 0)
                continue;
            if (visited[ni])
                continue;
            visited[ni] = 1;
            q[tail++] = ni;
        }
    }

    if (cnt == 0) {
        reach.fill(0);
        reach[start] = 1;
        out_moves[cnt++] = start;
    }

    if (out_reach_mask)
        *out_reach_mask = reach;
    if (out_comp_mask)
        *out_comp_mask = comp;
    return cnt;
}

inline void apply_simultaneous_turn(
    State& st,
    const std::array<int, M_MAX>& move_to,
    std::array<std::int64_t, M_MAX>* score = nullptr) {
    std::array<std::uint8_t, M_MAX> ox{};
    std::array<std::uint8_t, M_MAX> oy{};
    for (int p = 0; p < st.m; p++) {
        ox[p] = st.ex[p];
        oy[p] = st.ey[p];
    }

    std::array<int, CELL_MAX> cnt{};
    cnt.fill(0);
    std::array<std::array<int, M_MAX>, CELL_MAX> plist{};

    for (int p = 0; p < st.m; p++) {
        const int idx = move_to[p];
        const int x = idx / N;
        const int y = idx % N;
        st.ex[p] = static_cast<std::uint8_t>(x);
        st.ey[p] = static_cast<std::uint8_t>(y);
        plist[idx][cnt[idx]++] = p;
    }

    std::array<std::uint8_t, M_MAX> alive{};
    alive.fill(1);

    // Collision resolution
    for (int i = 0; i < CELL_MAX; i++) {
        if (cnt[i] <= 1)
            continue;
        const int o = static_cast<int>(st.owner[i]);
        int owner_piece = -1;
        if (o != -1) {
            for (int k = 0; k < cnt[i]; k++) {
                if (plist[i][k] == o) {
                    owner_piece = o;
                    break;
                }
            }
        }
        if (owner_piece != -1) {
            for (int k = 0; k < cnt[i]; k++) {
                const int p = plist[i][k];
                if (p == owner_piece)
                    continue;
                alive[p] = 0;
            }
        } else {
            for (int k = 0; k < cnt[i]; k++)
                alive[plist[i][k]] = 0;
        }
    }

    // Territory update / attack
    for (int p = 0; p < st.m; p++) {
        if (!alive[p])
            continue;
        const int idx = cell_index(static_cast<int>(st.ex[p]), static_cast<int>(st.ey[p]));
        const int old_o = static_cast<int>(st.owner[idx]);
        const int old_l = static_cast<int>(st.level[idx]);
        const int o = static_cast<int>(st.owner[idx]);
        if (o == -1) {
            st.owner[idx] = static_cast<std::int8_t>(p);
            st.level[idx] = 1;
        } else if (o == p) {
            if (static_cast<int>(st.level[idx]) < st.u_max)
                st.level[idx]++;
        } else {
            // attack
            st.level[idx]--;
            if (st.level[idx] == 0) {
                st.owner[idx] = static_cast<std::int8_t>(p);
                st.level[idx] = 1;
            } else {
                alive[p] = 0;
            }
        }

        if (score != nullptr) {
            const int new_o = static_cast<int>(st.owner[idx]);
            const int new_l = static_cast<int>(st.level[idx]);
            if (old_o != new_o || old_l != new_l) {
                const std::int64_t v = static_cast<std::int64_t>(st.value[idx]);
                if (0 <= old_o && old_o < st.m)
                    (*score)[old_o] -= v * static_cast<std::int64_t>(old_l);
                if (0 <= new_o && new_o < st.m)
                    (*score)[new_o] += v * static_cast<std::int64_t>(new_l);
            }
        }
    }

    // Restore collected pieces
    for (int p = 0; p < st.m; p++) {
        if (alive[p])
            continue;
        st.ex[p] = ox[p];
        st.ey[p] = oy[p];
    }
}

}  // namespace ahc061::exp001
