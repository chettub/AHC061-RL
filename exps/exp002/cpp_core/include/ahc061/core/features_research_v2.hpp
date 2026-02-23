#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ahc061/core/feature_common.hpp"
#include "ahc061/core/features.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

static constexpr int FEATURE_RESEARCH_V2_C = FEATURE_C + 2 + 8 + 8 + 1;

static constexpr int CH_R2_POS0_X_NORM = FEATURE_C + 0;
static constexpr int CH_R2_POS0_Y_NORM = FEATURE_C + 1;
static constexpr int CH_R2_DIST_OWNER0 = FEATURE_C + 2;                  // +p (0..7)
static constexpr int CH_R2_DIST_COMP0 = CH_R2_DIST_OWNER0 + M_MAX;       // +p
static constexpr int CH_R2_DIST_CENTER = CH_R2_DIST_COMP0 + M_MAX;

static_assert(CH_R2_DIST_CENTER + 1 == FEATURE_RESEARCH_V2_C, "FEATURE_RESEARCH_V2_C mismatch");

inline void _compute_dist_to_sources_into(
    const std::array<std::uint8_t, CELL_MAX>& is_source,
    float* out_plane) {  // [CELL_MAX]
    // Exact Manhattan distance transform on a 4-neighbor grid.
    // Distances are equivalent to multi-source BFS here (no blocked cells),
    // but this 2-pass DP is lighter on branches.
    constexpr int INF = 1 << 20;
    std::array<int, CELL_MAX> dist{};
    bool has_source = false;
    for (int i = 0; i < CELL_MAX; i++) {
        if (is_source[static_cast<std::size_t>(i)]) {
            dist[static_cast<std::size_t>(i)] = 0;
            has_source = true;
        } else {
            dist[static_cast<std::size_t>(i)] = INF;
        }
    }

    if (!has_source) {
        for (int i = 0; i < CELL_MAX; i++)
            out_plane[i] = 1.0f;
        return;
    }

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            const int idx = cell_index(x, y);
            int d = dist[static_cast<std::size_t>(idx)];
            if (x > 0)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x - 1, y))] + 1);
            if (y > 0)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x, y - 1))] + 1);
            dist[static_cast<std::size_t>(idx)] = d;
        }
    }
    for (int x = N - 1; x >= 0; x--) {
        for (int y = N - 1; y >= 0; y--) {
            const int idx = cell_index(x, y);
            int d = dist[static_cast<std::size_t>(idx)];
            if (x + 1 < N)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x + 1, y))] + 1);
            if (y + 1 < N)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x, y + 1))] + 1);
            dist[static_cast<std::size_t>(idx)] = d;
        }
    }

    constexpr float INV_MAX_DIST = 1.0f / 18.0f;
    for (int i = 0; i < CELL_MAX; i++) {
        const int d = dist[static_cast<std::size_t>(i)];
        out_plane[i] = (d >= INF) ? 1.0f : (static_cast<float>(d) * INV_MAX_DIST);
    }
}

inline void write_features_research_v2_from_common(const FeatureCommon& common, float* out_board) {
    const State& st = *common.st;
    const int plane_size = CELL_MAX;

    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_RESEARCH_V2_C * plane_size) * sizeof(float));

    const float v_scale = 1.0f / 1000.0f;
    const float inv_u_fixed = 1.0f / 5.0f;

    // base planes (v/level/owner)
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_V_NORM * plane_size + idx] = static_cast<float>(st.value[idx]) * v_scale;
        out_board[CH_L_NORM * plane_size + idx] = static_cast<float>(st.level[idx]) * inv_u_fixed;

        const int o = static_cast<int>(st.owner[idx]);
        if (o == -1) {
            out_board[CH_NEUTRAL * plane_size + idx] = 1.0f;
        } else if (0 <= o && o < M_MAX) {
            const int np = common.old_to_new[static_cast<std::size_t>(o)];
            out_board[(CH_OWNER0 + np) * plane_size + idx] = 1.0f;
        }
    }

    const auto& comp = *common.comp;
    const auto& reach = *common.reach;

    // comp/reach planes (reordered)
    for (int old_p = 0; old_p < M_MAX; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        for (int idx = 0; idx < plane_size; idx++) {
            if (comp[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)])
                out_board[(CH_COMP0 + np) * plane_size + idx] = 1.0f;
            if (reach[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)])
                out_board[(CH_REACH0 + np) * plane_size + idx] = 1.0f;
        }
    }

    // next distributions (reordered)
    for (int old_p = 0; old_p < M_MAX; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        for (int idx = 0; idx < plane_size; idx++)
            out_board[(CH_NEXT0 + np) * plane_size + idx] =
                common.next[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)];
    }

    // score planes (constant per board, reordered) - scaled by 50000, no clip
    constexpr float INV_SCORE_SCALE = 1.0f / 50000.0f;
    for (int old_p = 0; old_p < st.m; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        const float v = static_cast<float>(common.score_raw[static_cast<std::size_t>(old_p)]) * INV_SCORE_SCALE;
        for (int idx = 0; idx < plane_size; idx++)
            out_board[(CH_SCORE0 + np) * plane_size + idx] = v;
    }

    // constant planes
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_TURN_FRAC * plane_size + idx] = common.turn_frac;
        out_board[CH_M_NORM * plane_size + idx] = common.m_norm;
        out_board[CH_U_NORM * plane_size + idx] = common.u_norm;
    }

    // research_v1 extras: pos0
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_R2_POS0_X_NORM * plane_size + idx] = common.pos0_x_norm;
        out_board[CH_R2_POS0_Y_NORM * plane_size + idx] = common.pos0_y_norm;
    }

    // dist_owner[p], dist_comp[p] (reordered)
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> owner_source{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> comp_source{};
    for (int p = 0; p < M_MAX; p++) {
        owner_source[static_cast<std::size_t>(p)].fill(0);
        comp_source[static_cast<std::size_t>(p)].fill(0);
    }

    for (int idx = 0; idx < plane_size; idx++) {
        const int o = static_cast<int>(st.owner[idx]);
        if (0 <= o && o < st.m) {
            const int np = common.old_to_new[static_cast<std::size_t>(o)];
            owner_source[static_cast<std::size_t>(np)][static_cast<std::size_t>(idx)] = 1;
        }
    }
    for (int old_p = 0; old_p < st.m; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        for (int idx = 0; idx < plane_size; idx++) {
            if (comp[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)])
                comp_source[static_cast<std::size_t>(np)][static_cast<std::size_t>(idx)] = 1;
        }
    }

    std::array<float, CELL_MAX> dist_plane{};
    for (int p = 0; p < M_MAX; p++) {
        _compute_dist_to_sources_into(owner_source[static_cast<std::size_t>(p)], dist_plane.data());
        for (int idx = 0; idx < plane_size; idx++)
            out_board[(CH_R2_DIST_OWNER0 + p) * plane_size + idx] = dist_plane[static_cast<std::size_t>(idx)];

        _compute_dist_to_sources_into(comp_source[static_cast<std::size_t>(p)], dist_plane.data());
        for (int idx = 0; idx < plane_size; idx++)
            out_board[(CH_R2_DIST_COMP0 + p) * plane_size + idx] = dist_plane[static_cast<std::size_t>(idx)];
    }

    // dist_center (abs(x-4.5)+abs(y-4.5))/9
    constexpr float CX = 4.5f;
    constexpr float CY = 4.5f;
    constexpr float INV_MAX_CENTER = 1.0f / 9.0f;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            const int idx = cell_index(x, y);
            const float dx = std::abs(static_cast<float>(x) - CX);
            const float dy = std::abs(static_cast<float>(y) - CY);
            out_board[CH_R2_DIST_CENTER * plane_size + idx] = (dx + dy) * INV_MAX_CENTER;
        }
    }
}

}  // namespace ahc061::exp002
