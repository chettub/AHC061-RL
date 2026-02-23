#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ahc061/core/feature_common.hpp"
#include "ahc061/core/features_research_v2.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

// research_v4 layout (149ch):
//  - global  : 19ch (0..18)
//  - players : 8 * 16ch = 128ch (19..146)
//  - pos0    : 2ch (147..148)
static constexpr int FEATURE_RESEARCH_V4_C = 149;

// global
static constexpr int CH_R4_V_NORM = 0;
static constexpr int CH_R4_L_NORM = 1;
static constexpr int CH_R4_NEUTRAL = 2;
static constexpr int CH_R4_TURN_FRAC = 3;
static constexpr int CH_R4_DIST_CENTER = 4;
static constexpr int CH_R4_X_NORM = 5;
static constexpr int CH_R4_Y_NORM = 6;
static constexpr int CH_R4_M2_ONEHOT = 7;   // +k (0..6) => m=2..8
static constexpr int CH_R4_U1_ONEHOT = 14;  // +k (0..4) => u_max=1..5

// player block
static constexpr int CH_R4_PLAYER_BLOCK0 = 19;
static constexpr int R4_PLAYER_STRIDE = 16;
static constexpr int OFF_R4_OWNER = 0;
static constexpr int OFF_R4_COMP = 1;
static constexpr int OFF_R4_REACH = 2;
static constexpr int OFF_R4_NEXT = 3;
static constexpr int OFF_R4_SCORE = 4;
static constexpr int OFF_R4_DIST_OWNER = 5;
static constexpr int OFF_R4_DIST_COMP = 6;
static constexpr int OFF_R4_OWNER_LEVEL_SUM = 7;
static constexpr int OFF_R4_OWNER_LEVEL_VALUE_SUM = 8;
static constexpr int OFF_R4_COMP_LEVEL_SUM = 9;
static constexpr int OFF_R4_COMP_LEVEL_VALUE_SUM = 10;
static constexpr int OFF_R4_EST_WA_NORM = 11;
static constexpr int OFF_R4_EST_WB_NORM = 12;
static constexpr int OFF_R4_EST_WC_NORM = 13;
static constexpr int OFF_R4_EST_WD_NORM = 14;
static constexpr int OFF_R4_EST_EPS = 15;

// pos0
static constexpr int CH_R4_POS0_X_NORM = CH_R4_PLAYER_BLOCK0 + M_MAX * R4_PLAYER_STRIDE;
static constexpr int CH_R4_POS0_Y_NORM = CH_R4_POS0_X_NORM + 1;

static_assert(CH_R4_U1_ONEHOT + 5 == CH_R4_PLAYER_BLOCK0, "research_v4 global layout mismatch");
static_assert(CH_R4_PLAYER_BLOCK0 + M_MAX * R4_PLAYER_STRIDE == CH_R4_POS0_X_NORM, "research_v4 player layout mismatch");
static_assert(CH_R4_POS0_Y_NORM + 1 == FEATURE_RESEARCH_V4_C, "FEATURE_RESEARCH_V4_C mismatch");

inline int _r4_player_ch(int np, int offset) { return CH_R4_PLAYER_BLOCK0 + np * R4_PLAYER_STRIDE + offset; }

inline void write_features_research_v4_from_common(const FeatureCommon& common, float* out_board) {
    const State& st = *common.st;
    constexpr int plane_size = CELL_MAX;
    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_RESEARCH_V4_C * plane_size) * sizeof(float));

    const float v_scale = 1.0f / 1000.0f;
    const float inv_u_fixed = 1.0f / 5.0f;

    // base planes: v/l/neutral + owner(np)
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_R4_V_NORM * plane_size + idx] = static_cast<float>(st.value[idx]) * v_scale;
        out_board[CH_R4_L_NORM * plane_size + idx] = static_cast<float>(st.level[idx]) * inv_u_fixed;

        const int o = static_cast<int>(st.owner[idx]);
        if (o == -1) {
            out_board[CH_R4_NEUTRAL * plane_size + idx] = 1.0f;
        } else if (0 <= o && o < M_MAX) {
            const int np = common.old_to_new[static_cast<std::size_t>(o)];
            out_board[_r4_player_ch(np, OFF_R4_OWNER) * plane_size + idx] = 1.0f;
        }
    }

    const auto& comp = *common.comp;
    const auto& reach = *common.reach;

    // comp/reach planes (reordered)
    for (int old_p = 0; old_p < M_MAX; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        for (int idx = 0; idx < plane_size; idx++) {
            if (comp[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)])
                out_board[_r4_player_ch(np, OFF_R4_COMP) * plane_size + idx] = 1.0f;
            if (reach[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)])
                out_board[_r4_player_ch(np, OFF_R4_REACH) * plane_size + idx] = 1.0f;
        }
    }

    // next distributions (reordered)
    for (int old_p = 0; old_p < M_MAX; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        for (int idx = 0; idx < plane_size; idx++) {
            out_board[_r4_player_ch(np, OFF_R4_NEXT) * plane_size + idx] =
                common.next[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)];
        }
    }

    // score planes (reordered, raw/50000, no clip)
    constexpr float INV_SCORE_SCALE = 1.0f / 50000.0f;
    for (int old_p = 0; old_p < st.m; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        const float v = static_cast<float>(common.score_raw[static_cast<std::size_t>(old_p)]) * INV_SCORE_SCALE;
        for (int idx = 0; idx < plane_size; idx++)
            out_board[_r4_player_ch(np, OFF_R4_SCORE) * plane_size + idx] = v;
    }

    // global constant planes
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_R4_TURN_FRAC * plane_size + idx] = common.turn_frac;
        out_board[CH_R4_POS0_X_NORM * plane_size + idx] = common.pos0_x_norm;
        out_board[CH_R4_POS0_Y_NORM * plane_size + idx] = common.pos0_y_norm;
    }

    const int m = st.m;
    if (2 <= m && m <= 8) {
        const int ch = CH_R4_M2_ONEHOT + (m - 2);
        for (int idx = 0; idx < plane_size; idx++)
            out_board[ch * plane_size + idx] = 1.0f;
    }
    const int u = st.u_max;
    if (1 <= u && u <= 5) {
        const int ch = CH_R4_U1_ONEHOT + (u - 1);
        for (int idx = 0; idx < plane_size; idx++)
            out_board[ch * plane_size + idx] = 1.0f;
    }

    // dist_owner[np], dist_comp[np]
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
    for (int np = 0; np < M_MAX; np++) {
        _compute_dist_to_sources_into(owner_source[static_cast<std::size_t>(np)], dist_plane.data());
        for (int idx = 0; idx < plane_size; idx++) {
            out_board[_r4_player_ch(np, OFF_R4_DIST_OWNER) * plane_size + idx] =
                dist_plane[static_cast<std::size_t>(idx)];
        }

        _compute_dist_to_sources_into(comp_source[static_cast<std::size_t>(np)], dist_plane.data());
        for (int idx = 0; idx < plane_size; idx++) {
            out_board[_r4_player_ch(np, OFF_R4_DIST_COMP) * plane_size + idx] =
                dist_plane[static_cast<std::size_t>(idx)];
        }
    }

    // Additional per-player aggregate stats (reordered, constant planes):
    // - owner_level_sum:         sum(level) over owned cells
    // - owner_level_value_sum:   sum(level*value) over owned cells
    // - comp_level_sum:          sum(level) over current-position connected owned component
    // - comp_level_value_sum:    sum(level*value) over that component
    std::array<double, M_MAX> owner_level_sum{};
    std::array<double, M_MAX> owner_level_value_sum{};
    std::array<double, M_MAX> comp_level_sum{};
    std::array<double, M_MAX> comp_level_value_sum{};
    owner_level_sum.fill(0.0);
    owner_level_value_sum.fill(0.0);
    comp_level_sum.fill(0.0);
    comp_level_value_sum.fill(0.0);

    for (int idx = 0; idx < plane_size; idx++) {
        const int old_o = static_cast<int>(st.owner[idx]);
        if (0 <= old_o && old_o < st.m) {
            const int np = common.old_to_new[static_cast<std::size_t>(old_o)];
            const double lv = static_cast<double>(st.level[idx]);
            const double lvv = lv * static_cast<double>(st.value[idx]);
            owner_level_sum[static_cast<std::size_t>(np)] += lv;
            owner_level_value_sum[static_cast<std::size_t>(np)] += lvv;
        }
    }

    for (int old_p = 0; old_p < st.m; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        double sum_l = 0.0;
        double sum_lv = 0.0;
        for (int idx = 0; idx < plane_size; idx++) {
            if (!comp[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)])
                continue;
            const double lv = static_cast<double>(st.level[idx]);
            sum_l += lv;
            sum_lv += lv * static_cast<double>(st.value[idx]);
        }
        comp_level_sum[static_cast<std::size_t>(np)] = sum_l;
        comp_level_value_sum[static_cast<std::size_t>(np)] = sum_lv;
    }

    constexpr double INV_LEVEL_SUM_SCALE = 1.0 / (100.0 * 5.0);       // align with level/5
    constexpr double INV_LEVEL_VALUE_SUM_SCALE = 1.0 / (100.0 * 5.0 * 1000.0);  // align with value/1000 * level/5
    for (int np = 0; np < M_MAX; np++) {
        const float owner_l = static_cast<float>(owner_level_sum[static_cast<std::size_t>(np)] * INV_LEVEL_SUM_SCALE);
        const float owner_lv =
            static_cast<float>(owner_level_value_sum[static_cast<std::size_t>(np)] * INV_LEVEL_VALUE_SUM_SCALE);
        const float comp_l = static_cast<float>(comp_level_sum[static_cast<std::size_t>(np)] * INV_LEVEL_SUM_SCALE);
        const float comp_lv =
            static_cast<float>(comp_level_value_sum[static_cast<std::size_t>(np)] * INV_LEVEL_VALUE_SUM_SCALE);
        for (int idx = 0; idx < plane_size; idx++) {
            out_board[_r4_player_ch(np, OFF_R4_OWNER_LEVEL_SUM) * plane_size + idx] = owner_l;
            out_board[_r4_player_ch(np, OFF_R4_OWNER_LEVEL_VALUE_SUM) * plane_size + idx] = owner_lv;
            out_board[_r4_player_ch(np, OFF_R4_COMP_LEVEL_SUM) * plane_size + idx] = comp_l;
            out_board[_r4_player_ch(np, OFF_R4_COMP_LEVEL_VALUE_SUM) * plane_size + idx] = comp_lv;
        }
    }

    // Estimated policy parameters from C++ estimator (reordered, constant planes):
    //   w_norm[4], eps
    if (common.adf_beta != nullptr) {
        for (int old_p = 0; old_p < st.m; old_p++) {
            const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
            const OpponentParam est = (*common.adf_beta)[static_cast<std::size_t>(old_p)].mean_param();
            double sumw = est.wa + est.wb + est.wc + est.wd;
            if (!(sumw > 0.0))
                sumw = 1.0;
            const float wa = static_cast<float>(est.wa / sumw);
            const float wb = static_cast<float>(est.wb / sumw);
            const float wc = static_cast<float>(est.wc / sumw);
            const float wd = static_cast<float>(est.wd / sumw);
            const float eps = static_cast<float>(est.eps);
            for (int idx = 0; idx < plane_size; idx++) {
                out_board[_r4_player_ch(np, OFF_R4_EST_WA_NORM) * plane_size + idx] = wa;
                out_board[_r4_player_ch(np, OFF_R4_EST_WB_NORM) * plane_size + idx] = wb;
                out_board[_r4_player_ch(np, OFF_R4_EST_WC_NORM) * plane_size + idx] = wc;
                out_board[_r4_player_ch(np, OFF_R4_EST_WD_NORM) * plane_size + idx] = wd;
                out_board[_r4_player_ch(np, OFF_R4_EST_EPS) * plane_size + idx] = eps;
            }
        }
    }

    // dist_center (abs(x-4.5)+abs(y-4.5))/9, x_norm, y_norm
    constexpr float CX = 4.5f;
    constexpr float CY = 4.5f;
    constexpr float INV_MAX_CENTER = 1.0f / 9.0f;
    const float inv_nm1 = (N > 1) ? (1.0f / static_cast<float>(N - 1)) : 0.0f;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            const int idx = cell_index(x, y);
            const float dx = std::abs(static_cast<float>(x) - CX);
            const float dy = std::abs(static_cast<float>(y) - CY);
            out_board[CH_R4_DIST_CENTER * plane_size + idx] = (dx + dy) * INV_MAX_CENTER;
            out_board[CH_R4_X_NORM * plane_size + idx] = static_cast<float>(x) * inv_nm1;
            out_board[CH_R4_Y_NORM * plane_size + idx] = static_cast<float>(y) * inv_nm1;
        }
    }
}

}  // namespace ahc061::exp002
