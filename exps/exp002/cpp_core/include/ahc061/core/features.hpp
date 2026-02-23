#pragma once

#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "ahc061/core/feature_common.hpp"
#include "ahc061/core/pf.hpp"
#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

static constexpr int FEATURE_C = 46;

static constexpr int CH_V_NORM = 0;
static constexpr int CH_L_NORM = 1;
static constexpr int CH_NEUTRAL = 2;
static constexpr int CH_OWNER0 = 3;  // +p (0..7)
static constexpr int CH_COMP0 = CH_OWNER0 + M_MAX;   // +p
static constexpr int CH_REACH0 = CH_COMP0 + M_MAX;   // +p
static constexpr int CH_NEXT0 = CH_REACH0 + M_MAX;   // +p
static constexpr int CH_SCORE0 = CH_NEXT0 + M_MAX;    // +p (0..7), normalized by 500000
static constexpr int CH_TURN_FRAC = CH_SCORE0 + M_MAX;
static constexpr int CH_M_NORM = CH_TURN_FRAC + 1;
static constexpr int CH_U_NORM = CH_M_NORM + 1;

static_assert(CH_U_NORM + 1 == FEATURE_C, "FEATURE_C mismatch");

inline int feature_channels() { return FEATURE_C; }

inline void write_features_submit_v1_from_common(const FeatureCommon& common, float* out_board) {
    const State& st = *common.st;
    const int plane_size = CELL_MAX;

    // zero init
    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_C * plane_size) * sizeof(float));

    const float v_scale = 1.0f / 1000.0f;
    const float inv_u = (st.u_max > 0) ? (1.0f / static_cast<float>(st.u_max)) : 0.0f;

    // base planes (v/level/owner)
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_V_NORM * plane_size + idx] = static_cast<float>(st.value[idx]) * v_scale;
        out_board[CH_L_NORM * plane_size + idx] = static_cast<float>(st.level[idx]) * inv_u;

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
            out_board[(CH_NEXT0 + np) * plane_size + idx] = common.next[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)];
    }

    // score planes (constant per board, reordered)
    for (int old_p = 0; old_p < st.m; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        const float v = common.score_plane[static_cast<std::size_t>(old_p)];
        for (int idx = 0; idx < plane_size; idx++)
            out_board[(CH_SCORE0 + np) * plane_size + idx] = v;
    }

    // constant planes
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_TURN_FRAC * plane_size + idx] = common.turn_frac;
        out_board[CH_M_NORM * plane_size + idx] = common.m_norm;
        out_board[CH_U_NORM * plane_size + idx] = common.u_norm;
    }
}

inline void extract_features_into(
    const State& st,
    int turn,
    const std::array<ParticleFilterSMC, M_MAX>* pf,
    bool pf_enabled,
    float* out_board,                // [C][100]
    std::uint8_t* out_action_mask,    // [100], for player0
    std::array<std::array<int, CELL_MAX>, M_MAX>* out_moves = nullptr,
    std::array<int, M_MAX>* out_move_cnt = nullptr,
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>* out_comp = nullptr,
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>* out_reach = nullptr) {
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> comp_local{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> reach_local{};
    std::array<std::array<int, CELL_MAX>, M_MAX> moves_local{};
    std::array<int, M_MAX> move_cnt_local{};

    auto& comp = out_comp ? *out_comp : comp_local;
    auto& reach = out_reach ? *out_reach : reach_local;
    auto& moves = out_moves ? *out_moves : moves_local;
    auto& move_cnt = out_move_cnt ? *out_move_cnt : move_cnt_local;

    FeatureCommon common{};
    compute_feature_common_into(st, turn, pf, pf_enabled, common, out_action_mask, moves, move_cnt, comp, reach);
    write_features_submit_v1_from_common(common, out_board);
}

}  // namespace ahc061::exp002
