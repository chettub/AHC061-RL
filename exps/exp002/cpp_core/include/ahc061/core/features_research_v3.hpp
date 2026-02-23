#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ahc061/core/features_research_v2.hpp"

namespace ahc061::exp002 {

// research_v3:
// - same as research_v2 for spatial features
// - next[p] uses A-softmax(tau) UT expectation (computed in FeatureCommon)
// - m/u_max are provided as one-hot (global planes); scalar m_norm/u_norm are zeroed

static constexpr int FEATURE_RESEARCH_V3_C = FEATURE_RESEARCH_V2_C + 7 + 5;

static constexpr int CH_R3_M2_ONEHOT = FEATURE_RESEARCH_V2_C + 0;  // +k (0..6) => m=2..8
static constexpr int CH_R3_U1_ONEHOT = CH_R3_M2_ONEHOT + 7;        // +k (0..4) => u_max=1..5

static_assert(CH_R3_U1_ONEHOT + 5 == FEATURE_RESEARCH_V3_C, "FEATURE_RESEARCH_V3_C mismatch");

inline void write_features_research_v3_from_common(const FeatureCommon& common, float* out_board) {
    const State& st = *common.st;
    constexpr int plane_size = CELL_MAX;

    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_RESEARCH_V3_C * plane_size) * sizeof(float));

    // Fill research_v2 part (writes up to FEATURE_RESEARCH_V2_C channels).
    write_features_research_v2_from_common(common, out_board);

    // Replace scalar m/u_max planes with one-hot (keep turn_frac as-is).
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_M_NORM * plane_size + idx] = 0.0f;
        out_board[CH_U_NORM * plane_size + idx] = 0.0f;
    }

    const int m = st.m;
    if (2 <= m && m <= 8) {
        const int ch = CH_R3_M2_ONEHOT + (m - 2);
        for (int idx = 0; idx < plane_size; idx++)
            out_board[ch * plane_size + idx] = 1.0f;
    }

    const int u = st.u_max;
    if (1 <= u && u <= 5) {
        const int ch = CH_R3_U1_ONEHOT + (u - 1);
        for (int idx = 0; idx < plane_size; idx++)
            out_board[ch * plane_size + idx] = 1.0f;
    }
}

}  // namespace ahc061::exp002

