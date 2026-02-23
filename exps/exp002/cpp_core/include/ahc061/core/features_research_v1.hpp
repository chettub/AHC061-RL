#pragma once

#include <cstddef>
#include <cstring>

#include "ahc061/core/feature_common.hpp"
#include "ahc061/core/features.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

static constexpr int FEATURE_RESEARCH_V1_C = FEATURE_C + 2;

static constexpr int CH_POS0_X_NORM = FEATURE_C + 0;
static constexpr int CH_POS0_Y_NORM = FEATURE_C + 1;

static_assert(CH_POS0_Y_NORM + 1 == FEATURE_RESEARCH_V1_C, "FEATURE_RESEARCH_V1_C mismatch");

inline void write_features_research_v1_from_common(const FeatureCommon& common, float* out_board) {
    const int plane_size = CELL_MAX;
    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_RESEARCH_V1_C * plane_size) * sizeof(float));

    write_features_submit_v1_from_common(common, out_board);

    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_POS0_X_NORM * plane_size + idx] = common.pos0_x_norm;
        out_board[CH_POS0_Y_NORM * plane_size + idx] = common.pos0_y_norm;
    }
}

}  // namespace ahc061::exp002

