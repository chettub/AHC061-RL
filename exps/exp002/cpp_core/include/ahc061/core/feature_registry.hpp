#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ahc061/core/feature_common.hpp"
#include "ahc061/core/features.hpp"
#include "ahc061/core/features_research_v1.hpp"
#include "ahc061/core/features_research_v2.hpp"
#include "ahc061/core/features_research_v3.hpp"
#include "ahc061/core/features_research_v4.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

struct FeatureSet {
    const char* id;
    int channels;
    bool submit_supported;
    NextMode next_mode;
    void (*write_from_common)(const FeatureCommon&, float* out_board);
};

inline constexpr std::array<FeatureSet, 5> FEATURE_SETS = {
    FeatureSet{
        .id = "submit_v1",
        .channels = FEATURE_C,
        .submit_supported = true,
        .next_mode = NextMode::k_uniform_or_pf,
        .write_from_common = &write_features_submit_v1_from_common,
    },
    FeatureSet{
        .id = "research_v1",
        .channels = FEATURE_RESEARCH_V1_C,
        .submit_supported = false,
        .next_mode = NextMode::k_uniform_or_pf,
        .write_from_common = &write_features_research_v1_from_common,
    },
    FeatureSet{
        .id = "research_v2",
        .channels = FEATURE_RESEARCH_V2_C,
        .submit_supported = false,
        .next_mode = NextMode::k_uniform_or_pf,
        .write_from_common = &write_features_research_v2_from_common,
    },
    FeatureSet{
        .id = "research_v3",
        .channels = FEATURE_RESEARCH_V3_C,
        .submit_supported = false,
        .next_mode = NextMode::k_a_softmax_ut,
        .write_from_common = &write_features_research_v3_from_common,
    },
    FeatureSet{
        .id = "research_v4",
        .channels = FEATURE_RESEARCH_V4_C,
        .submit_supported = false,
        .next_mode = NextMode::k_adf_beta,
        .write_from_common = &write_features_research_v4_from_common,
    },
};

inline const FeatureSet& get_feature_set(std::string_view feature_id) {
    for (const auto& fs : FEATURE_SETS) {
        if (feature_id == fs.id)
            return fs;
    }
    throw std::runtime_error("unknown feature_id: " + std::string(feature_id));
}

inline int feature_channels(std::string_view feature_id) { return get_feature_set(feature_id).channels; }

inline bool feature_submit_supported(std::string_view feature_id) { return get_feature_set(feature_id).submit_supported; }

inline std::vector<std::string> feature_ids() {
    std::vector<std::string> out;
    out.reserve(FEATURE_SETS.size());
    for (const auto& fs : FEATURE_SETS)
        out.emplace_back(fs.id);
    return out;
}

}  // namespace ahc061::exp002
