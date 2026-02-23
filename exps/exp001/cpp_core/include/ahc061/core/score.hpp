#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

#include "ahc061/core/state.hpp"

namespace ahc061::exp001 {

inline std::pair<std::int64_t, std::int64_t> compute_score_s0_sa(const State& st) {
    std::array<std::int64_t, M_MAX> s{};
    s.fill(0);
    for (int i = 0; i < CELL_MAX; i++) {
        const int o = static_cast<int>(st.owner[i]);
        if (0 <= o && o < st.m) {
            s[o] += static_cast<std::int64_t>(st.value[i]) * static_cast<std::int64_t>(st.level[i]);
        }
    }
    std::int64_t sa = 0;
    for (int p = 1; p < st.m; p++)
        sa = std::max(sa, s[p]);
    return {s[0], sa};
}

inline double compute_phi(std::int64_t s0, std::int64_t sa) {
    // phi = log(1 + s0 / (sa + eps))
    constexpr double EPS = 1e-9;
    const double denom = static_cast<double>(sa) + EPS;
    const double ratio = static_cast<double>(s0) / denom;
    return std::log1p(ratio);
}

inline double compute_phi(const State& st) {
    const auto [s0, sa] = compute_score_s0_sa(st);
    return compute_phi(s0, sa);
}

inline std::int64_t compute_official_score(std::int64_t s0, std::int64_t sa) {
    // round(1e5 * log2(1 + s0/sa))
    if (sa <= 0)
        return 100000;
    const double ratio = static_cast<double>(s0) / static_cast<double>(sa);
    const double v = 100000.0 * std::log2(1.0 + ratio);
    return static_cast<std::int64_t>(std::llround(v));
}

inline std::int64_t compute_official_score(const State& st) {
    const auto [s0, sa] = compute_score_s0_sa(st);
    return compute_official_score(s0, sa);
}

}  // namespace ahc061::exp001

