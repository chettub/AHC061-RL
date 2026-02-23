#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ahc061/core/state.hpp"

namespace ahc061::exp001 {

struct ToolsCase {
    State st{};
    std::array<OpponentParam, M_MAX> opponent_param{};
    std::vector<std::array<double, M_MAX>> r1;  // [t][p] (p=0 unused)
    std::vector<std::array<double, M_MAX>> r2;  // [t][p]
};

inline ToolsCase load_tools_case(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs)
        throw std::runtime_error("failed to open tools input: " + path);

    ToolsCase tc{};
    int n = 0;
    int m = 0;
    int t_max = 0;
    int u_max = 0;
    if (!(ifs >> n >> m >> t_max >> u_max))
        throw std::runtime_error("failed to read header from tools input: " + path);
    if (n != N)
        throw std::runtime_error("unsupported N (expected 10): " + std::to_string(n));
    if (m < 2 || m > M_MAX)
        throw std::runtime_error("invalid M: " + std::to_string(m));
    if (t_max <= 0)
        throw std::runtime_error("invalid T: " + std::to_string(t_max));
    if (u_max <= 0)
        throw std::runtime_error("invalid U: " + std::to_string(u_max));

    tc.st.m = m;
    tc.st.t_max = t_max;
    tc.st.u_max = u_max;

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            int v = 0;
            if (!(ifs >> v))
                throw std::runtime_error("failed to read V from tools input: " + path);
            tc.st.value[cell_index(x, y)] = v;
        }
    }

    std::vector<int> sx(m), sy(m);
    for (int p = 0; p < m; p++) {
        if (!(ifs >> sx[p] >> sy[p]))
            throw std::runtime_error("failed to read start pos from tools input: " + path);
        tc.st.ex[p] = static_cast<std::uint8_t>(sx[p]);
        tc.st.ey[p] = static_cast<std::uint8_t>(sy[p]);
    }

    tc.st.owner.fill(-1);
    tc.st.level.fill(0);
    for (int p = 0; p < m; p++) {
        const int idx = cell_index(sx[p], sy[p]);
        tc.st.owner[idx] = static_cast<std::int8_t>(p);
        tc.st.level[idx] = 1;
    }

    // AI params for p=1..m-1 (p=0 is unused).
    tc.opponent_param.fill(OpponentParam{});
    for (int p = 1; p < m; p++) {
        double wa = 0, wb = 0, wc = 0, wd = 0, eps = 0;
        if (!(ifs >> wa >> wb >> wc >> wd >> eps))
            throw std::runtime_error("failed to read ai params from tools input: " + path);
        tc.opponent_param[p].wa = wa;
        tc.opponent_param[p].wb = wb;
        tc.opponent_param[p].wc = wc;
        tc.opponent_param[p].wd = wd;
        tc.opponent_param[p].eps = eps;
    }

    tc.r1.assign(t_max, {});
    tc.r2.assign(t_max, {});
    for (int t = 0; t < t_max; t++) {
        tc.r1[t].fill(0.0);
        tc.r2[t].fill(0.0);
        for (int p = 1; p < m; p++) {
            double a = 0.0;
            double b = 0.0;
            if (!(ifs >> a >> b))
                throw std::runtime_error("failed to read r1/r2 from tools input: " + path);
            tc.r1[t][p] = a;
            tc.r2[t][p] = b;
        }
    }

    return tc;
}

}  // namespace ahc061::exp001

