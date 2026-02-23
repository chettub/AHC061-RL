#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ahc061/core/features.hpp"
#include "ahc061/core/generator.hpp"
#include "ahc061/core/opponent_ai.hpp"
#include "ahc061/core/pf.hpp"
#include "ahc061/core/rules.hpp"
#include "ahc061/core/score.hpp"
#include "ahc061/core/tools_input.hpp"

namespace ahc061::exp001 {

struct EnvInstance {
    State st{};
    int turn = 0;
    double phi = 0.0;
    std::array<std::int64_t, M_MAX> score{};

    bool tools_mode = false;
    std::array<OpponentParam, M_MAX> opponent_param_true{};
    std::vector<std::array<double, M_MAX>> tools_r1;
    std::vector<std::array<double, M_MAX>> tools_r2;

    XorShift64 rng{};
    std::array<ParticleFilterSMC, M_MAX> pf{};

    mutable bool cache_valid = false;
    mutable int cache_turn = -1;
    mutable std::array<std::array<int, CELL_MAX>, M_MAX> cache_moves{};
    mutable std::array<int, M_MAX> cache_move_cnt{};
    mutable std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> cache_comp{};
    mutable std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> cache_reach{};

    void reset_pf(std::uint64_t base_seed) {
        for (int p = 0; p < M_MAX; p++) {
            const std::uint64_t s = base_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL) ^
                                    0x243f6a8885a308d3ULL;
            pf[p].reset(s);
        }
    }

    void reset_random(std::uint64_t seed) {
        tools_mode = false;
        tools_r1.clear();
        tools_r2.clear();

        generate_random_case(seed, st, opponent_param_true);
        rng = XorShift64(seed ^ 0xdeadbeefcafebabeULL);

        turn = 0;
        score.fill(0);
        for (int i = 0; i < CELL_MAX; i++) {
            const int o = static_cast<int>(st.owner[i]);
            if (0 <= o && o < st.m) {
                score[o] += static_cast<std::int64_t>(st.value[i]) * static_cast<std::int64_t>(st.level[i]);
            }
        }
        std::int64_t sa = 0;
        for (int p = 1; p < st.m; p++)
            sa = std::max(sa, score[p]);
        phi = compute_phi(score[0], sa);

        const std::uint64_t pf_seed = compute_case_seed_for_pf(st) ^ (seed * 0x94d049bb133111ebULL);
        reset_pf(pf_seed);
        cache_valid = false;
        cache_turn = -1;
    }

    void reset_from_tools(const ToolsCase& tc, std::uint64_t pf_seed_extra = 0) {
        tools_mode = true;
        st = tc.st;
        opponent_param_true = tc.opponent_param;
        tools_r1 = tc.r1;
        tools_r2 = tc.r2;

        rng = XorShift64(1);
        turn = 0;
        score.fill(0);
        for (int i = 0; i < CELL_MAX; i++) {
            const int o = static_cast<int>(st.owner[i]);
            if (0 <= o && o < st.m) {
                score[o] += static_cast<std::int64_t>(st.value[i]) * static_cast<std::int64_t>(st.level[i]);
            }
        }
        std::int64_t sa = 0;
        for (int p = 1; p < st.m; p++)
            sa = std::max(sa, score[p]);
        phi = compute_phi(score[0], sa);

        const std::uint64_t pf_seed = compute_case_seed_for_pf(st) ^ pf_seed_extra;
        reset_pf(pf_seed);
        cache_valid = false;
        cache_turn = -1;
    }

    int current_pos0_cell() const {
        return cell_index(static_cast<int>(st.ex[0]), static_cast<int>(st.ey[0]));
    }

    bool is_legal_move0(int action_cell) const {
        if (action_cell < 0 || action_cell >= CELL_MAX)
            return false;
        std::array<int, CELL_MAX> moves{};
        std::array<std::uint8_t, CELL_MAX> reach{};
        enumerate_legal_moves(st, 0, moves, nullptr, &reach);
        return reach[action_cell] != 0;
    }

    std::pair<float, bool> step(int action_cell, bool pf_enabled) {
        if (turn >= st.t_max)
            return {0.0f, true};
        const bool cache_ok = cache_valid && cache_turn == turn;
        bool legal = false;
        if (cache_ok) {
            legal = (0 <= action_cell && action_cell < CELL_MAX && cache_reach[0][action_cell] != 0);
        } else {
            legal = is_legal_move0(action_cell);
        }
        if (!legal) {
            throw std::runtime_error("illegal action for player0: cell=" + std::to_string(action_cell));
        }

        const State st_start = st;

        std::array<int, M_MAX> move_to{};
        move_to.fill(0);
        std::array<int, M_MAX> tx_cell{};
        tx_cell.fill(0);

        move_to[0] = action_cell;
        tx_cell[0] = action_cell;

        for (int p = 1; p < st.m; p++) {
            double r1 = 0.0;
            double r2 = 0.0;
            if (tools_mode) {
                r1 = tools_r1[turn][p];
                r2 = tools_r2[turn][p];
            } else {
                r1 = rng.next_double01();
                r2 = rng.next_double01();
            }
            int dest = 0;
            if (cache_ok) {
                const int cnt = cache_move_cnt[p];
                dest = select_move_ai_like_from_moves(
                    st_start,
                    p,
                    opponent_param_true[p],
                    r1,
                    r2,
                    cache_moves[p].data(),
                    cnt);
            } else {
                dest = select_move_ai_like(st_start, p, opponent_param_true[p], r1, r2);
            }
            move_to[p] = dest;
            tx_cell[p] = dest;
        }

        apply_simultaneous_turn(st, move_to, &score);
        turn++;

        if (pf_enabled) {
            for (int p = 1; p < st.m; p++) {
                const MoveSummary sum = cache_ok ? summarize_ai_observation_from_moves(
                                                       st_start,
                                                       p,
                                                       tx_cell[p],
                                                       cache_moves[p].data(),
                                                       cache_move_cnt[p])
                                                 : summarize_ai_observation(st_start, p, tx_cell[p]);
                pf[p].update(sum);
            }
        }

        std::int64_t sa = 0;
        for (int p = 1; p < st.m; p++)
            sa = std::max(sa, score[p]);
        const double new_phi = compute_phi(score[0], sa);
        const float reward = static_cast<float>(new_phi - phi);
        phi = new_phi;
        cache_valid = false;
        cache_turn = -1;
        const bool done = (turn >= st.t_max);
        return {reward, done};
    }

    void observe_into(float* out_board, std::uint8_t* out_mask, bool pf_enabled) const {
        extract_features_into(
            st,
            turn,
            &pf,
            pf_enabled,
            out_board,
            out_mask,
            &cache_moves,
            &cache_move_cnt,
            &cache_comp,
            &cache_reach);
        cache_valid = true;
        cache_turn = turn;
    }

    std::pair<std::int64_t, std::int64_t> score_s0_sa() const {
        std::int64_t sa = 0;
        for (int p = 1; p < st.m; p++)
            sa = std::max(sa, score[p]);
        return {score[0], sa};
    }
    std::int64_t official_score() const {
        const auto [s0, sa] = score_s0_sa();
        return compute_official_score(s0, sa);
    }
};

}  // namespace ahc061::exp001
