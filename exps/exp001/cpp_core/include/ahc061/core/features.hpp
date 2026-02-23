#pragma once

#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "ahc061/core/pf.hpp"
#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp001 {

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
    const int plane_size = CELL_MAX;

    // zero init
    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_C * plane_size) * sizeof(float));
    std::memset(out_action_mask, 0, static_cast<std::size_t>(plane_size) * sizeof(std::uint8_t));

    const float v_scale = 1.0f / 1000.0f;
    const float inv_u = (st.u_max > 0) ? (1.0f / static_cast<float>(st.u_max)) : 0.0f;

    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_V_NORM * plane_size + idx] = static_cast<float>(st.value[idx]) * v_scale;
        out_board[CH_L_NORM * plane_size + idx] = static_cast<float>(st.level[idx]) * inv_u;

        const int o = static_cast<int>(st.owner[idx]);
        if (o == -1) {
            out_board[CH_NEUTRAL * plane_size + idx] = 1.0f;
        } else if (0 <= o && o < M_MAX) {
            out_board[(CH_OWNER0 + o) * plane_size + idx] = 1.0f;
        }
    }

    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> comp_local{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> reach_local{};
    std::array<std::array<int, CELL_MAX>, M_MAX> moves_local{};
    std::array<int, M_MAX> move_cnt_local{};

    auto& comp = out_comp ? *out_comp : comp_local;
    auto& reach = out_reach ? *out_reach : reach_local;
    auto& moves = out_moves ? *out_moves : moves_local;
    auto& move_cnt = out_move_cnt ? *out_move_cnt : move_cnt_local;
    move_cnt.fill(0);

    for (int p = 0; p < st.m; p++) {
        move_cnt[p] = enumerate_legal_moves(st, p, moves[p], &comp[p], &reach[p]);
    }
    for (int p = st.m; p < M_MAX; p++) {
        move_cnt[p] = 0;
        comp[p].fill(0);
        reach[p].fill(0);
    }

    for (int p = 0; p < M_MAX; p++) {
        for (int idx = 0; idx < plane_size; idx++) {
            if (comp[p][idx])
                out_board[(CH_COMP0 + p) * plane_size + idx] = 1.0f;
            if (reach[p][idx])
                out_board[(CH_REACH0 + p) * plane_size + idx] = 1.0f;
        }
    }

    if (st.m > 0) {
        for (int idx = 0; idx < plane_size; idx++)
            out_action_mask[idx] = reach[0][idx] ? 1 : 0;
    }

    // next distributions
    for (int p = 0; p < st.m; p++) {
        const int n = move_cnt[p];
        if (n <= 0)
            continue;

        if (p == 0 || !pf_enabled || pf == nullptr || p >= st.m) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++) {
                const int idx = moves[p][i];
                out_board[(CH_NEXT0 + p) * plane_size + idx] = prob;
            }
            continue;
        }

        // PF mixture distribution for opponents
        std::array<int, 4> vmax{};
        std::array<int, 4> cnt_vmax{};
        vmax.fill(0);
        cnt_vmax.fill(0);

        std::array<int, CELL_MAX> cat{};
        std::array<int, CELL_MAX> v{};
        bool has_nonzero = false;
        for (int i = 0; i < n; i++) {
            const int idx = moves[p][i];
            const int c = categorize_move_for_ai(st, p, idx);
            cat[i] = c;
            v[i] = st.value[idx];
            if (c >= 0) {
                has_nonzero = true;
                if (v[i] > vmax[c]) {
                    vmax[c] = v[i];
                    cnt_vmax[c] = 1;
                } else if (v[i] == vmax[c]) {
                    cnt_vmax[c]++;
                }
            }
        }

        if (!has_nonzero) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++) {
                out_board[(CH_NEXT0 + p) * plane_size + moves[p][i]] = prob;
            }
            continue;
        }

        std::array<std::array<int, CELL_MAX>, 4> best_move_ids{};
        std::array<int, 4> best_cnt{};
        best_cnt.fill(0);
        for (int i = 0; i < n; i++) {
            const int c = cat[i];
            if (c < 0 || c >= 4)
                continue;
            if (v[i] == vmax[c])
                best_move_ids[c][best_cnt[c]++] = i;
        }

        const auto& pf_p = (*pf)[p];
        std::array<double, ParticleFilterSMC::P> w{};
        pf_p.normalized_weights(w);

        double eps_mean = 0.0;
        for (int i = 0; i < ParticleFilterSMC::P; i++)
            eps_mean += w[i] * pf_p.particle_theta(i)[4];

        std::array<double, CELL_MAX> dist{};
        for (int i = 0; i < n; i++)
            dist[i] = eps_mean / static_cast<double>(n);

        // Greedy mass is uniform over best moves within each winning category for a particle.
        // Accumulate "per-move" mass per category first, then distribute to the best moves once.
        std::array<double, 4> greedy_mass_per_move{};
        greedy_mass_per_move.fill(0.0);
        for (int i = 0; i < ParticleFilterSMC::P; i++) {
            const auto& th = pf_p.particle_theta(i);
            const double wa = th[0];
            const double wb = th[1];
            const double wc = th[2];
            const double wd = th[3];
            const double eps = th[4];

            const double s0 = wa * static_cast<double>(vmax[0]);
            const double s1 = wb * static_cast<double>(vmax[1]);
            const double s2 = wc * static_cast<double>(vmax[2]);
            const double s3 = wd * static_cast<double>(vmax[3]);
            const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
            const double tol = 1e-12 * std::max(1.0, std::abs(mx));

            std::array<std::uint8_t, 4> is_best{};
            const std::array<double, 4> sc{s0, s1, s2, s3};
            int k = 0;
            for (int c = 0; c < 4; c++) {
                if (std::abs(sc[c] - mx) <= tol) {
                    is_best[c] = 1;
                    k += cnt_vmax[c];
                }
            }
            if (k <= 0)
                continue;

            const double mass = w[i] * (1.0 - eps) / static_cast<double>(k);
            for (int c = 0; c < 4; c++) {
                if (!is_best[c])
                    continue;
                greedy_mass_per_move[c] += mass;
            }
        }

        for (int c = 0; c < 4; c++) {
            const double add = greedy_mass_per_move[c];
            if (add == 0.0)
                continue;
            for (int bi = 0; bi < best_cnt[c]; bi++) {
                const int move_id = best_move_ids[c][bi];
                dist[move_id] += add;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += dist[i];
        if (!(sum > 0.0)) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++)
                out_board[(CH_NEXT0 + p) * plane_size + moves[p][i]] = prob;
            continue;
        }

        const double inv_sum = 1.0 / sum;
        for (int i = 0; i < n; i++) {
            const int idx = moves[p][i];
            out_board[(CH_NEXT0 + p) * plane_size + idx] = static_cast<float>(dist[i] * inv_sum);
        }
    }

    // score planes (constant per board)
    // score[p] = sum_{owner==p} value * level, normalized by 500000.
    // (In the official generator, sum(value)=100000 and u_max<=5, so score<=500000.)
    std::array<std::int64_t, M_MAX> score{};
    score.fill(0);
    constexpr double INV_SCORE_SCALE = 1.0 / 500000.0;  // for planes
    for (int idx = 0; idx < plane_size; idx++) {
        const int o = static_cast<int>(st.owner[idx]);
        if (0 <= o && o < st.m) {
            score[static_cast<std::size_t>(o)] += static_cast<std::int64_t>(st.value[idx]) *
                                                  static_cast<std::int64_t>(st.level[idx]);
        }
    }
    for (int p = 0; p < st.m; p++) {
        const double raw = static_cast<double>(score[static_cast<std::size_t>(p)]) * INV_SCORE_SCALE;
        const float v = static_cast<float>(std::min(1.0, std::max(0.0, raw)));
        for (int idx = 0; idx < plane_size; idx++)
            out_board[(CH_SCORE0 + p) * plane_size + idx] = v;
    }

    // Reorder opponent-related planes (p>=1) by descending score.
    // This is done at feature generation time (just before passing obs to Python),
    // so it does not affect simulation dynamics / RNG / PF updates.
    if (st.m >= 3) {  // at least 2 opponents
        std::array<int, M_MAX> opp_old{};
        int opp_n = 0;
        for (int p = 1; p < st.m; p++)
            opp_old[opp_n++] = p;

        std::sort(opp_old.begin(), opp_old.begin() + opp_n, [&](int a, int b) {
            const auto sa = score[static_cast<std::size_t>(a)];
            const auto sb = score[static_cast<std::size_t>(b)];
            if (sa != sb)
                return sa > sb;  // higher score first
            return a < b;        // stable tie-break
        });

        std::array<int, M_MAX> new_to_old{};
        for (int p = 0; p < M_MAX; p++)
            new_to_old[p] = p;
        for (int new_p = 1; new_p < st.m; new_p++)
            new_to_old[new_p] = opp_old[static_cast<std::size_t>(new_p - 1)];

        auto permute_group = [&](int ch_base) {
            std::array<std::array<float, CELL_MAX>, M_MAX> buf{};
            for (int old_p = 1; old_p < st.m; old_p++) {
                const float* src = out_board + (ch_base + old_p) * plane_size;
                std::memcpy(buf[static_cast<std::size_t>(old_p)].data(), src, plane_size * sizeof(float));
            }
            for (int new_p = 1; new_p < st.m; new_p++) {
                const int old_p = new_to_old[static_cast<std::size_t>(new_p)];
                float* dst = out_board + (ch_base + new_p) * plane_size;
                std::memcpy(dst, buf[static_cast<std::size_t>(old_p)].data(), plane_size * sizeof(float));
            }
        };

        permute_group(CH_OWNER0);
        permute_group(CH_COMP0);
        permute_group(CH_REACH0);
        permute_group(CH_NEXT0);
        permute_group(CH_SCORE0);
    }

    // constant planes
    const float turn_frac = (st.t_max > 0) ? (static_cast<float>(turn) / static_cast<float>(st.t_max)) : 0.0f;
    const float m_norm = static_cast<float>(st.m) / static_cast<float>(M_MAX);
    const float u_norm = static_cast<float>(st.u_max) / 5.0f;
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_TURN_FRAC * plane_size + idx] = turn_frac;
        out_board[CH_M_NORM * plane_size + idx] = m_norm;
        out_board[CH_U_NORM * plane_size + idx] = u_norm;
    }
}

}  // namespace ahc061::exp001
