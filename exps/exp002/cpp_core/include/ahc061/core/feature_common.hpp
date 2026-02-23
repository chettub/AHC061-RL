#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>

#include "ahc061/core/adf_beta_estimator.hpp"
#include "ahc061/core/a_softmax_laplace.hpp"
#include "ahc061/core/opponent_ai.hpp"
#include "ahc061/core/pf.hpp"
#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

enum class NextMode : std::uint8_t {
    k_uniform_or_pf = 0,
    k_a_softmax_ut = 1,
    k_adf_beta = 2,
};

struct FeatureCommon {
    const State* st = nullptr;
    int turn = 0;
    bool pf_enabled = true;
    const std::array<ParticleFilterSMC, M_MAX>* pf = nullptr;
    const std::array<ASoftmaxLaplaceEstimator, M_MAX>* a_softmax = nullptr;
    const std::array<AdfBetaEstimator, M_MAX>* adf_beta = nullptr;

    const std::array<std::array<int, CELL_MAX>, M_MAX>* moves = nullptr;
    const std::array<int, M_MAX>* move_cnt = nullptr;
    const std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>* comp = nullptr;
    const std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>* reach = nullptr;

    std::array<std::array<float, CELL_MAX>, M_MAX> next{};  // [old_p][cell] (old_p=actual player id)

    std::array<std::int64_t, M_MAX> score_raw{};
    std::array<float, M_MAX> score_plane{};  // normalized/clamped for planes

    std::array<int, M_MAX> old_to_new{};  // opponent reorder mapping (0 stays 0)

    float turn_frac = 0.0f;
    float m_norm = 0.0f;
    float u_norm = 0.0f;
    float pos0_x_norm = 0.0f;
    float pos0_y_norm = 0.0f;
};

inline void compute_feature_common_base_into(
    const State& st,
    int turn,
    const std::array<ParticleFilterSMC, M_MAX>* pf,
    const std::array<ASoftmaxLaplaceEstimator, M_MAX>* a_softmax,
    const std::array<AdfBetaEstimator, M_MAX>* adf_beta,
    bool pf_enabled,
    FeatureCommon& out,
    std::uint8_t* out_action_mask,  // [100]
    std::array<std::array<int, CELL_MAX>, M_MAX>& moves,
    std::array<int, M_MAX>& move_cnt,
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>& comp,
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>& reach,
    const std::array<std::int64_t, M_MAX>* precomputed_score_raw = nullptr) {
    out.st = &st;
    out.turn = turn;
    out.pf = pf;
    out.a_softmax = a_softmax;
    out.adf_beta = adf_beta;
    out.pf_enabled = pf_enabled;
    out.moves = &moves;
    out.move_cnt = &move_cnt;
    out.comp = &comp;
    out.reach = &reach;

    // enumerate legal moves (also used as cache for faster opponent simulation)
    move_cnt.fill(0);
    for (int p = 0; p < st.m; p++) {
        move_cnt[p] = enumerate_legal_moves(st, p, moves[p], &comp[p], &reach[p]);
    }
    for (int p = st.m; p < M_MAX; p++) {
        move_cnt[p] = 0;
        comp[p].fill(0);
        reach[p].fill(0);
    }

    // action mask for player0
    for (int idx = 0; idx < CELL_MAX; idx++)
        out_action_mask[idx] = (st.m > 0 && reach[0][idx]) ? 1 : 0;

    // score (raw + plane scalar)
    out.score_raw.fill(0);
    if (precomputed_score_raw != nullptr) {
        for (int p = 0; p < st.m; p++)
            out.score_raw[static_cast<std::size_t>(p)] = (*precomputed_score_raw)[static_cast<std::size_t>(p)];
    } else {
        for (int idx = 0; idx < CELL_MAX; idx++) {
            const int o = static_cast<int>(st.owner[idx]);
            if (0 <= o && o < st.m) {
                out.score_raw[static_cast<std::size_t>(o)] += static_cast<std::int64_t>(st.value[idx]) *
                                                             static_cast<std::int64_t>(st.level[idx]);
            }
        }
    }

    out.score_plane.fill(0.0f);
    constexpr double INV_SCORE_SCALE = 1.0 / 500000.0;  // for planes
    for (int p = 0; p < st.m; p++) {
        const double raw = static_cast<double>(out.score_raw[static_cast<std::size_t>(p)]) * INV_SCORE_SCALE;
        out.score_plane[static_cast<std::size_t>(p)] = static_cast<float>(std::min(1.0, std::max(0.0, raw)));
    }

    // opponent reorder (by descending score)
    for (int p = 0; p < M_MAX; p++)
        out.old_to_new[p] = p;
    if (st.m >= 3) {  // at least 2 opponents
        std::array<int, M_MAX> opp_old{};
        int opp_n = 0;
        for (int p = 1; p < st.m; p++)
            opp_old[opp_n++] = p;

        std::sort(opp_old.begin(), opp_old.begin() + opp_n, [&](int a, int b) {
            const auto sa = out.score_raw[static_cast<std::size_t>(a)];
            const auto sb = out.score_raw[static_cast<std::size_t>(b)];
            if (sa != sb)
                return sa > sb;  // higher score first
            return a < b;        // stable tie-break
        });

        for (int i = 0; i < opp_n; i++) {
            const int old_p = opp_old[static_cast<std::size_t>(i)];
            out.old_to_new[static_cast<std::size_t>(old_p)] = i + 1;
        }
    }

    // constant scalars
    out.turn_frac = (st.t_max > 0) ? (static_cast<float>(turn) / static_cast<float>(st.t_max)) : 0.0f;
    out.m_norm = static_cast<float>(st.m) / static_cast<float>(M_MAX);
    out.u_norm = static_cast<float>(st.u_max) / 5.0f;
    const float inv_nm1 = (N > 1) ? (1.0f / static_cast<float>(N - 1)) : 0.0f;
    out.pos0_x_norm = static_cast<float>(st.ex[0]) * inv_nm1;
    out.pos0_y_norm = static_cast<float>(st.ey[0]) * inv_nm1;
}

inline bool _chol3(
    const std::array<std::array<double, 3>, 3>& s,
    std::array<std::array<double, 3>, 3>& out_l) {
    for (int r = 0; r < 3; r++)
        out_l[static_cast<std::size_t>(r)].fill(0.0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            for (int k = 0; k < j; k++)
                sum -= out_l[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
                       out_l[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
            if (i == j) {
                if (!(sum > 1e-18))
                    return false;
                out_l[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = std::sqrt(sum);
            } else {
                const double denom = out_l[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)];
                if (!(denom > 1e-18))
                    return false;
                out_l[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = sum / denom;
            }
        }
    }
    return true;
}

inline void compute_feature_common_next_into(
    const State& st,
    const std::array<ParticleFilterSMC, M_MAX>* pf,
    const std::array<ASoftmaxLaplaceEstimator, M_MAX>* a_softmax,
    const std::array<AdfBetaEstimator, M_MAX>* adf_beta,
    bool pf_enabled,
    NextMode next_mode,
    const std::array<std::array<int, CELL_MAX>, M_MAX>& moves,
    const std::array<int, M_MAX>& move_cnt,
    FeatureCommon& out) {
    for (int p = 0; p < M_MAX; p++)
        out.next[static_cast<std::size_t>(p)].fill(0.0f);

    for (int p = 0; p < st.m; p++) {
        const int n = move_cnt[static_cast<std::size_t>(p)];
        if (n <= 0)
            continue;

        // player0 (and fallback): uniform over legal moves.
        if (p == 0) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++) {
                const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
                out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = prob;
            }
            continue;
        }

        if (next_mode == NextMode::k_a_softmax_ut) {
            if (a_softmax == nullptr) {
                const float prob = 1.0f / static_cast<float>(n);
                for (int i = 0; i < n; i++) {
                    const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
                    out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = prob;
                }
                continue;
            }

            const auto& est = (*a_softmax)[static_cast<std::size_t>(p)];
            const std::array<double, 3>& mu = est.mu;
            const std::array<std::array<double, 3>, 3>& lam = est.lambda;
            std::array<std::array<double, 3>, 3> sigma{};
            for (int r = 0; r < 3; r++)
                sigma[static_cast<std::size_t>(r)].fill(0.0);

            for (int j = 0; j < 3; j++) {
                std::array<double, 3> b{0.0, 0.0, 0.0};
                b[static_cast<std::size_t>(j)] = 1.0;
                const std::array<double, 3> x = ASoftmaxLaplaceEstimator::solve3(lam, b);
                for (int r = 0; r < 3; r++)
                    sigma[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(r)];
            }

            std::array<std::array<double, 3>, 3> l{};
            bool ok = _chol3(sigma, l);
            if (!ok) {
                // Add diagonal jitter to covariance and retry.
                for (int it = 0; it < 3 && !ok; it++) {
                    const double j = std::pow(10.0, -12.0 + static_cast<double>(it) * 2.0);
                    for (int k = 0; k < 3; k++)
                        sigma[static_cast<std::size_t>(k)][static_cast<std::size_t>(k)] += j;
                    ok = _chol3(sigma, l);
                }
            }

            if (!ok) {
                const float prob = 1.0f / static_cast<float>(n);
                for (int i = 0; i < n; i++) {
                    const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
                    out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = prob;
                }
                continue;
            }

            const double eps = ASoftmaxLaplaceEstimator::clip(est.eps, 0.0, 1.0);
            constexpr double SQRT_D = 1.7320508075688772935;  // sqrt(3)
            constexpr double W = 1.0 / 6.0;                   // 1/(2d), d=3

            std::array<float, CELL_MAX> tmp_dist{};
            for (int col = 0; col < 3; col++) {
                std::array<double, 3> dir{l[0][static_cast<std::size_t>(col)], l[1][static_cast<std::size_t>(col)], l[2][static_cast<std::size_t>(col)]};

                for (int sgn = -1; sgn <= 1; sgn += 2) {
                    std::array<double, 3> dlt = mu;
                    for (int k = 0; k < 3; k++) {
                        dlt[static_cast<std::size_t>(k)] += static_cast<double>(sgn) * SQRT_D * dir[static_cast<std::size_t>(k)];
                        dlt[static_cast<std::size_t>(k)] = ASoftmaxLaplaceEstimator::clip(
                            dlt[static_cast<std::size_t>(k)],
                            -est.delta_clip,
                            est.delta_clip);
                    }

                    OpponentParam param{};
                    param.wa = 1.0;
                    param.wb = std::exp(dlt[0]);
                    param.wc = std::exp(dlt[1]);
                    param.wd = std::exp(dlt[2]);
                    param.eps = eps;

                    compute_move_dist_ai_like_from_moves(
                        st,
                        p,
                        param,
                        moves[static_cast<std::size_t>(p)].data(),
                        n,
                        tmp_dist.data());
                    for (int idx = 0; idx < CELL_MAX; idx++) {
                        out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] += static_cast<float>(W) * tmp_dist[static_cast<std::size_t>(idx)];
                    }
                }
            }
            continue;
        }

        if (next_mode == NextMode::k_adf_beta) {
            if (adf_beta == nullptr) {
                const float prob = 1.0f / static_cast<float>(n);
                for (int i = 0; i < n; i++) {
                    const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
                    out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = prob;
                }
                continue;
            }

            const OpponentParam param = (*adf_beta)[static_cast<std::size_t>(p)].mean_param();
            std::array<float, CELL_MAX> tmp_dist{};
            compute_move_dist_ai_like_from_moves(
                st,
                p,
                param,
                moves[static_cast<std::size_t>(p)].data(),
                n,
                tmp_dist.data());
            for (int idx = 0; idx < CELL_MAX; idx++)
                out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = tmp_dist[static_cast<std::size_t>(idx)];
            continue;
        }

        // Uniform or PF mixture mode.
        if (!pf_enabled || pf == nullptr || p >= st.m) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++) {
                const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
                out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = prob;
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
            const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
            const int c = categorize_move_for_ai(st, p, idx);
            cat[static_cast<std::size_t>(i)] = c;
            v[static_cast<std::size_t>(i)] = st.value[idx];
            if (c >= 0) {
                has_nonzero = true;
                if (v[static_cast<std::size_t>(i)] > vmax[static_cast<std::size_t>(c)]) {
                    vmax[static_cast<std::size_t>(c)] = v[static_cast<std::size_t>(i)];
                    cnt_vmax[static_cast<std::size_t>(c)] = 1;
                } else if (v[static_cast<std::size_t>(i)] == vmax[static_cast<std::size_t>(c)]) {
                    cnt_vmax[static_cast<std::size_t>(c)]++;
                }
            }
        }

        if (!has_nonzero) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++) {
                out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)])] = prob;
            }
            continue;
        }

        std::array<std::array<int, CELL_MAX>, 4> best_move_ids{};
        std::array<int, 4> best_cnt{};
        best_cnt.fill(0);
        for (int i = 0; i < n; i++) {
            const int c = cat[static_cast<std::size_t>(i)];
            if (c < 0 || c >= 4)
                continue;
            if (v[static_cast<std::size_t>(i)] == vmax[static_cast<std::size_t>(c)])
                best_move_ids[static_cast<std::size_t>(c)][static_cast<std::size_t>(best_cnt[static_cast<std::size_t>(c)]++)] = i;
        }

        const auto& pf_p = (*pf)[static_cast<std::size_t>(p)];
        std::array<double, ParticleFilterSMC::P> w{};
        pf_p.normalized_weights(w);

        double eps_mean = 0.0;
        for (int i = 0; i < ParticleFilterSMC::P; i++)
            eps_mean += w[static_cast<std::size_t>(i)] * pf_p.particle_theta(i)[4];

        std::array<double, CELL_MAX> dist{};
        for (int i = 0; i < n; i++)
            dist[static_cast<std::size_t>(i)] = eps_mean / static_cast<double>(n);

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
            const double eps_i = th[4];

            const double s0 = wa * static_cast<double>(vmax[0]);
            const double s1 = wb * static_cast<double>(vmax[1]);
            const double s2 = wc * static_cast<double>(vmax[2]);
            const double s3 = wd * static_cast<double>(vmax[3]);
            const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
            const double tol = 1e-12 * std::max(1.0, std::abs(mx));

            std::array<std::uint8_t, 4> is_best{};
            is_best.fill(0);
            const std::array<double, 4> sc{s0, s1, s2, s3};
            int k = 0;
            for (int c = 0; c < 4; c++) {
                if (std::abs(sc[static_cast<std::size_t>(c)] - mx) <= tol) {
                    is_best[static_cast<std::size_t>(c)] = 1;
                    k += cnt_vmax[static_cast<std::size_t>(c)];
                }
            }
            if (k <= 0)
                continue;

            const double mass = w[static_cast<std::size_t>(i)] * (1.0 - eps_i) / static_cast<double>(k);
            for (int c = 0; c < 4; c++) {
                if (!is_best[static_cast<std::size_t>(c)])
                    continue;
                greedy_mass_per_move[static_cast<std::size_t>(c)] += mass;
            }
        }

        for (int c = 0; c < 4; c++) {
            const double add = greedy_mass_per_move[static_cast<std::size_t>(c)];
            if (add == 0.0)
                continue;
            for (int bi = 0; bi < best_cnt[static_cast<std::size_t>(c)]; bi++) {
                const int move_id = best_move_ids[static_cast<std::size_t>(c)][static_cast<std::size_t>(bi)];
                dist[static_cast<std::size_t>(move_id)] += add;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += dist[static_cast<std::size_t>(i)];
        if (!(sum > 0.0)) {
            const float prob = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; i++) {
                out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)])] = prob;
            }
            continue;
        }

        const double inv_sum = 1.0 / sum;
        for (int i = 0; i < n; i++) {
            const int idx = moves[static_cast<std::size_t>(p)][static_cast<std::size_t>(i)];
            out.next[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] = static_cast<float>(dist[static_cast<std::size_t>(i)] * inv_sum);
        }
    }
}

inline void compute_feature_common_into(
    const State& st,
    int turn,
    const std::array<ParticleFilterSMC, M_MAX>* pf,
    bool pf_enabled,
    FeatureCommon& out,
    std::uint8_t* out_action_mask,  // [100]
    std::array<std::array<int, CELL_MAX>, M_MAX>& moves,
    std::array<int, M_MAX>& move_cnt,
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>& comp,
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX>& reach,
    NextMode next_mode = NextMode::k_uniform_or_pf,
    const std::array<ASoftmaxLaplaceEstimator, M_MAX>* a_softmax = nullptr,
    const std::array<AdfBetaEstimator, M_MAX>* adf_beta = nullptr,
    const std::array<std::int64_t, M_MAX>* precomputed_score_raw = nullptr) {
    compute_feature_common_base_into(
        st,
        turn,
        pf,
        a_softmax,
        adf_beta,
        pf_enabled,
        out,
        out_action_mask,
        moves,
        move_cnt,
        comp,
        reach,
        precomputed_score_raw);
    compute_feature_common_next_into(st, pf, a_softmax, adf_beta, pf_enabled, next_mode, moves, move_cnt, out);
}

}  // namespace ahc061::exp002
