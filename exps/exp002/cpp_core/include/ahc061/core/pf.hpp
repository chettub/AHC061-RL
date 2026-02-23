#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "ahc061/core/move_summary.hpp"
#include "ahc061/core/rules.hpp"

namespace ahc061::exp002 {

#ifndef AHC061_PF_PARTICLES
#define AHC061_PF_PARTICLES 1024
#endif

inline MoveSummary summarize_ai_observation(const State& st_start, int p, int action_cell) {
    std::array<int, CELL_MAX> moves{};
    const int cnt = enumerate_legal_moves(st_start, p, moves);
    return summarize_ai_observation_from_moves(st_start, p, action_cell, moves.data(), cnt);
}

struct ParticleFilterSMC {
    static constexpr int P = AHC061_PF_PARTICLES;
    static constexpr double PI = 3.141592653589793238462643383279502884;

    std::array<std::array<double, 5>, P> theta{};
    std::array<double, P> log_w{};  // 正規化済み（exp(log_w) の和が 1）
    XorShift64 rng;
    bool has_next_gauss = false;
    double next_gauss = 0.0;

    int resample_count = 0;
    int invalid_obs_count = 0;

    ParticleFilterSMC() : rng(1) { init_prior(); }
    explicit ParticleFilterSMC(std::uint64_t seed) : rng(seed) { init_prior(); }

    void reset(std::uint64_t seed) {
        rng = XorShift64(seed);
        has_next_gauss = false;
        next_gauss = 0.0;
        resample_count = 0;
        invalid_obs_count = 0;
        init_prior();
    }

    double uniform(double lo, double hi) { return lo + (hi - lo) * rng.next_double01(); }

    double normal01() {
        if (has_next_gauss) {
            has_next_gauss = false;
            return next_gauss;
        }
        double u1 = rng.next_double01();
        double u2 = rng.next_double01();
        if (u1 < 1e-12)
            u1 = 1e-12;
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double ang = 2.0 * PI * u2;
        const double z0 = r * std::cos(ang);
        const double z1 = r * std::sin(ang);
        has_next_gauss = true;
        next_gauss = z1;
        return z0;
    }

    static double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void init_prior() {
        for (int i = 0; i < P; i++) {
            theta[i][0] = uniform(0.3, 1.0);  // wa
            theta[i][1] = uniform(0.3, 1.0);  // wb
            theta[i][2] = uniform(0.3, 1.0);  // wc
            theta[i][3] = uniform(0.3, 1.0);  // wd
            theta[i][4] = uniform(0.1, 0.5);  // eps
            log_w[i] = -std::log(static_cast<double>(P));
        }
    }

    static double loglik(const MoveSummary& sum, const std::array<double, 5>& th) {
        constexpr double MIN_PROB = 1e-300;
        if (!sum.action_in_b || sum.n <= 0)
            return std::log(MIN_PROB);

        const double wa = th[0];
        const double wb = th[1];
        const double wc = th[2];
        const double wd = th[3];
        const double eps = th[4];

        const double prand = eps / static_cast<double>(sum.n);

        double pgreedy = 0.0;
        if (sum.all_zero) {
            pgreedy = 1.0 / static_cast<double>(sum.n);
        } else {
            const double s0 = wa * static_cast<double>(sum.vmax[0]);
            const double s1 = wb * static_cast<double>(sum.vmax[1]);
            const double s2 = wc * static_cast<double>(sum.vmax[2]);
            const double s3 = wd * static_cast<double>(sum.vmax[3]);
            const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
            const double tol = 1e-12 * std::max(1.0, std::abs(mx));
            std::array<std::uint8_t, 4> is_best{};
            int k = 0;
            const std::array<double, 4> sc{s0, s1, s2, s3};
            for (int c = 0; c < 4; c++) {
                if (std::abs(sc[c] - mx) <= tol) {
                    is_best[c] = 1;
                    k += sum.cnt[c];
                }
            }
            if (k > 0 && sum.action_cat >= 0 && sum.action_cat < 4) {
                const int c = sum.action_cat;
                if (is_best[c] && sum.action_value == sum.vmax[c])
                    pgreedy = 1.0 / static_cast<double>(k);
            }
        }

        const double p = std::max(MIN_PROB, prand + (1.0 - eps) * pgreedy);
        return std::log(p);
    }

    void renormalize(std::array<double, P>* out_w = nullptr, double* out_ess = nullptr) {
        double mx = log_w[0];
        for (int i = 1; i < P; i++)
            mx = std::max(mx, log_w[i]);
        double s = 0.0;
        std::array<double, P> w{};
        for (int i = 0; i < P; i++) {
            w[i] = std::exp(log_w[i] - mx);
            s += w[i];
        }
        if (!(s > 0.0)) {
            init_prior();
            w.fill(1.0 / static_cast<double>(P));
            if (out_w)
                *out_w = w;
            if (out_ess)
                *out_ess = static_cast<double>(P);
            return;
        }
        const double log_z = mx + std::log(s);
        double ess_inv = 0.0;
        for (int i = 0; i < P; i++) {
            log_w[i] -= log_z;
            w[i] = std::exp(log_w[i]);
            ess_inv += w[i] * w[i];
        }
        if (out_w)
            *out_w = w;
        if (out_ess)
            *out_ess = (ess_inv > 0.0) ? (1.0 / ess_inv) : 1.0;
    }

    void update(const MoveSummary& sum) {
        if (!sum.action_in_b)
            invalid_obs_count++;
        for (int i = 0; i < P; i++)
            log_w[i] += loglik(sum, theta[i]);

        std::array<double, P> w{};
        double ess = 0.0;
        renormalize(&w, &ess);
        if (ess >= 0.5 * static_cast<double>(P))
            return;

        // Liu–West (対角) 用の平均・分散（resample 前の重みで計算）
        std::array<double, 5> mu{};
        mu.fill(0.0);
        for (int i = 0; i < P; i++) {
            for (int d = 0; d < 5; d++)
                mu[d] += w[i] * theta[i][d];
        }
        std::array<double, 5> var{};
        var.fill(0.0);
        for (int i = 0; i < P; i++) {
            for (int d = 0; d < 5; d++) {
                const double diff = theta[i][d] - mu[d];
                var[d] += w[i] * diff * diff;
            }
        }
        for (int d = 0; d < 5; d++)
            var[d] = std::max(var[d], 1e-6);

        // systematic resample
        std::array<std::array<double, 5>, P> new_theta{};
        int i = 0;
        double cdf = w[0];
        const double u0 = rng.next_double01() / static_cast<double>(P);
        for (int j = 0; j < P; j++) {
            const double u = u0 + static_cast<double>(j) / static_cast<double>(P);
            while (u > cdf && i + 1 < P) {
                i++;
                cdf += w[i];
            }
            new_theta[j] = theta[i];
        }
        theta = new_theta;
        for (int k = 0; k < P; k++)
            log_w[k] = -std::log(static_cast<double>(P));
        resample_count++;

        // Liu–West move
        constexpr double a = 0.98;
        const double s = std::sqrt(1.0 - a * a);
        for (int k = 0; k < P; k++) {
            for (int d = 0; d < 5; d++) {
                const double noise = s * std::sqrt(var[d]) * normal01();
                theta[k][d] = a * theta[k][d] + (1.0 - a) * mu[d] + noise;
            }
            theta[k][0] = clip(theta[k][0], 0.3, 1.0);
            theta[k][1] = clip(theta[k][1], 0.3, 1.0);
            theta[k][2] = clip(theta[k][2], 0.3, 1.0);
            theta[k][3] = clip(theta[k][3], 0.3, 1.0);
            theta[k][4] = clip(theta[k][4], 0.1, 0.5);
        }
    }

    void normalized_weights(std::array<double, P>& w) const {
        double mx = log_w[0];
        for (int i = 1; i < P; i++)
            mx = std::max(mx, log_w[i]);
        double s = 0.0;
        for (int i = 0; i < P; i++) {
            w[i] = std::exp(log_w[i] - mx);
            s += w[i];
        }
        if (!(s > 0.0)) {
            w.fill(1.0 / static_cast<double>(P));
            return;
        }
        const double inv_s = 1.0 / s;
        for (int i = 0; i < P; i++)
            w[i] *= inv_s;
    }

    const std::array<double, 5>& particle_theta(int idx) const { return theta[idx]; }
};

}  // namespace ahc061::exp002
