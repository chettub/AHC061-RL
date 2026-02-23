#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "ahc061/core/move_summary.hpp"

namespace ahc061::exp002 {

struct ASoftmaxLaplaceEstimator {
    // Fixed hyperparams for research_v3.
    static constexpr double TAU = 0.1;
    static constexpr double PRIOR_STD = 0.5;
    static constexpr double EPS0 = 0.5;

    double tau = TAU;
    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double damping = 1.0;
    double jitter = 1e-6;

    std::int64_t n_obs = 0;
    double gamma_sum = 0.0;

    std::array<double, 3> mu{};                      // delta mean
    std::array<std::array<double, 3>, 3> lambda{};  // precision (inverse covariance)

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    static inline void softmax4_masked(
        const std::array<double, 4>& u_over_tau,
        const std::array<std::uint8_t, 4>& present,
        std::array<double, 4>& out_p) {
        double mx = -1e300;
        bool any = false;
        for (int k = 0; k < 4; k++) {
            if (!present[static_cast<std::size_t>(k)])
                continue;
            any = true;
            mx = std::max(mx, u_over_tau[static_cast<std::size_t>(k)]);
        }
        if (!any) {
            out_p.fill(0.0);
            return;
        }
        double s = 0.0;
        std::array<double, 4> e{};
        e.fill(0.0);
        for (int k = 0; k < 4; k++) {
            if (!present[static_cast<std::size_t>(k)])
                continue;
            const double ek = std::exp(u_over_tau[static_cast<std::size_t>(k)] - mx);
            e[static_cast<std::size_t>(k)] = ek;
            s += ek;
        }
        if (!(s > 0.0)) {
            out_p.fill(0.0);
            return;
        }
        const double inv_s = 1.0 / s;
        for (int k = 0; k < 4; k++)
            out_p[static_cast<std::size_t>(k)] = e[static_cast<std::size_t>(k)] * inv_s;
    }

    // Solve A x = b for 3x3 A using Gaussian elimination with partial pivoting.
    // Returns {0,0,0} on failure.
    static inline std::array<double, 3> solve3(
        const std::array<std::array<double, 3>, 3>& a,
        const std::array<double, 3>& b) {
        std::array<std::array<double, 4>, 3> m{};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = a[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            m[static_cast<std::size_t>(r)][3] = b[static_cast<std::size_t>(r)];
        }

        for (int col = 0; col < 3; col++) {
            int piv = col;
            double best = std::abs(m[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)]);
            for (int r = col + 1; r < 3; r++) {
                const double v = std::abs(m[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)]);
                if (v > best) {
                    best = v;
                    piv = r;
                }
            }
            if (!(best > 1e-18))
                return {0.0, 0.0, 0.0};
            if (piv != col)
                std::swap(m[static_cast<std::size_t>(piv)], m[static_cast<std::size_t>(col)]);

            const double div = m[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
            const double inv = 1.0 / div;
            for (int c = col; c < 4; c++)
                m[static_cast<std::size_t>(col)][static_cast<std::size_t>(c)] *= inv;

            for (int r = 0; r < 3; r++) {
                if (r == col)
                    continue;
                const double f = m[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)];
                if (f == 0.0)
                    continue;
                for (int c = col; c < 4; c++)
                    m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] -= f * m[static_cast<std::size_t>(col)][static_cast<std::size_t>(c)];
            }
        }

        return {m[0][3], m[1][3], m[2][3]};
    }

    void reset(double prior_std = PRIOR_STD, double eps0 = EPS0) {
        tau = TAU;
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        gamma_sum = 0.0;
        mu.fill(0.0);

        const double inv_var = 1.0 / (prior_std * prior_std);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                lambda[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? inv_var : 0.0;
        }
    }

    void update(const MoveSummary& sum) {
        if (!sum.action_in_b)
            return;
        const int n = sum.n;
        if (n <= 0)
            return;

        const double u_rand = 1.0 / static_cast<double>(n);

        // When greedy is uniform too, the observation is uninformative for (eps, delta).
        if (sum.all_zero)
            return;

        // Present categories = those with at least one move in that category (and nonzero greedy value).
        std::array<std::uint8_t, 4> present{};
        for (int k = 0; k < 4; k++) {
            const int cnt = sum.cnt[static_cast<std::size_t>(k)];
            const int vm = sum.vmax[static_cast<std::size_t>(k)];
            present[static_cast<std::size_t>(k)] = (cnt > 0 && vm > 0) ? 1 : 0;
        }

        // Utilities (log space) / tau.
        std::array<double, 4> u_over_tau{};
        u_over_tau.fill(0.0);
        if (present[0])
            u_over_tau[0] = std::log(static_cast<double>(sum.vmax[0])) / tau;
        if (present[1])
            u_over_tau[1] = (mu[0] + std::log(static_cast<double>(sum.vmax[1]))) / tau;
        if (present[2])
            u_over_tau[2] = (mu[1] + std::log(static_cast<double>(sum.vmax[2]))) / tau;
        if (present[3])
            u_over_tau[3] = (mu[2] + std::log(static_cast<double>(sum.vmax[3]))) / tau;

        std::array<double, 4> p_cat{};
        softmax4_masked(u_over_tau, present, p_cat);

        // Greedy probability of choosing the exact observed move.
        double g = 0.0;
        if (0 <= sum.action_cat && sum.action_cat < 4) {
            const int c = sum.action_cat;
            if (present[static_cast<std::size_t>(c)] && sum.action_value == sum.vmax[static_cast<std::size_t>(c)] &&
                sum.cnt[static_cast<std::size_t>(c)] > 0) {
                g = p_cat[static_cast<std::size_t>(c)] / static_cast<double>(sum.cnt[static_cast<std::size_t>(c)]);
            } else {
                g = 0.0;
            }
        } else {
            g = 0.0;
        }

        double gamma = 1.0;
        if (g > 0.0) {
            const double p_total = eps * u_rand + (1.0 - eps) * g;
            gamma = (p_total > 0.0) ? ((eps * u_rand) / p_total) : 1.0;
        } else {
            gamma = 1.0;
        }

        // epsilon: running average of responsibility, then clip.
        n_obs++;
        gamma_sum += gamma;
        eps = clip(gamma_sum / static_cast<double>(n_obs), eps_min, eps_max);

        // delta: ADF-like Gaussian update on multinomial logit (B,C,D vs baseline A).
        const double w = 1.0 - gamma;
        if (w <= 1e-9)
            return;

        const double pb = p_cat[1];
        const double pc = p_cat[2];
        const double pd = p_cat[3];

        const double yb = (sum.action_cat == 1) ? 1.0 : 0.0;
        const double yc = (sum.action_cat == 2) ? 1.0 : 0.0;
        const double yd = (sum.action_cat == 3) ? 1.0 : 0.0;

        // grad = w/tau * (P - y)
        const double scale_g = w / tau;
        std::array<double, 3> grad{scale_g * (pb - yb), scale_g * (pc - yc), scale_g * (pd - yd)};

        // Hessian = w/tau^2 * (Diag(P) - P P^T)
        const double scale_h = w / (tau * tau);
        std::array<std::array<double, 3>, 3> h{};
        for (int i = 0; i < 3; i++)
            h[static_cast<std::size_t>(i)].fill(0.0);
        h[0][0] = scale_h * (pb - pb * pb);
        h[1][1] = scale_h * (pc - pc * pc);
        h[2][2] = scale_h * (pd - pd * pd);
        h[0][1] = scale_h * (-pb * pc);
        h[0][2] = scale_h * (-pb * pd);
        h[1][2] = scale_h * (-pc * pd);
        h[1][0] = h[0][1];
        h[2][0] = h[0][2];
        h[2][1] = h[1][2];

        std::array<std::array<double, 3>, 3> lambda_new = lambda;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                lambda_new[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] += h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            lambda_new[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] += jitter;
        }

        const std::array<double, 3> step = solve3(lambda_new, grad);
        for (int i = 0; i < 3; i++) {
            mu[static_cast<std::size_t>(i)] = clip(
                mu[static_cast<std::size_t>(i)] - damping * step[static_cast<std::size_t>(i)],
                -delta_clip,
                delta_clip);
        }
        lambda = lambda_new;
    }
};

}  // namespace ahc061::exp002
