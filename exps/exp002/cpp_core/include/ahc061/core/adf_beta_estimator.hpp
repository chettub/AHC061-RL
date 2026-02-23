#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

#include "ahc061/core/move_summary.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp002 {

struct AdfBetaEstimator {
    // Recommended setting from exp003 large survey + tuned search.
    static constexpr double PRIOR_STD = 0.325;
    static constexpr double EPS0 = 0.30;

    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double jitter = 1e-15;

    std::array<double, 3> mu{};
    std::array<std::array<double, 3>, 3> sigma{};

    double beta_a = 1.0;
    double beta_b = 1.0;

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    static inline double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    static inline std::array<double, 3> mat_vec3(
        const std::array<std::array<double, 3>, 3>& a,
        const std::array<double, 3>& x) {
        return {
            a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
            a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
            a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
        };
    }

    static inline void symmetrize3(std::array<std::array<double, 3>, 3>& a) {
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                const double v = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = v;
                a[j][i] = v;
            }
        }
    }

    static inline double phi(double x) {
        static constexpr double INV_SQRT2PI = 0.39894228040143267794;
        return INV_SQRT2PI * std::exp(-0.5 * x * x);
    }

    static inline double sf(double x) {
        if (x > 38.0)
            return 0.0;
        if (x > 8.0) {
            const double p = phi(x);
            return (x > 0.0) ? (p / x) : 1.0;
        }
        return 0.5 * std::erfc(x / 1.4142135623730950488);
    }

    static inline int delta_idx_from_cat(int cat) {
        if (cat == 1)
            return 0;
        if (cat == 2)
            return 1;
        if (cat == 3)
            return 2;
        return -1;
    }

    static inline std::array<double, 3> make_a_vec(int cat_i, int cat_j) {
        std::array<double, 3> a{};
        a.fill(0.0);
        const int ii = delta_idx_from_cat(cat_i);
        const int jj = delta_idx_from_cat(cat_j);
        if (ii >= 0)
            a[ii] += 1.0;
        if (jj >= 0)
            a[jj] -= 1.0;
        return a;
    }

    static inline bool trunc_halfspace_into(
        const std::array<double, 3>& mu_in,
        const std::array<std::array<double, 3>, 3>& sigma_in,
        const std::array<double, 3>& a,
        double b,
        double& out_p,
        std::array<double, 3>& mu_out,
        std::array<std::array<double, 3>, 3>& sigma_out) {
        const double m = dot3(a, mu_in);
        const std::array<double, 3> sigma_a = mat_vec3(sigma_in, a);
        const double s2 = dot3(a, sigma_a);
        if (s2 <= 1e-18) {
            out_p = (m >= b) ? 1.0 : 0.0;
            mu_out = mu_in;
            sigma_out = sigma_in;
            return out_p > 0.0;
        }
        const double s = std::sqrt(s2);
        const double alpha = (b - m) / s;
        const double p = sf(alpha);
        out_p = p;
        if (!(p > 0.0)) {
            mu_out = mu_in;
            sigma_out = sigma_in;
            return false;
        }
        double lam = 0.0;
        if (alpha > 8.0) {
            lam = alpha + 1.0 / std::max(1e-12, alpha);
        } else {
            lam = phi(alpha) / p;
        }
        const double kappa = lam * (lam - alpha);

        mu_out = mu_in;
        for (int i = 0; i < 3; i++)
            mu_out[i] += sigma_a[i] * (lam / s);

        sigma_out = sigma_in;
        const double coef = kappa / s2;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                sigma_out[i][j] -= sigma_a[i] * sigma_a[j] * coef;
        }
        symmetrize3(sigma_out);
        for (int i = 0; i < 3; i++)
            sigma_out[i][i] += 1e-15;
        return true;
    }

    void beta_update_linear(double alpha, double beta) {
        // Moment-matched Beta update for linear likelihood l(u)=alpha+beta*u.
        alpha = std::max(0.0, alpha);
        if (!std::isfinite(alpha) || !std::isfinite(beta))
            return;

        const double s = beta_a + beta_b;
        if (!(s > 0.0))
            return;

        if (alpha + beta < 0.0)
            beta = -alpha;

        double w0 = 0.0;
        double w1 = 0.0;
        const double e0 = beta_a / s;
        const double e0_2 = (beta_a * (beta_a + 1.0)) / (s * (s + 1.0));
        double e1 = e0;
        double e1_2 = e0_2;

        if (beta >= 0.0) {
            w0 = alpha;
            w1 = beta * (beta_a / s);
            const double s1 = s + 1.0;
            e1 = (beta_a + 1.0) / s1;
            e1_2 = ((beta_a + 1.0) * (beta_a + 2.0)) / (s1 * (s1 + 1.0));
        } else {
            const double alpha1 = std::max(0.0, alpha + beta);
            w0 = alpha1;
            w1 = (-beta) * (beta_b / s);
            const double s1 = s + 1.0;
            e1 = beta_a / s1;
            e1_2 = (beta_a * (beta_a + 1.0)) / (s1 * (s1 + 1.0));
        }

        const double w = w0 + w1;
        if (!(w > 0.0))
            return;
        const double inv_w = 1.0 / w;
        double m1 = (w0 * e0 + w1 * e1) * inv_w;
        double m2 = (w0 * e0_2 + w1 * e1_2) * inv_w;

        m1 = clip(m1, 1e-6, 1.0 - 1e-6);
        double var = m2 - m1 * m1;
        const double var_max = 0.999 * m1 * (1.0 - m1);
        var = std::min(std::max(var, 1e-8), std::max(1e-8, var_max));

        const double k = m1 * (1.0 - m1) / var - 1.0;
        beta_a = std::max(1e-3, m1 * k);
        beta_b = std::max(1e-3, (1.0 - m1) * k);
    }

    void reset(double prior_std = PRIOR_STD, double eps0 = EPS0) {
        mu.fill(0.0);
        const double v = prior_std * prior_std;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                sigma[i][j] = (i == j) ? v : 0.0;
        }

        const double u0_raw = (clip(eps0, eps_min, eps_max) - eps_min) / (eps_max - eps_min);
        const double u0 = clip(u0_raw, 1e-3, 1.0 - 1e-3);
        constexpr double CONC = 2.0;
        beta_a = std::max(1e-3, u0 * CONC);
        beta_b = std::max(1e-3, (1.0 - u0) * CONC);
    }

    double eps_mean() const {
        const double s = beta_a + beta_b;
        if (!(s > 0.0))
            return 0.3;
        const double u = beta_a / s;
        return clip(eps_min + (eps_max - eps_min) * u, eps_min, eps_max);
    }

    void update(const MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero)
            return;

        const int n = sum.n;
        const int c_star = sum.action_cat;
        const int v_star = sum.action_value;

        const bool greedy_possible =
            (0 <= c_star && c_star < 4) && (v_star == sum.vmax[static_cast<std::size_t>(c_star)]) &&
            (sum.cnt[static_cast<std::size_t>(c_star)] > 0);
        const double g = greedy_possible ? static_cast<double>(sum.cnt[static_cast<std::size_t>(c_star)]) : 1.0;

        double m = 0.0;
        std::array<double, 3> mu_g = mu;
        std::array<std::array<double, 3>, 3> sigma_g = sigma;
        bool ok = false;
        if (greedy_possible) {
            std::array<std::pair<double, int>, 3> cons{};
            int cons_n = 0;
            const double log_v_star = std::log(static_cast<double>(std::max(1, v_star)));
            for (int c = 0; c < 4; c++) {
                if (c == c_star)
                    continue;
                if (sum.cnt[static_cast<std::size_t>(c)] <= 0)
                    continue;
                const int vm = sum.vmax[static_cast<std::size_t>(c)];
                if (vm <= 0)
                    continue;
                const double b = std::log(static_cast<double>(vm)) - log_v_star;
                cons[static_cast<std::size_t>(cons_n++)] = {b, c};
            }
            std::sort(cons.begin(), cons.begin() + cons_n, [&](const auto& a, const auto& b) {
                if (a.first != b.first)
                    return a.first > b.first;
                return a.second < b.second;
            });

            double log_m = 0.0;
            ok = true;
            for (int i = 0; i < cons_n; i++) {
                const double b = cons[static_cast<std::size_t>(i)].first;
                const int c_other = cons[static_cast<std::size_t>(i)].second;
                const std::array<double, 3> a = make_a_vec(c_star, c_other);
                double p_i = 0.0;
                std::array<double, 3> mu2{};
                std::array<std::array<double, 3>, 3> sig2{};
                if (!trunc_halfspace_into(mu_g, sigma_g, a, b, p_i, mu2, sig2)) {
                    ok = false;
                    break;
                }
                log_m += std::log(std::max(1e-300, p_i));
                mu_g = mu2;
                sigma_g = sig2;
            }
            if (ok) {
                m = (log_m < -700.0) ? 0.0 : std::exp(log_m);
            } else {
                m = 0.0;
            }
        } else {
            ok = false;
            m = 0.0;
        }

        const double inv_n = 1.0 / static_cast<double>(n);
        const double inv_g = 1.0 / std::max(1.0, g);
        const double alpha = eps_min * inv_n + (1.0 - eps_min) * m * inv_g;
        const double beta = (eps_max - eps_min) * (inv_n - m * inv_g);
        beta_update_linear(alpha, beta);

        const double eps_bar = eps_mean();
        if (greedy_possible && ok && m > 0.0) {
            const double c0 = eps_bar * inv_n;
            const double c1 = (1.0 - eps_bar) * inv_g;
            const double z = c0 + c1 * m;
            const double lam = (z > 0.0) ? ((c1 * m) / z) : 0.0;
            const double w1 = 1.0 - lam;
            const double w2 = lam;

            std::array<double, 3> mu_new{};
            for (int i = 0; i < 3; i++)
                mu_new[i] = w1 * mu[i] + w2 * mu_g[i];

            std::array<std::array<double, 3>, 3> exx{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    const double m1 = mu[i] * mu[j];
                    const double m2 = mu_g[i] * mu_g[j];
                    exx[i][j] = w1 * (sigma[i][j] + m1) + w2 * (sigma_g[i][j] + m2);
                }
            }

            std::array<std::array<double, 3>, 3> sigma_new{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    sigma_new[i][j] = exx[i][j] - mu_new[i] * mu_new[j];
            }
            symmetrize3(sigma_new);
            for (int i = 0; i < 3; i++)
                sigma_new[i][i] += jitter;

            for (int i = 0; i < 3; i++)
                mu[i] = clip(mu_new[i], -delta_clip, delta_clip);
            sigma = sigma_new;
        }
    }

    OpponentParam mean_param() const {
        const double vb = std::max(0.0, sigma[0][0]);
        const double vc = std::max(0.0, sigma[1][1]);
        const double vd = std::max(0.0, sigma[2][2]);
        const double db = clip(mu[0] + 0.5 * vb, -delta_clip, delta_clip);
        const double dc = clip(mu[1] + 0.5 * vc, -delta_clip, delta_clip);
        const double dd = clip(mu[2] + 0.5 * vd, -delta_clip, delta_clip);

        OpponentParam out{};
        out.wa = 1.0;
        out.wb = std::exp(db);
        out.wc = std::exp(dc);
        out.wd = std::exp(dd);
        out.eps = clip(eps_mean(), eps_min, eps_max);
        return out;
    }
};

}  // namespace ahc061::exp002
