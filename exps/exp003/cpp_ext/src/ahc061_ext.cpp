#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#define AHC061_EXP003_HAS_EIGEN 1
#else
#define AHC061_EXP003_HAS_EIGEN 0
#endif

#include "ahc061/core/generator.hpp"
#include "ahc061/core/opponent_ai.hpp"
#include "ahc061/core/pf.hpp"
#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

namespace ahc061::exp003 {

namespace core = ahc061::exp002;

struct ParamNorm {
    std::array<double, 4> w_norm{};
    double eps = 0.0;
};

static inline ParamNorm wnorm_eps_from_delta_eps(const std::array<double, 3>& delta, double eps) {
    // delta = (log(wb/wa), log(wc/wa), log(wd/wa))
    const double rb = std::exp(delta[0]);
    const double rc = std::exp(delta[1]);
    const double rd = std::exp(delta[2]);
    const double s = 1.0 + rb + rc + rd;
    const double inv_s = (s > 0.0) ? (1.0 / s) : 1.0;
    ParamNorm out{};
    out.w_norm[0] = 1.0 * inv_s;
    out.w_norm[1] = rb * inv_s;
    out.w_norm[2] = rc * inv_s;
    out.w_norm[3] = rd * inv_s;
    out.eps = eps;
    return out;
}

static inline ParamNorm normalize_true_param(const core::OpponentParam& op) {
    ParamNorm out{};
    double s = op.wa + op.wb + op.wc + op.wd;
    if (!(s > 0.0))
        s = 1.0;
    out.w_norm[0] = op.wa / s;
    out.w_norm[1] = op.wb / s;
    out.w_norm[2] = op.wc / s;
    out.w_norm[3] = op.wd / s;
    out.eps = op.eps;
    return out;
}

static inline ParamNorm estimate_param_mean_norm(const core::ParticleFilterSMC& pf) {
    ParamNorm out{};

    double mx = pf.log_w[0];
    for (int i = 1; i < core::ParticleFilterSMC::P; i++)
        mx = std::max(mx, pf.log_w[i]);

    double s = 0.0;
    for (int i = 0; i < core::ParticleFilterSMC::P; i++)
        s += std::exp(pf.log_w[i] - mx);

    if (!(s > 0.0)) {
        const double inv_p = 1.0 / static_cast<double>(core::ParticleFilterSMC::P);
        for (int i = 0; i < core::ParticleFilterSMC::P; i++) {
            const auto& th = pf.theta[i];
            double sw = th[0] + th[1] + th[2] + th[3];
            if (!(sw > 0.0))
                sw = 1.0;
            out.w_norm[0] += inv_p * th[0] / sw;
            out.w_norm[1] += inv_p * th[1] / sw;
            out.w_norm[2] += inv_p * th[2] / sw;
            out.w_norm[3] += inv_p * th[3] / sw;
            out.eps += inv_p * th[4];
        }
        return out;
    }

    const double inv_s = 1.0 / s;
    for (int i = 0; i < core::ParticleFilterSMC::P; i++) {
        const double wi = std::exp(pf.log_w[i] - mx) * inv_s;
        const auto& th = pf.theta[i];
        double sw = th[0] + th[1] + th[2] + th[3];
        if (!(sw > 0.0))
            sw = 1.0;
        out.w_norm[0] += wi * th[0] / sw;
        out.w_norm[1] += wi * th[1] / sw;
        out.w_norm[2] += wi * th[2] / sw;
        out.w_norm[3] += wi * th[3] / sw;
        out.eps += wi * th[4];
    }
    return out;
}

struct SoftmaxLaplaceEstimator {
    // Proposal A (simplified):
    // - 4-category softmax over log(Vmax) + delta (delta is log ratios to A)
    // - EM-like responsibility updates epsilon
    // - ADF-like Gaussian update for delta mean/precision (deterministic)
    double tau = 0.2;
    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double damping = 1.0;
    double jitter = 1e-6;

    std::int64_t n_obs = 0;
    double gamma_sum = 0.0;

    std::array<double, 3> mu{};  // delta mean
    std::array<std::array<double, 3>, 3> lambda{};  // precision (inverse covariance)

    void reset(double prior_std, double eps0) {
        tau = 0.2;
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        gamma_sum = 0.0;
        mu.fill(0.0);
        const double inv_var = 1.0 / (prior_std * prior_std);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                lambda[i][j] = (i == j) ? inv_var : 0.0;
        }
    }

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    static inline void softmax4_masked(const std::array<double, 4>& u_over_tau, const std::array<std::uint8_t, 4>& present, std::array<double, 4>& out_p) {
        double mx = -1e300;
        bool any = false;
        for (int k = 0; k < 4; k++) {
            if (!present[k])
                continue;
            any = true;
            mx = std::max(mx, u_over_tau[k]);
        }
        if (!any) {
            out_p.fill(0.0);
            return;
        }
        double s = 0.0;
        std::array<double, 4> e{};
        e.fill(0.0);
        for (int k = 0; k < 4; k++) {
            if (!present[k])
                continue;
            e[k] = std::exp(u_over_tau[k] - mx);
            s += e[k];
        }
        if (!(s > 0.0)) {
            out_p.fill(0.0);
            return;
        }
        const double inv_s = 1.0 / s;
        for (int k = 0; k < 4; k++)
            out_p[k] = e[k] * inv_s;
    }

    static inline std::array<double, 3> solve3(const std::array<std::array<double, 3>, 3>& a, const std::array<double, 3>& b) {
        // Gaussian elimination with partial pivoting for 3x3.
        std::array<std::array<double, 4>, 3> m{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                m[i][j] = a[i][j];
            m[i][3] = b[i];
        }
        for (int col = 0; col < 3; col++) {
            int piv = col;
            for (int r = col; r < 3; r++) {
                if (std::abs(m[r][col]) > std::abs(m[piv][col]))
                    piv = r;
            }
            if (piv != col)
                std::swap(m[piv], m[col]);
            double div = m[col][col];
            if (std::abs(div) < 1e-12)
                div = (div >= 0.0) ? 1e-12 : -1e-12;
            const double inv = 1.0 / div;
            for (int c = col; c < 4; c++)
                m[col][c] *= inv;
            for (int r = 0; r < 3; r++) {
                if (r == col)
                    continue;
                const double f = m[r][col];
                if (f == 0.0)
                    continue;
                for (int c = col; c < 4; c++)
                    m[r][c] -= f * m[col][c];
            }
        }
        return {m[0][3], m[1][3], m[2][3]};
    }

    void update(const core::MoveSummary& sum) {
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
        for (int k = 0; k < 4; k++)
            present[k] = (sum.cnt[k] > 0 && sum.vmax[k] > 0) ? 1 : 0;

        // Utilities (log space) / tau.
        std::array<double, 4> u_over_tau{};
        for (int k = 0; k < 4; k++)
            u_over_tau[k] = 0.0;
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
            if (present[c] && sum.action_value == sum.vmax[c] && sum.cnt[c] > 0) {
                g = p_cat[c] / static_cast<double>(sum.cnt[c]);
            } else {
                g = 0.0;
            }
        } else {
            g = 0.0;
        }

        double gamma = 1.0;
        if (g > 0.0) {
            const double p_total = eps * u_rand + (1.0 - eps) * g;
            if (p_total > 0.0)
                gamma = (eps * u_rand) / p_total;
            else
                gamma = 1.0;
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
                lambda_new[i][j] += h[i][j];
            lambda_new[i][i] += jitter;
        }

        const std::array<double, 3> step = solve3(lambda_new, grad);
        for (int i = 0; i < 3; i++)
            mu[i] = clip(mu[i] - damping * step[i], -delta_clip, delta_clip);
        lambda = lambda_new;
    }

    ParamNorm mean_param_norm() const {
        return wnorm_eps_from_delta_eps({mu[0], mu[1], mu[2]}, eps);
    }
};

#if AHC061_EXP003_HAS_EIGEN
struct SoftmaxLaplaceEstimatorEigen {
    double tau = 0.2;
    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double damping = 1.0;
    double jitter = 1e-6;

    std::int64_t n_obs = 0;
    double gamma_sum = 0.0;

    Eigen::Matrix<double, 3, 1> mu = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 3> lambda = Eigen::Matrix<double, 3, 3>::Zero();

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(double prior_std, double eps0) {
        tau = 0.2;
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        gamma_sum = 0.0;
        mu.setZero();
        const double inv_var = 1.0 / (prior_std * prior_std);
        lambda.setZero();
        lambda.diagonal().setConstant(inv_var);
    }

    static inline void softmax4_masked(
        const std::array<double, 4>& u_over_tau,
        const std::array<std::uint8_t, 4>& present,
        std::array<double, 4>& out_p) {
        double mx = -1e300;
        bool any = false;
        for (int k = 0; k < 4; k++) {
            if (!present[k])
                continue;
            any = true;
            mx = std::max(mx, u_over_tau[k]);
        }
        if (!any) {
            out_p.fill(0.0);
            return;
        }
        double s = 0.0;
        std::array<double, 4> e{};
        e.fill(0.0);
        for (int k = 0; k < 4; k++) {
            if (!present[k])
                continue;
            e[k] = std::exp(u_over_tau[k] - mx);
            s += e[k];
        }
        if (!(s > 0.0)) {
            out_p.fill(0.0);
            return;
        }
        const double inv_s = 1.0 / s;
        for (int k = 0; k < 4; k++)
            out_p[k] = e[k] * inv_s;
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b)
            return;
        const int n = sum.n;
        if (n <= 0)
            return;

        if (sum.all_zero)
            return;

        std::array<std::uint8_t, 4> present{};
        for (int k = 0; k < 4; k++)
            present[k] = (sum.cnt[k] > 0 && sum.vmax[k] > 0) ? 1 : 0;

        std::array<double, 4> u_over_tau{};
        u_over_tau.fill(0.0);
        if (present[0])
            u_over_tau[0] = std::log(static_cast<double>(sum.vmax[0])) / tau;
        if (present[1])
            u_over_tau[1] = (mu(0) + std::log(static_cast<double>(sum.vmax[1]))) / tau;
        if (present[2])
            u_over_tau[2] = (mu(1) + std::log(static_cast<double>(sum.vmax[2]))) / tau;
        if (present[3])
            u_over_tau[3] = (mu(2) + std::log(static_cast<double>(sum.vmax[3]))) / tau;

        std::array<double, 4> p_cat{};
        softmax4_masked(u_over_tau, present, p_cat);

        double g = 0.0;
        if (0 <= sum.action_cat && sum.action_cat < 4) {
            const int c = sum.action_cat;
            if (present[c] && sum.action_value == sum.vmax[c] && sum.cnt[c] > 0) {
                g = p_cat[c] / static_cast<double>(sum.cnt[c]);
            } else {
                g = 0.0;
            }
        } else {
            g = 0.0;
        }

        double gamma = 1.0;
        if (g > 0.0) {
            const double u_rand = 1.0 / static_cast<double>(n);
            const double p_total = eps * u_rand + (1.0 - eps) * g;
            gamma = (p_total > 0.0) ? ((eps * u_rand) / p_total) : 1.0;
        } else {
            gamma = 1.0;
        }

        n_obs++;
        gamma_sum += gamma;
        eps = clip(gamma_sum / static_cast<double>(n_obs), eps_min, eps_max);

        const double w = 1.0 - gamma;
        if (w <= 1e-9)
            return;

        const double pb = p_cat[1];
        const double pc = p_cat[2];
        const double pd = p_cat[3];

        const double yb = (sum.action_cat == 1) ? 1.0 : 0.0;
        const double yc = (sum.action_cat == 2) ? 1.0 : 0.0;
        const double yd = (sum.action_cat == 3) ? 1.0 : 0.0;

        const double scale_g = w / tau;
        Eigen::Matrix<double, 3, 1> grad;
        grad << scale_g * (pb - yb), scale_g * (pc - yc), scale_g * (pd - yd);

        const double scale_h = w / (tau * tau);
        Eigen::Matrix<double, 3, 3> h = Eigen::Matrix<double, 3, 3>::Zero();
        h(0, 0) = scale_h * (pb - pb * pb);
        h(1, 1) = scale_h * (pc - pc * pc);
        h(2, 2) = scale_h * (pd - pd * pd);
        h(0, 1) = scale_h * (-pb * pc);
        h(0, 2) = scale_h * (-pb * pd);
        h(1, 2) = scale_h * (-pc * pd);
        h(1, 0) = h(0, 1);
        h(2, 0) = h(0, 2);
        h(2, 1) = h(1, 2);

        Eigen::Matrix<double, 3, 3> lambda_new = lambda + h;
        lambda_new.diagonal().array() += jitter;

        const Eigen::Matrix<double, 3, 1> step = lambda_new.ldlt().solve(grad);
        mu.noalias() -= damping * step;
        mu = mu.cwiseMax(-delta_clip).cwiseMin(delta_clip);
        lambda = lambda_new;
    }

    ParamNorm mean_param_norm() const {
        return wnorm_eps_from_delta_eps({mu(0), mu(1), mu(2)}, eps);
    }
};
#endif

struct IneqTruncGaussEstimator {
    // Inequality constraints in log-ratio space:
    // delta = (x_b-x_a, x_c-x_a, x_d-x_a), x_t = log w_t.
    // Maintain delta ~ N(mu, Sigma) and update deterministically with halfspace truncation + moment matching.
    double eps = 0.50;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double jitter = 1e-15;

    std::int64_t n_obs = 0;
    double q_random_sum = 0.0;  // sum of P(random | obs)

    std::array<double, 3> mu{};  // mean of delta
    std::array<std::array<double, 3>, 3> sigma{};  // covariance of delta

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(double prior_std, double eps0) {
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        q_random_sum = 0.0;
        mu.fill(0.0);
        const double v = prior_std * prior_std;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                sigma[i][j] = (i == j) ? v : 0.0;
        }
    }

    static inline double phi(double x) {
        static constexpr double INV_SQRT2PI = 0.39894228040143267793994605993438;
        return INV_SQRT2PI * std::exp(-0.5 * x * x);
    }

    static inline double sf(double x) {
        // 1 - Phi(x) = 0.5 * erfc(x / sqrt(2)). Use Mills ratio for large x.
        if (x > 38.0)
            return 0.0;
        if (x > 8.0) {
            const double p = phi(x);
            return (x > 0.0) ? (p / x) : 1.0;
        }
        return 0.5 * std::erfc(x / 1.4142135623730950488);
    }

    static inline double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    static inline std::array<double, 3> mat_vec3(const std::array<std::array<double, 3>, 3>& a, const std::array<double, 3>& x) {
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

    static inline int delta_idx_from_cat(int cat) {
        // MoveSummary action_cat: 0:A, 1:B, 2:C, 3:D.
        if (cat == 1)
            return 0;
        if (cat == 2)
            return 1;
        if (cat == 3)
            return 2;
        return -1;  // A or invalid
    }

    static inline std::array<double, 3> make_a_vec(int cat_i, int cat_j) {
        // Constraint: x_i - x_j >= b, with baseline x_A = 0 and deltas for (B,C,D).
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
            for (int j = 0; j < 3; j++) {
                sigma_out[i][j] -= sigma_a[i] * sigma_a[j] * coef;
            }
        }
        symmetrize3(sigma_out);
        for (int i = 0; i < 3; i++)
            sigma_out[i][i] += 1e-15;
        return true;
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero)
            return;

        // Greedy impossible if action category is not A-D (e.g., own territory L=U => -1).
        if (!(0 <= sum.action_cat && sum.action_cat < 4)) {
            n_obs++;
            q_random_sum += 1.0;
            eps = clip(q_random_sum / static_cast<double>(n_obs), eps_min, eps_max);
            return;
        }

        const int c_star = sum.action_cat;
        const int v_star = sum.action_value;
        if (v_star != sum.vmax[static_cast<std::size_t>(c_star)]) {
            n_obs++;
            q_random_sum += 1.0;
            eps = clip(q_random_sum / static_cast<double>(n_obs), eps_min, eps_max);
            return;
        }
        const int k_star = sum.cnt[static_cast<std::size_t>(c_star)];
        if (k_star <= 0)
            return;

        const double u_rand = 1.0 / static_cast<double>(sum.n);
        const double log_pr = std::log(std::max(1e-300, eps)) + std::log(u_rand);

        // Sequential constraints + conditional moments (ADF).
        std::array<double, 3> mu_g = mu;
        std::array<std::array<double, 3>, 3> sigma_g = sigma;

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
            cons[cons_n++] = {b, c};
        }
        std::sort(cons.begin(), cons.begin() + cons_n, [&](const auto& a, const auto& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        double log_p_cond = 0.0;
        bool ok = true;
        for (int i = 0; i < cons_n; i++) {
            const double b = cons[i].first;
            const int c_other = cons[i].second;
            const std::array<double, 3> a = make_a_vec(c_star, c_other);
            double p_i = 0.0;
            std::array<double, 3> mu2{};
            std::array<std::array<double, 3>, 3> sig2{};
            if (!trunc_halfspace_into(mu_g, sigma_g, a, b, p_i, mu2, sig2)) {
                ok = false;
                break;
            }
            log_p_cond += std::log(std::max(1e-300, p_i));
            mu_g = mu2;
            sigma_g = sig2;
        }

        double r_g = 0.0;
        if (ok) {
            const double log_pg = std::log(std::max(1e-300, 1.0 - eps)) + log_p_cond - std::log(static_cast<double>(k_star));
            const double d = log_pr - log_pg;
            if (d > 50.0)
                r_g = 0.0;
            else if (d < -50.0)
                r_g = 1.0;
            else
                r_g = 1.0 / (1.0 + std::exp(d));
        } else {
            r_g = 0.0;
        }

        const double q_random = 1.0 - r_g;
        n_obs++;
        q_random_sum += q_random;
        eps = clip(q_random_sum / static_cast<double>(n_obs), eps_min, eps_max);

        // Mixture moment matching: (random => prior) + (greedy => conditional).
        const double w1 = 1.0 - r_g;
        const double w2 = r_g;

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
            for (int j = 0; j < 3; j++) {
                sigma_new[i][j] = exx[i][j] - mu_new[i] * mu_new[j];
            }
        }
        symmetrize3(sigma_new);
        for (int i = 0; i < 3; i++)
            sigma_new[i][i] += jitter;

        for (int i = 0; i < 3; i++)
            mu[i] = clip(mu_new[i], -delta_clip, delta_clip);
        sigma = sigma_new;
    }

    ParamNorm mean_param_norm() const {
        // Use log-normal mean for ratios.
        const double vb = std::max(0.0, sigma[0][0]);
        const double vc = std::max(0.0, sigma[1][1]);
        const double vd = std::max(0.0, sigma[2][2]);
        const std::array<double, 3> d{mu[0] + 0.5 * vb, mu[1] + 0.5 * vc, mu[2] + 0.5 * vd};
        return wnorm_eps_from_delta_eps(d, eps);
    }
};

struct IneqTruncGaussBetaEpsEstimator {
    // "log w Gaussian + halfspace truncation ADF + epsilon Beta (linear likelihood)".
    // Work in identifiable log-ratio space:
    // delta = (x_b-x_a, x_c-x_a, x_d-x_a), x_t = log w_t.
    // Keep delta ~ N(mu, Sigma).
    // Keep u=(eps-0.1)/0.4 ~ Beta(a,b) to represent eps in [0.1,0.5].
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double jitter = 1e-15;

    std::array<double, 3> mu{};  // mean of delta
    std::array<std::array<double, 3>, 3> sigma{};  // covariance of delta

    double beta_a = 1.0;  // Beta(a,b) on u in [0,1]
    double beta_b = 1.0;

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(double prior_std, double eps0) {
        mu.fill(0.0);
        const double v = prior_std * prior_std;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                sigma[i][j] = (i == j) ? v : 0.0;
        }

        const double u0_raw = (clip(eps0, eps_min, eps_max) - 0.1) / 0.4;
        const double u0 = clip(u0_raw, 1e-3, 1.0 - 1e-3);
        constexpr double CONC = 2.0;  // weak prior concentration
        beta_a = std::max(1e-3, u0 * CONC);
        beta_b = std::max(1e-3, (1.0 - u0) * CONC);
    }

    double eps_mean() const {
        const double s = beta_a + beta_b;
        if (!(s > 0.0))
            return 0.3;
        const double u = beta_a / s;
        return clip(0.1 + 0.4 * u, eps_min, eps_max);
    }

    static inline double phi(double x) {
        static constexpr double INV_SQRT2PI = 0.39894228040143267793994605993438;
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

    static inline double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    static inline std::array<double, 3> mat_vec3(const std::array<std::array<double, 3>, 3>& a, const std::array<double, 3>& x) {
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
        // Update Beta(a,b) on u in [0,1] with likelihood l(u)=alpha+beta*u (approx).
        // Resulting posterior is a 2-component Beta mixture; moment-match back to one Beta.
        alpha = std::max(0.0, alpha);
        if (!std::isfinite(alpha) || !std::isfinite(beta))
            return;

        const double s = beta_a + beta_b;
        if (!(s > 0.0))
            return;

        // Ensure l(u) >= 0 at u=1 (numerical safety).
        if (alpha + beta < 0.0)
            beta = -alpha;

        double w0 = 0.0;
        double w1 = 0.0;
        double e0 = beta_a / s;
        double e0_2 = (beta_a * (beta_a + 1.0)) / (s * (s + 1.0));
        double e1 = e0;
        double e1_2 = e0_2;

        if (beta >= 0.0) {
            // mixture: alpha*Beta(a,b) + beta*(a/(a+b))*Beta(a+1,b)
            w0 = alpha;
            w1 = beta * (beta_a / s);
            const double s1 = s + 1.0;
            e1 = (beta_a + 1.0) / s1;
            e1_2 = ((beta_a + 1.0) * (beta_a + 2.0)) / (s1 * (s1 + 1.0));
        } else {
            // l(u) = (alpha+beta) + (-beta)*(1-u)
            const double alpha1 = std::max(0.0, alpha + beta);
            w0 = alpha1;
            w1 = (-beta) * (beta_b / s);
            const double s1 = s + 1.0;
            e1 = beta_a / s1;  // Beta(a,b+1)
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
        const double a_new = std::max(1e-3, m1 * k);
        const double b_new = std::max(1e-3, (1.0 - m1) * k);
        beta_a = a_new;
        beta_b = b_new;
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero) {
            // Uninformative: random and greedy are both uniform.
            return;
        }

        const int n = sum.n;
        const int c_star = sum.action_cat;
        const int v_star = sum.action_value;

        const bool greedy_possible = (0 <= c_star && c_star < 4) && (v_star == sum.vmax[static_cast<std::size_t>(c_star)]) && (sum.cnt[static_cast<std::size_t>(c_star)] > 0);
        const double g = greedy_possible ? static_cast<double>(sum.cnt[static_cast<std::size_t>(c_star)]) : 1.0;

        // Build constraints and run ADF truncation if possible.
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
                cons[cons_n++] = {b, c};
            }
            std::sort(cons.begin(), cons.begin() + cons_n, [&](const auto& a, const auto& b) {
                if (a.first != b.first)
                    return a.first > b.first;
                return a.second < b.second;
            });

            double log_m = 0.0;
            ok = true;
            for (int i = 0; i < cons_n; i++) {
                const double b = cons[i].first;
                const int c_other = cons[i].second;
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
                if (log_m < -700.0)
                    m = 0.0;
                else
                    m = std::exp(log_m);
            } else {
                m = 0.0;
            }
        } else {
            // greedy impossible => m=0.
            ok = false;
            m = 0.0;
        }

        // ---- epsilon Beta update (linear likelihood) ----
        const double inv_n = 1.0 / static_cast<double>(n);
        const double inv_g = 1.0 / std::max(1.0, g);
        const double alpha = 0.1 * inv_n + 0.9 * m * inv_g;
        double beta = 0.4 * (inv_n - m * inv_g);
        beta_update_linear(alpha, beta);

        // ---- delta Gaussian mixture update ----
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

    ParamNorm mean_param_norm() const {
        const double vb = std::max(0.0, sigma[0][0]);
        const double vc = std::max(0.0, sigma[1][1]);
        const double vd = std::max(0.0, sigma[2][2]);
        const std::array<double, 3> d{mu[0] + 0.5 * vb, mu[1] + 0.5 * vc, mu[2] + 0.5 * vd};
        return wnorm_eps_from_delta_eps(d, eps_mean());
    }
};

static inline double safe_beta_mean_u(double a, double b) {
    const double s = a + b;
    if (!(s > 0.0))
        return 0.5;
    const double u = a / s;
    return std::min(1.0 - 1e-6, std::max(1e-6, u));
}

static inline double safe_beta_var_u(double a, double b) {
    const double s = a + b;
    if (!(s > 0.0))
        return 0.0;
    const double den = s * s * (s + 1.0);
    if (!(den > 0.0))
        return 0.0;
    return std::max(0.0, (a * b) / den);
}

static inline void beta_update_linear_moment(double alpha, double beta, double& a, double& b) {
    auto clip = [](double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); };
    alpha = std::max(0.0, alpha);
    if (!std::isfinite(alpha) || !std::isfinite(beta))
        return;

    const double s = a + b;
    if (!(s > 0.0))
        return;

    if (alpha + beta < 0.0)
        beta = -alpha;

    double w0 = 0.0;
    double w1 = 0.0;
    const double e0 = a / s;
    const double e0_2 = (a * (a + 1.0)) / (s * (s + 1.0));
    double e1 = e0;
    double e1_2 = e0_2;

    if (beta >= 0.0) {
        w0 = alpha;
        w1 = beta * (a / s);
        const double s1 = s + 1.0;
        e1 = (a + 1.0) / s1;
        e1_2 = ((a + 1.0) * (a + 2.0)) / (s1 * (s1 + 1.0));
    } else {
        const double alpha1 = std::max(0.0, alpha + beta);
        w0 = alpha1;
        w1 = (-beta) * (b / s);
        const double s1 = s + 1.0;
        e1 = a / s1;
        e1_2 = (a * (a + 1.0)) / (s1 * (s1 + 1.0));
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
    a = std::max(1e-3, m1 * k);
    b = std::max(1e-3, (1.0 - m1) * k);
}

static inline double greedy_prob_from_delta(const core::MoveSummary& sum, const std::array<double, 3>& delta) {
    if (!sum.action_in_b || sum.n <= 0)
        return 0.0;

    if (sum.all_zero)
        return 1.0 / static_cast<double>(sum.n);

    const double rb = std::exp(delta[0]);
    const double rc = std::exp(delta[1]);
    const double rd = std::exp(delta[2]);

    const double s0 = static_cast<double>(sum.vmax[0]);
    const double s1 = static_cast<double>(sum.vmax[1]) * rb;
    const double s2 = static_cast<double>(sum.vmax[2]) * rc;
    const double s3 = static_cast<double>(sum.vmax[3]) * rd;
    const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
    const double tol = 1e-12 * std::max(1.0, std::abs(mx));

    std::array<std::uint8_t, 4> is_best{};
    int k = 0;
    const std::array<double, 4> sc{s0, s1, s2, s3};
    for (int c = 0; c < 4; c++) {
        if (std::abs(sc[static_cast<std::size_t>(c)] - mx) <= tol) {
            is_best[static_cast<std::size_t>(c)] = 1;
            k += sum.cnt[static_cast<std::size_t>(c)];
        }
    }

    if (k <= 0 || sum.action_cat < 0 || sum.action_cat >= 4)
        return 0.0;
    const int c = sum.action_cat;
    if (!is_best[static_cast<std::size_t>(c)])
        return 0.0;
    if (sum.action_value != sum.vmax[static_cast<std::size_t>(c)])
        return 0.0;
    return 1.0 / static_cast<double>(k);
}

static inline std::array<double, 3> delta_from_param_norm(const ParamNorm& p) {
    const double wa = std::max(1e-12, p.w_norm[0]);
    const double wb = std::max(1e-12, p.w_norm[1]);
    const double wc = std::max(1e-12, p.w_norm[2]);
    const double wd = std::max(1e-12, p.w_norm[3]);
    return {std::log(wb / wa), std::log(wc / wa), std::log(wd / wa)};
}

static inline double obs_prob_from_param_norm(const core::MoveSummary& sum, const ParamNorm& p) {
    if (!sum.action_in_b || sum.n <= 0)
        return 1e-300;
    const double eps = std::min(0.5, std::max(0.1, p.eps));
    const double inv_n = 1.0 / static_cast<double>(sum.n);
    const std::array<double, 3> delta = delta_from_param_norm(p);
    const double pg = greedy_prob_from_delta(sum, delta);
    return std::max(1e-300, eps * inv_n + (1.0 - eps) * pg);
}

struct IneqTruncGaussBetaEpEstimator {
    IneqTruncGaussBetaEpsEstimator base{};
    std::int64_t n_updates = 0;
    int warmup_double_pass = 28;
    int steady_double_pass_interval = 4;
    double double_pass_uncertainty_threshold = 0.16;

    void reset(double prior_std, double eps0) {
        base.reset(prior_std, eps0);
        n_updates = 0;
    }

    double uncertainty_metric() const {
        const double tr_sigma =
            std::max(0.0, base.sigma[0][0]) + std::max(0.0, base.sigma[1][1]) + std::max(0.0, base.sigma[2][2]);
        const double u_var = safe_beta_var_u(base.beta_a, base.beta_b);
        const double eps_var = 0.16 * u_var;
        return tr_sigma + 4.0 * eps_var;
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero)
            return;
        n_updates++;

        const int n = sum.n;
        const int c_star = sum.action_cat;
        const int v_star = sum.action_value;

        const bool greedy_possible =
            (0 <= c_star && c_star < 4) &&
            (v_star == sum.vmax[static_cast<std::size_t>(c_star)]) &&
            (sum.cnt[static_cast<std::size_t>(c_star)] > 0);
        const double g = greedy_possible ? static_cast<double>(sum.cnt[static_cast<std::size_t>(c_star)]) : 1.0;

        std::array<std::pair<double, int>, 3> cons{};
        int cons_n = 0;
        if (greedy_possible) {
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
                cons[cons_n++] = {b, c};
            }
            std::sort(cons.begin(), cons.begin() + cons_n, [&](const auto& a, const auto& b) {
                if (a.first != b.first)
                    return a.first > b.first;
                return a.second < b.second;
            });
        }

        const double unc_now = uncertainty_metric();
        bool run_reverse_pass = (cons_n >= 2);
        if (run_reverse_pass && n_updates > warmup_double_pass) {
            const bool periodic =
                ((n_updates - warmup_double_pass) % std::max(1, steady_double_pass_interval) == 0);
            const bool uncertain = (unc_now >= double_pass_uncertainty_threshold);
            run_reverse_pass = periodic || uncertain;
        }

        auto run_pass = [&](bool reverse, double& out_m, std::array<double, 3>& out_mu, std::array<std::array<double, 3>, 3>& out_sigma) {
            out_mu = base.mu;
            out_sigma = base.sigma;
            if (cons_n == 0) {
                out_m = 1.0;
                return true;
            }
            double log_m = 0.0;
            for (int it = 0; it < cons_n; it++) {
                const int idx = reverse ? (cons_n - 1 - it) : it;
                const double b = cons[static_cast<std::size_t>(idx)].first;
                const int c_other = cons[static_cast<std::size_t>(idx)].second;
                const std::array<double, 3> a = IneqTruncGaussBetaEpsEstimator::make_a_vec(c_star, c_other);
                double p_i = 0.0;
                std::array<double, 3> mu2{};
                std::array<std::array<double, 3>, 3> sig2{};
                if (!IneqTruncGaussBetaEpsEstimator::trunc_halfspace_into(out_mu, out_sigma, a, b, p_i, mu2, sig2)) {
                    out_m = 0.0;
                    return false;
                }
                log_m += std::log(std::max(1e-300, p_i));
                out_mu = mu2;
                out_sigma = sig2;
            }
            out_m = (log_m < -700.0) ? 0.0 : std::exp(log_m);
            return out_m > 0.0;
        };

        double m = 0.0;
        bool ok = false;
        std::array<double, 3> mu_g = base.mu;
        std::array<std::array<double, 3>, 3> sigma_g = base.sigma;
        if (greedy_possible) {
            double m_f = 0.0;
            double m_r = 0.0;
            std::array<double, 3> mu_f{};
            std::array<double, 3> mu_r{};
            std::array<std::array<double, 3>, 3> sig_f{};
            std::array<std::array<double, 3>, 3> sig_r{};
            const bool ok_f = run_pass(false, m_f, mu_f, sig_f);
            const bool ok_r = run_reverse_pass ? run_pass(true, m_r, mu_r, sig_r) : false;

            if (ok_f && ok_r && run_reverse_pass) {
                ok = true;
                m = 0.5 * (m_f + m_r);
                const double wf_raw = std::max(1e-300, m_f);
                const double wr_raw = std::max(1e-300, m_r);
                const double inv_wr = 1.0 / (wf_raw + wr_raw);
                const double wf = wf_raw * inv_wr;
                const double wr = wr_raw * inv_wr;

                for (int i = 0; i < 3; i++)
                    mu_g[i] = wf * mu_f[i] + wr * mu_r[i];

                std::array<std::array<double, 3>, 3> exx{};
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        const double m1 = mu_f[i] * mu_f[j];
                        const double m2 = mu_r[i] * mu_r[j];
                        exx[i][j] = wf * (sig_f[i][j] + m1) + wr * (sig_r[i][j] + m2);
                    }
                }
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++)
                        sigma_g[i][j] = exx[i][j] - mu_g[i] * mu_g[j];
                }
                IneqTruncGaussBetaEpsEstimator::symmetrize3(sigma_g);
                for (int i = 0; i < 3; i++)
                    sigma_g[i][i] += base.jitter;
            } else if (ok_f) {
                ok = true;
                m = m_f;
                mu_g = mu_f;
                sigma_g = sig_f;
            } else if (ok_r && run_reverse_pass) {
                ok = true;
                m = m_r;
                mu_g = mu_r;
                sigma_g = sig_r;
            } else {
                ok = false;
                m = 0.0;
            }
        }

        const double inv_n = 1.0 / static_cast<double>(n);
        const double inv_g = 1.0 / std::max(1.0, g);
        const double alpha = 0.1 * inv_n + 0.9 * m * inv_g;
        const double beta = 0.4 * (inv_n - m * inv_g);
        base.beta_update_linear(alpha, beta);

        const double eps_bar = base.eps_mean();
        if (greedy_possible && ok && m > 0.0) {
            const double c0 = eps_bar * inv_n;
            const double c1 = (1.0 - eps_bar) * inv_g;
            const double z = c0 + c1 * m;
            const double lam = (z > 0.0) ? ((c1 * m) / z) : 0.0;

            const double w1 = 1.0 - lam;
            const double w2 = lam;

            std::array<double, 3> mu_new{};
            for (int i = 0; i < 3; i++)
                mu_new[i] = w1 * base.mu[i] + w2 * mu_g[i];

            std::array<std::array<double, 3>, 3> exx{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    const double m1 = base.mu[i] * base.mu[j];
                    const double m2 = mu_g[i] * mu_g[j];
                    exx[i][j] = w1 * (base.sigma[i][j] + m1) + w2 * (sigma_g[i][j] + m2);
                }
            }

            std::array<std::array<double, 3>, 3> sigma_new{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    sigma_new[i][j] = exx[i][j] - mu_new[i] * mu_new[j];
            }
            IneqTruncGaussBetaEpsEstimator::symmetrize3(sigma_new);
            for (int i = 0; i < 3; i++)
                sigma_new[i][i] += base.jitter;

            for (int i = 0; i < 3; i++)
                base.mu[i] = IneqTruncGaussBetaEpsEstimator::clip(mu_new[i], -base.delta_clip, base.delta_clip);
            base.sigma = sigma_new;
        }
    }

    ParamNorm mean_param_norm() const { return base.mean_param_norm(); }
};

struct RBPFDeltaBetaEstimator {
    static constexpr double DELTA_LO = -1.2039728043259361;  // log(0.3)
    static constexpr double DELTA_HI = 1.2039728043259361;   // log(1/0.3)
    static constexpr double MIN_PROB = 1e-300;

    int k_particles = 64;
    std::vector<std::array<double, 3>> delta{};
    std::vector<double> log_w{};
    std::vector<double> beta_a{};
    std::vector<double> beta_b{};

    double eps_min = 0.10;
    double eps_max = 0.50;
    core::XorShift64 rng{1};
    bool has_next_gauss = false;
    double next_gauss = 0.0;

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

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
        const double ang = 2.0 * 3.141592653589793238462643383279502884 * u2;
        const double z0 = r * std::cos(ang);
        const double z1 = r * std::sin(ang);
        has_next_gauss = true;
        next_gauss = z1;
        return z0;
    }

    void reset(std::uint64_t seed, int particles, double eps0) {
        rng = core::XorShift64(seed);
        has_next_gauss = false;
        next_gauss = 0.0;

        k_particles = std::max(1, particles);
        delta.assign(static_cast<std::size_t>(k_particles), std::array<double, 3>{});
        log_w.assign(static_cast<std::size_t>(k_particles), -std::log(static_cast<double>(k_particles)));
        beta_a.assign(static_cast<std::size_t>(k_particles), 1.0);
        beta_b.assign(static_cast<std::size_t>(k_particles), 1.0);

        const double u0_raw = (clip(eps0, eps_min, eps_max) - 0.1) / 0.4;
        const double u0 = clip(u0_raw, 1e-3, 1.0 - 1e-3);
        constexpr double CONC = 2.0;
        const double a0 = std::max(1e-3, u0 * CONC);
        const double b0 = std::max(1e-3, (1.0 - u0) * CONC);

        constexpr double PRIOR_STD = 0.42;
        for (int i = 0; i < k_particles; i++) {
            delta[static_cast<std::size_t>(i)] = {
                clip(PRIOR_STD * normal01(), DELTA_LO, DELTA_HI),
                clip(PRIOR_STD * normal01(), DELTA_LO, DELTA_HI),
                clip(PRIOR_STD * normal01(), DELTA_LO, DELTA_HI),
            };
            beta_a[static_cast<std::size_t>(i)] = a0;
            beta_b[static_cast<std::size_t>(i)] = b0;
        }
    }

    void renormalize(std::vector<double>& w, double* out_ess = nullptr) {
        const int k = k_particles;
        w.assign(static_cast<std::size_t>(k), 1.0 / static_cast<double>(k));
        if (k <= 0)
            return;
        double mx = log_w[0];
        for (int i = 1; i < k; i++)
            mx = std::max(mx, log_w[static_cast<std::size_t>(i)]);
        double s = 0.0;
        for (int i = 0; i < k; i++) {
            const double wi = std::exp(log_w[static_cast<std::size_t>(i)] - mx);
            w[static_cast<std::size_t>(i)] = wi;
            s += wi;
        }
        if (!(s > 0.0)) {
            const double uni = 1.0 / static_cast<double>(k);
            for (int i = 0; i < k; i++) {
                log_w[static_cast<std::size_t>(i)] = std::log(uni);
                w[static_cast<std::size_t>(i)] = uni;
            }
            if (out_ess)
                *out_ess = static_cast<double>(k);
            return;
        }
        const double log_z = mx + std::log(s);
        double ess_inv = 0.0;
        for (int i = 0; i < k; i++) {
            log_w[static_cast<std::size_t>(i)] -= log_z;
            const double wi = std::exp(log_w[static_cast<std::size_t>(i)]);
            w[static_cast<std::size_t>(i)] = wi;
            ess_inv += wi * wi;
        }
        if (out_ess)
            *out_ess = (ess_inv > 0.0) ? (1.0 / ess_inv) : 1.0;
    }

    void resample_and_move(const std::vector<double>& w) {
        const int k = k_particles;
        if (k <= 0)
            return;

        std::array<double, 3> mu{};
        mu.fill(0.0);
        std::array<double, 3> var{};
        var.fill(0.0);
        for (int i = 0; i < k; i++) {
            const double wi = w[static_cast<std::size_t>(i)];
            for (int d = 0; d < 3; d++)
                mu[static_cast<std::size_t>(d)] += wi * delta[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
        }
        for (int i = 0; i < k; i++) {
            const double wi = w[static_cast<std::size_t>(i)];
            for (int d = 0; d < 3; d++) {
                const double diff = delta[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] - mu[static_cast<std::size_t>(d)];
                var[static_cast<std::size_t>(d)] += wi * diff * diff;
            }
        }
        for (int d = 0; d < 3; d++)
            var[static_cast<std::size_t>(d)] = std::max(var[static_cast<std::size_t>(d)], 1e-6);

        std::vector<std::array<double, 3>> new_delta(static_cast<std::size_t>(k));
        std::vector<double> new_a(static_cast<std::size_t>(k), 1.0);
        std::vector<double> new_b(static_cast<std::size_t>(k), 1.0);

        int i = 0;
        double cdf = w[0];
        const double u0 = rng.next_double01() / static_cast<double>(k);
        for (int j = 0; j < k; j++) {
            const double u = u0 + static_cast<double>(j) / static_cast<double>(k);
            while (u > cdf && i + 1 < k) {
                i++;
                cdf += w[static_cast<std::size_t>(i)];
            }
            new_delta[static_cast<std::size_t>(j)] = delta[static_cast<std::size_t>(i)];
            new_a[static_cast<std::size_t>(j)] = beta_a[static_cast<std::size_t>(i)];
            new_b[static_cast<std::size_t>(j)] = beta_b[static_cast<std::size_t>(i)];
        }

        delta.swap(new_delta);
        beta_a.swap(new_a);
        beta_b.swap(new_b);
        const double log_uni = -std::log(static_cast<double>(k));
        for (int j = 0; j < k; j++)
            log_w[static_cast<std::size_t>(j)] = log_uni;

        constexpr double a_lw = 0.985;
        const double s_lw = std::sqrt(1.0 - a_lw * a_lw);
        for (int p = 0; p < k; p++) {
            for (int d = 0; d < 3; d++) {
                const double noise = s_lw * std::sqrt(var[static_cast<std::size_t>(d)]) * normal01();
                double x = a_lw * delta[static_cast<std::size_t>(p)][static_cast<std::size_t>(d)] + (1.0 - a_lw) * mu[static_cast<std::size_t>(d)] + noise;
                x = clip(x, DELTA_LO, DELTA_HI);
                delta[static_cast<std::size_t>(p)][static_cast<std::size_t>(d)] = x;
            }
        }
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        const int k = k_particles;
        if (k <= 0)
            return;

        const double inv_n = 1.0 / static_cast<double>(sum.n);
        for (int i = 0; i < k; i++) {
            const double pg = greedy_prob_from_delta(sum, delta[static_cast<std::size_t>(i)]);
            const double alpha = pg + 0.1 * (inv_n - pg);
            const double beta = 0.4 * (inv_n - pg);
            const double u_mean = safe_beta_mean_u(beta_a[static_cast<std::size_t>(i)], beta_b[static_cast<std::size_t>(i)]);
            double pred = alpha + beta * u_mean;
            pred = std::max(MIN_PROB, pred);
            log_w[static_cast<std::size_t>(i)] += std::log(pred);
            beta_update_linear_moment(alpha, beta, beta_a[static_cast<std::size_t>(i)], beta_b[static_cast<std::size_t>(i)]);
        }

        std::vector<double> w;
        double ess = 0.0;
        renormalize(w, &ess);
        if (ess < 0.52 * static_cast<double>(k))
            resample_and_move(w);
    }

    ParamNorm mean_param_norm() const {
        ParamNorm out{};
        out.w_norm.fill(0.25);
        out.eps = 0.3;

        if (k_particles <= 0)
            return out;

        double mx = log_w[0];
        for (int i = 1; i < k_particles; i++)
            mx = std::max(mx, log_w[static_cast<std::size_t>(i)]);
        double s = 0.0;
        for (int i = 0; i < k_particles; i++)
            s += std::exp(log_w[static_cast<std::size_t>(i)] - mx);
        if (!(s > 0.0))
            return out;

        out.w_norm.fill(0.0);
        out.eps = 0.0;
        const double inv_s = 1.0 / s;
        for (int i = 0; i < k_particles; i++) {
            const double wi = std::exp(log_w[static_cast<std::size_t>(i)] - mx) * inv_s;
            ParamNorm p = wnorm_eps_from_delta_eps(delta[static_cast<std::size_t>(i)], clip(0.1 + 0.4 * safe_beta_mean_u(beta_a[static_cast<std::size_t>(i)], beta_b[static_cast<std::size_t>(i)]), eps_min, eps_max));
            for (int d = 0; d < 4; d++)
                out.w_norm[static_cast<std::size_t>(d)] += wi * p.w_norm[static_cast<std::size_t>(d)];
            out.eps += wi * p.eps;
        }
        return out;
    }

    double uncertainty_metric() const {
        if (k_particles <= 0)
            return 0.0;
        double mx = log_w[0];
        for (int i = 1; i < k_particles; i++)
            mx = std::max(mx, log_w[static_cast<std::size_t>(i)]);
        double s = 0.0;
        for (int i = 0; i < k_particles; i++)
            s += std::exp(log_w[static_cast<std::size_t>(i)] - mx);
        if (!(s > 0.0))
            return 0.0;
        const double inv_s = 1.0 / s;
        std::array<double, 3> mu{};
        mu.fill(0.0);
        double eps_mu = 0.0;
        double eps_var_in = 0.0;
        for (int i = 0; i < k_particles; i++) {
            const double wi = std::exp(log_w[static_cast<std::size_t>(i)] - mx) * inv_s;
            for (int d = 0; d < 3; d++)
                mu[static_cast<std::size_t>(d)] += wi * delta[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
            const double u_mean = safe_beta_mean_u(beta_a[static_cast<std::size_t>(i)], beta_b[static_cast<std::size_t>(i)]);
            const double u_var = safe_beta_var_u(beta_a[static_cast<std::size_t>(i)], beta_b[static_cast<std::size_t>(i)]);
            eps_mu += wi * (0.1 + 0.4 * u_mean);
            eps_var_in += wi * (0.16 * u_var);
        }
        double tr = 0.0;
        double eps_var = eps_var_in;
        for (int i = 0; i < k_particles; i++) {
            const double wi = std::exp(log_w[static_cast<std::size_t>(i)] - mx) * inv_s;
            for (int d = 0; d < 3; d++) {
                const double diff = delta[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] - mu[static_cast<std::size_t>(d)];
                tr += wi * diff * diff;
            }
            const double u_mean = safe_beta_mean_u(beta_a[static_cast<std::size_t>(i)], beta_b[static_cast<std::size_t>(i)]);
            const double eps_i = 0.1 + 0.4 * u_mean;
            const double de = eps_i - eps_mu;
            eps_var += wi * de * de;
        }
        return tr + eps_var;
    }
};

struct HybridAdfRbpfEstimator {
    IneqTruncGaussBetaEpEstimator adf{};
    RBPFDeltaBetaEstimator rbpf{};
    int warmup_turns = 8;
    double gate_low = 0.08;
    double gate_high = 0.30;
    int rbpf_update_interval = 2;
    double score_decay = 0.985;
    double score_adf = 0.0;
    double score_rbpf = 0.0;
    int turn = 0;
    bool rbpf_ready = false;
    double mix_w = 0.0;

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(std::uint64_t seed, double prior_std, double eps0, int rbpf_particles) {
        adf.reset(prior_std, eps0);
        rbpf.reset(seed ^ 0x9e3779b97f4a7c15ULL, rbpf_particles, eps0);
        turn = 0;
        rbpf_ready = false;
        mix_w = 0.0;
        score_adf = 0.0;
        score_rbpf = 0.0;
    }

    void update(const core::MoveSummary& sum) {
        const ParamNorm adf_before = adf.mean_param_norm();
        const double p_adf = obs_prob_from_param_norm(sum, adf_before);
        const double p_rbpf_pred = rbpf_ready ? obs_prob_from_param_norm(sum, rbpf.mean_param_norm()) : p_adf;
        adf.update(sum);
        turn++;

        const double unc = adf.uncertainty_metric();
        const double w = clip((unc - gate_low) / std::max(1e-9, gate_high - gate_low), 0.0, 1.0);
        const bool do_rbpf = (turn <= warmup_turns) || (w > 1e-9 && ((turn % std::max(1, rbpf_update_interval) == 0) || w > 0.65));
        if (do_rbpf) {
            rbpf.update(sum);
            rbpf_ready = true;
        }

        score_adf = score_decay * score_adf + std::log(std::max(1e-300, p_adf));
        if (rbpf_ready) {
            score_rbpf = score_decay * score_rbpf + std::log(std::max(1e-300, p_rbpf_pred));
        } else {
            score_rbpf = score_decay * score_rbpf;
        }

        if (!rbpf_ready) {
            mix_w = 0.0;
            return;
        }

        const double score_gap = score_rbpf - score_adf;
        const double score_weight = 1.0 / (1.0 + std::exp(-(score_gap + 0.05) * 3.5));
        mix_w = w * score_weight;
        mix_w = clip(mix_w, 0.0, 0.65);
    }

    ParamNorm mean_param_norm() const {
        const ParamNorm a = adf.mean_param_norm();
        if (!(mix_w > 0.0) || !rbpf_ready)
            return a;
        const ParamNorm b = rbpf.mean_param_norm();
        ParamNorm out{};
        const double w_eps = mix_w;
        const double w_delta = std::min(0.25, 0.40 * mix_w);
        const double wa_delta = 1.0 - w_delta;
        for (int d = 0; d < 4; d++)
            out.w_norm[static_cast<std::size_t>(d)] = wa_delta * a.w_norm[static_cast<std::size_t>(d)] + w_delta * b.w_norm[static_cast<std::size_t>(d)];
        out.eps = (1.0 - w_eps) * a.eps + w_eps * b.eps;
        double sw = 0.0;
        for (int d = 0; d < 4; d++)
            sw += out.w_norm[static_cast<std::size_t>(d)];
        if (sw > 0.0) {
            const double inv = 1.0 / sw;
            for (int d = 0; d < 4; d++)
                out.w_norm[static_cast<std::size_t>(d)] *= inv;
        } else {
            out.w_norm = {0.25, 0.25, 0.25, 0.25};
        }
        return out;
    }
};

struct PGSoftmaxDiagEstimator {
    double tau = 0.12;
    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double jitter = 1e-6;

    std::int64_t n_obs = 0;
    double gamma_sum = 0.0;

    std::array<double, 3> mu{};
    std::array<double, 3> lambda_diag{};

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    static inline double sigmoid(double x) {
        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        const double z = std::exp(x);
        return z / (1.0 + z);
    }

    static inline double pg_omega(double eta) {
        const double a = std::abs(eta);
        if (a < 1e-6)
            return 0.25;
        return 0.5 * std::tanh(0.5 * a) / a;
    }

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
            e[static_cast<std::size_t>(k)] = std::exp(u_over_tau[static_cast<std::size_t>(k)] - mx);
            s += e[static_cast<std::size_t>(k)];
        }
        if (!(s > 0.0)) {
            out_p.fill(0.0);
            return;
        }
        const double inv_s = 1.0 / s;
        for (int k = 0; k < 4; k++)
            out_p[static_cast<std::size_t>(k)] = e[static_cast<std::size_t>(k)] * inv_s;
    }

    void reset(double prior_std, double eps0, double tau_in) {
        tau = std::max(1e-6, tau_in);
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        gamma_sum = 0.0;
        mu.fill(0.0);
        const double inv_var = 1.0 / (prior_std * prior_std);
        for (int i = 0; i < 3; i++)
            lambda_diag[static_cast<std::size_t>(i)] = inv_var;
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero)
            return;

        std::array<std::uint8_t, 4> present{};
        for (int k = 0; k < 4; k++)
            present[static_cast<std::size_t>(k)] = (sum.cnt[static_cast<std::size_t>(k)] > 0 && sum.vmax[static_cast<std::size_t>(k)] > 0) ? 1 : 0;

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

        double g = 0.0;
        if (0 <= sum.action_cat && sum.action_cat < 4) {
            const int c = sum.action_cat;
            if (present[static_cast<std::size_t>(c)] && sum.action_value == sum.vmax[static_cast<std::size_t>(c)] && sum.cnt[static_cast<std::size_t>(c)] > 0)
                g = p_cat[static_cast<std::size_t>(c)] / static_cast<double>(sum.cnt[static_cast<std::size_t>(c)]);
        }

        const double u_rand = 1.0 / static_cast<double>(sum.n);
        double gamma = 1.0;
        if (g > 0.0) {
            const double p_total = eps * u_rand + (1.0 - eps) * g;
            gamma = (p_total > 0.0) ? ((eps * u_rand) / p_total) : 1.0;
        }

        n_obs++;
        gamma_sum += gamma;
        eps = clip(gamma_sum / static_cast<double>(n_obs), eps_min, eps_max);

        const double w = 1.0 - gamma;
        if (w <= 1e-9)
            return;

        const double base_u = present[0] ? u_over_tau[0] : 0.0;
        for (int k = 1; k <= 3; k++) {
            if (!present[static_cast<std::size_t>(k)])
                continue;
            const int d = k - 1;
            const double eta = u_over_tau[static_cast<std::size_t>(k)] - base_u;
            const double p = sigmoid(eta);
            const double y = (sum.action_cat == k) ? 1.0 : 0.0;
            const double grad = (w / tau) * (p - y);
            const double h = (w / (tau * tau)) * pg_omega(eta);
            lambda_diag[static_cast<std::size_t>(d)] += h + jitter;
            mu[static_cast<std::size_t>(d)] = clip(mu[static_cast<std::size_t>(d)] - grad / lambda_diag[static_cast<std::size_t>(d)], -delta_clip, delta_clip);
        }
    }

    ParamNorm mean_param_norm() const { return wnorm_eps_from_delta_eps(mu, eps); }
};

struct LuceMMEstimator {
    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    std::int64_t n_obs = 0;
    double random_resp_sum = 0.0;

    std::array<double, 4> q{};
    std::array<double, 4> wins{};
    std::array<std::array<double, 4>, 4> matches{};

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(double eps0) {
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        random_resp_sum = 0.0;
        q.fill(1.0);
        wins.fill(1e-3);
        for (int i = 0; i < 4; i++)
            matches[static_cast<std::size_t>(i)].fill(0.0);
    }

    double greedy_prob(const core::MoveSummary& sum) const {
        if (!sum.action_in_b || sum.n <= 0)
            return 0.0;
        if (sum.all_zero)
            return 1.0 / static_cast<double>(sum.n);

        const double s0 = q[0] * static_cast<double>(sum.vmax[0]);
        const double s1 = q[1] * static_cast<double>(sum.vmax[1]);
        const double s2 = q[2] * static_cast<double>(sum.vmax[2]);
        const double s3 = q[3] * static_cast<double>(sum.vmax[3]);
        const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
        const double tol = 1e-12 * std::max(1.0, std::abs(mx));
        const std::array<double, 4> sc{s0, s1, s2, s3};
        std::array<std::uint8_t, 4> is_best{};
        int k = 0;
        for (int c = 0; c < 4; c++) {
            if (std::abs(sc[static_cast<std::size_t>(c)] - mx) <= tol) {
                is_best[static_cast<std::size_t>(c)] = 1;
                k += sum.cnt[static_cast<std::size_t>(c)];
            }
        }
        if (k <= 0 || sum.action_cat < 0 || sum.action_cat >= 4)
            return 0.0;
        const int c = sum.action_cat;
        if (!is_best[static_cast<std::size_t>(c)])
            return 0.0;
        if (sum.action_value != sum.vmax[static_cast<std::size_t>(c)])
            return 0.0;
        return 1.0 / static_cast<double>(k);
    }

    void mm_step() {
        std::array<double, 4> q_new = q;
        for (int i = 0; i < 4; i++) {
            double den = 0.0;
            for (int j = 0; j < 4; j++) {
                if (i == j)
                    continue;
                const double n_ij = matches[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (n_ij <= 0.0)
                    continue;
                den += n_ij / std::max(1e-9, q[static_cast<std::size_t>(i)] + q[static_cast<std::size_t>(j)]);
            }
            if (den > 1e-12)
                q_new[static_cast<std::size_t>(i)] = std::max(1e-6, wins[static_cast<std::size_t>(i)] / den);
        }
        double s = 0.0;
        for (int i = 0; i < 4; i++)
            s += q_new[static_cast<std::size_t>(i)];
        if (s > 0.0) {
            const double inv = 4.0 / s;
            for (int i = 0; i < 4; i++)
                q[static_cast<std::size_t>(i)] = std::max(1e-6, q_new[static_cast<std::size_t>(i)] * inv);
        }
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;

        const double inv_n = 1.0 / static_cast<double>(sum.n);
        const double pg = greedy_prob(sum);
        const double p_total = eps * inv_n + (1.0 - eps) * pg;
        const double gamma = (p_total > 0.0) ? ((eps * inv_n) / p_total) : 1.0;
        n_obs++;
        random_resp_sum += gamma;
        eps = clip(random_resp_sum / static_cast<double>(n_obs), eps_min, eps_max);

        const double w = 1.0 - gamma;
        if (w <= 1e-9)
            return;

        const int c_star = sum.action_cat;
        if (!(0 <= c_star && c_star < 4))
            return;
        if (sum.action_value != sum.vmax[static_cast<std::size_t>(c_star)])
            return;

        std::array<std::uint8_t, 4> present{};
        int present_n = 0;
        for (int c = 0; c < 4; c++) {
            const bool ok = (sum.cnt[static_cast<std::size_t>(c)] > 0 && sum.vmax[static_cast<std::size_t>(c)] > 0);
            present[static_cast<std::size_t>(c)] = ok ? 1 : 0;
            if (ok)
                present_n++;
        }
        if (!present[static_cast<std::size_t>(c_star)] || present_n <= 1)
            return;

        wins[static_cast<std::size_t>(c_star)] += w;
        for (int c = 0; c < 4; c++) {
            if (c == c_star || !present[static_cast<std::size_t>(c)])
                continue;
            matches[static_cast<std::size_t>(c_star)][static_cast<std::size_t>(c)] += w;
            matches[static_cast<std::size_t>(c)][static_cast<std::size_t>(c_star)] += w;
        }
        mm_step();
    }

    ParamNorm mean_param_norm() const {
        ParamNorm out{};
        double s = 0.0;
        for (int i = 0; i < 4; i++)
            s += std::max(1e-9, q[static_cast<std::size_t>(i)]);
        if (!(s > 0.0))
            s = 1.0;
        for (int i = 0; i < 4; i++)
            out.w_norm[static_cast<std::size_t>(i)] = std::max(1e-9, q[static_cast<std::size_t>(i)]) / s;
        out.eps = eps;
        return out;
    }
};

static inline double halton(std::uint64_t index_1based, std::uint32_t base) {
    double f = 1.0;
    double r = 0.0;
    std::uint64_t i = index_1based;
    while (i > 0) {
        f /= static_cast<double>(base);
        r += f * static_cast<double>(i % static_cast<std::uint64_t>(base));
        i /= static_cast<std::uint64_t>(base);
    }
    return r;
}

static inline double clamp01(double x) { return std::min(1.0, std::max(0.0, x)); }

struct FixedSupportISEstimator {
    // Proposal A: deterministic importance sampling with fixed support points (Halton).
    // theta = (delta_b, delta_c, delta_d, eps), delta = log ratios to A.
    static constexpr double MIN_PROB = 1e-300;
    static constexpr double DELTA_LO = -1.2039728043259361;  // log(0.3)
    static constexpr double DELTA_HI = 1.2039728043259361;   // log(1/0.3)

    const std::vector<std::array<double, 4>>* particles = nullptr;
    std::vector<double> log_w{};  // normalized: sum exp(log_w)=1
    ParamNorm cached_mean{};

    static std::vector<std::array<double, 4>> make_particles(int k_points) {
        if (k_points <= 0)
            k_points = 1;
        std::vector<std::array<double, 4>> out(static_cast<std::size_t>(k_points));
        for (int i = 0; i < k_points; i++) {
            const double u0 = halton(static_cast<std::uint64_t>(i + 1), 2);
            const double u1 = halton(static_cast<std::uint64_t>(i + 1), 3);
            const double u2 = halton(static_cast<std::uint64_t>(i + 1), 5);
            const double u3 = halton(static_cast<std::uint64_t>(i + 1), 7);
            const double d0 = DELTA_LO + (DELTA_HI - DELTA_LO) * clamp01(u0);
            const double d1 = DELTA_LO + (DELTA_HI - DELTA_LO) * clamp01(u1);
            const double d2 = DELTA_LO + (DELTA_HI - DELTA_LO) * clamp01(u2);
            const double eps = 0.1 + 0.4 * clamp01(u3);
            out[static_cast<std::size_t>(i)] = {d0, d1, d2, eps};
        }
        return out;
    }

    void reset(const std::vector<std::array<double, 4>>* particles_in) {
        particles = particles_in;
        const std::size_t k_points = (particles != nullptr) ? particles->size() : 1;
        log_w.assign(k_points, -std::log(static_cast<double>(k_points)));
        cached_mean = ParamNorm{};
        cached_mean.w_norm.fill(0.25);
        cached_mean.eps = 0.3;
        renormalize_and_cache_mean();
    }

    static inline double loglik(const core::MoveSummary& sum, const std::array<double, 4>& th) {
        if (!sum.action_in_b || sum.n <= 0)
            return std::log(MIN_PROB);

        const double rb = std::exp(th[0]);
        const double rc = std::exp(th[1]);
        const double rd = std::exp(th[2]);
        const double eps = th[3];

        const double prand = eps / static_cast<double>(sum.n);

        double pgreedy = 0.0;
        if (sum.all_zero) {
            pgreedy = 1.0 / static_cast<double>(sum.n);
        } else {
            const double s0 = static_cast<double>(sum.vmax[0]);
            const double s1 = static_cast<double>(sum.vmax[1]) * rb;
            const double s2 = static_cast<double>(sum.vmax[2]) * rc;
            const double s3 = static_cast<double>(sum.vmax[3]) * rd;
            const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
            const double tol = 1e-12 * std::max(1.0, std::abs(mx));
            std::array<std::uint8_t, 4> is_best{};
            int k = 0;
            const std::array<double, 4> sc{s0, s1, s2, s3};
            for (int c = 0; c < 4; c++) {
                if (std::abs(sc[c] - mx) <= tol) {
                    is_best[static_cast<std::size_t>(c)] = 1;
                    k += sum.cnt[static_cast<std::size_t>(c)];
                }
            }
            if (k > 0 && sum.action_cat >= 0 && sum.action_cat < 4) {
                const int c = sum.action_cat;
                if (is_best[static_cast<std::size_t>(c)] && sum.action_value == sum.vmax[static_cast<std::size_t>(c)])
                    pgreedy = 1.0 / static_cast<double>(k);
            }
        }

        const double p = std::max(MIN_PROB, prand + (1.0 - eps) * pgreedy);
        return std::log(p);
    }

    void update(const core::MoveSummary& sum) {
        if (particles == nullptr)
            return;
        const auto k_points = static_cast<int>(log_w.size());
        for (int i = 0; i < k_points; i++)
            log_w[static_cast<std::size_t>(i)] += loglik(sum, (*particles)[static_cast<std::size_t>(i)]);
        renormalize_and_cache_mean();
    }

    void renormalize_and_cache_mean() {
        if (particles == nullptr)
            return;
        const auto k_points = static_cast<int>(log_w.size());
        if (k_points <= 0)
            return;
        double mx = log_w[0];
        for (int i = 1; i < k_points; i++)
            mx = std::max(mx, log_w[static_cast<std::size_t>(i)]);
        double s = 0.0;
        for (int i = 0; i < k_points; i++)
            s += std::exp(log_w[static_cast<std::size_t>(i)] - mx);
        if (!(s > 0.0)) {
            const double inv = 1.0 / static_cast<double>(k_points);
            for (int i = 0; i < k_points; i++)
                log_w[static_cast<std::size_t>(i)] = std::log(inv);
            cached_mean = ParamNorm{};
            cached_mean.w_norm.fill(0.25);
            cached_mean.eps = 0.3;
            return;
        }
        const double log_z = mx + std::log(s);
        cached_mean = ParamNorm{};
        cached_mean.w_norm.fill(0.0);
        cached_mean.eps = 0.0;
        for (int i = 0; i < k_points; i++) {
            log_w[static_cast<std::size_t>(i)] -= log_z;
            const double wi = std::exp(log_w[static_cast<std::size_t>(i)]);
            const auto& th = (*particles)[static_cast<std::size_t>(i)];
            const ParamNorm pn = wnorm_eps_from_delta_eps({th[0], th[1], th[2]}, th[3]);
            for (int d = 0; d < 4; d++)
                cached_mean.w_norm[static_cast<std::size_t>(d)] += wi * pn.w_norm[static_cast<std::size_t>(d)];
            cached_mean.eps += wi * pn.eps;
        }
    }

    ParamNorm mean_param_norm() const { return cached_mean; }
};

struct SoftmaxFullLaplaceEstimator {
    // Proposal C: softmax over all legal moves (excluding full territory when non-full exists),
    // and online Laplace-ish update for delta with deterministic epsilon update.
    double tau = 0.2;
    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double damping = 1.0;
    double jitter = 1e-6;

    std::int64_t n_obs = 0;
    double gamma_sum = 0.0;

    std::array<double, 3> mu{};  // delta mean
    std::array<std::array<double, 3>, 3> lambda{};  // precision

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(double prior_std, double eps0) {
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        gamma_sum = 0.0;
        mu.fill(0.0);
        const double inv_var = 1.0 / (prior_std * prior_std);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                lambda[i][j] = (i == j) ? inv_var : 0.0;
        }
    }

    static inline std::array<double, 3> solve3(const std::array<std::array<double, 3>, 3>& a, const std::array<double, 3>& b) {
        return SoftmaxLaplaceEstimator::solve3(a, b);
    }

    void update_from_moves(const core::State& st_start, int p, int action_cell, const int* moves, int cnt) {
        if (cnt <= 0)
            return;

        bool action_in_b = false;
        int action_cat = -2;

        bool any_non_full = false;
        double mx = -1e300;
        for (int i = 0; i < cnt; i++) {
            const int idx = moves[i];
            const int cat = core::categorize_move_for_ai(st_start, p, idx);
            if (idx == action_cell) {
                action_in_b = true;
                action_cat = cat;
            }
            if (cat < 0)
                continue;
            any_non_full = true;
            const int v = st_start.value[idx];
            if (v <= 0)
                continue;
            const double base = std::log(static_cast<double>(v));
            double add = 0.0;
            if (cat == 1)
                add = mu[0];
            else if (cat == 2)
                add = mu[1];
            else if (cat == 3)
                add = mu[2];
            const double u = (base + add) / tau;
            mx = std::max(mx, u);
        }
        if (!action_in_b)
            return;
        if (!any_non_full)
            return;

        const double u_rand = 1.0 / static_cast<double>(cnt);
        const double prand = eps * u_rand;

        if (!(0 <= action_cat && action_cat < 4)) {
            const double gamma = 1.0;
            n_obs++;
            gamma_sum += gamma;
            eps = clip(gamma_sum / static_cast<double>(n_obs), eps_min, eps_max);
            return;
        }

        double denom = 0.0;
        std::array<double, 4> cat_mass{};
        cat_mass.fill(0.0);
        double chosen_mass = 0.0;
        for (int i = 0; i < cnt; i++) {
            const int idx = moves[i];
            const int cat = core::categorize_move_for_ai(st_start, p, idx);
            if (cat < 0)
                continue;
            const int v = st_start.value[idx];
            if (v <= 0)
                continue;
            const double base = std::log(static_cast<double>(v));
            double add = 0.0;
            if (cat == 1)
                add = mu[0];
            else if (cat == 2)
                add = mu[1];
            else if (cat == 3)
                add = mu[2];
            const double u = (base + add) / tau;
            const double e = std::exp(u - mx);
            denom += e;
            cat_mass[static_cast<std::size_t>(cat)] += e;
            if (idx == action_cell)
                chosen_mass = e;
        }
        if (!(denom > 0.0) || !(chosen_mass > 0.0)) {
            const double gamma = 1.0;
            n_obs++;
            gamma_sum += gamma;
            eps = clip(gamma_sum / static_cast<double>(n_obs), eps_min, eps_max);
            return;
        }

        const double p_chosen = chosen_mass / denom;
        const double p_total = std::max(1e-300, prand + (1.0 - eps) * p_chosen);
        const double gamma = prand / p_total;

        n_obs++;
        gamma_sum += gamma;
        eps = clip(gamma_sum / static_cast<double>(n_obs), eps_min, eps_max);

        const double w = 1.0 - gamma;
        if (!(w > 1e-9))
            return;

        const double p_b = cat_mass[1] / denom;
        const double p_c = cat_mass[2] / denom;
        const double p_d = cat_mass[3] / denom;

        const std::array<double, 3> p_param{p_b, p_c, p_d};
        const std::array<double, 3> y_param{
            (action_cat == 1) ? 1.0 : 0.0,
            (action_cat == 2) ? 1.0 : 0.0,
            (action_cat == 3) ? 1.0 : 0.0,
        };

        const double scale_g = w / tau;
        std::array<double, 3> grad{};
        for (int i = 0; i < 3; i++)
            grad[i] = scale_g * (p_param[i] - y_param[i]);

        const double scale_h = w / (tau * tau);
        std::array<std::array<double, 3>, 3> h{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double v = -p_param[i] * p_param[j];
                if (i == j)
                    v += p_param[i];
                h[i][j] = scale_h * v;
            }
        }

        std::array<std::array<double, 3>, 3> lambda_new{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                lambda_new[i][j] = lambda[i][j] + h[i][j];
            lambda_new[i][i] += jitter;
        }

        const std::array<double, 3> step = solve3(lambda_new, grad);
        for (int i = 0; i < 3; i++)
            mu[i] = clip(mu[i] - damping * step[i], -delta_clip, delta_clip);
        lambda = lambda_new;
    }

    ParamNorm mean_param_norm() const { return wnorm_eps_from_delta_eps(mu, eps); }
};

struct GridFilterEstimator {
    // Proposal D: deterministic 3D grid over delta=(log rb, log rc, log rd), with epsilon estimated
    // from "certain random" events and treated as a point estimate.
    static constexpr double MIN_PROB = 1e-300;
    static constexpr double DELTA_LO = -1.2039728043259361;  // log(0.3)
    static constexpr double DELTA_HI = 1.2039728043259361;   // log(1/0.3)

    int n = 0;
    std::vector<double> rb{};
    std::vector<double> rc{};
    std::vector<double> rd{};
    std::vector<double> log_w{};  // normalized

    double eps = 0.30;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double sum_i = 0.0;
    double sum_d = 0.0;
    double prior_strength = 5.0;
    double eps0 = 0.30;

    ParamNorm cached_mean{};

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }

    void reset(int grid_n, double eps0_in) {
        n = std::max(3, grid_n);
        eps0 = clip(eps0_in, eps_min, eps_max);
        eps = eps0;
        sum_i = 0.0;
        sum_d = 0.0;

        rb.resize(static_cast<std::size_t>(n));
        rc.resize(static_cast<std::size_t>(n));
        rd.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; i++) {
            const double u = static_cast<double>(i) / static_cast<double>(n - 1);
            const double d = DELTA_LO + (DELTA_HI - DELTA_LO) * u;
            const double r = std::exp(d);
            rb[static_cast<std::size_t>(i)] = r;
            rc[static_cast<std::size_t>(i)] = r;
            rd[static_cast<std::size_t>(i)] = r;
        }

        const std::size_t sz = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        log_w.assign(sz, -std::log(static_cast<double>(sz)));
        cached_mean = ParamNorm{};
        cached_mean.w_norm.fill(0.25);
        cached_mean.eps = eps;
        renormalize_and_cache_mean();
    }

    static inline bool certainly_random(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return false;
        if (sum.all_zero)
            return false;
        if (sum.action_cat == -1)
            return true;
        if (0 <= sum.action_cat && sum.action_cat < 4) {
            const int c = sum.action_cat;
            return sum.action_value != sum.vmax[static_cast<std::size_t>(c)];
        }
        return true;
    }

    static inline void count_types(
        const core::State& st_start,
        int p,
        const int* moves,
        int cnt,
        std::array<int, 4>& out_n_type,
        int& out_n_full) {
        out_n_type.fill(0);
        out_n_full = 0;
        for (int i = 0; i < cnt; i++) {
            const int idx = moves[i];
            const int cat = core::categorize_move_for_ai(st_start, p, idx);
            if (cat >= 0)
                out_n_type[static_cast<std::size_t>(cat)]++;
            else if (cat == -1)
                out_n_full++;
        }
    }

    void update_eps_from_moves(const core::State& st_start, int p, const int* moves, int cnt, const core::MoveSummary& sum) {
        if (!sum.action_in_b || cnt <= 0)
            return;
        if (sum.all_zero)
            return;
        std::array<int, 4> n_type{};
        int n_full = 0;
        count_types(st_start, p, moves, cnt, n_type, n_full);
        int d_size = 0;
        for (int c = 0; c < 4; c++)
            d_size += n_type[static_cast<std::size_t>(c)] - sum.cnt[static_cast<std::size_t>(c)];
        d_size += n_full;
        const double d_frac = static_cast<double>(std::max(0, d_size)) / static_cast<double>(cnt);
        if (!(d_frac > 1e-12))
            return;
        const double i_t = certainly_random(sum) ? 1.0 : 0.0;
        sum_i += i_t;
        sum_d += d_frac;
        const double num = sum_i + prior_strength * eps0;
        const double den = sum_d + prior_strength;
        eps = clip(num / std::max(1e-12, den), eps_min, eps_max);
    }

    void update_from_moves(const core::State& st_start, int p, const int* moves, int cnt, const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero)
            return;

        update_eps_from_moves(st_start, p, moves, cnt, sum);
        const std::size_t sz = log_w.size();
        if (sz == 0)
            return;

        for (int ib = 0; ib < n; ib++) {
            for (int ic = 0; ic < n; ic++) {
                for (int id = 0; id < n; id++) {
                    const std::size_t gidx =
                        static_cast<std::size_t>(ib) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n) +
                        static_cast<std::size_t>(ic) * static_cast<std::size_t>(n) +
                        static_cast<std::size_t>(id);
                    const double r_b = rb[static_cast<std::size_t>(ib)];
                    const double r_c = rc[static_cast<std::size_t>(ic)];
                    const double r_d = rd[static_cast<std::size_t>(id)];

                    const double prand = eps / static_cast<double>(sum.n);

                    const double s0 = static_cast<double>(sum.vmax[0]);
                    const double s1 = static_cast<double>(sum.vmax[1]) * r_b;
                    const double s2 = static_cast<double>(sum.vmax[2]) * r_c;
                    const double s3 = static_cast<double>(sum.vmax[3]) * r_d;
                    const double mx = std::max(std::max(s0, s1), std::max(s2, s3));
                    const double tol = 1e-12 * std::max(1.0, std::abs(mx));
                    std::array<std::uint8_t, 4> is_best{};
                    int k = 0;
                    const std::array<double, 4> sc{s0, s1, s2, s3};
                    for (int c = 0; c < 4; c++) {
                        if (std::abs(sc[c] - mx) <= tol) {
                            is_best[static_cast<std::size_t>(c)] = 1;
                            k += sum.cnt[static_cast<std::size_t>(c)];
                        }
                    }

                    double pgreedy = 0.0;
                    if (k > 0 && 0 <= sum.action_cat && sum.action_cat < 4) {
                        const int c = sum.action_cat;
                        if (is_best[static_cast<std::size_t>(c)] && sum.action_value == sum.vmax[static_cast<std::size_t>(c)])
                            pgreedy = 1.0 / static_cast<double>(k);
                    }
                    const double p = std::max(MIN_PROB, prand + (1.0 - eps) * pgreedy);
                    log_w[gidx] += std::log(p);
                }
            }
        }
        renormalize_and_cache_mean();
    }

    void renormalize_and_cache_mean() {
        const std::size_t sz = log_w.size();
        if (sz == 0)
            return;
        double mx = log_w[0];
        for (std::size_t i = 1; i < sz; i++)
            mx = std::max(mx, log_w[i]);
        double s = 0.0;
        for (std::size_t i = 0; i < sz; i++)
            s += std::exp(log_w[i] - mx);
        if (!(s > 0.0)) {
            const double inv = 1.0 / static_cast<double>(sz);
            for (std::size_t i = 0; i < sz; i++)
                log_w[i] = std::log(inv);
            cached_mean = ParamNorm{};
            cached_mean.w_norm.fill(0.25);
            cached_mean.eps = eps;
            return;
        }
        const double log_z = mx + std::log(s);
        cached_mean = ParamNorm{};
        cached_mean.w_norm.fill(0.0);
        cached_mean.eps = eps;
        for (int ib = 0; ib < n; ib++) {
            for (int ic = 0; ic < n; ic++) {
                for (int id = 0; id < n; id++) {
                    const std::size_t gidx =
                        static_cast<std::size_t>(ib) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n) +
                        static_cast<std::size_t>(ic) * static_cast<std::size_t>(n) +
                        static_cast<std::size_t>(id);
                    log_w[gidx] -= log_z;
                    const double wi = std::exp(log_w[gidx]);
                    const double r_b = rb[static_cast<std::size_t>(ib)];
                    const double r_c = rc[static_cast<std::size_t>(ic)];
                    const double r_d = rd[static_cast<std::size_t>(id)];
                    const double denom = 1.0 + r_b + r_c + r_d;
                    const double inv = (denom > 0.0) ? (1.0 / denom) : 1.0;
                    cached_mean.w_norm[0] += wi * (1.0 * inv);
                    cached_mean.w_norm[1] += wi * (r_b * inv);
                    cached_mean.w_norm[2] += wi * (r_c * inv);
                    cached_mean.w_norm[3] += wi * (r_d * inv);
                }
            }
        }
    }

    ParamNorm mean_param_norm() const { return cached_mean; }
};

#if AHC061_EXP003_HAS_EIGEN
struct IneqTruncGaussEstimatorEigen {
    double eps = 0.50;
    double eps_min = 0.10;
    double eps_max = 0.50;
    double delta_clip = 1.3;
    double jitter = 1e-15;

    std::int64_t n_obs = 0;
    double q_random_sum = 0.0;

    Eigen::Matrix<double, 3, 1> mu = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 3> sigma = Eigen::Matrix<double, 3, 3>::Zero();

    static inline double clip(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }
    static inline double phi(double x) {
        static constexpr double INV_SQRT2PI = 0.39894228040143267793994605993438;
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

    void reset(double prior_std, double eps0) {
        eps = clip(eps0, eps_min, eps_max);
        n_obs = 0;
        q_random_sum = 0.0;
        mu.setZero();
        sigma.setZero();
        sigma.diagonal().setConstant(prior_std * prior_std);
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

    static inline Eigen::Matrix<double, 3, 1> make_a_vec(int cat_i, int cat_j) {
        Eigen::Matrix<double, 3, 1> a = Eigen::Matrix<double, 3, 1>::Zero();
        const int ii = delta_idx_from_cat(cat_i);
        const int jj = delta_idx_from_cat(cat_j);
        if (ii >= 0)
            a(ii) += 1.0;
        if (jj >= 0)
            a(jj) -= 1.0;
        return a;
    }

    static inline bool trunc_halfspace_into(
        const Eigen::Matrix<double, 3, 1>& mu_in,
        const Eigen::Matrix<double, 3, 3>& sigma_in,
        const Eigen::Matrix<double, 3, 1>& a,
        double b,
        double& out_p,
        Eigen::Matrix<double, 3, 1>& mu_out,
        Eigen::Matrix<double, 3, 3>& sigma_out) {
        const double m = a.dot(mu_in);
        const Eigen::Matrix<double, 3, 1> sigma_a = sigma_in * a;
        const double s2 = a.dot(sigma_a);
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

        mu_out = mu_in + sigma_a * (lam / s);
        sigma_out = sigma_in - (sigma_a * sigma_a.transpose()) * (kappa / s2);
        sigma_out = 0.5 * (sigma_out + sigma_out.transpose());
        sigma_out.diagonal().array() += 1e-15;
        return true;
    }

    void update(const core::MoveSummary& sum) {
        if (!sum.action_in_b || sum.n <= 0)
            return;
        if (sum.all_zero)
            return;

        if (!(0 <= sum.action_cat && sum.action_cat < 4)) {
            n_obs++;
            q_random_sum += 1.0;
            eps = clip(q_random_sum / static_cast<double>(n_obs), eps_min, eps_max);
            return;
        }

        const int c_star = sum.action_cat;
        const int v_star = sum.action_value;
        if (v_star != sum.vmax[static_cast<std::size_t>(c_star)]) {
            n_obs++;
            q_random_sum += 1.0;
            eps = clip(q_random_sum / static_cast<double>(n_obs), eps_min, eps_max);
            return;
        }
        const int k_star = sum.cnt[static_cast<std::size_t>(c_star)];
        if (k_star <= 0)
            return;

        const double u_rand = 1.0 / static_cast<double>(sum.n);
        const double log_pr = std::log(std::max(1e-300, eps)) + std::log(u_rand);

        Eigen::Matrix<double, 3, 1> mu_g = mu;
        Eigen::Matrix<double, 3, 3> sigma_g = sigma;

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
            cons[cons_n++] = {b, c};
        }
        std::sort(cons.begin(), cons.begin() + cons_n, [&](const auto& a, const auto& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        double log_p_cond = 0.0;
        bool ok = true;
        for (int i = 0; i < cons_n; i++) {
            const double b = cons[i].first;
            const int c_other = cons[i].second;
            const Eigen::Matrix<double, 3, 1> a = make_a_vec(c_star, c_other);
            double p_i = 0.0;
            Eigen::Matrix<double, 3, 1> mu2;
            Eigen::Matrix<double, 3, 3> sig2;
            if (!trunc_halfspace_into(mu_g, sigma_g, a, b, p_i, mu2, sig2)) {
                ok = false;
                break;
            }
            log_p_cond += std::log(std::max(1e-300, p_i));
            mu_g = mu2;
            sigma_g = sig2;
        }

        double r_g = 0.0;
        if (ok) {
            const double log_pg = std::log(std::max(1e-300, 1.0 - eps)) + log_p_cond - std::log(static_cast<double>(k_star));
            const double d = log_pr - log_pg;
            if (d > 50.0)
                r_g = 0.0;
            else if (d < -50.0)
                r_g = 1.0;
            else
                r_g = 1.0 / (1.0 + std::exp(d));
        } else {
            r_g = 0.0;
        }

        const double q_random = 1.0 - r_g;
        n_obs++;
        q_random_sum += q_random;
        eps = clip(q_random_sum / static_cast<double>(n_obs), eps_min, eps_max);

        const double w1 = 1.0 - r_g;
        const double w2 = r_g;

        const Eigen::Matrix<double, 3, 1> mu_new = w1 * mu + w2 * mu_g;
        const Eigen::Matrix<double, 3, 3> exx =
            w1 * (sigma + mu * mu.transpose()) + w2 * (sigma_g + mu_g * mu_g.transpose());
        Eigen::Matrix<double, 3, 3> sigma_new = exx - mu_new * mu_new.transpose();
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose());
        sigma_new.diagonal().array() += 1e-15;

        mu = mu_new.cwiseMax(-delta_clip).cwiseMin(delta_clip);
        sigma = sigma_new;
    }

    ParamNorm mean_param_norm() const {
        const double vb = std::max(0.0, sigma(0, 0));
        const double vc = std::max(0.0, sigma(1, 1));
        const double vd = std::max(0.0, sigma(2, 2));
        const std::array<double, 3> d{mu(0) + 0.5 * vb, mu(1) + 0.5 * vc, mu(2) + 0.5 * vd};
        return wnorm_eps_from_delta_eps(d, eps);
    }
};
#endif

static inline std::int64_t pick_grain(std::int64_t n) {
    const std::int64_t threads = static_cast<std::int64_t>(at::get_num_threads());
    if (threads <= 1)
        return n;
    return std::max<std::int64_t>(1, n / threads);
}

static inline std::size_t idx(int m, int turn, int t_max) {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(t_max) + static_cast<std::size_t>(turn);
}

template <class Estimator, class InitFn, class UpdateFn, class MeanFn>
static void run_case_with_summary_estimator(
    std::uint64_t seed,
    int t_max,
    InitFn&& init_estimator,
    UpdateFn&& update_estimator,
    MeanFn&& mean_param,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<Estimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        init_estimator(est[p], p, seed);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            update_estimator(est[p], sum, p, turn);

            const ParamNorm est_p = mean_param(est[p]);
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

static void run_case(
    std::uint64_t seed,
    int t_max,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    // PF seed (match exp002::EnvInstance::reset_random).
    std::array<core::ParticleFilterSMC, core::M_MAX> pf{};
    const std::uint64_t pf_seed = core::compute_case_seed_for_pf(st) ^ (seed * 0x94d049bb133111ebULL);
    for (int p = 0; p < m; p++) {
        std::uint64_t s = pf_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL);
        s ^= 0x243f6a8885a308d3ULL;
        pf[p].reset(s);
    }

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        // player0: uniform random over legal moves.
        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        // opponents: AI-like (true params).
        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        // update PF + accumulate error (based on pre-move state).
        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            pf[p].update(sum);

            const ParamNorm est = estimate_param_mean_norm(pf[p]);
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        // apply turn.
        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

static void run_case_softmax_laplace(
    std::uint64_t seed,
    int t_max,
    double tau,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<SoftmaxLaplaceEstimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++) {
        est[p].reset(prior_std, eps0);
        est[p].tau = tau;
    }

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        // player0: uniform random over legal moves.
        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        // opponents: AI-like (true params).
        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update(sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

#if AHC061_EXP003_HAS_EIGEN
static void run_case_softmax_laplace_eigen(
    std::uint64_t seed,
    int t_max,
    double tau,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<SoftmaxLaplaceEstimatorEigen, core::M_MAX> est{};
    for (int p = 0; p < m; p++) {
        est[p].reset(prior_std, eps0);
        est[p].tau = tau;
    }

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        // player0: uniform random over legal moves.
        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        // opponents: AI-like (true params).
        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update(sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}
#endif

static void run_case_ineq_trunc_gauss(
    std::uint64_t seed,
    int t_max,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<IneqTruncGaussEstimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        est[p].reset(prior_std, eps0);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        // player0: uniform random over legal moves.
        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        // opponents: AI-like (true params).
        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update(sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

static void run_case_ineq_trunc_gauss_beta_eps(
    std::uint64_t seed,
    int t_max,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<IneqTruncGaussBetaEpsEstimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        est[p].reset(prior_std, eps0);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update(sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

#if AHC061_EXP003_HAS_EIGEN
static void run_case_ineq_trunc_gauss_eigen(
    std::uint64_t seed,
    int t_max,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<IneqTruncGaussEstimatorEigen, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        est[p].reset(prior_std, eps0);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        // player0: uniform random over legal moves.
        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        // opponents: AI-like (true params).
        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update(sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}
#endif

static void run_case_fixed_is(
    std::uint64_t seed,
    int t_max,
    int is_points,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    const auto particles = FixedSupportISEstimator::make_particles(is_points);
    std::array<FixedSupportISEstimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        est[p].reset(&particles);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update(sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

static void run_case_softmax_full_laplace(
    std::uint64_t seed,
    int t_max,
    double tau,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<SoftmaxFullLaplaceEstimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++) {
        est[p].reset(prior_std, eps0);
        est[p].tau = tau;
    }

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            est[p].update_from_moves(st, p, move_to[p], moves[p].data(), move_cnt[p]);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

static void run_case_grid_filter(
    std::uint64_t seed,
    int t_max,
    int grid_n,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    if (m < 2 || m > core::M_MAX)
        return;
    case_count[static_cast<std::size_t>(m)]++;

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<GridFilterEstimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        est[p].reset(grid_n, eps0);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    const int t = std::min(st.t_max, t_max);
    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        const std::size_t base = idx(m, turn, t_max);
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            est[p].update_from_moves(st, p, moves[p].data(), move_cnt[p], sum);

            const ParamNorm est_p = est[p].mean_param_norm();
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);

            double err_w_l1 = 0.0;
            for (int d = 0; d < 4; d++)
                err_w_l1 += std::abs(est_p.w_norm[static_cast<std::size_t>(d)] - tru.w_norm[static_cast<std::size_t>(d)]);
            const double err_eps_abs = std::abs(est_p.eps - tru.eps);
            const double err_mae5 = (err_w_l1 + err_eps_abs) / 5.0;

            sum_w_l1[base] += err_w_l1;
            sum_eps_abs[base] += err_eps_abs;
            sum_mae5[base] += err_mae5;
            count[base] += 1;
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }
}

static void run_case_ineq_trunc_gauss_beta_ep(
    std::uint64_t seed,
    int t_max,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    run_case_with_summary_estimator<IneqTruncGaussBetaEpEstimator>(
        seed,
        t_max,
        [prior_std, eps0](IneqTruncGaussBetaEpEstimator& e, int, std::uint64_t) { e.reset(prior_std, eps0); },
        [](IneqTruncGaussBetaEpEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const IneqTruncGaussBetaEpEstimator& e) { return e.mean_param_norm(); },
        sum_w_l1,
        sum_eps_abs,
        sum_mae5,
        count,
        case_count);
}

static void run_case_rbpf_delta_beta(
    std::uint64_t seed,
    int t_max,
    int rbpf_particles,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    run_case_with_summary_estimator<RBPFDeltaBetaEstimator>(
        seed,
        t_max,
        [rbpf_particles, eps0](RBPFDeltaBetaEstimator& e, int p, std::uint64_t case_seed) {
            const std::uint64_t s = case_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL) ^ 0x243f6a8885a308d3ULL;
            e.reset(s, rbpf_particles, eps0);
        },
        [](RBPFDeltaBetaEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const RBPFDeltaBetaEstimator& e) { return e.mean_param_norm(); },
        sum_w_l1,
        sum_eps_abs,
        sum_mae5,
        count,
        case_count);
}

static void run_case_hybrid_adf_rbpf(
    std::uint64_t seed,
    int t_max,
    double prior_std,
    double eps0,
    int rbpf_particles,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    run_case_with_summary_estimator<HybridAdfRbpfEstimator>(
        seed,
        t_max,
        [prior_std, eps0, rbpf_particles](HybridAdfRbpfEstimator& e, int p, std::uint64_t case_seed) {
            const std::uint64_t s = case_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x94d049bb133111ebULL);
            e.reset(s, prior_std, eps0, rbpf_particles);
        },
        [](HybridAdfRbpfEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const HybridAdfRbpfEstimator& e) { return e.mean_param_norm(); },
        sum_w_l1,
        sum_eps_abs,
        sum_mae5,
        count,
        case_count);
}

static void run_case_pg_softmax_diag(
    std::uint64_t seed,
    int t_max,
    double tau,
    double prior_std,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    run_case_with_summary_estimator<PGSoftmaxDiagEstimator>(
        seed,
        t_max,
        [tau, prior_std, eps0](PGSoftmaxDiagEstimator& e, int, std::uint64_t) { e.reset(prior_std, eps0, tau); },
        [](PGSoftmaxDiagEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const PGSoftmaxDiagEstimator& e) { return e.mean_param_norm(); },
        sum_w_l1,
        sum_eps_abs,
        sum_mae5,
        count,
        case_count);
}

static void run_case_luce_mm(
    std::uint64_t seed,
    int t_max,
    double eps0,
    std::vector<double>& sum_w_l1,
    std::vector<double>& sum_eps_abs,
    std::vector<double>& sum_mae5,
    std::vector<std::int64_t>& count,
    std::array<std::int64_t, core::M_MAX + 1>& case_count) {
    run_case_with_summary_estimator<LuceMMEstimator>(
        seed,
        t_max,
        [eps0](LuceMMEstimator& e, int, std::uint64_t) { e.reset(eps0); },
        [](LuceMMEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const LuceMMEstimator& e) { return e.mean_param_norm(); },
        sum_w_l1,
        sum_eps_abs,
        sum_mae5,
        count,
        case_count);
}

static inline void check_bench_inputs(torch::Tensor seeds, int t_max) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
}

template <class Estimator, class InitFn, class UpdateFn, class MeanFn>
static pybind11::dict trace_case_with_summary_estimator(
    std::uint64_t seed,
    int t_max,
    InitFn&& init_estimator,
    UpdateFn&& update_estimator,
    MeanFn&& mean_param) {
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");

    core::State st{};
    std::array<core::OpponentParam, core::M_MAX> opp_true{};
    core::generate_random_case(seed, st, opp_true);

    const int m = st.m;
    const int t = (m >= 2 && m <= core::M_MAX) ? std::min(st.t_max, t_max) : 0;

    auto true_param = torch::zeros({t, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto pred_param = torch::zeros({t, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto* true_ptr = true_param.data_ptr<double>();
    auto* pred_ptr = pred_param.data_ptr<double>();

    pybind11::dict out{};
    out["m"] = m;
    out["t"] = t;
    out["true_param"] = true_param;
    out["pred_param"] = pred_param;
    if (m < 2 || m > core::M_MAX || t <= 0) {
        return out;
    }

    core::XorShift64 rng_opp(seed ^ 0xdeadbeefcafebabeULL);
    core::XorShift64 rng_p0(seed ^ 0x1234567890abcdefULL);

    std::array<Estimator, core::M_MAX> est{};
    for (int p = 0; p < m; p++)
        init_estimator(est[p], p, seed, st);

    std::array<std::array<int, core::CELL_MAX>, core::M_MAX> moves{};
    std::array<int, core::M_MAX> move_cnt{};
    std::array<int, core::M_MAX> move_to{};

    for (int turn = 0; turn < t; turn++) {
        for (int p = 0; p < m; p++)
            move_cnt[p] = core::enumerate_legal_moves(st, p, moves[p]);

        {
            const int n0 = move_cnt[0];
            const int pick = (n0 >= 2) ? rng_p0.next_int(0, n0 - 1) : 0;
            move_to[0] = moves[0][pick];
        }

        for (int p = 1; p < m; p++) {
            const double r1 = rng_opp.next_double01();
            const double r2 = rng_opp.next_double01();
            move_to[p] = core::select_move_ai_like_from_moves(
                st,
                p,
                opp_true[static_cast<std::size_t>(p)],
                r1,
                r2,
                moves[p].data(),
                move_cnt[p]);
        }

        std::array<double, 5> tru_acc{};
        std::array<double, 5> pred_acc{};
        tru_acc.fill(0.0);
        pred_acc.fill(0.0);

        int n_opp = 0;
        for (int p = 1; p < m; p++) {
            const core::MoveSummary sum = core::summarize_ai_observation_from_moves(
                st,
                p,
                move_to[p],
                moves[p].data(),
                move_cnt[p]);
            update_estimator(est[p], sum, p, turn);

            const ParamNorm est_p = mean_param(est[p]);
            const ParamNorm tru = normalize_true_param(opp_true[static_cast<std::size_t>(p)]);
            for (int d = 0; d < 4; d++) {
                pred_acc[static_cast<std::size_t>(d)] += est_p.w_norm[static_cast<std::size_t>(d)];
                tru_acc[static_cast<std::size_t>(d)] += tru.w_norm[static_cast<std::size_t>(d)];
            }
            pred_acc[4] += est_p.eps;
            tru_acc[4] += tru.eps;
            n_opp++;
        }

        if (n_opp > 0) {
            const double inv = 1.0 / static_cast<double>(n_opp);
            for (int d = 0; d < 5; d++) {
                true_ptr[static_cast<std::size_t>(turn) * 5 + static_cast<std::size_t>(d)] =
                    tru_acc[static_cast<std::size_t>(d)] * inv;
                pred_ptr[static_cast<std::size_t>(turn) * 5 + static_cast<std::size_t>(d)] =
                    pred_acc[static_cast<std::size_t>(d)] * inv;
            }
        }

        std::array<int, core::M_MAX> move_to_all{};
        move_to_all.fill(0);
        for (int p = 0; p < m; p++)
            move_to_all[p] = move_to[p];
        core::apply_simultaneous_turn(st, move_to_all, nullptr);
    }

    return out;
}

template <class RunCaseFn>
static pybind11::dict bench_with_runner(torch::Tensor seeds, int t_max, RunCaseFn&& run_case_fn) {
    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_fn(seed, t_max, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_pf_estimation(torch::Tensor seeds, int t_max) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case(seed, t_max, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_softmax_laplace_estimation(
    torch::Tensor seeds,
    int t_max,
    double tau,
    double prior_std,
    double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_softmax_laplace(seed, t_max, tau, prior_std, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_ineq_trunc_gauss_estimation(
    torch::Tensor seeds,
    int t_max,
    double prior_std,
    double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_ineq_trunc_gauss(seed, t_max, prior_std, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_ineq_trunc_gauss_beta_eps_estimation(
    torch::Tensor seeds,
    int t_max,
    double prior_std,
    double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_ineq_trunc_gauss_beta_eps(seed, t_max, prior_std, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_ineq_trunc_gauss_beta_ep_estimation(
    torch::Tensor seeds,
    int t_max,
    double prior_std,
    double eps0) {
    check_bench_inputs(seeds, t_max);
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");
    return bench_with_runner(seeds, t_max, [prior_std, eps0](
                                               std::uint64_t seed,
                                               int t_max_in,
                                               std::vector<double>& sum_w_l1,
                                               std::vector<double>& sum_eps_abs,
                                               std::vector<double>& sum_mae5,
                                               std::vector<std::int64_t>& count,
                                               std::array<std::int64_t, core::M_MAX + 1>& case_count) {
        run_case_ineq_trunc_gauss_beta_ep(seed, t_max_in, prior_std, eps0, sum_w_l1, sum_eps_abs, sum_mae5, count, case_count);
    });
}

pybind11::dict bench_rbpf_delta_beta_estimation(
    torch::Tensor seeds,
    int t_max,
    int rbpf_particles,
    double eps0) {
    check_bench_inputs(seeds, t_max);
    TORCH_CHECK(rbpf_particles >= 1, "rbpf_particles must be >= 1");
    return bench_with_runner(seeds, t_max, [rbpf_particles, eps0](
                                               std::uint64_t seed,
                                               int t_max_in,
                                               std::vector<double>& sum_w_l1,
                                               std::vector<double>& sum_eps_abs,
                                               std::vector<double>& sum_mae5,
                                               std::vector<std::int64_t>& count,
                                               std::array<std::int64_t, core::M_MAX + 1>& case_count) {
        run_case_rbpf_delta_beta(seed, t_max_in, rbpf_particles, eps0, sum_w_l1, sum_eps_abs, sum_mae5, count, case_count);
    });
}

pybind11::dict bench_hybrid_adf_rbpf_estimation(
    torch::Tensor seeds,
    int t_max,
    double prior_std,
    double eps0,
    int rbpf_particles) {
    check_bench_inputs(seeds, t_max);
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");
    TORCH_CHECK(rbpf_particles >= 1, "rbpf_particles must be >= 1");
    return bench_with_runner(seeds, t_max, [prior_std, eps0, rbpf_particles](
                                               std::uint64_t seed,
                                               int t_max_in,
                                               std::vector<double>& sum_w_l1,
                                               std::vector<double>& sum_eps_abs,
                                               std::vector<double>& sum_mae5,
                                               std::vector<std::int64_t>& count,
                                               std::array<std::int64_t, core::M_MAX + 1>& case_count) {
        run_case_hybrid_adf_rbpf(seed, t_max_in, prior_std, eps0, rbpf_particles, sum_w_l1, sum_eps_abs, sum_mae5, count, case_count);
    });
}

pybind11::dict bench_pg_softmax_diag_estimation(
    torch::Tensor seeds,
    int t_max,
    double tau,
    double prior_std,
    double eps0) {
    check_bench_inputs(seeds, t_max);
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");
    return bench_with_runner(seeds, t_max, [tau, prior_std, eps0](
                                               std::uint64_t seed,
                                               int t_max_in,
                                               std::vector<double>& sum_w_l1,
                                               std::vector<double>& sum_eps_abs,
                                               std::vector<double>& sum_mae5,
                                               std::vector<std::int64_t>& count,
                                               std::array<std::int64_t, core::M_MAX + 1>& case_count) {
        run_case_pg_softmax_diag(seed, t_max_in, tau, prior_std, eps0, sum_w_l1, sum_eps_abs, sum_mae5, count, case_count);
    });
}

pybind11::dict bench_luce_mm_estimation(torch::Tensor seeds, int t_max, double eps0) {
    check_bench_inputs(seeds, t_max);
    return bench_with_runner(seeds, t_max, [eps0](
                                               std::uint64_t seed,
                                               int t_max_in,
                                               std::vector<double>& sum_w_l1,
                                               std::vector<double>& sum_eps_abs,
                                               std::vector<double>& sum_mae5,
                                               std::vector<std::int64_t>& count,
                                               std::array<std::int64_t, core::M_MAX + 1>& case_count) {
        run_case_luce_mm(seed, t_max_in, eps0, sum_w_l1, sum_eps_abs, sum_mae5, count, case_count);
    });
}

pybind11::dict trace_pf_estimation(std::int64_t seed, int t_max) {
    return trace_case_with_summary_estimator<core::ParticleFilterSMC>(
        static_cast<std::uint64_t>(seed),
        t_max,
        [](core::ParticleFilterSMC& e, int p, std::uint64_t seed_in, const core::State& st) {
            const std::uint64_t pf_seed = core::compute_case_seed_for_pf(st) ^ (seed_in * 0x94d049bb133111ebULL);
            std::uint64_t s = pf_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL);
            s ^= 0x243f6a8885a308d3ULL;
            e.reset(s);
        },
        [](core::ParticleFilterSMC& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const core::ParticleFilterSMC& e) { return estimate_param_mean_norm(e); });
}

pybind11::dict trace_ineq_trunc_gauss_beta_eps_estimation(
    std::int64_t seed,
    int t_max,
    double prior_std,
    double eps0) {
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");
    return trace_case_with_summary_estimator<IneqTruncGaussBetaEpsEstimator>(
        static_cast<std::uint64_t>(seed),
        t_max,
        [prior_std, eps0](IneqTruncGaussBetaEpsEstimator& e, int, std::uint64_t, const core::State&) { e.reset(prior_std, eps0); },
        [](IneqTruncGaussBetaEpsEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const IneqTruncGaussBetaEpsEstimator& e) { return e.mean_param_norm(); });
}

pybind11::dict trace_ineq_trunc_gauss_beta_ep_estimation(
    std::int64_t seed,
    int t_max,
    double prior_std,
    double eps0) {
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");
    return trace_case_with_summary_estimator<IneqTruncGaussBetaEpEstimator>(
        static_cast<std::uint64_t>(seed),
        t_max,
        [prior_std, eps0](IneqTruncGaussBetaEpEstimator& e, int, std::uint64_t, const core::State&) { e.reset(prior_std, eps0); },
        [](IneqTruncGaussBetaEpEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const IneqTruncGaussBetaEpEstimator& e) { return e.mean_param_norm(); });
}

pybind11::dict trace_hybrid_adf_rbpf_estimation(
    std::int64_t seed,
    int t_max,
    double prior_std,
    double eps0,
    int rbpf_particles) {
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");
    TORCH_CHECK(rbpf_particles >= 1, "rbpf_particles must be >= 1");
    return trace_case_with_summary_estimator<HybridAdfRbpfEstimator>(
        static_cast<std::uint64_t>(seed),
        t_max,
        [prior_std, eps0, rbpf_particles](HybridAdfRbpfEstimator& e, int, std::uint64_t seed_in, const core::State&) {
            e.reset(seed_in, prior_std, eps0, rbpf_particles);
        },
        [](HybridAdfRbpfEstimator& e, const core::MoveSummary& sum, int, int) { e.update(sum); },
        [](const HybridAdfRbpfEstimator& e) { return e.mean_param_norm(); });
}

pybind11::dict bench_fixed_is_estimation(torch::Tensor seeds, int t_max, int is_points) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(is_points >= 1, "is_points must be >= 1");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_fixed_is(seed, t_max, is_points, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_softmax_full_laplace_estimation(
    torch::Tensor seeds,
    int t_max,
    double tau,
    double prior_std,
    double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_softmax_full_laplace(seed, t_max, tau, prior_std, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_grid_filter_estimation(torch::Tensor seeds, int t_max, int grid_n, double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(grid_n >= 3, "grid_n must be >= 3");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_grid_filter(seed, t_max, grid_n, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

#if AHC061_EXP003_HAS_EIGEN
pybind11::dict bench_softmax_laplace_eigen_estimation(
    torch::Tensor seeds,
    int t_max,
    double tau,
    double prior_std,
    double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(tau > 0.0, "tau must be > 0");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_softmax_laplace_eigen(seed, t_max, tau, prior_std, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}

pybind11::dict bench_ineq_trunc_gauss_eigen_estimation(
    torch::Tensor seeds,
    int t_max,
    double prior_std,
    double eps0) {
    TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
    TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
    TORCH_CHECK(t_max >= 1, "t_max must be >= 1");
    TORCH_CHECK(prior_std > 0.0, "prior_std must be > 0");

    const auto n = static_cast<std::int64_t>(seeds.size(0));
    const auto total = static_cast<std::size_t>(core::M_MAX + 1) * static_cast<std::size_t>(t_max);

    auto sum_w_l1 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_eps_abs = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto sum_mae5 = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto count = torch::zeros({core::M_MAX + 1, t_max}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto case_count = torch::zeros({core::M_MAX + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* sum_w_ptr = sum_w_l1.data_ptr<double>();
    auto* sum_eps_ptr = sum_eps_abs.data_ptr<double>();
    auto* sum_mae_ptr = sum_mae5.data_ptr<double>();
    auto* count_ptr = count.data_ptr<std::int64_t>();
    auto* case_count_ptr = case_count.data_ptr<std::int64_t>();

    const auto* seed_ptr = seeds.data_ptr<std::int64_t>();

    std::mutex mu;
    const auto grain = pick_grain(n);
    at::parallel_for(0, n, grain, [&](std::int64_t begin, std::int64_t end) {
        std::vector<double> local_sum_w(total, 0.0);
        std::vector<double> local_sum_eps(total, 0.0);
        std::vector<double> local_sum_mae(total, 0.0);
        std::vector<std::int64_t> local_count(total, 0);
        std::array<std::int64_t, core::M_MAX + 1> local_case_count{};
        local_case_count.fill(0);

        for (std::int64_t i = begin; i < end; i++) {
            const std::uint64_t seed = static_cast<std::uint64_t>(seed_ptr[i]);
            run_case_ineq_trunc_gauss_eigen(seed, t_max, prior_std, eps0, local_sum_w, local_sum_eps, local_sum_mae, local_count, local_case_count);
        }

        std::lock_guard<std::mutex> lock(mu);
        for (std::size_t i = 0; i < total; i++) {
            sum_w_ptr[i] += local_sum_w[i];
            sum_eps_ptr[i] += local_sum_eps[i];
            sum_mae_ptr[i] += local_sum_mae[i];
            count_ptr[i] += local_count[i];
        }
        for (int m = 0; m <= core::M_MAX; m++)
            case_count_ptr[m] += local_case_count[static_cast<std::size_t>(m)];
    });

    pybind11::dict out{};
    out["sum_w_l1"] = sum_w_l1;
    out["sum_eps_abs"] = sum_eps_abs;
    out["sum_mae5"] = sum_mae5;
    out["count"] = count;
    out["case_count"] = case_count;
    return out;
}
#else
pybind11::dict bench_softmax_laplace_eigen_estimation(torch::Tensor, int, double, double, double) {
    TORCH_CHECK(false, "Eigen is not available. Install Eigen3 and/or set EIGEN3_INCLUDE_DIR.");
}
pybind11::dict bench_ineq_trunc_gauss_eigen_estimation(torch::Tensor, int, double, double) {
    TORCH_CHECK(false, "Eigen is not available. Install Eigen3 and/or set EIGEN3_INCLUDE_DIR.");
}
#endif

}  // namespace ahc061::exp003

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bench_pf_estimation", &ahc061::exp003::bench_pf_estimation, pybind11::arg("seeds"), pybind11::arg("t_max") = 100);
    m.def("trace_pf_estimation", &ahc061::exp003::trace_pf_estimation, pybind11::arg("seed"), pybind11::arg("t_max") = 100);
    m.def(
        "bench_fixed_is_estimation",
        &ahc061::exp003::bench_fixed_is_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("is_points") = 256);
    m.def(
        "bench_softmax_laplace_estimation",
        &ahc061::exp003::bench_softmax_laplace_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("tau") = 0.1,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_softmax_full_laplace_estimation",
        &ahc061::exp003::bench_softmax_full_laplace_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("tau") = 0.1,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_softmax_laplace_eigen_estimation",
        &ahc061::exp003::bench_softmax_laplace_eigen_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("tau") = 0.1,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_ineq_trunc_gauss_estimation",
        &ahc061::exp003::bench_ineq_trunc_gauss_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_ineq_trunc_gauss_beta_eps_estimation",
        &ahc061::exp003::bench_ineq_trunc_gauss_beta_eps_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "trace_ineq_trunc_gauss_beta_eps_estimation",
        &ahc061::exp003::trace_ineq_trunc_gauss_beta_eps_estimation,
        pybind11::arg("seed"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_ineq_trunc_gauss_beta_ep_estimation",
        &ahc061::exp003::bench_ineq_trunc_gauss_beta_ep_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "trace_ineq_trunc_gauss_beta_ep_estimation",
        &ahc061::exp003::trace_ineq_trunc_gauss_beta_ep_estimation,
        pybind11::arg("seed"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_rbpf_delta_beta_estimation",
        &ahc061::exp003::bench_rbpf_delta_beta_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("rbpf_particles") = 128,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_hybrid_adf_rbpf_estimation",
        &ahc061::exp003::bench_hybrid_adf_rbpf_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.35,
        pybind11::arg("eps0") = 0.30,
        pybind11::arg("rbpf_particles") = 64);
    m.def(
        "trace_hybrid_adf_rbpf_estimation",
        &ahc061::exp003::trace_hybrid_adf_rbpf_estimation,
        pybind11::arg("seed"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.35,
        pybind11::arg("eps0") = 0.30,
        pybind11::arg("rbpf_particles") = 64);
    m.def(
        "bench_pg_softmax_diag_estimation",
        &ahc061::exp003::bench_pg_softmax_diag_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("tau") = 0.11,
        pybind11::arg("prior_std") = 0.4,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_luce_mm_estimation",
        &ahc061::exp003::bench_luce_mm_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_ineq_trunc_gauss_eigen_estimation",
        &ahc061::exp003::bench_ineq_trunc_gauss_eigen_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("prior_std") = 0.5,
        pybind11::arg("eps0") = 0.50);
    m.def(
        "bench_grid_filter_estimation",
        &ahc061::exp003::bench_grid_filter_estimation,
        pybind11::arg("seeds"),
        pybind11::arg("t_max") = 100,
        pybind11::arg("grid_n") = 11,
        pybind11::arg("eps0") = 0.50);
    m.attr("M_MAX") = ahc061::exp002::M_MAX;
    m.attr("PF_PARTICLES") = ahc061::exp002::ParticleFilterSMC::P;
    m.attr("HAS_EIGEN") = AHC061_EXP003_HAS_EIGEN;
}
