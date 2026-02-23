#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ahc061/core/env.hpp"
#include "ahc061/core/feature_registry.hpp"

namespace ahc061::exp002 {

class BatchEnv {
public:
    explicit BatchEnv(int batch_size, std::string feature_id, bool pf_enabled)
        : batch_size_(batch_size),
          feature_id_(std::move(feature_id)),
          pf_enabled_(pf_enabled),
          envs_(static_cast<std::size_t>(batch_size)) {
        if (batch_size_ <= 0)
            throw std::runtime_error("batch_size must be positive");
        (void)get_feature_set(feature_id_);
    }

    int batch_size() const { return batch_size_; }
    bool pf_enabled() const { return pf_enabled_; }
    void set_pf_enabled(bool v) { pf_enabled_ = v; }

    std::string feature_id() const { return feature_id_; }
    void set_feature_id(std::string feature_id) {
        (void)get_feature_set(feature_id);
        feature_id_ = std::move(feature_id);
    }

    int feature_channels() const { return ahc061::exp002::feature_channels(feature_id_); }
    int feature_channels_of(std::string feature_id) const { return ahc061::exp002::feature_channels(feature_id); }

    static std::int64_t pick_grain(std::int64_t n) {
        const std::int64_t threads = static_cast<std::int64_t>(at::get_num_threads());
        if (threads <= 1)
            return n;
        // Each env instance has roughly uniform cost, so avoid oversplitting.
        return std::max<std::int64_t>(1, n / threads);
    }

    static void mode_to_update_flags(NextMode mode, bool pf_enabled, bool& update_pf, bool& update_a_softmax, bool& update_adf_beta) {
        update_pf = pf_enabled && (mode == NextMode::k_uniform_or_pf);
        update_a_softmax = (mode == NextMode::k_a_softmax_ut);
        update_adf_beta = (mode == NextMode::k_adf_beta);
    }

    static void write_aux_targets_for_env(
        const EnvInstance& env,
        float* dist_out,          // [M_MAX, CELL_MAX]
        float* param_out,         // [M_MAX, 5]
        std::uint8_t* valid_out)  // [M_MAX]
    {
        // Clear outputs.
        std::fill(dist_out, dist_out + M_MAX * CELL_MAX, 0.0f);
        std::fill(param_out, param_out + M_MAX * 5, 0.0f);
        std::fill(valid_out, valid_out + M_MAX, static_cast<std::uint8_t>(0));

        const int m = env.st.m;
        // old->new reorder mapping (match feature_common.hpp: sort opponents by descending score)
        std::array<int, M_MAX> old_to_new{};
        for (int p = 0; p < M_MAX; p++)
            old_to_new[p] = p;
        if (m >= 3) {
            std::array<int, M_MAX> opp_old{};
            int opp_n = 0;
            for (int p = 1; p < m; p++)
                opp_old[opp_n++] = p;
            std::sort(opp_old.begin(), opp_old.begin() + opp_n, [&](int a, int b) {
                const auto sa = env.score[static_cast<std::size_t>(a)];
                const auto sb = env.score[static_cast<std::size_t>(b)];
                if (sa != sb)
                    return sa > sb;
                return a < b;
            });
            for (int j = 0; j < opp_n; j++) {
                const int old_p = opp_old[static_cast<std::size_t>(j)];
                old_to_new[static_cast<std::size_t>(old_p)] = j + 1;
            }
        }

        std::array<float, CELL_MAX> dist_local{};
        std::array<int, CELL_MAX> moves_local{};
        for (int old_p = 1; old_p < m; old_p++) {
            const int new_p = old_to_new[static_cast<std::size_t>(old_p)];
            valid_out[new_p] = 1;

            const auto& op = env.opponent_param_true[static_cast<std::size_t>(old_p)];
            double sumw = op.wa + op.wb + op.wc + op.wd;
            if (!(sumw > 0.0))
                sumw = 1.0;
            {
                const std::ptrdiff_t base = static_cast<std::ptrdiff_t>(new_p) * 5;
                param_out[base + 0] = static_cast<float>(op.wa / sumw);
                param_out[base + 1] = static_cast<float>(op.wb / sumw);
                param_out[base + 2] = static_cast<float>(op.wc / sumw);
                param_out[base + 3] = static_cast<float>(op.wd / sumw);
                param_out[base + 4] = static_cast<float>(op.eps);
            }

            const int* moves_ptr = nullptr;
            int cnt = 0;
            const bool cache_ok = env.cache_valid && env.cache_turn == env.turn;
            if (cache_ok) {
                cnt = env.cache_move_cnt[static_cast<std::size_t>(old_p)];
                moves_ptr = env.cache_moves[static_cast<std::size_t>(old_p)].data();
            } else {
                cnt = enumerate_legal_moves(env.st, old_p, moves_local);
                moves_ptr = moves_local.data();
            }

            compute_move_dist_ai_like_from_moves(env.st, old_p, op, moves_ptr, cnt, dist_local.data());
            float* out = dist_out + static_cast<std::ptrdiff_t>(new_p) * CELL_MAX;
            std::copy(dist_local.begin(), dist_local.end(), out);
        }
    }

    void reset_random(torch::Tensor seeds) {
        TORCH_CHECK(seeds.device().is_cpu(), "seeds must be on CPU");
        TORCH_CHECK(seeds.dim() == 1, "seeds must be 1D");
        TORCH_CHECK(seeds.size(0) == batch_size_, "seeds size mismatch");
        TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
        const auto* seed_ptr = seeds.data_ptr<std::int64_t>();
        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                envs_[static_cast<std::size_t>(i)].reset_random(static_cast<std::uint64_t>(seed_ptr[i]));
            }
        });
    }

    void reset_from_tools(const std::vector<std::string>& paths) {
        if (static_cast<int>(paths.size()) != batch_size_)
            throw std::runtime_error("paths size mismatch");
        for (int i = 0; i < batch_size_; i++) {
            const auto tc = load_tools_case(paths[static_cast<std::size_t>(i)]);
            envs_[static_cast<std::size_t>(i)].reset_from_tools(tc, 0);
        }
    }

    void reset_from_tools_seeded(const std::vector<std::string>& paths, torch::Tensor pf_seeds_extra) {
        TORCH_CHECK(static_cast<int>(paths.size()) == batch_size_, "paths size mismatch");
        TORCH_CHECK(pf_seeds_extra.device().is_cpu(), "pf_seeds_extra must be on CPU");
        TORCH_CHECK(pf_seeds_extra.dim() == 1, "pf_seeds_extra must be 1D");
        TORCH_CHECK(pf_seeds_extra.size(0) == batch_size_, "pf_seeds_extra size mismatch");
        TORCH_CHECK(pf_seeds_extra.scalar_type() == torch::kInt64, "pf_seeds_extra must be int64");
        auto acc = pf_seeds_extra.accessor<std::int64_t, 1>();

        for (int i = 0; i < batch_size_; i++) {
            const auto tc = load_tools_case(paths[static_cast<std::size_t>(i)]);
            envs_[static_cast<std::size_t>(i)].reset_from_tools(tc, static_cast<std::uint64_t>(acc[i]));
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> observe() const {
        const int c = feature_channels();
        auto board = torch::empty(
            {batch_size_, c, N, N},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto mask = torch::empty(
            {batch_size_, CELL_MAX},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        observe_into(board, mask);
        return {board, mask};
    }

    void observe_into(torch::Tensor board, torch::Tensor mask) const {
        observe_into_feature(board, mask, feature_id_);
    }

    void observe_into_feature(torch::Tensor board, torch::Tensor mask, const std::string& feature_id) const {
        const int c = ahc061::exp002::feature_channels(feature_id);
        TORCH_CHECK(board.device().is_cpu(), "board must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board.scalar_type() == torch::kFloat32, "board must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board.is_contiguous(), "board must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board.sizes() == torch::IntArrayRef({batch_size_, c, N, N}), "board shape mismatch");
        TORCH_CHECK(mask.sizes() == torch::IntArrayRef({batch_size_, CELL_MAX}), "mask shape mismatch");

        auto* board_ptr = board.data_ptr<float>();
        auto* mask_ptr = mask.data_ptr<std::uint8_t>();
        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                const auto& env = envs_[static_cast<std::size_t>(i)];
                env.observe_feature_into(
                    feature_id,
                    board_ptr + static_cast<std::ptrdiff_t>(i) * c * CELL_MAX,
                    mask_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
                    pf_enabled_);
            }
        });
    }

    void observe_pair_into(
        torch::Tensor board_a,
        torch::Tensor board_b,
        torch::Tensor mask,
        const std::string& feature_id_a,
        const std::string& feature_id_b) const {
        const int ca = ahc061::exp002::feature_channels(feature_id_a);
        const int cb = ahc061::exp002::feature_channels(feature_id_b);

        TORCH_CHECK(board_a.device().is_cpu(), "board_a must be on CPU");
        TORCH_CHECK(board_b.device().is_cpu(), "board_b must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board_a.scalar_type() == torch::kFloat32, "board_a must be float32");
        TORCH_CHECK(board_b.scalar_type() == torch::kFloat32, "board_b must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board_a.is_contiguous(), "board_a must be contiguous");
        TORCH_CHECK(board_b.is_contiguous(), "board_b must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board_a.sizes() == torch::IntArrayRef({batch_size_, ca, N, N}), "board_a shape mismatch");
        TORCH_CHECK(board_b.sizes() == torch::IntArrayRef({batch_size_, cb, N, N}), "board_b shape mismatch");
        TORCH_CHECK(mask.sizes() == torch::IntArrayRef({batch_size_, CELL_MAX}), "mask shape mismatch");

        auto* a_ptr = board_a.data_ptr<float>();
        auto* b_ptr = board_b.data_ptr<float>();
        auto* m_ptr = mask.data_ptr<std::uint8_t>();

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                const auto& env = envs_[static_cast<std::size_t>(i)];
                env.observe_pair_into(
                    feature_id_a,
                    a_ptr + static_cast<std::ptrdiff_t>(i) * ca * CELL_MAX,
                    feature_id_b,
                    b_ptr + static_cast<std::ptrdiff_t>(i) * cb * CELL_MAX,
                    m_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
                    pf_enabled_);
            }
        });
    }

    void aux_targets_into(
        torch::Tensor move_dist,   // [B, M_MAX, CELL_MAX], float32
        torch::Tensor opp_param,   // [B, M_MAX, 5], float32 (w_norm[4], eps)
        torch::Tensor opp_valid) const {  // [B, M_MAX], uint8 (0/1), new-order, p=0 always 0
        TORCH_CHECK(move_dist.device().is_cpu(), "move_dist must be on CPU");
        TORCH_CHECK(opp_param.device().is_cpu(), "opp_param must be on CPU");
        TORCH_CHECK(opp_valid.device().is_cpu(), "opp_valid must be on CPU");
        TORCH_CHECK(move_dist.scalar_type() == torch::kFloat32, "move_dist must be float32");
        TORCH_CHECK(opp_param.scalar_type() == torch::kFloat32, "opp_param must be float32");
        TORCH_CHECK(opp_valid.scalar_type() == torch::kUInt8, "opp_valid must be uint8");
        TORCH_CHECK(move_dist.is_contiguous(), "move_dist must be contiguous");
        TORCH_CHECK(opp_param.is_contiguous(), "opp_param must be contiguous");
        TORCH_CHECK(opp_valid.is_contiguous(), "opp_valid must be contiguous");
        TORCH_CHECK(
            move_dist.sizes() == torch::IntArrayRef({batch_size_, M_MAX, CELL_MAX}),
            "move_dist shape mismatch");
        TORCH_CHECK(
            opp_param.sizes() == torch::IntArrayRef({batch_size_, M_MAX, 5}),
            "opp_param shape mismatch");
        TORCH_CHECK(
            opp_valid.sizes() == torch::IntArrayRef({batch_size_, M_MAX}),
            "opp_valid shape mismatch");

        auto* dist_ptr = move_dist.data_ptr<float>();
        auto* param_ptr = opp_param.data_ptr<float>();
        auto* valid_ptr = opp_valid.data_ptr<std::uint8_t>();

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                const auto& env = envs_[static_cast<std::size_t>(i)];
                write_aux_targets_for_env(
                    env,
                    dist_ptr + static_cast<std::ptrdiff_t>(i) * M_MAX * CELL_MAX,
                    param_ptr + static_cast<std::ptrdiff_t>(i) * M_MAX * 5,
                    valid_ptr + static_cast<std::ptrdiff_t>(i) * M_MAX);
            }
        });
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> aux_targets() const {
        auto move_dist = torch::empty(
            {batch_size_, M_MAX, CELL_MAX},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto opp_param = torch::empty(
            {batch_size_, M_MAX, 5},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto opp_valid = torch::empty(
            {batch_size_, M_MAX},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        aux_targets_into(move_dist, opp_param, opp_valid);
        return {move_dist, opp_param, opp_valid};
    }

    void step_into(torch::Tensor actions, torch::Tensor reward, torch::Tensor done) {
        TORCH_CHECK(actions.device().is_cpu(), "actions must be on CPU");
        TORCH_CHECK(actions.dim() == 1, "actions must be 1D");
        TORCH_CHECK(actions.size(0) == batch_size_, "actions size mismatch");
        TORCH_CHECK(actions.scalar_type() == torch::kInt64, "actions must be int64");
        TORCH_CHECK(actions.is_contiguous(), "actions must be contiguous");

        TORCH_CHECK(reward.device().is_cpu(), "reward must be on CPU");
        TORCH_CHECK(done.device().is_cpu(), "done must be on CPU");
        TORCH_CHECK(reward.dim() == 1, "reward must be 1D");
        TORCH_CHECK(done.dim() == 1, "done must be 1D");
        TORCH_CHECK(reward.size(0) == batch_size_, "reward size mismatch");
        TORCH_CHECK(done.size(0) == batch_size_, "done size mismatch");
        TORCH_CHECK(reward.scalar_type() == torch::kFloat32, "reward must be float32");
        TORCH_CHECK(done.scalar_type() == torch::kUInt8, "done must be uint8");
        TORCH_CHECK(reward.is_contiguous(), "reward must be contiguous");
        TORCH_CHECK(done.is_contiguous(), "done must be contiguous");

        const auto* act_ptr = actions.data_ptr<std::int64_t>();
        auto* r_ptr = reward.data_ptr<float>();
        auto* d_ptr = done.data_ptr<std::uint8_t>();
        const bool update_pf = pf_enabled_;
        const bool update_a_softmax = true;
        const bool update_adf_beta = true;

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                auto& env = envs_[static_cast<std::size_t>(i)];
                const auto [rew, is_done] =
                    env.step(static_cast<int>(act_ptr[i]), update_pf, update_a_softmax, update_adf_beta);
                r_ptr[i] = rew;
                d_ptr[i] = is_done ? 1 : 0;
            }
        });
    }

    std::tuple<torch::Tensor, torch::Tensor> step(torch::Tensor actions) {
        auto reward = torch::empty({batch_size_}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto done = torch::empty({batch_size_}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        step_into(actions, reward, done);
        return {reward, done};
    }

    void step_observe_into(
        torch::Tensor actions,
        torch::Tensor board,
        torch::Tensor mask,
        torch::Tensor reward,
        torch::Tensor done) {
        const int c = feature_channels();
        TORCH_CHECK(board.device().is_cpu(), "board must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board.scalar_type() == torch::kFloat32, "board must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board.is_contiguous(), "board must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board.sizes() == torch::IntArrayRef({batch_size_, c, N, N}), "board shape mismatch");
        TORCH_CHECK(mask.sizes() == torch::IntArrayRef({batch_size_, CELL_MAX}), "mask shape mismatch");

        TORCH_CHECK(actions.device().is_cpu(), "actions must be on CPU");
        TORCH_CHECK(actions.dim() == 1, "actions must be 1D");
        TORCH_CHECK(actions.size(0) == batch_size_, "actions size mismatch");
        TORCH_CHECK(actions.scalar_type() == torch::kInt64, "actions must be int64");
        TORCH_CHECK(actions.is_contiguous(), "actions must be contiguous");

        TORCH_CHECK(reward.device().is_cpu(), "reward must be on CPU");
        TORCH_CHECK(done.device().is_cpu(), "done must be on CPU");
        TORCH_CHECK(reward.dim() == 1, "reward must be 1D");
        TORCH_CHECK(done.dim() == 1, "done must be 1D");
        TORCH_CHECK(reward.size(0) == batch_size_, "reward size mismatch");
        TORCH_CHECK(done.size(0) == batch_size_, "done size mismatch");
        TORCH_CHECK(reward.scalar_type() == torch::kFloat32, "reward must be float32");
        TORCH_CHECK(done.scalar_type() == torch::kUInt8, "done must be uint8");
        TORCH_CHECK(reward.is_contiguous(), "reward must be contiguous");
        TORCH_CHECK(done.is_contiguous(), "done must be contiguous");

        const auto* act_ptr = actions.data_ptr<std::int64_t>();
        auto* r_ptr = reward.data_ptr<float>();
        auto* d_ptr = done.data_ptr<std::uint8_t>();
        auto* board_ptr = board.data_ptr<float>();
        auto* mask_ptr = mask.data_ptr<std::uint8_t>();
        const auto& fs = get_feature_set(feature_id_);
        bool update_pf = false;
        bool update_a_softmax = false;
        bool update_adf_beta = false;
        mode_to_update_flags(fs.next_mode, pf_enabled_, update_pf, update_a_softmax, update_adf_beta);

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                auto& env = envs_[static_cast<std::size_t>(i)];
                const auto [rew, is_done] =
                    env.step(static_cast<int>(act_ptr[i]), update_pf, update_a_softmax, update_adf_beta);
                r_ptr[i] = rew;
                d_ptr[i] = is_done ? 1 : 0;
                env.observe_feature_into(
                    feature_id_,
                    board_ptr + static_cast<std::ptrdiff_t>(i) * c * CELL_MAX,
                    mask_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
                    pf_enabled_);
            }
        });
    }

    void step_observe_aux_into(
        torch::Tensor actions,
        torch::Tensor board,
        torch::Tensor mask,
        torch::Tensor reward,
        torch::Tensor done,
        torch::Tensor move_dist,
        torch::Tensor opp_param,
        torch::Tensor opp_valid) {
        const int c = feature_channels();
        TORCH_CHECK(board.device().is_cpu(), "board must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board.scalar_type() == torch::kFloat32, "board must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board.is_contiguous(), "board must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board.sizes() == torch::IntArrayRef({batch_size_, c, N, N}), "board shape mismatch");
        TORCH_CHECK(mask.sizes() == torch::IntArrayRef({batch_size_, CELL_MAX}), "mask shape mismatch");

        TORCH_CHECK(actions.device().is_cpu(), "actions must be on CPU");
        TORCH_CHECK(actions.dim() == 1, "actions must be 1D");
        TORCH_CHECK(actions.size(0) == batch_size_, "actions size mismatch");
        TORCH_CHECK(actions.scalar_type() == torch::kInt64, "actions must be int64");
        TORCH_CHECK(actions.is_contiguous(), "actions must be contiguous");

        TORCH_CHECK(reward.device().is_cpu(), "reward must be on CPU");
        TORCH_CHECK(done.device().is_cpu(), "done must be on CPU");
        TORCH_CHECK(reward.dim() == 1, "reward must be 1D");
        TORCH_CHECK(done.dim() == 1, "done must be 1D");
        TORCH_CHECK(reward.size(0) == batch_size_, "reward size mismatch");
        TORCH_CHECK(done.size(0) == batch_size_, "done size mismatch");
        TORCH_CHECK(reward.scalar_type() == torch::kFloat32, "reward must be float32");
        TORCH_CHECK(done.scalar_type() == torch::kUInt8, "done must be uint8");
        TORCH_CHECK(reward.is_contiguous(), "reward must be contiguous");
        TORCH_CHECK(done.is_contiguous(), "done must be contiguous");

        TORCH_CHECK(move_dist.device().is_cpu(), "move_dist must be on CPU");
        TORCH_CHECK(opp_param.device().is_cpu(), "opp_param must be on CPU");
        TORCH_CHECK(opp_valid.device().is_cpu(), "opp_valid must be on CPU");
        TORCH_CHECK(move_dist.scalar_type() == torch::kFloat32, "move_dist must be float32");
        TORCH_CHECK(opp_param.scalar_type() == torch::kFloat32, "opp_param must be float32");
        TORCH_CHECK(opp_valid.scalar_type() == torch::kUInt8, "opp_valid must be uint8");
        TORCH_CHECK(move_dist.is_contiguous(), "move_dist must be contiguous");
        TORCH_CHECK(opp_param.is_contiguous(), "opp_param must be contiguous");
        TORCH_CHECK(opp_valid.is_contiguous(), "opp_valid must be contiguous");
        TORCH_CHECK(
            move_dist.sizes() == torch::IntArrayRef({batch_size_, M_MAX, CELL_MAX}),
            "move_dist shape mismatch");
        TORCH_CHECK(
            opp_param.sizes() == torch::IntArrayRef({batch_size_, M_MAX, 5}),
            "opp_param shape mismatch");
        TORCH_CHECK(
            opp_valid.sizes() == torch::IntArrayRef({batch_size_, M_MAX}),
            "opp_valid shape mismatch");

        const auto* act_ptr = actions.data_ptr<std::int64_t>();
        auto* r_ptr = reward.data_ptr<float>();
        auto* d_ptr = done.data_ptr<std::uint8_t>();
        auto* board_ptr = board.data_ptr<float>();
        auto* mask_ptr = mask.data_ptr<std::uint8_t>();
        auto* dist_ptr = move_dist.data_ptr<float>();
        auto* param_ptr = opp_param.data_ptr<float>();
        auto* valid_ptr = opp_valid.data_ptr<std::uint8_t>();
        const auto& fs = get_feature_set(feature_id_);
        bool update_pf = false;
        bool update_a_softmax = false;
        bool update_adf_beta = false;
        mode_to_update_flags(fs.next_mode, pf_enabled_, update_pf, update_a_softmax, update_adf_beta);

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                auto& env = envs_[static_cast<std::size_t>(i)];
                const auto [rew, is_done] =
                    env.step(static_cast<int>(act_ptr[i]), update_pf, update_a_softmax, update_adf_beta);
                r_ptr[i] = rew;
                d_ptr[i] = is_done ? 1 : 0;
                env.observe_feature_into(
                    feature_id_,
                    board_ptr + static_cast<std::ptrdiff_t>(i) * c * CELL_MAX,
                    mask_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
                    pf_enabled_);
                write_aux_targets_for_env(
                    env,
                    dist_ptr + static_cast<std::ptrdiff_t>(i) * M_MAX * CELL_MAX,
                    param_ptr + static_cast<std::ptrdiff_t>(i) * M_MAX * 5,
                    valid_ptr + static_cast<std::ptrdiff_t>(i) * M_MAX);
            }
        });
    }

    void step_observe_pair_into(
        torch::Tensor actions,
        torch::Tensor board_a,
        torch::Tensor board_b,
        torch::Tensor mask,
        torch::Tensor reward,
        torch::Tensor done,
        const std::string& feature_id_a,
        const std::string& feature_id_b) {
        const int ca = ahc061::exp002::feature_channels(feature_id_a);
        const int cb = ahc061::exp002::feature_channels(feature_id_b);

        TORCH_CHECK(board_a.device().is_cpu(), "board_a must be on CPU");
        TORCH_CHECK(board_b.device().is_cpu(), "board_b must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board_a.scalar_type() == torch::kFloat32, "board_a must be float32");
        TORCH_CHECK(board_b.scalar_type() == torch::kFloat32, "board_b must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board_a.is_contiguous(), "board_a must be contiguous");
        TORCH_CHECK(board_b.is_contiguous(), "board_b must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board_a.sizes() == torch::IntArrayRef({batch_size_, ca, N, N}), "board_a shape mismatch");
        TORCH_CHECK(board_b.sizes() == torch::IntArrayRef({batch_size_, cb, N, N}), "board_b shape mismatch");
        TORCH_CHECK(mask.sizes() == torch::IntArrayRef({batch_size_, CELL_MAX}), "mask shape mismatch");

        TORCH_CHECK(actions.device().is_cpu(), "actions must be on CPU");
        TORCH_CHECK(actions.dim() == 1, "actions must be 1D");
        TORCH_CHECK(actions.size(0) == batch_size_, "actions size mismatch");
        TORCH_CHECK(actions.scalar_type() == torch::kInt64, "actions must be int64");
        TORCH_CHECK(actions.is_contiguous(), "actions must be contiguous");

        TORCH_CHECK(reward.device().is_cpu(), "reward must be on CPU");
        TORCH_CHECK(done.device().is_cpu(), "done must be on CPU");
        TORCH_CHECK(reward.dim() == 1, "reward must be 1D");
        TORCH_CHECK(done.dim() == 1, "done must be 1D");
        TORCH_CHECK(reward.size(0) == batch_size_, "reward size mismatch");
        TORCH_CHECK(done.size(0) == batch_size_, "done size mismatch");
        TORCH_CHECK(reward.scalar_type() == torch::kFloat32, "reward must be float32");
        TORCH_CHECK(done.scalar_type() == torch::kUInt8, "done must be uint8");
        TORCH_CHECK(reward.is_contiguous(), "reward must be contiguous");
        TORCH_CHECK(done.is_contiguous(), "done must be contiguous");

        const auto* act_ptr = actions.data_ptr<std::int64_t>();
        auto* r_ptr = reward.data_ptr<float>();
        auto* d_ptr = done.data_ptr<std::uint8_t>();
        auto* a_ptr = board_a.data_ptr<float>();
        auto* b_ptr = board_b.data_ptr<float>();
        auto* m_ptr = mask.data_ptr<std::uint8_t>();
        const auto& fa = get_feature_set(feature_id_a);
        const auto& fb = get_feature_set(feature_id_b);
        bool update_pf_a = false;
        bool update_a_softmax_a = false;
        bool update_adf_beta_a = false;
        mode_to_update_flags(fa.next_mode, pf_enabled_, update_pf_a, update_a_softmax_a, update_adf_beta_a);
        bool update_pf_b = false;
        bool update_a_softmax_b = false;
        bool update_adf_beta_b = false;
        mode_to_update_flags(fb.next_mode, pf_enabled_, update_pf_b, update_a_softmax_b, update_adf_beta_b);
        const bool update_pf = update_pf_a || update_pf_b;
        const bool update_a_softmax = update_a_softmax_a || update_a_softmax_b;
        const bool update_adf_beta = update_adf_beta_a || update_adf_beta_b;

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                auto& env = envs_[static_cast<std::size_t>(i)];
                const auto [rew, is_done] =
                    env.step(static_cast<int>(act_ptr[i]), update_pf, update_a_softmax, update_adf_beta);
                r_ptr[i] = rew;
                d_ptr[i] = is_done ? 1 : 0;
                env.observe_pair_into(
                    feature_id_a,
                    a_ptr + static_cast<std::ptrdiff_t>(i) * ca * CELL_MAX,
                    feature_id_b,
                    b_ptr + static_cast<std::ptrdiff_t>(i) * cb * CELL_MAX,
                    m_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
                    pf_enabled_);
            }
        });
    }

    torch::Tensor pos0() const {
        auto out = torch::empty({batch_size_}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* a_ptr = out.data_ptr<std::int64_t>();
        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                a_ptr[i] = static_cast<std::int64_t>(envs_[static_cast<std::size_t>(i)].current_pos0_cell());
            }
        });
        return out;
    }

    torch::Tensor official_score() const {
        auto out = torch::empty({batch_size_}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* a_ptr = out.data_ptr<std::int64_t>();
        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                a_ptr[i] = envs_[static_cast<std::size_t>(i)].official_score();
            }
        });
        return out;
    }

    torch::Tensor score_s0_sa() const {
        auto out = torch::empty({batch_size_, 2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* a_ptr = out.data_ptr<std::int64_t>();
        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                const auto [s0, sa] = envs_[static_cast<std::size_t>(i)].score_s0_sa();
                a_ptr[static_cast<std::ptrdiff_t>(i) * 2 + 0] = s0;
                a_ptr[static_cast<std::ptrdiff_t>(i) * 2 + 1] = sa;
            }
        });
        return out;
    }

private:
    int batch_size_ = 0;
    std::string feature_id_{};
    bool pf_enabled_ = true;
    std::vector<EnvInstance> envs_{};
};

}  // namespace ahc061::exp002

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using ahc061::exp002::BatchEnv;

    m.def("feature_ids", []() { return ahc061::exp002::feature_ids(); });
    m.def("feature_channels", [](const std::string& feature_id) { return ahc061::exp002::feature_channels(feature_id); });
    m.def(
        "feature_submit_supported",
        [](const std::string& feature_id) { return ahc061::exp002::feature_submit_supported(feature_id); });
    pybind11::class_<BatchEnv>(m, "BatchEnv")
        .def(
            pybind11::init<int, std::string, bool>(),
            pybind11::arg("batch_size"),
            pybind11::arg("feature_id") = std::string("submit_v1"),
            pybind11::arg("pf_enabled") = true)
        .def_property_readonly("batch_size", &BatchEnv::batch_size)
        .def_property_readonly("pf_enabled", &BatchEnv::pf_enabled)
        .def("set_pf_enabled", &BatchEnv::set_pf_enabled, pybind11::arg("v"))
        .def_property_readonly("feature_id", &BatchEnv::feature_id)
        .def("set_feature_id", &BatchEnv::set_feature_id, pybind11::arg("feature_id"))
        .def("feature_channels", &BatchEnv::feature_channels)
        .def("feature_channels_of", &BatchEnv::feature_channels_of, pybind11::arg("feature_id"))
        .def("reset_random", &BatchEnv::reset_random, pybind11::arg("seeds"))
        .def("reset_from_tools", &BatchEnv::reset_from_tools, pybind11::arg("paths"))
        .def("reset_from_tools_seeded", &BatchEnv::reset_from_tools_seeded, pybind11::arg("paths"), pybind11::arg("pf_seeds_extra"))
        .def("observe", &BatchEnv::observe)
        .def("observe_into", &BatchEnv::observe_into, pybind11::arg("board"), pybind11::arg("mask"))
        .def(
            "observe_into_feature",
            &BatchEnv::observe_into_feature,
            pybind11::arg("board"),
            pybind11::arg("mask"),
            pybind11::arg("feature_id"))
        .def(
            "observe_pair_into",
            &BatchEnv::observe_pair_into,
            pybind11::arg("board_a"),
            pybind11::arg("board_b"),
            pybind11::arg("mask"),
            pybind11::arg("feature_id_a"),
            pybind11::arg("feature_id_b"))
        .def(
            "aux_targets_into",
            &BatchEnv::aux_targets_into,
            pybind11::arg("move_dist"),
            pybind11::arg("opp_param"),
            pybind11::arg("opp_valid"))
        .def("aux_targets", &BatchEnv::aux_targets)
        .def("step_into", &BatchEnv::step_into, pybind11::arg("actions"), pybind11::arg("reward"), pybind11::arg("done"))
        .def("step", &BatchEnv::step, pybind11::arg("actions"))
        .def(
            "step_observe_into",
            &BatchEnv::step_observe_into,
            pybind11::arg("actions"),
            pybind11::arg("board"),
            pybind11::arg("mask"),
            pybind11::arg("reward"),
            pybind11::arg("done"))
        .def(
            "step_observe_aux_into",
            &BatchEnv::step_observe_aux_into,
            pybind11::arg("actions"),
            pybind11::arg("board"),
            pybind11::arg("mask"),
            pybind11::arg("reward"),
            pybind11::arg("done"),
            pybind11::arg("move_dist"),
            pybind11::arg("opp_param"),
            pybind11::arg("opp_valid"))
        .def(
            "step_observe_pair_into",
            &BatchEnv::step_observe_pair_into,
            pybind11::arg("actions"),
            pybind11::arg("board_a"),
            pybind11::arg("board_b"),
            pybind11::arg("mask"),
            pybind11::arg("reward"),
            pybind11::arg("done"),
            pybind11::arg("feature_id_a"),
            pybind11::arg("feature_id_b"))
        .def("pos0", &BatchEnv::pos0)
        .def("official_score", &BatchEnv::official_score)
        .def("score_s0_sa", &BatchEnv::score_s0_sa);
}
