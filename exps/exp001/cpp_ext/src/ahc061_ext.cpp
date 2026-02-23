#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ahc061/core/env.hpp"

namespace ahc061::exp001 {

class BatchEnv {
public:
    explicit BatchEnv(int batch_size, bool pf_enabled)
        : batch_size_(batch_size), pf_enabled_(pf_enabled), envs_(static_cast<std::size_t>(batch_size)) {
        if (batch_size_ <= 0)
            throw std::runtime_error("batch_size must be positive");
    }

    int batch_size() const { return batch_size_; }
    bool pf_enabled() const { return pf_enabled_; }
    void set_pf_enabled(bool v) { pf_enabled_ = v; }

    int feature_channels() const { return FEATURE_C; }

    static std::int64_t pick_grain(std::int64_t n) {
        const std::int64_t threads = static_cast<std::int64_t>(at::get_num_threads());
        if (threads <= 1)
            return n;
        return std::max<std::int64_t>(1, n / (threads * 4));
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
        auto board = torch::empty(
            {batch_size_, FEATURE_C, N, N},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto mask = torch::empty(
            {batch_size_, CELL_MAX},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        observe_into(board, mask);
        return {board, mask};
    }

    void observe_into(torch::Tensor board, torch::Tensor mask) const {
        TORCH_CHECK(board.device().is_cpu(), "board must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board.scalar_type() == torch::kFloat32, "board must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board.is_contiguous(), "board must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board.sizes() == torch::IntArrayRef({batch_size_, FEATURE_C, N, N}), "board shape mismatch");
        TORCH_CHECK(mask.sizes() == torch::IntArrayRef({batch_size_, CELL_MAX}), "mask shape mismatch");

        auto* board_ptr = board.data_ptr<float>();
        auto* mask_ptr = mask.data_ptr<std::uint8_t>();
        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                const auto& env = envs_[static_cast<std::size_t>(i)];
                env.observe_into(
                    board_ptr + static_cast<std::ptrdiff_t>(i) * FEATURE_C * CELL_MAX,
                    mask_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
                    pf_enabled_);
            }
        });
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

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                auto& env = envs_[static_cast<std::size_t>(i)];
                const auto [rew, is_done] = env.step(static_cast<int>(act_ptr[i]), pf_enabled_);
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
        TORCH_CHECK(board.device().is_cpu(), "board must be on CPU");
        TORCH_CHECK(mask.device().is_cpu(), "mask must be on CPU");
        TORCH_CHECK(board.scalar_type() == torch::kFloat32, "board must be float32");
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
        TORCH_CHECK(board.is_contiguous(), "board must be contiguous");
        TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(board.sizes() == torch::IntArrayRef({batch_size_, FEATURE_C, N, N}), "board shape mismatch");
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

        const auto grain = pick_grain(batch_size_);
        at::parallel_for(0, batch_size_, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; i++) {
                auto& env = envs_[static_cast<std::size_t>(i)];
                const auto [rew, is_done] = env.step(static_cast<int>(act_ptr[i]), pf_enabled_);
                r_ptr[i] = rew;
                d_ptr[i] = is_done ? 1 : 0;
                env.observe_into(
                    board_ptr + static_cast<std::ptrdiff_t>(i) * FEATURE_C * CELL_MAX,
                    mask_ptr + static_cast<std::ptrdiff_t>(i) * CELL_MAX,
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
    bool pf_enabled_ = true;
    std::vector<EnvInstance> envs_{};
};

}  // namespace ahc061::exp001

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using ahc061::exp001::BatchEnv;

    m.def("feature_channels", []() { return ahc061::exp001::FEATURE_C; });
    pybind11::class_<BatchEnv>(m, "BatchEnv")
        .def(pybind11::init<int, bool>(), pybind11::arg("batch_size"), pybind11::arg("pf_enabled") = true)
        .def_property_readonly("batch_size", &BatchEnv::batch_size)
        .def_property_readonly("pf_enabled", &BatchEnv::pf_enabled)
        .def("set_pf_enabled", &BatchEnv::set_pf_enabled, pybind11::arg("v"))
        .def("feature_channels", &BatchEnv::feature_channels)
        .def("reset_random", &BatchEnv::reset_random, pybind11::arg("seeds"))
        .def("reset_from_tools", &BatchEnv::reset_from_tools, pybind11::arg("paths"))
        .def("reset_from_tools_seeded", &BatchEnv::reset_from_tools_seeded, pybind11::arg("paths"), pybind11::arg("pf_seeds_extra"))
        .def("observe", &BatchEnv::observe)
        .def("observe_into", &BatchEnv::observe_into, pybind11::arg("board"), pybind11::arg("mask"))
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
        .def("pos0", &BatchEnv::pos0)
        .def("official_score", &BatchEnv::official_score)
        .def("score_s0_sa", &BatchEnv::score_s0_sa);
}
