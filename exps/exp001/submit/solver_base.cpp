// exp001 submit (TorchScript, CPU-only inference)
// NOTE: This file is NOT a single-file submission by itself.
// Use the make_submit_torchscript.py script to bundle local headers + embed the model.

#include <torch/script.h>
#include <ATen/Parallel.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ahc061/core/features.hpp"
#include "ahc061/core/generator.hpp"
#include "ahc061/core/pf.hpp"
#include "ahc061/core/rules.hpp"
#include "ahc061/core/state.hpp"

#include "model_ts_base64.inc"

namespace ahc061::exp001 {

#ifndef AHC061_EXP001_TTA_MODE
// 0: no TTA (single forward)
// 1: TTA sum (default)    : argmax_a log(sum_k p_k(a))
// 2: TTA prod             : argmax_a sum_k log p_k(a)
#define AHC061_EXP001_TTA_MODE 1
#endif

static constexpr int kTtaMode = AHC061_EXP001_TTA_MODE;

static std::vector<std::uint8_t> base64_decode(std::string_view s) {
    static constexpr std::array<std::int16_t, 256> kDec = [] {
        std::array<std::int16_t, 256> t{};
        t.fill(-1);
        for (int c = 'A'; c <= 'Z'; c++) t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(c - 'A');
        for (int c = 'a'; c <= 'z'; c++) t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(26 + c - 'a');
        for (int c = '0'; c <= '9'; c++) t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(52 + c - '0');
        t[static_cast<std::uint8_t>('+')] = 62;
        t[static_cast<std::uint8_t>('/')] = 63;
        return t;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(s.size() * 3 / 4);

    int val = 0;
    int valb = -8;
    for (unsigned char uc : s) {
        if (uc == '=')
            break;
        if (uc == '\r' || uc == '\n' || uc == ' ' || uc == '\t')
            continue;
        const int d = static_cast<int>(kDec[uc]);
        if (d < 0)
            continue;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<std::uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static torch::jit::Module load_module_from_embedded_base64() {
    const std::vector<std::uint8_t> bytes = base64_decode(std::string_view(kModelTsBase64));
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ss.seekg(0);
    torch::jit::Module module = torch::jit::load(ss, torch::kCPU);
    module.eval();
    return module;
}

static inline float logaddexp(float a, float b) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    if (a == neg_inf)
        return b;
    if (b == neg_inf)
        return a;
    if (a < b)
        std::swap(a, b);
    // a >= b
    return a + static_cast<float>(std::log1p(std::exp(static_cast<double>(b - a))));
}

static void masked_log_softmax_100(const float* logits, const std::uint8_t* mask, float* out_logp) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    float max_v = neg_inf;
    for (int i = 0; i < CELL_MAX; i++) {
        if (!mask[static_cast<std::size_t>(i)])
            continue;
        max_v = std::max(max_v, logits[i]);
    }
    if (!(max_v > neg_inf / 2)) {
        // no legal moves; fall back to uniform (should not happen due to env-side fixup)
        const float lp = -static_cast<float>(std::log(static_cast<double>(CELL_MAX)));
        for (int i = 0; i < CELL_MAX; i++) out_logp[i] = lp;
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < CELL_MAX; i++) {
        if (!mask[static_cast<std::size_t>(i)])
            continue;
        sum += std::exp(static_cast<double>(logits[i] - max_v));
    }
    const float logz = max_v + static_cast<float>(std::log(sum));
    for (int i = 0; i < CELL_MAX; i++) {
        if (!mask[static_cast<std::size_t>(i)]) {
            out_logp[i] = neg_inf;
        } else {
            out_logp[i] = logits[i] - logz;
        }
    }
}

static const std::array<std::array<int, CELL_MAX>, 8>& tta_perm() {
    static const std::array<std::array<int, CELL_MAX>, 8> p = [] {
        std::array<std::array<int, CELL_MAX>, 8> out{};
        for (int flip = 0; flip < 2; flip++) {
            for (int rot = 0; rot < 4; rot++) {
                const int k = flip * 4 + rot;
                for (int x = 0; x < N; x++) {
                    for (int y = 0; y < N; y++) {
                        int tx = x;
                        int ty = y;
                        if (flip)
                            ty = N - 1 - ty;
                        for (int r = 0; r < rot; r++) {
                            const int nx = ty;
                            const int ny = N - 1 - tx;
                            tx = nx;
                            ty = ny;
                        }
                        out[static_cast<std::size_t>(k)][static_cast<std::size_t>(cell_index(x, y))] =
                            cell_index(tx, ty);
                    }
                }
            }
        }
        return out;
    }();
    return p;
}

static const std::array<std::array<int, CELL_MAX>, 8>& tta_inv_perm() {
    static const std::array<std::array<int, CELL_MAX>, 8> ip = [] {
        std::array<std::array<int, CELL_MAX>, 8> out{};
        const auto& p = tta_perm();
        for (int k = 0; k < 8; k++) {
            for (int idx = 0; idx < CELL_MAX; idx++) {
                const int idx_t = p[static_cast<std::size_t>(k)][static_cast<std::size_t>(idx)];
                out[static_cast<std::size_t>(k)][static_cast<std::size_t>(idx_t)] = idx;
            }
        }
        return out;
    }();
    return ip;
}

static int select_action(
    torch::jit::Module& module,
    const std::array<float, FEATURE_C * CELL_MAX>& board,
    const std::array<std::uint8_t, CELL_MAX>& mask) {
    c10::InferenceMode guard;
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    if (kTtaMode == 0) {
        torch::Tensor input = torch::from_blob(
            const_cast<float*>(board.data()),
            {1, FEATURE_C, N, N},
            opts);
        torch::Tensor logits = module.forward({input}).toTensor();
        if (logits.dim() == 2)
            logits = logits.squeeze(0);
        logits = logits.contiguous();
        const float* lp = logits.data_ptr<float>();

        int best = -1;
        float best_v = -std::numeric_limits<float>::infinity();
        for (int idx = 0; idx < CELL_MAX; idx++) {
            if (!mask[static_cast<std::size_t>(idx)])
                continue;
            const float v = lp[idx];
            if (best < 0 || v > best_v) {
                best = idx;
                best_v = v;
            }
        }
        if (best < 0)
            best = cell_index(0, 0);
        return best;
    }

    // TTA (D4): batch=8 forward once, aggregate masked probabilities in original coordinates.
    std::array<float, 8 * FEATURE_C * CELL_MAX> board_t{};
    std::array<std::uint8_t, 8 * CELL_MAX> mask_t{};
    const auto& p = tta_perm();
    const auto& ip = tta_inv_perm();

    for (int k = 0; k < 8; k++) {
        for (int idx = 0; idx < CELL_MAX; idx++) {
            const int idx_t = p[static_cast<std::size_t>(k)][static_cast<std::size_t>(idx)];
            mask_t[static_cast<std::size_t>(k) * CELL_MAX + static_cast<std::size_t>(idx_t)] =
                mask[static_cast<std::size_t>(idx)];
            for (int c = 0; c < FEATURE_C; c++) {
                board_t[(static_cast<std::size_t>(k) * FEATURE_C + static_cast<std::size_t>(c)) * CELL_MAX +
                        static_cast<std::size_t>(idx_t)] =
                    board[static_cast<std::size_t>(c) * CELL_MAX + static_cast<std::size_t>(idx)];
            }
        }
    }

    torch::Tensor input = torch::from_blob(
        board_t.data(),
        {8, FEATURE_C, N, N},
        opts);
    torch::Tensor logits = module.forward({input}).toTensor();
    logits = logits.contiguous(); // [8, 100]
    const float* lp = logits.data_ptr<float>();

    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::array<float, CELL_MAX> acc{};
    if (kTtaMode == 1) {
        acc.fill(neg_inf);
    } else {
        acc.fill(0.0f);
    }

    std::array<float, CELL_MAX> logp{};
    for (int k = 0; k < 8; k++) {
        const float* logits_k = lp + static_cast<std::size_t>(k) * CELL_MAX;
        const std::uint8_t* mask_k = mask_t.data() + static_cast<std::size_t>(k) * CELL_MAX;
        masked_log_softmax_100(logits_k, mask_k, logp.data());

        for (int idx_t = 0; idx_t < CELL_MAX; idx_t++) {
            const int idx = ip[static_cast<std::size_t>(k)][static_cast<std::size_t>(idx_t)];
            const float v = logp[static_cast<std::size_t>(idx_t)];
            if (kTtaMode == 1) {
                acc[static_cast<std::size_t>(idx)] = logaddexp(acc[static_cast<std::size_t>(idx)], v);
            } else {
                acc[static_cast<std::size_t>(idx)] += v;
            }
        }
    }

    int best = -1;
    float best_v = neg_inf;
    for (int idx = 0; idx < CELL_MAX; idx++) {
        if (!mask[static_cast<std::size_t>(idx)])
            continue;
        const float v = acc[static_cast<std::size_t>(idx)];
        if (best < 0 || v > best_v) {
            best = idx;
            best_v = v;
        }
    }
    if (best < 0)
        best = cell_index(0, 0);
    return best;
}

}  // namespace ahc061::exp001

int main() {
    using namespace std;
    using namespace ahc061::exp001;

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n = 0, m = 0, t_max = 0, u_max = 0;
    if (!(cin >> n >> m >> t_max >> u_max))
        return 0;
    if (n != N) {
        cerr << "[ERROR] unexpected N=" << n << '\n';
        return 0;
    }
    if (m < 2 || m > M_MAX) {
        cerr << "[ERROR] unexpected M=" << m << '\n';
        return 0;
    }

    State st{};
    st.m = m;
    st.t_max = t_max;
    st.u_max = u_max;

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            int v = 0;
            cin >> v;
            st.value[cell_index(x, y)] = v;
        }
    }

    for (int p = 0; p < m; p++) {
        int sx = 0, sy = 0;
        cin >> sx >> sy;
        st.ex[p] = static_cast<std::uint8_t>(sx);
        st.ey[p] = static_cast<std::uint8_t>(sy);
    }
    st.owner.fill(-1);
    st.level.fill(0);
    for (int p = 0; p < m; p++) {
        const int idx = cell_index(static_cast<int>(st.ex[p]), static_cast<int>(st.ey[p]));
        st.owner[idx] = static_cast<std::int8_t>(p);
        st.level[idx] = 1;
    }

    // PF init (deterministic from the case)
    const std::uint64_t base_seed = compute_case_seed_for_pf(st);
    std::array<ParticleFilterSMC, M_MAX> pf{};
    for (int p = 0; p < m; p++) {
        const std::uint64_t s = base_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL) ^ 0x243f6a8885a308d3ULL;
        pf[p].reset(s);
    }

    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    torch::jit::Module module = load_module_from_embedded_base64();

    int turn = 0;
    for (turn = 0; turn < t_max; turn++) {
        const State st_start = st;

        std::array<float, FEATURE_C * CELL_MAX> board{};
        std::array<std::uint8_t, CELL_MAX> mask{};
        extract_features_into(st, turn, &pf, true, board.data(), mask.data());

        const int action_cell = select_action(module, board, mask);
        const int ax = action_cell / N;
        const int ay = action_cell % N;
        cout << ax << ' ' << ay << '\n' << flush;

        // Read turn results
        std::array<int, M_MAX> tx{};
        std::array<int, M_MAX> ty{};
        for (int p = 0; p < m; p++) {
            cin >> tx[p] >> ty[p];
        }

        // PF update from observed opponent moves
        for (int p = 1; p < m; p++) {
            const int c = cell_index(tx[p], ty[p]);
            const MoveSummary sum = summarize_ai_observation(st_start, p, c);
            pf[p].update(sum);
        }

        for (int p = 0; p < m; p++) {
            int ex = 0, ey = 0;
            cin >> ex >> ey;
            st.ex[p] = static_cast<std::uint8_t>(ex);
            st.ey[p] = static_cast<std::uint8_t>(ey);
        }

        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                int o = 0;
                cin >> o;
                st.owner[cell_index(x, y)] = static_cast<std::int8_t>(o);
            }
        }
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                int l = 0;
                cin >> l;
                st.level[cell_index(x, y)] = static_cast<std::uint8_t>(l);
            }
        }
    }

    return 0;
}
