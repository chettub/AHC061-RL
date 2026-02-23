#pragma once

#include <array>
#include <cstdint>

namespace ahc061::exp002 {

static constexpr int N = 10;
static constexpr int M_MAX = 8;
static constexpr int CELL_MAX = N * N;

static constexpr std::array<int, 4> DX{1, -1, 0, 0};
static constexpr std::array<int, 4> DY{0, 0, 1, -1};

constexpr bool in_bounds(int x, int y) { return 0 <= x && x < N && 0 <= y && y < N; }
constexpr int cell_index(int x, int y) { return x * N + y; }

// Neighbor indices in (R, D, L, U) order. -1 means out of bounds.
static constexpr std::array<std::array<int, 4>, CELL_MAX> NEIGH_RDLU = [] {
    std::array<std::array<int, 4>, CELL_MAX> nei{};
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            const int idx = cell_index(x, y);
            nei[idx][0] = (y + 1 < N) ? cell_index(x, y + 1) : -1;  // R
            nei[idx][1] = (x + 1 < N) ? cell_index(x + 1, y) : -1;  // D
            nei[idx][2] = (y - 1 >= 0) ? cell_index(x, y - 1) : -1; // L
            nei[idx][3] = (x - 1 >= 0) ? cell_index(x - 1, y) : -1; // U
        }
    }
    return nei;
}();

struct XorShift64 {
    std::uint64_t x = 88172645463325252ULL;
    XorShift64() = default;
    explicit XorShift64(std::uint64_t seed) : x(seed ? seed : 88172645463325252ULL) {}

    std::uint64_t next_u64() {
        x ^= x << 7;
        x ^= x >> 9;
        return x;
    }

    int next_int(int lo, int hi) {  // inclusive
        const std::uint64_t span = static_cast<std::uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    double next_double01() {
        // [0, 1)
        constexpr double INV_U64 = 1.0 / 18446744073709551616.0;  // 2^64
        return static_cast<double>(next_u64()) * INV_U64;
    }
};

struct OpponentParam {
    double wa = 0.65;
    double wb = 0.65;
    double wc = 0.65;
    double wd = 0.65;
    double eps = 0.30;
};

struct State {
    int m = 0;
    int t_max = 0;
    int u_max = 0;

    std::array<int, CELL_MAX> value{};
    std::array<std::int8_t, CELL_MAX> owner{};  // -1..M_MAX-1
    std::array<std::uint8_t, CELL_MAX> level{}; // 0..u_max

    std::array<std::uint8_t, M_MAX> ex{};  // piece x
    std::array<std::uint8_t, M_MAX> ey{};  // piece y
};

}  // namespace ahc061::exp002
