from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ..cpp_ext import load_ext


@dataclass(frozen=True)
class BenchConfig:
    seed_begin: int
    seed_end: int
    t_max: int
    methods: list[str]
    pf_particles: list[int]
    is_points: list[int]
    grid_n: int
    metric: str
    tau: float
    prior_std: float
    eps0: float
    rbpf_particles: int
    out_dir: Path
    verbose_build: bool


@dataclass(frozen=True)
class ThroughputRow:
    method: str
    label: str
    pf_particles: int
    tau: float
    prior_std: float
    eps0: float
    seed_begin: int
    seed_end: int
    t_max: int
    sec_bench: float
    total_updates: int

    @property
    def updates_per_sec(self) -> float:
        if self.sec_bench <= 0:
            return 0.0
        return float(self.total_updates) / float(self.sec_bench)


def _parse_pf_particles(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).replace(" ", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError(f"pf_particles must be >= 1: {v}")
        out.append(v)
    if not out:
        raise ValueError("pf_particles is empty")
    return out


def _parse_methods(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).replace(" ", ",").split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p not in (
            "pf",
            "a",
            "ineq",
            "a_eigen",
            "ineq_eigen",
            "is",
            "softmax_full",
            "grid",
            "adf",
            "adf_beta",
            "adf_beta_ep",
            "hybrid_adf_rbpf",
        ):
            raise ValueError(
                "unknown method: "
                f"{p} (supported: pf,a,ineq,a_eigen,ineq_eigen,is,softmax_full,grid,adf,adf_beta,adf_beta_ep,hybrid_adf_rbpf)"
            )
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError("methods is empty")
    return out


def _metric_columns(metric: str) -> tuple[str, str]:
    if metric == "mae5":
        return "mean_mae5", "MAE(wnorm[4] + eps)/5"
    if metric == "w_l1":
        return "mean_w_l1", "L1(wnorm[4])"
    if metric == "eps_abs":
        return "mean_eps_abs", "|eps|"
    raise ValueError(f"unknown metric: {metric}")


def _df_from_out(
    *,
    out: dict[str, torch.Tensor],
    t_max: int,
    pf_particles: int,
    method: str,
    tau: float,
    prior_std: float,
    eps0: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    sum_w_l1 = out["sum_w_l1"].cpu().numpy()
    sum_eps_abs = out["sum_eps_abs"].cpu().numpy()
    sum_mae5 = out["sum_mae5"].cpu().numpy()
    count = out["count"].cpu().numpy()
    case_count = out["case_count"].cpu().numpy()

    m_max = int(count.shape[0] - 1)
    rows: list[dict[str, object]] = []
    for m in range(2, m_max + 1):
        for turn in range(int(t_max)):
            cnt = int(count[m, turn])
            if cnt <= 0:
                mean_w_l1 = float("nan")
                mean_eps_abs = float("nan")
                mean_mae5 = float("nan")
            else:
                inv = 1.0 / float(cnt)
                mean_w_l1 = float(sum_w_l1[m, turn] * inv)
                mean_eps_abs = float(sum_eps_abs[m, turn] * inv)
                mean_mae5 = float(sum_mae5[m, turn] * inv)
            rows.append(
                {
                    "method": str(method),
                    "pf_particles": int(pf_particles),
                    "tau": float(tau),
                    "prior_std": float(prior_std),
                    "eps0": float(eps0),
                    "m": int(m),
                    "turn": int(turn),
                    "count": int(cnt),
                    "mean_w_l1": mean_w_l1,
                    "mean_eps_abs": mean_eps_abs,
                    "mean_mae5": mean_mae5,
                }
            )

    return pd.DataFrame(rows), case_count


def _run_pf_one(pf_particles: int, cfg: BenchConfig) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=pf_particles, verbose=cfg.verbose_build)

    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_pf_estimation(seeds, int(cfg.t_max))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=int(pf_particles),
        method="pf",
        tau=float("nan"),
        prior_std=float("nan"),
        eps0=float("nan"),
    ) + (
        ThroughputRow(
            method="pf",
            label=f"PF P={int(pf_particles)}",
            pf_particles=int(pf_particles),
            tau=float("nan"),
            prior_std=float("nan"),
            eps0=float("nan"),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_is_one(is_points: int, cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_fixed_is_estimation(seeds, int(cfg.t_max), int(is_points))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=int(is_points),
        method="is",
        tau=float("nan"),
        prior_std=float("nan"),
        eps0=float("nan"),
    ) + (
        ThroughputRow(
            method="is",
            label=f"A-IS K={int(is_points)}",
            pf_particles=int(is_points),
            tau=float("nan"),
            prior_std=float("nan"),
            eps0=float("nan"),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_softmax_full(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_softmax_full_laplace_estimation(seeds, int(cfg.t_max), float(cfg.tau), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="softmax_full",
        tau=float(cfg.tau),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="softmax_full",
            label=f"C-softmax(full) tau={cfg.tau:g}",
            pf_particles=0,
            tau=float(cfg.tau),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_grid(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_grid_filter_estimation(seeds, int(cfg.t_max), int(cfg.grid_n), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=int(cfg.grid_n),
        method="grid",
        tau=float("nan"),
        prior_std=float("nan"),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="grid",
            label=f"D-grid n={int(cfg.grid_n)}",
            pf_particles=int(cfg.grid_n),
            tau=float("nan"),
            prior_std=float("nan"),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_adf(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    # Alias of ineq (ADF / truncated Gaussian).
    df, case_count, thr = _run_method_ineq(cfg, build_pf_particles=build_pf_particles)
    df = df.copy()
    df["method"] = "adf"
    return (
        df,
        case_count,
        ThroughputRow(
            method="adf",
            label=f"B-ADF std={cfg.prior_std:g}",
            pf_particles=int(thr.pf_particles),
            tau=float(thr.tau),
            prior_std=float(thr.prior_std),
            eps0=float(thr.eps0),
            seed_begin=int(thr.seed_begin),
            seed_end=int(thr.seed_end),
            t_max=int(thr.t_max),
            sec_bench=float(thr.sec_bench),
            total_updates=int(thr.total_updates),
        ),
    )


def _run_method_adf_beta(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_ineq_trunc_gauss_beta_eps_estimation(seeds, int(cfg.t_max), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="adf_beta",
        tau=float("nan"),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="adf_beta",
            label=f"B-ADF+Beta std={cfg.prior_std:g}",
            pf_particles=0,
            tau=float("nan"),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_adf_beta_ep(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_ineq_trunc_gauss_beta_ep_estimation(seeds, int(cfg.t_max), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="adf_beta_ep",
        tau=float("nan"),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="adf_beta_ep",
            label=f"B-ADF+Beta+EP std={cfg.prior_std:g}",
            pf_particles=0,
            tau=float("nan"),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_hybrid_adf_rbpf(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_hybrid_adf_rbpf_estimation(
        seeds,
        int(cfg.t_max),
        float(cfg.prior_std),
        float(cfg.eps0),
        int(cfg.rbpf_particles),
    )
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=int(cfg.rbpf_particles),
        method="hybrid_adf_rbpf",
        tau=float("nan"),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="hybrid_adf_rbpf",
            label=f"Hybrid ADF-RBPF K={cfg.rbpf_particles}",
            pf_particles=int(cfg.rbpf_particles),
            tau=float("nan"),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_a(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_softmax_laplace_estimation(seeds, int(cfg.t_max), float(cfg.tau), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="a",
        tau=float(cfg.tau),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="a",
            label=f"A tau={cfg.tau:g}",
            pf_particles=0,
            tau=float(cfg.tau),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_a_eigen(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    if not bool(getattr(ext, "HAS_EIGEN", False)):
        raise RuntimeError("Eigen is not available. Install Eigen3 and/or set EIGEN3_INCLUDE_DIR.")
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_softmax_laplace_eigen_estimation(seeds, int(cfg.t_max), float(cfg.tau), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="a_eigen",
        tau=float(cfg.tau),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="a_eigen",
            label=f"A-softmax(Eigen) tau={cfg.tau:g}",
            pf_particles=0,
            tau=float(cfg.tau),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_ineq(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_ineq_trunc_gauss_estimation(seeds, int(cfg.t_max), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="ineq",
        tau=float("nan"),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="ineq",
            label=f"A-ineq std={cfg.prior_std:g}",
            pf_particles=0,
            tau=float("nan"),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def _run_method_ineq_eigen(cfg: BenchConfig, *, build_pf_particles: int) -> tuple[pd.DataFrame, np.ndarray, ThroughputRow]:
    ext = load_ext(pf_particles=int(build_pf_particles), verbose=cfg.verbose_build)
    if not bool(getattr(ext, "HAS_EIGEN", False)):
        raise RuntimeError("Eigen is not available. Install Eigen3 and/or set EIGEN3_INCLUDE_DIR.")
    seeds = torch.arange(int(cfg.seed_begin), int(cfg.seed_end) + 1, dtype=torch.int64, device="cpu")
    t0 = time.perf_counter()
    out = ext.bench_ineq_trunc_gauss_eigen_estimation(seeds, int(cfg.t_max), float(cfg.prior_std), float(cfg.eps0))
    t1 = time.perf_counter()
    sec_bench = float(t1 - t0)
    total_updates = int(out["count"].sum().item())

    return _df_from_out(
        out=out,
        t_max=int(cfg.t_max),
        pf_particles=0,
        method="ineq_eigen",
        tau=float("nan"),
        prior_std=float(cfg.prior_std),
        eps0=float(cfg.eps0),
    ) + (
        ThroughputRow(
            method="ineq_eigen",
            label=f"A-ineq(Eigen) std={cfg.prior_std:g}",
            pf_particles=0,
            tau=float("nan"),
            prior_std=float(cfg.prior_std),
            eps0=float(cfg.eps0),
            seed_begin=int(cfg.seed_begin),
            seed_end=int(cfg.seed_end),
            t_max=int(cfg.t_max),
            sec_bench=sec_bench,
            total_updates=total_updates,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=999)
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument(
        "--methods",
        type=str,
        default="pf,a,ineq",
        help="comma-separated: pf,a,ineq,a_eigen,ineq_eigen,is,softmax_full,grid,adf,adf_beta,adf_beta_ep,hybrid_adf_rbpf",
    )
    parser.add_argument("--pf-particles", type=str, default="16,64,256,1024")
    parser.add_argument("--is-points", type=str, default="256", help="method is: fixed support points K (comma-separated)")
    parser.add_argument("--grid-n", type=int, default=11, help="method grid: grid resolution per axis (>=3)")
    parser.add_argument("--metric", type=str, default="mae5", choices=["mae5", "w_l1", "eps_abs"])
    parser.add_argument("--tau", type=float, default=0.1, help="method a/softmax_full: softmax temperature")
    parser.add_argument("--prior-std", type=float, default=0.5, help="method a/ineq/softmax_full/adf: prior std")
    parser.add_argument("--eps0", type=float, default=0.5, help="method a/ineq/softmax_full/grid/adf: initial epsilon")
    parser.add_argument("--rbpf-particles", type=int, default=64, help="method hybrid_adf_rbpf: RBPF particle count")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parents[2]  # .../exps/exp003
    out_dir = (
        (exp_dir / "artifacts" / "bench_pf_particles")
        if args.out_dir is None
        else Path(str(args.out_dir)).expanduser()
    )

    cfg = BenchConfig(
        seed_begin=int(args.seed_begin),
        seed_end=int(args.seed_end),
        t_max=int(args.t_max),
        methods=_parse_methods(str(args.methods)),
        pf_particles=_parse_pf_particles(str(args.pf_particles)),
        is_points=_parse_pf_particles(str(args.is_points)),
        grid_n=int(args.grid_n),
        metric=str(args.metric),
        tau=float(args.tau),
        prior_std=float(args.prior_std),
        eps0=float(args.eps0),
        rbpf_particles=int(args.rbpf_particles),
        out_dir=out_dir.resolve(),
        verbose_build=bool(args.verbose_build),
    )
    if cfg.seed_end < cfg.seed_begin:
        raise ValueError(f"seed_end must be >= seed_begin: {cfg.seed_begin}..{cfg.seed_end}")
    if cfg.t_max <= 0:
        raise ValueError(f"t_max must be >= 1: {cfg.t_max}")
    if any(m in cfg.methods for m in ("a", "a_eigen", "softmax_full")) and cfg.tau <= 0:
        raise ValueError(f"tau must be > 0: {cfg.tau}")
    if any(m in cfg.methods for m in ("a", "a_eigen", "ineq", "ineq_eigen", "softmax_full", "adf", "adf_beta")) and cfg.prior_std <= 0:
        raise ValueError(f"prior_std must be > 0: {cfg.prior_std}")
    if "adf_beta_ep" in cfg.methods and cfg.prior_std <= 0:
        raise ValueError(f"prior_std must be > 0: {cfg.prior_std}")
    if "hybrid_adf_rbpf" in cfg.methods and cfg.prior_std <= 0:
        raise ValueError(f"prior_std must be > 0: {cfg.prior_std}")
    if "hybrid_adf_rbpf" in cfg.methods and cfg.rbpf_particles <= 0:
        raise ValueError(f"rbpf_particles must be >= 1: {cfg.rbpf_particles}")
    if "grid" in cfg.methods and cfg.grid_n < 3:
        raise ValueError(f"grid_n must be >= 3: {cfg.grid_n}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    dfs: list[pd.DataFrame] = []
    case_counts: dict[int, np.ndarray] = {}
    throughput_rows: list[ThroughputRow] = []
    if "pf" in cfg.methods:
        for pf_particles in cfg.pf_particles:
            df, case_count, thr = _run_pf_one(pf_particles, cfg)
            dfs.append(df)
            case_counts[int(pf_particles)] = case_count
            throughput_rows.append(thr)

    if "a" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_a(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-1] = case_count  # method A
        throughput_rows.append(thr)

    if "a_eigen" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_a_eigen(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-3] = case_count  # method a_eigen
        throughput_rows.append(thr)

    if "ineq" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_ineq(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-2] = case_count  # method ineq
        throughput_rows.append(thr)

    if "ineq_eigen" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_ineq_eigen(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-4] = case_count  # method ineq_eigen
        throughput_rows.append(thr)

    if "is" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        for k in cfg.is_points:
            df, case_count, thr = _run_is_one(int(k), cfg, build_pf_particles=build_pf_particles)
            dfs.append(df)
            case_counts[-5] = case_count  # method is (any)
            throughput_rows.append(thr)

    if "softmax_full" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_softmax_full(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-6] = case_count  # method softmax_full
        throughput_rows.append(thr)

    if "grid" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_grid(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-7] = case_count  # method grid
        throughput_rows.append(thr)

    if "adf" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_adf(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-8] = case_count  # method adf
        throughput_rows.append(thr)

    if "adf_beta" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_adf_beta(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-9] = case_count  # method adf_beta
        throughput_rows.append(thr)

    if "adf_beta_ep" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_adf_beta_ep(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-10] = case_count  # method adf_beta_ep
        throughput_rows.append(thr)

    if "hybrid_adf_rbpf" in cfg.methods:
        build_pf_particles = int(cfg.pf_particles[0]) if cfg.pf_particles else 16
        df, case_count, thr = _run_method_hybrid_adf_rbpf(cfg, build_pf_particles=build_pf_particles)
        dfs.append(df)
        case_counts[-11] = case_count  # method hybrid_adf_rbpf
        throughput_rows.append(thr)

    df_all = pd.concat(dfs, ignore_index=True)
    methods_tag = "-".join(cfg.methods)
    csv_path = cfg.out_dir / f"exp003_methods_{methods_tag}_seeds{cfg.seed_begin}-{cfg.seed_end}_t{cfg.t_max}_{ts}.csv"
    df_all.to_csv(csv_path, index=False)

    metric_col, metric_label = _metric_columns(cfg.metric)

    ms = sorted(int(m) for m in df_all["m"].unique().tolist())
    cols = 3
    rows = int(math.ceil(len(ms) / cols))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.2), squeeze=False, sharex=True)
    flat_axes = axes.reshape(-1)

    legend_handles = None
    legend_labels = None

    for ax, m in zip(flat_axes, ms, strict=False):
        sub_m = df_all[df_all["m"] == int(m)].sort_values(["method", "pf_particles", "turn"])

        if "pf" in cfg.methods:
            for pf_particles in cfg.pf_particles:
                sub = sub_m[(sub_m["method"] == "pf") & (sub_m["pf_particles"] == int(pf_particles))]
                if sub.empty:
                    continue
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.4, label=f"PF P={int(pf_particles)}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "a" in cfg.methods:
            sub = sub_m[sub_m["method"] == "a"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.8, label=f"A-softmax tau={cfg.tau:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "ineq" in cfg.methods:
            sub = sub_m[sub_m["method"] == "ineq"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.8, label=f"A-ineq std={cfg.prior_std:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "a_eigen" in cfg.methods:
            sub = sub_m[sub_m["method"] == "a_eigen"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.8, label=f"A-softmax(Eigen) tau={cfg.tau:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "ineq_eigen" in cfg.methods:
            sub = sub_m[sub_m["method"] == "ineq_eigen"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.8, label=f"A-ineq(Eigen) std={cfg.prior_std:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "is" in cfg.methods:
            for k in cfg.is_points:
                sub = sub_m[(sub_m["method"] == "is") & (sub_m["pf_particles"] == int(k))]
                if sub.empty:
                    continue
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.6, label=f"A-IS K={int(k)}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "softmax_full" in cfg.methods:
            sub = sub_m[sub_m["method"] == "softmax_full"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.8, label=f"C-softmax(full) tau={cfg.tau:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "grid" in cfg.methods:
            sub = sub_m[sub_m["method"] == "grid"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.6, label=f"D-grid n={int(cfg.grid_n)}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "adf" in cfg.methods:
            sub = sub_m[sub_m["method"] == "adf"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.6, label=f"B-ADF std={cfg.prior_std:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "adf_beta" in cfg.methods:
            sub = sub_m[sub_m["method"] == "adf_beta"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.6, label=f"B-ADF+Beta std={cfg.prior_std:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "adf_beta_ep" in cfg.methods:
            sub = sub_m[sub_m["method"] == "adf_beta_ep"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.7, label=f"B-ADF+Beta+EP std={cfg.prior_std:g}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        if "hybrid_adf_rbpf" in cfg.methods:
            sub = sub_m[sub_m["method"] == "hybrid_adf_rbpf"]
            if not sub.empty:
                x = (sub["turn"].to_numpy(dtype=np.int32) + 1).astype(np.int32)
                y = sub[metric_col].to_numpy(dtype=np.float64)
                (line,) = ax.plot(x, y, linewidth=1.7, label=f"Hybrid ADF-RBPF K={cfg.rbpf_particles}")
                if legend_handles is None:
                    legend_handles = []
                    legend_labels = []
                if line.get_label() not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(line.get_label())

        # cases count (should be identical across pf_particles; take first)
        n_cases = None
        if "pf" in cfg.methods:
            for pf_particles in cfg.pf_particles:
                cc = case_counts[int(pf_particles)]
                if int(m) < len(cc):
                    n_cases = int(cc[int(m)])
                    break
        if n_cases is None and "a" in cfg.methods:
            cc = case_counts[-1]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "a_eigen" in cfg.methods:
            cc = case_counts[-3]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "ineq" in cfg.methods:
            cc = case_counts[-2]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "ineq_eigen" in cfg.methods:
            cc = case_counts[-4]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "is" in cfg.methods:
            cc = case_counts[-5]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "softmax_full" in cfg.methods:
            cc = case_counts[-6]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "grid" in cfg.methods:
            cc = case_counts[-7]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "adf" in cfg.methods:
            cc = case_counts[-8]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "adf_beta" in cfg.methods:
            cc = case_counts[-9]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "adf_beta_ep" in cfg.methods:
            cc = case_counts[-10]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        if n_cases is None and "hybrid_adf_rbpf" in cfg.methods:
            cc = case_counts[-11]
            if int(m) < len(cc):
                n_cases = int(cc[int(m)])
        title = f"m={int(m)}"
        if n_cases is not None:
            title += f" (cases={n_cases})"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    # hide unused axes
    for ax in flat_axes[len(ms) :]:
        ax.axis("off")

    fig.supxlabel("turn")
    fig.supylabel(metric_label)

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 6),
            fontsize="small",
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()

    png_path = cfg.out_dir / f"exp003_methods_{methods_tag}_{cfg.metric}_seeds{cfg.seed_begin}-{cfg.seed_end}_t{cfg.t_max}_{ts}.png"
    fig.savefig(png_path, dpi=200)

    # Throughput summary.
    thr_df = pd.DataFrame(
        [
            {
                "method": r.method,
                "label": r.label,
                "pf_particles": r.pf_particles,
                "tau": r.tau,
                "prior_std": r.prior_std,
                "eps0": r.eps0,
                "seed_begin": r.seed_begin,
                "seed_end": r.seed_end,
                "t_max": r.t_max,
                "sec_bench": r.sec_bench,
                "total_updates": r.total_updates,
                "updates_per_sec": r.updates_per_sec,
            }
            for r in throughput_rows
        ]
    )
    thr_csv_path = cfg.out_dir / f"exp003_throughput_methods_{methods_tag}_seeds{cfg.seed_begin}-{cfg.seed_end}_t{cfg.t_max}_{ts}.csv"
    thr_df.to_csv(thr_csv_path, index=False)

    import matplotlib.pyplot as plt

    thr_png_path = cfg.out_dir / f"exp003_throughput_methods_{methods_tag}_seeds{cfg.seed_begin}-{cfg.seed_end}_t{cfg.t_max}_{ts}.png"
    labels = thr_df["label"].tolist()
    y = thr_df["updates_per_sec"].to_numpy(dtype=np.float64)
    fig2, ax2 = plt.subplots(1, 1, figsize=(max(6.0, 1.2 * len(labels)), 3.8))
    ax2.bar(np.arange(len(labels), dtype=np.int32), y)
    ax2.set_xticks(np.arange(len(labels), dtype=np.int32))
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("opponent-updates / sec")
    ax2.set_title(f"Throughput (seeds {cfg.seed_begin}-{cfg.seed_end}, t_max={cfg.t_max})")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(thr_png_path, dpi=200)

    print(f"[OK] csv: {csv_path}")
    print(f"[OK] png: {png_path}")
    print(f"[OK] throughput_csv: {thr_csv_path}")
    print(f"[OK] throughput_png: {thr_png_path}")


if __name__ == "__main__":
    main()
