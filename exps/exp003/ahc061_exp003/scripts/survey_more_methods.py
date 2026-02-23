from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..cpp_ext import load_ext


@dataclass(frozen=True)
class MethodRun:
    method: str
    cfg_key: str
    fn_name: str
    kwargs: dict[str, float | int]


def _summarize_out(out: dict[str, torch.Tensor]) -> tuple[float, float, float, int]:
    sum_mae5 = float(out["sum_mae5"].sum().item())
    sum_w_l1 = float(out["sum_w_l1"].sum().item())
    sum_eps_abs = float(out["sum_eps_abs"].sum().item())
    total_updates = int(out["count"].sum().item())
    if total_updates <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    inv = 1.0 / float(total_updates)
    return sum_mae5 * inv, sum_w_l1 * inv, sum_eps_abs * inv, total_updates


def _pareto(df: pd.DataFrame) -> pd.DataFrame:
    points = df[["mae5", "updates_per_sec"]].to_numpy(dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(len(points)):
            if i == j:
                continue
            # j dominates i if mae5 <= and ups >= (with one strict)
            if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
                if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                    keep[i] = False
                    break
    return df[keep].copy().sort_values(["mae5", "updates_per_sec"], ascending=[True, False]).reset_index(drop=True)


def _build_runs() -> list[MethodRun]:
    runs: list[MethodRun] = []

    # Current best baseline from prior survey.
    runs.append(
        MethodRun(
            method="adf_beta",
            cfg_key="adf_beta:prior=0.350,eps0=0.300",
            fn_name="bench_ineq_trunc_gauss_beta_eps_estimation",
            kwargs={"prior_std": 0.35, "eps0": 0.30},
        )
    )

    # 1) EP-like extension (order-symmetrized ADF+Beta).
    runs.extend(
        [
            MethodRun(
                method="adf_beta_ep",
                cfg_key=f"adf_beta_ep:prior={p:.3f},eps0={e:.3f}",
                fn_name="bench_ineq_trunc_gauss_beta_ep_estimation",
                kwargs={"prior_std": p, "eps0": e},
            )
            for p, e in [(0.35, 0.30), (0.40, 0.30), (0.35, 0.40)]
        ]
    )

    # 2) RBPF(delta particles + beta eps).
    runs.extend(
        [
            MethodRun(
                method="rbpf_delta_beta",
                cfg_key=f"rbpf_delta_beta:K={k},eps0={e:.3f}",
                fn_name="bench_rbpf_delta_beta_estimation",
                kwargs={"rbpf_particles": k, "eps0": e},
            )
            for k, e in [(64, 0.30), (128, 0.30), (192, 0.30), (128, 0.50)]
        ]
    )

    # 3) Hybrid ADF-EP + RBPF.
    runs.extend(
        [
            MethodRun(
                method="hybrid_adf_rbpf",
                cfg_key=f"hybrid:prior={p:.3f},eps0={e:.3f},K={k}",
                fn_name="bench_hybrid_adf_rbpf_estimation",
                kwargs={"prior_std": p, "eps0": e, "rbpf_particles": k},
            )
            for p, e, k in [(0.35, 0.30, 48), (0.35, 0.30, 64), (0.40, 0.30, 64)]
        ]
    )

    # 4) PG-inspired softmax diagonal update.
    runs.extend(
        [
            MethodRun(
                method="pg_softmax_diag",
                cfg_key=f"pg_softmax_diag:tau={t:.3f},prior={p:.3f},eps0={e:.3f}",
                fn_name="bench_pg_softmax_diag_estimation",
                kwargs={"tau": t, "prior_std": p, "eps0": e},
            )
            for t, p, e in [(0.10, 0.35, 0.50), (0.11, 0.40, 0.50), (0.12, 0.35, 0.30)]
        ]
    )

    # 5) Luce/MM online ranking style.
    runs.extend(
        [
            MethodRun(
                method="luce_mm",
                cfg_key=f"luce_mm:eps0={e:.3f}",
                fn_name="bench_luce_mm_estimation",
                kwargs={"eps0": e},
            )
            for e in (0.30, 0.50)
        ]
    )

    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=999)
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--build-pf-particles", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if args.seed_end < args.seed_begin:
        raise ValueError("seed_end must be >= seed_begin")
    if args.t_max <= 0:
        raise ValueError("t_max must be >= 1")
    if args.build_pf_particles <= 0:
        raise ValueError("build_pf_particles must be >= 1")

    exp_dir = Path(__file__).resolve().parents[2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        exp_dir / "artifacts" / f"survey_more_methods_{ts}"
        if args.out_dir is None
        else Path(str(args.out_dir)).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = torch.arange(int(args.seed_begin), int(args.seed_end) + 1, dtype=torch.int64, device="cpu")
    ext = load_ext(pf_particles=int(args.build_pf_particles), verbose=bool(args.verbose_build))

    rows: list[dict[str, object]] = []
    runs = _build_runs()

    for i, run in enumerate(runs, start=1):
        fn = getattr(ext, run.fn_name)
        kwargs = {k: v for k, v in run.kwargs.items()}
        print(f"[{i}/{len(runs)}] {run.cfg_key}")
        t0 = time.perf_counter()
        out = fn(seeds, int(args.t_max), **kwargs)
        t1 = time.perf_counter()
        sec = float(t1 - t0)
        mae5, w_l1, eps_abs, total_updates = _summarize_out(out)
        updates_per_sec = float(total_updates) / sec if sec > 0 else 0.0
        rows.append(
            {
                "method": run.method,
                "cfg_key": run.cfg_key,
                "fn_name": run.fn_name,
                "params_json": json.dumps(run.kwargs, sort_keys=True),
                "seed_begin": int(args.seed_begin),
                "seed_end": int(args.seed_end),
                "t_max": int(args.t_max),
                "sec": sec,
                "updates_per_sec": updates_per_sec,
                "mae5": mae5,
                "w_l1": w_l1,
                "eps_abs": eps_abs,
                "total_updates": total_updates,
            }
        )

    df = pd.DataFrame(rows).sort_values(["mae5", "updates_per_sec"], ascending=[True, False]).reset_index(drop=True)
    csv_all = out_dir / "all_configs.csv"
    df.to_csv(csv_all, index=False)

    best = (
        df.sort_values(["method", "mae5", "updates_per_sec"], ascending=[True, True, False])
        .groupby("method", as_index=False)
        .first()
        .sort_values(["mae5", "updates_per_sec"], ascending=[True, False])
        .reset_index(drop=True)
    )
    csv_best = out_dir / "best_per_method.csv"
    best.to_csv(csv_best, index=False)

    pareto = _pareto(df)
    csv_pareto = out_dir / "pareto.csv"
    pareto.to_csv(csv_pareto, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.8))
    for method, sub in df.groupby("method"):
        ax.scatter(sub["updates_per_sec"], sub["mae5"], s=28, alpha=0.8, label=method)
    ax.scatter(pareto["updates_per_sec"], pareto["mae5"], s=80, marker="x", color="black", label="pareto")
    ax.set_xlabel("updates_per_sec")
    ax.set_ylabel("mae5 (lower is better)")
    ax.set_title(f"exp003 more methods survey ({args.seed_begin}-{args.seed_end}, t={args.t_max})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small", frameon=False)
    fig.tight_layout()
    png_scatter = out_dir / "plot_scatter_pareto.png"
    fig.savefig(png_scatter, dpi=180)

    fig2, ax2 = plt.subplots(1, 1, figsize=(max(7.0, 1.2 * len(best)), 4.0))
    labels = best["method"].tolist()
    y = best["mae5"].to_numpy(dtype=np.float64)
    ax2.bar(np.arange(len(labels), dtype=np.int32), y)
    ax2.set_xticks(np.arange(len(labels), dtype=np.int32))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("best mae5")
    ax2.set_title("Best config per method")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    png_best = out_dir / "plot_best_per_method.png"
    fig2.savefig(png_best, dpi=180)

    summary = {
        "out_dir": str(out_dir),
        "num_configs": int(len(df)),
        "best_overall": best.iloc[0].to_dict() if len(best) > 0 else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] all_configs: {csv_all}")
    print(f"[OK] best_per_method: {csv_best}")
    print(f"[OK] pareto: {csv_pareto}")
    print(f"[OK] scatter: {png_scatter}")
    print(f"[OK] best_plot: {png_best}")


if __name__ == "__main__":
    main()
