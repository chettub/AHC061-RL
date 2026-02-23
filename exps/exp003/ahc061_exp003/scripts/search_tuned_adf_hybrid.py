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
class Cfg:
    method: str
    cfg_key: str
    fn_name: str
    kwargs: dict[str, float | int]


def _summarize_out(out: dict[str, torch.Tensor]) -> tuple[float, float, float, int]:
    total_updates = int(out["count"].sum().item())
    if total_updates <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    inv = 1.0 / float(total_updates)
    mae5 = float(out["sum_mae5"].sum().item()) * inv
    w_l1 = float(out["sum_w_l1"].sum().item()) * inv
    eps_abs = float(out["sum_eps_abs"].sum().item()) * inv
    return mae5, w_l1, eps_abs, total_updates


def _run_one(ext, seeds: torch.Tensor, t_max: int, cfg: Cfg) -> dict[str, object]:
    fn = getattr(ext, cfg.fn_name)
    t0 = time.perf_counter()
    out = fn(seeds, int(t_max), **cfg.kwargs)
    t1 = time.perf_counter()
    sec = float(t1 - t0)
    mae5, w_l1, eps_abs, total_updates = _summarize_out(out)
    ups = float(total_updates) / sec if sec > 0 else 0.0
    return {
        "method": cfg.method,
        "cfg_key": cfg.cfg_key,
        "fn_name": cfg.fn_name,
        "params_json": json.dumps(cfg.kwargs, sort_keys=True),
        "sec": sec,
        "updates_per_sec": ups,
        "mae5": mae5,
        "w_l1": w_l1,
        "eps_abs": eps_abs,
        "total_updates": total_updates,
    }


def _build_cfgs() -> list[Cfg]:
    cfgs: list[Cfg] = []

    # Baseline and EP: refined neighborhood around known optimum.
    priors = [0.30, 0.325, 0.35, 0.375, 0.40]
    eps_list = [0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36]

    for p in priors:
        for e in eps_list:
            cfgs.append(
                Cfg(
                    method="adf_beta",
                    cfg_key=f"adf_beta:prior={p:.3f},eps0={e:.3f}",
                    fn_name="bench_ineq_trunc_gauss_beta_eps_estimation",
                    kwargs={"prior_std": float(p), "eps0": float(e)},
                )
            )
            cfgs.append(
                Cfg(
                    method="adf_beta_ep",
                    cfg_key=f"adf_beta_ep:prior={p:.3f},eps0={e:.3f}",
                    fn_name="bench_ineq_trunc_gauss_beta_ep_estimation",
                    kwargs={"prior_std": float(p), "eps0": float(e)},
                )
            )

    # Hybrid: widen particle count and init parameters.
    hy_priors = [0.30, 0.325, 0.35, 0.375, 0.40]
    hy_eps = [0.24, 0.28, 0.30, 0.32, 0.36]
    hy_k = [24, 32, 40, 48, 64, 80]
    for p in hy_priors:
        for e in hy_eps:
            for k in hy_k:
                cfgs.append(
                    Cfg(
                        method="hybrid_adf_rbpf",
                        cfg_key=f"hybrid:prior={p:.3f},eps0={e:.3f},K={k}",
                        fn_name="bench_hybrid_adf_rbpf_estimation",
                        kwargs={"prior_std": float(p), "eps0": float(e), "rbpf_particles": int(k)},
                    )
                )

    return cfgs


def _pareto(df: pd.DataFrame) -> pd.DataFrame:
    pts = df[["mae5", "updates_per_sec"]].to_numpy(dtype=np.float64)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in range(len(pts)):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    keep[i] = False
                    break
    return df[keep].copy().sort_values(["mae5", "updates_per_sec"], ascending=[True, False]).reset_index(drop=True)


def _select_stage2(stage1: pd.DataFrame) -> pd.DataFrame:
    picks = []

    # Method-wise strong candidates.
    for m, sub in stage1.groupby("method"):
        picks.append(sub.sort_values(["mae5", "updates_per_sec"], ascending=[True, False]).head(12))
        picks.append(sub.sort_values(["updates_per_sec", "mae5"], ascending=[False, True]).head(6))

    # Global Pareto from stage1.
    picks.append(_pareto(stage1).head(20))

    out = pd.concat(picks, ignore_index=True)
    out = out.drop_duplicates(subset=["cfg_key"]).reset_index(drop=True)
    return out


def _plot_scatter(df: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8))
    for method, sub in df.groupby("method"):
        ax.scatter(sub["updates_per_sec"], sub["mae5"], s=20, alpha=0.75, label=method)
    pare = _pareto(df)
    if len(pare) > 0:
        ax.scatter(pare["updates_per_sec"], pare["mae5"], s=70, marker="x", color="black", label="pareto")
    ax.set_xlabel("updates_per_sec")
    ax.set_ylabel("mae5")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def _plot_heatmap_2d(df: pd.DataFrame, method: str, out_png: Path) -> None:
    sub = df[df["method"] == method].copy()
    if len(sub) == 0:
        return
    sub["prior_std"] = sub["params_json"].map(lambda s: float(json.loads(s).get("prior_std", np.nan)))
    sub["eps0"] = sub["params_json"].map(lambda s: float(json.loads(s).get("eps0", np.nan)))
    piv = sub.pivot_table(index="prior_std", columns="eps0", values="mae5", aggfunc="min")

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.0))
    im = ax.imshow(piv.to_numpy(dtype=np.float64), origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in piv.columns], rotation=20)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv.index])
    ax.set_xlabel("eps0")
    ax.set_ylabel("prior_std")
    ax.set_title(f"{method} mae5 heatmap")
    fig.colorbar(im, ax=ax, label="mae5")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def _plot_hybrid_k(df: pd.DataFrame, out_png: Path) -> None:
    sub = df[df["method"] == "hybrid_adf_rbpf"].copy()
    if len(sub) == 0:
        return
    sub["rbpf_particles"] = sub["params_json"].map(lambda s: int(json.loads(s).get("rbpf_particles", 0)))
    g = sub.groupby("rbpf_particles", as_index=False).agg({"mae5": "min", "updates_per_sec": "max"})

    fig, ax1 = plt.subplots(1, 1, figsize=(6.8, 4.2))
    ax1.plot(g["rbpf_particles"], g["mae5"], marker="o", label="best mae5")
    ax1.set_xlabel("rbpf_particles")
    ax1.set_ylabel("mae5")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(g["rbpf_particles"], g["updates_per_sec"], marker="s", color="tab:orange", label="max updates/s")
    ax2.set_ylabel("updates_per_sec")
    ax1.set_title("Hybrid tradeoff by rbpf_particles")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=999)
    parser.add_argument("--holdout-begin", type=int, default=1000)
    parser.add_argument("--holdout-end", type=int, default=1999)
    parser.add_argument("--screen-end", type=int, default=199)
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--build-pf-particles", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parents[2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        exp_dir / "artifacts" / f"search_tuned_adf_hybrid_{ts}"
        if args.out_dir is None
        else Path(str(args.out_dir)).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = load_ext(pf_particles=int(args.build_pf_particles), verbose=bool(args.verbose_build))
    cfgs = _build_cfgs()

    seeds_screen = torch.arange(int(args.seed_begin), int(args.screen_end) + 1, dtype=torch.int64, device="cpu")
    seeds_train = torch.arange(int(args.seed_begin), int(args.seed_end) + 1, dtype=torch.int64, device="cpu")
    seeds_hold = torch.arange(int(args.holdout_begin), int(args.holdout_end) + 1, dtype=torch.int64, device="cpu")

    # Stage 1: screening
    rows1: list[dict[str, object]] = []
    for i, cfg in enumerate(cfgs, start=1):
        if i % 20 == 0 or i == 1 or i == len(cfgs):
            print(f"[stage1 {i}/{len(cfgs)}] {cfg.cfg_key}")
        row = _run_one(ext, seeds_screen, int(args.t_max), cfg)
        row.update({"split": "screen", "seed_begin": int(args.seed_begin), "seed_end": int(args.screen_end), "t_max": int(args.t_max)})
        rows1.append(row)
    df1 = pd.DataFrame(rows1)
    df1.to_csv(out_dir / "stage1_screen.csv", index=False)

    sel = _select_stage2(df1)
    sel.to_csv(out_dir / "stage1_selected_for_stage2.csv", index=False)

    # Stage 2: full train
    rows2: list[dict[str, object]] = []
    for i, (_, r) in enumerate(sel.iterrows(), start=1):
        cfg = Cfg(method=str(r["method"]), cfg_key=str(r["cfg_key"]), fn_name=str(r["fn_name"]), kwargs=json.loads(str(r["params_json"])))
        if i % 10 == 0 or i == 1 or i == len(sel):
            print(f"[stage2 {i}/{len(sel)}] {cfg.cfg_key}")
        row = _run_one(ext, seeds_train, int(args.t_max), cfg)
        row.update({"split": "train", "seed_begin": int(args.seed_begin), "seed_end": int(args.seed_end), "t_max": int(args.t_max)})
        rows2.append(row)
    df2 = pd.DataFrame(rows2).sort_values(["mae5", "updates_per_sec"], ascending=[True, False]).reset_index(drop=True)
    df2.to_csv(out_dir / "stage2_train.csv", index=False)

    # Stage 3: holdout on strongest train candidates.
    top_hold = (
        pd.concat(
            [
                df2.sort_values(["method", "mae5", "updates_per_sec"], ascending=[True, True, False]).groupby("method", as_index=False).head(10),
                _pareto(df2).head(20),
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["cfg_key"])
        .reset_index(drop=True)
    )
    rows3: list[dict[str, object]] = []
    for i, (_, r) in enumerate(top_hold.iterrows(), start=1):
        cfg = Cfg(method=str(r["method"]), cfg_key=str(r["cfg_key"]), fn_name=str(r["fn_name"]), kwargs=json.loads(str(r["params_json"])))
        if i % 10 == 0 or i == 1 or i == len(top_hold):
            print(f"[stage3 {i}/{len(top_hold)}] {cfg.cfg_key}")
        row = _run_one(ext, seeds_hold, int(args.t_max), cfg)
        row.update({"split": "holdout", "seed_begin": int(args.holdout_begin), "seed_end": int(args.holdout_end), "t_max": int(args.t_max)})
        rows3.append(row)
    df3 = pd.DataFrame(rows3).sort_values(["mae5", "updates_per_sec"], ascending=[True, False]).reset_index(drop=True)
    df3.to_csv(out_dir / "stage3_holdout.csv", index=False)

    best2 = (
        df2.sort_values(["method", "mae5", "updates_per_sec"], ascending=[True, True, False])
        .groupby("method", as_index=False)
        .first()
        .sort_values(["mae5", "updates_per_sec"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best3 = (
        df3.sort_values(["method", "mae5", "updates_per_sec"], ascending=[True, True, False])
        .groupby("method", as_index=False)
        .first()
        .sort_values(["mae5", "updates_per_sec"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best2.to_csv(out_dir / "best_per_method_train.csv", index=False)
    best3.to_csv(out_dir / "best_per_method_holdout.csv", index=False)
    _pareto(df2).to_csv(out_dir / "pareto_train.csv", index=False)
    _pareto(df3).to_csv(out_dir / "pareto_holdout.csv", index=False)

    _plot_scatter(df2, out_dir / "plot_stage2_train_scatter_pareto.png", "Stage2 train scatter")
    _plot_scatter(df3, out_dir / "plot_stage3_holdout_scatter_pareto.png", "Stage3 holdout scatter")
    _plot_heatmap_2d(df2, "adf_beta", out_dir / "plot_heatmap_adf_beta_train.png")
    _plot_heatmap_2d(df2, "adf_beta_ep", out_dir / "plot_heatmap_adf_beta_ep_train.png")
    _plot_hybrid_k(df2, out_dir / "plot_hybrid_k_tradeoff_train.png")

    summary = {
        "out_dir": str(out_dir),
        "n_cfg_stage1": int(len(df1)),
        "n_cfg_stage2": int(len(df2)),
        "n_cfg_stage3": int(len(df3)),
        "best_train": df2.iloc[0].to_dict() if len(df2) > 0 else None,
        "best_holdout": df3.iloc[0].to_dict() if len(df3) > 0 else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] best_train: {out_dir / 'best_per_method_train.csv'}")
    print(f"[OK] best_holdout: {out_dir / 'best_per_method_holdout.csv'}")


if __name__ == "__main__":
    main()
