from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _group_weighted_mean(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key_vals, sub in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        cnt = sub["count"].to_numpy(dtype=np.float64)
        wsum = float(cnt.sum())
        if wsum <= 0.0:
            mw = float("nan")
            me = float("nan")
            mm = float("nan")
        else:
            mw = float((sub["mean_w_l1"].to_numpy(dtype=np.float64) * cnt).sum() / wsum)
            me = float((sub["mean_eps_abs"].to_numpy(dtype=np.float64) * cnt).sum() / wsum)
            mm = float((sub["mean_mae5"].to_numpy(dtype=np.float64) * cnt).sum() / wsum)
        r: dict[str, object] = {k: v for k, v in zip(keys, key_vals, strict=True)}
        r["count"] = int(wsum)
        r["mean_w_l1"] = mw
        r["mean_eps_abs"] = me
        r["mean_mae5"] = mm
        rows.append(r)
    return pd.DataFrame(rows)


def _build_existing_label(row: pd.Series) -> str:
    m = str(row["method"])
    if m == "pf":
        return f"PF P={int(row['pf_particles'])}"
    if m == "adf_beta":
        return f"ADF+Beta std={float(row['prior_std']):g}"
    if m == "adf_beta_ep":
        return f"ADF+Beta+EP std={float(row['prior_std']):g}"
    if m == "hybrid_adf_rbpf":
        return f"Hybrid ADF-RBPF K={int(row['pf_particles'])}"
    return m


def _key_str(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in cols:
        if c not in df.columns:
            part = pd.Series([""] * len(df), index=df.index, dtype="object")
        else:
            vals = df[c]
            if pd.api.types.is_float_dtype(vals):
                part = vals.map(lambda x: "" if pd.isna(x) else f"{float(x):.12g}")
            else:
                part = vals.map(lambda x: "" if pd.isna(x) else str(x))
        out = out + "|" + part
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-csv", type=str, required=True, help="bench_pf_particles main csv")
    parser.add_argument("--throughput-csv", type=str, required=True, help="bench_pf_particles throughput csv")
    parser.add_argument("--ckpt-summary-csv", type=str, required=True, help="bench_ckpt_hidden_estimation summary.csv")
    parser.add_argument("--ckpt-per-turn-csv", type=str, required=True, help="bench_ckpt_hidden_estimation per_turn.csv")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--title", type=str, default="CKPT vs existing estimators")
    args = parser.parse_args()

    methods_csv = Path(args.methods_csv).expanduser().resolve()
    throughput_csv = Path(args.throughput_csv).expanduser().resolve()
    ckpt_summary_csv = Path(args.ckpt_summary_csv).expanduser().resolve()
    ckpt_per_turn_csv = Path(args.ckpt_per_turn_csv).expanduser().resolve()

    if args.out_dir is None:
        exp_dir = Path(__file__).resolve().parents[2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (exp_dir / "artifacts" / f"compare_ckpt_vs_existing_{ts}").resolve()
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_df = pd.read_csv(methods_csv)
    thr_df = pd.read_csv(throughput_csv)
    ckpt_df = pd.read_csv(ckpt_summary_csv)
    ckpt_turn_df = pd.read_csv(ckpt_per_turn_csv)

    key_cols = ["method", "pf_particles", "tau", "prior_std", "eps0"]
    method_summary = _group_weighted_mean(method_df, keys=key_cols)
    method_summary["_k"] = _key_str(method_summary, key_cols)
    thr_df["_k"] = _key_str(thr_df, key_cols)
    thr_map = dict(zip(thr_df["_k"], thr_df["updates_per_sec"], strict=False))

    method_summary["updates_per_sec"] = method_summary["_k"].map(lambda k: float(thr_map.get(k, np.nan)))
    method_summary["label"] = method_summary.apply(_build_existing_label, axis=1)
    method_summary["source"] = "existing"

    # per-turn existing summary (aggregate over m)
    keys_turn = key_cols + ["turn"]
    method_turn = _group_weighted_mean(method_df, keys=keys_turn)
    method_turn["_k"] = _key_str(method_turn, key_cols)
    label_map = dict(zip(method_summary["_k"], method_summary["label"], strict=False))
    method_turn["label"] = method_turn["_k"].map(lambda k: str(label_map.get(k, "existing")))
    method_turn["source"] = "existing"

    ckpt_out = ckpt_df.copy()
    ckpt_out["label"] = ckpt_out["ckpt_name"].map(lambda s: f"CKPT {s}")
    ckpt_out["source"] = "ckpt"

    ckpt_turn_out = ckpt_turn_df.copy()
    ckpt_turn_out["label"] = ckpt_turn_out["ckpt_name"].map(lambda s: f"CKPT {s}")
    ckpt_turn_out["source"] = "ckpt"

    combined = pd.concat(
        [
            method_summary[["label", "source", "count", "mean_w_l1", "mean_eps_abs", "mean_mae5", "updates_per_sec"]],
            ckpt_out[["label", "source", "count", "mean_w_l1", "mean_eps_abs", "mean_mae5", "updates_per_sec"]],
        ],
        ignore_index=True,
    )
    combined["mean_sqe5"] = combined["mean_mae5"].to_numpy(dtype=np.float64) ** 2.0
    combined = combined.sort_values(["mean_mae5", "mean_w_l1"], ascending=[True, True]).reset_index(drop=True)

    combined_turn = pd.concat(
        [
            method_turn[["label", "source", "turn", "count", "mean_w_l1", "mean_eps_abs", "mean_mae5"]],
            ckpt_turn_out[["label", "source", "turn", "count", "mean_w_l1", "mean_eps_abs", "mean_mae5"]],
        ],
        ignore_index=True,
    )
    combined_turn["mean_sqe5"] = combined_turn["mean_mae5"].to_numpy(dtype=np.float64) ** 2.0

    csv_summary = out_dir / "combined_summary.csv"
    csv_turn = out_dir / "combined_per_turn.csv"
    combined.to_csv(csv_summary, index=False)
    combined_turn.to_csv(csv_turn, index=False)

    # Bar: mean_mae5 ranking
    fig, ax = plt.subplots(1, 1, figsize=(max(8.0, 1.1 * len(combined)), 4.6))
    xs = np.arange(len(combined), dtype=np.int32)
    ys = combined["mean_mae5"].to_numpy(dtype=np.float64)
    colors = np.where(combined["source"].to_numpy(dtype=str) == "ckpt", "#1f77b4", "#ff7f0e")
    ax.bar(xs, ys, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(combined["label"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("mean_mae5 (lower is better)")
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    png_bar = out_dir / "plot_combined_mean_mae5.png"
    fig.savefig(png_bar, dpi=180)

    # Bar: mean_sqe5 ranking
    fig_sq, ax_sq = plt.subplots(1, 1, figsize=(max(8.0, 1.1 * len(combined)), 4.6))
    ys_sq = combined["mean_sqe5"].to_numpy(dtype=np.float64)
    ax_sq.bar(xs, ys_sq, color=colors)
    ax_sq.set_xticks(xs)
    ax_sq.set_xticklabels(combined["label"].tolist(), rotation=20, ha="right")
    ax_sq.set_ylabel("mean_sqe5 = mean_mae5^2 (lower is better)")
    ax_sq.set_title(f"{args.title} (squared)")
    ax_sq.grid(True, axis="y", alpha=0.25)
    fig_sq.tight_layout()
    png_bar_sq = out_dir / "plot_combined_mean_sqe5.png"
    fig_sq.savefig(png_bar_sq, dpi=180)

    # Scatter: accuracy vs throughput
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.2, 5.4))
    for src, sub in combined.groupby("source"):
        c = "#1f77b4" if src == "ckpt" else "#ff7f0e"
        ax2.scatter(sub["updates_per_sec"], sub["mean_mae5"], s=46, alpha=0.9, label=src, color=c)
        for _, r in sub.iterrows():
            ax2.annotate(
                str(r["label"]),
                (float(r["updates_per_sec"]), float(r["mean_mae5"])),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
    ax2.set_xlabel("updates_per_sec")
    ax2.set_ylabel("mean_mae5 (lower is better)")
    ax2.set_title(f"{args.title} (accuracy-throughput)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", frameon=False)
    fig2.tight_layout()
    png_scatter = out_dir / "plot_combined_scatter_mae5_vs_throughput.png"
    fig2.savefig(png_scatter, dpi=180)

    # Scatter: squared accuracy vs throughput
    fig2_sq, ax2_sq = plt.subplots(1, 1, figsize=(7.2, 5.4))
    for src, sub in combined.groupby("source"):
        c = "#1f77b4" if src == "ckpt" else "#ff7f0e"
        ax2_sq.scatter(sub["updates_per_sec"], sub["mean_sqe5"], s=46, alpha=0.9, label=src, color=c)
        for _, r in sub.iterrows():
            ax2_sq.annotate(
                str(r["label"]),
                (float(r["updates_per_sec"]), float(r["mean_sqe5"])),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
    ax2_sq.set_xlabel("updates_per_sec")
    ax2_sq.set_ylabel("mean_sqe5 = mean_mae5^2 (lower is better)")
    ax2_sq.set_title(f"{args.title} (squared accuracy-throughput)")
    ax2_sq.grid(True, alpha=0.25)
    ax2_sq.legend(loc="best", frameon=False)
    fig2_sq.tight_layout()
    png_scatter_sq = out_dir / "plot_combined_scatter_sqe5_vs_throughput.png"
    fig2_sq.savefig(png_scatter_sq, dpi=180)

    # Per-turn curves
    fig3, ax3 = plt.subplots(1, 1, figsize=(8.0, 4.8))
    label_order = combined["label"].tolist()
    for lb in label_order:
        sub = combined_turn[combined_turn["label"] == lb].sort_values("turn")
        if len(sub) == 0:
            continue
        src = str(sub["source"].iloc[0])
        lw = 1.9 if src == "ckpt" else 1.4
        alpha = 0.95 if src == "ckpt" else 0.8
        ax3.plot(
            sub["turn"].to_numpy(dtype=np.int32),
            sub["mean_mae5"].to_numpy(dtype=np.float64),
            linewidth=lw,
            alpha=alpha,
            label=lb,
        )
    ax3.set_xlabel("turn")
    ax3.set_ylabel("mean_mae5")
    ax3.set_title(f"{args.title} (per-turn)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", fontsize="small", frameon=False)
    fig3.tight_layout()
    png_turn = out_dir / "plot_combined_per_turn_mae5.png"
    fig3.savefig(png_turn, dpi=180)

    # Per-turn squared curves
    fig3_sq, ax3_sq = plt.subplots(1, 1, figsize=(8.0, 4.8))
    for lb in label_order:
        sub = combined_turn[combined_turn["label"] == lb].sort_values("turn")
        if len(sub) == 0:
            continue
        src = str(sub["source"].iloc[0])
        lw = 1.9 if src == "ckpt" else 1.4
        alpha = 0.95 if src == "ckpt" else 0.8
        ax3_sq.plot(
            sub["turn"].to_numpy(dtype=np.int32),
            sub["mean_sqe5"].to_numpy(dtype=np.float64),
            linewidth=lw,
            alpha=alpha,
            label=lb,
        )
    ax3_sq.set_xlabel("turn")
    ax3_sq.set_ylabel("mean_sqe5 = mean_mae5^2")
    ax3_sq.set_title(f"{args.title} (per-turn squared)")
    ax3_sq.grid(True, alpha=0.25)
    ax3_sq.legend(loc="best", fontsize="small", frameon=False)
    fig3_sq.tight_layout()
    png_turn_sq = out_dir / "plot_combined_per_turn_sqe5.png"
    fig3_sq.savefig(png_turn_sq, dpi=180)

    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] combined_summary_csv: {csv_summary}")
    print(f"[OK] combined_per_turn_csv: {csv_turn}")
    print(f"[OK] bar_png: {png_bar}")
    print(f"[OK] bar_sq_png: {png_bar_sq}")
    print(f"[OK] scatter_png: {png_scatter}")
    print(f"[OK] scatter_sq_png: {png_scatter_sq}")
    print(f"[OK] per_turn_png: {png_turn}")
    print(f"[OK] per_turn_sq_png: {png_turn_sq}")


if __name__ == "__main__":
    main()
