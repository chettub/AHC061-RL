from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-9
OFFICIAL_SCORE_SCALE = 100000.0 / math.log(2.0)


@dataclass(frozen=True)
class FileStat:
    file: str
    n_records: int
    n_unique_ids: int
    mean_score: float
    min_score: float
    max_score: float
    id_min: int
    id_max: int


@dataclass(frozen=True)
class CaseInfo:
    m: int
    u: int
    phi0: float


def _read_case_info_from_tools(tools_dir: Path, case_id: int) -> CaseInfo | None:
    p = tools_dir / f"{case_id:04d}.txt"
    if not p.is_file():
        return None

    with p.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) < 4:
            return None
        n, m, _, u = map(int, header[:4])
        if n <= 0 or m <= 1:
            return None

        values = [[0 for _ in range(n)] for _ in range(n)]
        for x in range(n):
            row = f.readline().strip().split()
            if len(row) != n:
                return None
            for y in range(n):
                values[x][y] = int(row[y])

        starts: list[tuple[int, int]] = []
        for _ in range(m):
            row = f.readline().strip().split()
            if len(row) < 2:
                return None
            sx, sy = map(int, row[:2])
            starts.append((sx, sy))

    sx0, sy0 = starts[0]
    s0 = float(values[sx0][sy0])
    sa = 0.0
    for pidx in range(1, m):
        sx, sy = starts[pidx]
        sa = max(sa, float(values[sx][sy]))
    phi0 = math.log1p(s0 / (sa + EPS))
    return CaseInfo(m=m, u=u, phi0=phi0)


def _load_results(paths: Iterable[Path]) -> tuple[pd.DataFrame, dict[int, list[float]]]:
    file_stats: list[FileStat] = []
    id_to_scores: dict[int, list[float]] = defaultdict(list)

    for path in sorted(paths):
        ids: list[int] = []
        scores: list[float] = []

        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"JSON parse failed: {path}:{i}: {e}") from e
                if "id" not in obj or "score" not in obj:
                    raise RuntimeError(f"Missing key in {path}:{i}: expected id/score")
                case_id = int(obj["id"])
                score = float(obj["score"])
                ids.append(case_id)
                scores.append(score)
                id_to_scores[case_id].append(score)

        if not scores:
            continue
        file_stats.append(
            FileStat(
                file=path.name,
                n_records=len(scores),
                n_unique_ids=len(set(ids)),
                mean_score=float(np.mean(scores)),
                min_score=float(np.min(scores)),
                max_score=float(np.max(scores)),
                id_min=min(ids),
                id_max=max(ids),
            )
        )

    if not file_stats:
        raise RuntimeError("No result rows were loaded.")

    file_df = pd.DataFrame([s.__dict__ for s in file_stats]).sort_values("mean_score", ascending=False)
    return file_df, id_to_scores


def _build_id_summary(
    id_to_scores: dict[int, list[float]],
    tools_dir: Path,
    *,
    min_files_per_id: int,
    total_files: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    skipped_missing_tools = 0
    skipped_low_coverage = 0

    for case_id, scores in sorted(id_to_scores.items()):
        num_files = len(scores)
        if num_files < min_files_per_id:
            skipped_low_coverage += 1
            continue
        info = _read_case_info_from_tools(tools_dir, case_id)
        if info is None:
            skipped_missing_tools += 1
            continue

        arr = np.array(scores, dtype=np.float64)
        rows.append(
            {
                "id": case_id,
                "m": int(info.m),
                "u": int(info.u),
                "phi0": float(info.phi0),
                "num_files": int(num_files),
                "coverage_ratio": float(num_files / max(1, total_files)),
                "best_score": float(arr.max()),
                "mean_score": float(arr.mean()),
                "std_score": float(arr.std(ddof=0)),
                "min_score": float(arr.min()),
            }
        )

    if not rows:
        raise RuntimeError("No id summary rows. Relax --min-files-per-id or check input files.")

    id_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    print(
        "[INFO] id rows:",
        len(id_df),
        "skipped_low_coverage:",
        skipped_low_coverage,
        "skipped_missing_tools:",
        skipped_missing_tools,
    )
    return id_df


def _build_mu_summary(id_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        id_df.groupby(["m", "u"], as_index=False)
        .agg(
            n_case=("id", "size"),
            mean_best=("best_score", "mean"),
            median_best=("best_score", "median"),
            min_best=("best_score", "min"),
            max_best=("best_score", "max"),
            mean_num_files=("num_files", "mean"),
            mean_coverage=("coverage_ratio", "mean"),
        )
        .sort_values(["m", "u"])
        .reset_index(drop=True)
    )
    for q in (0.10, 0.25, 0.75, 0.90):
        q_df = (
            id_df.groupby(["m", "u"], as_index=False)["best_score"]
            .quantile(q)
            .rename(columns={"best_score": f"p{int(q * 100):02d}_best"})
        )
        base = base.merge(q_df, on=["m", "u"], how="left")
    return base


def _build_reward_table(mu_df: pd.DataFrame, *, b_column: str) -> pd.DataFrame:
    if b_column not in mu_df.columns:
        raise RuntimeError(f"b_column not found in b_mu_summary: {b_column!r}")

    tbl = mu_df[["m", "u", "n_case", b_column]].copy()
    tbl = tbl.rename(columns={b_column: "b_value"})
    b_arr = tbl["b_value"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(b_arr)) or np.any(b_arr <= 0.0):
        raise RuntimeError("b_value must be positive finite values.")

    tbl["weight_raw"] = 1.0 / tbl["b_value"]

    p_case = tbl["n_case"].to_numpy(dtype=np.float64)
    p_case = p_case / max(EPS, float(p_case.sum()))
    mean_weight = float(np.sum(tbl["weight_raw"].to_numpy(dtype=np.float64) * p_case))
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        raise RuntimeError("Invalid mean weight for normalization.")

    tbl["weight_norm"] = tbl["weight_raw"] / mean_weight
    tbl["case_prob"] = p_case
    return tbl.sort_values(["m", "u"]).reset_index(drop=True)


def _build_fixed_alpha_calibration(
    *,
    id_to_scores: dict[int, list[float]],
    id_df: pd.DataFrame,
    reward_table: pd.DataFrame,
    b_column: str,
    results_glob: str,
    tools_dir: str,
    result_file_count: int,
) -> dict[str, object]:
    id_meta = id_df[["id", "m", "u", "phi0"]].set_index("id")
    mu_to_w: dict[tuple[int, int], float] = {}
    for row in reward_table.itertuples(index=False):
        mu_to_w[(int(row.m), int(row.u))] = float(row.weight_norm)

    sum_sq_old = 0.0
    sum_sq_weighted = 0.0
    num_samples = 0
    missing_id = 0
    missing_mu = 0

    for case_id, scores in id_to_scores.items():
        if case_id not in id_meta.index:
            missing_id += 1
            continue
        m = int(id_meta.at[case_id, "m"])
        u = int(id_meta.at[case_id, "u"])
        phi0 = float(id_meta.at[case_id, "phi0"])
        w = mu_to_w.get((m, u))
        if w is None:
            missing_mu += 1
            continue

        for score in scores:
            g_old = float(score) / OFFICIAL_SCORE_SCALE - phi0
            g_weighted = w * g_old
            sum_sq_old += g_old * g_old
            sum_sq_weighted += g_weighted * g_weighted
            num_samples += 1

    if num_samples <= 0:
        raise RuntimeError("No samples available for alpha calibration.")

    old_rms = math.sqrt(sum_sq_old / num_samples)
    weighted_rms = math.sqrt(sum_sq_weighted / num_samples)
    if not math.isfinite(weighted_rms) or weighted_rms <= 0.0:
        raise RuntimeError("Invalid weighted_rms in alpha calibration.")
    alpha_fixed = old_rms / weighted_rms

    return {
        "schema_version": 1,
        "alpha_mode": "fixed_historical",
        "alpha_fixed": float(alpha_fixed),
        "b_mu_column": str(b_column),
        "reward_formula": "r_new = alpha_fixed * weight_norm(M,U) * r_old",
        "old_return_rms": float(old_rms),
        "weighted_return_rms": float(weighted_rms),
        "num_samples": int(num_samples),
        "num_case_ids": int(len(id_df)),
        "results_file_count": int(result_file_count),
        "missing_id_count": int(missing_id),
        "missing_mu_count": int(missing_mu),
        "source_results_glob": str(results_glob),
        "source_tools_dir": str(tools_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _matrix_from_summary(summary: pd.DataFrame, metric: str) -> np.ndarray:
    mat = np.full((7, 5), np.nan, dtype=np.float64)  # M in [2,8], U in [1,5]
    for row in summary.itertuples(index=False):
        m = int(row.m)
        u = int(row.u)
        if 2 <= m <= 8 and 1 <= u <= 5:
            mat[m - 2, u - 1] = float(getattr(row, metric))
    return mat


def _plot_heatmaps(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("mean_best", "B(M,U) proxy mean"),
        ("median_best", "B(M,U) proxy median"),
        ("p90_best", "B(M,U) proxy p90"),
        ("n_case", "Num cases"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, (metric, title) in zip(axes, metrics):
        mat = _matrix_from_summary(summary, metric)
        cmap = "viridis" if metric != "n_case" else "magma"
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("U")
        ax.set_ylabel("M")
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([str(i) for i in range(1, 6)])
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels([str(i) for i in range(2, 9)])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    txt = "-"
                elif metric == "n_case":
                    txt = f"{int(mat[i, j])}"
                else:
                    txt = f"{mat[i, j]:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("B(M,U) proxy from local results (case-wise best score)", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_distributions(id_df: pd.DataFrame, out_path: Path) -> None:
    groups: list[tuple[str, np.ndarray]] = []
    for m in range(2, 9):
        for u in range(1, 6):
            vals = id_df.loc[(id_df["m"] == m) & (id_df["u"] == u), "best_score"].to_numpy()
            if vals.size == 0:
                continue
            groups.append((f"M{m}U{u}", vals))

    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    ax.boxplot([v for _, v in groups], tick_labels=[k for k, _ in groups], showfliers=False)
    ax.set_title("Distribution of case-wise best score by (M,U)")
    ax.set_xlabel("(M,U)")
    ax.set_ylabel("best score in local results")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_mu_lines(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for u in range(1, 6):
        s = summary[summary["u"] == u].sort_values("m")
        if s.empty:
            continue
        x = s["m"].to_numpy()
        y = s["mean_best"].to_numpy()
        y_lo = s["p25_best"].to_numpy()
        y_hi = s["p75_best"].to_numpy()
        ax.plot(x, y, marker="o", label=f"U={u}")
        ax.fill_between(x, y_lo, y_hi, alpha=0.15)
    ax.set_xticks(np.arange(2, 9))
    ax.set_xlabel("M")
    ax.set_ylabel("best score (mean / IQR)")
    ax.set_title("B(M,U) proxy trend by M (line) and IQR (band)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze B(M,U) proxy from local result files.")
    parser.add_argument("--results-glob", type=str, default="results/*.res")
    parser.add_argument("--tools-dir", type=str, default="tools/in")
    parser.add_argument("--min-files-per-id", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="exps/exp004/artifacts/b_mu_analysis")
    parser.add_argument(
        "--reward-b-column",
        type=str,
        default="mean_best",
        help="Which B(M,U) column is used for reward table (e.g. mean_best/median_best/p90_best).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    result_paths = sorted(repo_root.glob(args.results_glob))
    if not result_paths:
        raise RuntimeError(f"No files matched: {args.results_glob}")

    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = (repo_root / args.tools_dir).resolve()

    file_df, id_to_scores = _load_results(result_paths)
    id_df = _build_id_summary(
        id_to_scores,
        tools_dir,
        min_files_per_id=int(args.min_files_per_id),
        total_files=len(file_df),
    )
    mu_df = _build_mu_summary(id_df)
    reward_df = _build_reward_table(mu_df, b_column=str(args.reward_b_column))
    calib = _build_fixed_alpha_calibration(
        id_to_scores=id_to_scores,
        id_df=id_df,
        reward_table=reward_df,
        b_column=str(args.reward_b_column),
        results_glob=str(args.results_glob),
        tools_dir=str(args.tools_dir),
        result_file_count=len(file_df),
    )

    file_csv = out_dir / "file_summary.csv"
    id_csv = out_dir / "id_summary.csv"
    mu_csv = out_dir / "b_mu_summary.csv"
    reward_csv = out_dir / "b_mu_reward_table.csv"
    alpha_json = out_dir / "reward_calibration_fixed_alpha.json"

    file_df.to_csv(file_csv, index=False)
    id_df.to_csv(id_csv, index=False)
    mu_df.to_csv(mu_csv, index=False)
    reward_df.to_csv(reward_csv, index=False)
    alpha_json.write_text(json.dumps(calib, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    heatmap_png = out_dir / "b_mu_heatmaps.png"
    dist_png = out_dir / "b_mu_boxplot.png"
    line_png = out_dir / "b_mu_lines.png"
    _plot_heatmaps(mu_df, heatmap_png)
    _plot_distributions(id_df, dist_png)
    _plot_mu_lines(mu_df, line_png)

    print(f"[DONE] files={len(file_df)} ids={len(id_df)}")
    print(f"[DONE] alpha_fixed={float(calib['alpha_fixed']):.8f}")
    print(f"[DONE] wrote: {file_csv}")
    print(f"[DONE] wrote: {id_csv}")
    print(f"[DONE] wrote: {mu_csv}")
    print(f"[DONE] wrote: {reward_csv}")
    print(f"[DONE] wrote: {alpha_json}")
    print(f"[DONE] wrote: {heatmap_png}")
    print(f"[DONE] wrote: {dist_png}")
    print(f"[DONE] wrote: {line_png}")


if __name__ == "__main__":
    main()

