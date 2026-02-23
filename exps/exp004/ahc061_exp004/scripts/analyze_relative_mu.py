from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_mu_from_tools(tools_dir: Path, case_id: int) -> tuple[int, int] | None:
    p = tools_dir / f"{case_id:04d}.txt"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split()
    if len(first) < 4:
        return None
    _, m, _, u = map(int, first[:4])
    return m, u


def _round_half_up_nonnegative(x: float) -> int:
    return int(math.floor(x + 0.5))


def _load_result_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for path in sorted(paths):
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
                rows.append(
                    {
                        "file": path.name,
                        "id": int(obj["id"]),
                        "score": float(obj["score"]),
                    }
                )
    if not rows:
        raise RuntimeError("No result rows were loaded.")

    df = pd.DataFrame(rows).sort_values(["file", "id"]).reset_index(drop=True)
    dup = df.duplicated(subset=["file", "id"])
    if bool(dup.any()):
        bad = df.loc[dup, ["file", "id"]].head(1).to_dict("records")[0]
        raise RuntimeError(f"Duplicate (file,id) detected: {bad}")
    return df


def _build_relative_case_rows(
    target_df: pd.DataFrame,
    ref_case_df: pd.DataFrame,
    tools_dir: Path,
    *,
    min_ref_files_per_id: int,
) -> pd.DataFrame:
    merged = target_df.merge(ref_case_df, on="id", how="left")
    rows: list[dict[str, int | float | str]] = []
    skipped_missing_tools = 0
    skipped_missing_ref = 0
    skipped_low_ref = 0
    skipped_nonpositive_ref = 0

    for row in merged.itertuples(index=False):
        if pd.isna(row.ref_best_abs_score) or pd.isna(row.ref_num_files):
            skipped_missing_ref += 1
            continue
        ref_num_files = int(row.ref_num_files)
        if ref_num_files < min_ref_files_per_id:
            skipped_low_ref += 1
            continue
        ref_best_abs_score = float(row.ref_best_abs_score)
        if ref_best_abs_score <= 0.0:
            skipped_nonpositive_ref += 1
            continue
        mu = _read_mu_from_tools(tools_dir, int(row.id))
        if mu is None:
            skipped_missing_tools += 1
            continue
        m, u = mu
        score = float(row.score)
        relative_ratio = score / ref_best_abs_score
        relative_score = _round_half_up_nonnegative(1.0e9 * relative_ratio)
        rows.append(
            {
                "file": str(row.file),
                "id": int(row.id),
                "m": m,
                "u": u,
                "abs_score": score,
                "ref_best_abs_score": ref_best_abs_score,
                "ref_num_files": ref_num_files,
                "relative_ratio": relative_ratio,
                "relative_score": relative_score,
            }
        )

    if not rows:
        raise RuntimeError("No rows for relative score analysis. Check globs and filters.")

    rel_df = pd.DataFrame(rows).sort_values(["file", "id"]).reset_index(drop=True)
    print(
        "[INFO] relative rows:",
        len(rel_df),
        "skipped_missing_ref:",
        skipped_missing_ref,
        "skipped_low_ref:",
        skipped_low_ref,
        "skipped_nonpositive_ref:",
        skipped_nonpositive_ref,
        "skipped_missing_tools:",
        skipped_missing_tools,
    )
    return rel_df


def _build_relative_mu_summary(rel_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        rel_df.groupby(["file", "m", "u"], as_index=False)
        .agg(
            n_case=("id", "size"),
            mean_relative_score=("relative_score", "mean"),
            median_relative_score=("relative_score", "median"),
            mean_relative_ratio=("relative_ratio", "mean"),
            min_relative_score=("relative_score", "min"),
            max_relative_score=("relative_score", "max"),
            mean_abs_score=("abs_score", "mean"),
        )
        .sort_values(["file", "m", "u"])
        .reset_index(drop=True)
    )
    for q in (0.10, 0.25, 0.75, 0.90):
        q_df = (
            rel_df.groupby(["file", "m", "u"], as_index=False)["relative_score"]
            .quantile(q)
            .rename(columns={"relative_score": f"p{int(q * 100):02d}_relative_score"})
        )
        base = base.merge(q_df, on=["file", "m", "u"], how="left")
    return base


def _matrix_from_summary(summary: pd.DataFrame, metric: str) -> np.ndarray:
    mat = np.full((7, 5), np.nan, dtype=np.float64)  # M in [2,8], U in [1,5]
    for row in summary.itertuples(index=False):
        m = int(row.m)
        u = int(row.u)
        if 2 <= m <= 8 and 1 <= u <= 5:
            mat[m - 2, u - 1] = float(getattr(row, metric))
    return mat


def _plot_mean_relative_heatmap(summary: pd.DataFrame, out_path: Path, *, title: str) -> None:
    mat_score = _matrix_from_summary(summary, "mean_relative_score")
    mat_count = _matrix_from_summary(summary, "n_case")

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    im = ax.imshow(mat_score, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0e9)
    ax.set_title(title)
    ax.set_xlabel("U")
    ax.set_ylabel("M")
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([str(i) for i in range(1, 6)])
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels([str(i) for i in range(2, 9)])

    for i in range(mat_score.shape[0]):
        for j in range(mat_score.shape[1]):
            if np.isnan(mat_score[i, j]) or np.isnan(mat_count[i, j]):
                txt = "-"
            else:
                score_m = mat_score[i, j] / 1.0e6
                txt = f"{score_m:.1f}M\nn={int(mat_count[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean relative score (1e9 scale)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze mean relative score heatmap by (M,U) from local result files."
    )
    parser.add_argument("--reference-results-glob", type=str, default="results/*.res")
    parser.add_argument("--target-results-glob", type=str, required=True)
    parser.add_argument("--tools-dir", type=str, default="tools/in")
    parser.add_argument("--min-ref-files-per-id", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="exps/exp004/artifacts/relative_mu_analysis")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    ref_paths = sorted(repo_root.glob(args.reference_results_glob))
    if not ref_paths:
        raise RuntimeError(f"No files matched --reference-results-glob: {args.reference_results_glob}")
    target_paths = sorted(repo_root.glob(args.target_results_glob))
    if not target_paths:
        raise RuntimeError(f"No files matched --target-results-glob: {args.target_results_glob}")

    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = (repo_root / args.tools_dir).resolve()

    ref_df = _load_result_rows(ref_paths)
    target_df = _load_result_rows(target_paths)

    ref_case_df = (
        ref_df.groupby("id", as_index=False)
        .agg(
            ref_best_abs_score=("score", "max"),
            ref_num_files=("file", "nunique"),
        )
        .sort_values("id")
        .reset_index(drop=True)
    )

    rel_df = _build_relative_case_rows(
        target_df,
        ref_case_df,
        tools_dir,
        min_ref_files_per_id=int(args.min_ref_files_per_id),
    )
    mu_df = _build_relative_mu_summary(rel_df)

    ref_csv = out_dir / "reference_case_best.csv"
    rel_csv = out_dir / "target_relative_case_scores.csv"
    mu_csv = out_dir / "relative_mu_summary.csv"
    ref_case_df.to_csv(ref_csv, index=False)
    rel_df.to_csv(rel_csv, index=False)
    mu_df.to_csv(mu_csv, index=False)

    target_files = sorted(mu_df["file"].unique().tolist())
    heatmap_paths: list[Path] = []
    if len(target_files) == 1:
        sub = mu_df[mu_df["file"] == target_files[0]].copy()
        p = out_dir / "relative_mu_heatmap.png"
        _plot_mean_relative_heatmap(
            sub,
            p,
            title=f"Mean relative score by (M,U): {target_files[0]}",
        )
        heatmap_paths.append(p)
    else:
        for file_name in target_files:
            sub = mu_df[mu_df["file"] == file_name].copy()
            p = out_dir / f"relative_mu_heatmap__{_safe_name(file_name)}.png"
            _plot_mean_relative_heatmap(
                sub,
                p,
                title=f"Mean relative score by (M,U): {file_name}",
            )
            heatmap_paths.append(p)

    print(f"[DONE] reference_files={len(ref_paths)} target_files={len(target_files)} rows={len(rel_df)}")
    print(f"[DONE] wrote: {ref_csv}")
    print(f"[DONE] wrote: {rel_csv}")
    print(f"[DONE] wrote: {mu_csv}")
    for p in heatmap_paths:
        print(f"[DONE] wrote: {p}")


if __name__ == "__main__":
    main()
