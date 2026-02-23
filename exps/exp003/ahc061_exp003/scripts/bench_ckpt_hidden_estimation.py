from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exps.exp002.ahc061_exp002.ckpt import model_spec_from_ckpt, normalize_state_dict_keys, torch_load_maybe_weights_only
from exps.exp002.ahc061_exp002.env import BatchEnv
from exps.exp002.ahc061_exp002.models import build_policy_value_model, masked_logits


@dataclass(frozen=True)
class CkptMeta:
    path: Path
    arch_name: str
    feature_id: str
    in_channels: int
    hidden: int
    blocks: int
    upd: int
    mtime: float


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


def _parse_ckpt_list(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).replace("\n", ",").replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("ckpt list is empty")
    return parts


def _resolve_ckpts(
    *,
    repo_root: Path,
    ckpts_arg: str | None,
    ckpt_glob: str,
    latest_only: bool,
    max_ckpts: int,
    require_arch: str | None,
    require_feature_id: str | None,
) -> list[CkptMeta]:
    if ckpts_arg is not None:
        candidates = _parse_ckpt_list(ckpts_arg)
    else:
        pattern = ckpt_glob
        if not Path(pattern).is_absolute():
            pattern = str((repo_root / pattern).resolve())
        candidates = sorted(glob.glob(pattern))

    metas: list[CkptMeta] = []
    for c in candidates:
        p = Path(c).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            continue
        ckpt = torch_load_maybe_weights_only(p)
        ms = model_spec_from_ckpt(ckpt)
        arch_name = str(ms.arch_name)
        feature_id = str(ms.feature_id)
        if require_arch is not None and arch_name != require_arch:
            continue
        if require_feature_id is not None and feature_id != require_feature_id:
            continue
        metas.append(
            CkptMeta(
                path=p,
                arch_name=arch_name,
                feature_id=feature_id,
                in_channels=int(ms.in_channels),
                hidden=int(ms.hidden),
                blocks=int(ms.blocks),
                upd=int(ckpt.get("upd", -1)),
                mtime=float(p.stat().st_mtime),
            )
        )

    metas.sort(key=lambda x: x.mtime, reverse=True)
    if latest_only and metas:
        metas = metas[:1]
    if max_ckpts > 0:
        metas = metas[: int(max_ckpts)]
    return metas


def _safe_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path.resolve())


@torch.inference_mode()
def _eval_one_ckpt(
    *,
    meta: CkptMeta,
    repo_root: Path,
    seed_begin: int,
    seed_end: int,
    t_max: int,
    batch_size: int,
    device: torch.device,
    action: str,
    no_pf: bool,
    verbose_build: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    if seed_end < seed_begin:
        raise ValueError(f"seed_end must be >= seed_begin: {seed_begin}..{seed_end}")
    if t_max <= 0:
        raise ValueError(f"t_max must be >= 1: {t_max}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1: {batch_size}")

    ckpt = torch_load_maybe_weights_only(meta.path)

    env = BatchEnv(
        batch_size=int(batch_size),
        feature_id=str(meta.feature_id),
        pf_enabled=not bool(no_pf),
        verbose_build=bool(verbose_build),
    )
    if int(meta.in_channels) != int(env.feature_channels):
        raise RuntimeError(
            f"[CKPT] in_channels mismatch: ckpt={int(meta.in_channels)} env(feature_id={meta.feature_id!r})={int(env.feature_channels)}"
        )

    model = build_policy_value_model(
        str(meta.arch_name),
        in_channels=int(meta.in_channels),
        hidden_channels=int(meta.hidden),
        blocks=int(meta.blocks),
        feature_id=str(meta.feature_id),
    ).to(device)

    missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt["model"]), strict=False)
    if unexpected:
        raise RuntimeError(f"[CKPT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    bad_missing = [k for k in missing if not k.startswith("opp_move_head.") and not k.startswith("opp_param_head.")]
    if bad_missing:
        raise RuntimeError(f"[CKPT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
    if any(k.startswith("opp_param_head.") for k in missing):
        raise RuntimeError(f"[CKPT] opp_param_head is missing in {meta.path.name}; hidden-parameter estimation cannot be evaluated.")
    model.eval()

    m_max = int(env.spec.m_max)

    board = torch.empty((batch_size, int(meta.in_channels), 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((batch_size, 100), dtype=torch.uint8, device="cpu")
    next_board = torch.empty_like(board)
    next_mask = torch.empty_like(mask)
    reward = torch.empty((batch_size,), dtype=torch.float32, device="cpu")
    done = torch.empty((batch_size,), dtype=torch.uint8, device="cpu")
    opp_move_dist = torch.empty((batch_size, m_max, 100), dtype=torch.float32, device="cpu")
    opp_param_true = torch.empty((batch_size, m_max, 5), dtype=torch.float32, device="cpu")
    opp_valid = torch.empty((batch_size, m_max), dtype=torch.uint8, device="cpu")

    sum_w_l1 = 0.0
    sum_eps_abs = 0.0
    sum_mae5 = 0.0
    count = 0

    turn_sum_w_l1 = np.zeros((t_max,), dtype=np.float64)
    turn_sum_eps_abs = np.zeros((t_max,), dtype=np.float64)
    turn_sum_mae5 = np.zeros((t_max,), dtype=np.float64)
    turn_count = np.zeros((t_max,), dtype=np.int64)

    sample_mask = torch.zeros((batch_size,), dtype=torch.bool, device="cpu")
    seed_total = int(seed_end - seed_begin + 1)

    t0 = time.perf_counter()
    for s0 in range(int(seed_begin), int(seed_end) + 1, int(batch_size)):
        n_valid = min(int(batch_size), int(seed_end) - int(s0) + 1)
        seeds = torch.empty((batch_size,), dtype=torch.int64, device="cpu")
        seeds[:n_valid].copy_(torch.arange(int(s0), int(s0) + int(n_valid), dtype=torch.int64, device="cpu"))
        fill_seed = int(seeds[int(n_valid) - 1].item()) if int(n_valid) > 0 else int(s0)
        if int(n_valid) < int(batch_size):
            seeds[int(n_valid) :].fill_(fill_seed)
        sample_mask.zero_()
        sample_mask[:n_valid] = True

        env.reset_random(seeds)
        env.observe_into(board, mask)
        env.aux_targets_into(opp_move_dist, opp_param_true, opp_valid)

        for turn in range(int(t_max)):
            board_dev = board.to(device, non_blocking=(device.type == "cuda"))
            mask_dev = mask.to(device, non_blocking=(device.type == "cuda"))
            logits, _, _, opp_param_logits = model(board_dev, with_aux=True)
            logits = masked_logits(logits.float(), mask_dev)
            if action == "greedy":
                actions = torch.argmax(logits, dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(1)

            pred = opp_param_logits.float().to("cpu")
            pred_w = torch.softmax(pred[..., :4], dim=-1)
            pred_eps = torch.sigmoid(pred[..., 4])
            tgt_w = opp_param_true[..., :4]
            tgt_eps = opp_param_true[..., 4]

            valid = opp_valid != 0
            if int(n_valid) < int(batch_size):
                valid = valid & sample_mask.view(batch_size, 1)
            n = int(valid.sum().item())
            if n > 0:
                w_l1_each = torch.abs(pred_w - tgt_w).sum(dim=-1)
                eps_abs_each = torch.abs(pred_eps - tgt_eps)
                mae5_each = (w_l1_each + eps_abs_each) * 0.2

                sw = float(w_l1_each[valid].sum().item())
                se = float(eps_abs_each[valid].sum().item())
                sm = float(mae5_each[valid].sum().item())

                sum_w_l1 += sw
                sum_eps_abs += se
                sum_mae5 += sm
                count += n

                turn_sum_w_l1[turn] += sw
                turn_sum_eps_abs[turn] += se
                turn_sum_mae5[turn] += sm
                turn_count[turn] += n

            if turn + 1 < int(t_max):
                env.step_observe_into(actions.to(dtype=torch.int64, device="cpu"), next_board, next_mask, reward, done)
                env.aux_targets_into(opp_move_dist, opp_param_true, opp_valid)
                board, next_board = next_board, board
                mask, next_mask = next_mask, mask
    t1 = time.perf_counter()

    sec_total = float(t1 - t0)
    mean_w_l1 = float(sum_w_l1 / count) if count > 0 else float("nan")
    mean_eps_abs = float(sum_eps_abs / count) if count > 0 else float("nan")
    mean_mae5 = float(sum_mae5 / count) if count > 0 else float("nan")
    updates_per_sec = float(count / sec_total) if sec_total > 0 else 0.0

    row = {
        "ckpt": _safe_rel(meta.path, repo_root),
        "ckpt_name": meta.path.name,
        "mtime": datetime.fromtimestamp(meta.mtime).isoformat(timespec="seconds"),
        "upd": int(meta.upd),
        "arch_name": str(meta.arch_name),
        "feature_id": str(meta.feature_id),
        "in_channels": int(meta.in_channels),
        "hidden": int(meta.hidden),
        "blocks": int(meta.blocks),
        "seed_begin": int(seed_begin),
        "seed_end": int(seed_end),
        "num_seeds": int(seed_total),
        "t_max": int(t_max),
        "batch_size": int(batch_size),
        "pf_enabled": not bool(no_pf),
        "action": str(action),
        "count": int(count),
        "mean_w_l1": mean_w_l1,
        "mean_eps_abs": mean_eps_abs,
        "mean_mae5": mean_mae5,
        "sec_total": sec_total,
        "updates_per_sec": updates_per_sec,
    }

    turn_rows: list[dict[str, object]] = []
    for t in range(int(t_max)):
        cnt = int(turn_count[t])
        if cnt > 0:
            inv = 1.0 / float(cnt)
            tw = float(turn_sum_w_l1[t] * inv)
            te = float(turn_sum_eps_abs[t] * inv)
            tm = float(turn_sum_mae5[t] * inv)
        else:
            tw = float("nan")
            te = float("nan")
            tm = float("nan")
        turn_rows.append(
            {
                "ckpt": row["ckpt"],
                "ckpt_name": row["ckpt_name"],
                "turn": int(t + 1),
                "count": cnt,
                "mean_w_l1": tw,
                "mean_eps_abs": te,
                "mean_mae5": tm,
            }
        )

    return row, pd.DataFrame(turn_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=str, default=None, help="comma-separated ckpt paths (if set, --ckpt-glob is ignored)")
    parser.add_argument("--ckpt-glob", type=str, default="checkpoints/*_0p999.pt")
    parser.add_argument("--latest-only", action="store_true", help="evaluate only the newest ckpt by mtime after filtering")
    parser.add_argument("--max-ckpts", type=int, default=0, help="0 means no limit")
    parser.add_argument("--require-arch", type=str, default="dwres_ppconcat_v1")
    parser.add_argument("--require-feature-id", type=str, default="research_v4")
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=999)
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--action", type=str, choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    exp_dir = Path(__file__).resolve().parents[2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        exp_dir / "artifacts" / f"bench_ckpt_hidden_estimation_{ts}"
        if args.out_dir is None
        else Path(str(args.out_dir)).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = _resolve_ckpts(
        repo_root=repo_root,
        ckpts_arg=args.ckpts,
        ckpt_glob=str(args.ckpt_glob),
        latest_only=bool(args.latest_only),
        max_ckpts=int(args.max_ckpts),
        require_arch=(str(args.require_arch) if args.require_arch else None),
        require_feature_id=(str(args.require_feature_id) if args.require_feature_id else None),
    )
    if not metas:
        raise RuntimeError("no checkpoint found after filtering")

    device = _pick_device(str(args.device))
    print(f"[INFO] device={device}")
    print(f"[INFO] selected_ckpts={len(metas)}")
    for i, m in enumerate(metas, start=1):
        print(
            f"  [{i}] {m.path}  mtime={datetime.fromtimestamp(m.mtime).isoformat(timespec='seconds')} "
            f"upd={m.upd} arch={m.arch_name} feature={m.feature_id}"
        )

    summary_rows: list[dict[str, object]] = []
    turn_dfs: list[pd.DataFrame] = []

    for i, meta in enumerate(metas, start=1):
        print(f"[{i}/{len(metas)}] evaluating {meta.path.name}")
        row, turn_df = _eval_one_ckpt(
            meta=meta,
            repo_root=repo_root,
            seed_begin=int(args.seed_begin),
            seed_end=int(args.seed_end),
            t_max=int(args.t_max),
            batch_size=int(args.batch_size),
            device=device,
            action=str(args.action),
            no_pf=bool(args.no_pf),
            verbose_build=bool(args.verbose_build),
        )
        summary_rows.append(row)
        turn_dfs.append(turn_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_mae5", "mean_w_l1"], ascending=[True, True]).reset_index(drop=True)
    turn_df_all = pd.concat(turn_dfs, ignore_index=True)

    csv_summary = out_dir / "summary.csv"
    csv_turn = out_dir / "per_turn.csv"
    summary_df.to_csv(csv_summary, index=False)
    turn_df_all.to_csv(csv_turn, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 1.2 * len(summary_df)), 4.2))
    xs = np.arange(len(summary_df), dtype=np.int32)
    ys = summary_df["mean_mae5"].to_numpy(dtype=np.float64)
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xticklabels(summary_df["ckpt_name"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("mean_mae5 (lower is better)")
    ax.set_title("Checkpoint hidden-parameter estimation")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    png_bar = out_dir / "plot_mean_mae5.png"
    fig.savefig(png_bar, dpi=180)

    if len(summary_df) >= 1:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8.0, 4.6))
        for _, r in summary_df.iterrows():
            sub = turn_df_all[turn_df_all["ckpt"] == r["ckpt"]].sort_values("turn")
            x = sub["turn"].to_numpy(dtype=np.int32)
            y = sub["mean_mae5"].to_numpy(dtype=np.float64)
            ax2.plot(x, y, linewidth=1.6, label=str(r["ckpt_name"]))
        ax2.set_xlabel("turn")
        ax2.set_ylabel("mean_mae5")
        ax2.set_title("Per-turn hidden-parameter estimation error")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best", fontsize="small", frameon=False)
        fig2.tight_layout()
        png_turn = out_dir / "plot_per_turn_mae5.png"
        fig2.savefig(png_turn, dpi=180)
    else:
        png_turn = None

    summary_json = out_dir / "summary.json"
    best = summary_df.iloc[0].to_dict() if len(summary_df) > 0 else None
    summary_json.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "out_dir": str(out_dir),
                "n_ckpts": int(len(summary_df)),
                "best": best,
                "args": vars(args),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] summary_csv: {csv_summary}")
    print(f"[OK] per_turn_csv: {csv_turn}")
    print(f"[OK] mae5_bar_png: {png_bar}")
    if png_turn is not None:
        print(f"[OK] mae5_turn_png: {png_turn}")
    print(f"[OK] summary_json: {summary_json}")


if __name__ == "__main__":
    main()
