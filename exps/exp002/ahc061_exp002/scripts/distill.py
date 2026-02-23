from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..ckpt import ModelSpec, TrainSpec, ckpt_model_dict, normalize_state_dict_keys, model_spec_from_ckpt, torch_load_maybe_weights_only
from ..env import BatchEnv
from ..models import build_policy_value_model, masked_logits

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

AUTO_DDP_LAUNCHED_ENV = "AHC061_EXP002_DISTILL_AUTO_DDP_LAUNCHED"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    if isinstance(m, DDP):
        m = m.module
    # torch.compile wraps modules into an OptimizedModule that holds the original module in `_orig_mod`.
    orig = getattr(m, "_orig_mod", None)
    return orig if isinstance(orig, torch.nn.Module) else m


def _pick_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum_tensor(x: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _all_reduce_max_tensor(x: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def _set_seeds(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def _ema_decay_to_name(decay: float) -> str:
    decay_str = format(float(decay), ".10f").rstrip("0").rstrip(".")
    if decay_str == "":
        decay_str = "0"
    return f"ema_decay_{decay_str.replace('.', 'p')}"


def _ema_decay_to_suffix(decay: float) -> str:
    decay_str = format(float(decay), ".10f").rstrip("0").rstrip(".")
    if decay_str == "":
        decay_str = "0"
    return decay_str.replace(".", "p")


def _ema_out_path(base_path: Path, decay: float) -> Path:
    # Example: output.pt -> output_pt_ema_0p999.pt
    base_name = base_path.name.replace(".", "_")
    return base_path.with_name(f"{base_name}_ema_{_ema_decay_to_suffix(decay)}.pt")


def _split_path_stem_suffix(base_path: Path) -> tuple[str, str]:
    name = base_path.name
    compound_suffixes = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tar.lzma")
    for cs in compound_suffixes:
        if name.endswith(cs):
            return name[: -len(cs)], cs
    suffix = str(base_path.suffix)
    if suffix == "":
        return name, ".pt"
    return str(base_path.stem), suffix


def _sgdr_cycle_out_path(base_path: Path, *, cycle_idx: int, upd: int) -> Path:
    stem, suffix = _split_path_stem_suffix(base_path)
    return base_path.with_name(f"{stem}_sgdr_cycle{int(cycle_idx):04d}_upd{int(upd):04d}{suffix}")


class _WarmupThenSgdrScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        t0_steps: int,
        t_mult: int,
        eta_min: float,
    ) -> None:
        self._optimizer = optimizer
        self._warmup_steps = int(max(0, warmup_steps))
        self._step_count = 0
        self._base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        self._t_mult = int(max(1, t_mult))
        self._next_cycle_len = int(max(1, t0_steps))
        self._next_cycle_end_post = int(self._next_cycle_len)
        self._next_cycle_idx = 1
        self._sgdr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(t0_steps),
            T_mult=int(self._t_mult),
            eta_min=float(eta_min),
        )

    def step(self) -> int | None:
        self._step_count += 1
        if self._warmup_steps > 0 and self._step_count <= self._warmup_steps:
            scale = float(self._step_count) / float(self._warmup_steps)
            for i, pg in enumerate(self._optimizer.param_groups):
                pg["lr"] = self._base_lrs[i] * scale
            return None
        post_warmup_step = self._step_count - self._warmup_steps
        if post_warmup_step <= 0:
            return None
        self._sgdr.step(float(post_warmup_step - 1))
        if int(post_warmup_step) == int(self._next_cycle_end_post):
            ended_idx = int(self._next_cycle_idx)
            self._next_cycle_idx += 1
            self._next_cycle_len *= int(self._t_mult)
            self._next_cycle_end_post += int(self._next_cycle_len)
            return ended_idx
        return None


def _validate_ema_decays(decays: list[float], *, source: str) -> list[float]:
    out: list[float] = []
    seen_names: set[str] = set()
    for i, d in enumerate(decays):
        v = float(d)
        if not (0.0 < v < 1.0):
            raise RuntimeError(f"{source}: decay at index {i} must be in (0, 1), got {v}")
        name = _ema_decay_to_name(v)
        if name in seen_names:
            raise RuntimeError(f"{source}: duplicate decay after normalization: {v}")
        seen_names.add(name)
        out.append(v)
    return out


def _parse_ema_decays_arg(raw: str) -> list[float]:
    s = str(raw).strip()
    if s == "" or s.lower() == "off":
        return []
    parts = [p.strip() for p in s.split(",")]
    if any(p == "" for p in parts):
        raise RuntimeError("--ema-decays contains an empty item")
    vals: list[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError as e:
            raise RuntimeError(f"--ema-decays contains a non-float item: {p!r}") from e
    return _validate_ema_decays(vals, source="--ema-decays")


@dataclass
class EmaTracker:
    name: str
    decay: float
    model: torch.nn.Module
    steps: int
    param_pairs: list[tuple[torch.Tensor, torch.Tensor]]
    buf_pairs: list[tuple[torch.Tensor, torch.Tensor]]


def _build_student_arch_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    name = str(args.student_arch).lower()
    if "ppconcat" not in name:
        return {}
    out: dict[str, Any] = {}
    if args.student_pp_player_hidden is not None:
        out["player_hidden_channels"] = int(args.student_pp_player_hidden)
    if args.student_pp_set_layers is not None:
        out["player_set_layers"] = int(args.student_pp_set_layers)
    if args.student_pp_set_heads is not None:
        out["player_set_heads"] = int(args.student_pp_set_heads)
    if args.student_pp_set_ff_mult is not None:
        out["player_set_ff_mult"] = float(args.student_pp_set_ff_mult)
    if args.student_pp_set_dropout is not None:
        out["player_set_dropout"] = float(args.student_pp_set_dropout)
    if args.student_pp_set_every is not None:
        out["player_set_every"] = int(args.student_pp_set_every)
    return out


def _validate_student_arch_cli_args(args: argparse.Namespace) -> None:
    if args.student_pp_player_hidden is not None and int(args.student_pp_player_hidden) <= 0:
        raise RuntimeError("--student-pp-player-hidden must be >= 1")
    if args.student_pp_set_layers is not None and int(args.student_pp_set_layers) < 0:
        raise RuntimeError("--student-pp-set-layers must be >= 0")
    if args.student_pp_set_heads is not None and int(args.student_pp_set_heads) <= 0:
        raise RuntimeError("--student-pp-set-heads must be >= 1")
    if args.student_pp_set_ff_mult is not None and float(args.student_pp_set_ff_mult) <= 0.0:
        raise RuntimeError("--student-pp-set-ff-mult must be > 0")
    if args.student_pp_set_dropout is not None:
        d = float(args.student_pp_set_dropout)
        if not (0.0 <= d < 1.0):
            raise RuntimeError("--student-pp-set-dropout must be in [0,1)")
    if args.student_pp_set_every is not None and int(args.student_pp_set_every) <= 0:
        raise RuntimeError("--student-pp-set-every must be >= 1")


def _build_tta_perm(n: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    cell_max = int(n) * int(n)
    perm = torch.empty((8, cell_max), dtype=torch.int64)
    for flip in range(2):
        for rot in range(4):
            k = flip * 4 + rot
            for x in range(n):
                for y in range(n):
                    tx = int(x)
                    ty = int(y)
                    if flip != 0:
                        ty = int(n - 1 - ty)
                    for _ in range(rot):
                        nx = int(ty)
                        ny = int(n - 1 - tx)
                        tx = nx
                        ty = ny
                    src = int(x) * int(n) + int(y)
                    dst = int(tx) * int(n) + int(ty)
                    perm[k, src] = int(dst)
    inv = torch.empty_like(perm)
    idx = torch.arange(cell_max, dtype=torch.int64)
    for k in range(8):
        inv[k, perm[k]] = idx
    return perm, inv


_TTA_PERM_CPU, _TTA_INV_PERM_CPU = _build_tta_perm()


@torch.inference_mode()
def _collect_teacher_targets_pair(
    env: BatchEnv,
    teacher: torch.nn.Module,
    device: torch.device,
    t_max: int,
    *,
    feature_id_teacher: str,
    feature_id_student: str,
    sample: bool,
    amp: bool,
    teacher_tta_mode: int,
    teacher_tta_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = env.batch_size
    c_t = env.feature_channels_of(feature_id_teacher)
    c_s = env.feature_channels_of(feature_id_student)

    obs_s = torch.empty((t_max, bsz, c_s, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((t_max, bsz, 100), dtype=torch.uint8, device="cpu")
    teacher_logits = torch.empty((t_max, bsz, 100), dtype=torch.float32, device="cpu")
    teacher_values = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    opp_move_dist = torch.empty((t_max, bsz, int(env.spec.m_max), 100), dtype=torch.float32, device="cpu")
    opp_param = torch.empty((t_max, bsz, int(env.spec.m_max), 5), dtype=torch.float32, device="cpu")
    opp_valid = torch.empty((t_max, bsz, int(env.spec.m_max)), dtype=torch.uint8, device="cpu")

    rewards = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    dones = torch.empty((t_max, bsz), dtype=torch.uint8, device="cpu")
    next_obs_t = torch.empty((bsz, c_t, 10, 10), dtype=torch.float32, device="cpu")
    next_obs_s = torch.empty((bsz, c_s, 10, 10), dtype=torch.float32, device="cpu")
    next_mask = torch.empty((bsz, 100), dtype=torch.uint8, device="cpu")

    use_cuda = device.type == "cuda"
    use_amp = use_cuda and bool(amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
    tta_mode = int(teacher_tta_mode)
    tta_k = int(teacher_tta_k) if tta_mode != 0 else 1
    use_tta = tta_mode != 0
    if use_tta:
        tta_dev = device if use_cuda else torch.device("cpu")
        tta_perm = _TTA_PERM_CPU[:tta_k].to(device=tta_dev)
        tta_inv = _TTA_INV_PERM_CPU[:tta_k].to(device=tta_dev)
    else:
        tta_perm = None
        tta_inv = None

    if use_cuda:
        board_dev = torch.empty((bsz, c_t, 10, 10), dtype=torch.float32, device=device)
        mask_dev = torch.empty((bsz, 100), dtype=torch.uint8, device=device)

    obs_t0 = torch.empty((bsz, c_t, 10, 10), dtype=torch.float32, device="cpu")
    env.observe_pair_into(
        obs_t0,
        obs_s[0],
        mask[0],
        feature_id_a=feature_id_teacher,
        feature_id_b=feature_id_student,
    )
    env.aux_targets_into(opp_move_dist[0], opp_param[0], opp_valid[0])

    teacher.eval()
    for t in range(t_max):
        if use_cuda:
            board_dev.copy_(obs_t0 if t == 0 else next_obs_t)
            mask_dev.copy_(mask[t])
            board = board_dev
            m = mask_dev
        else:
            board = obs_t0 if t == 0 else next_obs_t
            m = mask[t]

        if not use_tta:
            with autocast:
                logits, v, _, _ = teacher(board)
            logits = masked_logits(logits.float(), m)
            v = v.float()
        else:
            assert tta_perm is not None and tta_inv is not None
            board_flat = board.reshape(bsz, c_t, 100)
            acc_logp: torch.Tensor | None = None
            v = torch.zeros((bsz,), dtype=torch.float32, device=board.device)
            for tk in range(tta_k):
                idx_inv = tta_inv[tk]
                idx_perm = tta_perm[tk]
                board_t = board_flat.index_select(2, idx_inv).reshape(bsz, c_t, 10, 10)
                mask_t = m.index_select(1, idx_inv)
                with autocast:
                    logits_t, v_t, _, _ = teacher(board_t)
                logits_t = masked_logits(logits_t.float(), mask_t)
                v = v + v_t.float()
                logp_t = torch.log_softmax(logits_t, dim=1)
                logp_orig = logp_t.index_select(1, idx_perm)
                if acc_logp is None:
                    acc_logp = logp_orig
                elif tta_mode == 1:
                    acc_logp = torch.logaddexp(acc_logp, logp_orig)
                else:
                    acc_logp = acc_logp + logp_orig
            assert acc_logp is not None
            logits = masked_logits(acc_logp, m)
            v = v / float(tta_k)

        teacher_logits[t].copy_(logits.to("cpu"))
        teacher_values[t].copy_(v.to("cpu"))

        if sample:
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(1).to("cpu")
        else:
            actions = torch.argmax(logits, dim=1).to("cpu")

        if t + 1 < t_max:
            env.step_observe_pair_into(
                actions,
                next_obs_t,
                obs_s[t + 1],
                mask[t + 1],
                rewards[t],
                dones[t],
                feature_id_a=feature_id_teacher,
                feature_id_b=feature_id_student,
            )
            env.aux_targets_into(opp_move_dist[t + 1], opp_param[t + 1], opp_valid[t + 1])
        else:
            env.step_observe_pair_into(
                actions,
                next_obs_t,
                next_obs_s,
                next_mask,
                rewards[t],
                dones[t],
                feature_id_a=feature_id_teacher,
                feature_id_b=feature_id_student,
            )

    return obs_s, mask, teacher_logits, teacher_values, opp_move_dist, opp_param, opp_valid


def _forward_student_with_tta(
    student: torch.nn.Module,
    x: torch.Tensor,
    m: torch.Tensor,
    *,
    autocast: Any,
    tta_mode: int,
    tta_k: int,
    tta_perm: torch.Tensor | None,
    tta_inv: torch.Tensor | None,
    aux_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if tta_mode == 0:
        with autocast:
            slogits, svalue, s_opp_move_logits, s_opp_param_logits = student(x)
        slogits = masked_logits(slogits.float(), m)
        svalue = svalue.float()
        if aux_mode == "none":
            return slogits, svalue, None, None
        return slogits, svalue, s_opp_move_logits.float(), s_opp_param_logits.float()

    if tta_perm is None or tta_inv is None:
        raise RuntimeError("[TTA] internal error: student tta perm/inv is not initialized")
    if aux_mode not in ("all", "first", "none"):
        raise RuntimeError(f"[TTA] unknown student aux mode: {aux_mode!r}")

    bsz = int(x.shape[0])
    channels = int(x.shape[1])
    x_flat = x.reshape(bsz, channels, 100)
    acc_logp: torch.Tensor | None = None
    acc_value = torch.zeros((bsz,), dtype=torch.float32, device=x.device)
    acc_opp_move: torch.Tensor | None = None
    acc_opp_param: torch.Tensor | None = None

    for tk in range(int(tta_k)):
        idx_inv = tta_inv[tk]
        idx_perm = tta_perm[tk]
        x_t = x_flat.index_select(2, idx_inv).reshape(bsz, channels, 10, 10)
        m_t = m.index_select(1, idx_inv)

        with autocast:
            slogits_t, svalue_t, s_opp_move_logits_t, s_opp_param_logits_t = student(x_t)
        slogits_t = masked_logits(slogits_t.float(), m_t)
        logp_t = torch.log_softmax(slogits_t, dim=1)
        logp_orig = logp_t.index_select(1, idx_perm)
        if acc_logp is None:
            acc_logp = logp_orig
        elif int(tta_mode) == 1:
            acc_logp = torch.logaddexp(acc_logp, logp_orig)
        else:
            acc_logp = acc_logp + logp_orig
        acc_value = acc_value + svalue_t.float()

        if aux_mode == "all":
            om = s_opp_move_logits_t.float().index_select(2, idx_perm)
            op = s_opp_param_logits_t.float()
            if acc_opp_move is None:
                acc_opp_move = om
                acc_opp_param = op
            else:
                acc_opp_move = acc_opp_move + om
                acc_opp_param = acc_opp_param + op
        elif aux_mode == "first" and tk == 0:
            acc_opp_move = s_opp_move_logits_t.float().index_select(2, idx_perm)
            acc_opp_param = s_opp_param_logits_t.float()

    if acc_logp is None:
        raise RuntimeError("[TTA] internal error: student TTA accumulated logits is empty")

    slogits = masked_logits(acc_logp, m)
    svalue = acc_value / float(tta_k)
    if aux_mode == "all":
        if acc_opp_move is None or acc_opp_param is None:
            raise RuntimeError("[TTA] internal error: student TTA aux accumulator is empty")
        acc_opp_move = acc_opp_move / float(tta_k)
        acc_opp_param = acc_opp_param / float(tta_k)
    return slogits, svalue, acc_opp_move, acc_opp_param


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--teacher-feature-id", type=str, default=None)
    parser.add_argument("--teacher-tta-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--teacher-tta-k",
        type=int,
        choices=(2, 4, 8),
        default=8,
        help="Number of D4 transforms for teacher TTA when --teacher-tta-mode is 1 or 2.",
    )
    parser.add_argument(
        "--teacher-rollout-mode",
        type=str,
        choices=("sample", "greedy"),
        default="sample",
        help="Teacher action selection while collecting distillation trajectories.",
    )
    parser.add_argument("--student-tta-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--student-tta-k", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument(
        "--student-tta-aux",
        type=str,
        choices=("all", "first", "none"),
        default="all",
        help="Aux-head aggregation under student TTA: all=aggregate all views, first=use view-0 only, none=disable aux losses.",
    )

    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument(
        "--init-ckpt-load-mode",
        type=str,
        choices=("strict", "match"),
        default="strict",
        help="strict: require full architectural match (legacy behavior), match: load only keys with matching name/shape.",
    )
    parser.add_argument(
        "--init-ckpt-match-allow-empty",
        action="store_true",
        help="Allow --init-ckpt-load-mode match to proceed even when zero keys are loaded.",
    )
    parser.add_argument("--student-feature-id", type=str, default=None)
    parser.add_argument("--student-arch", type=str, default="dwres_v1")
    parser.add_argument("--student-hidden", type=int, default=64)
    parser.add_argument("--student-blocks", type=int, default=16)
    parser.add_argument("--student-pp-player-hidden", type=int, default=None)
    parser.add_argument("--student-pp-set-layers", type=int, default=None)
    parser.add_argument("--student-pp-set-heads", type=int, default=None)
    parser.add_argument("--student-pp-set-ff-mult", type=float, default=None)
    parser.add_argument("--student-pp-set-dropout", type=float, default=None)
    parser.add_argument("--student-pp-set-every", type=int, default=None)

    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay (>= 0).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (L2). Set 0 to disable clipping.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--policy-distill-mode",
        type=str,
        choices=("soft", "top1"),
        default="soft",
        help="Policy distillation target: soft uses teacher distribution, top1 uses teacher argmax class labels.",
    )
    parser.add_argument(
        "--top1-margin",
        type=float,
        default=0.0,
        help="Margin target m in relu(m - (logit_true - logit_best_other)) for top1 policy distillation.",
    )
    parser.add_argument(
        "--top1-margin-coef",
        type=float,
        default=0.0,
        help="Weight for top1 margin loss term; effective when --policy-distill-mode top1.",
    )
    parser.add_argument(
        "--value-distill-mode",
        type=str,
        choices=("on", "off"),
        default="on",
        help="Value distillation loss switch: on uses MSE to teacher value, off disables value loss term.",
    )
    parser.add_argument("--w-policy", type=float, default=1.0)
    parser.add_argument("--w-value", type=float, default=1.0)
    parser.add_argument("--w-aux-opp-move", type=float, default=0.05)
    parser.add_argument("--w-aux-opp-param", type=float, default=0.01)

    parser.add_argument("--scheduler", type=str, choices=("cosine", "sgdr", "none"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--cosine-cycles", type=float, default=0.5)
    parser.add_argument(
        "--sgdr-t0-steps",
        type=int,
        default=None,
        help="SGDR first cycle length in optimizer steps. Required when --scheduler sgdr.",
    )
    parser.add_argument("--sgdr-t-mult", type=int, default=2, help="SGDR cycle growth factor (>=1).")
    parser.add_argument("--sgdr-eta-min", type=float, default=0.0, help="SGDR minimum learning rate.")
    parser.add_argument(
        "--ema-decays",
        type=str,
        default="0.999",
        help='Comma-separated EMA decays (e.g. "0.995,0.999"), or "off" to disable EMA.',
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--distributed",
        type=str,
        choices=("auto", "off", "on"),
        default="auto",
        help="Distributed launch mode. auto: launch torchrun automatically when multiple CUDA devices are visible.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-teacher", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")

    parser.add_argument(
        "--out-ckpt",
        type=str,
        default="exps/exp002/artifacts/checkpoints_distill/ckpt_distill.pt",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Overwrite --out-ckpt every N updates with the same format as final save (0 disables periodic saves).",
    )

    parser.add_argument("--wandb-project", type=str, default="ahc061-exp002")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="distill")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-log-ckpt", action="store_true")
    args = parser.parse_args()

    if args.updates <= 0:
        raise RuntimeError("--updates must be >= 1")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be >= 1")
    if args.epochs <= 0:
        raise RuntimeError("--epochs must be >= 1")
    if args.minibatch <= 0:
        raise RuntimeError("--minibatch must be >= 1")
    if int(args.save_every) < 0:
        raise RuntimeError("--save-every must be >= 0")
    if args.tau <= 0:
        raise RuntimeError("--tau must be > 0")
    if not math.isfinite(float(args.weight_decay)) or float(args.weight_decay) < 0.0:
        raise RuntimeError("--weight-decay must be finite and >= 0")
    if not math.isfinite(float(args.max_grad_norm)) or float(args.max_grad_norm) < 0.0:
        raise RuntimeError("--max-grad-norm must be finite and >= 0")
    scheduler_mode = str(args.scheduler)
    if scheduler_mode != "none":
        if float(args.warmup_ratio) < 0.0 or float(args.warmup_ratio) > 1.0:
            raise RuntimeError("--warmup-ratio must be in [0, 1]")
        if args.warmup_steps is not None and int(args.warmup_steps) < 0:
            raise RuntimeError("--warmup-steps must be >= 0")
    if scheduler_mode == "cosine":
        if float(args.cosine_cycles) <= 0.0:
            raise RuntimeError("--cosine-cycles must be > 0")
    if scheduler_mode == "sgdr":
        if args.sgdr_t0_steps is None:
            raise RuntimeError("--sgdr-t0-steps is required when --scheduler sgdr")
        if int(args.sgdr_t0_steps) <= 0:
            raise RuntimeError("--sgdr-t0-steps must be >= 1")
        if int(args.sgdr_t_mult) < 1:
            raise RuntimeError("--sgdr-t-mult must be >= 1")
        if not math.isfinite(float(args.sgdr_eta_min)):
            raise RuntimeError("--sgdr-eta-min must be finite")
        if float(args.sgdr_eta_min) < 0.0:
            raise RuntimeError("--sgdr-eta-min must be >= 0")
        if float(args.sgdr_eta_min) > float(args.lr):
            raise RuntimeError("--sgdr-eta-min must be <= --lr")
    if str(args.policy_distill_mode) not in ("soft", "top1"):
        raise RuntimeError("--policy-distill-mode must be one of {soft,top1}")
    if float(args.top1_margin) < 0.0:
        raise RuntimeError("--top1-margin must be >= 0")
    if float(args.top1_margin_coef) < 0.0:
        raise RuntimeError("--top1-margin-coef must be >= 0")
    if str(args.value_distill_mode) not in ("on", "off"):
        raise RuntimeError("--value-distill-mode must be one of {on,off}")
    if float(args.w_aux_opp_move) < 0.0 or float(args.w_aux_opp_param) < 0.0:
        raise RuntimeError("--w-aux-opp-move/--w-aux-opp-param must be >= 0")
    if int(args.teacher_tta_mode) not in (0, 1, 2):
        raise RuntimeError("--teacher-tta-mode must be one of {0,1,2}")
    if int(args.teacher_tta_mode) != 0 and int(args.teacher_tta_k) not in (2, 4, 8):
        raise RuntimeError("--teacher-tta-k must be one of {2,4,8} when --teacher-tta-mode != 0")
    if str(args.teacher_rollout_mode) not in ("sample", "greedy"):
        raise RuntimeError("--teacher-rollout-mode must be one of {sample,greedy}")
    if int(args.student_tta_mode) not in (0, 1, 2):
        raise RuntimeError("--student-tta-mode must be one of {0,1,2}")
    if int(args.student_tta_mode) != 0 and int(args.student_tta_k) not in (2, 4, 8):
        raise RuntimeError("--student-tta-k must be one of {2,4,8} when --student-tta-mode != 0")
    if str(args.student_tta_aux) not in ("all", "first", "none"):
        raise RuntimeError("--student-tta-aux must be one of {all,first,none}")
    if int(args.teacher_tta_mode) == 0 and int(args.student_tta_mode) != 0:
        raise RuntimeError("--student-tta-mode must be 0 when --teacher-tta-mode is 0")
    _validate_student_arch_cli_args(args)
    ema_decays = _parse_ema_decays_arg(str(args.ema_decays))

    has_dist_env = any(k in os.environ for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if (not has_dist_env) and (str(args.distributed) != "off"):
        visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_cuda > 1:
            if os.environ.get(AUTO_DDP_LAUNCHED_ENV) == "1":
                print("[WARN] auto distributed launch guard is set; continuing in single-process mode.")
            else:
                launch_cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={visible_cuda}",
                    "-m",
                    "exps.exp002.ahc061_exp002.scripts.distill",
                    *sys.argv[1:],
                ]
                print(f"[AUTO-DDP] relaunch with torchrun ({visible_cuda} GPUs visible)")
                launch_env = os.environ.copy()
                launch_env[AUTO_DDP_LAUNCHED_ENV] = "1"
                rc = subprocess.run(launch_cmd, env=launch_env, check=False).returncode
                raise SystemExit(rc)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed and str(args.distributed) == "off":
        raise RuntimeError("--distributed off cannot be used with torchrun (WORLD_SIZE>1)")
    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed distillation requires CUDA")
        if args.device not in ("auto", "cuda") and (not str(args.device).startswith("cuda")):
            raise RuntimeError("Distributed distillation requires --device auto/cuda/cuda:N")
        dist.init_process_group(backend="nccl")
    is_main_process = rank == 0

    def pick_device() -> torch.device:
        if is_distributed:
            if args.device == "auto" or args.device == "cuda":
                return torch.device(f"cuda:{local_rank}")
            req = torch.device(args.device)
            if req.type != "cuda":
                raise RuntimeError("Distributed distillation requires CUDA device")
            if req.index is not None and int(req.index) != int(local_rank):
                raise RuntimeError(
                    f"--device={args.device!r} conflicts with LOCAL_RANK={local_rank}. "
                    f"Use --device auto/cuda with torchrun."
                )
            return torch.device(f"cuda:{local_rank}")
        return _pick_device(str(args.device))

    device = pick_device()
    if device.type == "cuda":
        if device.index is None:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_amp = device.type == "cuda" and (not args.no_amp)
    student_tta_mode = int(args.student_tta_mode)
    student_tta_k = int(args.student_tta_k) if student_tta_mode != 0 else 1
    student_tta_aux = str(args.student_tta_aux)
    teacher_rollout_mode = str(args.teacher_rollout_mode)
    teacher_rollout_sample = teacher_rollout_mode == "sample"
    use_student_tta = student_tta_mode != 0
    if use_student_tta:
        student_tta_perm = _TTA_PERM_CPU[:student_tta_k].to(device=device)
        student_tta_inv = _TTA_INV_PERM_CPU[:student_tta_k].to(device=device)
    else:
        student_tta_perm = None
        student_tta_inv = None

    if int(args.batch_size) % int(world_size) != 0:
        raise RuntimeError(f"--batch-size must be divisible by WORLD_SIZE ({args.batch_size} vs {world_size})")
    if int(args.minibatch) % int(world_size) != 0:
        raise RuntimeError(f"--minibatch must be divisible by WORLD_SIZE ({args.minibatch} vs {world_size})")
    local_batch_size = int(args.batch_size) // int(world_size)
    local_minibatch = int(args.minibatch) // int(world_size)
    if local_batch_size <= 0 or local_minibatch <= 0:
        raise RuntimeError("local batch/minibatch must be >= 1")

    seed_base = int(args.seed) + int(rank) * 1000003
    _set_seeds(seed_base, device)

    repo_root = Path(__file__).resolve().parents[4]

    out_path = Path(args.out_ckpt)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_ckpt_path = Path(args.teacher_ckpt)
    if not teacher_ckpt_path.is_absolute():
        teacher_ckpt_path = (repo_root / teacher_ckpt_path).resolve()
    teacher_ckpt = torch_load_maybe_weights_only(teacher_ckpt_path)

    t_ms = model_spec_from_ckpt(teacher_ckpt)
    teacher_arch_kwargs = dict(t_ms.arch_kwargs)
    teacher_feature_id = str(args.teacher_feature_id) if args.teacher_feature_id is not None else str(t_ms.feature_id)
    student_feature_id = str(args.student_feature_id) if args.student_feature_id is not None else teacher_feature_id
    student_arch_kwargs = _build_student_arch_kwargs_from_args(args)

    env = BatchEnv(batch_size=int(local_batch_size), feature_id=teacher_feature_id, pf_enabled=not args.no_pf, verbose_build=False)
    c_teacher = env.feature_channels_of(teacher_feature_id)
    c_student = env.feature_channels_of(student_feature_id)
    if int(t_ms.in_channels) != int(c_teacher):
        raise RuntimeError(f"[TEACHER] in_channels mismatch: ckpt={int(t_ms.in_channels)} env={int(c_teacher)}")

    teacher = build_policy_value_model(
        t_ms.arch_name,
        in_channels=int(c_teacher),
        hidden_channels=int(t_ms.hidden),
        blocks=int(t_ms.blocks),
        feature_id=str(teacher_feature_id),
        arch_kwargs=dict(teacher_arch_kwargs),
    ).to(device)
    missing, unexpected = teacher.load_state_dict(normalize_state_dict_keys(teacher_ckpt["model"]), strict=False)
    if unexpected:
        raise RuntimeError(f"[TEACHER] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    allowed_missing_prefixes = ("opp_move_head.", "opp_param_head.")
    bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
    if bad_missing:
        raise RuntimeError(f"[TEACHER] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
    if missing and is_main_process:
        print(f"[TEACHER] loaded with missing aux head keys: {len(missing)}")
    teacher.eval()

    student = build_policy_value_model(
        str(args.student_arch),
        in_channels=int(c_student),
        hidden_channels=int(args.student_hidden),
        blocks=int(args.student_blocks),
        feature_id=str(student_feature_id),
        arch_kwargs=dict(student_arch_kwargs),
    ).to(device)

    init_ckpt_path: Path | None = None
    init_ckpt: dict[str, Any] | None = None
    init_ckpt_mode = str(args.init_ckpt_load_mode)
    init_load_stats: dict[str, int | str | None] = {
        "init_ckpt": None,
        "init_ckpt_load_mode": str(init_ckpt_mode),
        "loaded_keys": 0,
        "student_total_keys": 0,
        "init_ckpt_total_keys": 0,
        "skipped_missing_key": 0,
        "skipped_shape_mismatch": 0,
        "skipped_dtype_mismatch": 0,
    }
    if args.init_ckpt is not None:
        init_ckpt_path = Path(args.init_ckpt)
        if not init_ckpt_path.is_absolute():
            init_ckpt_path = (repo_root / init_ckpt_path).resolve()
        init_ckpt = torch_load_maybe_weights_only(init_ckpt_path)
        init_sd = normalize_state_dict_keys(init_ckpt["model"])
        student_sd = student.state_dict()

        init_load_stats["init_ckpt"] = str(init_ckpt_path)
        init_load_stats["student_total_keys"] = int(len(student_sd))
        init_load_stats["init_ckpt_total_keys"] = int(len(init_sd))

        if init_ckpt_mode == "strict":
            ims = model_spec_from_ckpt(init_ckpt)
            if str(ims.arch_name) != str(args.student_arch):
                raise RuntimeError(f"[INIT] arch_name mismatch: ckpt={ims.arch_name!r} student={str(args.student_arch)!r}")
            if str(ims.feature_id) != str(student_feature_id):
                raise RuntimeError(f"[INIT] feature_id mismatch: ckpt={ims.feature_id!r} student={str(student_feature_id)!r}")
            if int(ims.in_channels) != int(c_student):
                raise RuntimeError(f"[INIT] in_channels mismatch: ckpt={int(ims.in_channels)} env={int(c_student)}")
            if int(ims.hidden) != int(args.student_hidden) or int(ims.blocks) != int(args.student_blocks):
                raise RuntimeError(
                    f"[INIT] model size mismatch: ckpt hidden/blocks={int(ims.hidden)}/{int(ims.blocks)} "
                    f"student={int(args.student_hidden)}/{int(args.student_blocks)}"
                )
            if dict(ims.arch_kwargs) != dict(student_arch_kwargs):
                raise RuntimeError(
                    f"[INIT] arch_kwargs mismatch: ckpt={dict(ims.arch_kwargs)!r} student={dict(student_arch_kwargs)!r}"
                )

            missing, unexpected = student.load_state_dict(init_sd, strict=False)
            if unexpected:
                raise RuntimeError(f"[INIT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
            allowed_missing_prefixes = ("opp_move_head.", "opp_param_head.")
            bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
            if bad_missing:
                raise RuntimeError(f"[INIT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
            if missing and is_main_process:
                print(f"[INIT] loaded with missing aux head keys: {len(missing)}")
            init_load_stats["loaded_keys"] = int(len(student_sd) - len(missing))
            init_load_stats["skipped_missing_key"] = int(max(0, len(init_sd) - int(init_load_stats["loaded_keys"])))
            if is_main_process:
                print(
                    "[INIT] strict loaded "
                    f"{int(init_load_stats['loaded_keys'])}/{len(student_sd)} student keys "
                    f"(init keys={len(init_sd)})"
                )
        else:
            matched_state: dict[str, torch.Tensor] = {}
            skipped_missing_key = 0
            skipped_shape_mismatch = 0
            skipped_dtype_mismatch = 0
            example_missing: list[str] = []
            example_shape: list[str] = []
            example_dtype: list[str] = []
            max_examples = 20

            for k, v in init_sd.items():
                dst = student_sd.get(k, None)
                if dst is None:
                    skipped_missing_key += 1
                    if len(example_missing) < max_examples:
                        example_missing.append(str(k))
                    continue
                if tuple(v.shape) != tuple(dst.shape):
                    skipped_shape_mismatch += 1
                    if len(example_shape) < max_examples:
                        example_shape.append(f"{k}: init={tuple(v.shape)} student={tuple(dst.shape)}")
                    continue
                if v.dtype == dst.dtype:
                    matched_state[k] = v
                    continue
                if torch.is_floating_point(v) and torch.is_floating_point(dst):
                    matched_state[k] = v.to(dtype=dst.dtype)
                    continue
                skipped_dtype_mismatch += 1
                if len(example_dtype) < max_examples:
                    example_dtype.append(f"{k}: init={v.dtype} student={dst.dtype}")

            loaded_keys = int(len(matched_state))
            if loaded_keys == 0 and not bool(args.init_ckpt_match_allow_empty):
                raise RuntimeError(
                    "[INIT] --init-ckpt-load-mode match loaded 0 keys. "
                    "Use --init-ckpt-match-allow-empty to allow this."
                )

            missing, unexpected = student.load_state_dict(matched_state, strict=False)
            if unexpected:
                raise RuntimeError(f"[INIT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

            init_load_stats["loaded_keys"] = loaded_keys
            init_load_stats["skipped_missing_key"] = int(skipped_missing_key)
            init_load_stats["skipped_shape_mismatch"] = int(skipped_shape_mismatch)
            init_load_stats["skipped_dtype_mismatch"] = int(skipped_dtype_mismatch)

            if is_main_process:
                print(
                    "[INIT] match loaded "
                    f"{loaded_keys}/{len(student_sd)} student keys "
                    f"(init keys={len(init_sd)}, missing={skipped_missing_key}, "
                    f"shape={skipped_shape_mismatch}, dtype={skipped_dtype_mismatch})"
                )
                if example_missing:
                    print("[INIT] match example missing keys: " + ", ".join(example_missing))
                if example_shape:
                    print("[INIT] match example shape mismatch: " + " | ".join(example_shape))
                if example_dtype:
                    print("[INIT] match example dtype mismatch: " + " | ".join(example_dtype))

    opt_kwargs: dict[str, Any] = {"lr": float(args.lr), "weight_decay": float(args.weight_decay)}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(student.parameters(), **opt_kwargs)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

    scheduler = None
    steps_per_epoch = int(math.ceil((int(env.spec.t_max) * int(args.batch_size)) / float(int(args.minibatch))))
    steps_per_epoch = max(1, steps_per_epoch)
    steps_per_update = int(args.epochs) * steps_per_epoch
    steps_per_update = max(1, steps_per_update)
    total_steps = int(args.updates) * steps_per_update
    total_steps = max(1, total_steps)
    warmup_steps = 0
    if str(args.scheduler) != "none":
        if args.warmup_steps is None:
            warmup_steps = int(round(total_steps * float(args.warmup_ratio)))
        else:
            warmup_steps = int(args.warmup_steps)
        warmup_steps = max(0, min(warmup_steps, total_steps))

    sgdr_t0_steps = int(args.sgdr_t0_steps) if args.sgdr_t0_steps is not None else 0
    sgdr_t_mult = int(args.sgdr_t_mult)
    sgdr_eta_min = float(args.sgdr_eta_min)

    if str(args.scheduler) == "cosine":
        try:
            from transformers import get_cosine_schedule_with_warmup  # type: ignore
        except Exception as e:
            raise RuntimeError(f"failed to import transformers.get_cosine_schedule_with_warmup: {e}") from e

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_steps),
            num_training_steps=int(total_steps),
            num_cycles=float(args.cosine_cycles),
        )
    elif str(args.scheduler) == "sgdr":
        scheduler = _WarmupThenSgdrScheduler(
            optimizer,
            warmup_steps=int(warmup_steps),
            t0_steps=int(sgdr_t0_steps),
            t_mult=int(sgdr_t_mult),
            eta_min=float(sgdr_eta_min),
        )

    run = None
    if args.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError("wandb is not available. Install it to run distill logging, or use --wandb-mode disabled.")
    if args.wandb_mode != "disabled" and is_main_process:
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        wandb_dir = (repo_root / "exps" / "exp002" / "artifacts" / "wandb_distill").resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "algo": "distill",
            "teacher_ckpt": str(teacher_ckpt_path),
            "teacher_feature_id": str(teacher_feature_id),
            "teacher_arch_kwargs": dict(teacher_arch_kwargs),
            "teacher_tta_mode": int(args.teacher_tta_mode),
            "teacher_tta_k": int(args.teacher_tta_k),
            "teacher_tta_k_effective": int(args.teacher_tta_k) if int(args.teacher_tta_mode) != 0 else 1,
            "teacher_rollout_mode": str(teacher_rollout_mode),
            "teacher_rollout_sample": bool(teacher_rollout_sample),
            "student_tta_mode": int(student_tta_mode),
            "student_tta_k": int(args.student_tta_k),
            "student_tta_k_effective": int(student_tta_k),
            "student_tta_aux": str(student_tta_aux),
            "student_feature_id": str(student_feature_id),
            "student_arch": str(args.student_arch),
            "student_arch_kwargs": dict(student_arch_kwargs),
            "student_hidden": int(args.student_hidden),
            "student_blocks": int(args.student_blocks),
            "init_ckpt": str(init_ckpt_path) if init_ckpt_path is not None else None,
            "init_ckpt_load_mode": str(init_load_stats["init_ckpt_load_mode"]),
            "init_loaded_keys": int(init_load_stats["loaded_keys"]),
            "init_student_total_keys": int(init_load_stats["student_total_keys"]),
            "init_ckpt_total_keys": int(init_load_stats["init_ckpt_total_keys"]),
            "init_skipped_missing_key": int(init_load_stats["skipped_missing_key"]),
            "init_skipped_shape_mismatch": int(init_load_stats["skipped_shape_mismatch"]),
            "init_skipped_dtype_mismatch": int(init_load_stats["skipped_dtype_mismatch"]),
            "updates": int(args.updates),
            "batch_size": int(args.batch_size),
            "batch_size_per_rank": int(local_batch_size),
            "t_max": int(env.spec.t_max),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_grad_norm": float(args.max_grad_norm),
            "epochs": int(args.epochs),
            "minibatch": int(args.minibatch),
            "minibatch_per_rank": int(local_minibatch),
            "tau": float(args.tau),
            "policy_distill_mode": str(args.policy_distill_mode),
            "top1_margin": float(args.top1_margin),
            "top1_margin_coef": float(args.top1_margin_coef),
            "value_distill_mode": str(args.value_distill_mode),
            "w_policy": float(args.w_policy),
            "w_value": float(args.w_value),
            "w_aux_opp_move": float(args.w_aux_opp_move),
            "w_aux_opp_param": float(args.w_aux_opp_param),
            "seed": int(args.seed),
            "pf_enabled": bool(not args.no_pf),
            "world_size": int(world_size),
            "rank": int(rank),
            "device": str(device),
            "amp_bf16": bool(use_amp),
            "compile": bool(args.compile),
            "compile_teacher": bool(args.compile_teacher),
            "compile_mode": str(args.compile_mode),
            "scheduler": str(args.scheduler),
            "warmup_steps": int(warmup_steps) if str(args.scheduler) != "none" else 0,
            "total_steps": int(total_steps) if str(args.scheduler) != "none" else 0,
            "steps_per_update": int(steps_per_update),
            "cosine_cycles": float(args.cosine_cycles) if str(args.scheduler) == "cosine" else 0.0,
            "sgdr_t0_steps": int(sgdr_t0_steps) if str(args.scheduler) == "sgdr" else 0,
            "sgdr_t_mult": int(sgdr_t_mult) if str(args.scheduler) == "sgdr" else 0,
            "sgdr_eta_min": float(sgdr_eta_min) if str(args.scheduler) == "sgdr" else 0.0,
            "ema_decays": [float(d) for d in ema_decays],
            "torch_version": torch.__version__,
            "out_ckpt": str(out_path),
            "save_every": int(args.save_every),
        }
        run = wandb.init(
            project=str(args.wandb_project),
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=tags if tags else None,
            mode=str(args.wandb_mode),
            dir=str(wandb_dir),
            config=config,
        )
        wandb.define_metric("upd")
        wandb.define_metric("distill/*", step_metric="upd")
        wandb.define_metric("time/*", step_metric="upd")
        wandb.define_metric("hp/*", step_metric="upd")

    if args.compile and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        try:
            student = torch.compile(student, mode=str(args.compile_mode))
        except Exception as e:
            if is_main_process:
                print(f"[WARN] torch.compile(student) failed ({type(e).__name__}: {e}); falling back to eager.")
        if args.compile_teacher:
            try:
                teacher = torch.compile(teacher, mode=str(args.compile_mode))
            except Exception as e:
                if is_main_process:
                    print(f"[WARN] torch.compile(teacher) failed ({type(e).__name__}: {e}); falling back to eager.")
            teacher.eval()

    if is_distributed:
        ddp_device_id = device.index if device.index is not None else local_rank
        student = DDP(
            student,
            device_ids=[int(ddp_device_id)],
            output_device=int(ddp_device_id),
            broadcast_buffers=False,
            gradient_as_bucket_view=False,
        )

    ema_trackers: list[EmaTracker] = []
    online_named_params = dict(_unwrap_model(student).named_parameters())
    online_named_bufs = dict(_unwrap_model(student).named_buffers())
    for decay in ema_decays:
        ema_name = _ema_decay_to_name(decay)
        ema_model = build_policy_value_model(
            str(args.student_arch),
            in_channels=int(c_student),
            hidden_channels=int(args.student_hidden),
            blocks=int(args.student_blocks),
            feature_id=str(student_feature_id),
            arch_kwargs=dict(student_arch_kwargs),
        ).to(device)
        ema_model.load_state_dict(normalize_state_dict_keys(_unwrap_model(student).state_dict()))
        ema_model.eval()

        ema_named_params = dict(ema_model.named_parameters())
        ema_param_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, ema_p in ema_named_params.items():
            if name not in online_named_params:
                raise RuntimeError(f"[EMA] parameter name mismatch: {name!r} not in online model")
            ema_param_pairs.append((ema_p, online_named_params[name]))

        ema_named_bufs = dict(ema_model.named_buffers())
        ema_buf_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, ema_b in ema_named_bufs.items():
            if name not in online_named_bufs:
                raise RuntimeError(f"[EMA] buffer name mismatch: {name!r} not in online model")
            ema_buf_pairs.append((ema_b, online_named_bufs[name]))

        ema_trackers.append(
            EmaTracker(
                name=ema_name,
                decay=float(decay),
                model=ema_model,
                steps=0,
                param_pairs=ema_param_pairs,
                buf_pairs=ema_buf_pairs,
            )
        )

    def ema_update() -> None:
        with torch.no_grad():
            for tr in ema_trackers:
                for ema_p, p in tr.param_pairs:
                    ema_p.detach().mul_(tr.decay).add_(p.detach(), alpha=1.0 - tr.decay)
                for ema_b, b in tr.buf_pairs:
                    ema_b.detach().copy_(b.detach())
                tr.steps += 1

    tau = float(args.tau)
    policy_distill_mode = str(args.policy_distill_mode)
    top1_margin = float(args.top1_margin)
    top1_margin_coef = float(args.top1_margin_coef)
    value_distill_mode = str(args.value_distill_mode)
    w_pi = float(args.w_policy)
    w_v = float(args.w_value)
    w_opp_move = float(args.w_aux_opp_move)
    w_opp_param = float(args.w_aux_opp_param)
    max_grad_norm = float(args.max_grad_norm)
    save_every = int(args.save_every)

    wandb_run_id = str(run.id) if run is not None else None
    model_spec = ModelSpec(
        arch_name=str(args.student_arch),
        feature_id=student_feature_id,
        in_channels=int(c_student),
        hidden=int(args.student_hidden),
        blocks=int(args.student_blocks),
        arch_kwargs=dict(student_arch_kwargs),
    )
    train_spec = TrainSpec(
        algo="distill",
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        t_max=int(env.spec.t_max),
        lr=float(args.lr),
        warmup_updates=0,
        epochs=int(args.epochs),
        minibatch=int(args.minibatch),
        gamma=1.0,
        gae_lambda=0.0,
        ent_coef=0.0,
        aux_opp_move_coef=float(args.w_aux_opp_move),
        aux_opp_param_coef=float(args.w_aux_opp_param),
        pf_enabled=bool(not args.no_pf),
        amp=bool(use_amp),
        rollout_amp=bool(use_amp),
        world_size=int(world_size),
    )

    def save_ckpt_bundle(*, upd_to_save: int, tag: str, save_path: Path | None = None) -> None:
        if not is_main_process:
            return
        target_path = out_path if save_path is None else save_path
        env_steps = int(upd_to_save) * int(args.batch_size) * int(env.spec.t_max)
        ckpt_out = ckpt_model_dict(
            model_state=normalize_state_dict_keys(_unwrap_model(student).state_dict()),
            upd=int(upd_to_save),
            env_steps=int(env_steps),
            model=model_spec,
            train=train_spec,
            wandb_run_id=wandb_run_id,
            run_dir=None,
        )
        ckpt_out["teacher"] = {
            "ckpt": str(teacher_ckpt_path),
            "arch_name": str(t_ms.arch_name),
            "arch_kwargs": dict(teacher_arch_kwargs),
            "feature_id": str(teacher_feature_id),
            "in_channels": int(c_teacher),
            "hidden": int(t_ms.hidden),
            "blocks": int(t_ms.blocks),
            "upd": int(teacher_ckpt.get("upd", -1)),
        }
        ckpt_out["distill"] = {
            "tau": float(tau),
            "policy_distill_mode": str(policy_distill_mode),
            "top1_margin": float(top1_margin),
            "top1_margin_coef": float(top1_margin_coef),
            "value_distill_mode": str(value_distill_mode),
            "init_ckpt": init_load_stats["init_ckpt"],
            "init_ckpt_load_mode": str(init_load_stats["init_ckpt_load_mode"]),
            "init_loaded_keys": int(init_load_stats["loaded_keys"]),
            "init_student_total_keys": int(init_load_stats["student_total_keys"]),
            "init_ckpt_total_keys": int(init_load_stats["init_ckpt_total_keys"]),
            "init_skipped_missing_key": int(init_load_stats["skipped_missing_key"]),
            "init_skipped_shape_mismatch": int(init_load_stats["skipped_shape_mismatch"]),
            "init_skipped_dtype_mismatch": int(init_load_stats["skipped_dtype_mismatch"]),
            "ema_decays": [float(d) for d in ema_decays],
            "w_policy": float(w_pi),
            "w_value": float(w_v),
            "weight_decay": float(args.weight_decay),
            "w_aux_opp_move": float(w_opp_move),
            "w_aux_opp_param": float(w_opp_param),
            "max_grad_norm": float(max_grad_norm),
            "teacher_tta_mode": int(args.teacher_tta_mode),
            "teacher_tta_k": int(args.teacher_tta_k),
            "teacher_rollout_mode": str(teacher_rollout_mode),
            "teacher_rollout_sample": bool(teacher_rollout_sample),
            "student_tta_mode": int(student_tta_mode),
            "student_tta_k": int(args.student_tta_k),
            "student_tta_aux": str(student_tta_aux),
            "scheduler": str(args.scheduler),
            "warmup_steps": int(warmup_steps) if str(args.scheduler) != "none" else 0,
            "total_steps": int(total_steps) if str(args.scheduler) != "none" else 0,
            "steps_per_update": int(steps_per_update),
            "cosine_cycles": float(args.cosine_cycles) if str(args.scheduler) == "cosine" else 0.0,
            "sgdr_t0_steps": int(sgdr_t0_steps) if str(args.scheduler) == "sgdr" else 0,
            "sgdr_t_mult": int(sgdr_t_mult) if str(args.scheduler) == "sgdr" else 0,
            "sgdr_eta_min": float(sgdr_eta_min) if str(args.scheduler) == "sgdr" else 0.0,
            "save_tag": str(tag),
            "save_path": str(target_path),
        }
        ema_entries_meta = [
            {"name": tr.name, "decay": float(tr.decay), "steps": int(tr.steps)}
            for tr in ema_trackers
        ]
        ckpt_out["ema"] = {"entries": ema_entries_meta}

        torch.save(ckpt_out, target_path)
        print(f"[OK] wrote: {target_path} ({tag}, upd={upd_to_save})")

        for tr in ema_trackers:
            ema_out = ckpt_model_dict(
                model_state=normalize_state_dict_keys(tr.model.state_dict()),
                upd=int(upd_to_save),
                env_steps=int(env_steps),
                model=model_spec,
                train=train_spec,
                wandb_run_id=wandb_run_id,
                run_dir=None,
            )
            ema_out["teacher"] = dict(ckpt_out["teacher"])
            ema_out["distill"] = dict(ckpt_out["distill"])
            ema_out["ema"] = {
                "name": str(tr.name),
                "decay": float(tr.decay),
                "steps": int(tr.steps),
                "entries": ema_entries_meta,
            }
            ema_out_path = _ema_out_path(target_path, float(tr.decay))
            torch.save(ema_out, ema_out_path)
            print(f"[OK] wrote: {ema_out_path} ({tag}, upd={upd_to_save})")

    saved_sgdr_cycles: set[int] = set()
    t0 = time.perf_counter()
    for upd in range(1, int(args.updates) + 1):
        seed_base = int(args.seed) + upd * 100000 + int(rank) * int(local_batch_size)
        seeds = torch.arange(int(local_batch_size), dtype=torch.int64) + seed_base
        env.reset_random(seeds)

        obs_s, mask, teacher_logits, teacher_values, opp_move_dist, opp_param, opp_valid = _collect_teacher_targets_pair(
            env,
            teacher,
            device,
            t_max=int(env.spec.t_max),
            feature_id_teacher=teacher_feature_id,
            feature_id_student=student_feature_id,
            sample=bool(teacher_rollout_sample),
            amp=bool(use_amp),
            teacher_tta_mode=int(args.teacher_tta_mode),
            teacher_tta_k=int(args.teacher_tta_k),
        )

        t_max, bsz = teacher_values.shape
        n = t_max * bsz
        obs_s = obs_s.reshape(n, c_student, 10, 10)
        mask = mask.reshape(n, 100)
        teacher_logits = teacher_logits.reshape(n, 100)
        teacher_values = teacher_values.reshape(n)
        opp_move_dist = opp_move_dist.reshape(n, int(env.spec.m_max), 100)
        opp_param = opp_param.reshape(n, int(env.spec.m_max), 5)
        opp_valid = opp_valid.reshape(n, int(env.spec.m_max))

        preload_ok = False
        obs_d = obs_s
        mask_d = mask
        tlog_d = teacher_logits
        tval_d = teacher_values
        omd_d = opp_move_dist
        op_d = opp_param
        ov_d = opp_valid
        if device.type == "cuda":
            try:
                obs_d = obs_s.to(device)
                mask_d = mask.to(device)
                tlog_d = teacher_logits.to(device)
                tval_d = teacher_values.to(device)
                omd_d = opp_move_dist.to(device)
                op_d = opp_param.to(device)
                ov_d = opp_valid.to(device)
                preload_ok = True
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()

        perm_device = device if preload_ok else torch.device("cpu")
        student.train()

        stats_cnt = 0
        sum_pi = 0.0
        sum_pi_margin = 0.0
        sum_v = 0.0
        sum_total = 0.0
        sum_opp_move = 0.0
        sum_opp_param = 0.0
        sum_teacher_entropy = 0.0
        sum_kl = 0.0
        sum_top1 = 0.0

        use_update_amp = device.type == "cuda" and bool(use_amp)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_update_amp)

        for _ in range(int(args.epochs)):
            perm = torch.randperm(n, device=perm_device)
            for start in range(0, n, int(local_minibatch)):
                mb = perm[start : start + int(local_minibatch)]
                x = obs_d[mb] if preload_ok else obs_s[mb].to(device)
                m = mask_d[mb] if preload_ok else mask[mb].to(device)
                tl = tlog_d[mb] if preload_ok else teacher_logits[mb].to(device)
                tv = tval_d[mb] if preload_ok else teacher_values[mb].to(device)
                omd = omd_d[mb] if preload_ok else opp_move_dist[mb].to(device)
                op = op_d[mb] if preload_ok else opp_param[mb].to(device)
                ov = ov_d[mb] if preload_ok else opp_valid[mb].to(device)

                slogits, svalue, s_opp_move_logits, s_opp_param_logits = _forward_student_with_tta(
                    student,
                    x,
                    m,
                    autocast=autocast,
                    tta_mode=student_tta_mode,
                    tta_k=student_tta_k,
                    tta_perm=student_tta_perm,
                    tta_inv=student_tta_inv,
                    aux_mode=student_tta_aux,
                )

                log_ps = torch.log_softmax(slogits / tau, dim=1)
                log_pt = torch.log_softmax(tl.float() / tau, dim=1)
                pt = log_pt.exp()

                cross_ent = -(pt * log_ps).sum(dim=1).mean()
                teacher_entropy = -(pt * log_pt).sum(dim=1).mean()
                kl = cross_ent - teacher_entropy
                teacher_top1 = torch.argmax(tl, dim=1)
                top1 = (teacher_top1 == torch.argmax(slogits, dim=1)).float().mean()

                loss_pi_margin = torch.zeros((), dtype=torch.float32, device=device)
                if policy_distill_mode == "top1":
                    loss_pi_ce = F.cross_entropy(slogits, teacher_top1)
                    if top1_margin_coef > 0.0 and top1_margin > 0.0:
                        true_logit = slogits.gather(1, teacher_top1.view(-1, 1)).squeeze(1)
                        target_mask = F.one_hot(teacher_top1, num_classes=int(slogits.shape[1])).to(torch.bool)
                        best_other = slogits.masked_fill(target_mask, -float("inf")).max(dim=1).values
                        loss_pi_margin = F.relu(top1_margin - (true_logit - best_other)).mean()
                    loss_pi = loss_pi_ce + float(top1_margin_coef) * loss_pi_margin
                else:
                    loss_pi = cross_ent * (tau * tau)
                loss_v_raw = F.mse_loss(svalue, tv.float())
                if value_distill_mode == "off":
                    loss_v = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    loss_v = loss_v_raw

                valid = ov != 0 if ov.dtype != torch.bool else ov

                loss_opp_move = torch.zeros((), dtype=torch.float32, device=device)
                if w_opp_move > 0.0 and s_opp_move_logits is not None:
                    logp_opp = torch.log_softmax(s_opp_move_logits.float(), dim=2)
                    ce = -(omd.float() * logp_opp).sum(dim=2)  # [N, 8]
                    denom = valid.float().sum().clamp(min=1.0)
                    loss_opp_move = (ce * valid.float()).sum() / denom

                loss_opp_param = torch.zeros((), dtype=torch.float32, device=device)
                if w_opp_param > 0.0 and s_opp_param_logits is not None:
                    pred_w = torch.softmax(s_opp_param_logits[..., :4].float(), dim=2)
                    pred_eps = torch.sigmoid(s_opp_param_logits[..., 4].float())
                    tgt_w = op[..., :4].float()
                    tgt_eps = op[..., 4].float()
                    loss_w = (pred_w - tgt_w).pow(2).mean(dim=2)  # [N, 8]
                    loss_eps = (pred_eps - tgt_eps).pow(2)  # [N, 8]
                    per = loss_w + loss_eps
                    denom = valid.float().sum().clamp(min=1.0)
                    loss_opp_param = (per * valid.float()).sum() / denom

                loss = w_pi * loss_pi + w_v * loss_v + w_opp_move * loss_opp_move + w_opp_param * loss_opp_param

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                sgdr_cycle_end_idx: int | None = None
                if scheduler is not None:
                    scheduler_ret = scheduler.step()
                    if isinstance(scheduler_ret, int):
                        sgdr_cycle_end_idx = int(scheduler_ret)
                if ema_trackers:
                    ema_update()
                if sgdr_cycle_end_idx is not None and sgdr_cycle_end_idx not in saved_sgdr_cycles:
                    saved_sgdr_cycles.add(int(sgdr_cycle_end_idx))
                    cycle_tag = f"sgdr_cycle_{int(sgdr_cycle_end_idx):04d}"
                    cycle_path = _sgdr_cycle_out_path(
                        out_path,
                        cycle_idx=int(sgdr_cycle_end_idx),
                        upd=int(upd),
                    )
                    save_ckpt_bundle(upd_to_save=int(upd), tag=cycle_tag, save_path=cycle_path)

                stats_cnt += 1
                sum_pi += float(loss_pi.item())
                sum_pi_margin += float(loss_pi_margin.item())
                sum_v += float(loss_v.item())
                sum_total += float(loss.item())
                sum_opp_move += float(loss_opp_move.item())
                sum_opp_param += float(loss_opp_param.item())
                sum_teacher_entropy += float(teacher_entropy.item())
                sum_kl += float(kl.item())
                sum_top1 += float(top1.item())

        stats_vec = torch.tensor(
            [
                float(sum_pi),
                float(sum_pi_margin),
                float(sum_v),
                float(sum_total),
                float(sum_opp_move),
                float(sum_opp_param),
                float(sum_teacher_entropy),
                float(sum_kl),
                float(sum_top1),
                float(stats_cnt),
            ],
            device=device,
            dtype=torch.float64,
        )
        _all_reduce_sum_tensor(stats_vec)
        global_stats_cnt = max(1.0, float(stats_vec[9].item()))
        mean_pi = float(stats_vec[0].item() / global_stats_cnt)
        mean_pi_margin = float(stats_vec[1].item() / global_stats_cnt)
        mean_v = float(stats_vec[2].item() / global_stats_cnt)
        mean_total = float(stats_vec[3].item() / global_stats_cnt)
        mean_opp_move = float(stats_vec[4].item() / global_stats_cnt)
        mean_opp_param = float(stats_vec[5].item() / global_stats_cnt)
        mean_teacher_entropy = float(stats_vec[6].item() / global_stats_cnt)
        mean_kl = float(stats_vec[7].item() / global_stats_cnt)
        mean_top1 = float(stats_vec[8].item() / global_stats_cnt)
        dt = float(time.perf_counter() - t0)
        dt_vec = torch.tensor([dt], device=device, dtype=torch.float64)
        _all_reduce_max_tensor(dt_vec)
        dt = float(dt_vec[0].item())
        lr_now = float(optimizer.param_groups[0]["lr"])
        if is_main_process:
            print(
                f"upd={upd:04d}/{int(args.updates):04d} "
                f"loss={mean_total:.6f} pi={mean_pi:.6f} pi_m={mean_pi_margin:.6f} v={mean_v:.6f} "
                f"opp_pi={mean_opp_move:.6f} opp_th={mean_opp_param:.6f} "
                f"h_t={mean_teacher_entropy:.6f} kl={mean_kl:.6f} top1={mean_top1:.4f} "
                f"lr={lr_now:.2e} dt={dt:.1f}s"
            )
        if run is not None:
            wandb.log(
                {
                    "upd": int(upd),
                    "distill/loss_total": float(mean_total),
                    "distill/loss_pi": float(mean_pi),
                    "distill/loss_pi_margin": float(mean_pi_margin),
                    "distill/loss_v": float(mean_v),
                    "distill/loss_opp_move": float(mean_opp_move),
                    "distill/loss_opp_param": float(mean_opp_param),
                    "distill/teacher_entropy": float(mean_teacher_entropy),
                    "distill/kl": float(mean_kl),
                    "distill/top1": float(mean_top1),
                    "hp/lr": float(lr_now),
                    "hp/weight_decay": float(args.weight_decay),
                    "hp/max_grad_norm": float(max_grad_norm),
                    "time/elapsed_s": float(dt),
                }
            )
        if save_every > 0 and (upd % save_every == 0) and (upd < int(args.updates)):
            save_ckpt_bundle(upd_to_save=int(upd), tag="periodic")

    student.eval()
    if is_main_process:
        save_ckpt_bundle(upd_to_save=int(args.updates), tag="final")
        if run is not None and bool(args.wandb_log_ckpt):
            art = wandb.Artifact(f"distill_ckpt_{run.id}", type="model")
            art.add_file(str(out_path))
            run.log_artifact(art)
    if is_distributed:
        dist.barrier()
    if run is not None:
        run.finish()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
