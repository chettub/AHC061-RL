from __future__ import annotations

import argparse
import math
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..env import BatchEnv
from ..models import build_policy_value_model, masked_logits, normalize_state_dict_keys


def _torch_load(path: Path, *, weights_only: bool) -> dict:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _torch_load_maybe_weights_only(path: Path) -> dict:
    try:
        return _torch_load(path, weights_only=True)
    except pickle.UnpicklingError:
        return _torch_load(path, weights_only=False)


def _pick_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


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


@torch.inference_mode()
def _collect_teacher_targets(
    env: BatchEnv,
    teacher: torch.nn.Module,
    device: torch.device,
    t_max: int,
    *,
    sample: bool,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = env.batch_size
    c = env.feature_channels

    obs = torch.empty((t_max, bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((t_max, bsz, 100), dtype=torch.uint8, device="cpu")
    teacher_logits = torch.empty((t_max, bsz, 100), dtype=torch.float32, device="cpu")
    teacher_values = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")

    rewards = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    dones = torch.empty((t_max, bsz), dtype=torch.uint8, device="cpu")
    next_obs = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    next_mask = torch.empty((bsz, 100), dtype=torch.uint8, device="cpu")

    use_cuda = device.type == "cuda"
    use_amp = use_cuda and bool(amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    if use_cuda:
        board_dev = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device=device)
        mask_dev = torch.empty((bsz, 100), dtype=torch.uint8, device=device)

    teacher.eval()

    env.observe_into(obs[0], mask[0])
    for t in range(t_max):
        if use_cuda:
            board_dev.copy_(obs[t])
            mask_dev.copy_(mask[t])
            board = board_dev
            m = mask_dev
        else:
            board = obs[t]
            m = mask[t]

        with autocast:
            logits, v = teacher(board)
        logits = masked_logits(logits.float(), m)
        v = v.float()

        teacher_logits[t].copy_(logits.to("cpu"))
        teacher_values[t].copy_(v.to("cpu"))

        if sample:
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(1).to("cpu")
        else:
            actions = torch.argmax(logits, dim=1).to("cpu")

        if t + 1 < t_max:
            env.step_observe_into(
                actions, obs[t + 1], mask[t + 1], rewards[t], dones[t]
            )
        else:
            env.step_observe_into(actions, next_obs, next_mask, rewards[t], dones[t])

    return obs, mask, teacher_logits, teacher_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--student-arch", type=str, default="dwres_v1")
    parser.add_argument("--student-hidden", type=int, default=64)
    parser.add_argument("--student-blocks", type=int, default=16)
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
    parser.add_argument("--w-policy", type=float, default=1.0)
    parser.add_argument("--w-value", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--out-ckpt",
        type=str,
        default="exps/exp001/artifacts/checkpoints_distill/ckpt_dw64b16_from_0850.pt",
    )
    args = parser.parse_args()

    if args.updates <= 0:
        raise RuntimeError("--updates must be >= 1")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be >= 1")
    if args.epochs <= 0:
        raise RuntimeError("--epochs must be >= 1")
    if args.minibatch <= 0:
        raise RuntimeError("--minibatch must be >= 1")
    if args.tau <= 0:
        raise RuntimeError("--tau must be > 0")
    if not math.isfinite(float(args.weight_decay)) or float(args.weight_decay) < 0.0:
        raise RuntimeError("--weight-decay must be finite and >= 0")
    if not math.isfinite(float(args.max_grad_norm)) or float(args.max_grad_norm) < 0.0:
        raise RuntimeError("--max-grad-norm must be finite and >= 0")

    device = _pick_device(args.device)
    use_amp = device.type == "cuda" and (not args.no_amp)

    _set_seeds(args.seed, device)

    repo_root = Path(__file__).resolve().parents[4]
    out_path = Path(args.out_ckpt)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_ckpt_path = Path(args.teacher_ckpt)
    if not teacher_ckpt_path.is_absolute():
        teacher_ckpt_path = (repo_root / teacher_ckpt_path).resolve()
    teacher_ckpt = _torch_load_maybe_weights_only(teacher_ckpt_path)

    for k in ("model", "in_channels", "hidden", "blocks", "upd"):
        if k not in teacher_ckpt:
            raise RuntimeError(
                f"[TEACHER] invalid checkpoint (missing key: {k!r}): {teacher_ckpt_path}"
            )

    env = BatchEnv(
        batch_size=args.batch_size, pf_enabled=not args.no_pf, verbose_build=False
    )
    if int(teacher_ckpt["in_channels"]) != int(env.feature_channels):
        raise RuntimeError(
            f"[TEACHER] in_channels mismatch: ckpt={int(teacher_ckpt['in_channels'])} env={int(env.feature_channels)}"
        )

    teacher_arch = str(teacher_ckpt.get("arch_name", "resnet_v1"))
    teacher_hidden = int(teacher_ckpt["hidden"])
    teacher_blocks = int(teacher_ckpt["blocks"])
    teacher = build_policy_value_model(
        teacher_arch,
        in_channels=int(env.feature_channels),
        hidden_channels=teacher_hidden,
        blocks=teacher_blocks,
    ).to(device)
    teacher.load_state_dict(normalize_state_dict_keys(teacher_ckpt["model"]))
    teacher.eval()

    student = build_policy_value_model(
        args.student_arch,
        in_channels=int(env.feature_channels),
        hidden_channels=int(args.student_hidden),
        blocks=int(args.student_blocks),
    ).to(device)

    opt_kwargs = {"lr": float(args.lr), "weight_decay": float(args.weight_decay)}
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

    tau = float(args.tau)
    w_pi = float(args.w_policy)
    w_v = float(args.w_value)
    max_grad_norm = float(args.max_grad_norm)

    t0 = time.perf_counter()
    for upd in range(1, int(args.updates) + 1):
        seeds = (
            torch.arange(env.batch_size, dtype=torch.int64)
            + int(args.seed)
            + upd * 100000
        )
        env.reset_random(seeds)

        obs, mask, teacher_logits, teacher_values = _collect_teacher_targets(
            env,
            teacher,
            device,
            t_max=int(env.spec.t_max),
            sample=True,
            amp=use_amp,
        )

        t_max, bsz = teacher_values.shape
        n = t_max * bsz
        obs = obs.reshape(n, env.feature_channels, 10, 10)
        mask = mask.reshape(n, 100)
        teacher_logits = teacher_logits.reshape(n, 100)
        teacher_values = teacher_values.reshape(n)

        preload_ok = False
        obs_d = obs
        mask_d = mask
        tlog_d = teacher_logits
        tval_d = teacher_values
        if device.type == "cuda":
            try:
                obs_d = obs.to(device)
                mask_d = mask.to(device)
                tlog_d = teacher_logits.to(device)
                tval_d = teacher_values.to(device)
                preload_ok = True
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()

        perm_device = device if preload_ok else torch.device("cpu")
        student.train()

        stats_cnt = 0
        sum_pi = 0.0
        sum_v = 0.0
        sum_total = 0.0
        sum_teacher_entropy = 0.0
        sum_kl = 0.0
        sum_top1 = 0.0

        use_update_amp = device.type == "cuda" and bool(use_amp)
        autocast = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_update_amp
        )

        for _ in range(int(args.epochs)):
            perm = torch.randperm(n, device=perm_device)
            for start in range(0, n, int(args.minibatch)):
                mb = perm[start : start + int(args.minibatch)]
                x = obs_d[mb] if preload_ok else obs[mb].to(device)
                m = mask_d[mb] if preload_ok else mask[mb].to(device)
                tl = tlog_d[mb] if preload_ok else teacher_logits[mb].to(device)
                tv = tval_d[mb] if preload_ok else teacher_values[mb].to(device)

                with autocast:
                    slogits, svalue = student(x)
                slogits = masked_logits(slogits.float(), m)
                svalue = svalue.float()

                log_ps = torch.log_softmax(slogits / tau, dim=1)

                log_pt = torch.log_softmax(tl.float() / tau, dim=1)
                pt = log_pt.exp()
                cross_ent = -(pt * log_ps).sum(dim=1).mean()
                teacher_entropy = -(pt * log_pt).sum(dim=1).mean()
                kl = cross_ent - teacher_entropy
                top1 = (
                    (torch.argmax(tl, dim=1) == torch.argmax(slogits, dim=1))
                    .float()
                    .mean()
                )

                loss_pi = cross_ent * (tau * tau)
                loss_v = F.mse_loss(svalue, tv.float())
                loss = w_pi * loss_pi + w_v * loss_v

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()

                stats_cnt += 1
                sum_pi += float(loss_pi.item())
                sum_v += float(loss_v.item())
                sum_total += float(loss.item())
                sum_teacher_entropy += float(teacher_entropy.item())
                sum_kl += float(kl.item())
                sum_top1 += float(top1.item())

        mean_pi = sum_pi / max(1, stats_cnt)
        mean_v = sum_v / max(1, stats_cnt)
        mean_total = sum_total / max(1, stats_cnt)
        mean_teacher_entropy = sum_teacher_entropy / max(1, stats_cnt)
        mean_kl = sum_kl / max(1, stats_cnt)
        mean_top1 = sum_top1 / max(1, stats_cnt)
        dt = time.perf_counter() - t0
        print(
            f"upd={upd:04d}/{int(args.updates):04d} "
            f"loss={mean_total:.6f} pi={mean_pi:.6f} v={mean_v:.6f} "
            f"h_t={mean_teacher_entropy:.6f} kl={mean_kl:.6f} top1={mean_top1:.4f} "
            f"dt={dt:.1f}s"
        )

    student.eval()
    ckpt_out = {
        "upd": int(args.updates),
        "in_channels": int(env.feature_channels),
        "hidden": int(args.student_hidden),
        "blocks": int(args.student_blocks),
        "arch_name": str(args.student_arch),
        "model": normalize_state_dict_keys(student.state_dict()),
        "teacher_ckpt": str(teacher_ckpt_path),
        "teacher_upd": int(teacher_ckpt.get("upd", -1)),
        "teacher_arch": str(teacher_arch),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "torch_version": str(torch.__version__),
    }
    torch.save(ckpt_out, out_path)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
