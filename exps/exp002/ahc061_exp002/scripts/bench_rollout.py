from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..ckpt import model_spec_from_ckpt, normalize_state_dict_keys, torch_load_maybe_weights_only
from ..env import BatchEnv
from ..models import build_policy_value_model
from ..ppo.rollout import collect_rollout, create_rollout_workspace


@dataclass(frozen=True)
class BenchResult:
    sec_total: float
    episodes: int
    steps_per_episode: int
    batch_size: int

    @property
    def steps(self) -> int:
        return int(self.episodes) * int(self.steps_per_episode) * int(self.batch_size)

    @property
    def sps(self) -> float:
        if self.sec_total <= 0:
            return 0.0
        return float(self.steps) / float(self.sec_total)


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_use_channels_last(*, device: torch.device, memory_format_arg: str) -> bool:
    if device.type != "cuda":
        return False
    mode = str(memory_format_arg).strip().lower()
    if mode == "channels_last":
        return True
    if mode == "nchw":
        return False
    return True


@torch.inference_mode()
def _bench_env_step_observe(
    env: BatchEnv,
    *,
    episodes: int,
    t_max: int,
    with_aux: bool,
    fused_step_aux: bool,
) -> BenchResult:
    bsz = int(env.batch_size)
    c = int(env.feature_channels)
    seeds = torch.empty((bsz,), dtype=torch.int64, device="cpu")

    board = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((bsz, 100), dtype=torch.uint8, device="cpu")
    next_board = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    next_mask = torch.empty((bsz, 100), dtype=torch.uint8, device="cpu")
    reward = torch.empty((bsz,), dtype=torch.float32, device="cpu")
    done = torch.empty((bsz,), dtype=torch.uint8, device="cpu")

    opp_move_dist = torch.empty((bsz, int(env.spec.m_max), 100), dtype=torch.float32, device="cpu")
    opp_param = torch.empty((bsz, int(env.spec.m_max), 5), dtype=torch.float32, device="cpu")
    opp_valid = torch.empty((bsz, int(env.spec.m_max)), dtype=torch.uint8, device="cpu")

    t0 = time.perf_counter()
    for ep in range(int(episodes)):
        seeds.copy_(torch.arange(bsz, dtype=torch.int64) + ep * 1000003)
        env.reset_random(seeds)

        env.observe_into(board, mask)
        if with_aux:
            env.aux_targets_into(opp_move_dist, opp_param, opp_valid)

        for _ in range(int(t_max)):
            # 追加のコストを入れないため、合法手のうち先頭(=argmax)を常に選ぶ
            actions = torch.argmax(mask, dim=1).to(dtype=torch.int64, device="cpu")
            if with_aux:
                if fused_step_aux:
                    env.step_observe_aux_into(actions, next_board, next_mask, reward, done, opp_move_dist, opp_param, opp_valid)
                else:
                    env.step_observe_into(actions, next_board, next_mask, reward, done)
                    env.aux_targets_into(opp_move_dist, opp_param, opp_valid)
            else:
                env.step_observe_into(actions, next_board, next_mask, reward, done)
            board, next_board = next_board, board
            mask, next_mask = next_mask, mask
    t1 = time.perf_counter()

    return BenchResult(sec_total=float(t1 - t0), episodes=int(episodes), steps_per_episode=int(t_max), batch_size=bsz)


@torch.inference_mode()
def _bench_collect_rollout(
    env: BatchEnv,
    model: torch.nn.Module,
    device: torch.device,
    *,
    episodes: int,
    t_max: int,
    sample: bool,
    amp: bool,
    channels_last: bool,
    collect_aux: bool,
    fused_step_aux: bool,
) -> BenchResult:
    workspace = create_rollout_workspace(
        env,
        t_max=int(t_max),
        device=device,
        channels_last=bool(channels_last),
        pin_memory=(device.type == "cuda"),
        collect_aux=bool(collect_aux),
    )
    _maybe_sync(device)
    t0 = time.perf_counter()
    for ep in range(int(episodes)):
        seeds = torch.arange(int(env.batch_size), dtype=torch.int64, device="cpu") + ep * 1000003
        env.reset_random(seeds)
        _ = collect_rollout(
            env,
            model,
            device,
            int(t_max),
            sample=bool(sample),
            amp=bool(amp),
            channels_last=bool(channels_last),
            collect_aux=bool(collect_aux),
            fused_step_aux=bool(fused_step_aux),
            workspace=workspace,
        )
    _maybe_sync(device)
    t1 = time.perf_counter()
    return BenchResult(
        sec_total=float(t1 - t0),
        episodes=int(episodes),
        steps_per_episode=int(t_max),
        batch_size=int(env.batch_size),
    )


def _format_result(title: str, r: BenchResult) -> str:
    step_sec = (r.sec_total / (r.episodes * r.steps_per_episode)) if (r.episodes * r.steps_per_episode) > 0 else 0.0
    return (
        f"[{title}] sec={r.sec_total:.3f}  steps={r.steps}  sps={r.sps:.1f}  "
        f"(per-step-batch sec={step_sec:.6f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["env", "env_aux", "rollout", "both"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-id", type=str, default="submit_v1")
    parser.add_argument("--arch", type=str, default="dwres_v1")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--memory-format", type=str, choices=("auto", "nchw", "channels_last"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument(
        "--no-rollout-aux-targets",
        action="store_true",
        help="Skip env.aux_targets_into() inside collect_rollout (effective only when measuring rollout speed).",
    )
    g_fused_aux = parser.add_mutually_exclusive_group()
    g_fused_aux.add_argument(
        "--fused-step-aux",
        dest="fused_step_aux",
        action="store_true",
        default=False,
        help="Use fused step+observe+aux path (default: off).",
    )
    g_fused_aux.add_argument(
        "--no-fused-step-aux",
        dest="fused_step_aux",
        action="store_false",
        help="Disable fused step+observe+aux path.",
    )
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    args = parser.parse_args()

    if args.torch_num_threads is not None:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(int(args.torch_num_interop_threads))

    device = _pick_device(str(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_channels_last = _resolve_use_channels_last(device=device, memory_format_arg=str(args.memory_format))

    ckpt: dict[str, Any] | None = None
    arch_kwargs: dict[str, Any] = {}
    if args.ckpt is not None:
        ckpt_path = Path(str(args.ckpt)).expanduser().resolve()
        ckpt = torch_load_maybe_weights_only(ckpt_path)
        ms = model_spec_from_ckpt(ckpt)
        feature_id = str(ms.feature_id)
        arch = str(ms.arch_name)
        hidden = int(ms.hidden)
        blocks = int(ms.blocks)
        in_channels = int(ms.in_channels)
        arch_kwargs = dict(ms.arch_kwargs)
    else:
        feature_id = str(args.feature_id)
        arch = str(args.arch)
        hidden = int(args.hidden)
        blocks = int(args.blocks)
        in_channels = None

    env = BatchEnv(
        batch_size=int(args.batch_size),
        feature_id=feature_id,
        pf_enabled=not args.no_pf,
        verbose_build=False,
    )
    if in_channels is None:
        in_channels = int(env.feature_channels)

    model = build_policy_value_model(
        arch,
        in_channels=int(in_channels),
        hidden_channels=int(hidden),
        blocks=int(blocks),
        feature_id=str(feature_id),
        arch_kwargs=dict(arch_kwargs),
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if ckpt is not None:
        missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt["model"]), strict=False)
        if unexpected:
            raise RuntimeError(f"[CKPT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
        allowed_missing_prefixes = ("opp_move_head.", "opp_param_head.")
        bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
        if bad_missing:
            raise RuntimeError(f"[CKPT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")

    compile_ok = False
    if bool(args.compile) and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        try:
            model = torch.compile(model, mode=str(args.compile_mode))
            compile_ok = True
        except Exception as e:
            print(f"[WARN] torch.compile failed ({type(e).__name__}: {e}); falling back to eager.")

    sample = not bool(args.no_sample)
    use_amp = (device.type == "cuda") and bool(args.amp)
    collect_aux_rollout = not bool(args.no_rollout_aux_targets)
    fused_step_aux = bool(args.fused_step_aux)
    pf_particles = os.environ.get("AHC061_PF_PARTICLES", "16")

    print(
        "[CONFIG]"
        f" mode={args.mode}"
        f" device={device}"
        f" compile={bool(args.compile)}(ok={compile_ok})"
        f" amp={use_amp}"
        f" memory_format={'channels_last' if use_channels_last else 'nchw'}"
        f" sample={sample}"
        f" collect_aux_rollout={collect_aux_rollout}"
        f" fused_step_aux={fused_step_aux}"
        f" pf_enabled={not args.no_pf}"
        f" batch={int(env.batch_size)}"
        f" t_max={int(args.t_max)}"
        f" pf_particles={pf_particles}"
        f" feature_id={feature_id}"
        f" arch={arch}"
        f" hidden={hidden}"
        f" blocks={blocks}"
        f" torch_threads={torch.get_num_threads()}"
        f" torch_interop={torch.get_num_interop_threads()}"
    )

    # warmup
    w = int(args.warmup)
    if w > 0:
        if args.mode in ("env", "env_aux", "both"):
            _ = _bench_env_step_observe(
                env,
                episodes=w,
                t_max=int(args.t_max),
                with_aux=(args.mode == "env_aux"),
                fused_step_aux=bool(fused_step_aux),
            )
        if args.mode in ("rollout", "both"):
            _ = _bench_collect_rollout(
                env,
                model,
                device,
                episodes=w,
                t_max=int(args.t_max),
                sample=sample,
                amp=use_amp,
                channels_last=use_channels_last,
                collect_aux=collect_aux_rollout,
                fused_step_aux=bool(fused_step_aux),
            )

    if args.mode == "env":
        r = _bench_env_step_observe(
            env,
            episodes=int(args.episodes),
            t_max=int(args.t_max),
            with_aux=False,
            fused_step_aux=bool(fused_step_aux),
        )
        print(_format_result("env(step_observe_into)", r))
        return
    if args.mode == "env_aux":
        r = _bench_env_step_observe(
            env,
            episodes=int(args.episodes),
            t_max=int(args.t_max),
            with_aux=True,
            fused_step_aux=bool(fused_step_aux),
        )
        print(_format_result("env(step_observe_into+aux_targets)", r))
        return
    if args.mode == "rollout":
        r = _bench_collect_rollout(
            env,
            model,
            device,
            episodes=int(args.episodes),
            t_max=int(args.t_max),
            sample=sample,
            amp=use_amp,
            channels_last=use_channels_last,
            collect_aux=collect_aux_rollout,
            fused_step_aux=bool(fused_step_aux),
        )
        print(_format_result("collect_rollout", r))
        return

    r0 = _bench_env_step_observe(
        env,
        episodes=int(args.episodes),
        t_max=int(args.t_max),
        with_aux=False,
        fused_step_aux=bool(fused_step_aux),
    )
    r1 = _bench_env_step_observe(
        env,
        episodes=int(args.episodes),
        t_max=int(args.t_max),
        with_aux=True,
        fused_step_aux=bool(fused_step_aux),
    )
    r2 = _bench_collect_rollout(
        env,
        model,
        device,
        episodes=int(args.episodes),
        t_max=int(args.t_max),
        sample=sample,
        amp=use_amp,
        channels_last=use_channels_last,
        collect_aux=collect_aux_rollout,
        fused_step_aux=bool(fused_step_aux),
    )
    print(_format_result("env(step_observe_into)", r0))
    print(_format_result("env(step_observe_into+aux_targets)", r1))
    print(_format_result("collect_rollout", r2))


if __name__ == "__main__":
    main()
