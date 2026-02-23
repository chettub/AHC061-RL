from __future__ import annotations

import argparse
import time

import torch

from ..env import BatchEnv
from ..models import PolicyValueNet, masked_logits


def _pick_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def bench_env_only(env: BatchEnv, steps: int) -> dict[str, float]:
    reward = torch.empty((env.batch_size,), dtype=torch.float32, device="cpu")
    done = torch.empty((env.batch_size,), dtype=torch.uint8, device="cpu")
    t0 = time.perf_counter()
    for _ in range(steps):
        a = env.pos0()
        env.step_into(a, reward, done)
    t1 = time.perf_counter()
    it_sec = t1 - t0
    env_steps = steps * env.batch_size
    return {
        "time/total_sec": it_sec,
        "time/env_sps": (env_steps / it_sec) if it_sec > 0 else 0.0,
    }


@torch.no_grad()
def bench_rollout(
    env: BatchEnv,
    model: PolicyValueNet,
    device: torch.device,
    steps: int,
    *,
    pin_memory: bool,
    sample: bool,
    amp: bool,
) -> dict[str, float]:
    bsz = env.batch_size
    c = env.feature_channels

    board_cpu = torch.empty(
        (bsz, c, 10, 10),
        dtype=torch.float32,
        device="cpu",
        pin_memory=pin_memory,
    )
    board_next_cpu = torch.empty(
        (bsz, c, 10, 10),
        dtype=torch.float32,
        device="cpu",
        pin_memory=pin_memory,
    )
    mask_cpu = torch.empty(
        (bsz, 100),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=pin_memory,
    )
    mask_next_cpu = torch.empty(
        (bsz, 100),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=pin_memory,
    )
    reward_cpu = torch.empty((bsz,), dtype=torch.float32, device="cpu")
    done_cpu = torch.empty((bsz,), dtype=torch.uint8, device="cpu")

    if device.type == "cuda":
        board_dev = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device=device)
        mask_dev = torch.empty((bsz, 100), dtype=torch.uint8, device=device)

    observe_sec = 0.0
    h2d_sec = 0.0
    forward_sec = 0.0
    sample_sec = 0.0
    step_sec = 0.0
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp))

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    env.observe_into(board_cpu, mask_cpu)
    t1 = time.perf_counter()
    observe_sec += t1 - t0

    for _ in range(steps):
        if device.type == "cuda":
            t0 = time.perf_counter()
            board_dev.copy_(board_cpu, non_blocking=pin_memory)
            mask_dev.copy_(mask_cpu, non_blocking=pin_memory)
            sync()
            t1 = time.perf_counter()
            h2d_sec += t1 - t0
            board = board_dev
            mask = mask_dev
        else:
            board = board_cpu
            mask = mask_cpu

        t0 = time.perf_counter()
        with autocast:
            logits, _ = model(board)
        logits = masked_logits(logits.float(), mask)
        sync()
        t1 = time.perf_counter()
        forward_sec += t1 - t0

        t0 = time.perf_counter()
        log_probs = torch.log_softmax(logits, dim=1)
        if sample:
            probs = log_probs.exp()
            actions = torch.multinomial(probs, 1).squeeze(1)
        else:
            actions = torch.argmax(log_probs, dim=1)
        actions_cpu = actions.to("cpu")
        sync()
        t1 = time.perf_counter()
        sample_sec += t1 - t0

        t0 = time.perf_counter()
        env.step_observe_into(actions_cpu, board_next_cpu, mask_next_cpu, reward_cpu, done_cpu)
        t1 = time.perf_counter()
        step_sec += t1 - t0
        board_cpu, board_next_cpu = board_next_cpu, board_cpu
        mask_cpu, mask_next_cpu = mask_next_cpu, mask_cpu

    total_sec = observe_sec + h2d_sec + forward_sec + sample_sec + step_sec
    env_steps = steps * env.batch_size
    return {
        "time/observe_sec": observe_sec,
        "time/h2d_sec": h2d_sec,
        "time/forward_sec": forward_sec,
        "time/sample_sec": sample_sec,
        "time/step_sec": step_sec,
        "time/total_sec": total_sec,
        "time/env_sps": (env_steps / total_sec) if total_sec > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rollout", choices=["rollout", "env"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    args = parser.parse_args()

    device = _pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    env = BatchEnv(batch_size=args.batch_size, pf_enabled=not args.no_pf, verbose_build=False)
    env.reset_random(torch.arange(args.batch_size, dtype=torch.int64))

    model = PolicyValueNet(in_channels=env.feature_channels, hidden_channels=args.hidden, blocks=args.blocks).to(device)
    if args.compile and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        model = torch.compile(model, mode=args.compile_mode)
    model.eval()

    # warmup (also triggers ext build)
    if args.mode == "env":
        bench_env_only(env, args.warmup)
    else:
        bench_rollout(env, model, device, args.warmup, pin_memory=args.pin, sample=not args.greedy, amp=args.amp)

    best = None
    best_rep = -1
    for rep in range(max(1, args.reps)):
        env.reset_random(torch.arange(args.batch_size, dtype=torch.int64) + 12345 + rep * 100000)
        if args.mode == "env":
            m = bench_env_only(env, args.steps)
        else:
            m = bench_rollout(env, model, device, args.steps, pin_memory=args.pin, sample=not args.greedy, amp=args.amp)
        print(f"rep={rep} total_sec={m['time/total_sec']:.6f} env_sps={m['time/env_sps']:.2f}")
        if best is None or m["time/total_sec"] < best["time/total_sec"]:
            best = m
            best_rep = rep

    assert best is not None
    print(f"best_rep={best_rep} mode={args.mode} device={device} batch={args.batch_size} steps={args.steps} pin={args.pin}")
    for k in sorted(best.keys()):
        print(f"{k} {best[k]:.6f}")


if __name__ == "__main__":
    main()
