from __future__ import annotations

import argparse
import time

import torch
from torch.distributions import Categorical

from ..env import BatchEnv
from ..models import PolicyValueNet, masked_logits
from ..ppo.gae import compute_gae
from ..ppo.rollout import Rollout, collect_rollout
from ..ppo.update import ppo_update


def _pick_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _warmup_rollout(
    env: BatchEnv, model: PolicyValueNet, device: torch.device, seed: int
) -> None:
    seeds = torch.arange(env.batch_size, dtype=torch.int64) + seed
    env.reset_random(seeds)
    collect_rollout(env, model, device, t_max=env.spec.t_max, sample=True)
    _sync(device)


@torch.no_grad()
def _collect_rollout_baseline(
    env: BatchEnv,
    model: PolicyValueNet,
    device: torch.device,
    *,
    sample: bool,
) -> Rollout:
    t_max = env.spec.t_max
    bsz = env.batch_size
    c = env.feature_channels

    obs = torch.empty((t_max, bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((t_max, bsz, 100), dtype=torch.uint8, device="cpu")
    actions = torch.empty((t_max, bsz), dtype=torch.int64, device="cpu")
    logp = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    values = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    rewards = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    dones = torch.empty((t_max, bsz), dtype=torch.uint8, device="cpu")

    model.eval()
    for t in range(t_max):
        env.observe_into(obs[t], mask[t])

        board = obs[t].to(device)
        m = mask[t].to(device)
        logits, v = model(board)
        logits = masked_logits(logits, m)
        dist = Categorical(logits=logits)
        if sample:
            a = dist.sample()
        else:
            a = torch.argmax(dist.probs, dim=1)

        actions[t] = a.to("cpu")
        logp[t] = dist.log_prob(a).to("cpu")
        values[t] = v.to("cpu")

        rew, done = env.step(actions[t])
        rewards[t] = rew
        dones[t] = done

    next_obs, next_mask = env.observe()
    board = next_obs.to(device)
    m = next_mask.to(device)
    _, last_v = model(board)
    last_value = last_v.to("cpu")

    return Rollout(
        obs=obs,
        mask=mask,
        actions=actions,
        logp=logp,
        values=values,
        rewards=rewards,
        dones=dones,
        last_value=last_value,
    )


def _ppo_update_baseline(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,  # [N, C, 10, 10]
    mask: torch.Tensor,  # [N, 100]
    actions: torch.Tensor,  # [N]
    old_logp: torch.Tensor,  # [N]
    old_values: torch.Tensor,  # [N]
    advantages: torch.Tensor,  # [N]
    returns: torch.Tensor,  # [N]
    *,
    clip_coef: float = 0.2,
    vf_clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
) -> None:
    logits, values = model(obs)
    logits = masked_logits(logits, mask)
    dist = Categorical(logits=logits)

    new_logp = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()

    with torch.no_grad():
        _ = (old_logp - new_logp).mean()
        _ = ((ratio - 1.0).abs() > clip_coef).float().mean()

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    if vf_clip_coef > 0.0:
        v_clipped = old_values + torch.clamp(
            values - old_values, -vf_clip_coef, vf_clip_coef
        )
        v_loss1 = (values - returns).pow(2)
        v_loss2 = (v_clipped - returns).pow(2)
        value_loss = torch.max(v_loss1, v_loss2).mean()
    else:
        value_loss = (values - returns).pow(2).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()


def bench_once_baseline(args, device: torch.device) -> dict[str, float]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    env = BatchEnv(
        batch_size=args.batch_size, pf_enabled=not args.no_pf, verbose_build=False
    )
    model = PolicyValueNet(
        in_channels=env.feature_channels,
        hidden_channels=args.hidden,
        blocks=args.blocks,
    ).to(device)
    if args.compile and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        model = torch.compile(model, mode=args.compile_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.warmup > 0:
        seeds = torch.arange(env.batch_size, dtype=torch.int64) + (args.seed + 12345)
        env.reset_random(seeds)
        _collect_rollout_baseline(env, model, device, sample=True)
        _sync(device)

    seeds = torch.arange(args.batch_size, dtype=torch.int64) + args.seed + 100000
    env.reset_random(seeds)

    t0 = time.perf_counter()
    rollout = _collect_rollout_baseline(env, model, device, sample=True)
    _sync(device)
    t1 = time.perf_counter()

    adv, ret = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        last_value=rollout.last_value,
        gamma=1.0,
        gae_lambda=0.95,
    )
    t2 = time.perf_counter()

    t_max, bsz = rollout.actions.shape
    n = t_max * bsz
    obs = rollout.obs.reshape(n, env.feature_channels, 10, 10)
    mask = rollout.mask.reshape(n, 100)
    actions = rollout.actions.reshape(n)
    old_logp = rollout.logp.reshape(n)
    old_values = rollout.values.reshape(n)
    advantages = adv.reshape(n)
    returns = ret.reshape(n)

    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    t3 = time.perf_counter()

    model.train()
    idx = torch.arange(n, dtype=torch.int64)
    for _ in range(args.epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, args.minibatch):
            mb = perm[start : start + args.minibatch]
            _ppo_update_baseline(
                model,
                optimizer,
                obs=obs[mb].to(device),
                mask=mask[mb].to(device),
                actions=actions[mb].to(device),
                old_logp=old_logp[mb].to(device),
                old_values=old_values[mb].to(device),
                advantages=advantages[mb].to(device),
                returns=returns[mb].to(device),
            )
    _sync(device)
    t4 = time.perf_counter()

    rollout_sec = t1 - t0
    gae_sec = t2 - t1
    flatten_sec = t3 - t2
    update_sec = t4 - t3
    total_sec = t4 - t0
    env_steps = args.batch_size * env.spec.t_max

    return {
        "time/rollout_sec": rollout_sec,
        "time/gae_sec": gae_sec,
        "time/flatten_sec": flatten_sec,
        "time/update_sec": update_sec,
        "time/total_sec": total_sec,
        "time/sps": (env_steps / total_sec) if total_sec > 0 else 0.0,
    }


def bench_once(args, device: torch.device) -> dict[str, float]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    env = BatchEnv(
        batch_size=args.batch_size, pf_enabled=not args.no_pf, verbose_build=False
    )
    model = PolicyValueNet(
        in_channels=env.feature_channels,
        hidden_channels=args.hidden,
        blocks=args.blocks,
    ).to(device)
    if args.compile and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        model = torch.compile(model, mode=args.compile_mode)
    opt_kwargs = {"lr": args.lr}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.warmup > 0:
        _warmup_rollout(env, model, device, args.seed + 12345)

    seeds = torch.arange(args.batch_size, dtype=torch.int64) + args.seed + 100000
    env.reset_random(seeds)

    t0 = time.perf_counter()
    rollout = collect_rollout(env, model, device, t_max=env.spec.t_max, sample=True)
    _sync(device)
    t1 = time.perf_counter()

    adv, ret = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        last_value=rollout.last_value,
        gamma=1.0,
        gae_lambda=0.95,
    )
    t2 = time.perf_counter()

    # flatten
    t_max, bsz = rollout.actions.shape
    n = t_max * bsz
    obs = rollout.obs.reshape(n, env.feature_channels, 10, 10)
    mask = rollout.mask.reshape(n, 100)
    actions = rollout.actions.reshape(n)
    old_logp = rollout.logp.reshape(n)
    old_values = rollout.values.reshape(n)
    advantages = adv.reshape(n)
    returns = ret.reshape(n)

    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    t3 = time.perf_counter()

    model.train()
    preload_ok = False
    obs_d = obs
    mask_d = (mask != 0) if mask.dtype != torch.bool else mask
    actions_d = actions
    old_logp_d = old_logp
    old_values_d = old_values
    advantages_d = advantages
    returns_d = returns
    if device.type == "cuda":
        try:
            obs_d = obs.to(device)
            mask_d = (
                (mask.to(device) != 0) if mask.dtype != torch.bool else mask.to(device)
            )
            actions_d = actions.to(device)
            old_logp_d = old_logp.to(device)
            old_values_d = old_values.to(device)
            advantages_d = advantages.to(device)
            returns_d = returns.to(device)
            preload_ok = True
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()

    perm_device = device if preload_ok else torch.device("cpu")
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=perm_device)
        for start in range(0, n, args.minibatch):
            mb = perm[start : start + args.minibatch]
            ppo_update(
                model,
                optimizer,
                obs=obs_d[mb] if preload_ok else obs[mb].to(device),
                mask=mask_d[mb] if preload_ok else mask[mb].to(device),
                actions=actions_d[mb] if preload_ok else actions[mb].to(device),
                old_logp=old_logp_d[mb] if preload_ok else old_logp[mb].to(device),
                old_values=old_values_d[mb]
                if preload_ok
                else old_values[mb].to(device),
                advantages=advantages_d[mb]
                if preload_ok
                else advantages[mb].to(device),
                returns=returns_d[mb] if preload_ok else returns[mb].to(device),
            )
    _sync(device)
    t4 = time.perf_counter()

    rollout_sec = t1 - t0
    gae_sec = t2 - t1
    flatten_sec = t3 - t2
    update_sec = t4 - t3
    total_sec = t4 - t0
    env_steps = args.batch_size * env.spec.t_max

    return {
        "time/rollout_sec": rollout_sec,
        "time/gae_sec": gae_sec,
        "time/flatten_sec": flatten_sec,
        "time/update_sec": update_sec,
        "time/total_sec": total_sec,
        "time/sps": (env_steps / total_sec) if total_sec > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl", type=str, default="both", choices=["both", "baseline", "optimized"]
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    args = parser.parse_args()

    device = _pick_device(args.device)

    def run_reps(fn) -> dict[str, float]:
        best = None
        for r in range(args.reps):
            m = fn(args, device)
            print(
                f"rep={r} total_sec={m['time/total_sec']:.6f} sps={m['time/sps']:.2f}"
            )
            if best is None or m["time/total_sec"] < best["time/total_sec"]:
                best = m
        assert best is not None
        return best

    baseline_best = None
    opt_best = None
    if args.impl in ("both", "baseline"):
        print("== baseline ==")
        baseline_best = run_reps(bench_once_baseline)
        for k in sorted(baseline_best.keys()):
            print(f"baseline/{k} {baseline_best[k]:.6f}")

    if args.impl in ("both", "optimized"):
        print("== optimized ==")
        opt_best = run_reps(bench_once)
        for k in sorted(opt_best.keys()):
            print(f"opt/{k} {opt_best[k]:.6f}")

    if baseline_best is not None and opt_best is not None:
        speedup = opt_best["time/sps"] / max(1e-9, baseline_best["time/sps"])
        print(f"speedup {speedup:.3f}x")


if __name__ == "__main__":
    main()
