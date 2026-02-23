from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

import torch

from ..env import BatchEnv
from ..env import tools_input_paths
from ..models import build_policy_value_model, masked_logits, normalize_state_dict_keys
from ..ppo.gae import compute_gae
from ..ppo.rollout import collect_rollout
from ..ppo.update import ppo_update

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


def _find_latest_ckpt(ckpt_dir: Path) -> Path:
    best_upd = -1
    best_path: Path | None = None
    for p in ckpt_dir.glob("ckpt_*.pt"):
        stem = p.stem  # ckpt_0001
        try:
            upd = int(stem.split("_", 1)[1])
        except Exception:
            continue
        if upd > best_upd:
            best_upd = upd
            best_path = p
    if best_path is None:
        raise FileNotFoundError(f"no checkpoint found in: {ckpt_dir}")
    return best_path


def _get_rng_state(device: torch.device) -> dict:
    state: dict = {
        "torch_cpu": torch.get_rng_state(),
        "py_random": random.getstate(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def _set_rng_state(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["py_random"])
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
        and ("torch_cuda_all" in state)
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    if "numpy" in state:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(state["numpy"])
        except Exception:
            pass


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for st in optimizer.state.values():
        if not isinstance(st, dict):
            continue
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps the original module in a proxy that has _orig_mod.
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    return model


def _require_equal(name: str, ckpt_v, cur_v) -> None:
    if ckpt_v != cur_v:
        raise RuntimeError(
            f"[RESUME][STRICT] {name} mismatch: ckpt={ckpt_v!r} current={cur_v!r}"
        )


def _torch_load(path: Path, *, weights_only: bool) -> dict:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        # torch<2.6: no weights_only kwarg, default behavior is equivalent to weights_only=False.
        return torch.load(path, **kwargs)


def _torch_load_maybe_weights_only(path: Path) -> dict:
    # Prefer safe weights-only load, but allow falling back for trusted local checkpoints
    # that contain non-tensor states (e.g., numpy RNG).
    try:
        return _torch_load(path, weights_only=True)
    except pickle.UnpicklingError:
        return _torch_load(path, weights_only=False)


@torch.inference_mode()
def _eval_tools(
    env: BatchEnv,
    model: torch.nn.Module,
    device: torch.device,
    paths: list[str],
    *,
    sample: bool,
) -> float:
    was_training = model.training
    model.eval()
    board = torch.empty(
        (env.batch_size, env.feature_channels, 10, 10),
        dtype=torch.float32,
        device="cpu",
    )
    mask = torch.empty((env.batch_size, 100), dtype=torch.uint8, device="cpu")
    scores: list[float] = []

    for i in range(0, len(paths), env.batch_size):
        chunk = paths[i : i + env.batch_size]
        if len(chunk) < env.batch_size:
            chunk = chunk + [chunk[-1]] * (env.batch_size - len(chunk))
            valid = len(paths) - i
        else:
            valid = env.batch_size

        env.reset_from_tools(chunk)
        for _ in range(env.spec.t_max):
            env.observe_into(board, mask)
            logits, _ = model(board.to(device))
            logits = masked_logits(logits.float(), mask.to(device))
            if sample:
                probs = torch.softmax(logits, dim=1)
                actions = torch.multinomial(probs, 1).squeeze(1).to("cpu")
            else:
                actions = torch.argmax(logits, dim=1).to("cpu")
            env.step(actions)
        sc = env.official_score().float()[:valid]
        scores.extend(sc.tolist())

    model.train(was_training)
    return float(sum(scores) / max(1, len(scores)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--arch", type=str, default="resnet_v1")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="ahc061-exp001")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=16)
    args = parser.parse_args()

    def pick_device() -> torch.device:
        if args.device != "auto":
            return torch.device(args.device)
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            arch = f"sm_{cap[0]}{cap[1]}"
            if arch in torch.cuda.get_arch_list():
                return torch.device("cuda")
        return torch.device("cpu")

    device = pick_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_amp = device.type == "cuda" and (not args.no_amp)
    rollout_amp = False

    exp_dir = Path(__file__).resolve().parents[2]
    ckpt_dir = exp_dir / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_full_dir = exp_dir / "artifacts" / "checkpoints_full"
    ckpt_full_dir.mkdir(parents=True, exist_ok=True)

    gamma = 1.0
    gae_lambda = 0.95

    resume_path: Path | None = None
    if (args.resume_ckpt is not None) + bool(args.resume_latest) + (
        args.init_ckpt is not None
    ) > 1:
        raise RuntimeError(
            "Specify only one of --resume-ckpt, --resume-latest, --init-ckpt"
        )
    if args.resume_latest:
        resume_path = _find_latest_ckpt(ckpt_full_dir)
    elif args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
    init_path = Path(args.init_ckpt) if args.init_ckpt else None

    resume_ckpt: dict | None = None
    if resume_path is not None:
        # Full resume requires non-tensor objects (optimizer/RNG/wandb id), so use weights_only=False.
        resume_ckpt = _torch_load(resume_path, weights_only=False)
        ckpt_version = int(resume_ckpt.get("ckpt_version", 1))
        if ckpt_version < 2:
            suggested = ckpt_full_dir / Path(resume_path).name
            raise RuntimeError(
                f"[RESUME] ckpt_version={ckpt_version} is too old for full resume. "
                "This checkpoint does not include optimizer/RNG state. "
                f"Use a full checkpoint under {ckpt_full_dir} (e.g. {suggested}), "
                "or create a new v2 full checkpoint with this updated trainer first."
            )

    env = BatchEnv(
        batch_size=args.batch_size, pf_enabled=not args.no_pf, verbose_build=False
    )
    init_ckpt: dict | None = None
    if init_path is not None:
        init_ckpt = _torch_load_maybe_weights_only(init_path)
        for k in ("model", "in_channels", "hidden", "blocks", "upd"):
            if k not in init_ckpt:
                raise RuntimeError(
                    f"[INIT] invalid checkpoint (missing key: {k!r}): {init_path}"
                )
        _require_equal(
            "arch_name", str(init_ckpt.get("arch_name", "resnet_v1")), str(args.arch)
        )
        _require_equal(
            "in_channels", int(init_ckpt["in_channels"]), int(env.feature_channels)
        )
        _require_equal("hidden", int(init_ckpt["hidden"]), int(args.hidden))
        _require_equal("blocks", int(init_ckpt["blocks"]), int(args.blocks))
    if resume_ckpt is not None:
        _require_equal(
            "arch_name", str(resume_ckpt.get("arch_name", "resnet_v1")), str(args.arch)
        )
        _require_equal(
            "batch_size", int(resume_ckpt["train"]["batch_size"]), int(args.batch_size)
        )
        _require_equal("seed", int(resume_ckpt["train"]["seed"]), int(args.seed))
        _require_equal("lr", float(resume_ckpt["train"]["lr"]), float(args.lr))
        _require_equal("epochs", int(resume_ckpt["train"]["epochs"]), int(args.epochs))
        _require_equal(
            "minibatch", int(resume_ckpt["train"]["minibatch"]), int(args.minibatch)
        )
        _require_equal("no_pf", bool(resume_ckpt["train"]["no_pf"]), bool(args.no_pf))
        _require_equal(
            "no_amp", bool(resume_ckpt["train"]["no_amp"]), bool(args.no_amp)
        )
        _require_equal("gamma", float(resume_ckpt["train"]["gamma"]), float(gamma))
        _require_equal(
            "gae_lambda", float(resume_ckpt["train"]["gae_lambda"]), float(gae_lambda)
        )
        _require_equal("t_max", int(resume_ckpt["train"]["t_max"]), int(env.spec.t_max))
        _require_equal(
            "in_channels", int(resume_ckpt["in_channels"]), int(env.feature_channels)
        )
        _require_equal("hidden", int(resume_ckpt["hidden"]), int(args.hidden))
        _require_equal("blocks", int(resume_ckpt["blocks"]), int(args.blocks))

    model = build_policy_value_model(
        args.arch,
        in_channels=int(env.feature_channels),
        hidden_channels=int(args.hidden),
        blocks=int(args.blocks),
    ).to(device)
    if init_ckpt is not None:
        model.load_state_dict(normalize_state_dict_keys(init_ckpt["model"]))
    opt_kwargs = {"lr": args.lr}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.wandb_mode == "disabled":
        if resume_ckpt is not None:
            raise RuntimeError(
                "[RESUME] --wandb-mode disabled is not supported for full resume."
            )
    elif wandb is None:
        raise RuntimeError("wandb is not available. Install it to run training.")

    start_upd = 1
    if resume_ckpt is not None:
        model.load_state_dict(normalize_state_dict_keys(resume_ckpt["model"]))
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        _optimizer_to_device(optimizer, device)
        _set_rng_state(resume_ckpt["rng"], device)
        start_upd = int(resume_ckpt["upd"]) + 1
        if start_upd > args.updates:
            raise RuntimeError(
                f"[RESUME] ckpt upd={resume_ckpt['upd']} already >= --updates={args.updates}"
            )
    else:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        try:
            import numpy as np  # type: ignore

            np.random.seed(args.seed)
        except Exception:
            pass

    compile_ok = False
    if args.compile and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            compile_ok = True
        except Exception as e:
            print(
                f"[WARN] torch.compile failed ({type(e).__name__}: {e}); falling back to eager."
            )

    run = None
    if args.wandb_mode != "disabled":
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_dir = exp_dir / "artifacts" / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "algo": "ppo",
            "updates": args.updates,
            "batch_size": args.batch_size,
            "t_max": env.spec.t_max,
            "lr": args.lr,
            "arch_name": str(args.arch),
            "hidden": args.hidden,
            "blocks": args.blocks,
            "epochs": args.epochs,
            "minibatch": args.minibatch,
            "seed": args.seed,
            "pf_enabled": not args.no_pf,
            "feature_channels": env.feature_channels,
            "device": str(device),
            "amp_update_bf16": use_amp,
            "amp_rollout_bf16": rollout_amp,
            "compile": bool(args.compile),
            "compile_mode": args.compile_mode,
            "compile_ok": compile_ok,
            "torch_version": torch.__version__,
            "resume_ckpt": str(resume_path) if resume_path is not None else None,
            "init_ckpt": str(init_path) if init_path is not None else None,
        }

        wandb_init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=tags if tags else None,
            mode=args.wandb_mode,
            dir=str(wandb_dir),
            config=config,
        )
        if resume_ckpt is not None:
            wandb_init_kwargs["id"] = str(resume_ckpt["wandb"]["run_id"])
            wandb_init_kwargs["resume"] = "allow"
        run = wandb.init(**wandb_init_kwargs)
        wandb.define_metric("env_steps")
        wandb.define_metric("train/*", step_metric="env_steps")
        wandb.define_metric("loss/*", step_metric="env_steps")
        wandb.define_metric("time/*", step_metric="env_steps")
        wandb.define_metric("hp/*", step_metric="env_steps")

    eval_paths: list[str] | None = None
    eval_env: BatchEnv | None = None
    if args.eval_seeds < 0:
        raise RuntimeError("--eval-seeds must be >= 0")
    if args.eval_every < 0:
        raise RuntimeError("--eval-every must be >= 0")
    if args.eval_batch <= 0:
        raise RuntimeError("--eval-batch must be >= 1")
    if args.eval_seeds > 0:
        eval_paths = tools_input_paths(0, args.eval_seeds - 1)
        eval_env = BatchEnv(
            batch_size=args.eval_batch, pf_enabled=not args.no_pf, verbose_build=False
        )

        eval_env_steps = (start_upd - 1) * args.batch_size * env.spec.t_max
        eval_upd = start_upd - 1
        eval_mean_greedy = _eval_tools(
            eval_env, model, device, eval_paths, sample=False
        )

        rng_backup = _get_rng_state(device)
        eval_sample_seed = int(args.seed) * 1000003 + int(eval_upd) * 10007 + 12345
        torch.manual_seed(eval_sample_seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_sample_seed)
        eval_mean_sample = _eval_tools(eval_env, model, device, eval_paths, sample=True)
        _set_rng_state(rng_backup, device)

        print(
            f"[EVAL] upd={eval_upd:04d} tools_mean_score_greedy={eval_mean_greedy:.3f} "
            f"tools_mean_score_sample={eval_mean_sample:.3f} n={args.eval_seeds}"
        )
        if args.wandb_mode != "disabled":
            wandb.log(
                {
                    "eval/mean_score": eval_mean_greedy,
                    "eval/mean_score_greedy": eval_mean_greedy,
                    "eval/mean_score_sample": eval_mean_sample,
                    "eval/n": int(args.eval_seeds),
                    "eval/sample_seed": int(eval_sample_seed),
                },
                step=int(eval_env_steps),
            )

    for upd in range(start_upd, args.updates + 1):
        seeds = (
            torch.arange(args.batch_size, dtype=torch.int64) + args.seed + upd * 100000
        )
        env.reset_random(seeds)

        t_rollout0 = time.perf_counter()
        rollout = collect_rollout(
            env, model, device, t_max=env.spec.t_max, sample=True, amp=rollout_amp
        )
        t_rollout1 = time.perf_counter()

        adv, ret = compute_gae(
            rewards=rollout.rewards,
            values=rollout.values,
            dones=rollout.dones,
            last_value=rollout.last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

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

        t_update0 = time.perf_counter()
        model.train()
        # Preload rollout tensors to the training device once per update to avoid per-minibatch HtoD copies.
        # If it OOMs, fall back to the old per-minibatch transfer path.
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
                    (mask.to(device) != 0)
                    if mask.dtype != torch.bool
                    else mask.to(device)
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
                # best effort cleanup; continue with per-minibatch transfer
                torch.cuda.empty_cache()

        perm_device = device if preload_ok else torch.device("cpu")
        stats_sum = None
        stats_cnt = 0
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=perm_device)
            for start in range(0, n, args.minibatch):
                mb = perm[start : start + args.minibatch]
                s = ppo_update(
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
                    amp=use_amp,
                )
                if stats_sum is None:
                    stats_sum = s
                else:
                    stats_sum.loss += s.loss
                    stats_sum.policy_loss += s.policy_loss
                    stats_sum.value_loss += s.value_loss
                    stats_sum.entropy += s.entropy
                    stats_sum.approx_kl += s.approx_kl
                    stats_sum.clipfrac += s.clipfrac
                stats_cnt += 1
        t_update1 = time.perf_counter()

        mean_ep_return = float(rollout.rewards.sum(dim=0).mean().item())
        mean_score = float(env.official_score().float().mean().item())

        if stats_sum is not None and stats_cnt > 0:
            stats_sum.loss /= stats_cnt
            stats_sum.policy_loss /= stats_cnt
            stats_sum.value_loss /= stats_cnt
            stats_sum.entropy /= stats_cnt
            stats_sum.approx_kl /= stats_cnt
            stats_sum.clipfrac /= stats_cnt
            print(
                f"upd={upd:04d} return={mean_ep_return:.6f} score={mean_score:.1f} "
                f"loss={stats_sum.loss:.4f} pi={stats_sum.policy_loss:.4f} v={stats_sum.value_loss:.4f} "
                f"ent={stats_sum.entropy:.4f} kl={stats_sum.approx_kl:.6f} clip={stats_sum.clipfrac:.3f}"
            )
        else:
            print(f"upd={upd:04d} return={mean_ep_return:.6f} score={mean_score:.1f}")

        if stats_sum is not None and stats_cnt > 0:
            env_steps = upd * args.batch_size * env.spec.t_max
            rollout_sec = t_rollout1 - t_rollout0
            update_sec = t_update1 - t_update0
            total_sec = rollout_sec + update_sec
            steps = args.batch_size * env.spec.t_max
            metrics = {
                "update": upd,
                "env_steps": env_steps,
                "train/mean_return": mean_ep_return,
                "train/mean_official_score": mean_score,
                "loss/total": stats_sum.loss,
                "loss/policy": stats_sum.policy_loss,
                "loss/value": stats_sum.value_loss,
                "loss/entropy": stats_sum.entropy,
                "loss/approx_kl": stats_sum.approx_kl,
                "loss/clipfrac": stats_sum.clipfrac,
                "time/rollout_sec": rollout_sec,
                "time/update_sec": update_sec,
                "time/total_sec": total_sec,
                "time/rollout_sps": (steps / rollout_sec) if rollout_sec > 0 else 0.0,
                "time/sps": (steps / total_sec) if total_sec > 0 else 0.0,
                "hp/lr": float(optimizer.param_groups[0]["lr"]),
            }
            if args.wandb_mode != "disabled":
                wandb.log(metrics, step=env_steps)

        if args.save_every > 0 and upd % args.save_every == 0:
            env_steps = upd * args.batch_size * env.spec.t_max
            ckpt_model = {
                "model": _unwrap_model(model).state_dict(),
                "in_channels": env.feature_channels,
                "arch_name": str(args.arch),
                "hidden": args.hidden,
                "blocks": args.blocks,
                "upd": upd,
            }
            ckpt = {
                "ckpt_version": 2,
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": _get_rng_state(device),
                "in_channels": env.feature_channels,
                "arch_name": str(args.arch),
                "hidden": args.hidden,
                "blocks": args.blocks,
                "upd": upd,
                "env_steps": env_steps,
                "train": {
                    "batch_size": int(args.batch_size),
                    "seed": int(args.seed),
                    "lr": float(args.lr),
                    "epochs": int(args.epochs),
                    "minibatch": int(args.minibatch),
                    "no_pf": bool(args.no_pf),
                    "no_amp": bool(args.no_amp),
                    "gamma": float(gamma),
                    "gae_lambda": float(gae_lambda),
                    "t_max": int(env.spec.t_max),
                },
                "wandb": {"run_id": str(run.id)}
                if run is not None
                else {"run_id": None},
            }
            ckpt_path = ckpt_dir / f"ckpt_{upd:04d}.pt"
            ckpt_full_path = ckpt_full_dir / f"ckpt_{upd:04d}.pt"
            torch.save(ckpt_model, ckpt_path)
            torch.save(ckpt, ckpt_full_path)
            if args.wandb_log_checkpoints and (run is not None):
                artifact = wandb.Artifact(name=f"ckpt_{upd:04d}", type="checkpoint")
                artifact.add_file(str(ckpt_path))
                run.log_artifact(artifact)

        if (
            eval_env is not None
            and eval_paths is not None
            and args.eval_every > 0
            and (upd % args.eval_every == 0)
        ):
            env_steps = upd * args.batch_size * env.spec.t_max
            eval_mean_greedy = _eval_tools(
                eval_env, model, device, eval_paths, sample=False
            )

            rng_backup = _get_rng_state(device)
            eval_sample_seed = int(args.seed) * 1000003 + int(upd) * 10007 + 12345
            torch.manual_seed(eval_sample_seed)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_sample_seed)
            eval_mean_sample = _eval_tools(
                eval_env, model, device, eval_paths, sample=True
            )
            _set_rng_state(rng_backup, device)

            print(
                f"[EVAL] upd={upd:04d} tools_mean_score_greedy={eval_mean_greedy:.3f} "
                f"tools_mean_score_sample={eval_mean_sample:.3f} n={args.eval_seeds}"
            )
            if args.wandb_mode != "disabled":
                wandb.log(
                    {
                        "eval/mean_score": eval_mean_greedy,
                        "eval/mean_score_greedy": eval_mean_greedy,
                        "eval/mean_score_sample": eval_mean_sample,
                        "eval/n": int(args.eval_seeds),
                        "eval/sample_seed": int(eval_sample_seed),
                    },
                    step=int(env_steps),
                )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
