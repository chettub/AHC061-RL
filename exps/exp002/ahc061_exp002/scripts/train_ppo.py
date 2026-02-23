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
from torch.nn.parallel import DistributedDataParallel as DDP

from ..ckpt import (
    ModelSpec,
    TrainSpec,
    ckpt_full_dict,
    ckpt_model_dict,
    ensure_dir,
    is_full_ckpt,
    model_spec_from_ckpt,
    normalize_state_dict_keys,
    train_spec_from_ckpt,
    wandb_run_id_from_ckpt,
    write_json,
)
from ..env import BatchEnv, tools_input_paths
from ..models import build_policy_value_model, masked_logits
from ..ppo.gae import compute_gae
from ..ppo.rollout import collect_rollout, create_rollout_workspace
from ..ppo.update import ppo_update
from ..run_dir import RunPaths, default_run_dir, init_run_dir

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

AUTO_DDP_LAUNCHED_ENV = "AHC061_EXP002_AUTO_DDP_LAUNCHED"


def _torch_load_any(path: Path, *, weights_only: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for st in optimizer.state.values():
        if not isinstance(st, dict):
            continue
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = getattr(m, "_orig_mod")
    return m


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


def _get_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
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


def _set_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["py_random"])
    if device.type == "cuda" and torch.cuda.is_available() and ("torch_cuda_all" in state):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    if "numpy" in state:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(state["numpy"])
        except Exception:
            pass


def _maybe_warn_override(name: str, cur_v, ckpt_v) -> None:
    if cur_v != ckpt_v:
        print(f"[RESUME][STRICT] override {name}: {cur_v!r} -> {ckpt_v!r}")


def _apply_arch_kwargs_to_args(args: argparse.Namespace, arch_kwargs: dict[str, Any]) -> None:
    args.pp_player_hidden = None
    args.pp_set_layers = None
    args.pp_set_heads = None
    args.pp_set_ff_mult = None
    args.pp_set_dropout = None
    args.pp_set_every = None
    if "player_hidden_channels" in arch_kwargs:
        args.pp_player_hidden = int(arch_kwargs["player_hidden_channels"])
    if "player_set_layers" in arch_kwargs:
        args.pp_set_layers = int(arch_kwargs["player_set_layers"])
    if "player_set_heads" in arch_kwargs:
        args.pp_set_heads = int(arch_kwargs["player_set_heads"])
    if "player_set_ff_mult" in arch_kwargs:
        args.pp_set_ff_mult = float(arch_kwargs["player_set_ff_mult"])
    if "player_set_dropout" in arch_kwargs:
        args.pp_set_dropout = float(arch_kwargs["player_set_dropout"])
    if "player_set_every" in arch_kwargs:
        args.pp_set_every = int(arch_kwargs["player_set_every"])


def _build_arch_kwargs_from_args(args: argparse.Namespace, arch_name: str) -> dict[str, Any]:
    name = str(arch_name).lower()
    if "ppconcat" not in name:
        return {}
    out: dict[str, Any] = {}
    if args.pp_player_hidden is not None:
        out["player_hidden_channels"] = int(args.pp_player_hidden)
    if args.pp_set_layers is not None:
        out["player_set_layers"] = int(args.pp_set_layers)
    if args.pp_set_heads is not None:
        out["player_set_heads"] = int(args.pp_set_heads)
    if args.pp_set_ff_mult is not None:
        out["player_set_ff_mult"] = float(args.pp_set_ff_mult)
    if args.pp_set_dropout is not None:
        out["player_set_dropout"] = float(args.pp_set_dropout)
    if args.pp_set_every is not None:
        out["player_set_every"] = int(args.pp_set_every)
    return out


def _validate_arch_cli_args(args: argparse.Namespace) -> None:
    if args.pp_player_hidden is not None and int(args.pp_player_hidden) <= 0:
        raise RuntimeError("--pp-player-hidden must be >= 1")
    if args.pp_set_layers is not None and int(args.pp_set_layers) < 0:
        raise RuntimeError("--pp-set-layers must be >= 0")
    if args.pp_set_heads is not None and int(args.pp_set_heads) <= 0:
        raise RuntimeError("--pp-set-heads must be >= 1")
    if args.pp_set_ff_mult is not None and float(args.pp_set_ff_mult) <= 0.0:
        raise RuntimeError("--pp-set-ff-mult must be > 0")
    if args.pp_set_dropout is not None:
        d = float(args.pp_set_dropout)
        if not (0.0 <= d < 1.0):
            raise RuntimeError("--pp-set-dropout must be in [0,1)")
    if args.pp_set_every is not None and int(args.pp_set_every) <= 0:
        raise RuntimeError("--pp-set-every must be >= 1")


def _extract_wandb_auto_name_index(run_name: str | None) -> int | None:
    if run_name is None:
        return None
    s = str(run_name).strip()
    if not s:
        return None
    pos = s.rfind("-")
    if pos < 0 or pos + 1 >= len(s):
        return None
    tail = s[pos + 1 :]
    if not tail.isdigit():
        return None
    try:
        return int(tail)
    except ValueError:
        return None


def _resolve_use_channels_last(
    *,
    device: torch.device,
    memory_format_arg: str,
    local_minibatch: int | None = None,
) -> bool:
    if device.type != "cuda":
        return False
    mode = str(memory_format_arg).strip().lower()
    if mode == "channels_last":
        return True
    if mode == "nchw":
        return False
    # auto:
    # For smaller minibatches, layout conversion overhead can dominate.
    if local_minibatch is not None and int(local_minibatch) <= 512:
        return False
    return True


def _tensor_nbytes(x: torch.Tensor) -> int:
    return int(x.numel()) * int(x.element_size())


def _should_cache_rollout_on_gpu(*, mode: str, device: torch.device, total_bytes: int) -> bool:
    if device.type != "cuda":
        return False
    m = str(mode).strip().lower()
    if m == "cpu":
        return False
    if m == "gpu":
        return True
    try:
        free_bytes, total_mem = torch.cuda.mem_get_info(device)
    except Exception:
        return False
    # Keep headroom for model/optimizer states, activations, and allocator fragmentation.
    reserve = max(int(total_mem) // 8, 1_500_000_000)
    return int(total_bytes) + int(reserve) <= int(free_bytes)


def _resolve_gae_device(*, mode: str, device: torch.device) -> torch.device:
    m = str(mode).strip().lower()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if device.type != "cuda":
            raise RuntimeError("--gae-device=cuda requires CUDA device")
        return device
    # auto (safe default): keep GAE on CPU unless explicitly set to cuda.
    return torch.device("cpu")


def _ema_decay_to_name(decay: float) -> str:
    decay_str = format(float(decay), ".10f").rstrip("0").rstrip(".")
    if decay_str == "":
        decay_str = "0"
    return f"ema_decay_{decay_str.replace('.', 'p')}"


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


def _parse_resume_ema_entries(ckpt: dict[str, Any]) -> list[dict[str, Any]]:
    ema = ckpt.get("ema")
    if not isinstance(ema, dict):
        raise RuntimeError("[RESUME] invalid checkpoint (missing key: 'ema')")
    entries = ema.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("[RESUME] invalid checkpoint (missing key: 'ema.entries')")
    ema_models = ckpt.get("ema_models")
    if not isinstance(ema_models, dict):
        raise RuntimeError("[RESUME] invalid checkpoint (missing key: 'ema_models')")

    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for i, ent in enumerate(entries):
        if not isinstance(ent, dict):
            raise RuntimeError(f"[RESUME] invalid checkpoint (ema.entries[{i}] must be dict)")
        if "name" not in ent or "decay" not in ent or "steps" not in ent:
            raise RuntimeError(f"[RESUME] invalid checkpoint (ema.entries[{i}] missing required fields)")
        name = str(ent["name"])
        decay = float(ent["decay"])
        steps = int(ent["steps"])
        if name != _ema_decay_to_name(decay):
            raise RuntimeError(
                f"[RESUME] invalid checkpoint (ema.entries[{i}].name mismatch: {name!r} vs decay {decay})"
            )
        if name in seen_names:
            raise RuntimeError(f"[RESUME] invalid checkpoint (duplicate ema entry name: {name!r})")
        seen_names.add(name)
        if name not in ema_models:
            raise RuntimeError(f"[RESUME] invalid checkpoint (ema_models missing state for {name!r})")
        out.append({"name": name, "decay": decay, "steps": steps})

    _validate_ema_decays([float(e["decay"]) for e in out], source="[RESUME] ema decays")
    return out


@dataclass
class EmaTracker:
    name: str
    decay: float
    model: torch.nn.Module
    steps: int
    param_pairs: list[tuple[torch.Tensor, torch.Tensor]]
    buf_pairs: list[tuple[torch.Tensor, torch.Tensor]]


def _split_counts(total: int, parts: int) -> list[int]:
    q, r = divmod(int(total), int(parts))
    return [q + (1 if i < r else 0) for i in range(int(parts))]


def _split_range(total: int, parts: int, index: int) -> tuple[int, int]:
    counts = _split_counts(int(total), int(parts))
    st = sum(counts[: int(index)])
    return st, st + counts[int(index)]


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
    board = torch.empty((env.batch_size, env.feature_channels, 10, 10), dtype=torch.float32, device="cpu")
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
            logits, _, _, _ = model(board.to(device))
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


def _resolve_resume_path(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.is_dir():
        cand = p / "checkpoints_full" / "ckpt_last.pt"
        if cand.is_file():
            return cand
        cand2 = p / "ckpt_last.pt"
        if cand2.is_file():
            return cand2
        raise FileNotFoundError(f"--resume dir does not contain checkpoints_full/ckpt_last.pt: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--aux-opp-move-coef", type=float, default=0.05)
    parser.add_argument("--aux-opp-param-coef", type=float, default=0.01)

    parser.add_argument("--feature-id", type=str, default="submit_v1")
    parser.add_argument("--arch", type=str, default="resnet_v1")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--pp-player-hidden", type=int, default=None)
    parser.add_argument("--pp-set-layers", type=int, default=None)
    parser.add_argument("--pp-set-heads", type=int, default=None)
    parser.add_argument("--pp-set-ff-mult", type=float, default=None)
    parser.add_argument("--pp-set-dropout", type=float, default=None)
    parser.add_argument("--pp-set-every", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=2048)
    g_shuffle = parser.add_mutually_exclusive_group()
    g_shuffle.add_argument(
        "--shuffle-minibatches",
        dest="shuffle_minibatches",
        action="store_true",
        default=True,
        help="Shuffle samples before PPO minibatch updates (default: on).",
    )
    g_shuffle.add_argument(
        "--no-shuffle-minibatches",
        dest="shuffle_minibatches",
        action="store_false",
        help="Disable sample-level shuffle and use contiguous minibatch slices (faster, but less stochastic).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    g_rollout_amp = parser.add_mutually_exclusive_group()
    g_rollout_amp.add_argument(
        "--rollout-amp",
        dest="rollout_amp",
        action="store_true",
        default=None,
        help="Enable bf16 AMP during rollout on CUDA (default: follow --no-amp).",
    )
    g_rollout_amp.add_argument(
        "--no-rollout-amp",
        dest="rollout_amp",
        action="store_false",
        default=None,
        help="Disable bf16 AMP during rollout.",
    )
    g_fused_aux = parser.add_mutually_exclusive_group()
    g_fused_aux.add_argument(
        "--fused-step-aux",
        dest="fused_step_aux",
        action="store_true",
        default=False,
        help="Use fused env step+observe+aux path during rollout (default: off).",
    )
    g_fused_aux.add_argument(
        "--no-fused-step-aux",
        dest="fused_step_aux",
        action="store_false",
        help="Disable fused env step+observe+aux path (for ablation).",
    )
    parser.add_argument(
        "--ema-decays",
        type=str,
        default="0.999",
        help='Comma-separated EMA decays (e.g. "0.995,0.999"), or "off" to disable EMA.',
    )

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument(
        "--memory-format",
        type=str,
        choices=("auto", "nchw", "channels_last"),
        default="auto",
        help="Tensor memory format for model/inputs on CUDA (auto: pick by local minibatch size).",
    )
    parser.add_argument(
        "--rollout-cache-device",
        type=str,
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Where flattened rollout tensors for PPO update are cached.",
    )
    parser.add_argument(
        "--gae-device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for GAE computation (auto: cpu).",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--distributed",
        type=str,
        choices=("auto", "off", "on"),
        default="auto",
        help="Distributed launch mode. auto: launch torchrun automatically when multiple CUDA devices are visible.",
    )
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)

    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-ckpt", type=str, default=None)

    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--save-last-every", type=int, default=10)

    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=16)

    parser.add_argument("--wandb-project", type=str, default="ahc061-exp002")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online")

    args = parser.parse_args()
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
                    "exps.exp002.ahc061_exp002.scripts.train_ppo",
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
            raise RuntimeError("Distributed training requires CUDA")
        if args.device not in ("auto", "cuda") and (not str(args.device).startswith("cuda")):
            raise RuntimeError("Distributed training requires --device auto/cuda/cuda:N")
        dist.init_process_group(backend="nccl")
    is_main_process = rank == 0

    if args.torch_num_threads is not None:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(args.torch_num_interop_threads))
        except RuntimeError as e:
            print(f"[WARN] torch.set_num_interop_threads failed ({type(e).__name__}: {e})")

    if args.updates <= 0:
        raise RuntimeError("--updates must be >= 1")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be >= 1")
    if args.epochs <= 0:
        raise RuntimeError("--epochs must be >= 1")
    if args.minibatch <= 0:
        raise RuntimeError("--minibatch must be >= 1")
    if args.save_every < 0 or args.save_last_every < 0:
        raise RuntimeError("--save-every/--save-last-every must be >= 0")
    if args.eval_every < 0 or args.eval_seeds < 0 or args.eval_batch <= 0:
        raise RuntimeError("--eval-every/--eval-seeds must be >= 0 and --eval-batch must be >= 1")
    if float(args.ent_coef) < 0.0:
        raise RuntimeError("--ent-coef must be >= 0")
    if float(args.aux_opp_move_coef) < 0.0 or float(args.aux_opp_param_coef) < 0.0:
        raise RuntimeError("--aux-opp-move-coef/--aux-opp-param-coef must be >= 0")
    _validate_arch_cli_args(args)

    if args.resume is not None and args.init_ckpt is not None:
        raise RuntimeError("Specify only one of --resume or --init-ckpt")
    if int(args.warmup_updates) < 0:
        raise RuntimeError("--warmup-updates must be >= 0")
    ema_decays = _parse_ema_decays_arg(str(args.ema_decays))

    def pick_device() -> torch.device:
        if is_distributed:
            if args.device == "auto" or args.device == "cuda":
                return torch.device(f"cuda:{local_rank}")
            req = torch.device(args.device)
            if req.type != "cuda":
                raise RuntimeError("Distributed training requires CUDA device")
            if req.index is not None and int(req.index) != int(local_rank):
                raise RuntimeError(
                    f"--device={args.device!r} conflicts with LOCAL_RANK={local_rank}. "
                    f"Use --device auto/cuda with torchrun."
                )
            return torch.device(f"cuda:{local_rank}")
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
        if device.index is None:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            torch.cuda.set_device(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    exp_dir = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[4]

    resume_ckpt: dict[str, Any] | None = None
    resume_path: Path | None = None
    resume_wandb_id: str | None = None
    resume_ema_entries: list[dict[str, Any]] | None = None
    if args.resume is not None:
        resume_path = _resolve_resume_path(args.resume)
        resume_ckpt = _torch_load_any(resume_path, weights_only=False)
        if not is_full_ckpt(resume_ckpt):
            raise RuntimeError(f"[RESUME] not a full checkpoint: {resume_path}")
        resume_wandb_id = wandb_run_id_from_ckpt(resume_ckpt)
        resume_ema_entries = _parse_resume_ema_entries(resume_ckpt)

    # Strict resume: ckpt優先で強制一致（差分があれば上書き）
    if resume_ckpt is not None:
        ms = model_spec_from_ckpt(resume_ckpt)
        ts = train_spec_from_ckpt(resume_ckpt)
        if ts is None:
            raise RuntimeError(f"[RESUME] invalid checkpoint (missing train_spec): {resume_path}")
        if resume_ema_entries is None:
            raise RuntimeError("[RESUME] internal error: resume_ema_entries is None")
        ckpt_ema_decays = [float(e["decay"]) for e in resume_ema_entries]
        _maybe_warn_override("ema_decays", ema_decays, ckpt_ema_decays)
        ema_decays = ckpt_ema_decays

        cur_arch_kwargs = _build_arch_kwargs_from_args(args, str(args.arch))
        _maybe_warn_override("arch", str(args.arch), ms.arch_name)
        _maybe_warn_override("feature_id", str(args.feature_id), ms.feature_id)
        _maybe_warn_override("hidden", int(args.hidden), int(ms.hidden))
        _maybe_warn_override("blocks", int(args.blocks), int(ms.blocks))
        _maybe_warn_override("arch_kwargs", cur_arch_kwargs, dict(ms.arch_kwargs))
        args.arch = ms.arch_name
        args.feature_id = ms.feature_id
        args.hidden = ms.hidden
        args.blocks = ms.blocks
        _apply_arch_kwargs_to_args(args, dict(ms.arch_kwargs))

        _maybe_warn_override("seed", int(args.seed), int(ts.seed))
        _maybe_warn_override("batch_size", int(args.batch_size), int(ts.batch_size))
        _maybe_warn_override("lr", float(args.lr), float(ts.lr))
        _maybe_warn_override("warmup_updates", int(args.warmup_updates), int(getattr(ts, "warmup_updates", 0)))
        _maybe_warn_override("epochs", int(args.epochs), int(ts.epochs))
        _maybe_warn_override("minibatch", int(args.minibatch), int(ts.minibatch))
        _maybe_warn_override("gamma", float(args.gamma), float(ts.gamma))
        _maybe_warn_override("gae_lambda", float(args.gae_lambda), float(ts.gae_lambda))
        _maybe_warn_override("ent_coef", float(args.ent_coef), float(getattr(ts, "ent_coef", 0.01)))
        _maybe_warn_override("aux_opp_move_coef", float(args.aux_opp_move_coef), float(getattr(ts, "aux_opp_move_coef", 0.0)))
        _maybe_warn_override("aux_opp_param_coef", float(args.aux_opp_param_coef), float(getattr(ts, "aux_opp_param_coef", 0.0)))
        _maybe_warn_override("pf_enabled", bool(not args.no_pf), bool(ts.pf_enabled))
        _maybe_warn_override("amp", bool(not args.no_amp), bool(ts.amp))
        _maybe_warn_override("rollout_amp", args.rollout_amp, bool(ts.rollout_amp))
        args.seed = ts.seed
        args.batch_size = ts.batch_size
        args.lr = ts.lr
        args.warmup_updates = getattr(ts, "warmup_updates", 0)
        args.epochs = ts.epochs
        args.minibatch = ts.minibatch
        args.gamma = ts.gamma
        args.gae_lambda = ts.gae_lambda
        args.ent_coef = getattr(ts, "ent_coef", 0.01)
        args.aux_opp_move_coef = getattr(ts, "aux_opp_move_coef", 0.0)
        args.aux_opp_param_coef = getattr(ts, "aux_opp_param_coef", 0.0)
        args.no_pf = not ts.pf_enabled
        args.no_amp = not ts.amp
        args.rollout_amp = bool(ts.rollout_amp)

    model_arch_kwargs = _build_arch_kwargs_from_args(args, str(args.arch))
    collect_aux_targets = (float(args.aux_opp_move_coef) > 0.0) or (float(args.aux_opp_param_coef) > 0.0)

    if int(args.batch_size) % int(world_size) != 0:
        raise RuntimeError(f"--batch-size must be divisible by WORLD_SIZE ({args.batch_size} vs {world_size})")
    if int(args.minibatch) % int(world_size) != 0:
        raise RuntimeError(f"--minibatch must be divisible by WORLD_SIZE ({args.minibatch} vs {world_size})")
    local_batch_size = int(args.batch_size) // int(world_size)
    local_minibatch = int(args.minibatch) // int(world_size)
    if local_batch_size <= 0 or local_minibatch <= 0:
        raise RuntimeError("local batch/minibatch must be >= 1")
    use_channels_last = _resolve_use_channels_last(
        device=device,
        memory_format_arg=str(args.memory_format),
        local_minibatch=int(local_minibatch),
    )

    # Run dir
    run_dir: Path | None = None
    if is_main_process:
        if args.run_dir is not None:
            run_dir = (repo_root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
        elif resume_ckpt is not None and (resume_ckpt.get("run_dir") is not None):
            run_dir = Path(str(resume_ckpt["run_dir"])).resolve()
        else:
            run_dir = default_run_dir(exp_dir=exp_dir, name=str(args.run_name) if args.run_name else None)
    run_dir_box = [str(run_dir) if run_dir is not None else ""]
    if is_distributed:
        dist.broadcast_object_list(run_dir_box, src=0)
    if run_dir_box[0] == "":
        raise RuntimeError("failed to determine run_dir")
    run_dir = Path(run_dir_box[0]).resolve()

    paths = RunPaths.from_run_dir(run_dir)

    if resume_ckpt is None:
        if is_main_process:
            if paths.run_dir.exists() and any(paths.run_dir.iterdir()):
                raise RuntimeError(f"[RUN] run_dir already exists and is not empty: {paths.run_dir}")
            init_run_dir(
                paths,
                config={
                    "algo": "ppo",
                    "updates": int(args.updates),
                    "batch_size": int(args.batch_size),
                    "batch_size_per_rank": int(local_batch_size),
                    "world_size": int(world_size),
                    "lr": float(args.lr),
                    "gamma": float(args.gamma),
                    "gae_lambda": float(args.gae_lambda),
                    "ent_coef": float(args.ent_coef),
                    "aux_opp_move_coef": float(args.aux_opp_move_coef),
                    "aux_opp_param_coef": float(args.aux_opp_param_coef),
                    "collect_aux_targets": bool(collect_aux_targets),
                    "feature_id": str(args.feature_id),
                    "arch_name": str(args.arch),
                    "arch_kwargs": dict(model_arch_kwargs),
                    "hidden": int(args.hidden),
                    "blocks": int(args.blocks),
                    "epochs": int(args.epochs),
                    "minibatch": int(args.minibatch),
                    "minibatch_per_rank": int(local_minibatch),
                    "shuffle_minibatches": bool(args.shuffle_minibatches),
                    "seed": int(args.seed),
                    "pf_enabled": bool(not args.no_pf),
                    "amp": bool(not args.no_amp),
                    "memory_format": ("channels_last" if use_channels_last else "nchw"),
                    "rollout_cache_device": str(args.rollout_cache_device),
                    "gae_device": str(args.gae_device),
                    "fused_step_aux": bool(args.fused_step_aux),
                    "ema_decays": [float(d) for d in ema_decays],
                    "device": str(device),
                    "torch_version": torch.__version__,
                },
            )
        if is_distributed:
            dist.barrier()
    else:
        if is_main_process:
            ensure_dir(paths.ckpt_dir)
            ensure_dir(paths.ckpt_full_dir)
            ensure_dir(paths.ckpt_ema_dir)
            ensure_dir(paths.wandb_dir)
        if is_distributed:
            dist.barrier()

    env = BatchEnv(
        batch_size=int(local_batch_size),
        feature_id=str(args.feature_id),
        pf_enabled=not args.no_pf,
        verbose_build=False,
    )

    init_ckpt: dict[str, Any] | None = None
    init_path: Path | None = None
    if args.init_ckpt is not None:
        init_path = (repo_root / args.init_ckpt).resolve() if not Path(args.init_ckpt).is_absolute() else Path(args.init_ckpt).resolve()
        init_ckpt = _torch_load_any(init_path, weights_only=True)
        ims = model_spec_from_ckpt(init_ckpt)
        if ims.arch_name != str(args.arch):
            raise RuntimeError(f"[INIT] arch mismatch: ckpt={ims.arch_name!r} current={str(args.arch)!r}")
        if ims.feature_id != str(args.feature_id):
            raise RuntimeError(f"[INIT] feature_id mismatch: ckpt={ims.feature_id!r} current={str(args.feature_id)!r}")
        if ims.hidden != int(args.hidden) or ims.blocks != int(args.blocks):
            raise RuntimeError(
                f"[INIT] model size mismatch: ckpt hidden/blocks={ims.hidden}/{ims.blocks} current={int(args.hidden)}/{int(args.blocks)}"
            )
        if dict(ims.arch_kwargs) != dict(model_arch_kwargs):
            raise RuntimeError(
                f"[INIT] arch_kwargs mismatch: ckpt={dict(ims.arch_kwargs)!r} current={dict(model_arch_kwargs)!r}"
            )

    model = build_policy_value_model(
        str(args.arch),
        in_channels=int(env.feature_channels),
        hidden_channels=int(args.hidden),
        blocks=int(args.blocks),
        feature_id=str(args.feature_id),
        arch_kwargs=dict(model_arch_kwargs),
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    opt_kwargs: dict[str, Any] = {"lr": float(args.lr)}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    start_upd = 1
    run = None

    if resume_ckpt is not None:
        ms = model_spec_from_ckpt(resume_ckpt)
        if ms.in_channels != int(env.feature_channels):
            raise RuntimeError(f"[RESUME] in_channels mismatch: ckpt={ms.in_channels} env={int(env.feature_channels)}")
        model.load_state_dict(normalize_state_dict_keys(resume_ckpt["model"]))
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        _optimizer_to_device(optimizer, device)
        _set_rng_state(resume_ckpt["rng"], device)
        start_upd = int(resume_ckpt["upd"]) + 1
        if start_upd > int(args.updates):
            raise RuntimeError(f"[RESUME] ckpt upd={int(resume_ckpt['upd'])} already >= --updates={int(args.updates)}")
    else:
        seed_base = int(args.seed) + int(rank) * 1000003
        torch.manual_seed(seed_base)
        random.seed(seed_base)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base)
        try:
            import numpy as np  # type: ignore

            np.random.seed(seed_base)
        except Exception:
            pass
        if init_ckpt is not None:
            missing, unexpected = model.load_state_dict(normalize_state_dict_keys(init_ckpt["model"]), strict=False)
            if unexpected:
                raise RuntimeError(f"[INIT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
            allowed_missing_prefixes = ("opp_move_head.", "opp_param_head.")
            bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
            if bad_missing:
                raise RuntimeError(f"[INIT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
            if missing:
                print(f"[INIT] loaded with missing aux head keys: {len(missing)}")

    compile_ok = False
    if args.compile and device.type == "cuda":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        try:
            model = torch.compile(model, mode=str(args.compile_mode))
            compile_ok = True
        except Exception as e:
            if is_main_process:
                print(f"[WARN] torch.compile failed ({type(e).__name__}: {e}); falling back to eager.")

    if is_distributed:
        ddp_device_id = device.index if device.index is not None else local_rank
        model = DDP(
            model,
            device_ids=[int(ddp_device_id)],
            output_device=int(ddp_device_id),
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    ema_trackers: list[EmaTracker] = []
    online_named_params = dict(_unwrap_model(model).named_parameters())
    online_named_bufs = dict(_unwrap_model(model).named_buffers())
    resume_ema_models: dict[str, Any] | None = None
    resume_ema_steps: dict[str, int] = {}
    if resume_ckpt is not None:
        if resume_ema_entries is None:
            raise RuntimeError("[RESUME] internal error: resume_ema_entries is None")
        raw_resume_ema_models = resume_ckpt.get("ema_models")
        if not isinstance(raw_resume_ema_models, dict):
            raise RuntimeError("[RESUME] invalid checkpoint (missing key: 'ema_models')")
        resume_ema_models = raw_resume_ema_models
        resume_ema_steps = {str(e["name"]): int(e["steps"]) for e in resume_ema_entries}

    for decay in ema_decays:
        ema_name = _ema_decay_to_name(decay)
        ema_model = build_policy_value_model(
            str(args.arch),
            in_channels=int(env.feature_channels),
            hidden_channels=int(args.hidden),
            blocks=int(args.blocks),
            feature_id=str(args.feature_id),
            arch_kwargs=dict(model_arch_kwargs),
        ).to(device)
        if resume_ckpt is not None:
            if resume_ema_models is None or ema_name not in resume_ema_models:
                raise RuntimeError(f"[RESUME] invalid checkpoint (ema_models missing {ema_name!r})")
            if ema_name not in resume_ema_steps:
                raise RuntimeError(f"[RESUME] invalid checkpoint (ema step missing {ema_name!r})")
            ema_model.load_state_dict(normalize_state_dict_keys(resume_ema_models[ema_name]))
            ema_steps = int(resume_ema_steps[ema_name])
        else:
            ema_model.load_state_dict(normalize_state_dict_keys(_unwrap_model(model).state_dict()))
            ema_steps = 0
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
                steps=int(ema_steps),
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

    use_amp = device.type == "cuda" and (not args.no_amp)
    if args.rollout_amp is None:
        rollout_amp = bool(use_amp)
    else:
        rollout_amp = (device.type == "cuda") and bool(args.rollout_amp)
    gae_device = _resolve_gae_device(mode=str(args.gae_device), device=device)

    base_lrs = [float(args.lr) for _ in optimizer.param_groups]

    def set_lr_scale(scale: float) -> None:
        s = float(scale)
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i] * s

    if args.wandb_mode != "disabled" and wandb is None:
        raise RuntimeError("wandb is not available. Install it to run training, or use --wandb-mode disabled.")

    if args.wandb_mode != "disabled" and is_main_process:
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        config = {
            "algo": "ppo",
            "updates": int(args.updates),
            "batch_size": int(args.batch_size),
            "batch_size_per_rank": int(local_batch_size),
            "t_max": int(env.spec.t_max),
            "lr": float(args.lr),
            "warmup_updates": int(args.warmup_updates),
            "clip_coef": float(args.clip_coef),
            "vf_clip_coef": float(args.vf_clip_coef),
            "ent_coef": float(args.ent_coef),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "aux_opp_move_coef": float(args.aux_opp_move_coef),
            "aux_opp_param_coef": float(args.aux_opp_param_coef),
            "collect_aux_targets": bool(collect_aux_targets),
            "feature_id": str(args.feature_id),
            "arch_name": str(args.arch),
            "arch_kwargs": dict(model_arch_kwargs),
            "hidden": int(args.hidden),
            "blocks": int(args.blocks),
            "epochs": int(args.epochs),
            "minibatch": int(args.minibatch),
            "minibatch_per_rank": int(local_minibatch),
            "shuffle_minibatches": bool(args.shuffle_minibatches),
            "seed": int(args.seed),
            "pf_enabled": bool(not args.no_pf),
            "ema_decays": [float(d) for d in ema_decays],
            "device": str(device),
            "memory_format": ("channels_last" if use_channels_last else "nchw"),
            "rollout_cache_device": str(args.rollout_cache_device),
            "gae_device": str(args.gae_device),
            "fused_step_aux": bool(args.fused_step_aux),
            "world_size": int(world_size),
            "rank": int(rank),
            "amp_update_bf16": bool(use_amp),
            "amp_rollout_bf16": bool(rollout_amp),
            "compile": bool(args.compile),
            "compile_mode": str(args.compile_mode),
            "compile_ok": bool(compile_ok),
            "torch_version": torch.__version__,
            "run_dir": str(paths.run_dir),
            "resume_ckpt": str(resume_path) if resume_path is not None else None,
            "init_ckpt": str(init_path) if init_path is not None else None,
        }

        wandb_init_kwargs: dict[str, Any] = dict(
            project=str(args.wandb_project),
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=tags if tags else None,
            mode=str(args.wandb_mode),
            dir=str(paths.wandb_dir),
            config=config,
        )
        if resume_ckpt is not None:
            rid = wandb_run_id_from_ckpt(resume_ckpt)
            if rid is not None:
                wandb_init_kwargs["id"] = rid
                wandb_init_kwargs["resume"] = "allow"
        run = wandb.init(**wandb_init_kwargs)
        # Use `upd` as the canonical x-axis step (requested).
        wandb.define_metric("upd")
        wandb.define_metric("env_steps", step_metric="upd")
        wandb.define_metric("train/*", step_metric="upd")
        wandb.define_metric("loss/*", step_metric="upd")
        wandb.define_metric("eval/*", step_metric="upd")
        wandb.define_metric("time/*", step_metric="upd")
        wandb.define_metric("hp/*", step_metric="upd")
        # For auto-generated names such as "trim-haze-52", prefix run_dir with "052_".
        if args.run_dir is None and resume_ckpt is None and args.wandb_name is None:
            auto_idx = _extract_wandb_auto_name_index(getattr(run, "name", None))
            if auto_idx is not None:
                prefixed_run_dir = paths.run_dir.parent / f"{auto_idx:03d}_{paths.run_dir.name}"
                if prefixed_run_dir != paths.run_dir:
                    if prefixed_run_dir.exists():
                        raise RuntimeError(f"[RUN] target run_dir already exists: {prefixed_run_dir}")
                    old_run_dir = paths.run_dir
                    old_run_dir.rename(prefixed_run_dir)
                    paths = RunPaths.from_run_dir(prefixed_run_dir.resolve())
                    print(f"[RUN] run_dir renamed by wandb auto index: {old_run_dir.name} -> {paths.run_dir.name}")
                    run.config.update({"run_dir": str(paths.run_dir)}, allow_val_change=True)

    if is_distributed:
        run_dir_box2 = [str(paths.run_dir) if is_main_process else ""]
        dist.broadcast_object_list(run_dir_box2, src=0)
        if run_dir_box2[0] == "":
            raise RuntimeError("failed to synchronize run_dir")
        paths = RunPaths.from_run_dir(Path(run_dir_box2[0]).resolve())

    # eval settings
    eval_paths: list[str] | None = None
    eval_env: BatchEnv | None = None
    rollout_cache_info_printed = False
    rollout_cache_decided = False
    rollout_cache_on_gpu = False

    def _set_eval_seed(seed: int) -> None:
        torch.manual_seed(int(seed))
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _collect_eval_task_values(local_values: torch.Tensor, *, label: str) -> list[float]:
        if not is_distributed:
            return [float(v) for v in local_values.to("cpu").tolist()]
        values = torch.where(torch.isnan(local_values), torch.zeros_like(local_values), local_values)
        counts = (~torch.isnan(local_values)).to(dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if is_main_process:
            counts_cpu = counts.to("cpu").tolist()
            bad = [i for i, c in enumerate(counts_cpu) if int(round(float(c))) != 1]
            if bad:
                raise RuntimeError(f"[EVAL] {label}: invalid task ownership at indices {bad[:8]}")
            return [float(v) for v in values.to("cpu").tolist()]
        return []

    def _run_eval(eval_upd: int, eval_env_steps: int) -> None:
        if eval_env is None or eval_paths is None:
            return
        eval_model = _unwrap_model(model)
        eval_models: list[tuple[str, torch.nn.Module]] = [("online", eval_model)] + [(tr.name, tr.model) for tr in ema_trackers]
        n_tasks = len(eval_models)
        if n_tasks <= 0:
            return
        task_start, task_end = _split_range(n_tasks, int(world_size), int(rank))

        local_greedy = torch.full((n_tasks,), float("nan"), dtype=torch.float64, device=device)
        for i, (_, m) in enumerate(eval_models):
            if int(i) < int(task_start) or int(i) >= int(task_end):
                continue
            local_greedy[i] = float(_eval_tools(eval_env, m, device, eval_paths, sample=False))
        greedy_vals = _collect_eval_task_values(local_greedy, label="greedy")

        eval_mean_greedy = 0.0
        ema_mean_greedy: dict[str, float] = {}
        if is_main_process:
            eval_mean_greedy = float(greedy_vals[0])
            for i, (name, _) in enumerate(eval_models):
                if i == 0:
                    continue
                ema_mean_greedy[name] = float(greedy_vals[i])

        rng_backup = _get_rng_state(device)
        eval_sample_seed = int(args.seed) * 1000003 + int(eval_upd) * 10007 + 12345
        local_sample = torch.full((n_tasks,), float("nan"), dtype=torch.float64, device=device)
        for i, (_, m) in enumerate(eval_models):
            if int(i) < int(task_start) or int(i) >= int(task_end):
                continue
            _set_eval_seed(eval_sample_seed)
            local_sample[i] = float(_eval_tools(eval_env, m, device, eval_paths, sample=True))
        _set_rng_state(rng_backup, device)
        sample_vals = _collect_eval_task_values(local_sample, label="sample")

        eval_mean_sample = 0.0
        ema_mean_sample: dict[str, float] = {}
        if is_main_process:
            eval_mean_sample = float(sample_vals[0])
            for i, (name, _) in enumerate(eval_models):
                if i == 0:
                    continue
                ema_mean_sample[name] = float(sample_vals[i])

        if is_main_process:
            msg = (
                f"[EVAL] upd={int(eval_upd):04d} tools_mean_score_greedy={eval_mean_greedy:.3f} "
                f"tools_mean_score_sample={eval_mean_sample:.3f}"
            )
            for tr in ema_trackers:
                msg += (
                    f" {tr.name}_mean_score_greedy={ema_mean_greedy[tr.name]:.3f}"
                    f" {tr.name}_mean_score_sample={ema_mean_sample[tr.name]:.3f}"
                )
            msg += f" n={int(args.eval_seeds)}"
            print(msg)

        if run is not None:
            metrics: dict[str, Any] = {
                "upd": int(eval_upd),
                "env_steps": int(eval_env_steps),
                "eval/mean_score": float(eval_mean_greedy),
                "eval/mean_score_greedy": float(eval_mean_greedy),
                "eval/mean_score_sample": float(eval_mean_sample),
                "eval/n": int(args.eval_seeds),
                "eval/sample_seed": int(eval_sample_seed),
            }
            for tr in ema_trackers:
                metrics[f"eval/{tr.name}/mean_score_greedy"] = float(ema_mean_greedy[tr.name])
                metrics[f"eval/{tr.name}/mean_score_sample"] = float(ema_mean_sample[tr.name])
                metrics[f"eval/{tr.name}/decay"] = float(tr.decay)
                metrics[f"eval/{tr.name}/steps"] = int(tr.steps)
            wandb.log(metrics, step=int(eval_upd))

    if int(args.eval_seeds) > 0:
        eval_paths = tools_input_paths(0, int(args.eval_seeds) - 1)
        eval_env = BatchEnv(
            batch_size=int(args.eval_batch),
            feature_id=str(args.feature_id),
            pf_enabled=not args.no_pf,
            verbose_build=False,
        )
        eval_upd = start_upd - 1
        eval_env_steps = int(eval_upd) * int(args.batch_size) * int(env.spec.t_max)
        _run_eval(eval_upd=eval_upd, eval_env_steps=eval_env_steps)

    rollout_workspace = create_rollout_workspace(
        env,
        t_max=int(env.spec.t_max),
        device=device,
        channels_last=use_channels_last,
        pin_memory=(device.type == "cuda"),
        collect_aux=bool(collect_aux_targets),
    )

    for upd in range(start_upd, int(args.updates) + 1):
        if int(args.warmup_updates) > 0:
            set_lr_scale(float(upd) / float(int(args.warmup_updates)))
        else:
            set_lr_scale(1.0)

        seed_base = int(args.seed) + upd * 100000 + int(rank) * int(local_batch_size)
        seeds = torch.arange(int(local_batch_size), dtype=torch.int64) + seed_base
        env.reset_random(seeds)

        t_rollout0 = time.perf_counter()
        rollout = collect_rollout(
            env,
            model,
            device,
            t_max=int(env.spec.t_max),
            sample=True,
            amp=rollout_amp,
            channels_last=use_channels_last,
            collect_aux=bool(collect_aux_targets),
            fused_step_aux=bool(args.fused_step_aux),
            workspace=rollout_workspace,
        )
        t_rollout1 = time.perf_counter()

        adv, ret = compute_gae(
            rewards=rollout.rewards,
            values=rollout.values,
            dones=rollout.dones,
            last_value=rollout.last_value,
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            device=gae_device,
        )

        t_max, bsz = rollout.actions.shape
        n = t_max * bsz
        obs = rollout.obs.reshape(n, env.feature_channels, 10, 10)
        mask = rollout.mask.reshape(n, 100)
        opp_move_dist = (
            rollout.opp_move_dist.reshape(n, int(env.spec.m_max), 100) if rollout.opp_move_dist is not None else None
        )
        opp_param = rollout.opp_param.reshape(n, int(env.spec.m_max), 5) if rollout.opp_param is not None else None
        opp_valid = rollout.opp_valid.reshape(n, int(env.spec.m_max)) if rollout.opp_valid is not None else None
        actions = rollout.actions.reshape(n)
        old_logp = rollout.logp.reshape(n)
        old_values = rollout.values.reshape(n)
        advantages = adv.reshape(n)
        returns = ret.reshape(n)

        rollout_cache_nbytes = (
            _tensor_nbytes(obs)
            + _tensor_nbytes(mask)
            + _tensor_nbytes(actions)
            + _tensor_nbytes(old_logp)
            + _tensor_nbytes(old_values)
            + _tensor_nbytes(advantages)
            + _tensor_nbytes(returns)
        )
        if opp_move_dist is not None:
            rollout_cache_nbytes += _tensor_nbytes(opp_move_dist)
        if opp_param is not None:
            rollout_cache_nbytes += _tensor_nbytes(opp_param)
        if opp_valid is not None:
            rollout_cache_nbytes += _tensor_nbytes(opp_valid)
        if not rollout_cache_decided:
            rollout_cache_on_gpu = _should_cache_rollout_on_gpu(
                mode=str(args.rollout_cache_device),
                device=device,
                total_bytes=int(rollout_cache_nbytes),
            )
            rollout_cache_decided = True
        if (not rollout_cache_on_gpu) and advantages.is_cuda:
            advantages = advantages.to(device="cpu")
            returns = returns.to(device="cpu")
        if is_main_process and (not rollout_cache_info_printed):
            cache_gib = float(rollout_cache_nbytes) / float(1024**3)
            cache_mode = "gpu" if rollout_cache_on_gpu else "cpu"
            print(
                f"[PERF] rollout cache device: {cache_mode} "
                f"(mode={str(args.rollout_cache_device)}, estimated={cache_gib:.2f} GiB) "
                f"memory_format={'channels_last' if use_channels_last else 'nchw'} "
                f"shuffle={'on' if bool(args.shuffle_minibatches) else 'off'} "
                f"aux_targets={'on' if collect_aux_targets else 'off'} "
                f"fused_step_aux={'on' if bool(args.fused_step_aux) else 'off'} "
                f"gae_device={str(gae_device)}"
            )
            rollout_cache_info_printed = True

        # normalize advantages
        adv_stat = torch.tensor(
            [
                float(advantages.sum().item()),
                float(advantages.square().sum().item()),
                float(advantages.numel()),
            ],
            device=device,
            dtype=torch.float64,
        )
        _all_reduce_sum_tensor(adv_stat)
        adv_mean = float((adv_stat[0] / adv_stat[2]).item())
        adv_var = float((adv_stat[1] / adv_stat[2] - (adv_stat[0] / adv_stat[2]).square()).item())
        if adv_var < 0.0:
            adv_var = 0.0
        advantages = (advantages - adv_mean) / math.sqrt(adv_var + 1e-8)

        model.train()

        t_update0 = time.perf_counter()
        stats_sum = None
        stats_cnt = 0
        local_steps = int(n)
        obs_dev = None
        mask_dev = None
        opp_move_dist_dev = None
        opp_param_dev = None
        opp_valid_dev = None
        actions_dev = None
        old_logp_dev = None
        old_values_dev = None
        advantages_dev = None
        returns_dev = None
        mask_cpu_bool = None
        opp_valid_cpu_bool = None

        if rollout_cache_on_gpu:
            obs_dev = obs.to(device=device, non_blocking=True)
            if use_channels_last:
                obs_dev = obs_dev.contiguous(memory_format=torch.channels_last)
            mask_dev = mask.to(device=device, dtype=torch.bool, non_blocking=True)
            if opp_move_dist is not None:
                opp_move_dist_dev = opp_move_dist.to(device=device, non_blocking=True)
            if opp_param is not None:
                opp_param_dev = opp_param.to(device=device, non_blocking=True)
            if opp_valid is not None:
                opp_valid_dev = opp_valid.to(device=device, dtype=torch.bool, non_blocking=True)
            actions_dev = actions.to(device=device, non_blocking=True)
            old_logp_dev = old_logp.to(device=device, non_blocking=True)
            old_values_dev = old_values.to(device=device, non_blocking=True)
            advantages_dev = advantages.to(device=device, non_blocking=True)
            returns_dev = returns.to(device=device, non_blocking=True)
        elif device.type == "cpu":
            mask_cpu_bool = mask != 0 if mask.dtype != torch.bool else mask
            if opp_valid is not None:
                opp_valid_cpu_bool = opp_valid != 0 if opp_valid.dtype != torch.bool else opp_valid

        single_mb = int(local_minibatch) >= int(n)
        for _ in range(int(args.epochs)):
            if single_mb:
                mb_iter = (slice(0, n),)
            else:
                perm = (
                    torch.randperm(n, device=device if rollout_cache_on_gpu else "cpu")
                    if bool(args.shuffle_minibatches)
                    else None
                )
                if perm is None:
                    mb_iter = (slice(start, start + int(local_minibatch)) for start in range(0, n, int(local_minibatch)))
                else:
                    mb_iter = (perm[start : start + int(local_minibatch)] for start in range(0, n, int(local_minibatch)))
            for mb in mb_iter:
                if rollout_cache_on_gpu:
                    assert obs_dev is not None
                    assert mask_dev is not None
                    assert actions_dev is not None
                    assert old_logp_dev is not None
                    assert old_values_dev is not None
                    assert advantages_dev is not None
                    assert returns_dev is not None
                    stats = ppo_update(
                        model,
                        optimizer,
                        obs_dev[mb],
                        mask_dev[mb],
                        (opp_move_dist_dev[mb] if opp_move_dist_dev is not None else None),
                        (opp_param_dev[mb] if opp_param_dev is not None else None),
                        (opp_valid_dev[mb] if opp_valid_dev is not None else None),
                        actions_dev[mb],
                        old_logp_dev[mb],
                        old_values_dev[mb],
                        advantages_dev[mb],
                        returns_dev[mb],
                        clip_coef=float(args.clip_coef),
                        vf_clip_coef=float(args.vf_clip_coef),
                        ent_coef=float(args.ent_coef),
                        aux_opp_move_coef=float(args.aux_opp_move_coef),
                        aux_opp_param_coef=float(args.aux_opp_param_coef),
                        amp=bool(use_amp),
                    )
                else:
                    if device.type == "cpu":
                        obs_mb = obs[mb]
                        if use_channels_last:
                            obs_mb = obs_mb.contiguous(memory_format=torch.channels_last)
                        assert mask_cpu_bool is not None
                        stats = ppo_update(
                            model,
                            optimizer,
                            obs_mb,
                            mask_cpu_bool[mb],
                            (opp_move_dist[mb] if opp_move_dist is not None else None),
                            (opp_param[mb] if opp_param is not None else None),
                            (opp_valid_cpu_bool[mb] if opp_valid_cpu_bool is not None else None),
                            actions[mb],
                            old_logp[mb],
                            old_values[mb],
                            advantages[mb],
                            returns[mb],
                            clip_coef=float(args.clip_coef),
                            vf_clip_coef=float(args.vf_clip_coef),
                            ent_coef=float(args.ent_coef),
                            aux_opp_move_coef=float(args.aux_opp_move_coef),
                            aux_opp_param_coef=float(args.aux_opp_param_coef),
                            amp=False,
                        )
                    else:
                        obs_mb = obs[mb].to(device=device, non_blocking=True)
                        if use_channels_last:
                            obs_mb = obs_mb.contiguous(memory_format=torch.channels_last)
                        stats = ppo_update(
                            model,
                            optimizer,
                            obs_mb,
                            mask[mb].to(device=device, dtype=torch.bool, non_blocking=True),
                            (opp_move_dist[mb].to(device=device, non_blocking=True) if opp_move_dist is not None else None),
                            (opp_param[mb].to(device=device, non_blocking=True) if opp_param is not None else None),
                            (opp_valid[mb].to(device=device, dtype=torch.bool, non_blocking=True) if opp_valid is not None else None),
                            actions[mb].to(device=device, non_blocking=True),
                            old_logp[mb].to(device=device, non_blocking=True),
                            old_values[mb].to(device=device, non_blocking=True),
                            advantages[mb].to(device=device, non_blocking=True),
                            returns[mb].to(device=device, non_blocking=True),
                            clip_coef=float(args.clip_coef),
                            vf_clip_coef=float(args.vf_clip_coef),
                            ent_coef=float(args.ent_coef),
                            aux_opp_move_coef=float(args.aux_opp_move_coef),
                            aux_opp_param_coef=float(args.aux_opp_param_coef),
                            amp=bool(use_amp),
                        )
                if ema_trackers:
                    ema_update()
                if stats_sum is None:
                    stats_sum = stats
                else:
                    stats_sum.loss += stats.loss
                    stats_sum.policy_loss += stats.policy_loss
                    stats_sum.value_loss += stats.value_loss
                    stats_sum.aux_opp_move_loss += stats.aux_opp_move_loss
                    stats_sum.aux_opp_param_loss += stats.aux_opp_param_loss
                    stats_sum.entropy += stats.entropy
                    stats_sum.approx_kl += stats.approx_kl
                    stats_sum.clipfrac += stats.clipfrac
                stats_cnt += 1
        t_update1 = time.perf_counter()

        if stats_sum is None:
            raise RuntimeError("ppo_update produced no stats")
        stats_core = torch.stack(
            [
                stats_sum.loss,
                stats_sum.policy_loss,
                stats_sum.value_loss,
                stats_sum.aux_opp_move_loss,
                stats_sum.aux_opp_param_loss,
                stats_sum.entropy,
                stats_sum.approx_kl,
                stats_sum.clipfrac,
            ]
        ).to(dtype=torch.float64)
        stats_vec = torch.cat(
            (
                stats_core,
                torch.tensor([float(stats_cnt)], device=stats_core.device, dtype=torch.float64),
            )
        )
        _all_reduce_sum_tensor(stats_vec)
        global_stats_cnt = max(1.0, float(stats_vec[8].item()))
        stats_sum.loss = float(stats_vec[0].item() / global_stats_cnt)
        stats_sum.policy_loss = float(stats_vec[1].item() / global_stats_cnt)
        stats_sum.value_loss = float(stats_vec[2].item() / global_stats_cnt)
        stats_sum.aux_opp_move_loss = float(stats_vec[3].item() / global_stats_cnt)
        stats_sum.aux_opp_param_loss = float(stats_vec[4].item() / global_stats_cnt)
        stats_sum.entropy = float(stats_vec[5].item() / global_stats_cnt)
        stats_sum.approx_kl = float(stats_vec[6].item() / global_stats_cnt)
        stats_sum.clipfrac = float(stats_vec[7].item() / global_stats_cnt)

        env_steps = upd * int(args.batch_size) * int(env.spec.t_max)
        score_now = env.official_score().float()
        metric_vec = torch.tensor(
            [
                float(rollout.rewards.sum().item()),
                float(rollout.rewards.shape[1]),
                float(score_now.sum().item()),
                float(score_now.numel()),
            ],
            device=device,
            dtype=torch.float64,
        )
        _all_reduce_sum_tensor(metric_vec)
        mean_ep_return = float((metric_vec[0] / metric_vec[1]).item())
        mean_score = float((metric_vec[2] / metric_vec[3]).item())

        rollout_sec = float(t_rollout1 - t_rollout0)
        update_sec = float(t_update1 - t_update0)
        total_sec = float(t_update1 - t_rollout0)
        time_vec = torch.tensor([rollout_sec, update_sec, total_sec], device=device, dtype=torch.float64)
        _all_reduce_max_tensor(time_vec)
        rollout_sec = float(time_vec[0].item())
        update_sec = float(time_vec[1].item())
        total_sec = float(time_vec[2].item())
        global_steps = int(local_steps) * int(world_size)

        if is_main_process:
            print(
                f"upd={upd:04d}/{int(args.updates):04d} "
                f"ret={mean_ep_return:+.4f} score={mean_score:.1f} "
                f"loss={stats_sum.loss:.4f} (pi={stats_sum.policy_loss:.4f} v={stats_sum.value_loss:.4f} "
                f"opp_pi={stats_sum.aux_opp_move_loss:.4f} opp_th={stats_sum.aux_opp_param_loss:.4f} ent={stats_sum.entropy:.4f}) "
                f"kl={stats_sum.approx_kl:.4f} clip={stats_sum.clipfrac:.3f} "
                f"sps={(global_steps / total_sec) if total_sec > 0 else 0.0:.0f}"
            )

        if run is not None:
            wandb.log(
                {
                    "upd": int(upd),
                    "env_steps": int(env_steps),
                    "train/mean_return": float(mean_ep_return),
                    "train/mean_official_score": float(mean_score),
                    "loss/total": float(stats_sum.loss),
                    "loss/policy": float(stats_sum.policy_loss),
                    "loss/value": float(stats_sum.value_loss),
                    "loss/aux_opp_move": float(stats_sum.aux_opp_move_loss),
                    "loss/aux_opp_param": float(stats_sum.aux_opp_param_loss),
                    "loss/entropy": float(stats_sum.entropy),
                    "loss/approx_kl": float(stats_sum.approx_kl),
                    "loss/clipfrac": float(stats_sum.clipfrac),
                    "time/rollout_sec": float(rollout_sec),
                    "time/update_sec": float(update_sec),
                    "time/total_sec": float(total_sec),
                    "time/rollout_sps": float((global_steps / rollout_sec) if rollout_sec > 0 else 0.0),
                    "time/sps": float((global_steps / total_sec) if total_sec > 0 else 0.0),
                    "hp/lr": float(optimizer.param_groups[0]["lr"]),
                    "hp/clip_coef": float(args.clip_coef),
                    "hp/vf_clip_coef": float(args.vf_clip_coef),
                    "hp/ent_coef": float(args.ent_coef),
                    "hp/aux_opp_move_coef": float(args.aux_opp_move_coef),
                    "hp/aux_opp_param_coef": float(args.aux_opp_param_coef),
                },
                step=int(upd),
            )

        # checkpoint saving
        def do_save_snapshot() -> bool:
            return int(args.save_every) > 0 and (upd % int(args.save_every) == 0 or upd == int(args.updates))

        def do_save_last() -> bool:
            return int(args.save_last_every) > 0 and (upd % int(args.save_last_every) == 0 or upd == int(args.updates))

        if (do_save_snapshot() or do_save_last()) and is_main_process:
            wandb_run_id = str(run.id) if run is not None else resume_wandb_id
            model_spec = ModelSpec(
                arch_name=str(args.arch),
                feature_id=str(args.feature_id),
                in_channels=int(env.feature_channels),
                hidden=int(args.hidden),
                blocks=int(args.blocks),
                arch_kwargs=dict(model_arch_kwargs),
            )
            train_spec = TrainSpec(
                algo="ppo",
                seed=int(args.seed),
                batch_size=int(args.batch_size),
                t_max=int(env.spec.t_max),
                lr=float(args.lr),
                warmup_updates=int(args.warmup_updates),
                epochs=int(args.epochs),
                minibatch=int(args.minibatch),
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
                ent_coef=float(args.ent_coef),
                aux_opp_move_coef=float(args.aux_opp_move_coef),
                aux_opp_param_coef=float(args.aux_opp_param_coef),
                pf_enabled=bool(not args.no_pf),
                amp=bool(use_amp),
                rollout_amp=bool(rollout_amp),
                world_size=int(world_size),
            )

            ckpt_model = ckpt_model_dict(
                model_state=_unwrap_model(model).state_dict(),
                upd=int(upd),
                env_steps=int(env_steps),
                model=model_spec,
                train=train_spec,
                wandb_run_id=wandb_run_id,
                run_dir=str(paths.run_dir),
            )
            ckpt_full = ckpt_full_dict(
                model_state=_unwrap_model(model).state_dict(),
                optimizer_state=optimizer.state_dict(),
                rng_state=_get_rng_state(device),
                upd=int(upd),
                env_steps=int(env_steps),
                model=model_spec,
                train=train_spec,
                wandb_run_id=wandb_run_id,
                run_dir=str(paths.run_dir),
            )
            ema_entries_meta = [
                {"name": tr.name, "decay": float(tr.decay), "steps": int(tr.steps)}
                for tr in ema_trackers
            ]
            ckpt_model["ema"] = {"entries": ema_entries_meta}
            ckpt_full["ema"] = {"entries": ema_entries_meta}
            ckpt_full["ema_models"] = {
                tr.name: normalize_state_dict_keys(tr.model.state_dict())
                for tr in ema_trackers
            }
            write_json(paths.ckpt_ema_dir / "manifest.json", {"entries": ema_entries_meta})

            ckpt_ema_by_name: dict[str, dict[str, Any]] = {}
            if ema_trackers:
                for tr in ema_trackers:
                    ckpt_ema = ckpt_model_dict(
                        model_state=tr.model.state_dict(),
                        upd=int(upd),
                        env_steps=int(env_steps),
                        model=model_spec,
                        train=train_spec,
                        wandb_run_id=wandb_run_id,
                        run_dir=str(paths.run_dir),
                    )
                    ckpt_ema["ema"] = {
                        "name": str(tr.name),
                        "decay": float(tr.decay),
                        "steps": int(tr.steps),
                        "entries": ema_entries_meta,
                    }
                    ckpt_ema_by_name[tr.name] = ckpt_ema

            if do_save_snapshot():
                model_path = paths.ckpt_dir / f"ckpt_{upd:04d}.pt"
                full_path = paths.ckpt_full_dir / f"ckpt_{upd:04d}.pt"
                torch.save(ckpt_model, model_path)
                torch.save(ckpt_full, full_path)
                for tr in ema_trackers:
                    ema_dir = paths.ckpt_ema_dir / tr.name
                    ensure_dir(ema_dir)
                    ema_path = ema_dir / f"ckpt_{upd:04d}.pt"
                    torch.save(ckpt_ema_by_name[tr.name], ema_path)
                if args.wandb_log_checkpoints and (run is not None):
                    artifact = wandb.Artifact(name=f"ckpt_{upd:04d}", type="checkpoint")
                    artifact.add_file(str(model_path))
                    run.log_artifact(artifact)

            if do_save_last():
                model_last = paths.ckpt_dir / "ckpt_last.pt"
                full_last = paths.ckpt_full_dir / "ckpt_last.pt"
                torch.save(ckpt_model, model_last)
                torch.save(ckpt_full, full_last)
                for tr in ema_trackers:
                    ema_dir = paths.ckpt_ema_dir / tr.name
                    ensure_dir(ema_dir)
                    ema_last = ema_dir / "ckpt_last.pt"
                    torch.save(ckpt_ema_by_name[tr.name], ema_last)
        if is_distributed and (do_save_snapshot() or do_save_last()):
            dist.barrier()

        # periodic eval
        if (
            eval_env is not None
            and eval_paths is not None
            and int(args.eval_every) > 0
            and (upd % int(args.eval_every) == 0)
        ):
            _run_eval(eval_upd=int(upd), eval_env_steps=int(env_steps))
        if is_distributed and int(args.eval_every) > 0 and (upd % int(args.eval_every) == 0):
            dist.barrier()

    if run is not None:
        run.finish()
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
