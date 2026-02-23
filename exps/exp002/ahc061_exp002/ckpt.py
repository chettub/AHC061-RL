from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


CKPT_VERSION = 1


def _torch_load(path: Path, *, weights_only: bool) -> dict[str, Any]:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def torch_load_maybe_weights_only(path: Path) -> dict[str, Any]:
    try:
        return _torch_load(path, weights_only=True)
    except pickle.UnpicklingError:
        return _torch_load(path, weights_only=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("_orig_mod.", "module.")
    # Backward compatibility for legacy key names.
    rename_prefixes = (
        ("player_front.", "player_front_blocks."),
    )
    changed = False
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if isinstance(nk, str) and nk.startswith(p):
                nk = nk[len(p) :]
                changed = True
        if isinstance(nk, str):
            for old, new in rename_prefixes:
                if nk.startswith(old):
                    nk = new + nk[len(old) :]
                    changed = True
                    break
        if nk in out:
            raise RuntimeError(f"duplicate key after normalization: {nk!r}")
        out[nk] = v
    return out if changed else state_dict


@dataclass(frozen=True)
class ModelSpec:
    arch_name: str
    feature_id: str
    in_channels: int
    hidden: int
    blocks: int
    arch_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainSpec:
    algo: str
    seed: int
    batch_size: int
    t_max: int
    lr: float
    warmup_updates: int
    epochs: int
    minibatch: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    aux_opp_move_coef: float
    aux_opp_param_coef: float
    pf_enabled: bool
    amp: bool
    rollout_amp: bool
    world_size: int = 1


@dataclass(frozen=True)
class CkptMeta:
    ckpt_version: int
    upd: int
    env_steps: int
    model: ModelSpec
    train: TrainSpec
    wandb_run_id: str | None
    run_dir: str | None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ckpt_model_dict(
    *,
    model_state: dict[str, Any],
    upd: int,
    env_steps: int,
    model: ModelSpec,
    train: TrainSpec | None = None,
    wandb_run_id: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ckpt_version": int(CKPT_VERSION),
        "kind": "model",
        "upd": int(upd),
        "env_steps": int(env_steps),
        "model": normalize_state_dict_keys(model_state),
        "model_spec": asdict(model),
        "wandb": {"run_id": str(wandb_run_id) if wandb_run_id is not None else None},
        "run_dir": str(run_dir) if run_dir is not None else None,
        "torch_version": str(torch.__version__),
    }
    if train is not None:
        out["train_spec"] = asdict(train)
    return out


def ckpt_full_dict(
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    rng_state: dict[str, Any],
    upd: int,
    env_steps: int,
    model: ModelSpec,
    train: TrainSpec,
    wandb_run_id: str | None,
    run_dir: str | None,
) -> dict[str, Any]:
    return {
        "ckpt_version": int(CKPT_VERSION),
        "kind": "full",
        "upd": int(upd),
        "env_steps": int(env_steps),
        "model": normalize_state_dict_keys(model_state),
        "optimizer": optimizer_state,
        "rng": rng_state,
        "model_spec": asdict(model),
        "train_spec": asdict(train),
        "wandb": {"run_id": str(wandb_run_id) if wandb_run_id is not None else None},
        "run_dir": str(run_dir) if run_dir is not None else None,
        "torch_version": str(torch.__version__),
    }


def is_full_ckpt(ckpt: dict[str, Any]) -> bool:
    return bool(ckpt.get("kind") == "full") or ("optimizer" in ckpt and "rng" in ckpt)


def model_spec_from_ckpt(ckpt: dict[str, Any]) -> ModelSpec:
    def _normalize_arch_kwargs(v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise RuntimeError(f"invalid model_spec.arch_kwargs type: {type(v).__name__}")
        return {str(k): v[k] for k in v.keys()}

    if "model_spec" in ckpt:
        ms = ckpt["model_spec"]
        return ModelSpec(
            arch_name=str(ms["arch_name"]),
            feature_id=str(ms.get("feature_id", "submit_v1")),
            in_channels=int(ms["in_channels"]),
            hidden=int(ms["hidden"]),
            blocks=int(ms["blocks"]),
            arch_kwargs=_normalize_arch_kwargs(ms.get("arch_kwargs", {})),
        )

    # Backward compatibility (exp001 / older)
    return ModelSpec(
        arch_name=str(ckpt.get("arch_name", "resnet_v1")),
        feature_id=str(ckpt.get("feature_id", "submit_v1")),
        in_channels=int(ckpt["in_channels"]),
        hidden=int(ckpt["hidden"]),
        blocks=int(ckpt["blocks"]),
        arch_kwargs=_normalize_arch_kwargs(ckpt.get("arch_kwargs", {})),
    )


def train_spec_from_ckpt(ckpt: dict[str, Any]) -> TrainSpec | None:
    ts = ckpt.get("train_spec") or ckpt.get("train")
    if ts is None:
        return None
    return TrainSpec(
        algo=str(ts.get("algo", "ppo")),
        seed=int(ts["seed"]),
        batch_size=int(ts["batch_size"]),
        t_max=int(ts["t_max"]),
        lr=float(ts["lr"]),
        warmup_updates=int(ts.get("warmup_updates", 0)),
        epochs=int(ts["epochs"]),
        minibatch=int(ts["minibatch"]),
        gamma=float(ts["gamma"]),
        gae_lambda=float(ts["gae_lambda"]),
        ent_coef=float(ts.get("ent_coef", 0.01)),
        aux_opp_move_coef=float(ts.get("aux_opp_move_coef", 0.0)),
        aux_opp_param_coef=float(ts.get("aux_opp_param_coef", 0.0)),
        pf_enabled=not bool(ts.get("no_pf", False)) if "no_pf" in ts else bool(ts["pf_enabled"]),
        amp=bool(ts.get("amp", not bool(ts.get("no_amp", False)))),
        rollout_amp=bool(ts.get("rollout_amp", bool(ts.get("amp", not bool(ts.get("no_amp", False)))))),
        world_size=int(ts.get("world_size", 1)),
    )


def wandb_run_id_from_ckpt(ckpt: dict[str, Any]) -> str | None:
    w = ckpt.get("wandb", {})
    rid = w.get("run_id")
    return str(rid) if rid is not None else None
