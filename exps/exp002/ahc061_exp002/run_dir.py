from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ckpt import ensure_dir, write_json


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_dir: Path
    ckpt_full_dir: Path
    ckpt_ema_dir: Path
    wandb_dir: Path

    @staticmethod
    def from_run_dir(run_dir: Path) -> "RunPaths":
        return RunPaths(
            run_dir=run_dir,
            ckpt_dir=run_dir / "checkpoints",
            ckpt_full_dir=run_dir / "checkpoints_full",
            ckpt_ema_dir=run_dir / "checkpoints_ema",
            wandb_dir=run_dir / "wandb",
        )


def default_run_dir(*, exp_dir: Path, name: str | None) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}"
    if name:
        stem += f"_{name}"
    return exp_dir / "artifacts" / "runs" / stem


def init_run_dir(paths: RunPaths, *, config: dict[str, Any]) -> None:
    ensure_dir(paths.run_dir)
    ensure_dir(paths.ckpt_dir)
    ensure_dir(paths.ckpt_full_dir)
    ensure_dir(paths.ckpt_ema_dir)
    ensure_dir(paths.wandb_dir)
    write_json(paths.run_dir / "config.json", config)
