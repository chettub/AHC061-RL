from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from ..env import BatchEnv, tools_input_paths
from ..models import build_policy_value_model, masked_logits, normalize_state_dict_keys

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


@torch.no_grad()
def run_episode_greedy(env: BatchEnv, model: torch.nn.Module, device: torch.device) -> None:
    model.eval()
    board = torch.empty((env.batch_size, env.feature_channels, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((env.batch_size, 100), dtype=torch.uint8, device="cpu")

    for _ in range(env.spec.t_max):
        env.observe_into(board, mask)
        logits, _ = model(board.to(device))
        logits = masked_logits(logits, mask.to(device))
        actions = torch.argmax(logits, dim=1).to("cpu")
        env.step(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ahc061-exp001")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
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

    ckpt_path = Path(args.ckpt)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    env = BatchEnv(batch_size=args.batch, pf_enabled=not args.no_pf, verbose_build=False)
    arch_name = str(ckpt.get("arch_name", "resnet_v1"))
    model = build_policy_value_model(
        arch_name,
        in_channels=int(ckpt["in_channels"]),
        hidden_channels=int(ckpt["hidden"]),
        blocks=int(ckpt["blocks"]),
    ).to(device)
    model.load_state_dict(normalize_state_dict_keys(ckpt["model"]))

    paths = tools_input_paths(args.seed_begin, args.seed_end)
    scores: list[float] = []

    if wandb is None:
        raise RuntimeError("wandb is not available. Install it to run evaluation.")

    exp_dir = Path(__file__).resolve().parents[2]
    wandb_dir = exp_dir / "artifacts" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=tags if tags else None,
        mode="online",
        dir=str(wandb_dir),
        config={
            "job": "eval_tools",
            "ckpt": args.ckpt,
            "arch_name": arch_name,
            "seed_begin": args.seed_begin,
            "seed_end": args.seed_end,
            "batch": args.batch,
            "pf_enabled": not args.no_pf,
            "device": str(device),
            "torch_version": torch.__version__,
        },
    )

    for i in range(0, len(paths), args.batch):
        chunk = paths[i : i + args.batch]
        if len(chunk) < args.batch:
            # pad with last to fill the batch, then discard extras
            chunk = chunk + [chunk[-1]] * (args.batch - len(chunk))
            valid = len(paths) - i
        else:
            valid = args.batch

        env.reset_from_tools(chunk)
        run_episode_greedy(env, model, device)
        sc = env.official_score().float()[:valid]
        scores.extend(sc.tolist())

    mean_score = sum(scores) / max(1, len(scores))
    print(f"tools seeds {args.seed_begin}-{args.seed_end}: mean_score={mean_score:.3f} n={len(scores)}")
    wandb.log(
        {
            "eval/mean_score": mean_score,
            "eval/n": len(scores),
            "eval/score_hist": wandb.Histogram(scores),
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
