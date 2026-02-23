from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..models import masked_logits


@dataclass
class Rollout:
    obs: torch.Tensor  # [T, B, C, 10, 10] (CPU)
    mask: torch.Tensor  # [T, B, 100] (CPU, uint8)
    actions: torch.Tensor  # [T, B] (CPU, int64)
    logp: torch.Tensor  # [T, B] (CPU, float32)
    values: torch.Tensor  # [T, B] (CPU, float32)
    rewards: torch.Tensor  # [T, B] (CPU, float32)
    dones: torch.Tensor  # [T, B] (CPU, uint8)
    last_value: torch.Tensor  # [B] (CPU, float32)


@torch.inference_mode()
def collect_rollout(
    env,
    model: nn.Module,
    device: torch.device,
    t_max: int,
    *,
    sample: bool,
    amp: bool = False,
) -> Rollout:
    bsz = env.batch_size
    c = env.feature_channels
    use_cuda = device.type == "cuda"
    use_amp = use_cuda and bool(amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    obs = torch.empty((t_max, bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((t_max, bsz, 100), dtype=torch.uint8, device="cpu")
    actions = torch.empty((t_max, bsz), dtype=torch.int64, device="cpu")
    logp = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    values = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    rewards = torch.empty((t_max, bsz), dtype=torch.float32, device="cpu")
    dones = torch.empty((t_max, bsz), dtype=torch.uint8, device="cpu")

    if use_cuda:
        board_dev = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device=device)
        mask_dev = torch.empty((bsz, 100), dtype=torch.uint8, device=device)
    next_obs = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device="cpu")
    next_mask = torch.empty((bsz, 100), dtype=torch.uint8, device="cpu")

    model.eval()

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
            logits, v = model(board)
        logits = masked_logits(logits.float(), m)
        v = v.float()
        log_probs = torch.log_softmax(logits, dim=1)
        if sample:
            probs = log_probs.exp()
            a = torch.multinomial(probs, 1).squeeze(1)
        else:
            a = torch.argmax(log_probs, dim=1)

        actions[t].copy_(a.to("cpu"))
        logp[t].copy_(log_probs.gather(1, a.view(-1, 1)).squeeze(1).to("cpu"))
        values[t].copy_(v.to("cpu"))

        if t + 1 < t_max:
            env.step_observe_into(actions[t], obs[t + 1], mask[t + 1], rewards[t], dones[t])
        else:
            env.step_observe_into(actions[t], next_obs, next_mask, rewards[t], dones[t])

    # bootstrap value
    if use_cuda:
        board_dev.copy_(next_obs)
        mask_dev.copy_(next_mask)
        board = board_dev
    else:
        board = next_obs
    with autocast:
        _, last_v = model(board)
    last_value = last_v.float().to("cpu")

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
