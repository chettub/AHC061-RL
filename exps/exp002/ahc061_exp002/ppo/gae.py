from __future__ import annotations

import torch


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,  # [T, B]
    values: torch.Tensor,  # [T, B]
    dones: torch.Tensor,  # [T, B] (0/1)
    last_value: torch.Tensor,  # [B]
    gamma: float,
    gae_lambda: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_device = rewards.device if device is None else torch.device(device)
    rewards_f = rewards.to(device=target_device, dtype=torch.float32, non_blocking=True)
    values_f = values.to(device=target_device, dtype=torch.float32, non_blocking=True)
    dones_f = dones.to(device=target_device, dtype=torch.float32, non_blocking=True)
    last_value_f = last_value.to(device=target_device, dtype=torch.float32, non_blocking=True)

    t_max = rewards_f.size(0)
    adv = torch.zeros_like(rewards_f)
    last_gae = torch.zeros_like(last_value_f)

    for t in reversed(range(t_max)):
        next_nonterminal = 1.0 - dones_f[t]
        next_value = last_value_f if t == t_max - 1 else values_f[t + 1]
        delta = rewards_f[t] + gamma * next_value * next_nonterminal - values_f[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + values_f
    return adv, ret
