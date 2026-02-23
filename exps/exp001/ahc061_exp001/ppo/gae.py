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
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards_f = rewards.float()
    values_f = values.float()
    dones_f = dones.float()

    t_max = rewards_f.size(0)
    adv = torch.zeros_like(rewards_f)
    last_gae = torch.zeros_like(last_value.float())

    for t in reversed(range(t_max)):
        next_nonterminal = 1.0 - dones_f[t]
        next_value = last_value if t == t_max - 1 else values_f[t + 1]
        delta = rewards_f[t] + gamma * next_value * next_nonterminal - values_f[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + values_f
    return adv, ret

