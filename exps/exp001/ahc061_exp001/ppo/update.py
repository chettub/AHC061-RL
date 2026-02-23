from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..models import masked_logits


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float


def ppo_update(
    model: nn.Module,
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
    amp: bool | None = None,
) -> PPOStats:
    use_amp = obs.is_cuda if amp is None else bool(amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    with autocast:
        logits, values = model(obs)
    logits = masked_logits(logits.float(), mask)
    values = values.float()
    log_probs = torch.log_softmax(logits, dim=1)
    new_logp = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
    probs = log_probs.exp()
    ent_elem = torch.where(probs > 0.0, probs * log_probs, torch.zeros_like(probs))
    entropy = (-ent_elem.sum(dim=1)).mean()

    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()

    with torch.no_grad():
        approx_kl = (old_logp - new_logp).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    if vf_clip_coef > 0.0:
        v_clipped = old_values + torch.clamp(values - old_values, -vf_clip_coef, vf_clip_coef)
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

    return PPOStats(
        loss=float(loss.item()),
        policy_loss=float(policy_loss.item()),
        value_loss=float(value_loss.item()),
        entropy=float(entropy.item()),
        approx_kl=float(approx_kl.item()),
        clipfrac=float(clipfrac.item()),
    )
