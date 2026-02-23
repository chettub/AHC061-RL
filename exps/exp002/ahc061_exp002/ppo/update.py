from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..models import masked_logits


@dataclass
class PPOStats:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    aux_opp_move_loss: torch.Tensor
    aux_opp_param_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: torch.Tensor


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,  # [N, C, 10, 10]
    mask: torch.Tensor,  # [N, 100]
    opp_move_dist: torch.Tensor | None,  # [N, 8, 100], float32
    opp_param: torch.Tensor | None,  # [N, 8, 5], float32 (w_norm[4], eps)
    opp_valid: torch.Tensor | None,  # [N, 8], uint8/bool
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
    aux_opp_move_coef: float = 0.0,
    aux_opp_param_coef: float = 0.0,
    max_grad_norm: float = 1.0,
    amp: bool | None = None,
) -> PPOStats:
    use_amp = obs.is_cuda if amp is None else bool(amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    with autocast:
        logits, values, opp_move_logits, opp_param_logits = model(obs)
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

    aux_opp_move_loss_t = torch.zeros((), dtype=torch.float32, device=obs.device)
    if aux_opp_move_coef > 0.0 and (opp_move_dist is not None) and (opp_valid is not None):
        valid = opp_valid != 0 if opp_valid.dtype != torch.bool else opp_valid
        valid_flat = valid.reshape(-1)
        if bool(valid_flat.any().item()):
            logits_valid = opp_move_logits.float().reshape(-1, 100)[valid_flat]
            dist_valid = opp_move_dist.float().reshape(-1, 100)[valid_flat]
            logp_opp = torch.log_softmax(logits_valid, dim=1)
            aux_opp_move_loss_t = -(dist_valid * logp_opp).sum(dim=1).mean()

    aux_opp_param_loss_t = torch.zeros((), dtype=torch.float32, device=obs.device)
    if aux_opp_param_coef > 0.0 and (opp_param is not None) and (opp_valid is not None):
        valid = opp_valid != 0 if opp_valid.dtype != torch.bool else opp_valid
        valid_flat = valid.reshape(-1)
        if bool(valid_flat.any().item()):
            pred = opp_param_logits.float().reshape(-1, 5)[valid_flat]
            tgt = opp_param.float().reshape(-1, 5)[valid_flat]
            pred_w = torch.softmax(pred[:, :4], dim=1)
            pred_eps = torch.sigmoid(pred[:, 4])
            tgt_w = tgt[:, :4]
            tgt_eps = tgt[:, 4]
            loss_w = (pred_w - tgt_w).pow(2).mean(dim=1)
            loss_eps = (pred_eps - tgt_eps).pow(2)
            aux_opp_param_loss_t = (loss_w + loss_eps).mean()

    loss = (
        policy_loss
        + vf_coef * value_loss
        - ent_coef * entropy
        + aux_opp_move_coef * aux_opp_move_loss_t
        + aux_opp_param_coef * aux_opp_param_loss_t
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return PPOStats(
        loss=loss.detach(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        aux_opp_move_loss=aux_opp_move_loss_t.detach(),
        aux_opp_param_loss=aux_opp_param_loss_t.detach(),
        entropy=entropy.detach(),
        approx_kl=approx_kl.detach(),
        clipfrac=clipfrac.detach(),
    )
