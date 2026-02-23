from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..models import masked_logits


@dataclass
class Rollout:
    obs: torch.Tensor  # [T, B, C, 10, 10] (CPU)
    mask: torch.Tensor  # [T, B, 100] (CPU, uint8)
    opp_move_dist: torch.Tensor | None  # [T, B, 8, 100] (CPU, float32), true target (new-order, p=0 masked)
    opp_param: torch.Tensor | None  # [T, B, 8, 5] (CPU, float32), true target (w_norm[4], eps)
    opp_valid: torch.Tensor | None  # [T, B, 8] (CPU, uint8), 0/1 valid mask (new-order, p=0 always 0)
    actions: torch.Tensor  # [T, B] (CPU, int64)
    logp: torch.Tensor  # [T, B] (CPU, float32)
    values: torch.Tensor  # [T, B] (CPU, float32)
    rewards: torch.Tensor  # [T, B] (CPU, float32)
    dones: torch.Tensor  # [T, B] (CPU, uint8)
    last_value: torch.Tensor  # [B] (CPU, float32)


@dataclass
class RolloutWorkspace:
    obs: torch.Tensor  # [T, B, C, 10, 10] (CPU)
    mask: torch.Tensor  # [T, B, 100] (CPU, uint8)
    opp_move_dist: torch.Tensor | None  # [T, B, 8, 100] (CPU, float32)
    opp_param: torch.Tensor | None  # [T, B, 8, 5] (CPU, float32)
    opp_valid: torch.Tensor | None  # [T, B, 8] (CPU, uint8)
    actions: torch.Tensor  # [T, B] (CPU, int64)
    logp: torch.Tensor  # [T, B] (CPU, float32)
    values: torch.Tensor  # [T, B] (CPU, float32)
    rewards: torch.Tensor  # [T, B] (CPU, float32)
    dones: torch.Tensor  # [T, B] (CPU, uint8)
    next_obs: torch.Tensor  # [B, C, 10, 10] (CPU)
    next_mask: torch.Tensor  # [B, 100] (CPU, uint8)
    last_value: torch.Tensor  # [B] (CPU, float32)
    board_dev: torch.Tensor | None = None
    mask_dev: torch.Tensor | None = None
    logp_dev: torch.Tensor | None = None
    values_dev: torch.Tensor | None = None


def _empty_cpu(shape, *, dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)


def create_rollout_workspace(
    env,
    *,
    t_max: int,
    device: torch.device,
    channels_last: bool = False,
    pin_memory: bool = False,
    collect_aux: bool = True,
) -> RolloutWorkspace:
    bsz = int(env.batch_size)
    c = int(env.feature_channels)
    m_max = int(env.spec.m_max)
    use_cuda = device.type == "cuda"
    use_channels_last = use_cuda and bool(channels_last)
    use_pinned = bool(pin_memory and use_cuda)

    def _alloc_cpu_tensors(pin: bool) -> tuple[torch.Tensor, ...]:
        base: list[torch.Tensor | None] = [
            _empty_cpu((t_max, bsz, c, 10, 10), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz, 100), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.int64, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((t_max, bsz), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((bsz, c, 10, 10), dtype=torch.float32, pin_memory=pin),
            _empty_cpu((bsz, 100), dtype=torch.uint8, pin_memory=pin),
            _empty_cpu((bsz,), dtype=torch.float32, pin_memory=pin),
        ]
        if collect_aux:
            base.extend(
                [
                    _empty_cpu((t_max, bsz, m_max, 100), dtype=torch.float32, pin_memory=pin),
                    _empty_cpu((t_max, bsz, m_max, 5), dtype=torch.float32, pin_memory=pin),
                    _empty_cpu((t_max, bsz, m_max), dtype=torch.uint8, pin_memory=pin),
                ]
            )
        else:
            base.extend([None, None, None])
        return tuple(base)

    if use_pinned:
        try:
            (
                obs,
                mask,
                actions,
                logp,
                values,
                rewards,
                dones,
                next_obs,
                next_mask,
                last_value,
                opp_move_dist,
                opp_param,
                opp_valid,
            ) = _alloc_cpu_tensors(True)
        except RuntimeError:
            # Large pinned buffers can fail on some systems. Fall back to pageable CPU memory.
            (
                obs,
                mask,
                actions,
                logp,
                values,
                rewards,
                dones,
                next_obs,
                next_mask,
                last_value,
                opp_move_dist,
                opp_param,
                opp_valid,
            ) = _alloc_cpu_tensors(False)
    else:
        (
            obs,
            mask,
            actions,
            logp,
            values,
            rewards,
            dones,
            next_obs,
            next_mask,
            last_value,
            opp_move_dist,
            opp_param,
            opp_valid,
        ) = _alloc_cpu_tensors(False)

    board_dev = None
    mask_dev = None
    logp_dev = None
    values_dev = None
    if use_cuda:
        if use_channels_last:
            board_dev = torch.empty(
                (bsz, c, 10, 10),
                dtype=torch.float32,
                device=device,
                memory_format=torch.channels_last,
            )
        else:
            board_dev = torch.empty((bsz, c, 10, 10), dtype=torch.float32, device=device)
        mask_dev = torch.empty((bsz, 100), dtype=torch.uint8, device=device)
        # Avoid per-step tiny D2H copies; copy [T, B] to CPU once after rollout.
        logp_dev = torch.empty((t_max, bsz), dtype=torch.float32, device=device)
        values_dev = torch.empty((t_max, bsz), dtype=torch.float32, device=device)

    return RolloutWorkspace(
        obs=obs,
        mask=mask,
        opp_move_dist=opp_move_dist,
        opp_param=opp_param,
        opp_valid=opp_valid,
        actions=actions,
        logp=logp,
        values=values,
        rewards=rewards,
        dones=dones,
        next_obs=next_obs,
        next_mask=next_mask,
        last_value=last_value,
        board_dev=board_dev,
        mask_dev=mask_dev,
        logp_dev=logp_dev,
        values_dev=values_dev,
    )


@torch.inference_mode()
def collect_rollout(
    env,
    model: nn.Module,
    device: torch.device,
    t_max: int,
    *,
    sample: bool,
    amp: bool = False,
    channels_last: bool = False,
    collect_aux: bool = True,
    fused_step_aux: bool = False,
    workspace: RolloutWorkspace | None = None,
) -> Rollout:
    bsz = env.batch_size
    c = env.feature_channels
    use_cuda = device.type == "cuda"
    use_amp = use_cuda and bool(amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
    if workspace is None:
        workspace = create_rollout_workspace(
            env,
            t_max=int(t_max),
            device=device,
            channels_last=bool(channels_last),
            pin_memory=False,
            collect_aux=bool(collect_aux),
        )
    obs = workspace.obs
    mask = workspace.mask
    actions = workspace.actions
    logp = workspace.logp
    values = workspace.values
    rewards = workspace.rewards
    dones = workspace.dones
    opp_move_dist = workspace.opp_move_dist
    opp_param = workspace.opp_param
    opp_valid = workspace.opp_valid
    next_obs = workspace.next_obs
    next_mask = workspace.next_mask

    model.eval()

    env.observe_into(obs[0], mask[0])
    if collect_aux:
        if opp_move_dist is None or opp_param is None or opp_valid is None:
            raise RuntimeError("collect_aux=True requires aux tensors in workspace")
        env.aux_targets_into(opp_move_dist[0], opp_param[0], opp_valid[0])

    for t in range(t_max):
        if use_cuda:
            board_dev = workspace.board_dev
            mask_dev = workspace.mask_dev
            if board_dev is None or mask_dev is None:
                raise RuntimeError("CUDA rollout requires board_dev/mask_dev in workspace")
            board_dev.copy_(obs[t], non_blocking=True)
            mask_dev.copy_(mask[t], non_blocking=True)
            board = board_dev
            m = mask_dev
        else:
            board = obs[t]
            m = mask[t]

        with autocast:
            logits, v, _, _ = model(board, with_aux=False)
        logits = masked_logits(logits.float(), m)
        v = v.float()
        if sample:
            if use_cuda:
                # On CUDA, multinomial(log_softmax) is typically faster than Gumbel-Max.
                log_probs = torch.log_softmax(logits, dim=1)
                a = torch.multinomial(log_probs.exp(), 1).squeeze(1)
                step_logp = log_probs.gather(1, a.view(-1, 1)).squeeze(1)
            else:
                # Exact categorical sampling via Gumbel-Max; usually faster on CPU.
                u = torch.rand_like(logits)
                gumbel = -torch.log(-torch.log(u))
                a = torch.argmax(logits + gumbel, dim=1)
                lse = torch.logsumexp(logits, dim=1)
                step_logp = logits.gather(1, a.view(-1, 1)).squeeze(1) - lse
        else:
            a = torch.argmax(logits, dim=1)
            lse = torch.logsumexp(logits, dim=1)
            step_logp = logits.gather(1, a.view(-1, 1)).squeeze(1) - lse

        # actions are consumed immediately by env.step_observe_into(), so keep this copy synchronous.
        actions[t].copy_(a, non_blocking=False)
        if use_cuda:
            logp_dev = workspace.logp_dev
            values_dev = workspace.values_dev
            if logp_dev is None or values_dev is None:
                raise RuntimeError("CUDA rollout requires logp_dev/values_dev in workspace")
            logp_dev[t].copy_(step_logp, non_blocking=True)
            values_dev[t].copy_(v, non_blocking=True)
        else:
            logp[t].copy_(step_logp, non_blocking=True)
            values[t].copy_(v, non_blocking=True)

        if t + 1 < t_max:
            if collect_aux:
                if opp_move_dist is None or opp_param is None or opp_valid is None:
                    raise RuntimeError("collect_aux=True requires aux tensors in workspace")
                if fused_step_aux:
                    env.step_observe_aux_into(
                        actions[t],
                        obs[t + 1],
                        mask[t + 1],
                        rewards[t],
                        dones[t],
                        opp_move_dist[t + 1],
                        opp_param[t + 1],
                        opp_valid[t + 1],
                    )
                else:
                    env.step_observe_into(actions[t], obs[t + 1], mask[t + 1], rewards[t], dones[t])
                    env.aux_targets_into(opp_move_dist[t + 1], opp_param[t + 1], opp_valid[t + 1])
            else:
                env.step_observe_into(actions[t], obs[t + 1], mask[t + 1], rewards[t], dones[t])
        else:
            env.step_observe_into(actions[t], next_obs, next_mask, rewards[t], dones[t])

    if use_cuda:
        logp_dev = workspace.logp_dev
        values_dev = workspace.values_dev
        if logp_dev is None or values_dev is None:
            raise RuntimeError("CUDA rollout requires logp_dev/values_dev in workspace")
        logp.copy_(logp_dev, non_blocking=True)
        values.copy_(values_dev, non_blocking=True)

    # bootstrap value
    if use_cuda:
        board_dev = workspace.board_dev
        mask_dev = workspace.mask_dev
        if board_dev is None or mask_dev is None:
            raise RuntimeError("CUDA rollout requires board_dev/mask_dev in workspace")
        board_dev.copy_(next_obs, non_blocking=True)
        mask_dev.copy_(next_mask, non_blocking=True)
        board = board_dev
    else:
        board = next_obs
    with autocast:
        _, last_v, _, _ = model(board, with_aux=False)
    last_value = workspace.last_value
    last_value.copy_(last_v.float(), non_blocking=True)
    if use_cuda:
        # D2H copies above are non_blocking; ensure CPU tensors are fully ready before return.
        torch.cuda.current_stream(device).synchronize()

    return Rollout(
        obs=obs,
        mask=mask,
        opp_move_dist=opp_move_dist,
        opp_param=opp_param,
        opp_valid=opp_valid,
        actions=actions,
        logp=logp,
        values=values,
        rewards=rewards,
        dones=dones,
        last_value=last_value,
    )
