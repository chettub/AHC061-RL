from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_state_dict_keys(state_dict: dict) -> dict:
    prefixes = ("_orig_mod.", "module.")
    changed = False
    out: dict = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if isinstance(nk, str) and nk.startswith(p):
                nk = nk[len(p) :]
                changed = True
        if nk in out:
            raise RuntimeError(f"duplicate key after normalization: {nk!r}")
        out[nk] = v
    return out if changed else state_dict


def _pick_gn_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        g = _pick_gn_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.silu(y)
        y = self.conv2(y)
        y = self.gn2(y)
        return F.silu(x + y)


class DWResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        g = _pick_gn_groups(channels)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(g, channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.gn1(y)
        y = F.silu(y)
        y = self.pw(y)
        y = self.gn2(y)
        return F.silu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__()
        g = _pick_gn_groups(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(blocks)])

        self.policy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, bias=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # board: [B, C, 10, 10]
        x = self.stem(board)
        x = self.blocks(x)

        logits = self.policy_head(x).flatten(1)  # [B, 100]
        pooled = x.mean(dim=(2, 3))  # [B, H]
        value = self.value_head(pooled).squeeze(1)  # [B]
        return logits, value


class PolicyValueDWNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__()
        g = _pick_gn_groups(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[DWResidualBlock(hidden_channels) for _ in range(blocks)])

        self.policy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, bias=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # board: [B, C, 10, 10]
        x = self.stem(board)
        x = self.blocks(x)

        logits = self.policy_head(x).flatten(1)  # [B, 100]
        pooled = x.mean(dim=(2, 3))  # [B, H]
        value = self.value_head(pooled).squeeze(1)  # [B]
        return logits, value


def build_policy_value_model(
    arch_name: str,
    *,
    in_channels: int,
    hidden_channels: int,
    blocks: int,
) -> nn.Module:
    name = str(arch_name).lower()
    if name in ("resnet", "resnet_v1", "policyvaluenet"):
        return PolicyValueNet(in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks)
    if name in ("dwres", "dwres_v1", "policyvaluedwnet"):
        return PolicyValueDWNet(in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks)
    raise ValueError(f"unknown arch_name: {arch_name!r}")


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: uint8/bool/int (0/1), shape [B, 100]
    if mask.dtype != torch.bool:
        mask_bool = mask != 0
    else:
        mask_bool = mask
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask_bool, neg_inf)
