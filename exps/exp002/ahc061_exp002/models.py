from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


M_MAX = 8


def _pick_gn_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def _coord_grid(device: torch.device, dtype: torch.dtype, height: int, width: int) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    gy = yy.view(1, 1, height, 1).expand(1, 1, height, width)
    gx = xx.view(1, 1, 1, width).expand(1, 1, height, width)
    return torch.cat((gx, gy), dim=1)


class CoordEmbed2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        coord = _coord_grid(x.device, x.dtype, h, w).expand(bsz, -1, -1, -1)
        return torch.cat((x, coord), dim=1)


class GlobalFiLM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, channels)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        h = F.silu(self.fc1(pooled))
        gamma_beta = self.fc2(h)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, *, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        gate = F.silu(self.fc1(pooled))
        gate = torch.sigmoid(self.fc2(gate))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


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


class Full3x3PWResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        g = _pick_gn_groups(channels)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3x3(x)
        y = self.gn1(y)
        y = F.silu(y)
        y = self.pw(y)
        y = self.gn2(y)
        return F.silu(x + y)


class PlayerSetAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        ff_mult: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if heads <= 0:
            raise ValueError(f"heads must be >= 1, got {int(heads)}")
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        if ff_mult <= 0.0:
            raise ValueError(f"ff_mult must be > 0, got {float(ff_mult)}")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {float(dropout)}")

        ff_hidden = max(channels, int(round(float(channels) * float(ff_mult))))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop1 = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_hidden, channels),
        )
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor, present_mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: [B, P, C], present_mask: [B, P] bool
        key_padding_mask = None if present_mask is None else (~present_mask)
        y = self.norm1(tokens)
        # NOTE:
        # `need_weights=False` can route to efficient SDPA kernels that currently fail
        # under torch.compile backward with key_padding_mask ("last dimension must be contiguous").
        # We keep the same module weights and force a safe path by requesting attn weights.
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=True)
        tokens = tokens + self.drop1(y)
        z = self.ff(self.norm2(tokens))
        tokens = tokens + self.drop2(z)
        if present_mask is not None:
            tokens = tokens * present_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return tokens


class MainEnemyCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        query_channels: int,
        kv_channels: int,
        *,
        heads: int = 4,
        ff_mult: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if heads <= 0:
            raise ValueError(f"heads must be >= 1, got {int(heads)}")
        if query_channels % heads != 0:
            raise ValueError(f"query_channels ({query_channels}) must be divisible by heads ({heads})")
        if ff_mult <= 0.0:
            raise ValueError(f"ff_mult must be > 0, got {float(ff_mult)}")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {float(dropout)}")

        ff_hidden = max(query_channels, int(round(float(query_channels) * float(ff_mult))))
        self.q_norm = nn.LayerNorm(query_channels)
        self.kv_norm = nn.LayerNorm(kv_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_channels,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
            kdim=kv_channels,
            vdim=kv_channels,
        )
        self.drop1 = nn.Dropout(float(dropout))
        self.ff_norm = nn.LayerNorm(query_channels)
        self.ff = nn.Sequential(
            nn.Linear(query_channels, ff_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_hidden, query_channels),
        )
        self.drop2 = nn.Dropout(float(dropout))

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        kv_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q_tokens: [B, Nq, Cq], kv_tokens: [B, Nk, Ckv], kv_valid_mask: [B, Nk] bool
        key_padding_mask = None if kv_valid_mask is None else (~kv_valid_mask)
        q = self.q_norm(q_tokens)
        kv = self.kv_norm(kv_tokens)
        # Keep this on a safe path for torch.compile + mask backward stability.
        y, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=True)
        q_tokens = q_tokens + self.drop1(y)
        z = self.ff(self.ff_norm(q_tokens))
        q_tokens = q_tokens + self.drop2(z)
        return q_tokens


class PixelPlayerSelfAttentionDWBlock(nn.Module):
    def __init__(self, channels: int, *, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.o_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

        self.dw_block = DWResidualBlock(channels)

    def forward(self, x: torch.Tensor, player_present: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, P, C, H, W], attention is only across players at each pixel.
        bsz, pnum, channels, h, w = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(bsz * h * w, pnum, channels)  # [BHW, P, C]

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        q = q.view(bsz * h * w, pnum, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [BHW, heads, P, Dh]
        k = k.view(bsz * h * w, pnum, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz * h * w, pnum, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)  # [BHW, heads, P, P]
        if player_present is not None:
            # Mask key-side inactive players; inactive outputs are zeroed after DW block.
            key_mask = player_present.view(bsz, 1, 1, pnum).expand(bsz, h * w, 1, pnum).reshape(
                bsz * h * w, 1, 1, pnum
            )
            neg = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~key_mask, neg)
        attn = torch.softmax(attn_logits, dim=-1)

        attn_out = torch.matmul(attn, v)  # [BHW, heads, P, Dh]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(bsz * h * w, pnum, channels)
        tokens = self.norm(tokens + self.o_proj(attn_out))

        x = tokens.reshape(bsz, h, w, pnum, channels).permute(0, 3, 4, 1, 2).contiguous()
        x = self.dw_block(x.reshape(bsz * pnum, channels, h, w)).reshape(bsz, pnum, channels, h, w)
        if player_present is not None:
            x = x * player_present.view(bsz, pnum, 1, 1, 1).to(dtype=x.dtype)
        return x


class PlayerAxisMixResidualBlock(nn.Module):
    def __init__(
        self,
        player_count: int,
        channels: int,
        *,
        use_channel_gate: bool = True,
        alpha_scale: float = 0.2,
    ):
        super().__init__()
        if int(player_count) <= 0:
            raise ValueError(f"player_count must be >= 1, got {int(player_count)}")
        if int(channels) <= 0:
            raise ValueError(f"channels must be >= 1, got {int(channels)}")
        if float(alpha_scale) <= 0.0:
            raise ValueError(f"alpha_scale must be > 0, got {float(alpha_scale)}")

        self.player_count = int(player_count)
        self.channels = int(channels)
        self.use_channel_gate = bool(use_channel_gate)
        self.alpha_scale = float(alpha_scale)

        # Shared player-axis mixing (no bias) for stability and parameter efficiency.
        self.mix1 = nn.Parameter(torch.eye(self.player_count, dtype=torch.float32))
        self.mix2 = nn.Parameter(torch.eye(self.player_count, dtype=torch.float32))
        # Start from exact identity residual; gradients can grow interaction gradually.
        self.alpha_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))
        if self.use_channel_gate:
            self.gamma_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        else:
            self.register_buffer("_dummy_gamma", torch.empty(0), persistent=False)

    @staticmethod
    def _normalize_rowwise_abs(w: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        den = w.abs().sum(dim=-1, keepdim=True).clamp_min(float(eps))
        return w / den

    def _masked_weight(self, w: torch.Tensor, present_f: torch.Tensor) -> torch.Tensor:
        # w: [P, P], present_f: [B, P] in {0,1}
        w_eff = w.unsqueeze(0) * present_f.unsqueeze(1)  # mask key-side (source players)
        return self._normalize_rowwise_abs(w_eff)

    def forward(self, x: torch.Tensor, present_mask: torch.Tensor | None) -> torch.Tensor:
        # x: [B, P, C, H, W], present_mask: [B, P] bool
        bsz, pnum, channels, _, _ = x.shape
        if pnum != self.player_count:
            raise RuntimeError(f"player axis mismatch: expected {self.player_count}, got {pnum}")
        if channels != self.channels:
            raise RuntimeError(f"channel mismatch: expected {self.channels}, got {channels}")

        if present_mask is None:
            present_f = torch.ones((bsz, pnum), dtype=x.dtype, device=x.device)
        else:
            present_f = present_mask.to(dtype=x.dtype)
        present_map = present_f.view(bsz, pnum, 1, 1, 1)
        x_masked = x * present_map

        w1 = self._masked_weight(self.mix1.to(dtype=x.dtype), present_f)
        h = torch.einsum("bqk,bkchw->bqchw", w1, x_masked)
        h = F.silu(h)
        w2 = self._masked_weight(self.mix2.to(dtype=x.dtype), present_f)
        y = torch.einsum("bqk,bkchw->bqchw", w2, h)
        if self.use_channel_gate:
            gamma = (1.0 + torch.tanh(self.gamma_raw.to(dtype=x.dtype))).view(1, 1, channels, 1, 1)
            y = y * gamma

        alpha = self.alpha_scale * torch.tanh(self.alpha_raw.to(dtype=x.dtype))
        out = x_masked + alpha * y
        out = out * present_map
        return out


class MBConvSEBlock(nn.Module):
    def __init__(self, channels: int, *, expand_ratio: int = 2, se_ratio: float = 0.25):
        super().__init__()
        mid_channels = max(channels, channels * int(expand_ratio))
        g_mid = _pick_gn_groups(mid_channels)
        g_out = _pick_gn_groups(channels)

        self.expand = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.gn_expand = nn.GroupNorm(g_mid, mid_channels)
        self.dw = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.gn_dw = nn.GroupNorm(g_mid, mid_channels)
        self.se = SEBlock(mid_channels, se_ratio=se_ratio)
        self.project = nn.Conv2d(mid_channels, channels, kernel_size=1, padding=0, bias=False)
        self.gn_project = nn.GroupNorm(g_out, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.gn_expand(y)
        y = F.silu(y)
        y = self.dw(y)
        y = self.gn_dw(y)
        y = F.silu(y)
        y = self.se(y)
        y = self.project(y)
        y = self.gn_project(y)
        return F.silu(x + y)


class PolicyValueBase(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.policy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, bias=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.opp_move_head = nn.Conv2d(hidden_channels, M_MAX, kernel_size=1, padding=0, bias=True)
        self.opp_param_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, M_MAX * 5),
        )
        self.register_buffer("_dummy_aux0", torch.empty(0), persistent=False)

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _forward_policy_only(self, board: torch.Tensor) -> torch.Tensor:
        x_policy, _ = self._forward_features(board)
        return x_policy

    def forward_policy(self, board: torch.Tensor) -> torch.Tensor:
        x_policy = self._forward_policy_only(board)
        return self.policy_head(x_policy).flatten(1)

    def forward(
        self,
        board: torch.Tensor,
        *,
        with_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_policy, x_value = self._forward_features(board)
        logits = self.policy_head(x_policy).flatten(1)  # [B, 100]
        pooled = x_value.mean(dim=(2, 3))  # [B, H]
        value = self.value_head(pooled).squeeze(1)  # [B]
        if with_aux:
            opp_move_logits = self.opp_move_head(x_value).flatten(2)  # [B, M_MAX, 100]
            opp_param = self.opp_param_head(pooled).view(-1, M_MAX, 5)  # [B, M_MAX, 5]
        else:
            opp_move_logits = self._dummy_aux0
            opp_param = self._dummy_aux0
        return logits, value, opp_move_logits, opp_param


class PolicyValueNet(PolicyValueBase):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__(hidden_channels=hidden_channels)
        g = _pick_gn_groups(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(blocks)])

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(board)
        x = self.blocks(x)
        return x, x


class PolicyValueDWNet(PolicyValueBase):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__(hidden_channels=hidden_channels)
        g = _pick_gn_groups(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[DWResidualBlock(hidden_channels) for _ in range(blocks)])

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(board)
        x = self.blocks(x)
        return x, x


class PolicyValueDWPlayerConcatNet(PolicyValueBase):
    # research_v4 fixed layout:
    #   global  : 19ch [0:19]
    #   player  : 128ch [19:147] as (8 players x 16ch)
    #   pos0    : 2ch  [147:149]
    R4_GLOBAL_C = 19
    R4_PLAYER_PER_C = 16
    R4_PLAYER_BLOCK_C = M_MAX * R4_PLAYER_PER_C
    R4_TOTAL_C = R4_GLOBAL_C + R4_PLAYER_BLOCK_C + 2
    R4_REACH_OFFSET = 2  # owner, comp, reach, ...
    R4_M2_ONEHOT_OFFSET = 7  # global slice: m_onehot starts at ch7, values encode m=2..8
    PLAYER_BRANCH_N = M_MAX - 1  # process only p=1..7
    PLAYER_COMMON_INPUT_C = R4_GLOBAL_C + R4_PLAYER_PER_C
    PLAYER_ENEMY_FEAT_INPUT_C = R4_PLAYER_PER_C  # player p only

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        blocks: int = 6,
        *,
        feature_id: str | None = None,
        player_hidden_channels: int | None = None,
        use_full3x3_blocks: bool = False,
        player_set_layers: int = 0,
        player_set_heads: int = 4,
        player_set_ff_mult: float = 2.0,
        player_set_dropout: float = 0.0,
        player_set_every: int = 1,
        player_mix_layers: int = 0,
        player_mix_every: int = 1,
        player_mix_channel_gate: bool = False,
        player_mix_alpha_scale: float = 0.2,
    ):
        super().__init__(hidden_channels=hidden_channels)
        if in_channels != self.R4_TOTAL_C:
            raise ValueError(
                f"dwres_ppconcat_v1 requires in_channels={self.R4_TOTAL_C} (research_v4), got {int(in_channels)}"
            )
        if feature_id is not None and str(feature_id) != "research_v4":
            raise ValueError(f"dwres_ppconcat_v1 requires feature_id='research_v4', got {feature_id!r}")
        if blocks <= 0:
            raise ValueError(f"blocks must be >= 1, got {int(blocks)}")
        # Default: player branch width is half of main width for better throughput.
        player_hidden = int(max(1, hidden_channels // 2) if player_hidden_channels is None else player_hidden_channels)
        if player_hidden <= 0:
            raise ValueError(f"player_hidden_channels must be >= 1, got {player_hidden}")
        if int(player_set_layers) < 0:
            raise ValueError(f"player_set_layers must be >= 0, got {int(player_set_layers)}")
        if int(player_set_heads) <= 0:
            raise ValueError(f"player_set_heads must be >= 1, got {int(player_set_heads)}")
        if float(player_set_ff_mult) <= 0.0:
            raise ValueError(f"player_set_ff_mult must be > 0, got {float(player_set_ff_mult)}")
        if not (0.0 <= float(player_set_dropout) < 1.0):
            raise ValueError(f"player_set_dropout must be in [0,1), got {float(player_set_dropout)}")
        if int(player_set_every) <= 0:
            raise ValueError(f"player_set_every must be >= 1, got {int(player_set_every)}")
        if int(player_mix_layers) < 0:
            raise ValueError(f"player_mix_layers must be >= 0, got {int(player_mix_layers)}")
        if int(player_mix_every) <= 0:
            raise ValueError(f"player_mix_every must be >= 1, got {int(player_mix_every)}")
        if float(player_mix_alpha_scale) <= 0.0:
            raise ValueError(f"player_mix_alpha_scale must be > 0, got {float(player_mix_alpha_scale)}")
        self.player_hidden_channels = player_hidden
        self.player_set_layers = int(player_set_layers)
        self.player_set_every = int(player_set_every)
        self.player_mix_layers = int(player_mix_layers)
        self.player_mix_every = int(player_mix_every)

        g = _pick_gn_groups(hidden_channels)
        g_player = _pick_gn_groups(player_hidden)
        front_blocks = max(1, int(blocks) // 2)
        back_blocks = max(0, int(blocks) - front_blocks)
        block_cls: type[nn.Module]
        if use_full3x3_blocks:
            block_cls = Full3x3PWResidualBlock
        else:
            block_cls = DWResidualBlock

        # network1 (main): stem + residual front/back blocks
        self.main_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.main_front = nn.Sequential(*[block_cls(hidden_channels) for _ in range(front_blocks)])
        self.main_back = nn.Sequential(*[block_cls(hidden_channels) for _ in range(back_blocks)])

        # network2 (shared across players): per-player independent processing
        # Split stem into common (global+p0) and enemy-specific (id+p) projections.
        # This avoids recomputing common projection for each active enemy.
        self.player_stem_common = nn.Conv2d(self.PLAYER_COMMON_INPUT_C, player_hidden, kernel_size=1, padding=0, bias=False)
        self.player_stem_enemy_feat = nn.Conv2d(self.PLAYER_ENEMY_FEAT_INPUT_C, player_hidden, kernel_size=1, padding=0, bias=False)
        self.player_stem_enemy_id = nn.Embedding(self.PLAYER_BRANCH_N, player_hidden)
        self.player_stem_norm = nn.GroupNorm(g_player, player_hidden)
        self.player_stem_act = nn.SiLU()
        self.player_front_blocks = nn.ModuleList([block_cls(player_hidden) for _ in range(front_blocks)])
        self.player_set_blocks = nn.ModuleList()
        self.player_set_token_to_film = nn.ModuleList()
        self.player_mix_blocks = nn.ModuleList()
        for _ in range(self.player_set_layers):
            self.player_set_blocks.append(
                PlayerSetAttentionBlock(
                    player_hidden,
                    heads=int(player_set_heads),
                    ff_mult=float(player_set_ff_mult),
                    dropout=float(player_set_dropout),
                )
            )
            self.player_set_token_to_film.append(nn.Linear(player_hidden, player_hidden * 2))
        for _ in range(self.player_mix_layers):
            self.player_mix_blocks.append(
                PlayerAxisMixResidualBlock(
                    self.PLAYER_BRANCH_N,
                    player_hidden,
                    use_channel_gate=bool(player_mix_channel_gate),
                    alpha_scale=float(player_mix_alpha_scale),
                )
            )

        # merge: concat(main + all players) -> fuse -> residual add to main path
        self.merge_fuse_main = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=False)
        self.merge_fuse_player = nn.Conv2d(
            player_hidden * self.PLAYER_BRANCH_N, hidden_channels, kernel_size=1, padding=0, bias=False
        )
        self.merge_fuse_norm = nn.GroupNorm(g, hidden_channels)
        self.merge_fuse_act = nn.SiLU()

        self.register_buffer("enemy_player_index", torch.arange(1, M_MAX, dtype=torch.int64), persistent=False)

    def _split_research_v4(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = board.shape
        g = board[:, : self.R4_GLOBAL_C, :, :]
        p = board[:, self.R4_GLOBAL_C : self.R4_GLOBAL_C + self.R4_PLAYER_BLOCK_C, :, :]
        p = p.view(bsz, M_MAX, self.R4_PLAYER_PER_C, h, w)
        return g, p

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, c, h, w = board.shape
        if c != self.R4_TOTAL_C:
            raise RuntimeError(f"dwres_ppconcat_v1 expects C={self.R4_TOTAL_C}, got {int(c)}")

        x_main = self.main_stem(board)
        x_main = self.main_front(x_main)

        g, pfeat = self._split_research_v4(board)
        # Run network2 only for active opponents (p=1..7 and p < m).
        m_onehot = board[:, self.R4_M2_ONEHOT_OFFSET : self.R4_M2_ONEHOT_OFFSET + 7, 0, 0]
        m_each = torch.argmax(m_onehot, dim=1).to(torch.int64) + 2
        enemy_present = self.enemy_player_index.view(1, self.PLAYER_BRANCH_N) < m_each.view(-1, 1)

        hid = self.player_hidden_channels
        enemy_feat_all = pfeat[:, 1:, :, :, :]  # [B, 7, C_p, H, W]
        common_in = torch.cat((g, pfeat[:, 0, :, :, :]), dim=1)  # [B, Cg+Cp, H, W]
        common_proj = self.player_stem_common(common_in)  # [B, hid, H, W]
        enemy_feat_flat = enemy_feat_all.reshape(bsz * self.PLAYER_BRANCH_N, self.R4_PLAYER_PER_C, h, w)
        enemy_proj = self.player_stem_enemy_feat(enemy_feat_flat).view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
        enemy_proj = enemy_proj + self.player_stem_enemy_id.weight.view(1, self.PLAYER_BRANCH_N, hid, 1, 1).to(
            dtype=enemy_proj.dtype
        )
        py_all = common_proj.unsqueeze(1) + enemy_proj
        py_all = py_all.reshape(bsz * self.PLAYER_BRANCH_N, hid, h, w)
        py_all = self.player_stem_act(self.player_stem_norm(py_all))
        py_all = py_all.view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
        enemy_present_f = enemy_present.to(dtype=py_all.dtype).view(bsz, self.PLAYER_BRANCH_N, 1, 1, 1)
        set_idx = 0
        mix_idx = 0
        for blk_i, blk in enumerate(self.player_front_blocks):
            py_all = blk(py_all.reshape(bsz * self.PLAYER_BRANCH_N, hid, h, w)).view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
            py_all = py_all * enemy_present_f

            if set_idx < self.player_set_layers and ((blk_i + 1) % self.player_set_every == 0):
                tok = py_all.mean(dim=(3, 4))  # [B, 7, hid]
                tok = self.player_set_blocks[set_idx](tok, enemy_present)
                gamma_beta = self.player_set_token_to_film[set_idx](tok)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                py_all = py_all * (1.0 + gamma) + beta
                py_all = py_all * enemy_present_f
                set_idx += 1

            if mix_idx < self.player_mix_layers and ((blk_i + 1) % self.player_mix_every == 0):
                py_all = self.player_mix_blocks[mix_idx](py_all, enemy_present)
                py_all = py_all * enemy_present_f
                mix_idx += 1

        while set_idx < self.player_set_layers:
            tok = py_all.mean(dim=(3, 4))
            tok = self.player_set_blocks[set_idx](tok, enemy_present)
            gamma_beta = self.player_set_token_to_film[set_idx](tok)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            py_all = py_all * (1.0 + gamma) + beta
            py_all = py_all * enemy_present_f
            set_idx += 1

        while mix_idx < self.player_mix_layers:
            py_all = self.player_mix_blocks[mix_idx](py_all, enemy_present)
            py_all = py_all * enemy_present_f
            mix_idx += 1

        py_cat = py_all.reshape(bsz, -1, h, w)
        fused = self.merge_fuse_main(x_main) + self.merge_fuse_player(py_cat)
        fused = self.merge_fuse_act(self.merge_fuse_norm(fused))
        x = x_main + fused
        x = self.main_back(x)
        return x, x


class PolicyValueDWPlayerConcatPCatOnlyNet(PolicyValueBase):
    # research_v4 fixed layout:
    #   global  : 19ch [0:19]
    #   player  : 128ch [19:147] as (8 players x 16ch)
    #   pos0    : 2ch  [147:149]
    R4_GLOBAL_C = 19
    R4_PLAYER_PER_C = 16
    R4_PLAYER_BLOCK_C = M_MAX * R4_PLAYER_PER_C
    R4_TOTAL_C = R4_GLOBAL_C + R4_PLAYER_BLOCK_C + 2
    R4_M2_ONEHOT_OFFSET = 7  # global slice: m_onehot starts at ch7, values encode m=2..8
    PLAYER_BRANCH_N = M_MAX - 1  # process only p=1..7
    PLAYER_COMMON_INPUT_C = R4_GLOBAL_C + R4_PLAYER_PER_C
    PLAYER_ENEMY_FEAT_INPUT_C = R4_PLAYER_PER_C  # player p only

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        blocks: int = 6,
        *,
        feature_id: str | None = None,
        player_hidden_channels: int | None = None,
        use_full3x3_blocks: bool = False,
        player_set_layers: int = 0,
        player_set_heads: int = 4,
        player_set_ff_mult: float = 2.0,
        player_set_dropout: float = 0.0,
        player_set_every: int = 1,
        player_mix_layers: int = 0,
        player_mix_every: int = 1,
        player_mix_channel_gate: bool = False,
        player_mix_alpha_scale: float = 0.2,
    ):
        super().__init__(hidden_channels=hidden_channels)
        if in_channels != self.R4_TOTAL_C:
            raise ValueError(
                f"dwres_ppconcat_full_pcatonly_v1 requires in_channels={self.R4_TOTAL_C} (research_v4), got {int(in_channels)}"
            )
        if feature_id is not None and str(feature_id) != "research_v4":
            raise ValueError(f"dwres_ppconcat_full_pcatonly_v1 requires feature_id='research_v4', got {feature_id!r}")
        if blocks <= 0:
            raise ValueError(f"blocks must be >= 1, got {int(blocks)}")
        player_hidden = int(max(1, hidden_channels // 2) if player_hidden_channels is None else player_hidden_channels)
        if player_hidden <= 0:
            raise ValueError(f"player_hidden_channels must be >= 1, got {player_hidden}")
        if int(player_set_layers) < 0:
            raise ValueError(f"player_set_layers must be >= 0, got {int(player_set_layers)}")
        if int(player_set_heads) <= 0:
            raise ValueError(f"player_set_heads must be >= 1, got {int(player_set_heads)}")
        if float(player_set_ff_mult) <= 0.0:
            raise ValueError(f"player_set_ff_mult must be > 0, got {float(player_set_ff_mult)}")
        if not (0.0 <= float(player_set_dropout) < 1.0):
            raise ValueError(f"player_set_dropout must be in [0,1), got {float(player_set_dropout)}")
        if int(player_set_every) <= 0:
            raise ValueError(f"player_set_every must be >= 1, got {int(player_set_every)}")
        if int(player_mix_layers) < 0:
            raise ValueError(f"player_mix_layers must be >= 0, got {int(player_mix_layers)}")
        if int(player_mix_every) <= 0:
            raise ValueError(f"player_mix_every must be >= 1, got {int(player_mix_every)}")
        if float(player_mix_alpha_scale) <= 0.0:
            raise ValueError(f"player_mix_alpha_scale must be > 0, got {float(player_mix_alpha_scale)}")
        self.player_hidden_channels = player_hidden
        self.player_set_layers = int(player_set_layers)
        self.player_set_every = int(player_set_every)
        self.player_mix_layers = int(player_mix_layers)
        self.player_mix_every = int(player_mix_every)

        g = _pick_gn_groups(hidden_channels)
        g_player = _pick_gn_groups(player_hidden)
        front_blocks = max(1, int(blocks) // 2)
        back_blocks = max(0, int(blocks) - front_blocks)
        block_cls: type[nn.Module]
        if use_full3x3_blocks:
            block_cls = Full3x3PWResidualBlock
        else:
            block_cls = DWResidualBlock

        # Main trunk receives only player-concat fused features.
        self.main_back = nn.Sequential(*[block_cls(hidden_channels) for _ in range(back_blocks)])

        # network2 (shared across players): per-player independent processing
        self.player_stem_common = nn.Conv2d(self.PLAYER_COMMON_INPUT_C, player_hidden, kernel_size=1, padding=0, bias=False)
        self.player_stem_enemy_feat = nn.Conv2d(self.PLAYER_ENEMY_FEAT_INPUT_C, player_hidden, kernel_size=1, padding=0, bias=False)
        self.player_stem_enemy_id = nn.Embedding(self.PLAYER_BRANCH_N, player_hidden)
        self.player_stem_norm = nn.GroupNorm(g_player, player_hidden)
        self.player_stem_act = nn.SiLU()
        self.player_front_blocks = nn.ModuleList([block_cls(player_hidden) for _ in range(front_blocks)])
        self.player_set_blocks = nn.ModuleList()
        self.player_set_token_to_film = nn.ModuleList()
        self.player_mix_blocks = nn.ModuleList()
        for _ in range(self.player_set_layers):
            self.player_set_blocks.append(
                PlayerSetAttentionBlock(
                    player_hidden,
                    heads=int(player_set_heads),
                    ff_mult=float(player_set_ff_mult),
                    dropout=float(player_set_dropout),
                )
            )
            self.player_set_token_to_film.append(nn.Linear(player_hidden, player_hidden * 2))
        for _ in range(self.player_mix_layers):
            self.player_mix_blocks.append(
                PlayerAxisMixResidualBlock(
                    self.PLAYER_BRANCH_N,
                    player_hidden,
                    use_channel_gate=bool(player_mix_channel_gate),
                    alpha_scale=float(player_mix_alpha_scale),
                )
            )

        self.merge_fuse_player = nn.Conv2d(
            player_hidden * self.PLAYER_BRANCH_N, hidden_channels, kernel_size=1, padding=0, bias=False
        )
        self.merge_fuse_norm = nn.GroupNorm(g, hidden_channels)
        self.merge_fuse_act = nn.SiLU()

        self.register_buffer("enemy_player_index", torch.arange(1, M_MAX, dtype=torch.int64), persistent=False)

    def _split_research_v4(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = board.shape
        g = board[:, : self.R4_GLOBAL_C, :, :]
        p = board[:, self.R4_GLOBAL_C : self.R4_GLOBAL_C + self.R4_PLAYER_BLOCK_C, :, :]
        p = p.view(bsz, M_MAX, self.R4_PLAYER_PER_C, h, w)
        return g, p

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, c, h, w = board.shape
        if c != self.R4_TOTAL_C:
            raise RuntimeError(f"dwres_ppconcat_full_pcatonly_v1 expects C={self.R4_TOTAL_C}, got {int(c)}")

        g, pfeat = self._split_research_v4(board)
        # Run network2 only for active opponents (p=1..7 and p < m).
        m_onehot = board[:, self.R4_M2_ONEHOT_OFFSET : self.R4_M2_ONEHOT_OFFSET + 7, 0, 0]
        m_each = torch.argmax(m_onehot, dim=1).to(torch.int64) + 2
        enemy_present = self.enemy_player_index.view(1, self.PLAYER_BRANCH_N) < m_each.view(-1, 1)

        hid = self.player_hidden_channels
        enemy_feat_all = pfeat[:, 1:, :, :, :]  # [B, 7, C_p, H, W]
        common_in = torch.cat((g, pfeat[:, 0, :, :, :]), dim=1)  # [B, Cg+Cp, H, W]
        common_proj = self.player_stem_common(common_in)  # [B, hid, H, W]
        enemy_feat_flat = enemy_feat_all.reshape(bsz * self.PLAYER_BRANCH_N, self.R4_PLAYER_PER_C, h, w)
        enemy_proj = self.player_stem_enemy_feat(enemy_feat_flat).view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
        enemy_proj = enemy_proj + self.player_stem_enemy_id.weight.view(1, self.PLAYER_BRANCH_N, hid, 1, 1).to(
            dtype=enemy_proj.dtype
        )
        py_all = common_proj.unsqueeze(1) + enemy_proj
        py_all = py_all.reshape(bsz * self.PLAYER_BRANCH_N, hid, h, w)
        py_all = self.player_stem_act(self.player_stem_norm(py_all))
        py_all = py_all.view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
        enemy_present_f = enemy_present.to(dtype=py_all.dtype).view(bsz, self.PLAYER_BRANCH_N, 1, 1, 1)
        set_idx = 0
        mix_idx = 0
        for blk_i, blk in enumerate(self.player_front_blocks):
            py_all = blk(py_all.reshape(bsz * self.PLAYER_BRANCH_N, hid, h, w)).view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
            py_all = py_all * enemy_present_f

            if set_idx < self.player_set_layers and ((blk_i + 1) % self.player_set_every == 0):
                tok = py_all.mean(dim=(3, 4))  # [B, 7, hid]
                tok = self.player_set_blocks[set_idx](tok, enemy_present)
                gamma_beta = self.player_set_token_to_film[set_idx](tok)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                py_all = py_all * (1.0 + gamma) + beta
                py_all = py_all * enemy_present_f
                set_idx += 1

            if mix_idx < self.player_mix_layers and ((blk_i + 1) % self.player_mix_every == 0):
                py_all = self.player_mix_blocks[mix_idx](py_all, enemy_present)
                py_all = py_all * enemy_present_f
                mix_idx += 1

        while set_idx < self.player_set_layers:
            tok = py_all.mean(dim=(3, 4))
            tok = self.player_set_blocks[set_idx](tok, enemy_present)
            gamma_beta = self.player_set_token_to_film[set_idx](tok)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            py_all = py_all * (1.0 + gamma) + beta
            py_all = py_all * enemy_present_f
            set_idx += 1

        while mix_idx < self.player_mix_layers:
            py_all = self.player_mix_blocks[mix_idx](py_all, enemy_present)
            py_all = py_all * enemy_present_f
            mix_idx += 1

        py_cat = py_all.reshape(bsz, -1, h, w)
        x = self.merge_fuse_player(py_cat)
        x = self.merge_fuse_act(self.merge_fuse_norm(x))
        x = self.main_back(x)
        return x, x


class PolicyValueDWRelMixDualAttnNet(PolicyValueBase):
    # research_v4 fixed layout:
    #   global  : 19ch [0:19]
    #   player  : 128ch [19:147] as (8 players x 16ch)
    #   pos0    : 2ch  [147:149]
    R4_GLOBAL_C = 19
    R4_PLAYER_PER_C = 16
    R4_PLAYER_BLOCK_C = M_MAX * R4_PLAYER_PER_C
    R4_TOTAL_C = R4_GLOBAL_C + R4_PLAYER_BLOCK_C + 2
    R4_M2_ONEHOT_OFFSET = 7  # global slice: m_onehot starts at ch7, values encode m=2..8
    PLAYER_BRANCH_N = M_MAX - 1  # process only p=1..7
    PLAYER_COMMON_INPUT_C = R4_GLOBAL_C + R4_PLAYER_PER_C
    PLAYER_ENEMY_FEAT_INPUT_C = R4_PLAYER_PER_C  # player p only

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        blocks: int = 6,
        *,
        feature_id: str | None = None,
        player_hidden_channels: int | None = None,
        relmix_layers: int = 4,
        relmix_every: int = 2,
        relmix_coarse_size: int = 5,
        relmix_set_heads: int = 4,
        relmix_spatial_heads: int = 4,
        relmix_cross_heads: int = 4,
        relmix_ff_mult: float = 3.0,
        relmix_dropout: float = 0.0,
    ):
        super().__init__(hidden_channels=hidden_channels)
        if in_channels != self.R4_TOTAL_C:
            raise ValueError(
                f"dwres_relmix_dualattn_v3 requires in_channels={self.R4_TOTAL_C} (research_v4), got {int(in_channels)}"
            )
        if feature_id is not None and str(feature_id) != "research_v4":
            raise ValueError(f"dwres_relmix_dualattn_v3 requires feature_id='research_v4', got {feature_id!r}")
        if blocks <= 0:
            raise ValueError(f"blocks must be >= 1, got {int(blocks)}")
        if int(relmix_layers) < 0:
            raise ValueError(f"relmix_layers must be >= 0, got {int(relmix_layers)}")
        if int(relmix_every) <= 0:
            raise ValueError(f"relmix_every must be >= 1, got {int(relmix_every)}")
        if int(relmix_coarse_size) <= 0:
            raise ValueError(f"relmix_coarse_size must be >= 1, got {int(relmix_coarse_size)}")
        if int(relmix_cross_heads) <= 0:
            raise ValueError(f"relmix_cross_heads must be >= 1, got {int(relmix_cross_heads)}")
        if hidden_channels % int(relmix_cross_heads) != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by relmix_cross_heads ({int(relmix_cross_heads)})"
            )
        if float(relmix_ff_mult) <= 0.0:
            raise ValueError(f"relmix_ff_mult must be > 0, got {float(relmix_ff_mult)}")
        if not (0.0 <= float(relmix_dropout) < 1.0):
            raise ValueError(f"relmix_dropout must be in [0,1), got {float(relmix_dropout)}")

        player_hidden = int(
            hidden_channels if player_hidden_channels is None else player_hidden_channels
        )  # quality-prioritized default
        if player_hidden <= 0:
            raise ValueError(f"player_hidden_channels must be >= 1, got {player_hidden}")
        self.player_hidden_channels = player_hidden
        self.relmix_layers = int(relmix_layers)
        self.relmix_every = int(relmix_every)
        self.relmix_coarse_size = int(relmix_coarse_size)

        g = _pick_gn_groups(hidden_channels)
        g_player = _pick_gn_groups(player_hidden)
        front_blocks = max(1, int(blocks) // 2)
        back_blocks = max(0, int(blocks) - front_blocks)

        # Main branch (policy/value trunk backbone).
        self.main_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.main_front_blocks = nn.ModuleList([Full3x3PWResidualBlock(hidden_channels) for _ in range(front_blocks)])
        self.main_back = nn.Sequential(*[Full3x3PWResidualBlock(hidden_channels) for _ in range(back_blocks)])

        # Enemy branch (shared over p=1..7).
        self.enemy_stem_common = nn.Conv2d(self.PLAYER_COMMON_INPUT_C, player_hidden, kernel_size=1, padding=0, bias=False)
        self.enemy_stem_enemy_feat = nn.Conv2d(self.PLAYER_ENEMY_FEAT_INPUT_C, player_hidden, kernel_size=1, padding=0, bias=False)
        self.enemy_stem_enemy_id = nn.Embedding(self.PLAYER_BRANCH_N, player_hidden)
        self.enemy_stem_norm = nn.GroupNorm(g_player, player_hidden)
        self.enemy_stem_act = nn.SiLU()
        self.enemy_front_blocks = nn.ModuleList([Full3x3PWResidualBlock(player_hidden) for _ in range(front_blocks)])

        # Relational mixer blocks:
        # 1) coarse spatial player-attention
        # 2) global set-attention -> FiLM
        # 3) player0-centric cross-attention (main queries enemy keys/values)
        # 4) gated fusion into main branch
        self.relmix_spatial_attn = nn.ModuleList()
        self.relmix_set_attn = nn.ModuleList()
        self.relmix_set_to_film = nn.ModuleList()
        self.relmix_cross_attn = nn.ModuleList()
        self.relmix_enemy_to_main = nn.ModuleList()
        self.relmix_fuse = nn.ModuleList()
        for _ in range(self.relmix_layers):
            self.relmix_spatial_attn.append(
                PlayerSetAttentionBlock(
                    player_hidden,
                    heads=int(relmix_spatial_heads),
                    ff_mult=float(relmix_ff_mult),
                    dropout=float(relmix_dropout),
                )
            )
            self.relmix_set_attn.append(
                PlayerSetAttentionBlock(
                    player_hidden,
                    heads=int(relmix_set_heads),
                    ff_mult=float(relmix_ff_mult),
                    dropout=float(relmix_dropout),
                )
            )
            self.relmix_set_to_film.append(nn.Linear(player_hidden, player_hidden * 2))
            self.relmix_cross_attn.append(
                MainEnemyCrossAttentionBlock(
                    hidden_channels,
                    player_hidden,
                    heads=int(relmix_cross_heads),
                    ff_mult=float(relmix_ff_mult),
                    dropout=float(relmix_dropout),
                )
            )
            self.relmix_enemy_to_main.append(nn.Conv2d(player_hidden, hidden_channels, kernel_size=1, padding=0, bias=False))
            self.relmix_fuse.append(nn.Conv2d(hidden_channels * 3, hidden_channels * 2, kernel_size=1, padding=0, bias=False))

        # Final merge from enemy branch to main branch.
        self.merge_main = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=False)
        self.merge_enemy = nn.Conv2d(player_hidden, hidden_channels, kernel_size=1, padding=0, bias=False)
        self.merge_norm = nn.GroupNorm(g, hidden_channels)
        self.merge_act = nn.SiLU()

        self.register_buffer("enemy_player_index", torch.arange(1, M_MAX, dtype=torch.int64), persistent=False)

    def _split_research_v4(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = board.shape
        g = board[:, : self.R4_GLOBAL_C, :, :]
        p = board[:, self.R4_GLOBAL_C : self.R4_GLOBAL_C + self.R4_PLAYER_BLOCK_C, :, :]
        p = p.view(bsz, M_MAX, self.R4_PLAYER_PER_C, h, w)
        return g, p

    def _enemy_present_mask(self, board: torch.Tensor) -> torch.Tensor:
        m_onehot = board[:, self.R4_M2_ONEHOT_OFFSET : self.R4_M2_ONEHOT_OFFSET + 7, 0, 0]
        m_each = torch.argmax(m_onehot, dim=1).to(torch.int64) + 2
        return self.enemy_player_index.view(1, self.PLAYER_BRANCH_N) < m_each.view(-1, 1)

    @staticmethod
    def _masked_enemy_mean(py_all: torch.Tensor, enemy_present_f: torch.Tensor) -> torch.Tensor:
        den = enemy_present_f.sum(dim=1).clamp_min(1.0)
        return (py_all * enemy_present_f).sum(dim=1) / den

    def _apply_relmix_layer(
        self,
        rel_idx: int,
        x_main: torch.Tensor,
        py_all: torch.Tensor,
        enemy_present: torch.Tensor,
        enemy_present_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, pnum, hid, h, w = py_all.shape
        cs = int(self.relmix_coarse_size)

        # 1) Coarse spatial attention across players at each coarse cell.
        coarse = F.adaptive_avg_pool2d(py_all.reshape(bsz * pnum, hid, h, w), output_size=(cs, cs))
        coarse = coarse.view(bsz, pnum, hid, cs, cs)
        tok_sp = coarse.permute(0, 3, 4, 1, 2).reshape(bsz * cs * cs, pnum, hid)
        mask_sp = enemy_present.unsqueeze(1).expand(bsz, cs * cs, pnum).reshape(bsz * cs * cs, pnum)
        tok_sp = self.relmix_spatial_attn[rel_idx](tok_sp, mask_sp)
        coarse = tok_sp.view(bsz, cs, cs, pnum, hid).permute(0, 3, 4, 1, 2).contiguous()
        coarse_up = F.interpolate(
            coarse.reshape(bsz * pnum, hid, cs, cs),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).view(bsz, pnum, hid, h, w)
        py_all = (py_all + coarse_up) * enemy_present_f

        # 2) Global set-attention and FiLM feedback into per-player maps.
        tok_set = 0.5 * (py_all.mean(dim=(3, 4)) + py_all.amax(dim=(3, 4)))
        tok_set = self.relmix_set_attn[rel_idx](tok_set, enemy_present)
        gamma_beta = self.relmix_set_to_film[rel_idx](tok_set)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        py_all = (py_all * (1.0 + gamma) + beta) * enemy_present_f

        # 3) Player0-centric cross-attention (main query, enemy key/value).
        main_c = F.adaptive_avg_pool2d(x_main, output_size=(cs, cs))
        q_tok = main_c.flatten(2).transpose(1, 2).contiguous()  # [B, cs*cs, Hm]
        enemy_c = F.adaptive_avg_pool2d(py_all.reshape(bsz * pnum, hid, h, w), output_size=(cs, cs)).view(
            bsz, pnum, hid, cs, cs
        )
        kv_tok = enemy_c.permute(0, 1, 3, 4, 2).reshape(bsz, pnum * cs * cs, hid).contiguous()
        kv_valid = enemy_present.unsqueeze(-1).expand(bsz, pnum, cs * cs).reshape(bsz, pnum * cs * cs)
        q_tok = self.relmix_cross_attn[rel_idx](q_tok, kv_tok, kv_valid)
        ctx = q_tok.transpose(1, 2).reshape(bsz, x_main.shape[1], cs, cs).contiguous()
        ctx_up = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)

        # 4) Gated fusion into main features.
        enemy_mean = self._masked_enemy_mean(py_all, enemy_present_f)
        enemy_main = self.relmix_enemy_to_main[rel_idx](enemy_mean)
        fused_in = torch.cat((x_main, enemy_main, ctx_up), dim=1)
        delta_gate = self.relmix_fuse[rel_idx](fused_in)
        delta, gate = delta_gate.chunk(2, dim=1)
        x_main = x_main + torch.sigmoid(gate) * F.silu(delta)
        return x_main, py_all

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, c, h, w = board.shape
        if c != self.R4_TOTAL_C:
            raise RuntimeError(f"dwres_relmix_dualattn_v3 expects C={self.R4_TOTAL_C}, got {int(c)}")

        x_main = self.main_stem(board)
        g, pfeat = self._split_research_v4(board)
        enemy_present = self._enemy_present_mask(board)
        enemy_present_f = enemy_present.to(dtype=x_main.dtype).view(bsz, self.PLAYER_BRANCH_N, 1, 1, 1)

        hid = self.player_hidden_channels
        enemy_feat_all = pfeat[:, 1:, :, :, :]  # [B, 7, Cp, H, W]
        common_in = torch.cat((g, pfeat[:, 0, :, :, :]), dim=1)  # [B, Cg+Cp, H, W]
        common_proj = self.enemy_stem_common(common_in)  # [B, hid, H, W]
        enemy_feat_flat = enemy_feat_all.reshape(bsz * self.PLAYER_BRANCH_N, self.R4_PLAYER_PER_C, h, w)
        enemy_proj = self.enemy_stem_enemy_feat(enemy_feat_flat).view(bsz, self.PLAYER_BRANCH_N, hid, h, w)
        enemy_proj = enemy_proj + self.enemy_stem_enemy_id.weight.view(1, self.PLAYER_BRANCH_N, hid, 1, 1).to(
            dtype=enemy_proj.dtype
        )
        py_all = common_proj.unsqueeze(1) + enemy_proj
        py_all = py_all.reshape(bsz * self.PLAYER_BRANCH_N, hid, h, w)
        py_all = self.enemy_stem_act(self.enemy_stem_norm(py_all))
        py_all = py_all.view(bsz, self.PLAYER_BRANCH_N, hid, h, w) * enemy_present_f

        rel_idx = 0
        for blk_i, (main_blk, enemy_blk) in enumerate(zip(self.main_front_blocks, self.enemy_front_blocks)):
            x_main = main_blk(x_main)
            py_all = enemy_blk(py_all.reshape(bsz * self.PLAYER_BRANCH_N, hid, h, w)).view(
                bsz, self.PLAYER_BRANCH_N, hid, h, w
            )
            py_all = py_all * enemy_present_f
            if rel_idx < self.relmix_layers and ((blk_i + 1) % self.relmix_every == 0):
                x_main, py_all = self._apply_relmix_layer(rel_idx, x_main, py_all, enemy_present, enemy_present_f)
                rel_idx += 1

        while rel_idx < self.relmix_layers:
            x_main, py_all = self._apply_relmix_layer(rel_idx, x_main, py_all, enemy_present, enemy_present_f)
            rel_idx += 1

        enemy_mean = self._masked_enemy_mean(py_all, enemy_present_f)
        merged = self.merge_main(x_main) + self.merge_enemy(enemy_mean)
        merged = self.merge_act(self.merge_norm(merged))
        x = x_main + merged
        x = self.main_back(x)
        return x, x


class PolicyValueDWPixelPlayerAttnNoFFNNet(PolicyValueBase):
    # research_v4 fixed layout:
    #   global  : 19ch [0:19]
    #   player  : 128ch [19:147] as (8 players x 16ch)
    #   pos0    : 2ch  [147:149]
    R4_GLOBAL_C = 19
    R4_PLAYER_PER_C = 16
    R4_PLAYER_BLOCK_C = M_MAX * R4_PLAYER_PER_C
    R4_TOTAL_C = R4_GLOBAL_C + R4_PLAYER_BLOCK_C + 2
    R4_M2_ONEHOT_OFFSET = 7
    R4_PLAYER_INPUT_C = R4_GLOBAL_C + R4_PLAYER_PER_C  # [global + player_p]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        blocks: int = 6,
        *,
        feature_id: str | None = None,
        attention_heads: int = 4,
    ):
        super().__init__(hidden_channels=hidden_channels)
        if in_channels != self.R4_TOTAL_C:
            raise ValueError(
                f"dwres_pxattn_noffn_v1 requires in_channels={self.R4_TOTAL_C} (research_v4), got {int(in_channels)}"
            )
        if feature_id is not None and str(feature_id) != "research_v4":
            raise ValueError(f"dwres_pxattn_noffn_v1 requires feature_id='research_v4', got {feature_id!r}")
        if blocks <= 0:
            raise ValueError(f"blocks must be >= 1, got {int(blocks)}")

        g = _pick_gn_groups(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.R4_PLAYER_INPUT_C, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [PixelPlayerSelfAttentionDWBlock(hidden_channels, heads=int(attention_heads)) for _ in range(blocks)]
        )
        self.register_buffer("player_index", torch.arange(M_MAX, dtype=torch.int64), persistent=False)

    def _split_research_v4(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = board.shape
        g = board[:, : self.R4_GLOBAL_C, :, :]
        p = board[:, self.R4_GLOBAL_C : self.R4_GLOBAL_C + self.R4_PLAYER_BLOCK_C, :, :]
        p = p.view(bsz, M_MAX, self.R4_PLAYER_PER_C, h, w)
        return g, p

    def _player_present_mask(self, board: torch.Tensor) -> torch.Tensor:
        # m_onehot starts at ch7 and encodes m=2..8
        m_onehot = board[:, self.R4_M2_ONEHOT_OFFSET : self.R4_M2_ONEHOT_OFFSET + 7, 0, 0]
        m_each = torch.argmax(m_onehot, dim=1).to(torch.int64) + 2
        return self.player_index.view(1, M_MAX) < m_each.view(-1, 1)

    def _forward_trunk(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, c, h, w = board.shape
        if c != self.R4_TOTAL_C:
            raise RuntimeError(f"dwres_pxattn_noffn_v1 expects C={self.R4_TOTAL_C}, got {int(c)}")

        g, pfeat = self._split_research_v4(board)  # [B,19,H,W], [B,8,16,H,W]
        player_in = torch.cat((g.unsqueeze(1).expand(-1, M_MAX, -1, -1, -1), pfeat), dim=2)  # [B,8,35,H,W]
        x = self.stem(player_in.reshape(bsz * M_MAX, self.R4_PLAYER_INPUT_C, h, w)).reshape(
            bsz, M_MAX, -1, h, w
        )  # [B,8,C,H,W]

        present = self._player_present_mask(board)
        for blk in self.blocks:
            x = blk(x, present)

        x0 = x[:, 0, :, :, :]
        return x0, x, present

    def _forward_policy_only(self, board: torch.Tensor) -> torch.Tensor:
        x0, _, _ = self._forward_trunk(board)
        return x0

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0, _, _ = self._forward_trunk(board)
        return x0, x0

    def forward(
        self,
        board: torch.Tensor,
        *,
        with_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Policy/value use player0 map. Aux uses per-player hidden states.
        x0, x_all, present = self._forward_trunk(board)
        logits = self.policy_head(x0).flatten(1)  # [B, 100]
        pooled0 = x0.mean(dim=(2, 3))  # [B, H]
        value = self.value_head(pooled0).squeeze(1)  # [B]

        if with_aux:
            bsz, pnum, hid, h, w = x_all.shape
            flat = x_all.reshape(bsz * pnum, hid, h, w)
            move_all = self.opp_move_head(flat).view(bsz, pnum, M_MAX, h * w)  # [B, src_p, tgt_p, 100]

            pooled_all = x_all.mean(dim=(3, 4)).reshape(bsz * pnum, hid)  # [B*src_p, H]
            param_all = self.opp_param_head(pooled_all).view(bsz, pnum, M_MAX, 5)  # [B, src_p, tgt_p, 5]

            idx = self.player_index.to(device=board.device)
            opp_move_logits = move_all[:, idx, idx, :]  # [B, M_MAX, 100]
            opp_param = param_all[:, idx, idx, :]  # [B, M_MAX, 5]

            # Keep inactive slots benign; losses are masked by opp_valid on trainer side.
            present_f = present.to(dtype=opp_move_logits.dtype)
            opp_move_logits = opp_move_logits * present_f.unsqueeze(-1)
            opp_param = opp_param * present.to(dtype=opp_param.dtype).unsqueeze(-1)
        else:
            opp_move_logits = self._dummy_aux0
            opp_param = self._dummy_aux0
        return logits, value, opp_move_logits, opp_param


class PolicyValueDWGlobalCoordNet(PolicyValueBase):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__(hidden_channels=hidden_channels)
        g = _pick_gn_groups(hidden_channels)
        self.coord = CoordEmbed2d()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[DWResidualBlock(hidden_channels) for _ in range(blocks)])
        self.global_film = GlobalFiLM(hidden_channels)

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.coord(board)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_film(x)
        return x, x


class PolicyValueMBConvSEGlobalCoordNet(PolicyValueBase):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__(hidden_channels=hidden_channels)
        g = _pick_gn_groups(hidden_channels)
        self.coord = CoordEmbed2d()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[MBConvSEBlock(hidden_channels, expand_ratio=2, se_ratio=0.25) for _ in range(blocks)])
        self.global_film = GlobalFiLM(hidden_channels)

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.coord(board)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_film(x)
        return x, x


class PolicyValueMBConvSEGlobalCoordSplitNet(PolicyValueBase):
    def __init__(self, in_channels: int, hidden_channels: int = 64, blocks: int = 6):
        super().__init__(hidden_channels=hidden_channels)
        g = _pick_gn_groups(hidden_channels)
        self.coord = CoordEmbed2d()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.SiLU(),
        )
        shared_blocks = max(1, blocks // 2)
        branch_blocks = max(1, blocks - shared_blocks)
        self.shared_blocks = nn.Sequential(
            *[MBConvSEBlock(hidden_channels, expand_ratio=2, se_ratio=0.25) for _ in range(shared_blocks)]
        )
        self.policy_blocks = nn.Sequential(
            *[MBConvSEBlock(hidden_channels, expand_ratio=2, se_ratio=0.25) for _ in range(branch_blocks)]
        )
        self.value_blocks = nn.Sequential(
            *[MBConvSEBlock(hidden_channels, expand_ratio=2, se_ratio=0.25) for _ in range(branch_blocks)]
        )
        self.policy_film = GlobalFiLM(hidden_channels)
        self.value_film = GlobalFiLM(hidden_channels)

    def _forward_shared(self, board: torch.Tensor) -> torch.Tensor:
        x = self.coord(board)
        x = self.stem(x)
        return self.shared_blocks(x)

    def _forward_policy_only(self, board: torch.Tensor) -> torch.Tensor:
        x = self._forward_shared(board)
        x = self.policy_blocks(x)
        return self.policy_film(x)

    def _forward_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_shared(board)
        x_policy = self.policy_blocks(x)
        x_value = self.value_blocks(x)
        x_policy = self.policy_film(x_policy)
        x_value = self.value_film(x_value)
        return x_policy, x_value


def build_policy_value_model(
    arch_name: str,
    *,
    in_channels: int,
    hidden_channels: int,
    blocks: int,
    feature_id: str | None = None,
    arch_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    def _reject_unused(kwargs: dict[str, Any], target: str) -> None:
        if kwargs:
            keys = ", ".join(sorted(str(k) for k in kwargs.keys()))
            raise ValueError(f"{target} does not use arch_kwargs keys: {keys}")

    def _build_ppconcat(
        *,
        default_player_hidden: int | None,
        default_use_full3x3: bool,
        default_set_layers: int,
        default_mix_layers: int = 0,
        default_mix_every: int = 1,
        default_mix_channel_gate: bool = False,
        default_mix_alpha_scale: float = 0.2,
    ) -> nn.Module:
        kwargs = dict(arch_kwargs or {})
        player_hidden = kwargs.pop("player_hidden_channels", default_player_hidden)
        if player_hidden is not None:
            player_hidden = int(player_hidden)
        use_full3x3 = bool(kwargs.pop("use_full3x3_blocks", default_use_full3x3))
        player_set_layers = int(kwargs.pop("player_set_layers", default_set_layers))
        player_set_heads = int(kwargs.pop("player_set_heads", 4))
        player_set_ff_mult = float(kwargs.pop("player_set_ff_mult", 2.0))
        player_set_dropout = float(kwargs.pop("player_set_dropout", 0.0))
        player_set_every = int(kwargs.pop("player_set_every", 1))
        player_mix_layers = int(kwargs.pop("player_mix_layers", default_mix_layers))
        player_mix_every = int(kwargs.pop("player_mix_every", default_mix_every))
        player_mix_channel_gate = bool(kwargs.pop("player_mix_channel_gate", default_mix_channel_gate))
        player_mix_alpha_scale = float(kwargs.pop("player_mix_alpha_scale", default_mix_alpha_scale))
        _reject_unused(kwargs, "ppconcat arch")
        return PolicyValueDWPlayerConcatNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            feature_id=feature_id,
            player_hidden_channels=player_hidden,
            use_full3x3_blocks=use_full3x3,
            player_set_layers=player_set_layers,
            player_set_heads=player_set_heads,
            player_set_ff_mult=player_set_ff_mult,
            player_set_dropout=player_set_dropout,
            player_set_every=player_set_every,
            player_mix_layers=player_mix_layers,
            player_mix_every=player_mix_every,
            player_mix_channel_gate=player_mix_channel_gate,
            player_mix_alpha_scale=player_mix_alpha_scale,
        )

    def _build_ppconcat_pcatonly(
        *,
        default_player_hidden: int | None,
        default_use_full3x3: bool,
        default_set_layers: int,
        default_mix_layers: int = 0,
        default_mix_every: int = 1,
        default_mix_channel_gate: bool = False,
        default_mix_alpha_scale: float = 0.2,
    ) -> nn.Module:
        kwargs = dict(arch_kwargs or {})
        player_hidden = kwargs.pop("player_hidden_channels", default_player_hidden)
        if player_hidden is not None:
            player_hidden = int(player_hidden)
        use_full3x3 = bool(kwargs.pop("use_full3x3_blocks", default_use_full3x3))
        player_set_layers = int(kwargs.pop("player_set_layers", default_set_layers))
        player_set_heads = int(kwargs.pop("player_set_heads", 4))
        player_set_ff_mult = float(kwargs.pop("player_set_ff_mult", 2.0))
        player_set_dropout = float(kwargs.pop("player_set_dropout", 0.0))
        player_set_every = int(kwargs.pop("player_set_every", 1))
        player_mix_layers = int(kwargs.pop("player_mix_layers", default_mix_layers))
        player_mix_every = int(kwargs.pop("player_mix_every", default_mix_every))
        player_mix_channel_gate = bool(kwargs.pop("player_mix_channel_gate", default_mix_channel_gate))
        player_mix_alpha_scale = float(kwargs.pop("player_mix_alpha_scale", default_mix_alpha_scale))
        _reject_unused(kwargs, "ppconcat pcatonly arch")
        return PolicyValueDWPlayerConcatPCatOnlyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            feature_id=feature_id,
            player_hidden_channels=player_hidden,
            use_full3x3_blocks=use_full3x3,
            player_set_layers=player_set_layers,
            player_set_heads=player_set_heads,
            player_set_ff_mult=player_set_ff_mult,
            player_set_dropout=player_set_dropout,
            player_set_every=player_set_every,
            player_mix_layers=player_mix_layers,
            player_mix_every=player_mix_every,
            player_mix_channel_gate=player_mix_channel_gate,
            player_mix_alpha_scale=player_mix_alpha_scale,
        )

    def _build_relmix(
        *,
        default_player_hidden: int | None,
    ) -> nn.Module:
        kwargs = dict(arch_kwargs or {})
        player_hidden = kwargs.pop("player_hidden_channels", default_player_hidden)
        if player_hidden is not None:
            player_hidden = int(player_hidden)
        # CLI backward-compat for ppconcat args:
        #   player_set_layers  -> relmix_layers
        #   player_set_every   -> relmix_every
        #   player_set_heads   -> relmix_set_heads / relmix_spatial_heads (if not explicitly provided)
        #   player_set_ff_mult -> relmix_ff_mult
        #   player_set_dropout -> relmix_dropout
        relmix_layers = int(kwargs.pop("relmix_layers", kwargs.pop("player_set_layers", 4)))
        relmix_every = int(kwargs.pop("relmix_every", kwargs.pop("player_set_every", 2)))
        relmix_coarse_size = int(kwargs.pop("relmix_coarse_size", 5))
        pp_heads = kwargs.pop("player_set_heads", None)
        relmix_set_heads = int(kwargs.pop("relmix_set_heads", 4 if pp_heads is None else int(pp_heads)))
        relmix_spatial_heads = int(kwargs.pop("relmix_spatial_heads", relmix_set_heads))
        relmix_cross_heads = int(kwargs.pop("relmix_cross_heads", 4))
        relmix_ff_mult = float(kwargs.pop("relmix_ff_mult", kwargs.pop("player_set_ff_mult", 3.0)))
        relmix_dropout = float(kwargs.pop("relmix_dropout", kwargs.pop("player_set_dropout", 0.0)))
        _reject_unused(kwargs, "relmix arch")
        return PolicyValueDWRelMixDualAttnNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            feature_id=feature_id,
            player_hidden_channels=player_hidden,
            relmix_layers=relmix_layers,
            relmix_every=relmix_every,
            relmix_coarse_size=relmix_coarse_size,
            relmix_set_heads=relmix_set_heads,
            relmix_spatial_heads=relmix_spatial_heads,
            relmix_cross_heads=relmix_cross_heads,
            relmix_ff_mult=relmix_ff_mult,
            relmix_dropout=relmix_dropout,
        )

    name = str(arch_name).lower()
    if name in ("resnet", "resnet_v1", "policyvaluenet"):
        _reject_unused(dict(arch_kwargs or {}), str(arch_name))
        return PolicyValueNet(in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks)
    if name in ("dwres", "dwres_v1", "policyvaluedwnet"):
        _reject_unused(dict(arch_kwargs or {}), str(arch_name))
        return PolicyValueDWNet(in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks)
    if name in ("dwres_ppconcat", "dwres_ppconcat_v1", "policyvaluedwplayerconcatnet"):
        return _build_ppconcat(default_player_hidden=None, default_use_full3x3=False, default_set_layers=0)
    if name in ("dwres_ppconcat_full", "dwres_ppconcat_full_v1"):
        return _build_ppconcat(
            default_player_hidden=int(hidden_channels),
            default_use_full3x3=False,
            default_set_layers=0,
        )
    if name in ("dwres_ppconcat_full_pcatonly", "dwres_ppconcat_full_pcatonly_v1"):
        return _build_ppconcat_pcatonly(
            default_player_hidden=int(hidden_channels),
            default_use_full3x3=False,
            default_set_layers=0,
        )
    if name in ("dwres_ppconcat_h2", "dwres_ppconcat_h2_v1"):
        return _build_ppconcat(
            default_player_hidden=max(1, int(hidden_channels) // 2),
            default_use_full3x3=False,
            default_set_layers=0,
        )
    if name in (
        "dwres_ppconcat_full3x3",
        "dwres_ppconcat_full3x3_v1",
        "dwres_ppconcat_dense3x3",
        "dwres_ppconcat_dense3x3_v1",
    ):
        return _build_ppconcat(default_player_hidden=None, default_use_full3x3=True, default_set_layers=0)
    if name in (
        "dwres_ppconcat_full3x3_full",
        "dwres_ppconcat_full3x3_full_v1",
        "dwres_ppconcat_dense3x3_full",
        "dwres_ppconcat_dense3x3_full_v1",
    ):
        return _build_ppconcat(
            default_player_hidden=int(hidden_channels),
            default_use_full3x3=True,
            default_set_layers=0,
        )
    if name in (
        "dwres_ppconcat_full3x3_h2",
        "dwres_ppconcat_full3x3_h2_v1",
        "dwres_ppconcat_dense3x3_h2",
        "dwres_ppconcat_dense3x3_h2_v1",
    ):
        return _build_ppconcat(
            default_player_hidden=max(1, int(hidden_channels) // 2),
            default_use_full3x3=True,
            default_set_layers=0,
        )
    if name in (
        "dwres_ppconcat_full3x3_setiter_v2",
        "dwres_ppconcat_setiter_v2",
        "dwres_ppconcat_dense3x3_setiter_v2",
    ):
        return _build_ppconcat(default_player_hidden=None, default_use_full3x3=True, default_set_layers=3)
    if name in (
        "dwres_ppconcat_full3x3_pmix_gamma_v1",
        "dwres_ppconcat_dense3x3_pmix_gamma_v1",
        "dwres_ppconcat_pmix_gamma_v1",
    ):
        return _build_ppconcat(
            default_player_hidden=None,
            default_use_full3x3=True,
            default_set_layers=0,
            default_mix_layers=3,
            default_mix_every=2,
            default_mix_channel_gate=True,
            default_mix_alpha_scale=0.2,
        )
    if name in (
        "dwres_relmix_dualattn_v3",
        "dwres_ppconcat_relmix_dualattn_v3",
        "dwres_ppconcat_full3x3_relmix_v3",
    ):
        return _build_relmix(default_player_hidden=int(hidden_channels))
    if name in (
        "dwres_pxattn_noffn",
        "dwres_pxattn_noffn_v1",
        "dwres_pixel_player_attn_noffn",
        "dwres_pixel_player_attn_noffn_v1",
        "policyvaluedwpixelplayerattnnoffnnet",
    ):
        _reject_unused(dict(arch_kwargs or {}), str(arch_name))
        return PolicyValueDWPixelPlayerAttnNoFFNNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            feature_id=feature_id,
        )
    if name in ("dwres_gc", "dwres_gc_v1", "policyvaluedwglobalcoordnet"):
        _reject_unused(dict(arch_kwargs or {}), str(arch_name))
        return PolicyValueDWGlobalCoordNet(in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks)
    if name in ("mbconvse_gc", "mbconvse_gc_v1", "policyvaluembconvseglobalcoordnet"):
        _reject_unused(dict(arch_kwargs or {}), str(arch_name))
        return PolicyValueMBConvSEGlobalCoordNet(
            in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks
        )
    if name in ("mbconvse_gc_split", "mbconvse_gc_split_v1", "policyvaluembconvseglobalcoordsplitnet"):
        _reject_unused(dict(arch_kwargs or {}), str(arch_name))
        return PolicyValueMBConvSEGlobalCoordSplitNet(
            in_channels=in_channels, hidden_channels=hidden_channels, blocks=blocks
        )
    raise ValueError(f"unknown arch_name: {arch_name!r}")


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.uint8:
        # Fast path for uint8 masks from C++ env: 1=valid, 0=invalid.
        m = mask.to(dtype=logits.dtype)
        return logits - (1.0 - m) * 1.0e9
    if mask.dtype != torch.bool:
        mask = mask != 0
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, neg_inf)
