from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from .cpp_ext import load_ext


@dataclass(frozen=True)
class EnvSpec:
    n: int = 10
    t_max: int = 100
    m_max: int = 8


class BatchEnv:
    def __init__(
        self,
        batch_size: int,
        *,
        feature_id: str = "submit_v1",
        pf_enabled: bool = True,
        verbose_build: bool = False,
    ):
        self._ext = load_ext(verbose=verbose_build)
        self._env = self._ext.BatchEnv(batch_size=batch_size, feature_id=str(feature_id), pf_enabled=pf_enabled)
        self.spec = EnvSpec()

    @property
    def batch_size(self) -> int:
        return int(self._env.batch_size)

    @property
    def pf_enabled(self) -> bool:
        return bool(self._env.pf_enabled)

    @property
    def feature_id(self) -> str:
        return str(self._env.feature_id)

    @property
    def feature_channels(self) -> int:
        return int(self._env.feature_channels())

    def feature_channels_of(self, feature_id: str) -> int:
        return int(self._env.feature_channels_of(str(feature_id)))

    def set_pf_enabled(self, v: bool) -> None:
        self._env.set_pf_enabled(bool(v))

    def set_feature_id(self, feature_id: str) -> None:
        self._env.set_feature_id(str(feature_id))

    def reset_random(self, seeds: torch.Tensor) -> None:
        self._env.reset_random(seeds.to(dtype=torch.int64, device="cpu"))

    def reset_from_tools(self, paths: Sequence[str], pf_seeds_extra: Optional[torch.Tensor] = None) -> None:
        if pf_seeds_extra is None:
            self._env.reset_from_tools(list(paths))
            return
        pf_seeds_extra_cpu = pf_seeds_extra.to(dtype=torch.int64, device="cpu")
        self._env.reset_from_tools_seeded(list(paths), pf_seeds_extra_cpu)

    def observe(self):
        return self._env.observe()

    def observe_into(self, board: torch.Tensor, mask: torch.Tensor) -> None:
        self._env.observe_into(board, mask)

    def observe_into_feature(self, board: torch.Tensor, mask: torch.Tensor, feature_id: str) -> None:
        self._env.observe_into_feature(board, mask, str(feature_id))

    def observe_pair_into(
        self,
        board_a: torch.Tensor,
        board_b: torch.Tensor,
        mask: torch.Tensor,
        *,
        feature_id_a: str,
        feature_id_b: str,
    ) -> None:
        self._env.observe_pair_into(board_a, board_b, mask, str(feature_id_a), str(feature_id_b))

    def aux_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._env.aux_targets()

    def aux_targets_into(self, move_dist: torch.Tensor, opp_param: torch.Tensor, opp_valid: torch.Tensor) -> None:
        self._env.aux_targets_into(move_dist, opp_param, opp_valid)

    def step(self, actions: torch.Tensor):
        return self._env.step(actions.to(dtype=torch.int64, device="cpu"))

    def step_into(self, actions: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_into(actions_cpu, reward, done)

    def step_observe_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_observe_into(actions_cpu, board, mask, reward, done)

    def step_observe_aux_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        move_dist: torch.Tensor,
        opp_param: torch.Tensor,
        opp_valid: torch.Tensor,
    ) -> None:
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_observe_aux_into(actions_cpu, board, mask, reward, done, move_dist, opp_param, opp_valid)

    def step_observe_pair_into(
        self,
        actions: torch.Tensor,
        board_a: torch.Tensor,
        board_b: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        *,
        feature_id_a: str,
        feature_id_b: str,
    ) -> None:
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_observe_pair_into(
            actions_cpu,
            board_a,
            board_b,
            mask,
            reward,
            done,
            str(feature_id_a),
            str(feature_id_b),
        )

    def pos0(self) -> torch.Tensor:
        return self._env.pos0()

    def official_score(self) -> torch.Tensor:
        return self._env.official_score()

    def score_s0_sa(self) -> torch.Tensor:
        return self._env.score_s0_sa()


def tools_input_paths(seed_begin: int, seed_end: int) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    base = repo_root / "tools" / "in"
    out: list[str] = []
    for s in range(seed_begin, seed_end + 1):
        out.append(str(base / f"{s:04d}.txt"))
    return out
