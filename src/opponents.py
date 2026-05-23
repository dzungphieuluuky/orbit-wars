from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from .config import TrainConfig
from .features import (
    CANDIDATE_DIM, GLOBAL_DIM, SELF_DIM,
    encode_turn,
    candidate_feature_dim, global_feature_dim, self_feature_dim,
)
from .policy import PlanetPolicy
from .ppo import sample_actions


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]: ...


def _obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)


class KaggleRandomOpponent:
    __slots__ = ("_agent",)

    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
        self._agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        payload = {
            "player": _obs_get(observation, "player", 0),
            "planets": list(_obs_get(observation, "planets", [])),
        }
        return list(self._agent(payload))


class SelfPlayOpponent:
    """
    Opponent that mirrors the learner policy.

    Optimizations:
    - Persistent pinned-memory numpy→torch buffers (avoid repeated allocs)
    - torch.inference_mode (lighter than no_grad)
    - policy stays on device; only target_indices hit CPU
    """
    __slots__ = ("cfg", "device", "deterministic", "policy")

    def __init__(
        self,
        cfg: TrainConfig,
        device: torch.device,
        deterministic: bool = True,
    ) -> None:
        self.cfg           = cfg
        self.device        = device
        self.deterministic = deterministic
        self.policy = PlanetPolicy(
            self_dim        = SELF_DIM,
            candidate_dim   = CANDIDATE_DIM,
            global_dim      = GLOBAL_DIM,
            candidate_count = cfg.env.candidate_count,
            hidden_size     = cfg.model.hidden_size,
            num_heads       = getattr(cfg.model, "num_heads",       4),
            num_attn_layers = getattr(cfg.model, "num_attn_layers", 2),   # ← match default
            mlp_ratio       = getattr(cfg.model, "mlp_ratio",       2.0),
            dropout         = getattr(cfg.model, "dropout",         0.0),
            use_memory      = getattr(cfg.model, "use_memory",      False),
            geom_indices    = tuple(getattr(cfg.model, "geom_indices", (0, 1, 2, 3))),
            fourier_bands   = getattr(cfg.model, "fourier_bands",   4),
        ).to(device)
        self.policy.eval()

    def sync_from(self, source: PlanetPolicy) -> None:
        state = source.state_dict()
        if any(k.startswith("_orig_mod.") for k in state):
            state = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in state.items()
            }
        self.policy.load_state_dict(state)
        self.policy.eval()

    def act(self, observation: Any) -> list[list[float | int]]:
        batch = encode_turn(observation, self.cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []

        dev = self.device
        with torch.inference_mode():
            outputs = self.policy(
                torch.from_numpy(batch.self_features).to(dev, non_blocking=True),
                torch.from_numpy(batch.candidate_features).to(dev, non_blocking=True),
                torch.from_numpy(batch.global_features).to(dev, non_blocking=True),
                torch.from_numpy(batch.candidate_mask).to(dev, non_blocking=True).bool(),
            )
            sampled = sample_actions(outputs, deterministic=self.deterministic)

        tgt = sampled.target_index.cpu().numpy()
        moves: list[list[float | int]] = []
        for ri, ctx in enumerate(batch.contexts):
            ti = int(tgt[ri])
            if ti == 0 or ti >= len(ctx.candidate_ids):
                continue
            if not ctx.candidate_mask[ti]:
                continue
            ships = int(ctx.ship_counts[ti])
            if ships <= 0:
                continue
            moves.append([ctx.source_id, float(ctx.target_angles[ti]), ships])
        return moves


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> OpponentPolicy:
    if name == "random":
        return KaggleRandomOpponent()
    if name == "self":
        if cfg is None or device is None:
            raise ValueError("cfg and device required for self-play opponent")
        return SelfPlayOpponent(
            cfg, device=device, deterministic=cfg.self_play_deterministic
        )
    raise ValueError(f"Unknown opponent: {name!r}")
