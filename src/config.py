
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# frozen=True allows hashing and is slightly faster for attribute access
@dataclass(slots=True, frozen=True)
class EnvConfig:
    board_size: float = 100.0
    episode_steps: int = 500
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_planets: int = 48
    max_ships: float = 400.0
    max_production: float = 5.0


@dataclass(slots=True, frozen=True)
class ModelConfig:
    hidden_size:     int   = 128
    num_heads:       int   = 4
    num_attn_layers: int   = 1
    mlp_ratio:       float = 2.0
    dropout:         float = 0.0
    use_memory:      bool  = False
    geom_indices:    tuple = (0, 1, 2, 3)
    fourier_bands:   int   = 4


@dataclass(slots=True, frozen=True)
class PPOConfig:
    rollout_steps: int = 64
    num_envs: int = 8
    total_updates: int = 5000
    epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    lr: float = 0.0002
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class TrainConfig:
    run_name: str = "orbit_wars_ppo"
    save_dir: str = "artifacts"
    checkpoint_every: int = 100
    opponent: str = "self"
    self_play_update_interval: int = 50
    seed: int = 42
    device: str = "auto"
    log_every: int = 1
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


@lru_cache(maxsize=1)
def default_train_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "default_cfg.yaml"


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return _train_config_from_dict(data)


def _train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    # Build frozen sub-configs directly
    env_data   = data.get("env", {})
    model_data = data.get("model", {})
    ppo_data   = data.get("ppo", {})

    env_cfg   = _make_frozen(EnvConfig,   env_data)
    model_cfg = _make_frozen(ModelConfig, model_data)
    ppo_cfg   = _make_frozen(PPOConfig,   ppo_data)

    top_level = {k: v for k, v in data.items() if k not in {"env", "model", "ppo"}}
    cfg = TrainConfig(env=env_cfg, model=model_cfg, ppo=ppo_cfg)
    _update_dataclass(cfg, top_level)
    return cfg


def _make_frozen(cls, data: dict[str, Any]):
    """Instantiate a frozen dataclass, coercing types from a dict."""
    import dataclasses
    defaults = cls()
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce_value(data[f.name], getattr(defaults, f.name))
    return cls(**kwargs)


def _update_dataclass(
    instance: Any,
    values: dict[str, Any],
    skip: set[str] | None = None,
) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        object.__setattr__(instance, key, _coerce_value(value, default)) \
            if getattr(type(instance), "__dataclass_params__", None) and \
               getattr(type(instance).__dataclass_params__, "frozen", False) \
            else setattr(instance, key, _coerce_value(value, default))


# Lookup table avoids repeated isinstance checks
_BOOL_TRUE  = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})


def _coerce_value(value: Any, default: Any) -> Any:
    t = type(default)
    if t is bool:
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _BOOL_TRUE:  return True
            if low in _BOOL_FALSE: return False
        return bool(value)
    if t is int:
        return int(value)
    if t is float:
        return float(value)
    return value
