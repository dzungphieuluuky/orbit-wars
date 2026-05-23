
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TrainConfig, default_train_config_path, load_train_config
from src.features import (
    CANDIDATE_DIM,
    GLOBAL_DIM,
    SELF_DIM,
    TurnBatch,
    encode_turn,
)
from src.policy import PlanetPolicy, PolicyOutput
from src.ppo import sample_actions

# ── Types ─────────────────────────────────────────────────────────
Planet = namedtuple(
    "Planet",
    ["id", "owner", "x", "y", "radius", "ships", "production"],
)


@dataclass(slots=True)
class GameResult:
    game_idx:   int
    seed:       int
    reward:     float
    steps:      int
    label:      str       # "win" | "loss" | "draw"


@dataclass(slots=True)
class EvalSummary:
    wins:     int
    losses:   int
    draws:    int
    games:    int
    win_rate: float
    avg_steps: float


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained PlanetPolicy against the nearest-planet sniper."
    )
    p.add_argument("--config",       type=str, default=str(default_train_config_path()))
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Path to .pth checkpoint. Omit to evaluate an untrained policy.")
    p.add_argument("--games",        type=int, default=20)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--deterministic",action="store_true",
                   help="Use argmax instead of sampling for the learned policy.")
    p.add_argument("--player",       type=int, default=0, choices=[0, 1],
                   help="Which player slot the learned policy occupies (0 or 1).")
    return p.parse_args()


# ── Device / seed ─────────────────────────────────────────────────

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Policy construction ───────────────────────────────────────────

def build_policy(cfg: TrainConfig, device: torch.device) -> PlanetPolicy:
    """Construct PlanetPolicy from TrainConfig, matching train.py exactly."""
    m = cfg.model
    return PlanetPolicy(
        self_dim        = SELF_DIM,
        candidate_dim   = CANDIDATE_DIM,
        global_dim      = GLOBAL_DIM,
        candidate_count = cfg.env.candidate_count,
        hidden_size     = m.hidden_size,
        num_heads       = getattr(m, "num_heads",       4),
        num_attn_layers = getattr(m, "num_attn_layers", 2),
        mlp_ratio       = getattr(m, "mlp_ratio",       2.0),
        dropout         = getattr(m, "dropout",         0.0),
        use_memory      = getattr(m, "use_memory",      False),
        geom_indices    = tuple(getattr(m, "geom_indices", (0, 1, 2, 3))),
        fourier_bands   = getattr(m, "fourier_bands",   4),
    ).to(device)


def load_checkpoint(
    policy:          PlanetPolicy,
    checkpoint_path: str | None,
    device:          torch.device,
) -> int:
    """
    Load policy weights from checkpoint.

    Returns the update step stored in the checkpoint (0 if unavailable).
    Handles the torch.compile '_orig_mod.' prefix automatically.
    """
    if checkpoint_path is None:
        print("[eval] No checkpoint provided — using randomly initialised weights.")
        return 0

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt       = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("policy", ckpt)

    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }

    policy.load_state_dict(state_dict)
    update = int(ckpt.get("update", 0))
    print(f"[eval] Loaded checkpoint: {path.name}  (update={update})")
    return update


# ── Learned policy action ─────────────────────────────────────────

def build_moves(
    batch:         TurnBatch,
    policy:        PlanetPolicy,
    device:        torch.device,
    deterministic: bool,
    hidden_state:  torch.Tensor | None = None,
) -> tuple[list[list[float | int]], torch.Tensor | None]:
    """
    Run the learned policy on a TurnBatch and return game moves.

    Returns
    -------
    moves:
        List of [source_id, angle, ship_count] actions.
    next_hidden:
        Updated GRU hidden state (None if memory disabled).
    """
    if batch.self_features.shape[0] == 0:
        return [], hidden_state

    dev = device
    with torch.inference_mode():
        outputs: PolicyOutput = policy(
            torch.from_numpy(batch.self_features).to(dev,      non_blocking=True),
            torch.from_numpy(batch.candidate_features).to(dev, non_blocking=True),
            torch.from_numpy(batch.global_features).to(dev,    non_blocking=True),
            torch.from_numpy(batch.candidate_mask).to(dev,     non_blocking=True).bool(),
            hidden_state,
        )
        sampled = sample_actions(outputs, deterministic=deterministic)

    next_hidden    = outputs.hidden_state   # None when use_memory=False
    target_indices = sampled.target_index.cpu().numpy()

    moves: list[list[float | int]] = []
    for row_idx, ctx in enumerate(batch.contexts):
        ti = int(target_indices[row_idx])

        # Index 0 = no-op; out-of-range or masked = invalid
        if ti == 0:
            continue
        if ti >= len(ctx.candidate_ids):
            continue
        if not ctx.candidate_mask[ti]:
            continue

        ships = int(ctx.ship_counts[ti])
        if ships <= 0:
            continue

        moves.append([
            ctx.source_id,
            float(ctx.target_angles[ti]),
            ships,
        ])

    return moves, next_hidden


# ── Sniper baseline ───────────────────────────────────────────────

def nearest_planet_sniper(obs: Any) -> list[list[float | int]]:
    """
    Deterministic greedy baseline: each owned planet attacks
    the nearest non-owned planet if it has enough ships.
    """
    player     = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw        = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets    = [Planet(*p) for p in raw]
    my_planets = [p for p in planets if p.owner == player]
    targets    = [p for p in planets if p.owner != player]

    if not targets:
        return []

    moves: list[list[float | int]] = []
    for mine in my_planets:
        nearest  = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        needed   = max(nearest.ships + 1, 20)
        if mine.ships < needed:
            continue
        angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
        moves.append([mine.id, angle, needed])

    return moves


# ── State extraction helpers ──────────────────────────────────────

def _get(state: Any, key: str, default: Any = None) -> Any:
    return state.get(key, default) if isinstance(state, dict) \
           else getattr(state, key, default)

def _obs(state: Any)    -> Any:  return _get(state, "observation")
def _status(state: Any) -> str:  return str(_get(state, "status", "UNKNOWN"))
def _reward(state: Any) -> float:
    v = _get(state, "reward", 0.0)
    return 0.0 if v is None else float(v)

def _label(reward: float) -> str:
    if reward > 0: return "win"
    if reward < 0: return "loss"
    return "draw"


# ── Single game ───────────────────────────────────────────────────

def play_one_game(
    cfg:           TrainConfig,
    policy:        PlanetPolicy,
    device:        torch.device,
    *,
    seed:          int,
    deterministic: bool,
    learner_player:int = 0,
) -> GameResult:
    """
    Play one full game between the learned policy and the sniper.

    Args:
        learner_player: 0 = learned policy plays as player 0 (default).
                        1 = learned policy plays as player 1.
    """
    from kaggle_environments import make

    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "randomSeed": int(seed)},
        debug=False,
    )
    env.reset(num_agents=2)
    states = env.step([[], []])

    lp = learner_player
    sp = 1 - lp

    player_obs   = _obs(states[lp])
    opponent_obs = _obs(states[sp])
    done         = _status(states[lp]) != "ACTIVE"
    step_count   = 0
    hidden       : torch.Tensor | None = None

    while not done:
        # Learned policy action
        batch          = encode_turn(player_obs, cfg.env, env_index=0)
        player_action, hidden = build_moves(
            batch, policy, device, deterministic, hidden
        )

        # Sniper action
        opponent_action = nearest_planet_sniper(opponent_obs)

        # Step environment
        joint  = [None, None]
        joint[lp] = player_action
        joint[sp] = opponent_action
        states = env.step(joint)

        player_obs   = _obs(states[lp])
        opponent_obs = _obs(states[sp])
        done         = _status(states[lp]) != "ACTIVE"
        step_count  += 1

    reward = _reward(states[lp])
    return GameResult(
        game_idx = 0,           # filled in by caller
        seed     = seed,
        reward   = reward,
        steps    = step_count,
        label    = _label(reward),
    )


# ── Multi-game evaluation ─────────────────────────────────────────

def evaluate(
    cfg:           TrainConfig,
    policy:        PlanetPolicy,
    device:        torch.device,
    *,
    games:         int,
    base_seed:     int,
    deterministic: bool,
    learner_player:int = 0,
    verbose:       bool = True,
) -> EvalSummary:
    """Run `games` episodes and return aggregated statistics."""
    wins = losses = draws = 0
    total_steps = 0

    for game_idx in range(games):
        result          = play_one_game(
            cfg, policy, device,
            seed           = base_seed + game_idx,
            deterministic  = deterministic,
            learner_player = learner_player,
        )
        result.game_idx = game_idx + 1
        total_steps    += result.steps

        if result.label == "win":
            wins   += 1
        elif result.label == "loss":
            losses += 1
        else:
            draws  += 1

        if verbose:
            print(
                f"game={result.game_idx:>3} "
                f"seed={result.seed} "
                f"result={result.label:<4} "
                f"reward={result.reward:+.1f} "
                f"steps={result.steps}"
            )

    total    = max(games, 1)
    win_rate = wins / total

    return EvalSummary(
        wins      = wins,
        losses    = losses,
        draws     = draws,
        games     = games,
        win_rate  = win_rate,
        avg_steps = total_steps / total,
    )


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    cfg    = load_train_config(args.config)

    device = resolve_device(
        args.device if args.device != "auto" else cfg.device
    )
    seed_everything(args.seed)

    # Build and load policy
    policy = build_policy(cfg, device)
    load_checkpoint(policy, args.checkpoint, device)
    policy.eval()

    print(
        f"\n[eval] device={device} "
        f"games={args.games} "
        f"deterministic={args.deterministic} "
        f"learner_player={args.player}\n"
    )

    summary = evaluate(
        cfg,
        policy,
        device,
        games          = args.games,
        base_seed      = args.seed,
        deterministic  = args.deterministic,
        learner_player = args.player,
        verbose        = True,
    )

    print(
        f"\nsummary "
        f"wins={summary.wins} "
        f"losses={summary.losses} "
        f"draws={summary.draws} "
        f"games={summary.games} "
        f"avg_steps={summary.avg_steps:.1f}"
    )
    print(f"win_rate={summary.win_rate:.4f}")


if __name__ == "__main__":
    main()
