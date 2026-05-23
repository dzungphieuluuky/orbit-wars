from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TrainConfig
from .features import TurnBatch, encode_turn
from .opponents import OpponentPolicy


@dataclass(slots=True)
class StepResult:
    batch:  TurnBatch
    reward: float
    done:   bool
    info:   dict[str, Any]


def _get(state: Any, key: str, default: Any = None) -> Any:
    """Unified attribute / dict access — single branch per call."""
    return state.get(key, default) if isinstance(state, dict) \
           else getattr(state, key, default)


def _obs(state: Any) -> Any:
    return _get(state, "observation")

def _status(state: Any) -> str:
    return str(_get(state, "status", "UNKNOWN"))

def _reward(state: Any) -> float:
    v = _get(state, "reward", 0.0)
    return 0.0 if v is None else float(v)


def _terminal_reward(ps: Any, os: Any) -> float:
    pr = _reward(ps)
    return 0.0 if (pr > 0.0 and _reward(os) > 0.0) else pr


class OrbitWarsEnv:
    __slots__ = (
        "cfg", "opponent", "make_fn", "env_index",
        "env", "last_obs", "last_opp_obs",
        "episode_index", "learner_player",
    )

    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg           = cfg
        self.opponent      = opponent
        self.make_fn       = make_fn
        self.env_index     = env_index
        self.env           = None
        self.last_obs      = None
        self.last_opp_obs  = None
        self.episode_index = 0
        self.learner_player= 0

    def reset(self, seed: int | None = None) -> TurnBatch:
        from kaggle_environments import make as _make
        make_fn = self.make_fn or _make

        cfg_dict: dict[str, Any] = {}
        if seed is not None:
            cfg_dict["seed"] = cfg_dict["randomSeed"] = int(seed)

        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0

        self.env = make_fn("orbit_wars", configuration=cfg_dict, debug=False)
        self.env.reset(num_agents=2)
        states = self.env.step([[], []])

        lp = self.learner_player
        self.last_obs     = _obs(states[lp])
        self.last_opp_obs = _obs(states[1 - lp])
        self.episode_index += 1

        return encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")

        opp_action = self.opponent.act(self.last_opp_obs)
        lp = self.learner_player
        joint = ([player_action, opp_action] if lp == 0
                 else [opp_action, player_action])

        states = self.env.step(joint)
        ps = states[lp]
        os = states[1 - lp]

        self.last_obs     = _obs(ps)
        self.last_opp_obs = _obs(os)

        done   = _status(ps) != "ACTIVE"
        reward = _terminal_reward(ps, os) if done else 0.0
        batch  = encode_turn(self.last_obs, self.cfg.env, env_index=self.env_index)

        return StepResult(
            batch  = batch,
            reward = reward,
            done   = done,
            info   = {
                "learner_player" : lp,
                "player_status"  : _status(ps),
                "opponent_status": _status(os),
                "reward"         : reward,
            },
        )
