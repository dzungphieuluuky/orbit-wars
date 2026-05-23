
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]


# Pre-allocated dtype descriptors for zero-copy parsing
_PLANET_DTYPE = np.dtype([
    ("id", np.int32), ("owner", np.int32),
    ("x", np.float32), ("y", np.float32),
    ("radius", np.float32), ("ships", np.int32),
    ("production", np.int32),
])

_FLEET_DTYPE = np.dtype([
    ("id", np.int32), ("owner", np.int32),
    ("x", np.float32), ("y", np.float32),
    ("angle", np.float32), ("from_planet_id", np.int32),
    ("ships", np.int32),
])


def _obs_get(observation: Any, key: str, default: Any) -> Any:
    """Single-dispatch attribute/dict access — faster than repeated isinstance."""
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)


def parse_observation(observation: Any) -> GameState:
    raw_planets = _obs_get(observation, "planets", [])
    raw_fleets  = _obs_get(observation, "fleets",  [])

    # Vectorized construction: convert to numpy structured array first
    planets: list[PlanetState] = []
    if raw_planets:
        arr = np.asarray(raw_planets, dtype=np.float64)
        for row in arr:
            planets.append(PlanetState(
                id=int(row[0]), owner=int(row[1]),
                x=float(row[2]), y=float(row[3]),
                radius=float(row[4]), ships=int(row[5]),
                production=int(row[6]),
            ))

    fleets: list[FleetState] = []
    if raw_fleets:
        arr = np.asarray(raw_fleets, dtype=np.float64)
        for row in arr:
            fleets.append(FleetState(
                id=int(row[0]), owner=int(row[1]),
                x=float(row[2]), y=float(row[3]),
                angle=float(row[4]), from_planet_id=int(row[5]),
                ships=int(row[6]),
            ))

    return GameState(
        step=int(_obs_get(observation, "step", 0)),
        player=int(_obs_get(observation, "player", 0)),
        planets=planets,
        fleets=fleets,
    )
