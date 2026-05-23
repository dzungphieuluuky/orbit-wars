
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .game_types import GameState, parse_observation

# ── Constants (module-level) ─────────────────────────────────────
BOARD_CX: float = 50.0
BOARD_CY: float = 50.0
ROTATION_RADIUS_LIMIT: float = 50.0
SUN_RADIUS: float = 10.0
PLANET_LAUNCH_OFFSET: float = 0.1

# Pre-computed reciprocals to replace divisions in hot paths
_INV_MAX_PLANETS_48: float = 1.0 / 48.0
_INV_MAX_SHIPS_400:  float = 1.0 / 400.0
_INV_MAX_PROD_5:     float = 1.0 / 5.0
_INV_BOARD_100:      float = 1.0 / 100.0
_INV_RADIUS_5:       float = 1.0 / 5.0


@dataclass(slots=True)
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    ship_counts: list[int]
    target_angles: list[float]


@dataclass(slots=True)
class TurnBatch:
    self_features: np.ndarray       # (N, 11)  float32
    candidate_features: np.ndarray  # (N, K, 14) float32
    global_features: np.ndarray     # (N, 8)  float32
    candidate_mask: np.ndarray      # (N, K)  bool
    contexts: list[DecisionContext]
    state: GameState


# ── Dimension constants (avoids function call overhead) ───────────
SELF_DIM      = 11
CANDIDATE_DIM = 14
GLOBAL_DIM    = 8


def self_feature_dim()      -> int: return SELF_DIM
def candidate_feature_dim() -> int: return CANDIDATE_DIM
def global_feature_dim()    -> int: return GLOBAL_DIM


# ── Pure NumPy Helper Functions ────────────────────────────────────

def _dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def _is_rotating(x: float, y: float, radius: float) -> bool:
    """Check if planet is in rotating ring around sun."""
    dx = x - BOARD_CX
    dy = y - BOARD_CY
    return math.sqrt(dx * dx + dy * dy) + radius < ROTATION_RADIUS_LIMIT


def _seg_dist(px: float, py: float,
              x1: float, y1: float,
              x2: float, y2: float) -> float:
    """Minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0.0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_sq))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def _crosses_sun(sx: float, sy: float, sr: float,
                 angle: float,
                 tx: float, ty: float) -> bool:
    """Check if trajectory from source planet crosses the sun."""
    off = sr + PLANET_LAUNCH_OFFSET
    bx = sx + math.cos(angle) * off
    by = sy + math.sin(angle) * off
    return _seg_dist(BOARD_CX, BOARD_CY, bx, by, tx, ty) < SUN_RADIUS


def _top_k_by_dist(dists: np.ndarray,
                   mask: np.ndarray,
                   k: int,
                   n: int) -> np.ndarray:
    """Select top k indices by distance among valid positions (numpy version)."""
    # Get valid indices and distances
    valid_idx = np.where(mask)[0]
    valid_dist = dists[valid_idx]

    # Return top k indices (or fewer if not enough valid)
    if len(valid_idx) == 0:
        return np.full(k, -1, dtype=np.int32)

    # Partial sort to find top k
    num_to_select = min(k, len(valid_idx))
    top_indices = np.argsort(valid_dist)[:num_to_select]
    result_indices = valid_idx[top_indices].astype(np.int32)

    # Pad with -1 if needed
    result = np.full(k, -1, dtype=np.int32)
    result[:len(result_indices)] = result_indices
    return result


def _candidates_one(
    src_idx: int,
    planet_ids: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    owners: np.ndarray,
    player_id: int,
    k: int, n: int,
    eq: int, nq: int, fq: int,
) -> np.ndarray:
    """Select candidate targets for a single source planet."""
    sx, sy = xs[src_idx], ys[src_idx]

    # Build distance array (invalid slots → huge sentinel)
    dists = np.full(n, 1e9, dtype=np.float32)
    valid_mask = (planet_ids != -1) & (np.arange(n) != src_idx)
    for i in np.where(valid_mask)[0]:
        dists[i] = _dist2(sx, sy, xs[i], ys[i])

    # Category masks
    em = valid_mask & (owners != -1) & (owners != player_id)  # enemy
    nm = valid_mask & (owners == -1)                           # neutral
    fm = valid_mask & (owners == player_id)                    # friendly

    # Select top-k from each category
    et = _top_k_by_dist(dists, em, eq, n)
    nt = _top_k_by_dist(dists, nm, nq, n)
    ft = _top_k_by_dist(dists, fm, fq, n)

    # Merge results
    result = np.full(k, -1, dtype=np.int32)
    pos = 0
    for arr in (et, nt, ft):
        for idx in arr:
            if idx != -1 and pos < k:
                result[pos] = idx
                pos += 1

    # Fallback: fill remaining slots by proximity
    if pos < k:
        used = np.zeros(n, dtype=bool)
        used[result[:pos][result[:pos] != -1]] = True

        # Sort by distance and fill remaining
        order = np.argsort(dists)
        for idx in order:
            if pos >= k:
                break
            if valid_mask[idx] and not used[idx]:
                result[pos] = idx
                pos += 1

    return result


def _build_all_candidates(
    src_indices: np.ndarray, m: int,
    planet_ids: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    owners: np.ndarray,
    player_id: int, k: int, n: int,
    eq: int, nq: int, fq: int,
) -> np.ndarray:
    """Build candidate target arrays for all source planets (numpy version)."""
    out = np.full((m, k), -1, dtype=np.int32)
    for i in range(m):
        si = src_indices[i]
        if si != -1:
            out[i] = _candidates_one(
                si, planet_ids, xs, ys, owners,
                player_id, k, n, eq, nq, fq,
            )
    return out


def _self_feat(
    si: int,
    xs: np.ndarray, ys: np.ndarray,
    radii: np.ndarray, ships: np.ndarray,
    production: np.ndarray,
    my_cnt_n: float, en_cnt_n: float,
    my_shp_n: float, en_shp_n: float,
    inv_board: float, inv_ms: float, inv_mp: float,
) -> np.ndarray:
    """Compute self features for a source planet."""
    f = np.empty(SELF_DIM, dtype=np.float32)
    max_ships = 1.0 / inv_ms
    f[0]  = 1.0
    f[1]  = xs[si] * inv_board
    f[2]  = ys[si] * inv_board
    f[3]  = radii[si] * _INV_RADIUS_5
    f[4]  = min(ships[si], max_ships) * inv_ms
    f[5]  = production[si] * inv_mp
    f[6]  = 1.0 if _is_rotating(xs[si], ys[si], radii[si]) else 0.0
    f[7]  = my_cnt_n
    f[8]  = en_cnt_n
    f[9]  = my_shp_n
    f[10] = en_shp_n
    return f


def _cand_feats(
    si: int,
    tgt_indices: np.ndarray,
    k: int,
    xs: np.ndarray, ys: np.ndarray,
    radii: np.ndarray, ships: np.ndarray,
    production: np.ndarray, owners: np.ndarray,
    player_id: int,
    inv_board: float, inv_ms: float, inv_mp: float,
):
    """Compute candidate features for a source planet."""
    cf    = np.zeros((k, CANDIDATE_DIM), dtype=np.float32)
    cmask = np.zeros(k, dtype=bool)
    scnt  = np.zeros(k, dtype=np.int32)
    cids  = np.full(k, -1, dtype=np.int32)
    tang  = np.zeros(k, dtype=np.float32)

    cmask[0] = True   # no-op slot always valid

    sx, sy, sr = xs[si], ys[si], radii[si]
    ss = ships[si]
    max_ships = 1.0 / inv_ms

    for j in range(k):
        ti = tgt_indices[j]
        if ti == -1:
            continue
        tx, ty   = xs[ti], ys[ti]
        tsh      = ships[ti]
        tprod    = production[ti]
        town     = owners[ti]
        tr       = radii[ti]

        dx = tx - sx
        dy = ty - sy
        angle = math.atan2(dy, dx)
        dist  = math.sqrt(dx * dx + dy * dy)

        sun  = _crosses_sun(sx, sy, sr, angle, tx, ty)
        need = max(int(tsh) + 1, 20)

        cf[j, 0]  = 1.0
        cf[j, 1]  = 1.0 if town == -1 else 0.0
        cf[j, 2]  = 1.0 if town == player_id else 0.0
        cf[j, 3]  = 1.0 if (town != -1 and town != player_id) else 0.0
        cf[j, 4]  = tx * inv_board
        cf[j, 5]  = ty * inv_board
        cf[j, 6]  = dx * inv_board
        cf[j, 7]  = dy * inv_board
        cf[j, 8]  = dist * inv_board
        cf[j, 9]  = min(tsh, max_ships) * inv_ms
        cf[j, 10] = tprod * inv_mp
        cf[j, 11] = 1.0 if _is_rotating(tx, ty, tr) else 0.0
        cf[j, 12] = 1.0 if sun else 0.0
        cf[j, 13] = min(ss, max_ships) * inv_ms

        scnt[j]  = need
        cids[j]  = ti
        tang[j]  = angle
        cmask[j] = (need > 0) and (not sun) and (ss >= need)

    return cf, cmask, scnt, cids, tang


# ── Public API ────────────────────────────────────────────────────

def encode_turn(
    observation: Any,
    env_cfg: EnvConfig,
    *,
    env_index: int = 0,
) -> TurnBatch:
    state = (observation if isinstance(observation, GameState)
             else parse_observation(observation))

    # Unpack config once — avoids repeated attribute lookups in loops
    board_size    = env_cfg.board_size
    k             = env_cfg.candidate_count
    n             = env_cfg.max_planets
    max_ships     = env_cfg.max_ships
    max_prod      = env_cfg.max_production
    episode_steps = env_cfg.episode_steps

    # Pre-computed reciprocals
    inv_board = 1.0 / board_size
    inv_ms    = 1.0 / max_ships
    inv_mp    = 1.0 / max_prod
    inv_n     = 1.0 / n

    eq = k // 3
    nq = k // 3
    fq = k - eq - nq

    # ── Fill padded planet arrays in one pass ─────────────────────
    planet_ids = np.full(n, -1,  dtype=np.int32)
    owners     = np.full(n, -2,  dtype=np.int32)
    xs         = np.zeros(n,     dtype=np.float32)
    ys         = np.zeros(n,     dtype=np.float32)
    radii      = np.zeros(n,     dtype=np.float32)
    ships      = np.zeros(n,     dtype=np.float32)
    production = np.zeros(n,     dtype=np.float32)

    planets = state.planets
    np_limit = min(len(planets), n)
    for i in range(np_limit):
        p = planets[i]
        planet_ids[i] = p.id
        owners[i]     = p.owner
        xs[i]         = p.x
        ys[i]         = p.y
        radii[i]      = p.radius
        ships[i]      = p.ships
        production[i] = p.production

    # ── Boolean masks (vectorised) ────────────────────────────────
    valid   = planet_ids != -1
    my_mask = valid & (owners == state.player)
    en_mask = valid & (owners != state.player) & (owners != -1)
    ne_mask = valid & (owners == -1)

    my_planet_indices = np.where(my_mask)[0].astype(np.int32)
    m = len(my_planet_indices)

    if m == 0:
        empty = TurnBatch(
            self_features     = np.zeros((0, SELF_DIM),    dtype=np.float32),
            candidate_features= np.zeros((0, k, CANDIDATE_DIM), dtype=np.float32),
            global_features   = np.zeros((0, GLOBAL_DIM), dtype=np.float32),
            candidate_mask    = np.zeros((0, k),           dtype=bool),
            contexts          = [],
            state             = state,
        )
        return empty

    # ── Scalar global stats (cheap numpy reductions) ──────────────
    my_cnt      = float(my_mask.sum())
    en_cnt      = float(en_mask.sum())
    my_shp      = float(ships[my_mask].sum())
    en_shp      = float(ships[en_mask].sum())
    ne_cnt      = float(ne_mask.sum())
    max_tot_shp = n * max_ships

    my_cnt_n = my_cnt * inv_n
    en_cnt_n = en_cnt * inv_n
    my_shp_n = my_shp / max_tot_shp
    en_shp_n = en_shp / max_tot_shp

    # ── Candidate IDs (pure numpy) ────────────────────────────────
    cid_mat = _build_all_candidates(
        my_planet_indices, m,
        planet_ids, xs, ys, owners,
        state.player, k, n, eq, nq, fq,
    )

    # ── Per-source features ───────────────────────────────────────
    # Pre-allocate output arrays — avoids repeated np.array(list) conversions
    self_arr = np.empty((m, SELF_DIM),        dtype=np.float32)
    cand_arr = np.empty((m, k, CANDIDATE_DIM),dtype=np.float32)
    mask_arr = np.zeros((m, k),               dtype=bool)
    contexts: list[DecisionContext] = []

    for i, si in enumerate(my_planet_indices):
        si_int = int(si)

        self_arr[i] = _self_feat(
            si_int, xs, ys, radii, ships, production,
            my_cnt_n, en_cnt_n, my_shp_n, en_shp_n,
            inv_board, inv_ms, inv_mp,
        )

        cf, cm, sc, ci, ta = _cand_feats(
            si_int, cid_mat[i], k,
            xs, ys, radii, ships, production, owners,
            state.player, inv_board, inv_ms, inv_mp,
        )
        cand_arr[i] = cf
        mask_arr[i] = cm

        contexts.append(DecisionContext(
            env_index    = env_index,
            source_id    = int(planet_ids[si_int]),
            candidate_ids= ci.tolist(),
            candidate_mask= cm,
            ship_counts  = sc.tolist(),
            target_angles= ta.tolist(),
        ))

    # ── Global features (broadcast) ───────────────────────────────
    # Current features: step, my_cnt, en_cnt, neutral_cnt, my_ships, en_ships
    # TODO: Implement fleet-in-flight counting (reserved slots 6-7):
    #   - Track num_friendly_fleets, num_enemy_fleets
    #   - Could improve policy awareness of in-transit units
    gf = np.array([
        state.step / episode_steps,
        my_cnt_n, en_cnt_n,
        ne_cnt * inv_n,
        my_shp_n, en_shp_n,
        0.0,  # Reserved: num_friendly_fleets (not yet implemented)
        0.0,  # Reserved: num_enemy_fleets (not yet implemented)
    ], dtype=np.float32)
    global_arr = np.broadcast_to(gf[None, :], (m, GLOBAL_DIM)).copy()

    return TurnBatch(
        self_features     = self_arr,
        candidate_features= cand_arr,
        global_features   = global_arr,
        candidate_mask    = mask_arr,
        contexts          = contexts,
        state             = state,
    )
