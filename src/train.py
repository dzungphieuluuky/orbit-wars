from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import TrainConfig, default_train_config_path, load_train_config
from .env import OrbitWarsEnv
from .features import (
    CANDIDATE_DIM, GLOBAL_DIM, SELF_DIM,
    TurnBatch,
    candidate_feature_dim, global_feature_dim, self_feature_dim,
)
from .game_types import PlanetState
from .opponents import SelfPlayOpponent, build_opponent
from .policy import PlanetPolicy
from .ppo import TransitionBatch, ppo_update, sample_actions


# ── Helpers ───────────────────────────────────────────────────────

def is_kaggle() -> bool: return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
def is_colab()  -> bool: return "google.colab" in str(globals().get("get_ipython", lambda: "")())


@dataclass(slots=True)
class StepGroup:
    indices: list[int]
    reward:  float
    done:    bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(default_train_config_path()))
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep speed; non-determinism is acceptable in RL
    torch.backends.cudnn.benchmark     = True


def _find_planet(planets: list[PlanetState], pid: int) -> PlanetState | None:
    for p in planets:
        if p.id == pid:
            return p
    return None


# ── Batch merging ─────────────────────────────────────────────────

def merge_batches(batches: list[TurnBatch]) -> TurnBatch:
    """Concatenate TurnBatches; avoids re-allocating when any batch is empty."""
    if not batches:
        raise ValueError("batches must not be empty")

    parts_sf, parts_cf, parts_gf, parts_cm = [], [], [], []
    all_ctx: list = []
    for b in batches:
        if b.self_features.shape[0] > 0:
            parts_sf.append(b.self_features)
            parts_cf.append(b.candidate_features)
            parts_gf.append(b.global_features)
            parts_cm.append(b.candidate_mask)
        all_ctx.extend(b.contexts)

    k = batches[0].candidate_features.shape[1]
    if parts_sf:
        sf = np.concatenate(parts_sf, axis=0)
        cf = np.concatenate(parts_cf, axis=0)
        gf = np.concatenate(parts_gf, axis=0)
        cm = np.concatenate(parts_cm, axis=0)
    else:
        sf = np.zeros((0, SELF_DIM),      dtype=np.float32)
        cf = np.zeros((0, k, CANDIDATE_DIM), dtype=np.float32)
        gf = np.zeros((0, GLOBAL_DIM),    dtype=np.float32)
        cm = np.zeros((0, k),             dtype=bool)

    return TurnBatch(
        self_features     = sf,
        candidate_features= cf,
        global_features   = gf,
        candidate_mask    = cm,
        contexts          = all_ctx,
        state             = batches[0].state,
    )


# ── Bootstrap helper ──────────────────────────────────────────────

def bootstrap_values(
    policy: PlanetPolicy,
    batches: list[TurnBatch],
    device: torch.device,
) -> list[float]:
    merged = merge_batches(batches)
    if merged.self_features.shape[0] == 0:
        return [0.0] * len(batches)

    # Cumulative offsets to slice per-env values
    sizes   = [b.self_features.shape[0] for b in batches]
    offsets = np.concatenate([[0], np.cumsum(sizes[:-1])]).astype(int)

    with torch.inference_mode():
        out = policy(
            torch.from_numpy(merged.self_features).to(device, non_blocking=True),
            torch.from_numpy(merged.candidate_features).to(device, non_blocking=True),
            torch.from_numpy(merged.global_features).to(device, non_blocking=True),
            torch.from_numpy(merged.candidate_mask).to(device, non_blocking=True).bool(),
        )
    vals = out.value.cpu().numpy()

    result = []
    for i, (start, sz) in enumerate(zip(offsets, sizes)):
        result.append(0.0 if sz == 0 else float(vals[start: start + sz].mean()))
    return result


# ── Rollout collection ────────────────────────────────────────────

def collect_rollout(
    envs:      list[OrbitWarsEnv],
    batches:   list[TurnBatch],
    policy:    PlanetPolicy,
    cfg:       TrainConfig,
    device:    torch.device,
    next_seed: int,
) -> tuple[TransitionBatch, list[TurnBatch], int, dict[str, float]]:
    k   = cfg.env.candidate_count
    T   = cfg.ppo.rollout_steps
    NE  = len(envs)
    γ   = cfg.ppo.gamma

    # Pre-allocate Python lists for rollout storage
    # Upper bound: T * NE * max_planets entries
    self_rows:  list[np.ndarray] = []
    cand_rows:  list[np.ndarray] = []
    glob_rows:  list[np.ndarray] = []
    mask_rows:  list[np.ndarray] = []
    tgt_idx:    list[int]        = []
    log_probs:  list[float]      = []
    values:     list[float]      = []
    groups_per_env: list[list[StepGroup]] = [[] for _ in range(NE)]
    ep_rewards: list[float]      = []
    running_r:  list[float]      = [0.0] * NE

    for _ in range(T):
        # ── Merge env observations → single batch ────────────────
        merged = merge_batches(batches)
        sizes  = [b.self_features.shape[0] for b in batches]
        offsets= np.concatenate([[0], np.cumsum(sizes[:-1])]).astype(int)

        M = merged.self_features.shape[0]

        if M > 0:
            with torch.inference_mode():
                out = policy(
                    torch.from_numpy(merged.self_features).to(device, non_blocking=True),
                    torch.from_numpy(merged.candidate_features).to(device, non_blocking=True),
                    torch.from_numpy(merged.global_features).to(device, non_blocking=True),
                    torch.from_numpy(merged.candidate_mask).to(device, non_blocking=True).bool(),
                )
                sampled  = sample_actions(out, deterministic=False)
                row_vals = out.value.cpu().numpy()            # (M,)
                s_tidx   = sampled.target_index.cpu().numpy() # (M,)
                s_lp     = sampled.log_prob.cpu().numpy()     # (M,)
        else:
            row_vals = np.zeros(0, dtype=np.float32)
            s_tidx   = np.zeros(0, dtype=np.int64)
            s_lp     = np.zeros(0, dtype=np.float32)

        # ── Dispatch actions to each env ─────────────────────────
        next_batches: list[TurnBatch] = []
        for ei, env in enumerate(envs):
            b     = batches[ei]
            start = int(offsets[ei])
            moves: list[list[float | int]] = []
            g_indices: list[int] = []

            for li, ctx in enumerate(b.contexts):
                gi = start + li
                self_rows.append(b.self_features[li])
                cand_rows.append(b.candidate_features[li])
                glob_rows.append(b.global_features[li])
                mask_rows.append(b.candidate_mask[li])
                values.append(float(row_vals[gi]) if M > 0 else 0.0)

                ti = int(s_tidx[gi]) if M > 0 else 0
                tgt_idx.append(ti)
                log_probs.append(float(s_lp[gi]) if M > 0 else 0.0)
                g_indices.append(len(values) - 1)

                valid_send = (
                    ti > 0
                    and ti < len(ctx.candidate_ids)
                    and ctx.candidate_mask[ti]
                    and ctx.ship_counts[ti] > 0
                )
                if not valid_send:
                    continue
                ships = int(ctx.ship_counts[ti])
                src   = _find_planet(b.state.planets, ctx.source_id)
                if src is None or src.ships < ships:
                    continue
                moves.append([ctx.source_id, float(ctx.target_angles[ti]), ships])

            result = env.step(moves)
            running_r[ei] += float(result.reward)
            groups_per_env[ei].append(StepGroup(
                indices=g_indices, reward=float(result.reward), done=result.done
            ))

            if result.done:
                ep_rewards.append(running_r[ei])
                running_r[ei] = 0.0
                next_seed += 1
                next_batches.append(env.reset(seed=next_seed))
            else:
                next_batches.append(result.batch)

        batches = next_batches

    # ── Compute returns & advantages ──────────────────────────────
    n_rows   = len(values)
    returns  = np.zeros(n_rows, dtype=np.float32)
    advs     = np.zeros(n_rows, dtype=np.float32)
    boot     = bootstrap_values(policy, batches, device)

    for ei, groups in enumerate(groups_per_env):
        fut = boot[ei]
        for g in reversed(groups):
            fut = g.reward + γ * fut * (1.0 - float(g.done))
            for idx in g.indices:
                returns[idx] = fut
                advs[idx]    = fut - values[idx]

    # ── Assemble TransitionBatch ──────────────────────────────────
    # Use np.stack (avoids per-element copy that np.array(list) does)
    tb = TransitionBatch(
        self_features     = torch.from_numpy(
            np.stack(self_rows).reshape(-1, SELF_DIM)),
        candidate_features= torch.from_numpy(
            np.stack(cand_rows).reshape(-1, k, CANDIDATE_DIM)),
        global_features   = torch.from_numpy(
            np.stack(glob_rows).reshape(-1, GLOBAL_DIM)),
        candidate_mask    = torch.from_numpy(
            np.stack(mask_rows).reshape(-1, k)),
        target_index      = torch.tensor(tgt_idx,  dtype=torch.long),
        log_prob          = torch.tensor(log_probs, dtype=torch.float32),
        returns           = torch.from_numpy(returns),
        advantages        = torch.from_numpy(advs),
    )

    stats = {
        "episode_reward_mean": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "episodes_finished"  : float(len(ep_rewards)),
        "samples"            : float(n_rows),
    }
    return tb, batches, next_seed, stats


# ── Checkpointing ─────────────────────────────────────────────────

def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update:   int,
    policy:   PlanetPolicy,
    optimizer: torch.optim.Optimizer,
    cfg:      TrainConfig,
) -> None:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "update"   : update,
        "policy"   : policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config"   : cfg,
    }
    # Write numbered checkpoint, then overwrite "last" atomically
    numbered = run_dir / f"ckpt_{update:06d}.pth"
    torch.save(payload, numbered)
    last = run_dir / "ckpt_last.pth"
    torch.save(payload, last)


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_train_config(args.config)
    cfg.save_dir = "artifacts"

    seed_everything(cfg.seed)
    device   = resolve_device(cfg.device)
    opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)

    envs      = [OrbitWarsEnv(cfg, opponent, env_index=i)
                 for i in range(cfg.ppo.num_envs)]
    next_seed = cfg.seed
    batches   = []
    for env in envs:
        batches.append(env.reset(seed=next_seed))
        next_seed += 1

    # ── Build policy ──────────────────────────────────────────────
    policy_raw = PlanetPolicy(
        self_dim        = SELF_DIM,
        candidate_dim   = CANDIDATE_DIM,
        global_dim      = GLOBAL_DIM,
        candidate_count = cfg.env.candidate_count,
        hidden_size     = cfg.model.hidden_size,
        num_heads       = getattr(cfg.model, "num_heads",       4),
        num_attn_layers = getattr(cfg.model, "num_attn_layers", 2),
        mlp_ratio       = getattr(cfg.model, "mlp_ratio",       2.0),
        dropout         = getattr(cfg.model, "dropout",         0.0),
        use_memory      = getattr(cfg.model, "use_memory",      False),
        geom_indices    = tuple(getattr(cfg.model, "geom_indices", (0, 1, 2, 3))),
        fourier_bands   = getattr(cfg.model, "fourier_bands",   4),
    ).to(device)
    # ── Compile (keeps policy_raw as uncompiled reference) ────────
    policy = policy_raw
    if hasattr(torch, "compile"):
        try:
            # policy = torch.compile(policy_raw, mode="reduce-overhead")
            # print("torch.compile: enabled")

            print("torch.compile disabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    print("Model Architecture:\n")
    print(policy)

    # ── Sync opponent from uncompiled weights ─────────────────────
    if isinstance(opponent, SelfPlayOpponent):
        opponent.sync_from(policy_raw)

    # ── Optimizer on uncompiled parameters ───────────────────────
    optimizer = torch.optim.AdamW(
        policy_raw.parameters(),
        lr           = cfg.ppo.lr,
        weight_decay = 1e-5,
        fused        = (device.type == "cuda"),
    )

    save_dir = Path(cfg.save_dir)

    for update in range(1, cfg.ppo.total_updates + 1):
        # policy (compiled) used for forward passes
        tb, batches, next_seed, stats = collect_rollout(
            envs, batches, policy, cfg, device, next_seed
        )
        metrics = ppo_update(
            policy, optimizer, tb,
            clip_coef     = cfg.ppo.clip_coef,
            ent_coef      = cfg.ppo.ent_coef,
            vf_coef       = cfg.ppo.vf_coef,
            max_grad_norm = cfg.ppo.max_grad_norm,
            epochs        = cfg.ppo.epochs,
            minibatch_size= cfg.ppo.minibatch_size,
            device        = device,
        )

        # policy_raw used for everything that touches state_dict
        if (isinstance(opponent, SelfPlayOpponent)
                and update % cfg.self_play_update_interval == 0):
            opponent.sync_from(policy_raw)             # ← raw

        if update % cfg.log_every == 0:
            print(
                f"update={update:>5} "
                f"reward={stats['episode_reward_mean']:+.4f} "
                f"ep={int(stats['episodes_finished']):>3} "
                f"samples={int(stats['samples']):>6} "
                f"loss={metrics['loss']:.4f} "
                f"ent={metrics['entropy']:.4f}",
                flush=True,
            )

        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(
                save_dir, cfg.run_name, update,
                policy_raw,                            # ← raw
                optimizer, cfg,
            )
            print(f"  ↳ checkpoint saved → {save_dir / cfg.run_name}",
                  flush=True)


if __name__ == "__main__":
    main()
