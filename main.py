
import os
import sys
import torch
import numpy as np
from pathlib import Path

class _Path_hybrid(type(Path())):
    """Helper class to ensure compatibility with Path-like operations."""
    pass

# ---- Path Resolution ----
def _find_weights_path():
    candidates = [
        _Path_hybrid("/kaggle_simulations/agent/weights.npz"),
        _Path_hybrid.cwd() / "weights.npz",
        _Path_hybrid("weights.npz"),
    ]
    try:
        candidates.insert(0, _Path_hybrid(__file__).resolve().parent / "weights.npz")
    except NameError:
        pass
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

_WEIGHTS_PATH = _find_weights_path()

from src.config import TrainConfig
from src.features import SELF_DIM, CANDIDATE_DIM, GLOBAL_DIM, encode_turn
from src.policy import PlanetPolicy
from src.ppo import sample_actions


# Global variables
_POLICY = None
_CONFIG = None
_DEVICE = torch.device("cpu")

def load_agent():
    global _POLICY, _CONFIG

    if not os.path.exists(_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found at {_WEIGHTS_PATH}")

    # Load weights from npz
    weights_data = np.load(_WEIGHTS_PATH, allow_pickle=True)

    # Assuming config was pickled in the npz or use defaults
    _CONFIG = TrainConfig()

    # Initialize policy structure
    _POLICY = PlanetPolicy(
        self_dim        = SELF_DIM,
        candidate_dim   = CANDIDATE_DIM,
        global_dim      = GLOBAL_DIM,
        candidate_count = _CONFIG.env.candidate_count,
        hidden_size     = _CONFIG.model.hidden_size,
        num_heads       = getattr(_CONFIG.model, "num_heads",       4),
        num_attn_layers = getattr(_CONFIG.model, "num_attn_layers", 2),
        mlp_ratio       = getattr(_CONFIG.model, "mlp_ratio",       2.0),
        dropout         = getattr(_CONFIG.model, "dropout",         0.0),
        use_memory      = getattr(_CONFIG.model, "use_memory",      False),
        geom_indices    = tuple(getattr(_CONFIG.model, "geom_indices", (0, 1, 2, 3))),
        fourier_bands   = getattr(_CONFIG.model, "fourier_bands",   4),
    ).to(_DEVICE)

    # Map numpy arrays back to torch tensors
    state_dict = {k: torch.from_numpy(v) for k, v in weights_data.items()}
    _POLICY.load_state_dict(state_dict)
    _POLICY.eval()

def agent(obs, config=None):
    global _POLICY, _CONFIG

    if _POLICY is None:
        load_agent()

    batch = encode_turn(obs, _CONFIG.env, env_index=0)
    if batch.self_features.shape[0] == 0: return []

    with torch.inference_mode():
        outputs = _POLICY(
            torch.from_numpy(batch.self_features).to(_DEVICE),
            torch.from_numpy(batch.candidate_features).to(_DEVICE),
            torch.from_numpy(batch.global_features).to(_DEVICE),
            torch.from_numpy(batch.candidate_mask).to(_DEVICE).bool(),
            None,   # hidden_state — stateless inference for submission agent
        )
        sampled = sample_actions(outputs, deterministic=True)

    target_indices = sampled.target_index.detach().cpu().numpy()
    moves = []
    for row_idx, context in enumerate(batch.contexts):
        target_idx = int(target_indices[row_idx])
        if target_idx == 0 or target_idx >= len(context.candidate_ids): continue
        if not context.candidate_mask[target_idx]: continue
        ships = int(context.ship_counts[target_idx])
        if ships <= 0: continue
        moves.append([context.source_id, float(context.target_angles[target_idx]), ships])

    return moves
