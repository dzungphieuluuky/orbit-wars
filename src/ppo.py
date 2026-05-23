from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: Tensor
    log_prob: Tensor
    entropy: Tensor


@dataclass(slots=True)
class TransitionBatch:
    self_features:      Tensor
    candidate_features: Tensor
    global_features:    Tensor
    candidate_mask:     Tensor
    target_index:       Tensor
    log_prob:           Tensor
    returns:            Tensor
    advantages:         Tensor


# Pre-built zero tensor to avoid allocation in the hot path
_ZERO = torch.zeros(1)



def _safe_logits(logits: Tensor) -> Tensor:
    """Replace fully-invalid rows with a uniform distribution at slot 0."""
    invalid = ~torch.isfinite(logits).any(dim=-1)   # (N,)
    if not invalid.any():
        return logits
    out = logits.clone()
    out[invalid, 0] = 0.0
    return out


def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:
    logits = _safe_logits(outputs.target_logits)
    if deterministic:
        idx = logits.argmax(dim=-1)
    else:
        idx = Categorical(logits=logits).sample()

    lp, ent = _log_prob_and_entropy(logits, idx)
    return SampledAction(target_index=idx, log_prob=lp, entropy=ent)



def _log_prob_and_entropy(logits: Tensor, idx: Tensor):
    dist = Categorical(logits=logits)
    return dist.log_prob(idx), dist.entropy()


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: Tensor,
) -> tuple[Tensor, Tensor]:
    return _log_prob_and_entropy(_safe_logits(outputs.target_logits), target_index)



def _ppo_loss(
    advantages: Tensor,
    ratio: Tensor,
    clip_coef: float,
    returns_mb: Tensor,
    value_mb: Tensor,
    entropy: Tensor,
    vf_coef: float,
    ent_coef: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Clipped surrogate objective
    pg1 = -advantages * ratio
    pg2 = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.maximum(pg1, pg2).mean()

    # Value loss (MSE)
    value_loss = 0.5 * F.mse_loss(value_mb, returns_mb)

    entropy_mean = entropy.mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    return total, policy_loss, value_loss, entropy_mean


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    N = batch.self_features.shape[0]
    if N == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # ── Transfer to device once, pin if possible ──────────────────
    pin = device.type == "cuda"

    def _to(t: Tensor) -> Tensor:
        return t.to(device, non_blocking=True)

    sf   = _to(batch.self_features)
    cf   = _to(batch.candidate_features)
    gf   = _to(batch.global_features)
    cm   = _to(batch.candidate_mask).bool()
    olp  = _to(batch.log_prob)
    tidx = _to(batch.target_index)
    ret  = _to(batch.returns)
    adv  = _to(batch.advantages)

    # Normalise advantages once (not per minibatch)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    mb  = min(N, max(1, minibatch_size))
    acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0

    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, mb):
            idx = perm[start: start + mb]

            outputs = policy(sf[idx], cf[idx], gf[idx], cm[idx])
            nlp, ent = action_log_prob_and_entropy(outputs, tidx[idx])

            # ratio in log-space for numerical stability
            ratio = (nlp - olp[idx]).exp()

            loss, pl, vl, em = _ppo_loss(
                adv[idx], ratio, clip_coef,
                ret[idx], outputs.value,
                ent, vf_coef, ent_coef,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            # Accumulate metrics — detach early to free graph memory
            acc["loss"]        += loss.detach().item()
            acc["policy_loss"] += pl.detach().item()
            acc["value_loss"]  += vl.detach().item()
            acc["entropy"]     += em.detach().item()
            n_updates += 1

    inv = 1.0 / max(n_updates, 1)
    return {k: v * inv for k, v in acc.items()}
