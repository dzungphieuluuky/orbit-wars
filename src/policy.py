
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Data classes ──────────────────────────────────────────────────

@dataclass(slots=True)
class PolicyOutput:
    target_logits: Tensor           # (B, K)
    value:         Tensor           # (B,)
    hidden_state:  Tensor | None    # (B, h) or None


# ── Helpers ───────────────────────────────────────────────────────

def _mlp(
    in_dim:  int,
    hidden:  int,
    out_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers += [
        nn.Linear(hidden, out_dim),
        nn.SiLU(),
    ]
    return nn.Sequential(*layers)


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """
    Mean-pool x over dim=1 using boolean mask.

    Args:
        x:    (B, K, h)
        mask: (B, K) — True = valid

    Returns:
        (B, h)
    """
    x_masked = x.masked_fill(~mask.unsqueeze(-1), 0.0)
    denom    = mask.sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1)
    return x_masked.sum(dim=1) / denom                      # (B, h)


# ── Fourier Features ──────────────────────────────────────────────

class FourierFeatures(nn.Module):
    """
    Random Fourier Features for geometric position encoding.

    Projects input coordinates through sin/cos at learned frequencies,
    giving the network a richer geometric prior without hand-crafting
    positional encodings.

    Output dim = in_dim * num_bands * 2  (sin + cos per band per input)
    """

    def __init__(self, in_dim: int, num_bands: int) -> None:
        super().__init__()
        self.out_dim = in_dim * num_bands * 2
        # Frequencies are fixed (not learned) — standard RFF
        self.register_buffer(
            "freqs",
            torch.randn(in_dim, num_bands) * 2.0 * math.pi,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_dim)
        proj = x.unsqueeze(-1) * self.freqs          # (..., in_dim, num_bands)
        proj = proj.reshape(*x.shape[:-1], -1)       # (..., in_dim * num_bands)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (..., out_dim)


# ── Attention ─────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with fused QKV projection.
    Uses F.scaled_dot_product_attention for Flash Attention support.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.dropout   = dropout
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim,     bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        qkv     = self.qkv(x).split(D, dim=-1)
        q, k, v = [t.view(B, L, H, Dh).transpose(1, 2) for t in qkv]

        # Convert (B, L) bool mask → additive attention bias
        attn_bias = None
        if mask is not None:
            # (B, L) → (B, 1, 1, L): True=valid, False=masked
            attn_bias = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(
                ~mask[:, None, None, :], float("-inf")
            )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = attn_bias,
            dropout_p = self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block — more stable in RL than post-norm."""

    def __init__(
        self,
        dim:       int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        mlp_dim    = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = MultiHeadAttention(dim, num_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_dim, dim),
        )
        self.drop  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ── GRU Memory ────────────────────────────────────────────────────

class GRUMemory(nn.Module):
    """Single-step GRU for temporal memory across environment steps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(
        self,
        x:      Tensor,             # (B, h)
        hidden: Tensor | None,      # (B, h) or None
    ) -> tuple[Tensor, Tensor]:
        next_h = self.gru(x, hidden)
        return next_h, next_h


# ── Cross-Attention Pooling ───────────────────────────────────────

class CrossAttentionPooling(nn.Module):
    """
    Learned query attends over candidate set → single summary vector.

    Richer than mean-pooling: learns WHICH candidates matter for
    value estimation rather than treating all equally.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn  = MultiHeadAttention(dim, num_heads, dropout)
        self.norm  = nn.LayerNorm(dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x:    (B, K, h)
            mask: (B, K) bool — True = valid

        Returns:
            (B, h) — pooled summary
        """
        B = x.shape[0]
        q = self.query.expand(B, 1, -1)               # (B, 1, h)

        # Prepend learnable query token; mask its slot as always valid
        xq = torch.cat([q, x], dim=1)                 # (B, K+1, h)
        if mask is not None:
            mask_q = F.pad(mask, (1, 0), value=True)  # (B, K+1)
        else:
            mask_q = None

        out = self.attn(self.norm(xq), mask_q)        # (B, K+1, h)
        return out[:, 0, :]                            # (B, h) — query output


# ── Main Policy ───────────────────────────────────────────────────

class PlanetPolicy(nn.Module):
    """
    Planet targeting policy with:
      - Fourier geometry encoding
      - Per-stream MLP encoders (self / global / candidate)
      - Context injection (source planet + global state → each candidate)
      - Relational self-attention across candidates
      - Optional GRU temporal memory
      - Cross-attention value pooling

    Args:
        self_dim:        Feature dimension of source planet features.
        candidate_dim:   Feature dimension of each candidate planet.
        global_dim:      Feature dimension of global game state.
        candidate_count: Number of candidate targets (K).
        hidden_size:     Internal representation dimension (h).
        num_heads:       Attention heads. Must divide hidden_size.
        num_attn_layers: Number of transformer blocks.
        mlp_ratio:       FFN expansion ratio inside transformer blocks.
        dropout:         Dropout rate. Use 0.0 for RL (entropy handles exploration).
        use_memory:      Whether to use GRU temporal memory across steps.
        geom_indices:    Indices of geometric features in candidate_features.
        fourier_bands:   Frequency bands for Fourier geometry encoding.
    """

    def __init__(
        self,
        self_dim:        int,
        candidate_dim:   int,
        global_dim:      int,
        candidate_count: int,
        hidden_size:     int   = 128,
        num_heads:       int   = 4,
        num_attn_layers: int   = 2,
        mlp_ratio:       float = 2.0,
        dropout:         float = 0.0,
        use_memory:      bool  = False,      # BUG FIX: default False (opt-in)
        geom_indices:    tuple[int, ...] = (0, 1, 2, 3),
        fourier_bands:   int   = 4,
    ) -> None:
        super().__init__()

        self.candidate_count = candidate_count
        self.use_memory      = use_memory

        h        = hidden_size
        ctx_dim  = h * 3   # self_h + global_h + cand_h

        # --- Geometry encoder ------------------------------------
        self.geom_encoder = FourierFeatures(
            in_dim   = len(geom_indices),
            num_bands= fourier_bands,
        )
        geom_dim = self.geom_encoder.out_dim

        # Cache geom indices as buffer (avoids list() every forward)
        # BUG FIX: was list(geom_indices) allocated on every call
        self.register_buffer(
            "geom_idx",
            torch.tensor(list(geom_indices), dtype=torch.long),
            persistent=False,
        )

        # --- Encoders --------------------------------------------
        self.self_encoder      = _mlp(self_dim,                h, h, dropout)
        self.global_encoder    = _mlp(global_dim,              h, h, dropout)
        self.candidate_encoder = _mlp(candidate_dim + geom_dim, h, h, dropout)

        # --- Context projection ----------------------------------
        # BUG FIX: add LayerNorm to normalize before first attention layer
        self.context_proj = nn.Sequential(
            nn.Linear(ctx_dim, h),
            nn.SiLU(),
            nn.LayerNorm(h),
        )

        # --- Transformer layers ----------------------------------
        self.attn_layers = nn.ModuleList([
            TransformerBlock(h, num_heads, mlp_ratio, dropout)
            for _ in range(num_attn_layers)
        ])

        # --- Temporal memory (optional) --------------------------
        self.memory: GRUMemory | None = GRUMemory(h) if use_memory else None

        # --- Policy head -----------------------------------------
        self.target_head = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1, bias=False),
        )

        # --- Value pooling + head --------------------------------
        self.value_pool = CrossAttentionPooling(h, num_heads, dropout)
        self.value_head = nn.Sequential(
            # value_ctx + self_h + global_h — explicit skip connections
            # stabilize value learning in RL even though value_ctx
            # implicitly contains self/global information already.
            nn.Linear(ctx_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1, bias=False),  # BUG FIX: consistent bias=False
        )

        # BUG FIX: -1e9 → float("-inf") for correct masked softmax
        self._neg_inf = float("-inf")

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Orthogonal init on MLP/projection layers only.
        Specialized submodules (GRUMemory, FourierFeatures,
        MultiHeadAttention) keep their own default init.
        """
        # BUG FIX: only init direct MLP/projection children,
        # not specialized submodules which have their own init
        own_modules = [
            self.self_encoder,
            self.global_encoder,
            self.candidate_encoder,
            self.context_proj,
            self.target_head,
            self.value_head,
        ]
        for module in own_modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        self_features:      Tensor,         # (B, self_dim)
        candidate_features: Tensor,         # (B, K, candidate_dim)
        global_features:    Tensor,         # (B, global_dim)
        candidate_mask:     Tensor,         # (B, K) bool — True = valid
        hidden_state:       Tensor | None = None,  # (B, h)
    ) -> PolicyOutput:
        B, K, _ = candidate_features.shape

        # BUG FIX: validate hidden_state batch size
        if self.use_memory and hidden_state is not None:
            if hidden_state.shape[0] != B:
                raise ValueError(
                    f"hidden_state batch size {hidden_state.shape[0]} != "
                    f"input batch size {B}. Reset hidden state after env.reset()."
                )

        # BUG FIX: warn when memory is enabled but hidden_state is never passed
        if self.use_memory and hidden_state is None and self.training:
            warnings.warn(
                "use_memory=True but hidden_state=None — GRU resets every step. "
                "Pass hidden_state=prev_output.hidden_state for temporal memory.",
                UserWarning, stacklevel=2,
            )

        # --- Geometry encoding -----------------------------------
        geom   = candidate_features[..., self.geom_idx]  # BUG FIX: cached buffer
        geom   = self.geom_encoder(geom)                  # (B, K, geom_dim)

        # --- Encode each stream ----------------------------------
        self_h   = self.self_encoder(self_features)       # (B, h)
        global_h = self.global_encoder(global_features)   # (B, h)

        cand_in  = torch.cat([candidate_features, geom], dim=-1)  # (B, K, cand_dim+geom_dim)
        cand_h   = self.candidate_encoder(
            cand_in.reshape(B * K, -1)
        ).reshape(B, K, -1)                               # (B, K, h)

        # --- Inject context into candidates ----------------------
        self_exp   = self_h.unsqueeze(1).expand(B, K, -1)
        global_exp = global_h.unsqueeze(1).expand(B, K, -1)
        x = self.context_proj(
            torch.cat([self_exp, global_exp, cand_h], dim=-1)
        )                                                  # (B, K, h)

        # --- Relational attention --------------------------------
        for layer in self.attn_layers:
            x = layer(x, candidate_mask)                  # (B, K, h)

        # --- Temporal memory -------------------------------------
        next_hidden: Tensor | None = None
        if self.use_memory and self.memory is not None:
            pooled              = _masked_mean(x, candidate_mask)  # (B, h)
            memory_out, next_hidden = self.memory(pooled, hidden_state)
            x = x + memory_out.unsqueeze(1)               # (B, K, h) broadcast

        # --- Policy head -----------------------------------------
        logits = self.target_head(x).squeeze(-1)          # (B, K)
        logits = logits.masked_fill(~candidate_mask, self._neg_inf)

        # --- Value head ------------------------------------------
        value_ctx   = self.value_pool(x, candidate_mask)  # (B, h)
        value_input = torch.cat([value_ctx, self_h, global_h], dim=-1)
        value       = self.value_head(value_input).squeeze(-1)  # (B,)

        return PolicyOutput(
            target_logits = logits,
            value         = value,
            hidden_state  = next_hidden,
        )
