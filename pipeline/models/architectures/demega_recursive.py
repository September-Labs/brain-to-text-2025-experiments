from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.architectures.demega_ctc import (
    DisentangledSelfAttention,
    prepare_attention_mask,
    make_log_bucket_position,
)
from models.architectures.demega_recursive_blocks import MEGConformerLayer, MoERNN


class RotaryEmbedding(nn.Module):
    """
    RoPE embedding optimized for small sequences.
    Uses smaller base for better position resolution on short sequences.
    """
    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        base: float = 1000.0,  # Smaller base for small sequences
        attention_scaling: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires even head_dim; got {dim}")
        self.dim = dim
        self.base = float(base)
        self.attention_scaling = float(attention_scaling)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, 
        x_like: torch.Tensor, 
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_like: tensor to infer dtype/device
            position_ids: [bs, seq] absolute positions
        Returns:
            cos, sin: [bs, seq, dim]
        """
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        
        device_type = x_like.device.type if x_like.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv.float() @ pos.float()).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        
        return cos.to(dtype=x_like.dtype, device=x_like.device), sin.to(dtype=x_like.dtype, device=x_like.device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim halves: (x1, x2) -> (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to tensor x. Shapes: x [bs, seq, dim], cos/sin [bs, seq, dim]."""
    return (x * cos) + (rotate_half(x) * sin)


class DisentangledCrossAttention(nn.Module):
    """
    DeBERTa-style disentangled cross-attention.
    
    Extends disentangled attention to cross-attention scenarios where
    queries come from decoder and keys/values come from encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        pos_att_type: tuple[str, ...] = ("c2p", "p2c"),
        position_buckets: int = 32,
        max_relative_positions: int = 128,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.pos_att_type = pos_att_type

        self.query_proj = nn.Linear(hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout_attn = nn.Dropout(attention_dropout) if attention_dropout > 0.0 else nn.Identity()
        self.dropout_out = nn.Dropout(hidden_dropout) if hidden_dropout > 0.0 else nn.Identity()

        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.pos_ebd_size = position_buckets if position_buckets > 0 else max_relative_positions

        # Relative position embeddings
        self.rel_embeddings = nn.Embedding(self.pos_ebd_size * 2, hidden_size)
        self.norm_rel_ebd = nn.LayerNorm(hidden_size)

        # Position projections for disentangled attention
        if "c2p" in self.pos_att_type:
            self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size)
        if "p2c" in self.pos_att_type:
            self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size)

    def get_rel_embedding(self) -> torch.Tensor:
        rel_embeddings = self.rel_embeddings.weight
        return self.norm_rel_ebd(rel_embeddings)

    def _shape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return (
            x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        )

    def _build_cross_relative_position(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build relative position matrix for cross-attention."""
        q_ids = torch.arange(query_len, dtype=torch.long, device=device)
        k_ids = torch.arange(key_len, dtype=torch.long, device=device)
        rel_pos_ids = q_ids[:, None] - k_ids[None, :]
        
        if self.position_buckets > 0:
            rel_pos_ids = make_log_bucket_position(
                rel_pos_ids, self.position_buckets, self.max_relative_positions
            )
        
        return rel_pos_ids.unsqueeze(0).to(torch.long)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_states: [B, T_q, D] decoder states
            key_value_states: [B, T_kv, D] encoder states
            attention_mask: [B, T_kv] mask for encoder (True = masked)
        Returns:
            output: [B, T_q, D]
        """
        batch_size, query_len, _ = query_states.shape
        _, key_len, _ = key_value_states.shape

        q = self._shape_qkv(self.query_proj(query_states))
        k = self._shape_qkv(self.key_proj(key_value_states))
        v = self._shape_qkv(self.value_proj(key_value_states))

        # Compute scale factor
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(self.head_dim * scale_factor, dtype=q.dtype, device=q.device))

        # Content-to-content attention
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / scale

        # Add disentangled attention bias
        rel_embeddings = self.get_rel_embedding()
        relative_pos = self._build_cross_relative_position(query_len, key_len, q.device)
        rel_att = self._disentangled_cross_attention_bias(q, k, relative_pos, rel_embeddings)
        attn_scores = attn_scores + rel_att

        # Reshape for mask application
        attn_scores = attn_scores.view(batch_size, self.num_heads, query_len, key_len)

        if attention_mask is not None:
            # attention_mask: [B, T_kv] -> [B, 1, 1, T_kv]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_probs = self.dropout_attn(attn_probs)

        attn_probs_flat = attn_probs.view(batch_size * self.num_heads, query_len, key_len)
        ctx = torch.bmm(attn_probs_flat, v)
        ctx = (
            ctx.view(batch_size, self.num_heads, query_len, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, query_len, self.all_head_size)
        )

        ctx = self.out_proj(ctx)
        ctx = self.dropout_out(ctx)

        return ctx

    def _disentangled_cross_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute disentangled attention bias for cross-attention."""
        query_len = query_layer.size(-2)
        key_len = key_layer.size(-2)
        att_span = self.pos_ebd_size

        rel_embeddings = rel_embeddings[: (att_span * 2), :].unsqueeze(0)

        if "c2p" in self.pos_att_type:
            pos_key_layer = self._shape_qkv(self.pos_key_proj(rel_embeddings))
        if "p2c" in self.pos_att_type:
            pos_query_layer = self._shape_qkv(self.pos_query_proj(rel_embeddings))

        repeat_factor = query_layer.size(0) // self.num_heads
        if "c2p" in self.pos_att_type:
            pos_key_layer = pos_key_layer.repeat(repeat_factor, 1, 1)
        if "p2c" in self.pos_att_type:
            pos_query_layer = pos_query_layer.repeat(repeat_factor, 1, 1)

        score = 0.0

        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(
                torch.tensor(
                    self.head_dim * (2 if "p2c" in self.pos_att_type else 1),
                    dtype=query_layer.dtype,
                    device=query_layer.device,
                )
            )
            c2p = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p = torch.gather(
                c2p, dim=-1,
                index=c2p_pos.expand(query_layer.size(0), query_len, key_len)
            )
            score = score + (c2p / scale)

        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(
                torch.tensor(
                    self.head_dim * (2 if "c2p" in self.pos_att_type else 1),
                    dtype=query_layer.dtype,
                    device=query_layer.device,
                )
            )
            p2c_pos = torch.clamp(-relative_pos + att_span, 0, att_span * 2 - 1)
            p2c = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c = torch.gather(
                p2c, dim=-1,
                index=p2c_pos.expand(key_layer.size(0), key_len, query_len)
            ).transpose(-1, -2)
            score = score + (p2c / scale)

        return score


class DisentangledCrossAttentionLayer(nn.Module):
    """Cross-attention layer with disentangled attention mechanism."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        position_buckets: int = 32,
        max_relative_positions: int = 128,
        pos_att_type: tuple[str, ...] = ("c2p", "p2c"),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Self-attention using DisentangledSelfAttention
        self.self_attention = DisentangledSelfAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            relative_attention=True,
            position_buckets=position_buckets,
            max_relative_positions=max_relative_positions,
            pos_att_type=pos_att_type,
        )

        # Cross-attention using DisentangledCrossAttention
        self.cross_attention = DisentangledCrossAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            position_buckets=position_buckets,
            max_relative_positions=max_relative_positions,
            pos_att_type=pos_att_type,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        query_positions: Optional[torch.LongTensor] = None,  # Not used, kept for API compatibility
        encoder_positions: Optional[torch.LongTensor] = None,  # Not used, kept for API compatibility
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # === Self-attention ===
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attention(x_norm)
        x = residual + self.dropout(attn_out)

        # === Cross-attention ===
        residual = x
        x_norm = self.norm2(x)
        cross_out = self.cross_attention(x_norm, encoder_out, encoder_mask)
        x = residual + self.dropout(cross_out)

        # === FFN ===
        residual = x
        x = residual + self.ffn(self.norm3(x))

        return x


class RoPECrossAttentionLayer(nn.Module):
    """Cross-attention layer with RoPE for queries and keys."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 1000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Self-attention projections
        self.self_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-attention projections
        self.cross_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # RoPE for self-attention and cross-attention
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, T, D] -> [B, H, T, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_from_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, H, T, head_dim] -> [B, T, D]"""
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        query_positions: torch.LongTensor,
        encoder_positions: torch.LongTensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_q, _ = x.shape
        _, T_kv, _ = encoder_out.shape
        
        # === Self-attention with RoPE ===
        residual = x
        x_norm = self.norm1(x)
        
        q = self._reshape_for_attention(self.self_q_proj(x_norm))  # [B, H, T, head_dim]
        k = self._reshape_for_attention(self.self_k_proj(x_norm))
        v = self._reshape_for_attention(self.self_v_proj(x_norm))
        
        # Apply RoPE to Q and K (need to reshape for RoPE which expects [B, T, dim])
        cos, sin = self.rope(x, query_positions)  # [B, T, head_dim]
        cos = cos.unsqueeze(1)  # [B, 1, T, head_dim]
        sin = sin.unsqueeze(1)
        
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        
        attn_out = self._reshape_from_attention(attn_out)
        x = residual + self.dropout(self.self_out_proj(attn_out))
        
        # === Cross-attention with RoPE ===
        residual = x
        x_norm = self.norm2(x)
        
        q = self._reshape_for_attention(self.cross_q_proj(x_norm))
        k = self._reshape_for_attention(self.cross_k_proj(encoder_out))
        v = self._reshape_for_attention(self.cross_v_proj(encoder_out))
        
        # RoPE for query positions
        cos_q, sin_q = self.rope(x, query_positions)
        cos_q = cos_q.unsqueeze(1)
        sin_q = sin_q.unsqueeze(1)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        
        # RoPE for encoder positions
        cos_k, sin_k = self.rope(encoder_out, encoder_positions)
        cos_k = cos_k.unsqueeze(1)
        sin_k = sin_k.unsqueeze(1)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if encoder_mask is not None:
            # encoder_mask: [B, T_kv] -> [B, 1, 1, T_kv]
            attn_weights = attn_weights.masked_fill(
                encoder_mask.unsqueeze(1).unsqueeze(2), 
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        
        attn_out = self._reshape_from_attention(attn_out)
        x = residual + self.dropout(self.cross_out_proj(attn_out))
        
        # === FFN ===
        residual = x
        x = residual + self.ffn(self.norm3(x))
        
        return x


class RefinementDecoder(nn.Module):
    """
    Refinement decoder supporting multiple attention mechanisms.
    
    Supports:
    - "rope": RoPE-based attention (default)
    - "disentangled": DeBERTa-style disentangled attention
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_type: str = "rope",
        # RoPE parameters
        rope_base: float = 1000.0,
        # Disentangled attention parameters
        position_buckets: int = 32,
        max_relative_positions: int = 128,
        pos_att_type: tuple[str, ...] = ("c2p", "p2c"),
    ):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == "rope":
            self.layers = nn.ModuleList([
                RoPECrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ])
        elif attention_type == "disentangled":
            self.layers = nn.ModuleList([
                DisentangledCrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    position_buckets=position_buckets,
                    max_relative_positions=max_relative_positions,
                    pos_att_type=pos_att_type,
                )
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}. Must be 'rope' or 'disentangled'.")
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        query_positions: torch.LongTensor,
        encoder_positions: torch.LongTensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, query_positions, encoder_positions, encoder_mask)
        return self.norm(x)


# Keep the old class name for backward compatibility
class RoPERefinementDecoder(RefinementDecoder):
    """Backward compatible alias for RefinementDecoder with RoPE attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        rope_base: float = 1000.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention_type="rope",
            rope_base=rope_base,
        )


class IterativeRefinementCTC(nn.Module):
    """
    CTC with iterative refinement supporting multiple attention mechanisms.
    
    Attention type options:
    - "rope": RoPE-based attention
        - No maximum sequence length limitation
        - Better extrapolation to unseen lengths
        - Relative position information encoded naturally
        - Works well with small sequences when using smaller base
    - "disentangled": DeBERTa-style disentangled attention
        - Separates content and position attention
        - Content-to-position (c2p) and position-to-content (p2c) components
        - Uses log-bucketed relative positions for efficiency
    """

    def __init__(
        self,
        neural_dim: int,
        n_days: int,
        n_classes: int,
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        num_refinement_layers: int = 2,
        num_iterations: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.3,
        norm_type: str = "pre",
        patch_size: int = 16,
        patch_stride: int = 12,
        input_dropout: float = 0.2,
        position_buckets: int = 32,
        max_relative_positions: int = 128,
        # Attention type selection
        refinement_attention_type: str = "rope",  # "rope" or "disentangled"
        # RoPE parameters
        rope_base: float = 1000.0,  # Smaller base for small sequences
        # Disentangled attention parameters
        pos_att_type: tuple[str, ...] = ("c2p", "p2c"),
        # MoE RNN parameters
        use_moe_rnn: bool = True,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_rnn_type: str = "lstm",
        moe_rnn_layers: int = 1,
        moe_rnn_hidden: Optional[int] = None,
        moe_bidirectional: bool = True,
        moe_load_balance_weight: float = 0.01,
        # Refinement options
        use_confidence_weighting: bool = True,
        refinement_loss_weight: float = 1.0,
        iteration_loss_decay: float = 0.8,
        **kwargs
    ) -> None:
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_iterations = num_iterations
        self.use_moe_rnn = use_moe_rnn
        self.use_confidence_weighting = use_confidence_weighting
        self.refinement_loss_weight = refinement_loss_weight
        self.iteration_loss_decay = iteration_loss_decay
        self.refinement_attention_type = refinement_attention_type

        # Day-specific input layers
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # Patch convolution
        if self.patch_size > 0 and self.patch_stride > 0:
            self.patch_conv = nn.Conv1d(
                self.neural_dim,
                self.neural_dim,
                kernel_size=self.patch_size,
                stride=self.patch_stride,
                padding=0,
                groups=self.neural_dim,
            )
        else:
            self.patch_conv = None

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Conv1d(self.neural_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv1d(self.neural_dim, hidden_dim, kernel_size=1)

        # Encoder
        self.encoder = nn.ModuleList([
            MEGConformerLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 2,
                dropout=dropout_rate,
                norm_type=norm_type,
                position_buckets=position_buckets,
                max_relative_positions=max_relative_positions,
            )
            for _ in range(num_encoder_layers)
        ])

        # MoE RNN
        if self.use_moe_rnn:
            moe_rnn_hidden = moe_rnn_hidden or hidden_dim
            self.moe_rnn = MoERNN(
                input_dim=hidden_dim,
                hidden_dim=moe_rnn_hidden,
                output_dim=hidden_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                rnn_type=moe_rnn_type,
                rnn_layers=moe_rnn_layers,
                dropout=dropout_rate,
                bidirectional=moe_bidirectional,
                load_balance_weight=moe_load_balance_weight,
            )
            self.moe_residual_gate = nn.Parameter(torch.zeros(1))
        else:
            self.moe_rnn = None

        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Phoneme embeddings (no positional - using RoPE or disentangled)
        self.phoneme_embeddings = nn.Embedding(n_classes + 1, hidden_dim)
        self.latent_token_id = n_classes

        # Confidence embedding
        if use_confidence_weighting:
            self.confidence_proj = nn.Linear(1, hidden_dim)

        # Refinement decoder with selectable attention type
        self.refinement_decoder = RefinementDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_refinement_layers,
            num_heads=num_heads,
            dropout=dropout_rate,
            attention_type=refinement_attention_type,
            rope_base=rope_base,
            position_buckets=position_buckets,
            max_relative_positions=max_relative_positions,
            pos_att_type=pos_att_type,
        )

        # Classifiers
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # Iteration-specific layer norms
        self.iteration_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_iterations)
        ])

    def encode(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode input features."""
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)

        x = x.transpose(1, 2)
        if self.patch_conv is not None:
            x = self.patch_conv(x)

        x_proj = self.input_projection(x)
        x_skip = self.input_skip(x)
        x = (x_proj + x_skip).transpose(1, 2)

        for layer in self.encoder:
            x = layer(x)

        load_balance_loss = None
        if self.use_moe_rnn and self.moe_rnn is not None:
            moe_out, load_balance_loss = self.moe_rnn(x)
            gate = torch.sigmoid(self.moe_residual_gate)
            x = gate * moe_out + (1 - gate) * x

        x = self.encoder_norm(x)
        return x, load_balance_loss

    def refine_step(
        self,
        queries: torch.Tensor,
        encoder_out: torch.Tensor,
        iteration: int,
        query_positions: torch.LongTensor,
        encoder_positions: torch.LongTensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single refinement step."""
        queries = self.iteration_norms[iteration](queries)

        refined = self.refinement_decoder(
            queries, 
            encoder_out, 
            query_positions,
            encoder_positions,
            encoder_mask,
        )

        logits = self.classifier(refined)
        return refined, logits

    def initialize_queries(
        self,
        encoder_out: torch.Tensor,
        initial_logits: torch.Tensor,
        use_predictions: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Initialize refinement queries using soft embeddings for gradient flow."""
        batch_size, seq_len, _ = encoder_out.shape
        device = encoder_out.device

        if use_predictions:
            # Use softmax (not argmax) to maintain differentiability
            probs = F.softmax(initial_logits / temperature, dim=-1)
            
            # Soft embeddings: weighted sum of phoneme embeddings
            # [B, T, n_classes] @ [n_classes, hidden_dim] -> [B, T, hidden_dim]
            phoneme_weights = self.phoneme_embeddings.weight[:self.n_classes]
            token_emb = torch.matmul(probs, phoneme_weights)

            if self.use_confidence_weighting:
                confidence = probs.max(dim=-1).values
                conf_emb = self.confidence_proj(confidence.unsqueeze(-1))
                token_emb = token_emb + conf_emb
        else:
            latent_ids = torch.full(
                (batch_size, seq_len),
                self.latent_token_id,
                device=device,
                dtype=torch.long,
            )
            token_emb = self.phoneme_embeddings(latent_ids)

        return token_emb

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        return_all_iterations: bool = False,
        temperature: float = 1.0,  # Add temperature for sharpening during inference
    ) -> dict[str, torch.Tensor | list | None]:
        # Encode
        encoder_out, load_balance_loss = self.encode(x, day_idx)
        batch_size, seq_len, _ = encoder_out.shape
        device = x.device

        # Create position indices for attention
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Initial CTC prediction
        initial_logits = self.classifier(encoder_out)

        # Initialize queries with soft embeddings
        queries = self.initialize_queries(
            encoder_out, initial_logits, 
            use_predictions=True, 
            temperature=temperature
        )

        all_logits = [initial_logits]

        # Iterative refinement
        for i in range(self.num_iterations):
            refined, logits = self.refine_step(
                queries, 
                encoder_out, 
                i,
                query_positions=positions,
                encoder_positions=positions,
            )
            all_logits.append(logits)

            if i < self.num_iterations - 1:
                # CRITICAL: Use soft embeddings to maintain gradient flow
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Soft embedding lookup (differentiable)
                phoneme_weights = self.phoneme_embeddings.weight[:self.n_classes]
                token_emb = torch.matmul(probs, phoneme_weights)

                if self.use_confidence_weighting:
                    confidence = probs.max(dim=-1).values
                    conf_emb = self.confidence_proj(confidence.unsqueeze(-1))
                    token_emb = token_emb + conf_emb

                # Residual connection
                queries = 0.5 * token_emb + 0.5 * refined

        output = {
            "logits": all_logits[-1],
            "initial_logits": initial_logits,
            "load_balance_loss": load_balance_loss,
        }

        if return_all_iterations:
            output["all_logits"] = all_logits

        return output

    def compute_loss(
        self,
        output: dict,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
        blank_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute CTC loss with iteration weighting."""
        ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        all_logits = output.get("all_logits", [output["initial_logits"], output["logits"]])

        total_loss = 0.0
        losses = {}

        for i, logits in enumerate(all_logits):
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

            max_len = logits.shape[1]
            clamped_lengths = torch.clamp(input_lengths, min=1, max=max_len)

            loss = ctc_loss_fn(log_probs, labels, clamped_lengths, label_lengths)
            loss = loss.mean()

            if i == 0:
                weight = 0.5
            else:
                weight = self.refinement_loss_weight * (
                    self.iteration_loss_decay ** (len(all_logits) - 1 - i)
                )

            total_loss += weight * loss
            losses[f"loss_iter_{i}"] = loss.detach()

        if output.get("load_balance_loss") is not None:
            total_loss += output["load_balance_loss"]
            losses["load_balance_loss"] = output["load_balance_loss"].detach()

        losses["total_loss"] = total_loss

        return losses
