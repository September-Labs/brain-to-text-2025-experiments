from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def prepare_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)
    return attention_mask


@torch.jit.script
def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int):
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos / mid)
            / torch.log(torch.tensor((max_position - 1) / mid))
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)

    q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


@torch.jit.script
def build_rpos(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    position_buckets: int,
    max_relative_positions: int,
):
    if key_layer.size(-2) != query_layer.size(-2):
        return build_relative_position(
            key_layer, key_layer, bucket_size=position_buckets, max_position=max_relative_positions
        )
    return relative_pos


class DisentangledSelfAttention(nn.Module):
    """DeBERTa-style disentangled self-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_bias: bool = True,
        pos_att_type: Iterable[str] = ("c2p", "p2c"),
        relative_attention: bool = True,
        position_buckets: int = -1,
        max_relative_positions: int = -1,
        share_att_key: bool = False,
        max_position_embeddings: Optional[int] = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size

        self.query_proj = nn.Linear(hidden_size, self.all_head_size, bias=attention_bias)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size, bias=attention_bias)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size, bias=attention_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)

        self.dropout_attn = nn.Dropout(attention_dropout) if attention_dropout > 0.0 else nn.Identity()
        self.dropout_out = nn.Dropout(hidden_dropout) if hidden_dropout > 0.0 else nn.Identity()
        self.pos_dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0.0 else nn.Identity()

        self.share_att_key = bool(share_att_key)
        self.pos_att_type = tuple(pos_att_type) if pos_att_type is not None else tuple()
        self.relative_attention = bool(relative_attention)

        self.position_buckets = int(position_buckets)
        self.max_relative_positions = int(max_relative_positions)
        if self.max_relative_positions < 1:
            self.max_relative_positions = int(max_position_embeddings or 512)

        self.pos_ebd_size = self.position_buckets if self.position_buckets > 0 else self.max_relative_positions

        if self.relative_attention:
            self.rel_embeddings = nn.Embedding(self.pos_ebd_size * 2, hidden_size)
            self.norm_rel_ebd = nn.LayerNorm(hidden_size)

        if self.relative_attention and not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and hasattr(self, "norm_rel_ebd"):
            rel_embeddings = self.norm_rel_ebd(rel_embeddings)
        return rel_embeddings

    def _shape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return (
            x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query_states is None:
            query_states = hidden_states

        batch_size, seq_len, _ = hidden_states.shape

        q = self._shape_qkv(self.query_proj(query_states))
        k = self._shape_qkv(self.key_proj(hidden_states))
        v = self._shape_qkv(self.value_proj(hidden_states))

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(q, scale_factor).to(dtype=q.dtype, device=q.device)
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / scale

        if self.relative_attention and (("c2p" in self.pos_att_type) or ("p2c" in self.pos_att_type)):
            rel_embeddings = self.get_rel_embedding()
            rel_att = self._disentangled_attention_bias(q, k, relative_pos, rel_embeddings)
            attn_scores = attn_scores + rel_att

        attn_scores = attn_scores.view(batch_size, self.num_heads, seq_len, seq_len)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask).to(attn_scores.dtype)
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_probs = self.dropout_attn(attn_probs)

        attn_probs_flat = attn_probs.view(batch_size * self.num_heads, seq_len, seq_len)
        ctx = torch.bmm(attn_probs_flat, v)
        ctx = (
            ctx.view(batch_size, self.num_heads, seq_len, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )

        ctx = self.out_proj(ctx)
        ctx = self.dropout_out(ctx)

        if output_attentions:
            return ctx, attn_probs
        return ctx, None

    def _disentangled_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: Optional[torch.Tensor],
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer, key_layer, bucket_size=self.position_buckets, max_position=self.max_relative_positions
            )
        elif relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0)
        elif relative_pos.dim() == 3:
            if relative_pos.size(0) != 1:
                relative_pos = relative_pos[:1]
        else:
            raise ValueError(f"relative_pos must have dim 2 or 3; got {relative_pos.dim()}")

        seq_len = query_layer.size(-2)
        att_span = self.pos_ebd_size

        rel_embeddings = rel_embeddings[: (att_span * 2), :].unsqueeze(0)

        if self.share_att_key:
            pos_query_layer = self._shape_qkv(self.query_proj(rel_embeddings))
            pos_key_layer = self._shape_qkv(self.key_proj(rel_embeddings))
        else:
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
            scale = scaled_size_sqrt(pos_key_layer, 2 if ("p2c" in self.pos_att_type) else 1).to(
                dtype=query_layer.dtype, device=query_layer.device
            )
            c2p = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p = torch.gather(c2p, dim=-1, index=c2p_pos.expand(query_layer.size(0), seq_len, seq_len))
            score = score + (c2p / scale)

        if "p2c" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_query_layer, 2 if ("c2p" in self.pos_att_type) else 1).to(
                dtype=query_layer.dtype, device=query_layer.device
            )
            r_pos = build_rpos(
                query_layer,
                key_layer,
                relative_pos,
                position_buckets=self.position_buckets,
                max_relative_positions=self.max_relative_positions,
            )
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c = torch.gather(p2c, dim=-1, index=p2c_pos.expand(key_layer.size(0), seq_len, seq_len)).transpose(
                -1, -2
            )
            score = score + (p2c / scale)

        return score


class MEGConformerLayer(nn.Module):
    """Conformer layer with pre-layer normalization and DeBERTa attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        ff_dim: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        norm_type: str = "pre",
    ) -> None:
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        self.norm_type = norm_type

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

        self.attention = DisentangledSelfAttention(
            hidden_size=dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            relative_attention=True,
            position_buckets=32,
            max_relative_positions=128,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "pre":
            res = x
            x_norm = self.ln1(x)
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = res + self.dropout(x_conv)

            res = x
            x_norm = self.ln2(x)
            attn_out, _ = self.attention(x_norm)
            x = res + self.dropout(attn_out)

            res = x
            x_norm = self.ln3(x)
            ff_out = self.ffn(x_norm)
            x = res + ff_out
        else:
            res = x
            x_conv = x.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = self.ln1(x_conv + res)

            res = x
            attn_out, _ = self.attention(x)
            x = self.ln2(self.dropout(attn_out) + res)

            res = x
            x = self.ffn(x)
            x = self.ln3(x + res)

        return x


class DeMEGaCTC(nn.Module):
    """DeMEGa-style encoder with CTC heads and day-specific input layers."""

    def __init__(
        self,
        neural_dim: int,
        n_days: int,
        n_classes: int,
        hidden_dim: int = 128,
        num_conformers: int = 4,
        num_heads: int = 4,
        dropout_rate: float = 0.5,
        norm_type: str = "pre",
        patch_size: int = 14,
        patch_stride: int = 4,
        input_dropout: float = 0.2,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        use_contrastive: bool = True,
        use_ipa_features: bool = True,
        ipa_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_contrastive = use_contrastive
        self.use_ipa_features = use_ipa_features

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

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

        self.input_projection = nn.Sequential(
            nn.Conv1d(self.neural_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv1d(self.neural_dim, hidden_dim, kernel_size=1)

        self.meg_encoder = nn.ModuleList(
            [
                MEGConformerLayer(hidden_dim, num_heads, hidden_dim * 2, dropout=dropout_rate, norm_type=norm_type)
                for _ in range(num_conformers)
            ]
        )

        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.frame_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

        if self.use_ipa_features:
            self.ipa_predictor = nn.Sequential(
                nn.Linear(embedding_dim, ipa_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(ipa_hidden_dim, 14),
            )
        else:
            self.ipa_predictor = None

        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, projection_dim),
            )
        else:
            self.projection_head = None

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
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

        for layer in self.meg_encoder:
            x = layer(x)

        x = self.feature_norm(x)
        logits = self.classifier(x)

        output: dict[str, torch.Tensor | None] = {"logits": logits}

        if self.use_ipa_features or self.use_contrastive:
            embeddings = self.frame_proj(x)
            output["embeddings"] = embeddings
            if self.use_ipa_features:
                output["ipa_logits"] = self.ipa_predictor(embeddings)
            if self.use_contrastive:
                output["proj"] = self.projection_head(embeddings)

        return output
