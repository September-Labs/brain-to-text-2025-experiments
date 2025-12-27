from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from b2txt.zipf import ZipfWeightLearner


class MEGConformerLayer(nn.Module):
    """Lightweight Conformer-style block for neural time series."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ff_dim = ff_dim or (2 * dim)

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
        )

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.ln1(x_conv + res)

        res = x
        attn_out, _ = self.attention(x, x, x)
        x = self.ln2(self.dropout(attn_out) + res)

        res = x
        x = self.ffn(x)
        x = self.ln3(x + res)
        return x


class LCSCTCZipf(nn.Module):
    """CTC encoder with session conditioning and Zipf-aware auxiliary heads."""

    def __init__(
        self,
        neural_dim: int,
        n_days: int,
        n_classes: int,
        hidden_dim: int = 256,
        num_conformers: int = 4,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        patch_size: int = 0,
        patch_stride: int = 1,
        aux_dim: int = 64,
        use_day_embedding: bool = True,
        use_zipf: bool = True,
        zipf_alpha: float = 0.99,
    ) -> None:
        super().__init__()
        self.neural_dim = int(neural_dim)
        self.n_days = int(n_days)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.aux_dim = int(aux_dim)
        self.use_day_embedding = bool(use_day_embedding)
        self.use_zipf = bool(use_zipf)

        if self.use_day_embedding:
            self.day_scale = nn.Embedding(self.n_days, self.neural_dim)
            self.day_bias = nn.Embedding(self.n_days, self.neural_dim)
            nn.init.zeros_(self.day_scale.weight)
            nn.init.zeros_(self.day_bias.weight)
            self.day_dropout = nn.Dropout(dropout_rate)

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
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conformers = nn.ModuleList(
            [
                MEGConformerLayer(hidden_dim, num_heads=num_heads, ff_dim=hidden_dim * 2, dropout=dropout_rate)
                for _ in range(num_conformers)
            ]
        )

        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.n_classes)
        self.aux_proj = nn.Linear(hidden_dim, self.aux_dim)
        self.label_embedding = nn.Embedding(self.n_classes, self.aux_dim)

        self.zipf_learner = None
        if self.use_zipf:
            self.zipf_learner = ZipfWeightLearner(self.n_classes, self.aux_dim, alpha=zipf_alpha)

    def forward(self, x: torch.Tensor, day_idx: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if self.use_day_embedding and day_idx is not None:
            scale = 1.0 + self.day_scale(day_idx).unsqueeze(1)
            bias = self.day_bias(day_idx).unsqueeze(1)
            x = self.day_dropout(x * scale + bias)

        x = x.transpose(1, 2)
        if self.patch_conv is not None:
            x = self.patch_conv(x)
        x = self.input_projection(x)
        x = x.transpose(1, 2)

        for layer in self.conformers:
            x = layer(x)

        x = self.feature_norm(x)
        logits = self.classifier(x)
        aux = self.aux_proj(x)

        return {"logits": logits, "aux": aux}
