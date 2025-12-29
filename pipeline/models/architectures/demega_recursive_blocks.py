from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from models.architectures.demega_ctc import DisentangledSelfAttention


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
        position_buckets: int = 32,
        max_relative_positions: int = 128,
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
            position_buckets=position_buckets,
            max_relative_positions=max_relative_positions,
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


class RNNExpert(nn.Module):
    """Single RNN expert that can be LSTM, GRU, or vanilla RNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_dim * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return output


class MoERNN(nn.Module):
    """Mixture of Experts layer with RNN experts."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        rnn_type: str = "lstm",
        rnn_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_weight = load_balance_weight

        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, num_experts),
        )

        self.experts = nn.ModuleList(
            [
                RNNExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    num_layers=rnn_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
                for _ in range(num_experts)
            ]
        )

        expert_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(expert_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        router_input = x.mean(dim=1)
        router_logits = self.router(router_input)

        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        load_balance_loss = self._compute_load_balance_loss(router_probs)

        combined_output = torch.zeros(
            batch_size,
            seq_len,
            self.experts[0].output_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]

            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    weights = expert_weights[mask].unsqueeze(-1).unsqueeze(-1)
                    combined_output[mask] += weights * expert_output

        output = self.output_projection(combined_output)
        output = self.dropout(output)

        return output, load_balance_loss

    def _compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        expert_usage = router_probs.mean(dim=0)
        target = torch.ones_like(expert_usage) / self.num_experts
        return self.load_balance_weight * F.mse_loss(expert_usage, target)
