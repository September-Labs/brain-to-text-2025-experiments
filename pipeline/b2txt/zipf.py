from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZipfWeightLearner(nn.Module):
    """Learn Zipf priors and MEG feature prototypes for phonemes."""

    def __init__(self, vocab_size: int, feature_dim: int, alpha: float = 0.99) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)

        self.register_buffer("phoneme_counts", torch.ones(self.vocab_size))
        self.register_buffer("total_count", torch.tensor(float(self.vocab_size)))

        self.register_buffer("meg_prototypes", torch.zeros(self.vocab_size, self.feature_dim))
        self.register_buffer("prototype_counts", torch.ones(self.vocab_size))

        self.zipf_s = nn.Parameter(torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.phoneme_meg_attention = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def update_statistics(self, phonemes: torch.Tensor, features: torch.Tensor) -> None:
        """EMA updates for phoneme counts and prototypes."""
        if phonemes.numel() == 0:
            return
        phonemes = phonemes.to(dtype=torch.long)
        features = features.to(dtype=self.meg_prototypes.dtype)

        for phoneme in phonemes.tolist():
            self.phoneme_counts[phoneme] = self.alpha * self.phoneme_counts[phoneme] + (1.0 - self.alpha)
            self.total_count = self.alpha * self.total_count + (1.0 - self.alpha)

        for phoneme, feat in zip(phonemes.tolist(), features):
            old = self.meg_prototypes[phoneme]
            self.meg_prototypes[phoneme] = self.alpha * old + (1.0 - self.alpha) * feat
            self.prototype_counts[phoneme] += 1.0

    def get_zipf_weights(self) -> torch.Tensor:
        frequencies = self.phoneme_counts / self.total_count.clamp_min(1.0)
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(sorted_freqs)
        ranks[sorted_indices] = torch.arange(
            1, self.vocab_size + 1, dtype=frequencies.dtype, device=frequencies.device
        )
        zipf_weights = 1.0 / torch.pow(ranks.clamp_min(1.0), self.zipf_s)
        return zipf_weights / zipf_weights.sum().clamp_min(1e-8)

    def compute_similarity(self, features: torch.Tensor) -> torch.Tensor:
        norm_prototypes = F.normalize(self.meg_prototypes, p=2, dim=1)
        norm_features = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(norm_features, norm_prototypes.T)
        return similarity / self.temperature.clamp_min(1e-4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return Zipf + MEG weighted priors for a batch of features."""
        batch_size = features.size(0)
        zipf_priors = self.get_zipf_weights().unsqueeze(0).expand(batch_size, -1)
        meg_similarity = self.compute_similarity(features)

        combined = torch.cat(
            [
                features.unsqueeze(1).expand(-1, self.vocab_size, -1),
                self.meg_prototypes.unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        attn = self.phoneme_meg_attention(combined.reshape(batch_size * self.vocab_size, -1))
        attn = attn.reshape(batch_size, self.vocab_size)

        weights = zipf_priors * (1.0 + meg_similarity) * attn
        return F.softmax(weights, dim=-1)
