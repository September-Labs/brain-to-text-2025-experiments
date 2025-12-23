from __future__ import annotations

from typing import Dict, Iterable

import torch


class BaseTask:
    """Base task interface for model training/evaluation."""

    def __init__(self, config: Dict[str, object], device: torch.device) -> None:
        self.config = config
        self.device = device

    def training_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        raise NotImplementedError

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        raise NotImplementedError

    def predict_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        raise NotImplementedError

    def init_validation_state(self) -> Dict[str, float]:
        return {}

    def accumulate_validation(self, state: Dict[str, float], batch_stats: Dict[str, float]) -> Dict[str, float]:
        for key, value in batch_stats.items():
            state[key] = state.get(key, 0.0) + float(value)
        return state

    def finalize_validation(self, state: Dict[str, float]) -> Dict[str, float]:
        return state

    def format_predictions(
        self,
        batch: Dict[str, torch.Tensor],
        predictions: Dict[str, object],
    ) -> Iterable[Dict[str, object]]:
        raise NotImplementedError
