from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Dict, Iterable, List

import torch

from b2txt.augmentations import apply_transforms
from b2txt.constants import LOGIT_TO_PHONEME
from b2txt.ctc import compute_output_lengths, greedy_decode, edit_distance
from b2txt.metrics import word_error_stats
from b2txt.tasks.base import BaseTask


class CTCPhonemeTask(BaseTask):
    """CTC phoneme decoding task for brain-to-text data."""

    def __init__(self, config: Dict[str, object], device: torch.device) -> None:
        super().__init__(config, device)
        self.data_cfg = config["data"]
        self.model_cfg = config["model"]
        self.ctc_cfg = config.get("ctc", {})
        self.blank_id = int(self.ctc_cfg.get("blank_id", 0))
        self.loss_fn = torch.nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        self.wer_cfg = config.get("competition", {}).get("wer", {})
        self.wer_enabled = bool(self.wer_cfg.get("enabled", False))
        self.wer_strict = bool(self.wer_cfg.get("strict", False))
        self.wer_max_trials = self.wer_cfg.get("max_trials")
        self.decode_predictions = bool(self.wer_cfg.get("decode_predictions", False))
        self.decoder = None

        if self.wer_enabled:
            decoder_cfg = self.wer_cfg.get("decoder")
            if not decoder_cfg:
                msg = "WER enabled but no decoder configured."
                if self.wer_strict:
                    raise ValueError(msg)
                warnings.warn(msg)
                self.wer_enabled = False
            else:
                try:
                    module = importlib.import_module(decoder_cfg["module"])
                    decoder_cls = getattr(module, decoder_cfg["class"])
                    params = decoder_cfg.get("params", {})
                    self.decoder = decoder_cls(**params)
                except Exception as exc:
                    msg = f"Failed to initialize decoder: {exc}"
                    if self.wer_strict:
                        raise
                    warnings.warn(msg)
                    self.wer_enabled = False

    def forward_model(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        day_indices: torch.Tensor,
    ) -> torch.Tensor:
        signature = inspect.signature(model.forward)
        if "day_idx" in signature.parameters:
            return model(features, day_indices)
        return model(features)

    def _prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        features = batch["input_features"].to(self.device)
        n_time_steps = batch["n_time_steps"].to(self.device)
        day_indices = batch["day_indicies"].to(self.device)

        features, n_time_steps = apply_transforms(
            features,
            n_time_steps,
            self.data_cfg.get("transforms", {}),
            self.device,
            mode=mode,
        )

        output_lengths = compute_output_lengths(n_time_steps, self.model_cfg, self.ctc_cfg)
        output_lengths = torch.clamp(output_lengths, min=1)

        return {
            "features": features,
            "n_time_steps": n_time_steps,
            "output_lengths": output_lengths,
            "day_indices": day_indices,
        }

    def training_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="train")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        logits = self.forward_model(model, prepared["features"], prepared["day_indices"])
        max_len = logits.shape[1]
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=max_len)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        loss = self.loss_fn(
            log_probs,
            labels,
            output_lengths,
            phone_seq_lens,
        )
        loss = torch.mean(loss)

        return {"loss": loss}

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        prepared = self._prepare_batch(batch, mode="eval")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        logits = self.forward_model(model, prepared["features"], prepared["day_indices"])
        max_len = logits.shape[1]
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=max_len)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        loss = self.loss_fn(
            log_probs,
            labels,
            output_lengths,
            phone_seq_lens,
        )
        loss = torch.mean(loss)

        per_edits = 0
        per_targets = 0
        preds = greedy_decode(logits, blank_id=self.blank_id)
        for idx, pred in enumerate(preds):
            target_len = int(phone_seq_lens[idx].item())
            if target_len == 0:
                continue
            target_seq = labels[idx, :target_len].tolist()
            per_edits += edit_distance(pred, target_seq)
            per_targets += target_len

        wer_edits = 0
        wer_words = 0
        if self.wer_enabled and self.decoder and "sentence_labels" in batch:
            logits_np = logits.detach().float().cpu().numpy()
            trial_count = len(preds)
            if self._wer_remaining is not None:
                trial_count = min(trial_count, self._wer_remaining)
            for idx in range(trial_count):
                pred_sentence = self.decoder.decode_logits(logits_np[idx])
                true_sentence = batch["sentence_labels"][idx]
                edits, words = word_error_stats(true_sentence, pred_sentence)
                wer_edits += edits
                wer_words += words
            if self._wer_remaining is not None:
                self._wer_remaining -= trial_count

        return {
            "loss_sum": float(loss.item()),
            "count": 1.0,
            "per_edits": float(per_edits),
            "per_targets": float(per_targets),
            "wer_edits": float(wer_edits),
            "wer_words": float(wer_words),
        }

    def init_validation_state(self) -> Dict[str, float]:
        if self.wer_max_trials is None:
            self._wer_remaining = None
        else:
            self._wer_remaining = int(self.wer_max_trials)
        return {
            "loss_sum": 0.0,
            "count": 0.0,
            "per_edits": 0.0,
            "per_targets": 0.0,
            "wer_edits": 0.0,
            "wer_words": 0.0,
        }

    def accumulate_validation(self, state: Dict[str, float], batch_stats: Dict[str, float]) -> Dict[str, float]:
        state["loss_sum"] += batch_stats.get("loss_sum", 0.0)
        state["count"] += batch_stats.get("count", 0.0)
        state["per_edits"] += batch_stats.get("per_edits", 0.0)
        state["per_targets"] += batch_stats.get("per_targets", 0.0)
        state["wer_edits"] += batch_stats.get("wer_edits", 0.0)
        state["wer_words"] += batch_stats.get("wer_words", 0.0)
        return state

    def finalize_validation(self, state: Dict[str, float]) -> Dict[str, float]:
        avg_loss = state["loss_sum"] / max(1.0, state["count"])
        per = None
        if state["per_targets"] > 0:
            per = state["per_edits"] / state["per_targets"]
        metrics = {"val_loss": avg_loss, "val_per": per}
        if state["wer_words"] > 0:
            metrics["val_wer"] = state["wer_edits"] / state["wer_words"]
        return metrics

    def predict_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="eval")
        logits = self.forward_model(model, prepared["features"], prepared["day_indices"])
        preds = greedy_decode(logits, blank_id=self.blank_id)
        output: Dict[str, object] = {"pred_ids": preds}
        if self.decoder and self.decode_predictions:
            logits_np = logits.detach().float().cpu().numpy()
            output["pred_sentences"] = [
                self.decoder.decode_logits(logits_np[i]) for i in range(len(preds))
            ]
        return output

    def format_predictions(
        self,
        batch: Dict[str, torch.Tensor],
        predictions: Dict[str, object],
    ) -> Iterable[Dict[str, object]]:
        day_indices = batch["day_indicies"].tolist()
        block_nums = batch["block_nums"].tolist()
        trial_nums = batch["trial_nums"].tolist()
        pred_ids = predictions["pred_ids"]
        pred_sentences = predictions.get("pred_sentences")

        sessions = self.data_cfg["sessions"]
        transcriptions = None
        if batch.get("transcriptions") is not None:
            transcriptions = [self._decode_transcription(x) for x in batch["transcriptions"]]

        for idx, seq in enumerate(pred_ids):
            entry = {
                "session": sessions[int(day_indices[idx])],
                "block_num": int(block_nums[idx]),
                "trial_num": int(trial_nums[idx]),
                "pred_phonemes": " ".join(LOGIT_TO_PHONEME[token] for token in seq),
            }
            if transcriptions:
                entry["transcription"] = transcriptions[idx]
            if pred_sentences:
                entry["pred_sentence"] = pred_sentences[idx]
            yield entry

    @staticmethod
    def _decode_transcription(arr: torch.Tensor) -> str:
        values = arr.tolist()
        if 0 in values:
            values = values[: values.index(0)]
        return "".join(chr(v) for v in values)
