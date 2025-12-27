from __future__ import annotations

import importlib
import inspect
import os
import warnings
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from b2txt.alignment import ctc_forced_align
from b2txt.augmentations import apply_transforms
from b2txt.constants import LOGIT_TO_PHONEME
from b2txt.ctc import compute_output_lengths, greedy_decode, edit_distance
from b2txt.metrics import word_error_stats
from b2txt.tasks.base import BaseTask


def masked_mean(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    max_len = features.size(1)
    mask = torch.arange(max_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
    summed = (features * mask.unsqueeze(-1)).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom.unsqueeze(-1)


class LCSCTCZipfTask(BaseTask):
    """CTC phoneme decoding with optional LCS alignment and Zipf prior."""

    def __init__(self, config: Dict[str, object], device: torch.device) -> None:
        super().__init__(config, device)
        self.data_cfg = config["data"]
        self.model_cfg = config["model"]
        self.ctc_cfg = config.get("ctc", {})
        self.blank_id = int(self.ctc_cfg.get("blank_id", 0))
        self.loss_fn = torch.nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        self.boundary_id = int(self.data_cfg.get("n_classes", len(LOGIT_TO_PHONEME))) - 1

        task_params = (config.get("task") or {}).get("params", {}) if isinstance(config.get("task"), dict) else {}
        self.align_weight = float(task_params.get("align_weight", 0.0))
        self.use_zipf = bool(task_params.get("use_zipf", False))
        self.zipf_boost = float(task_params.get("zipf_boost", 0.0))
        self.zipf_apply_train = bool(task_params.get("zipf_apply_train", False))
        self.zipf_apply_eval = bool(task_params.get("zipf_apply_eval", True))
        self.zipf_apply_blank = bool(task_params.get("zipf_apply_blank", False))
        self.zipf_update = bool(task_params.get("zipf_update", True))
        self.zipf_update_every = int(task_params.get("zipf_update_every", 1))
        self.use_torchaudio_align = bool(
            task_params.get("use_torchaudio_align", os.environ.get("B2TXT_DISABLE_TORCHAUDIO_ALIGN") is None)
        )

        self._update_step = 0

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
                    if hasattr(self.decoder, "health_check") and self.decoder.healthcheck:
                        ok, msg = self.decoder.health_check()
                        if not ok:
                            if self.wer_strict:
                                raise ValueError(msg)
                            warnings.warn(msg)
                            self.wer_enabled = False
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
    ) -> dict[str, torch.Tensor]:
        signature = inspect.signature(model.forward)
        if "day_idx" in signature.parameters:
            output = model(features, day_indices)
        else:
            output = model(features)
        if isinstance(output, dict):
            return output
        return {"logits": output}

    def _prepare_batch(self, batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
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

    def _apply_zipf_bias(self, logits: torch.Tensor, zipf_weights: torch.Tensor) -> torch.Tensor:
        log_prior = torch.log(zipf_weights.clamp_min(1e-8))
        if not self.zipf_apply_blank:
            log_prior[:, self.blank_id] = 0.0
        return logits + (self.zipf_boost * log_prior)

    def _compute_alignment_and_updates(
        self,
        model: torch.nn.Module,
        aux: torch.Tensor,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        phone_seq_lens: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], List[torch.Tensor]]:
        align_loss = log_probs.new_zeros(())
        align_count = 0
        zipf_phonemes: List[int] = []
        zipf_features: List[torch.Tensor] = []

        batch_size = labels.shape[0]
        for i in range(batch_size):
            t_len = int(output_lengths[i].item())
            tgt_len = int(phone_seq_lens[i].item())
            if t_len == 0 or tgt_len == 0:
                continue
            targets = labels[i, :tgt_len]

            aligned_labels, aligned_pos = ctc_forced_align(
                log_probs[i, :t_len],
                targets,
                blank_id=self.blank_id,
                use_torchaudio=self.use_torchaudio_align,
            )

            if self.align_weight > 0.0:
                frame_logits = aux[i, :t_len] @ model.label_embedding.weight.T
                frame_logits = frame_logits.to(dtype=log_probs.dtype)
                valid = (aligned_labels > 0) & (aligned_labels < self.boundary_id)
                if valid.any():
                    align_loss = align_loss + F.cross_entropy(
                        frame_logits[valid], aligned_labels[valid].to(dtype=torch.long), reduction="mean"
                    )
                    align_count += 1

            if self.use_zipf and self.zipf_update:
                aux_i = aux[i, :t_len]
                for pos in range(tgt_len):
                    mask = aligned_pos == pos
                    if not mask.any():
                        continue
                    label_id = int(targets[pos].item())
                    if label_id <= 0 or label_id >= self.boundary_id:
                        continue
                    zipf_phonemes.append(label_id)
                    zipf_features.append(aux_i[mask].mean(dim=0))

        if align_count > 0:
            align_loss = align_loss / align_count

        return align_loss, zipf_phonemes, zipf_features

    def training_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="train")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        output = self.forward_model(model, prepared["features"], prepared["day_indices"])
        logits = output["logits"]
        aux = output.get("aux")
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])

        if self.use_zipf and self.zipf_apply_train and aux is not None and model.zipf_learner is not None:
            pooled = masked_mean(aux, output_lengths)
            zipf_weights = model.zipf_learner(pooled)
            logits = self._apply_zipf_bias(logits, zipf_weights)

        log_probs = logits.log_softmax(dim=-1)
        log_probs_t = log_probs.transpose(0, 1)
        ctc_loss = self.loss_fn(log_probs_t, labels, output_lengths, phone_seq_lens)
        ctc_loss = torch.mean(ctc_loss)

        align_loss = log_probs.new_zeros(())
        zipf_phonemes: List[int] = []
        zipf_features: List[torch.Tensor] = []
        needs_alignment = (self.align_weight > 0.0) or (self.use_zipf and self.zipf_update)
        if needs_alignment and aux is not None:
            align_loss, zipf_phonemes, zipf_features = self._compute_alignment_and_updates(
                model, aux, log_probs, labels, phone_seq_lens, output_lengths
            )

        loss = ctc_loss + (self.align_weight * align_loss)

        if self.use_zipf and self.zipf_update and model.zipf_learner is not None:
            self._update_step += 1
            if self.zipf_update_every <= 1 or (self._update_step % self.zipf_update_every == 0):
                if zipf_phonemes:
                    phoneme_tensor = torch.tensor(
                        zipf_phonemes, device=aux.device, dtype=torch.long
                    )
                    feat_tensor = torch.stack(zipf_features, dim=0)
                    with torch.no_grad():
                        model.zipf_learner.update_statistics(phoneme_tensor, feat_tensor)

        return {
            "loss": loss,
            "ctc_loss": float(ctc_loss.item()),
            "align_loss": float(align_loss.item()) if self.align_weight > 0.0 else 0.0,
        }

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        prepared = self._prepare_batch(batch, mode="eval")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        output = self.forward_model(model, prepared["features"], prepared["day_indices"])
        logits = output["logits"]
        aux = output.get("aux")
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])

        if self.use_zipf and self.zipf_apply_eval and aux is not None and model.zipf_learner is not None:
            pooled = masked_mean(aux, output_lengths)
            zipf_weights = model.zipf_learner(pooled)
            logits = self._apply_zipf_bias(logits, zipf_weights)

        log_probs = logits.log_softmax(dim=-1)
        log_probs_t = log_probs.transpose(0, 1)
        ctc_loss = self.loss_fn(log_probs_t, labels, output_lengths, phone_seq_lens)
        ctc_loss = torch.mean(ctc_loss)

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
            "loss_sum": float(ctc_loss.item()),
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
        output = self.forward_model(model, prepared["features"], prepared["day_indices"])
        logits = output["logits"]
        aux = output.get("aux")
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])

        if self.use_zipf and self.zipf_apply_eval and aux is not None and model.zipf_learner is not None:
            pooled = masked_mean(aux, output_lengths)
            zipf_weights = model.zipf_learner(pooled)
            logits = self._apply_zipf_bias(logits, zipf_weights)

        preds = greedy_decode(logits, blank_id=self.blank_id)
        output_dict: Dict[str, object] = {"pred_ids": preds}
        if self.decoder and self.decode_predictions:
            logits_np = logits.detach().float().cpu().numpy()
            output_dict["pred_sentences"] = [
                self.decoder.decode_logits(logits_np[i]) for i in range(len(preds))
            ]
        return output_dict

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
        for idx, seq in enumerate(pred_ids):
            entry = {
                "session": sessions[int(day_indices[idx])],
                "block_num": int(block_nums[idx]),
                "trial_num": int(trial_nums[idx]),
                "pred_phonemes": " ".join(LOGIT_TO_PHONEME[token] for token in seq),
            }
            if pred_sentences:
                entry["pred_sentence"] = pred_sentences[idx]
            yield entry
