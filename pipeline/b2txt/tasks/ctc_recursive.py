# In b2txt/tasks/iterative_ctc.py

from typing import Dict, Iterable

import torch
from torch import nn

from b2txt.ctc import compute_output_lengths, greedy_decode, edit_distance
from b2txt.metrics import word_error_stats
from b2txt.tasks.ctc_phoneme import CTCPhonemeTask


class IterativeRefinementCTCTask(CTCPhonemeTask):
    """Task for iterative refinement CTC models."""

    def __init__(self, config: Dict[str, object], device: torch.device) -> None:
        super().__init__(config, device)
        self.iteration_cfg = config.get("iteration", {})
        self.refinement_loss_weight = self.iteration_cfg.get("refinement_loss_weight", 1.0)
        self.iteration_loss_decay = self.iteration_cfg.get("iteration_loss_decay", 0.8)

    def forward_model(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        day_indices: torch.Tensor,
        **kwargs,
    ) -> dict:
        """Forward pass that handles dict output from iterative models."""
        output = model(features, day_indices, **kwargs)
        if isinstance(output, dict):
            return output
        return {"logits": output}

    def training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="train")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        output = model(
            prepared["features"],
            prepared["day_indices"],
            return_all_iterations=True,
        )

        if hasattr(model, "compute_loss"):
            losses = model.compute_loss(
                output,
                labels,
                prepared["output_lengths"],
                phone_seq_lens,
                blank_id=self.blank_id,
            )
            return {"loss": losses["total_loss"], **losses}

        all_logits = output.get("all_logits", [output["initial_logits"], output["logits"]])

        total_loss = 0.0
        for i, logits in enumerate(all_logits):
            max_len = logits.shape[1]
            output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=max_len)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

            loss = self.loss_fn(log_probs, labels, output_lengths, phone_seq_lens)
            loss = loss.mean()

            if i == 0:
                weight = 0.5
            else:
                weight = self.refinement_loss_weight * (
                    self.iteration_loss_decay ** (len(all_logits) - 1 - i)
                )

            total_loss += weight * loss

        if output.get("load_balance_loss") is not None:
            total_loss += output["load_balance_loss"]

        return {"loss": total_loss}

    def validation_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        prepared = self._prepare_batch(batch, mode="eval")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        output = model(prepared["features"], prepared["day_indices"])

        logits = output["logits"]
        initial_logits = output.get("initial_logits", logits)

        max_len = logits.shape[1]
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=max_len)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        loss = self.loss_fn(log_probs, labels, output_lengths, phone_seq_lens)
        loss = loss.mean()

        per_edits_initial = 0
        per_edits_refined = 0
        per_targets = 0
        wer_edits = 0
        wer_words = 0

        preds_initial = greedy_decode(initial_logits, blank_id=self.blank_id)
        preds_refined = greedy_decode(logits, blank_id=self.blank_id)

        for idx in range(len(preds_refined)):
            target_len = int(phone_seq_lens[idx].item())
            if target_len == 0:
                continue
            target_seq = labels[idx, :target_len].tolist()
            per_edits_initial += edit_distance(preds_initial[idx], target_seq)
            per_edits_refined += edit_distance(preds_refined[idx], target_seq)
            per_targets += target_len

        if self.wer_enabled and self.decoder and "sentence_labels" in batch:
            logits_np = logits.detach().float().cpu().numpy()
            lengths_np = output_lengths.detach().cpu().numpy()
            trial_count = len(preds_refined)
            if self._wer_remaining is not None:
                trial_count = min(trial_count, self._wer_remaining)
            for idx in range(trial_count):
                t_len = int(lengths_np[idx]) if idx < len(lengths_np) else logits_np.shape[1]
                pred_sentence = self.decoder.decode_logits(logits_np[idx, :t_len])
                true_sentence = batch["sentence_labels"][idx]
                edits, words = word_error_stats(true_sentence, pred_sentence)
                wer_edits += edits
                wer_words += words
            if self._wer_remaining is not None:
                self._wer_remaining -= trial_count

        return {
            "loss_sum": float(loss.item()),
            "count": 1.0,
            "per_edits": float(per_edits_refined),
            "per_edits_initial": float(per_edits_initial),
            "per_targets": float(per_targets),
            "wer_edits": float(wer_edits),
            "wer_words": float(wer_words),
        }

    def predict_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="eval")
        logits = self.forward_model(model, prepared["features"], prepared["day_indices"])["logits"]
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])
        preds = greedy_decode(logits, blank_id=self.blank_id)
        output: Dict[str, object] = {"pred_ids": preds}
        if self.decoder and self.decode_predictions:
            logits_np = logits.detach().float().cpu().numpy()
            lengths_np = output_lengths.detach().cpu().numpy()
            output["pred_sentences"] = [
                self.decoder.decode_logits(logits_np[i, : int(lengths_np[i])])
                for i in range(len(preds))
            ]
        return output

    def init_validation_state(self) -> Dict[str, float]:
        state = super().init_validation_state()
        state["per_edits_initial"] = 0.0
        return state

    def accumulate_validation(self, state: Dict[str, float], batch_stats: Dict[str, float]) -> Dict[str, float]:
        state = super().accumulate_validation(state, batch_stats)
        state["per_edits_initial"] += batch_stats.get("per_edits_initial", 0.0)
        return state

    def finalize_validation(self, state: Dict[str, float]) -> Dict[str, float]:
        metrics = super().finalize_validation(state)
        if state["per_targets"] > 0:
            metrics["val_per_initial"] = state["per_edits_initial"] / state["per_targets"]
        return metrics
