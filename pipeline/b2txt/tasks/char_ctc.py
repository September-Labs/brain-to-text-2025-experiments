from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from b2txt.augmentations import apply_transforms
from b2txt.ctc import compute_output_lengths, greedy_decode, edit_distance
from b2txt.metrics import normalize_sentence, word_error_stats
from b2txt.tasks.base import BaseTask


class CharCTCTask(BaseTask):
    """CTC character-level task for direct text decoding."""

    def __init__(self, config: Dict[str, object], device: torch.device) -> None:
        super().__init__(config, device)
        self.data_cfg = config["data"]
        self.model_cfg = config["model"]
        self.ctc_cfg = config.get("ctc", {})
        self.blank_id = int(self.ctc_cfg.get("blank_id", 0))
        self.loss_fn = torch.nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

        task_cfg = config.get("task", {})
        params = task_cfg.get("params", {}) if isinstance(task_cfg, dict) else {}
        vocab = params.get("vocab", "abcdefghijklmnopqrstuvwxyz' ")
        self.lowercase = bool(params.get("lowercase", True))
        self.drop_unknown = bool(params.get("drop_unknown", True))

        if isinstance(vocab, list):
            vocab_list = vocab
        else:
            vocab_list = list(vocab)

        self.id_to_char = [None] + vocab_list
        self.char_to_id = {ch: idx + 1 for idx, ch in enumerate(vocab_list)}

    def _normalize(self, sentence: str) -> str:
        normalized = normalize_sentence(sentence)
        if self.lowercase:
            normalized = normalized.lower()
        return normalized

    def _encode_sentence(self, sentence: str) -> List[int]:
        encoded: List[int] = []
        for ch in sentence:
            if ch in self.char_to_id:
                encoded.append(self.char_to_id[ch])
            elif not self.drop_unknown:
                raise ValueError(f"Unknown character '{ch}' not in vocab")
        return encoded

    def _decode_ids(self, ids: List[int]) -> str:
        chars: List[str] = []
        for token in ids:
            if token == self.blank_id:
                continue
            if token < len(self.id_to_char):
                ch = self.id_to_char[token]
                if ch is not None:
                    chars.append(ch)
        return "".join(chars)

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
            "output_lengths": output_lengths,
            "day_indices": day_indices,
        }

    def training_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="train")
        sentences = batch.get("sentence_labels", [])
        targets = [self._encode_sentence(self._normalize(s)) for s in sentences]
        target_lengths = torch.tensor([len(t) for t in targets], device=self.device, dtype=torch.int32)

        max_len = max(1, max(target_lengths).item() if len(target_lengths) else 1)
        target_tensor = torch.zeros((len(targets), max_len), device=self.device, dtype=torch.int64)
        for idx, seq in enumerate(targets):
            if seq:
                target_tensor[idx, : len(seq)] = torch.tensor(seq, device=self.device)

        logits = model(prepared["features"], prepared["day_indices"]) if "day_idx" in model.forward.__code__.co_varnames else model(prepared["features"])
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        loss = self.loss_fn(log_probs, target_tensor, output_lengths, target_lengths)
        loss = torch.mean(loss)

        return {"loss": loss}

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        prepared = self._prepare_batch(batch, mode="eval")
        sentences = batch.get("sentence_labels", [])
        targets = [self._encode_sentence(self._normalize(s)) for s in sentences]
        target_lengths = torch.tensor([len(t) for t in targets], device=self.device, dtype=torch.int32)

        max_len = max(1, max(target_lengths).item() if len(target_lengths) else 1)
        target_tensor = torch.zeros((len(targets), max_len), device=self.device, dtype=torch.int64)
        for idx, seq in enumerate(targets):
            if seq:
                target_tensor[idx, : len(seq)] = torch.tensor(seq, device=self.device)

        logits = model(prepared["features"], prepared["day_indices"]) if "day_idx" in model.forward.__code__.co_varnames else model(prepared["features"])
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

        loss = self.loss_fn(log_probs, target_tensor, output_lengths, target_lengths)
        loss = torch.mean(loss)

        preds = greedy_decode(logits, blank_id=self.blank_id)
        cer_edits = 0
        cer_chars = 0
        wer_edits = 0
        wer_words = 0

        for idx, pred_ids in enumerate(preds):
            pred_text = self._decode_ids(pred_ids)
            true_text = self._normalize(sentences[idx])
            cer_edits += edit_distance(list(pred_text), list(true_text))
            cer_chars += len(true_text)

            edits, words = word_error_stats(true_text, pred_text)
            wer_edits += edits
            wer_words += words

        return {
            "loss_sum": float(loss.item()),
            "count": 1.0,
            "cer_edits": float(cer_edits),
            "cer_chars": float(cer_chars),
            "wer_edits": float(wer_edits),
            "wer_words": float(wer_words),
        }

    def init_validation_state(self) -> Dict[str, float]:
        return {
            "loss_sum": 0.0,
            "count": 0.0,
            "cer_edits": 0.0,
            "cer_chars": 0.0,
            "wer_edits": 0.0,
            "wer_words": 0.0,
        }

    def accumulate_validation(self, state: Dict[str, float], batch_stats: Dict[str, float]) -> Dict[str, float]:
        state["loss_sum"] += batch_stats.get("loss_sum", 0.0)
        state["count"] += batch_stats.get("count", 0.0)
        state["cer_edits"] += batch_stats.get("cer_edits", 0.0)
        state["cer_chars"] += batch_stats.get("cer_chars", 0.0)
        state["wer_edits"] += batch_stats.get("wer_edits", 0.0)
        state["wer_words"] += batch_stats.get("wer_words", 0.0)
        return state

    def finalize_validation(self, state: Dict[str, float]) -> Dict[str, float]:
        avg_loss = state["loss_sum"] / max(1.0, state["count"])
        metrics = {"val_loss": avg_loss}
        if state["cer_chars"] > 0:
            metrics["val_cer"] = state["cer_edits"] / state["cer_chars"]
        if state["wer_words"] > 0:
            metrics["val_wer"] = state["wer_edits"] / state["wer_words"]
        return metrics

    def predict_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="eval")
        logits = model(prepared["features"], prepared["day_indices"]) if "day_idx" in model.forward.__code__.co_varnames else model(prepared["features"])
        preds = greedy_decode(logits, blank_id=self.blank_id)
        pred_texts = [self._decode_ids(seq) for seq in preds]
        return {"pred_ids": preds, "pred_texts": pred_texts}

    def format_predictions(
        self,
        batch: Dict[str, torch.Tensor],
        predictions: Dict[str, object],
    ) -> Iterable[Dict[str, object]]:
        day_indices = batch["day_indicies"].tolist()
        block_nums = batch["block_nums"].tolist()
        trial_nums = batch["trial_nums"].tolist()
        pred_texts = predictions.get("pred_texts", [])

        sessions = self.data_cfg["sessions"]
        for idx, pred_text in enumerate(pred_texts):
            yield {
                "session": sessions[int(day_indices[idx])],
                "block_num": int(block_nums[idx]),
                "trial_num": int(trial_nums[idx]),
                "pred_sentence": pred_text,
            }
