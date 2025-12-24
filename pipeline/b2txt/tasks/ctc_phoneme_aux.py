from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from b2txt.augmentations import apply_transforms
from b2txt.constants import LOGIT_TO_PHONEME
from b2txt.ctc import compute_output_lengths, greedy_decode, edit_distance
from b2txt.metrics import word_error_stats
from b2txt.tasks.base import BaseTask


def get_ipa_feature_matrix() -> Tuple[torch.Tensor, List[str]]:
    feature_names = [
        "consonantal",
        "syllabic",
        "sonorant",
        "voice",
        "nasal",
        "continuant",
        "labial",
        "coronal",
        "dorsal",
        "high",
        "low",
        "back",
        "round",
        "diphthong",
    ]

    rows = {
        "AA": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        "AE": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "AH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "AO": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        "AW": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        "AY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        "EH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "ER": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "EY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        "IH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "IY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "OW": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        "OY": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        "UH": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        "UW": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        "P": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "B": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "T": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "D": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "K": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "G": [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "CH": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "JH": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "F": [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "V": [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "TH": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "DH": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "S": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "Z": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "SH": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "ZH": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "HH": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "M": [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "N": [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "NG": [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "L": [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "R": [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "W": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        "Y": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    }

    phoneme_order = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    ]

    feature_matrix = torch.tensor([rows[p] for p in phoneme_order], dtype=torch.float32)
    return feature_matrix, feature_names


def supervised_nt_xent(
    emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1, eps: float = 1e-8
) -> torch.Tensor:
    n_samples = emb.shape[0]
    if n_samples <= 1:
        return emb.new_zeros(())

    sim = torch.matmul(emb, emb.T) / temperature
    logits_mask = torch.ones_like(sim, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & logits_mask

    sim_max, _ = torch.max(sim.masked_fill(~logits_mask, float("-inf")), dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1) + eps

    pos_exp_sim = exp_sim * pos_mask
    num = pos_exp_sim.sum(dim=1)

    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return emb.new_zeros(())

    loss_i = -torch.log((num[valid] + eps) / denom[valid])
    return loss_i.mean()


class CTCPhonemeAuxTask(BaseTask):
    """CTC phoneme decoding with IPA and contrastive auxiliary losses."""

    def __init__(self, config: Dict[str, object], device: torch.device) -> None:
        super().__init__(config, device)
        self.data_cfg = config["data"]
        self.model_cfg = config["model"]
        self.ctc_cfg = config.get("ctc", {})
        self.blank_id = int(self.ctc_cfg.get("blank_id", 0))
        self.loss_fn = torch.nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        self.boundary_id = int(self.data_cfg.get("n_classes", len(LOGIT_TO_PHONEME))) - 1

        task_params = (config.get("task") or {}).get("params", {}) if isinstance(config.get("task"), dict) else {}
        model_params = self.model_cfg.get("params", {})

        self.use_ipa_features = bool(task_params.get("use_ipa_features", model_params.get("use_ipa_features", False)))
        self.use_contrastive = bool(
            task_params.get("use_contrastive", model_params.get("use_contrastive", False))
        )

        self.ipa_feature_weight = float(
            task_params.get("ipa_feature_weight", model_params.get("ipa_feature_weight", 0.0))
        )
        self.contrastive_weight = float(
            task_params.get("contrastive_weight", model_params.get("contrastive_weight", 0.0))
        )
        self.contrastive_temperature = float(
            task_params.get("contrastive_temperature", model_params.get("contrastive_temperature", 0.1))
        )
        self.contrastive_max_samples = int(
            task_params.get("contrastive_max_samples", model_params.get("contrastive_max_samples", 2048))
        )

        self.ipa_features = None
        if self.use_ipa_features:
            feature_matrix, _ = get_ipa_feature_matrix()
            n_classes = int(self.data_cfg.get("n_classes", feature_matrix.shape[0] + 2))
            full = torch.zeros((n_classes, feature_matrix.shape[1]), dtype=feature_matrix.dtype)
            full[1 : 1 + feature_matrix.shape[0]] = feature_matrix
            self.ipa_features = full

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
    ) -> dict[str, torch.Tensor | None]:
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

    def _ctc_forced_align(
        self, log_probs: torch.Tensor, targets: torch.Tensor, blank_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_steps = log_probs.shape[0]
        target_len = targets.numel()
        if target_len == 0:
            labels = torch.full((time_steps,), blank_id, device=log_probs.device, dtype=torch.long)
            positions = torch.full((time_steps,), -1, device=log_probs.device, dtype=torch.long)
            return labels, positions

        extended = torch.full((target_len * 2 + 1,), blank_id, device=log_probs.device, dtype=torch.long)
        extended[1::2] = targets
        states = extended.numel()

        dp = log_probs.new_full((time_steps, states), -1e9)
        bp = torch.full((time_steps, states), -1, device=log_probs.device, dtype=torch.int32)

        dp[0, 0] = log_probs[0, blank_id]
        if states > 1:
            dp[0, 1] = log_probs[0, extended[1]]

        for t in range(1, time_steps):
            for s in range(states):
                best_score = dp[t - 1, s]
                best_state = s
                if s - 1 >= 0 and dp[t - 1, s - 1] > best_score:
                    best_score = dp[t - 1, s - 1]
                    best_state = s - 1
                if s - 2 >= 0 and extended[s] != blank_id and extended[s] != extended[s - 2]:
                    score = dp[t - 1, s - 2]
                    if score > best_score:
                        best_score = score
                        best_state = s - 2
                dp[t, s] = best_score + log_probs[t, extended[s]]
                bp[t, s] = best_state

        if states == 1:
            state = 0
        else:
            state = states - 1 if dp[-1, states - 1] >= dp[-1, states - 2] else states - 2

        state_seq: List[int] = []
        for t in range(time_steps - 1, -1, -1):
            state_seq.append(state)
            if t > 0:
                state = int(bp[t, state].item())
        state_seq.reverse()

        labels = torch.full((time_steps,), blank_id, device=log_probs.device, dtype=torch.long)
        positions = torch.full((time_steps,), -1, device=log_probs.device, dtype=torch.long)
        for t, s in enumerate(state_seq):
            if s % 2 == 1:
                pos = (s - 1) // 2
                labels[t] = targets[pos]
                positions[t] = pos
        return labels, positions

    def _compute_aux_losses(
        self,
        output: dict[str, torch.Tensor | None],
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        phone_seq_lens: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = labels.shape[0]
        ipa_loss = log_probs.new_zeros(())
        contrastive_loss = log_probs.new_zeros(())

        if not (self.use_ipa_features or self.use_contrastive):
            return ipa_loss, contrastive_loss

        ipa_logits = output.get("ipa_logits")
        proj = output.get("proj")
        if proj is None:
            proj = output.get("embeddings")
        alignments: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(batch_size):
            t_len = int(output_lengths[i].item())
            tgt_len = int(phone_seq_lens[i].item())
            tgt = labels[i, :tgt_len]
            aligned = self._ctc_forced_align(log_probs[i, :t_len], tgt, self.blank_id)
            alignments.append(aligned)

        if self.use_ipa_features and ipa_logits is not None and self.ipa_features is not None:
            ipa_features = self.ipa_features.to(device=ipa_logits.device, dtype=ipa_logits.dtype)
            loss_sum = log_probs.new_zeros(())
            count_sum = log_probs.new_zeros(())
            for i in range(batch_size):
                t_len = int(output_lengths[i].item())
                aligned_labels, _ = alignments[i]
                aligned_labels = aligned_labels[:t_len]
                ipa_targets = ipa_features[aligned_labels]
                mask = (aligned_labels > 0) & (aligned_labels < self.boundary_id)
                if not mask.any():
                    continue
                logits_i = ipa_logits[i, :t_len]
                loss = F.binary_cross_entropy_with_logits(logits_i, ipa_targets, reduction="none").mean(dim=-1)
                loss_sum = loss_sum + (loss * mask.float()).sum()
                count_sum = count_sum + mask.float().sum()
            ipa_loss = loss_sum / count_sum.clamp_min(1.0)

        if self.use_contrastive and proj is not None:
            emb_list: List[torch.Tensor] = []
            label_list: List[int] = []
            for i in range(batch_size):
                t_len = int(output_lengths[i].item())
                tgt_len = int(phone_seq_lens[i].item())
                if tgt_len == 0:
                    continue
                proj_i = proj[i, :t_len]
                _, aligned_pos = alignments[i]
                aligned_pos = aligned_pos[:t_len]
                for pos in range(tgt_len):
                    mask = aligned_pos == pos
                    if not mask.any():
                        continue
                    label_val = int(labels[i, pos].item())
                    if label_val <= 0 or label_val >= self.boundary_id:
                        continue
                    emb_list.append(proj_i[mask].mean(dim=0))
                    label_list.append(label_val)

            if emb_list:
                emb = torch.stack(emb_list, dim=0)
                label_tensor = torch.tensor(label_list, device=emb.device, dtype=torch.long)
                if self.contrastive_max_samples and emb.shape[0] > self.contrastive_max_samples:
                    perm = torch.randperm(emb.shape[0], device=emb.device)[: self.contrastive_max_samples]
                    emb = emb[perm]
                    label_tensor = label_tensor[perm]
                emb = F.normalize(emb, dim=-1)
                contrastive_loss = supervised_nt_xent(
                    emb, label_tensor, temperature=self.contrastive_temperature
                )

        return ipa_loss, contrastive_loss

    def training_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="train")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        output = self.forward_model(model, prepared["features"], prepared["day_indices"])
        logits = output["logits"]
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])
        log_probs = logits.log_softmax(dim=-1)
        log_probs_t = log_probs.transpose(0, 1)

        ctc_loss = self.loss_fn(
            log_probs_t,
            labels,
            output_lengths,
            phone_seq_lens,
        )
        ctc_loss = torch.mean(ctc_loss)

        ipa_loss, contrastive_loss = self._compute_aux_losses(
            output, log_probs, labels, phone_seq_lens, output_lengths
        )
        loss = ctc_loss + (self.ipa_feature_weight * ipa_loss) + (self.contrastive_weight * contrastive_loss)

        return {
            "loss": loss,
            "ctc_loss": float(ctc_loss.item()),
            "ipa_loss": float(ipa_loss.item()) if self.use_ipa_features else 0.0,
            "contrastive_loss": float(contrastive_loss.item()) if self.use_contrastive else 0.0,
        }

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        prepared = self._prepare_batch(batch, mode="eval")
        labels = batch["seq_class_ids"].to(self.device)
        phone_seq_lens = batch["phone_seq_lens"].to(self.device)

        output = self.forward_model(model, prepared["features"], prepared["day_indices"])
        logits = output["logits"]
        output_lengths = torch.clamp(prepared["output_lengths"], min=1, max=logits.shape[1])
        log_probs = logits.log_softmax(dim=-1)
        log_probs_t = log_probs.transpose(0, 1)

        ctc_loss = self.loss_fn(
            log_probs_t,
            labels,
            output_lengths,
            phone_seq_lens,
        )
        ctc_loss = torch.mean(ctc_loss)

        ipa_loss, contrastive_loss = self._compute_aux_losses(
            output, log_probs, labels, phone_seq_lens, output_lengths
        )
        loss = ctc_loss + (self.ipa_feature_weight * ipa_loss) + (self.contrastive_weight * contrastive_loss)

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
            "ipa_loss_sum": float(ipa_loss.item()) if self.use_ipa_features else 0.0,
            "contrastive_loss_sum": float(contrastive_loss.item()) if self.use_contrastive else 0.0,
            "aux_count": 1.0,
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
            "ipa_loss_sum": 0.0,
            "contrastive_loss_sum": 0.0,
            "aux_count": 0.0,
        }

    def accumulate_validation(self, state: Dict[str, float], batch_stats: Dict[str, float]) -> Dict[str, float]:
        state["loss_sum"] += batch_stats.get("loss_sum", 0.0)
        state["count"] += batch_stats.get("count", 0.0)
        state["per_edits"] += batch_stats.get("per_edits", 0.0)
        state["per_targets"] += batch_stats.get("per_targets", 0.0)
        state["wer_edits"] += batch_stats.get("wer_edits", 0.0)
        state["wer_words"] += batch_stats.get("wer_words", 0.0)
        state["ipa_loss_sum"] += batch_stats.get("ipa_loss_sum", 0.0)
        state["contrastive_loss_sum"] += batch_stats.get("contrastive_loss_sum", 0.0)
        state["aux_count"] += batch_stats.get("aux_count", 0.0)
        return state

    def finalize_validation(self, state: Dict[str, float]) -> Dict[str, float]:
        avg_loss = state["loss_sum"] / max(1.0, state["count"])
        per = None
        if state["per_targets"] > 0:
            per = state["per_edits"] / state["per_targets"]
        metrics = {"val_loss": avg_loss, "val_per": per}
        if state["wer_words"] > 0:
            metrics["val_wer"] = state["wer_edits"] / state["wer_words"]
        if self.use_ipa_features and state["aux_count"] > 0:
            metrics["val_ipa_loss"] = state["ipa_loss_sum"] / state["aux_count"]
        if self.use_contrastive and state["aux_count"] > 0:
            metrics["val_contrastive_loss"] = state["contrastive_loss_sum"] / state["aux_count"]
        return metrics

    def predict_step(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        prepared = self._prepare_batch(batch, mode="eval")
        output = self.forward_model(model, prepared["features"], prepared["day_indices"])
        logits = output["logits"]
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
