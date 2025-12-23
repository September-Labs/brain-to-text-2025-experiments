from __future__ import annotations

import time
from typing import Dict, List

import numpy as np


class RedisLmDecoder:
    """Decode phoneme logits to text using the NEJM redis LM service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        input_stream: str = "remote_lm_input",
        output_partial_stream: str = "remote_lm_output_partial",
        output_final_stream: str = "remote_lm_output_final",
        update_params: bool = False,
        acoustic_scale: float = 0.35,
        blank_penalty: float = 90.0,
        alpha: float = 0.55,
        flush_on_init: bool = False,
    ) -> None:
        import redis

        self.redis = redis.Redis(host=host, port=port, db=db)
        if flush_on_init:
            self.redis.flushall()

        self.input_stream = input_stream
        self.output_partial_stream = output_partial_stream
        self.output_final_stream = output_final_stream

        self.update_params = update_params
        self.acoustic_scale = acoustic_scale
        self.blank_penalty = blank_penalty
        self.alpha = alpha

        now = self._get_current_time_ms()
        self.partial_last_seen = now
        self.final_last_seen = now
        self.reset_last_seen = now
        self.update_last_seen = now

    def decode_logits(self, logits: np.ndarray) -> str:
        if logits.ndim == 3:
            logits = logits[0]
        logits = self._rearrange_logits(logits)

        self.reset_last_seen = self._reset_remote_lm(self.reset_last_seen)
        if self.update_params:
            self.update_last_seen = self._update_remote_lm_params(self.update_last_seen)

        self.partial_last_seen, _ = self._send_logits(self.partial_last_seen, logits)
        self.final_last_seen, lm_out = self._finalize_remote_lm(self.final_last_seen)

        return lm_out["candidate_sentences"][0]

    def _get_current_time_ms(self) -> int:
        t = self.redis.time()
        return int(t[0] * 1000 + t[1] / 1000)

    @staticmethod
    def _rearrange_logits(logits: np.ndarray) -> np.ndarray:
        return np.concatenate((logits[:, 0:1], logits[:, -1:], logits[:, 1:-1]), axis=-1)

    def _reset_remote_lm(self, last_seen: int) -> int:
        self.redis.xadd("remote_lm_reset", {"done": 0})
        time.sleep(0.001)
        response = []
        while not response:
            response = self.redis.xread({"remote_lm_done_resetting": last_seen}, count=1, block=10000)
        for entry_id, _ in response[0][1]:
            last_seen = entry_id
        return last_seen

    def _update_remote_lm_params(self, last_seen: int) -> int:
        entry_dict = {
            "acoustic_scale": self.acoustic_scale,
            "blank_penalty": self.blank_penalty,
            "alpha": self.alpha,
        }
        self.redis.xadd("remote_lm_update_params", entry_dict)
        time.sleep(0.001)
        response = []
        while not response:
            response = self.redis.xread(
                {"remote_lm_done_updating_params": last_seen}, count=1, block=10000
            )
        for entry_id, _ in response[0][1]:
            last_seen = entry_id
        return last_seen

    def _send_logits(self, last_seen: int, logits: np.ndarray) -> tuple[int, str]:
        self.redis.xadd(self.input_stream, {"logits": np.float32(logits).tobytes()})
        response = []
        while not response:
            response = self.redis.xread({self.output_partial_stream: last_seen}, count=1, block=10000)
        decoded = ""
        for entry_id, entry_data in response[0][1]:
            last_seen = entry_id
            decoded = entry_data[b"lm_response_partial"].decode()
        return last_seen, decoded

    def _finalize_remote_lm(self, last_seen: int) -> tuple[int, Dict[str, List[object]]]:
        self.redis.xadd("remote_lm_finalize", {"done": 0})
        time.sleep(0.005)
        response = []
        while not response:
            response = self.redis.xread({self.output_final_stream: last_seen}, count=1, block=10000)
        candidate_sentences: List[str] = []
        candidate_total_scores: List[float] = []

        for entry_id, entry_data in response[0][1]:
            last_seen = entry_id
            scores = entry_data[b"scoring"].decode().split(";")
            candidate_sentences = [str(c) for c in scores[::5]]
            candidate_total_scores = [float(c) for c in scores[4::5]]

        if not candidate_sentences or not candidate_total_scores:
            candidate_sentences = [""]
            candidate_total_scores = [0.0]
        else:
            sort_order = np.argsort(candidate_total_scores)[::-1]
            candidate_sentences = [candidate_sentences[i] for i in sort_order]
            candidate_total_scores = [candidate_total_scores[i] for i in sort_order]

        return last_seen, {
            "candidate_sentences": candidate_sentences,
            "candidate_total_scores": candidate_total_scores,
        }
