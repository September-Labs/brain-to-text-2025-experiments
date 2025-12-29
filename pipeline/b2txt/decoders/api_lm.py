from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

import numpy as np


class ApiLmDecoder:
    """Decode phoneme logits via the remote LM HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_env: str = "B2TXT_LM_API_KEY",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_wait: float = 1.0,
        nbest: int = 100,
        alpha: float = 0.55,
        blank_penalty: float = 90.0,
        reorder_logits: bool = True,
        healthcheck: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_wait = float(retry_wait)
        self.nbest = int(nbest)
        self.alpha = float(alpha)
        self.blank_penalty = float(blank_penalty)
        self.reorder_logits = bool(reorder_logits)
        self.healthcheck = bool(healthcheck)

        key = api_key
        if not key:
            if api_key_env:
                key = os.environ.get(api_key_env) or api_key_env
        if not key:
            raise ValueError("Missing API key. Set api_key or env var.")
        self.api_key = key

    def health_check(self) -> tuple[bool, str]:
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            status = payload.get("status", "ok")
            if status != "ok":
                return False, f"Health check failed: {payload}"
            return True, "OK"
        except Exception as exc:
            return False, f"Health check error: {exc}"

    @staticmethod
    def _rearrange_logits(logits: np.ndarray) -> np.ndarray:
        return np.concatenate((logits[:, 0:1], logits[:, -1:], logits[:, 1:-1]), axis=-1)

    def _post_decode(self, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.base_url}/decode"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-API-Key", self.api_key)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {429, 500, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_wait * (attempt + 1))
                    continue
                raise
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"Decode failed: {last_error}")

    def decode_logits(self, logits: np.ndarray) -> str:
        if logits.ndim == 3:
            logits = logits[0]
        if self.reorder_logits:
            logits = self._rearrange_logits(logits)

        logits = np.asarray(logits, dtype=np.float32)
        payload = {
            "logits_b64": base64.b64encode(logits.tobytes()).decode("ascii"),
            "shape": list(logits.shape),
            "dtype": "float32",
            "params": {
                "alpha": self.alpha,
                "blank_penalty": self.blank_penalty,
                "nbest": self.nbest,
            },
        }
        response = self._post_decode(payload)
        return response.get("final_text", "")
