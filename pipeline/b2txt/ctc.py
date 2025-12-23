from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def compute_output_lengths(
    n_time_steps: torch.Tensor,
    model_config: Dict[str, object],
    ctc_config: Dict[str, object] | None = None,
) -> torch.Tensor:
    """Compute CTC input lengths after time reduction."""
    output_cfg = (ctc_config or {}).get("output_length") if ctc_config else None

    if output_cfg:
        mode = output_cfg.get("type", "identity")
        if mode == "patch":
            patch_size = int(output_cfg.get("patch_size", 0))
            patch_stride = int(output_cfg.get("patch_stride", 1))
            if patch_size > 0 and patch_stride > 0:
                return ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
        if mode == "downsample":
            factor = float(output_cfg.get("factor", 1.0))
            if factor > 0:
                return torch.floor(n_time_steps / factor).to(torch.int32)
        return n_time_steps.to(torch.int32)

    params = model_config.get("params", {}) if model_config else {}
    patch_size = int(params.get("patch_size", 0))
    patch_stride = int(params.get("patch_stride", 1))
    if patch_size > 0 and patch_stride > 0:
        return ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)

    return n_time_steps.to(torch.int32)


def greedy_decode(
    logits: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """CTC greedy decode for a batch of logits."""
    pred = torch.argmax(logits, dim=-1).tolist()
    decoded: List[List[int]] = []
    for seq in pred:
        collapsed: List[int] = []
        prev = None
        for token in seq:
            if token == prev:
                continue
            prev = token
            if token == blank_id:
                continue
            collapsed.append(token)
        decoded.append(collapsed)
    return decoded


def edit_distance(a: Iterable[int], b: Iterable[int]) -> int:
    """Compute Levenshtein distance between two sequences."""
    a_list = list(a)
    b_list = list(b)

    if not a_list:
        return len(b_list)
    if not b_list:
        return len(a_list)

    prev_row = list(range(len(b_list) + 1))
    for i, a_val in enumerate(a_list, start=1):
        curr = [i]
        for j, b_val in enumerate(b_list, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (0 if a_val == b_val else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr
    return prev_row[-1]
