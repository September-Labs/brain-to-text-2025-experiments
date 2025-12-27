from __future__ import annotations

from typing import List, Tuple

import torch

try:
    import torchaudio.functional as AF
except Exception:  # pragma: no cover
    AF = None


def positions_from_aligned(
    aligned_labels: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    positions = [-1] * aligned_labels.numel()
    if targets.numel() == 0:
        return torch.tensor(positions, dtype=torch.long)

    tgt = targets.tolist()
    cur = -1
    prev_label = blank_id
    for t, label in enumerate(aligned_labels.tolist()):
        if label == blank_id:
            prev_label = blank_id
            continue
        if cur == -1:
            if label == tgt[0]:
                cur = 0
            positions[t] = cur
            prev_label = label
            continue
        if label == tgt[cur]:
            if prev_label == blank_id and cur + 1 < len(tgt) and tgt[cur + 1] == label:
                cur += 1
            positions[t] = cur
            prev_label = label
            continue
        if cur + 1 < len(tgt) and label == tgt[cur + 1]:
            cur += 1
            positions[t] = cur
        prev_label = label

    return torch.tensor(positions, dtype=torch.long)


def ctc_forced_align(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    use_torchaudio: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    time_steps = log_probs.shape[0]
    target_len = targets.numel()
    if target_len == 0:
        labels = torch.full((time_steps,), blank_id, device=log_probs.device, dtype=torch.long)
        positions = torch.full((time_steps,), -1, device=log_probs.device, dtype=torch.long)
        return labels, positions

    if use_torchaudio and AF is not None and hasattr(AF, "forced_align") and log_probs.is_cuda:
        try:
            align_log_probs = log_probs
            if align_log_probs.dtype != torch.float32:
                align_log_probs = align_log_probs.float()
            log_probs_b = align_log_probs.unsqueeze(0)
            targets_b = targets.unsqueeze(0)
            input_lengths = torch.tensor([time_steps], device=align_log_probs.device)
            target_lengths = torch.tensor([target_len], device=align_log_probs.device)
            aligned_labels, _ = AF.forced_align(
                log_probs_b,
                targets_b,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=blank_id,
            )
            aligned_labels = aligned_labels[0]
            pos_cpu = positions_from_aligned(
                aligned_labels.detach().cpu(),
                targets.detach().cpu(),
                blank_id,
            )
            aligned_pos = pos_cpu.to(device=log_probs.device)
            return aligned_labels, aligned_pos
        except Exception:
            pass

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
