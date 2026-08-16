# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Logit masking utilities for stable RL via dynamic vocabulary pruning.

Reference: "Taming the Tail: Stable RL via Dynamic Vocabulary Pruning in LLMs".
Port of aiic_verl/trainer/code/ppo/logit_masking.py.
"""

from __future__ import annotations

import math

import torch


def apply_minp_masking(
    logits: torch.Tensor,
    rho: float,
    mask_value: float = -50.0,
) -> tuple[torch.Tensor, float]:
    """Apply min-p masking to (already temperature-scaled) logits.

    Safe set: V_S(s) = {a | π(a|s) >= π_max(s) * rho}
    Equivalent in logit space: logit(a) >= logit_max + log(rho).

    Mask creation is under ``torch.no_grad()`` so only unmasked positions carry
    gradient. ``mask_value=-50`` (≈ prob 2e-22) is safe for BF16 backward through
    ``log_softmax``.

    Args:
        logits: Temperature-scaled logits ``(..., vocab_size)``.
        rho: Threshold in (0, 1]. Larger => more aggressive pruning.
        mask_value: Value written into masked positions. Default -50.

    Returns:
        (masked_logits, mask_ratio) where ``mask_ratio`` is the fraction of
        vocab entries removed (averaged over all positions), useful as a metric.
    """
    if rho <= 0.0:
        return logits, 0.0
    with torch.no_grad():
        threshold = logits.max(dim=-1, keepdim=True).values + math.log(rho)
        mask = logits < threshold
    mask_ratio = mask.float().mean().item()
    masked_logits = torch.where(mask, mask_value, logits)
    return masked_logits, mask_ratio
