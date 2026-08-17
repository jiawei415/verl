# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Custom reward function that layers a tool-use bonus on top of math_dapo scoring.

Wired via:
    custom_reward_function.path=examples/xujiawei/tool_bonus_reward.py
    custom_reward_function.name=compute_score

The extra signal counts `<tool_call>` opening tags in `solution_str` -- the
raw rollout text -- as a proxy for how many tool calls the model made. Only
active when SANDBOX_ENDPOINT / SANDBOX_LOCAL_FALLBACK is set so the reward
matches what the tool actually could return.
"""

import os
import re

from verl.utils.reward_score import math_dapo


TOOL_USE_BONUS = float(os.environ.get("TOOL_USE_BONUS", "0.2"))
NO_TOOL_PENALTY = float(os.environ.get("NO_TOOL_PENALTY", "0.1"))

_TOOL_CALL_RE = re.compile(r"<tool_call>", re.IGNORECASE)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """math_dapo strict-box score + tool-use shaping.

    Returns {"score": ..., "acc": ..., "pred": ..., "n_tool_calls": ...}.
    The `acc` field stays 0/1 so val metrics (val-core/*/acc/mean@N) remain a
    clean accuracy signal. Only `score` (the training reward) is shaped.
    """
    base = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    if not isinstance(base, dict):
        base = {"score": float(base), "acc": float(base) > 0, "pred": None}
    if base.get("pred") is None:
        base["pred"] = "[INVALID]"

    n_calls = len(_TOOL_CALL_RE.findall(solution_str or ""))
    base["n_tool_calls"] = n_calls

    correct = base.get("acc")
    shaped = base["score"]
    if n_calls > 0:
        shaped += TOOL_USE_BONUS
    else:
        shaped -= NO_TOOL_PENALTY
    # clamp to [-1, 1] so advantage normalization stays stable
    base["score"] = max(-1.0, min(1.0, shaped))
    return base
