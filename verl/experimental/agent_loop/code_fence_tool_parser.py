# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Text-mode tool parser: extracts ```python / ```py fenced code blocks and
returns each as an `execute_python` FunctionCall.

Coexists with the built-in Hermes / gpt-oss / qwen3_coder / gemma4 / search_r1
parsers. Register once by importing this module (done from
`verl/experimental/agent_loop/__init__.py`).

Use by setting `actor_rollout_ref.rollout.multi_turn.format=code_fence` and
providing the `execute_python` tool via `function_tool_path` or
`tool_config_path`.

System-prompt convention:
    - Write a ```python ... ``` block whenever you want to run code. Its stdout
      is fed back to you as the next user message.
    - Your FINAL submission is the LAST ```python ... ``` block you produce
      before the rollout ends (max turns reached or you stop producing blocks).
    - No `<tool_call>` XML or JSON wrapping.

Behaviour: extracts ALL fenced blocks in one response as separate
`execute_python` FunctionCalls, letting `rollout.multi_turn.max_parallel_calls`
gate whether they run in parallel or serially. Set that config to 1 to preserve
the one-block-per-turn semantics from aiic's MathAgent.
"""

from __future__ import annotations

import json
import logging
import os

import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Matches ```python\n...\n``` or ```py\n...\n``` fences (case-insensitive on
# language tag). Uses `regex` (from tool_parser deps) for possessive quantifier
# support and better performance on nested backticks.
_CODE_FENCE_RE = regex.compile(
    r"```[ \t]*(?:python|py)[ \t]*(?:\r?\n)(?P<code>.*?)(?:\r?\n)?```",
    regex.DOTALL | regex.IGNORECASE,
)


@ToolParser.register("code_fence")
class CodeFenceToolParser(ToolParser):
    """Extracts ```python fenced blocks as tool calls.

    Each fenced block becomes one FunctionCall. Non-fence text is preserved in
    the returned content so the model's reasoning stays intact for training.

    Tool name is read from `CODE_FENCE_TOOL_NAME` env var (default: `execute_python`).
    Set this per-run to match the registered tool. Examples:
      - code_multiturn: CODE_FENCE_TOOL_NAME=execute_python (default)
      - math_multiturn: CODE_FENCE_TOOL_NAME=code_interpreter
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_name = os.environ.get("CODE_FENCE_TOOL_NAME", "execute_python")

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] | None = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if "```" not in text:
            return text, []

        function_calls: list[FunctionCall] = []
        for m in _CODE_FENCE_RE.finditer(text):
            code = (m.group("code") or "").strip()
            if not code:
                continue
            function_calls.append(
                FunctionCall(
                    name=self.tool_name,
                    arguments=json.dumps({"code": code}, ensure_ascii=False),
                )
            )

        # Content: original text with fenced blocks stripped so the tool-call
        # step doesn't re-execute them via the model's own generation.
        content = _CODE_FENCE_RE.sub("", text)

        return content, function_calls
