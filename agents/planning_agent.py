"""
Inputs  (from Normalization Agent)
----------------------------------
clean_code   : str
compact_ast  : dict

Output
------
{
  "summary"      : str,        # ≤ 3 sentence overview
  "checklist"    : [str, …],   # ≤ 8 concise red-flag cues
}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from transformers import pipeline
from load_llm import get_remote_llm_result
from utils import get_wrapped_code

class PlanningAgent:
    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        temperature: float = 0.1,
        max_code_chars: int = 11_264,   # truncate oversized inputs
        max_ast_chars: int = 77_824,
        pipe: pipeline = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_code_chars = max_code_chars
        self.max_ast_chars = max_ast_chars

        self.pipe = pipe

    # ─────────────── public entry point ──────────
    def __call__(
        self,
        *,
        clean_code: str,
        compact_ast: Dict[str, Any]
    ) -> Dict[str, Any]:
        messages = self._build_messages(clean_code, compact_ast)
        raw_json = self._invoke_llm(messages)
        plan     = self._validate_plan(raw_json)
        return plan

    # ─────────────── prompt templates ────────────
    _SYSTEM_PROMPT = (
        "You are a senior C/C++ *defensive* security researcher.\n"
        "Your task: devise an initial **analysis plan** to detect whether a "
        "vulnerability exists in the function supplied by the user.\n\n"
        "Respond with **JSON only** - no markdown, no explanations outside the JSON.\n"
        "Required keys:\n"
        '  "summary"      : ≤ 3-sentence plain-language overview of the function.\n'
        '  "checklist"    : array (≤ 8) of concise red-flag cues (code patterns, '
        "data-flow hints, etc.) that indicate the existence of a vulnerability.\n"
        "Constraints:\n"
        "- Focus exclusively on *detection* and *prevention* - do NOT produce exploitation steps.\n"
        "- Do not invent symbols that are absent from the code or AST.\n"
        "- Prefer brevity; omit speculative items if unsure.\n"
    )

    _USER_TEMPLATE = (
        "=== Function Source Code (could be truncated if too long) ===\n"
        "{clean}\n"
        "=== Compact AST JSON (could be truncated if too long) ===\n"
        "{ast}\n"
    )

    # ─────────────── helpers ─────────────────────
    def _build_messages(
        self,
        clean_code: str,
        compact_ast: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        trimmed_code = (
            clean_code[: self.max_code_chars] + " …[truncated]…"
            if len(clean_code) > self.max_code_chars
            else clean_code
        )
        ast_json = json.dumps(compact_ast, separators=(",", ":"))
        trimmed_ast = (
            ast_json[: self.max_ast_chars] + " …[truncated]…"
            if len(ast_json) > self.max_ast_chars
            else ast_json
        )
        system_msg = {
            "role": "system",
            "content": self._SYSTEM_PROMPT,
        }
        user_msg = {
            "role": "user",
            "content": self._USER_TEMPLATE.format(
                clean=trimmed_code, ast=trimmed_ast
            ),
        }
        return [system_msg, user_msg]

    def _invoke_llm(self, messages: List[Dict[str, str]]) -> str:
        if not self.pipe:
            return get_remote_llm_result(self.model, messages, save_path="plan_agent_raw.jsonl")
        BATCH_SIZE = 1
        output = self.pipe(messages, batch_size=BATCH_SIZE)
        return output[0]["generated_text"][-1]['content']

    @staticmethod
    def _validate_plan(raw: str) -> Dict[str, Any]:
        """Ensure the LLM's response is valid and complete JSON."""
        # print(f"PlanningAgent: {raw}")
        json_string = raw
        if "```" in raw:
            json_string = get_wrapped_code(raw, lang="json")
            if not json_string:
                json_string = get_wrapped_code(raw, lang="")

        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON:\n{json_string}") from exc

        required = {"summary", "checklist"}
        missing  = required - data.keys()
        if missing:
            raise ValueError(f"Planning JSON missing keys {missing}:\n{json_string}")

        if not isinstance(data["checklist"], list) or len(data["checklist"]) > 8:
            raise ValueError("`checklist` must be a list of ≤ 8 strings.")

        return data
