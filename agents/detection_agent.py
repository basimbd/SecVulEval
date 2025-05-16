"""
Inputs (kwargs expected by __call__)
------------------------------------
clean_code   : str
compact_ast  : dict
summary      : str
checklist    : list[str]
context      : dict[str,dict]    # {symbol : {"code":..., "file":..., ...}}

Output
------
{
  "is_vulnerable"   : true | false,
  "vuln_statements" : [ {"line":17,"statement":"strcpy(buf, src);",
                         "reason":"unsafe copy"} ]
}
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from utils import get_wrapped_code
from load_llm import get_remote_llm_result

class DetectionAgent:
    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        temperature: float = 0.1,
        max_code_lines: int = 400,
        max_ast_chars: int = 51_200,
        max_ctx_chars: int = 11_264,
        pipe=None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_code_lines = max_code_lines
        self.max_ast_chars = max_ast_chars
        self.max_ctx_chars = max_ctx_chars

        self.pipe = pipe

    # ───────── public entry point ─────
    def __call__(
        self,
        *,
        clean_code: str,
        compact_ast: Dict[str, Any],
        summary: str,
        checklist: List[str],
        context: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        messages = self._build_messages(
            clean_code, compact_ast, summary, checklist, context
        )
        raw = self._invoke_llm(messages)
        retry = 1
        while retry > 0:
            try:
                return self._parse(raw)
            except Exception as exc:
                retry -= 1
                print(f"Retrying LLM call: {exc}")
                print(f"Raw LLM output: {raw}")
                raw = self._invoke_llm(messages)
        return {"is_vulnerable": False, "vuln_statements": []}

    # ───────── prompt building ────────
    _SYS = (
        "You are a C/C++ security analyst. Decide whether the given function "
        "contains a vulnerability *defensively* (no exploit "
        "instructions).  If vulnerable, list the minimal set of source "
        "statements that directly cause the issue. "
        "Only flag as vulnerable if you are absolutely sure. "
        "Not everything needs checks, so be sure to only identify the ones "
        "that can cause issues according to the other contexts given to you.\n\n"
        "Return JSON ONLY, schema:\n"
        '{{\n "is_vulnerable": <bool>,\n'
        ' "vuln_statements": [\n'
        '   {{"line": <0-based>, "statement": "<raw text>", "reason": "<why>"}}\n'
        ' ]\n}}\n'
        "Return an empty array if `is_vulnerable` is false."
    )

    def _build_messages(
        self,
        code: str,
        ast: Dict[str, Any],
        summary: str,
        checklist: List[str],
        context: Dict[str, Dict[str, Any]],
    ):
        numbered = self._number_lines(code)
        ast_str = self._truncate(json.dumps(ast, separators=(",", ":")), self.max_ast_chars)
        ctx_str = self._truncate(
            json.dumps(context, separators=(",", ":")), self.max_ctx_chars
        )
        user = (
            "### Function (line-numbered):\n"
            f"{numbered}\n\n"
            "### Compact AST (could be truncated if very long):\n"
            f"{ast_str}\n\n"
            "### Function summary from Planning Agent:\n"
            f"{summary}\n\n"
            "### Common Pitfall checklist for this function:\n"
            + "\n".join(f"- {c}" for c in checklist[:8])
            + "\n\n"
            "### Helpful external context definitions:\n"
            f"{ctx_str}\n"
        )

        return [
            {
                "role": "system",
                "content": self._SYS,
            },
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _number_lines(code: str) -> str:
        return "\n".join(f"{i:4d}: {line}" for i, line in enumerate(code.splitlines()))

    @staticmethod
    def _truncate(txt: str, limit: int) -> str:
        return txt if len(txt) <= limit else txt[: limit] + " …[truncated]…"

    # ───────── LLM call ───────────────
    def _invoke_llm(self, messages):
        if not self.pipe:
            return get_remote_llm_result(self.model, messages, self.temperature, save_path="det_agent_raw.jsonl")
        BATCH_SIZE = 1
        resp = self.pipe(messages, temperature=self.temperature, do_sample=True, batch_size=BATCH_SIZE)
        return resp[0]["generated_text"][-1]['content'].strip()

    # ───────── output validation ───────
    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        json_string = raw
        if "```" in raw:
            json_string = get_wrapped_code(raw, lang="json")
            if not json_string:
                json_string = get_wrapped_code(raw, lang="")
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as exc:
            raise ValueError("Detection Agent returned invalid JSON") from exc

        if "is_vulnerable" not in data or "vuln_statements" not in data:
            raise ValueError("Missing required keys in Detection output")

        if not isinstance(data["vuln_statements"], list):
            raise ValueError("`vuln_statements` must be a list")

        # normalise: ensure each entry has line & statement
        normd = []
        for e in data["vuln_statements"]:
            if not {"line", "statement"}.issubset(e):
                continue
            normd.append(
                {"line": int(e["line"]), "statement": e["statement"], "reason": e.get("reason", "")}
            )
        data["vuln_statements"] = normd
        return data
