"""
Re-checks Detection Agent's verdict.

Input (kwargs):
    clean_code      str
    compact_ast     dict
    summary         str
    checklist       list[str]
    context         dict           # Context Agent bag
    detection_out   dict           # Detection Agent JSON

Output (dict):
    {
        "agree": <bool>,             # with detection_out['is_vulnerable']
        "is_vulnerable": <bool>,
        "vuln_statements": [ {line:int, statement:str, reason:str}, ... ]
    }
"""

from __future__ import annotations
import json, os
from typing import Any, Dict, List
from utils import get_wrapped_code
from load_llm import get_remote_llm_result

class ValidationAgent:
    _SYS = (
        "You are a senior C/C++ security reviewer.\n"
        "Given the code, AST, planning cues and the previous "
        "Detection Agent's verdict whether vulnerable or not, decide whether you **agree**.\n"
        "If you disagree, provide your own corrected list of vulnerable "
        "statements. Only flag as vulnerable if you are absolutely sure. "
        "Not everything needs checks, so be sure to only identify the ones "
        "that can cause issues according to the other contexts given to you.\n\n"
        "Respond with JSON ONLY:\n"
        '{\n  "agree": <bool>,                  # do you agree with detection?\n'
        '  "is_vulnerable": <bool>,\n'
        '  "vuln_statements": '
        '[{"line":<int>, "statement":"...", "reason":"..."}]\n}\n'
        "Keep vuln_statements empty if is_vulnerable is false."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_code_chars: int = 11_264,
        max_ast_chars: int = 51_200,
        max_ctx_chars: int = 11_264,
        pipe=None,
    ) -> None:
        self.model = model
        self.temperature    = temperature
        self.max_code_chars = max_code_chars
        self.max_ast_chars  = max_ast_chars
        self.max_ctx_chars  = max_ctx_chars

        self.pipe = pipe

    # ─────────── call ───────────
    def __call__(
        self,
        *,
        clean_code: str,
        compact_ast: Dict[str, Any],
        summary: str,
        checklist: List[str],
        context: Dict[str, Dict],
        detection_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages = self._build_msgs(
            clean_code, compact_ast, summary,
            checklist, context, detection_out
        )
        if not self.pipe:
            raw = get_remote_llm_result(self.model, messages, temperature=self.temperature, save_path="val_agent_raw.jsonl")
        else:
            BATCH_SIZE = 1
            resp = self.pipe(messages, temperature=self.temperature, do_sample=True, batch_size=BATCH_SIZE)
            raw = resp[0]["generated_text"][-1]['content'].strip()
        output = self._parse(raw)
        if not output:
            output = detection_out
            output["agree"] = True
        return output

    # ───────── helper: prompt building ─────────
    def _build_msgs(
        self, code, ast, summary, checklist, ctx, det_out
    ):
        def trunc(txt, n): return txt if len(txt) <= n else txt[:n] + "…"

        user = (
            "### Code (truncated)\n" + trunc(code, self.max_code_chars) +
            "\n\n### AST (truncated JSON)\n" +
            trunc(json.dumps(ast, separators=(",", ":")), self.max_ast_chars) +
            "\n\n### Planning summary\n" + summary +
            "\n\n### Checklist cues\n" + "\n".join(f"- {c}" for c in checklist) +
            "\n\n### Context (truncated)\n" +
            trunc(json.dumps(ctx, separators=(",", ":")), self.max_ctx_chars) +
            "\n\n### Detection Agent output\n" +
            json.dumps(det_out, indent=2)
        )
        return [
            {"role": "system", "content": self._SYS},
            {"role": "user",   "content": user},
        ]

    # ───────── helper: JSON parse & sanity check ─────────
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
            # raise ValueError("Validation Agent: invalid JSON") from exc
            return None

        for key in ("agree", "is_vulnerable", "vuln_statements"):
            if key not in data:
                raise ValueError(f"Validation Agent: missing key {key}")
        if not isinstance(data["vuln_statements"], list):
            data["vuln_statements"] = []
        return data
