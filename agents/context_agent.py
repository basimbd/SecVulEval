"""
Inputs (after Planning Agent)
----------------------
clean_code      : str                    - Normalised source of the target
compact_ast     : dict | None            - Compact AST, may be None
planning_plan   : dict                   - Output of Planning Agent
backend         : retrieval backend obj  - must implement symbol-fetch methods

Return (dict)
-------------
{
  "context"      : { name: {kind, code, file, line} },
  "ctx_complete" : bool,
  "iterations"   : int
}
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional
from utils import get_wrapped_code
from load_llm import get_remote_llm_result

class ContextAgent:
    def __init__(
        self,
        *,
        backend,
        normaliser=None,
        loop_limit: int = 3,
        model: str = "gpt-4.1",
        temperature: float = 0.1,
        max_ctx_chars: int = 11_264,
        pipe = None,
    ) -> None:
        self.backend       = backend
        self.normaliser    = normaliser
        self.loop_limit    = loop_limit
        self.model         = model
        self.temperature   = temperature
        self.max_ctx_chars = max_ctx_chars

        self.pipe = pipe

    # ─────────────── public entry point ──────────
    def __call__(
        self,
        *,
        clean_code: str,
        compact_ast: Optional[Dict[str, Any]],
        planning_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctx: Dict[str, Dict[str, Any]] = {}
        # pending: List[Dict[str, str]] = []

        for iteration in range(1, self.loop_limit + 1):
            # ask LLM if more context needed
            pending = self._ask_llm(
                clean_code=clean_code,
                checklist=planning_plan["checklist"],
                collected=ctx,
            )
            # fetch pending symbols via backend
            for sym in pending:
                name, kind = sym["name"], sym["kind"]
                if name in ctx:
                    continue
                fetched = self._fetch_symbol(name, kind)
                if fetched:
                    if self.normaliser:
                        fetched["code"] = self.normaliser.clean_code(fetched["code"])
                    ctx[name] = {"kind": kind, **fetched}

            if not pending:
                return {"context": ctx, "ctx_complete": True, "iterations": iteration}

        return {"context": ctx, "ctx_complete": False, "iterations": self.loop_limit}

    # ───────────── symbol retrieval ───────────
    def _fetch_symbol(self, name: str, kind: str):
        kind = kind.lower()
        try:
            if kind == "function":
                return self.backend.get_def(name)
            if kind == "macro":
                return self.backend.get_macro(name)
            if kind == "global":
                return self.backend.get_global(name)
            if kind == "typedef":
                return self.backend.get_typedef(name)
            if kind == "struct":
                return self.backend.get_struct(name)
            if kind == "enum":
                return self.backend.get_enum(name)
        except AttributeError:
            return None
        return None

    # ───────────── LLM interaction ────────────
    _SYS_PROMPT = (
        "You are assisting in *defensive* C/C++ vulnerability analysis.\n"
        "Given the target function and the context already collected, decide "
        "whether additional external symbols are needed to detect a "
        "vulnerability.  External symbols can be functions, "
        "macros, global variables, typedefs, structs, or enums.\n\n"
        "Return JSON ONLY:\n"
        "{{\n  \"need_more\": bool,\n  \"new_symbols\": [ {{name, kind}}, ... ]\n}}\n"
        "If need_more is false, new_symbols must be an empty array.  List at "
        "most 6 symbols and never repeat ones already provided. You cannot get indefinite "
        "context from the user, so choose the most important context you need for "
        "vulnerability detection in this function.\n\n"
    )

    _USR_TEMPLATE = (
        "=== Target Function (might be truncated if too long) ===\n{code}\n\n"
        "=== Existing Context Symbols ===\n{ctx}\n\n"
        "=== Checklist Hints ===\n{checklist}\n"
    )

    def _ask_llm(self, *, clean_code, checklist, collected):
        preview_ctx = textwrap.shorten("; ".join(collected.keys()), width=120)
        user_msg = self._USR_TEMPLATE.format(
            code=textwrap.shorten(clean_code, width=800),
            ctx=preview_ctx or "<none>",
            checklist="; ".join(checklist),
        )
        messages = [
            {"role": "system", "content": self._SYS_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        if not self.pipe:
            txt = get_remote_llm_result(self.model, messages, self.temperature, save_path="cntxt_agent_raw.jsonl")
        else:
            BATCH_SIZE = 1
            resp = self.pipe(messages, temperature=self.temperature, do_sample=True, batch_size=BATCH_SIZE)
            txt = resp[0]["generated_text"][-1]['content'].strip()
        
        json_string = txt
        if "```" in txt:
            json_string = get_wrapped_code(txt, lang="json")
            if not json_string:
                json_string = get_wrapped_code(txt, lang="")
        
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError:
            return []  # fail safe
        if not data.get("need_more"):
            return []
        symbols = data.get("new_symbols", [])
        out = [s for s in symbols if isinstance(s, dict) and {"name", "kind"} <= s.keys()]
        return out[:6]  # enforce max length
