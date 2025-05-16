"""normalization_agent.py
--------------------------------------------------
A self-contained NormalizationAgent for the C/C++ vulnerability-detection
pipeline.  It cleans/normalises source code *and* serialises a Tree-sitter
AST so that downstream LLM agents have both the textual form and a
structural representation to reference.

Key design choices (unchanged)
==============================
1. **Comments kept by default.**  Developers often leave security-relevant
hints in comments ("// TODO: sanity-check length"), so *keep_comments*
defaults to **True**.
2. **Macro expansion optional.**  A full pre-processor pass can break when
headers are missing.  Therefore we *only* expand macros if the caller
explicitly asks for it **and** supplies a *clang_path*.
3. **Header-agnostic AST.**  Tree-sitter parses *syntactically* without
resolving includes, so missing headers are not a blocker.
4. **Single shared parser.**  The compiled language library is cached at
module import time to avoid recompilation overhead across agent calls.

Usage Example
-------------
```python
agent = NormalizationAgent(keep_comments=True)
result = agent.run(raw_code, file_type="cpp", vuln_type="buffer_overflow")
print(result["src"])
print(result["ast"][:500])  # JSON string (truncated)
```

Dependencies
------------
- `tree_sitter` (pip install tree_sitter)
- Optionally `clang`/`clang++` in $PATH for macro expansion

The class raises informative exceptions rather than swallowing errors, so
that the Orchestrator can decide how to recover.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import tree_sitter_c, tree_sitter_cpp
from tree_sitter import Language, Parser
# from .tree_sitter_parser import TreeSitterParser

# ---------------------------------------------------------------------------
# Build / load Tree-sitter C & C++ grammars exactly **once** per interpreter
# session.  The compiled .so file is stored under ~/.cache to persist across
# runs and keep the first-run penalty small.
# ---------------------------------------------------------------------------
# _TS_LANG_SO = Path.home() / ".cache" / "ts_langs_c_cpp.so"

# if not _TS_LANG_SO.exists():
#     _TS_LANG_SO.parent.mkdir(parents=True, exist_ok=True)
#     Language.build_library(
#         str(_TS_LANG_SO),
#         [
#             str(Path(__file__).with_suffix("").parent / "tree-sitter-c"),
#             str(Path(__file__).with_suffix("").parent / "tree-sitter-cpp"),
#         ],
#     )

TS_LANG_C = Language(tree_sitter_c.language())
TS_LANG_CPP = Language(tree_sitter_cpp.language())


class NormalizationAgent:
    """Clean raw C/C++ code and emit `(src, ast_json)`.

    Parameters
    ----------
    keep_comments
        If *True*, comments are preserved.  If *False*, they are replaced
        by whitespace of equal length to retain line numbering.
    expand_macros
        If *True*, run the file through `clang -E` to expand macros.
        Caller must supply *clang_path*.  Disabled by default because
        missing headers often break expansion.
    clang_path
        Path to the clang executable.  Required if *expand_macros* is True.
    style
        Optional clang‑format style string (e.g. "{BasedOnStyle: llvm, IndentWidth: 4}").
    """

    _COMMENT_PAT = re.compile(r"/\*.*?\*/|//.*?$", re.DOTALL | re.MULTILINE)

    def __init__(
        self,
        *,
        keep_comments: bool = True,
        expand_macros: bool = False,
        clang_path: Optional[str] = None,
        style: Optional[str] = None,
    ) -> None:
        if expand_macros and not clang_path:
            raise ValueError("clang_path must be provided when expand_macros=True")

        self.keep_comments = keep_comments
        self.expand_macros = expand_macros
        self.clang_path = clang_path or shutil.which("clang")
        self.style = style

        self._c_parser = Parser(TS_LANG_C)
        self._cpp_parser = Parser(TS_LANG_CPP)

    # ---------------------------- public API --------------------------------

    def run(
        self,
        src: str,
        *,
        file_type: str,
        vuln_type: str | None = None,
    ) -> Dict[str, Any]:
        """Return normalised text & AST JSON.

        Parameters
        ----------
        src
            Raw C/C++ source code.
        file_type
            Either "c" or "cpp"; selects the grammar **deterministically**.
        vuln_type
            Optional vulnerability label (carried through for convenience).
        """
        file_type = file_type.lower()
        if file_type not in {"c", "cpp", "c++", "cxx"}:
            raise ValueError("file_type must be 'c' or 'cpp'")

        if self.expand_macros:
            src = self._run_clang_preprocessor(src)

        src = self._normalise_whitespace(src)
        src = self._ensure_trailing_newline(src)

        if not self.keep_comments:
            src = self._strip_comments_preserve_lines(src)

        src = self._apply_clang_format(src)

        # ast_json = self._produce_ast_json(src, file_type)

        compact_ast, line_map = self.produce_ast_compact_json(
            src,
            file_type,
            keep_comments=self.keep_comments
        )

        return {
            "src": src,
            # "ast": ast_json,
            "compact_ast": {"ast": compact_ast, "line_map": line_map},
            "vuln_type": vuln_type,
        }

    # ----------------------- normalisation helpers -------------------------

    @staticmethod
    def _normalise_whitespace(code: str) -> str:
        # Convert tabs→4 spaces, unify line endings, strip trailing ws.
        lines = [l.expandtabs(4).rstrip() for l in code.splitlines()]
        return "\n".join(lines)

    @staticmethod
    def _ensure_trailing_newline(code: str) -> str:
        return code if code.endswith("\n") else code + "\n"

    def _strip_comments_preserve_lines(self, code: str) -> str:
        return re.sub(self._COMMENT_PAT, lambda m: " " * (m.end() - m.start()), code)

    # -------------------------- clang helpers ------------------------------

    def _run_clang_preprocessor(self, code: str) -> str:
        """Pipe the snippet through clang -E (macro expansion)."""
        if not self.clang_path:
            raise RuntimeError("clang not found; cannot preprocess")

        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [self.clang_path, "-E", "-P", tmp_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return res.stdout
        finally:
            os.unlink(tmp_path)

    def _apply_clang_format(self, code: str) -> str:
        if self.style is None:
            return code
        clang_format = shutil.which("clang-format")
        if clang_format is None:
            return code
        try:
            res = subprocess.run(
                [clang_format, f"-style={self.style}"],
                input=code,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return res.stdout
        except subprocess.CalledProcessError:
            return code

    # --------------------------- AST helpers -------------------------------

    def _produce_ast_json(self, code: str, file_type: str) -> str:
        parser = self._cpp_parser if file_type in {"cpp", "c++", "cxx"} else self._c_parser
        tree = parser.parse(code.encode("utf8"))
        return json.dumps(self._node_to_dict(tree.root_node))

    def _node_to_dict(self, node) -> Dict[str, Any]:  # recursive
        return {
            "type": node.type,
            "start_point": node.start_point,
            "end_point": node.end_point,
            "children": [self._node_to_dict(c) for c in node.children],
        }
    
    def produce_ast_compact_json(
        self,
        code: str,
        file_type: str,
        *,
        keep_comments: bool = False,
        structural_only: bool = True,
    ) -> tuple[dict, dict[str, int]]:
        """
        Efficiently return a compact AST JSON plus a {tag: line_nr} map.
        """
        code_bytes = code.encode("utf8")
        parser     = self._cpp_parser if file_type in ("cpp","c++","cxx") else self._c_parser
        ts_tree    = parser.parse(code_bytes)
        root       = ts_tree.root_node

        # 1) Locate the actual function_definition node and only traverse it
        func_node = None
        for c in root.named_children:
            if c.type == "function_definition":
                func_node = c
                break
        if func_node is None:
            func_node = root  # fallback to whole tree if no function found

        RELEVANT = {
            "function_definition","if_statement","for_statement","while_statement",
            "switch_statement","call_expression","assignment_expression",
            "return_statement","binary_expression","unary_expression",
            "expression_statement","declaration","break_statement",
            "continue_statement","goto_statement","do_statement",
            "case_statement","default_statement","labeled_statement",
            "asm_statement","throw_statement","try_statement",
            "catch_clause","for_range_loop","else_clause",
        }
        IDLITS = ("identifier","number_literal","string_literal")

        id_ctr   = 0
        line_map: dict[str,int] = {}

        def recurse(node) -> Optional[dict]:
            nonlocal id_ctr

            typ = node.type

            # skip comments if asked
            if not keep_comments and typ == "comment":
                return None

            # prune non‐structural wrappers
            if structural_only and typ not in RELEVANT and typ not in IDLITS:
                out: List[dict] = []
                for child in node.named_children:
                    cobj = recurse(child)
                    if cobj:
                        out.append(cobj)
                # flatten up one level
                return {"child": out} if out else None

            # build this node
            tag = f"{typ}#{id_ctr}"; id_ctr += 1
            ln  = node.start_point[0]
            line_map[tag] = ln

            obj: dict[str,Any] = {"tag": tag, "type": typ, "line": ln}

            # leaf: keep actual lexeme
            if typ in IDLITS:
                lex = code_bytes[node.start_byte:node.end_byte].decode("utf8", "ignore")
                obj["value"] = lex
                return obj

            # internal: recurse children exactly once
            kids: List[dict] = []
            for child in node.named_children:
                cobj = recurse(child)
                if cobj:
                    kids.append(cobj)
            if kids:
                obj["child"] = kids
            return obj

        # try:
        compact_root = recurse(func_node)
        # except MemoryError:
        #     # fallback to minimal stub
        #     compact_root = {"tag":"OOM","type":"error","line":0}
        #     line_map = {}

        return compact_root or {}, line_map


# ----------------------------- CLI demo ------------------------------------
if __name__ == "__main__":
    _SAMPLE = """// vulnerable example\nint foo(char *input) {\n    char buf[10];\n    strcpy(buf, input); // overflow\n    return 0;\n}\n"""

    agent = NormalizationAgent()
    result = agent.run(_SAMPLE, file_type="c", vuln_type="buffer_overflow")

    print("--- Normalised Source ---")
    print(result["src"])

    print("--- AST (first 400 chars) ---")
    print(result["compact_ast"]["ast"])
