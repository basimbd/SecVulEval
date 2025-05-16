"""symbol_backend.py
====================
Backend used by the Context Agent to fetch symbol definitions at an arbitrary
<repo, commit>.

Implements the following retrieval methods expected by the Context Agent
-----------------------------------------------------------------------
get_def(name)     → full function definition
get_macro(name)   → one-line #define
get_global(name)  → global variable declaration/definition
get_typedef(name) → typedef statement
get_struct(name)  → full struct definition
get_enum(name)    → full enum definition

Return schema (or None if not found):
    {
        "code": <source_snippet>,
        "file": <absolute_path>,
        "line": <0-based start line>
    }

Key design points
-----------------
* **Tree-sitter import style** ― per project preference:
    ```python
    import tree_sitter_c, tree_sitter_cpp
    from tree_sitter import Language, Parser
    TS_LANG_C   = Language(tree_sitter_c.language())
    TS_LANG_CPP = Language(tree_sitter_cpp.language())
    c_parser    = Parser(TS_LANG_C)
    cpp_parser  = Parser(TS_LANG_CPP)
    ```
* **Lazy SQLite + LRU** - first miss parses with Tree-sitter, then caches in
  a warm store (SQLite) plus a hot in-memory LRU.
* **Blob-SHA dedup** - identical files across commits share the same cache row
  (dedup by `git rev-parse <commit>:<file>` output).
* **No checkout** - reads file text via `git cat-file -p <sha>`.
"""
from __future__ import annotations

import re
import sqlite3
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Tree-sitter imports (user-preferred style) ---------------------------
import tree_sitter_c  # type: ignore
import tree_sitter_cpp  # type: ignore
from tree_sitter import Language, Parser

TS_LANG_C   = Language(tree_sitter_c.language())  # pylint: disable=no-member
TS_LANG_CPP = Language(tree_sitter_cpp.language())  # pylint: disable=no-member
C_PARSER     = Parser(TS_LANG_C)
CPP_PARSER   = Parser(TS_LANG_CPP)

# --- regex helpers --------------------------------------------------------
RGX_GLOBAL  = re.compile(r"^\s*(?:extern\s+)?[\w\s\*]+\b(?P<name>\w+)\b\s*(?:[=;{])", re.M)
RGX_MACRO   = lambda name: re.compile(rf"^\s*#\s*define\s+{re.escape(name)}\b[^\n]*", re.M)

_SOURCE_EXTS = [
    "*.c", "*.h", "*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx",
]

# ── kind-specific grep patterns (PCRE, single-line) ─────────────────────
def _grep_pattern(kind: str, name: str) -> str:
    escaped = re.escape(name)
    if kind == "function":
        # Matches first line of a def:   [modifiers] <ret> <name>(
        return rf"^\s*(?:static\s+|inline\s+|extern\s+|__\w+__\s+)*[\w\*\s]+\b{escaped}\s*\("
    if kind == "macro":
        return rf"^\s*#\s*define\s+{escaped}\b"
    if kind == "global":
        return rf"^\s*(?:extern\s+)?[\w\s\*\[\]]+\b{escaped}\b\s*(?:[=;{{])"
    if kind == "typedef":
        return rf"^\s*typedef\b[^\n]*\b{escaped}\b"
    if kind == "struct":
        return rf"^\s*(?:typedef\s+)?struct\s+\b{escaped}\b"
    if kind == "enum":
        return rf"^\s*(?:typedef\s+)?enum\s+\b{escaped}\b"
    return escaped  # fallback

# -------------------------------------------------------------------------
class SymbolBackend:
    """Lazy-parsing, cached symbol retriever (C & C++)."""

    # SQLite schema — warm store
    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS sym("    # repo, sha, kind, name → unique row
        "repo TEXT, sha TEXT, kind TEXT, name TEXT, "
        "path TEXT, line INT, code TEXT, "
        "PRIMARY KEY(repo, sha, kind, name))"
    )

    _KINDS = {"function", "macro", "global", "typedef", "struct", "enum"}

    # ------------------------------------------------------------------
    def __init__(
        self,
        repo: str | Path,
        commit: str,
        *,
        db_path: str | Path = "symcache.sqlite",
        lru_size: int = 20_000,
    ) -> None:
        self.repo   = str(Path(repo).resolve())
        self.commit = commit
        self.db     = sqlite3.connect(str(db_path))
        self.db.execute(self._SCHEMA)
        self._hot_get = lru_cache(maxsize=lru_size)(self._warm_get)

    # ------- public API (Context Agent expects these names) -------------
    def get_def(self, name: str):     return self._lookup(name, "function")
    def get_macro(self, name: str):   return self._lookup(name, "macro")
    def get_global(self, name: str):  return self._lookup(name, "global")
    def get_typedef(self, name: str): return self._lookup(name, "typedef")
    def get_struct(self, name: str):  return self._lookup(name, "struct")
    def get_enum(self, name: str):    return self._lookup(name, "enum")

    # ------------------ core lookup (lazy & cached) ---------------------
    def _lookup(self, name: str, kind: str) -> Optional[Dict[str, Any]]:
        if kind not in self._KINDS:
            return None

        for file_path, line_no in self._grep_candidates(name, kind):
            sha = self._blob_sha(file_path)
            cached_row = self._hot_get(self.repo, sha, kind, name)
            if cached_row:
                return cached_row

            source = self._git_show(sha)
            rec    = self._extract_symbol(source, file_path, name, kind, line_no)
            if rec:                                 # verified definition
                self._warm_put((self.repo, sha, kind, name), rec)
                self._hot_get.cache_clear()
                return rec
        return None

    # ---------------- warm-store helpers (SQLite) -----------------------
    def _warm_get(self, repo: str, sha: str, kind: str, name: str):
        row = self.db.execute(
            "SELECT path,line,code FROM sym WHERE repo=? AND sha=? AND kind=? AND name=?",
            (repo, sha, kind, name),
        ).fetchone()
        if row:
            path, line, code = row
            return {"file": path, "line": line, "code": code}
        return None

    def _warm_put(self, key: Tuple[str, str, str, str], rec: Dict[str, Any]):
        self.db.execute(
            "REPLACE INTO sym VALUES (?,?,?,?,?,?,?)",
            (*key, rec["file"], rec["line"], rec["code"]),
        )
        self.db.commit()

    # --------------------- git helpers ----------------------------------
    def _grep_candidates(self, name: str, kind: str) -> list[tuple[str, int]]:
        pattern = _grep_pattern(kind, name)
        cmd = ["git", "-C", self.repo, "grep", "-nP", "--full-name",
            "-e", pattern, self.commit, "--", *_SOURCE_EXTS]
        out = subprocess.run(cmd, capture_output=True, text=True, errors='replace').stdout.strip()

        results = []
        if out:
            for line in out.splitlines():
                _, f, n, _ = line.split(":", 3)
                results.append((f, int(n) - 1))
            return results          # fast path hit!

        # ── fallback: any file that mentions the name ──
        cmd = ["git", "-C", self.repo, "grep", "-l", "--full-name",
            name, self.commit, "--", *_SOURCE_EXTS]
        files = subprocess.run(cmd, capture_output=True, text=True, errors='replace').stdout.splitlines()
        return [(f.split(":", 1)[1] if ":" in f else f, 0) for f in files]      # line_hint=0 → Tree-sitter will search
        # return [(f, 0) for f in files]         

    @lru_cache(maxsize=100_000)
    def _blob_sha(self, path: str) -> str:
        cmd = ["git", "-C", self.repo, "rev-parse", f"{self.commit}:{path}"]
        return subprocess.check_output(cmd, text=True).strip()

    def _git_show(self, sha: str) -> str:
        return subprocess.check_output(["git", "-C", self.repo, "cat-file", "-p", sha], text=True, errors='replace')

    # ---------------- symbol extraction engine -------------------------
    def _extract_symbol(
        self,
        src: str,
        path: str,
        name: str,
        kind: str,
        line_hint: int,
    ):
        if kind == "macro":
            m = re.search(rf"^\s*#\s*define\s+{re.escape(name)}\b[^\n]*", src, re.M)
            return self._build_rec(self.repo, path, src, *m.span()) if m else None

        if kind == "global":
            for m in RGX_GLOBAL.finditer(src):
                if m.group("name") == name:
                    return self._build_rec(self.repo, path, src, *m.span())
            return None

        parser = (
            CPP_PARSER
            if path.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"))
            else C_PARSER
        )
        tree = parser.parse(src.encode())
        root = tree.root_node

        finder = {
            "function": self._find_function,
            "typedef":  self._find_typedef,
            "struct":   self._find_struct,
            "enum":     self._find_enum,
        }[kind]
        node = finder(root, name, line_hint)
        if not node:
            return None
        return {
            "code": src[node.start_byte : node.end_byte],
            "file": str(Path(self.repo, path)),
            "line": node.start_point[0],
        }

    # ── Utility helpers -----------------------------------------------------
    def _build_rec(self, repo: str, path: str, src: str, start: int, end: int):
        return {
            "code": src[start:end].rstrip(),
            "file": str(Path(repo, path)),
            "line": src[:start].count("\n"),
        }

    # ── Tree-sitter node finders (with line-hint speed-up) -----------------
    def _node_at_line(self, root, line: int):
        """Return smallest node that starts at `line` (fast narrow)."""
        queue = [root]
        while queue:
            n = queue.pop()
            if n.start_point[0] == line:
                return n
            queue.extend(c for c in n.children if c.start_point[0] <= line)
        return None


    def _find_function(self, root, ident: str, line_hint: int):
        n = self._node_at_line(root, line_hint)
        while n:
            if n.type == "function_definition":
                for child in n.children:
                    if child.type == "function_declarator":
                        for leaf in child.children:
                            if leaf.type == "identifier" and leaf.text.decode() == ident:
                                return n
            n = n.parent
        return None

    def _find_typedef(self, root, ident: str, line_hint: int):
        n = self._node_at_line(root, line_hint)
        while n:
            if n.type == "type_definition":
                # find the declarator list under this typedef
                decl_list = next((c for c in n.children 
                                if c.type == "init_declarator_list"), None)
                if decl_list:
                    # each child is an init_declarator → walk its subtree
                    for init_decl in decl_list.children:
                        for leaf in init_decl.walk():
                            if leaf.type in ("type_identifier", "identifier") and leaf.text.decode() == ident:
                                return n
            n = n.parent
        return None

    def _find_struct(self, root, ident: str, line_hint: int):
        n = self._node_at_line(root, line_hint)
        while n:
            if n.type == "struct_specifier":
                ids = [c for c in n.children if c.type == "type_identifier"]
                if ids and ids[0].text.decode() == ident:
                    return n
            n = n.parent
        return None

    def _find_enum(self, root, ident: str, line_hint: int):
        n = self._node_at_line(root, line_hint)
        while n:
            if n.type == "enum_specifier":
                ids = [c for c in n.children if c.type == "type_identifier"]
                if ids and ids[0].text.decode() == ident:
                    return n
            n = n.parent
        return None
