"""Split a Lean file into commands (declarations, `open`, `#check`, ...) and
find the tactic lines inside proofs, with exact line ranges, so pieces can be
deleted and the file put back together."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lean_translator.decls import DECL_KEYWORDS, MODIFIERS, OTHER_COMMANDS
from lean_translator.lexer import Token, tokenize

PREFIX = MODIFIERS | {"@["}
DEBUG = {"#check", "#eval", "#print", "#reduce", "#exit", "#lint", "#find", "#synth", "#simp", "#norm_num", "#help"}
# Never touched: deleting these changes how the rest of the file is read.
STRUCTURAL = {"import", "module", "section", "namespace", "end", "mutual", "prelude", "public"}


@dataclass
class Span:
    start: int  # 1-based first line (inclusive)
    end: int  # 1-based last line (inclusive)
    head: str  # "theorem", "open", "#check", "attribute", ...
    kind: str  # decl | debug | attribute | structural | other
    name: str = ""  # declaration name as written (may be empty)
    tokens: list[Token] = field(default_factory=list)
    by_line: int | None = None  # line holding the proof's top-level `by`, if any

    @property
    def idents(self) -> set[str]:
        return {t.text for t in self.tokens if t.kind == "ident"}


def _is_start(toks: list[Token], i: int) -> bool:
    t = toks[i]
    if t.col != 0:
        return False
    if t.kind == "doc" or t.kind == "moddoc":
        return True
    if t.text == "#" and i + 1 < len(toks) and toks[i + 1].kind == "ident":
        return True
    return t.text in DECL_KEYWORDS or t.text in MODIFIERS or t.text in OTHER_COMMANDS or t.text in STRUCTURAL


def _head(toks: list[Token]) -> tuple[str, int]:
    """First token that isn't a docstring, attribute or modifier."""
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind in ("doc", "moddoc"):
            i += 1
        elif t.text == "@[":
            depth = 0
            while i < len(toks):
                if toks[i].text in ("@[", "["):
                    depth += 1
                elif toks[i].text == "]":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            i += 1
        elif t.text in MODIFIERS:
            i += 1
        elif t.text == "#" and i + 1 < len(toks):
            return "#" + toks[i + 1].text, i
        else:
            return t.text, i
    return "", len(toks)


def split_commands(src: str) -> list[Span]:
    toks = tokenize(src)
    lines = src.split("\n")
    starts = [i for i in range(len(toks)) if _is_start(toks, i)]
    raw: list[list[Token]] = []
    for a, b in zip(starts, starts[1:] + [len(toks)]):
        raw.append(toks[a:b])
    # Merge docstring/attribute/modifier-only chunks into the command they precede.
    groups: list[list[Token]] = []
    carry: list[Token] = []
    for chunk in raw:
        head, _ = _head(carry + chunk)
        if not head:
            carry += chunk
            continue
        groups.append(carry + chunk)
        carry = []
    if carry:
        groups.append(carry)

    spans: list[Span] = []
    for g in groups:
        head, hi = _head(g)
        start = g[0].line
        # Claim `--` comment lines directly above (no blank line in between).
        while start > 1 and lines[start - 2].lstrip().startswith("--") and not lines[start - 2].startswith("--!"):
            start -= 1
        if head == "example":
            kind = "debug"
        elif head in DECL_KEYWORDS:
            kind = "decl"
        elif head in DEBUG or head.startswith("#"):
            kind = "debug" if head in DEBUG else "other"
        elif head == "attribute":
            kind = "attribute"
        elif head in STRUCTURAL or (head == "noncomputable" and hi + 1 < len(g) and g[hi + 1].text == "section"):
            kind = "structural"
        else:
            kind = "other"
        name = ""
        if kind == "decl" and hi + 1 < len(g) and g[hi + 1].kind == "ident":
            name = g[hi + 1].text
        by_line = None
        if kind == "decl":
            depth = 0
            for k, t in enumerate(g):
                if t.text in "([{⟨":
                    depth += 1
                elif t.text in ")]}⟩":
                    depth -= 1
                elif depth == 0 and t.text == ":=":
                    if k + 1 < len(g) and g[k + 1].text == "by":
                        by_line = g[k + 1].line
                    break
        spans.append(Span(start, g[-1].line, head, kind, name, g, by_line))

    # A command ends where the next begins (minus trailing blank lines).
    for s, nxt in zip(spans, spans[1:]):
        end = nxt.start - 1
        while end > s.end and not lines[end - 1].strip():
            end -= 1
        s.end = max(s.end, end)
    if spans:
        end = len(lines)
        while end > spans[-1].end and not lines[end - 1].strip():
            end -= 1
        spans[-1].end = max(spans[-1].end, end)
    return spans


def tactic_units(span: Span, lines: list[str]) -> list[tuple[int, int]]:
    """Line ranges of individual tactics in a `by` proof: a line plus the
    more-indented lines under it.  Only proofs whose `by` ends its line."""
    if span.by_line is None or lines[span.by_line - 1].rstrip().split()[-1] != "by":
        return []
    units = []
    ln = span.by_line + 1
    while ln <= span.end:
        text = lines[ln - 1]
        if not text.strip() or text.lstrip().startswith("--"):
            ln += 1
            continue
        indent = len(text) - len(text.lstrip())
        end = ln
        while end + 1 <= span.end:
            nxt = lines[end]
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            end += 1
        units.append((ln, end))
        ln = end + 1
    return units


_SIMP_ARGS = re.compile(r"\b(simp|simp_all|simp only|dsimp|rw|simp_rw|nlinarith|linarith|aesop|norm_num|field_simp|positivity|omega)(\??!?)\s*(only\s*)?\[([^\[\]]*)\]")


def lemma_args(line: str) -> list[tuple[int, int]]:
    """Character ranges of single arguments in `simp [a, b, c]`-style lists, each
    including one adjacent comma so the list stays well-formed when removed."""
    out = []
    for m in _SIMP_ARGS.finditer(line):
        if m.group(1) == "rw":
            continue  # rewrite order matters; dropping one is rarely what's wanted
        inner_start = m.start(4)
        parts, depth, cur = [], 0, inner_start
        for k, ch in enumerate(m.group(4)):
            pos = inner_start + k
            if ch in "([{⟨":
                depth += 1
            elif ch in ")]}⟩":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append((cur, pos))
                cur = pos + 1
        parts.append((cur, m.end(4)))
        if len(parts) == 1:
            continue  # `simp [x]` -> `simp []` is handled as plain line deletion's job
        for k, (a, b) in enumerate(parts):
            if k + 1 < len(parts):
                out.append((a, parts[k + 1][0]))  # arg + following comma
            else:
                out.append((parts[k - 1][1], b))  # preceding comma + last arg
    return out
