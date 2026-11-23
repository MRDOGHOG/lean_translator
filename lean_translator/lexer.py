"""A fast, forgiving tokenizer for Lean 4 surface syntax.

It never raises on weird input: anything it does not recognise becomes a
one-character ``sym`` token.  Comments are dropped, but doc comments
(``/-- ... -/``) are kept because they become training targets for the
neural refiner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Token", "tokenize"]


@dataclass(slots=True)
class Token:
    kind: str  # ident | num | str | char | sym | doc | moddoc
    text: str
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}:{self.text!r}@{self.line}:{self.col}"


# Multi-character symbols, longest first.  Order matters only among
# symbols that share a prefix, so we sort by length at import time.
_SYMBOLS = sorted(
    """
    := :: => -> <- <-> <;> <|> <| |> |>. >>= ≠ ≤ ≥ ↔ → ← ↦ ∀ ∃ ∃! ¬ ∧ ∨ ∘ ∈ ∉ ⊆ ⊂ ⊇ ⊃ ∩ ∪ ⋂ ⋃
    ∑ ∏ ∫ ∮ ⨆ ⨅ ⊔ ⊓ ⊤ ⊥ ∅ ᶜ • ∣ ∤ ⁻¹' ⁻¹ '' ≃ ≅ ≡ ⟨ ⟩ ⟪ ⟫ ⦃ ⦄ ‖ ↑ ⇑ ↥ ∂ ∞ × ⊗ ⊕ ∆ √ π
    ≃+* ≃* ≃+ ≃ₗ ≃o →+* →* →+ →ₗ →o →ₙ ↪ ↠ ⧸ ⁅ ⁆ ∠ ∡ ⬝ ∗ ⋆ ≪ ≫ ⋖ ⩿ ∼ ≈ ≺ ≼ ⊢
    ∑' ∏' ⨁ ⊗ₜ ≫ ⋙ ⬝ᵥ ⊣ ⇒ ⟶ ⥤ ≌ ⊠ ⊞ ▸ ≺ ⟮ ⟯
    @[ {{ }} ++ ** // && || == != <= >= ^^ .. ...
    """.split(),
    key=len,
    reverse=True,
)

# Characters that may start an identifier: letters (incl. Greek, script and
# blackboard letters), underscore and a few mathematical letter-likes.
_ID_START = r"[A-Za-z_À-ɏͰ-Ͽἀ-῿℀-⅏\U0001D400-\U0001D7FF]"
_ID_REST = r"[A-Za-z0-9_'!?À-ɏͰ-Ͽἀ-῿℀-⅏₀-ₜᵢ-ᵪ\U0001D400-\U0001D7FF]"
# λ and Π / Σ are keywords-ish symbols in Lean, not identifier starts.
_IDENT_PART = rf"(?:«[^»]*»|(?![λΠΣ]){_ID_START}{_ID_REST}*)"
# Superscript / subscript modifier letters that decorate operators, e.g. the
# `₊` in `‖x‖₊`, `ᵥ` in `*ᵥ`, `ᵀ` (transpose), `ᵒᵖ` (opposite).
_MOD = "[\u02B0-\u02FF\u1D2C-\u1DBF\u2070-\u207F\u208A-\u209C]"
_IDENT = rf"{_IDENT_PART}(?:\.{_IDENT_PART}|\.[0-9]+)*"

_MASTER = re.compile(
    "|".join(
        [
            r"(?P<ws>[ \t\r]+)",
            r"(?P<nl>\n)",
            r"(?P<lcomment>--[^\n]*)",
            r"(?P<bcomment>/-)",  # handled manually (nesting)
            r'(?P<str>"(?:[^"\\]|\\.)*")',
            r"(?P<char>'(?:[^'\\\n]|\\.)')",
            r"(?P<num>0x[0-9a-fA-F]+|0b[01]+|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            rf"(?P<ident>{_IDENT})",
            r"(?P<bracket>[()\[\]{}⟨⟩⦃⦄⟪⟫])",
            "(?P<sym>(?:" + "|".join(re.escape(s) for s in _SYMBOLS) + ")" + _MOD + "*)",
            r"(?P<other>." + _MOD + "*)",
        ]
    ),
    re.S,
)


def _skip_block_comment(src: str, i: int) -> int:
    """Return the index just past the block comment starting at ``src[i]``."""
    depth = 0
    n = len(src)
    while i < n:
        if src.startswith("/-", i):
            depth += 1
            i += 2
        elif src.startswith("-/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return n


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    line = 1
    line_start = 0
    match = _MASTER.match
    while i < n:
        m = match(src, i)
        kind = m.lastgroup
        text = m.group()
        col = i - line_start
        if kind == "nl":
            i += 1
            line += 1
            line_start = i
            continue
        if kind in ("ws", "lcomment"):
            i = m.end()
            continue
        if kind == "bcomment":
            j = _skip_block_comment(src, i)
            body = src[i:j]
            if body.startswith("/--"):
                tokens.append(Token("doc", body[3:-2].strip(), line, col))
            elif body.startswith("/-!"):
                tokens.append(Token("moddoc", body[3:-2].strip(), line, col))
            nls = body.count("\n")
            if nls:
                line += nls
                line_start = i + body.rfind("\n") + 1
            i = j
            continue
        if kind == "str":
            nls = text.count("\n")
            if nls:
                line += nls
                line_start = i + text.rfind("\n") + 1
        elif kind in ("other", "bracket"):
            kind = "sym"
        tokens.append(Token(kind, text, line, col))
        i = m.end()
    return tokens
