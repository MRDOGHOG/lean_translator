"""Lightweight multi-stage Lean 4 -> English translator.

Stage 1 (`gloss`): deterministic parser + rule-based glosser, Lean -> "Lean-English".
Stage 2 (`refiner`): small seq2seq transformer, Lean-English -> fluent English.
"""

from .decls import extract_decls
from .gloss import gloss_decl, gloss_name

__all__ = ["extract_decls", "gloss_decl", "gloss_name", "translate"]


def translate(src: str) -> list[str]:
    """Stage-1 translation of a Lean source string, one gloss per declaration."""
    return [gloss_decl(d).gloss for d in extract_decls(src)]
