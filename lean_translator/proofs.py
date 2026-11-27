"""Proof narrator: Lean tactic proofs -> English steps, by rules (no AI).

Each tactic line becomes one sentence from a template ("intro x" -> "Let x
be arbitrary."), and the formulas inside `have` / `show` / `suffices` go
through the same logic-rule glosser as the statements.  Nesting follows the
source: a `have ... := by` block or a `·` bullet becomes an indented
sub-list, so the English keeps the proof's structure.

    from lean_translator.proofs import narrate
    for depth, sentence in narrate(decl): ...
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .decls import Decl
from .gloss import Glosser
from .lexer import Token
from .parser import _detok, parse

__all__ = ["narrate", "Step", "UNKNOWN"]

UNKNOWN: Counter = Counter()  # tactics with no template, for coverage reports

_OPEN = {"(", "[", "{", "⟨", "⦃", "{{", "@["}
_CLOSE = {")", "]", "}", "⟩", "⦄", "}}"}
_MAX_MATH = 140  # longer formulas are shortened in the prose


@dataclass
class Step:
    col: int
    toks: list[Token]
    children: list["Step"] = field(default_factory=list)
    bullet: bool = False


# ---------------------------------------------------------------- structure
def _lines(toks: list[Token]) -> list[list[Token]]:
    out: list[list[Token]] = []
    for t in toks:
        if out and out[-1][0].line == t.line:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def _depth(toks: list[Token]) -> int:
    d = 0
    for t in toks:
        if t.text in _OPEN:
            d += 1
        elif t.text in _CLOSE:
            d -= 1
    return d


_STARTERS = {"·", ".", "|", "_", "case", "next", "<;>", "all_goals", "any_goals", "on_goal"}


def _units(toks: list[Token]) -> list[Step]:
    """Group the tokens of a tactic block into one Step per tactic, with columns."""
    units: list[Step] = []
    depth = 0
    for group in _lines(toks):
        first = group[0]
        cur = units[-1] if units else None
        bullet_body = (cur is not None and cur.toks[0].text in ("·", ".") and len(cur.toks) > 1
                       and first.col <= cur.toks[1].col)
        cont = cur is not None and not bullet_body and (
            depth > 0
            or (first.col > cur.col and cur.toks[-1].text not in ("by", "=>", "·", "do", "from")
                and first.text not in _STARTERS and not _opens_block(cur))
        )
        if cont:
            cur.toks.extend(group)
        else:
            units.append(Step(first.col, list(group)))
        depth = max(0, depth + _depth(group))
    # split `a; b` and bullets `· tac`
    out: list[Step] = []
    for u in units:
        out.extend(_split_unit(u))
    return out


def _opens_block(u: Step) -> bool:
    """Does this tactic take an indented block on the following lines (calc, induction ... with)?"""
    head = u.toks[0].text
    if head in ("calc", "conv", "classical"):
        return True
    return any(t.text == "with" for t in u.toks) and head in ("induction", "cases", "rcases", "match", "induction'")


def _split_unit(u: Step) -> list[Step]:
    toks = u.toks
    if toks and toks[0].text in ("·", "."):
        rest = toks[1:]
        b = Step(u.col, [], bullet=True)
        if rest:
            b.children = [Step(rest[0].col, s.toks) for s in _split_semis(rest)]
        return [b]
    return [Step(u.col, s.toks) for s in _split_semis(toks)]


def _split_semis(toks: list[Token]) -> list[Step]:
    parts, cur, d = [], [], 0
    for t in toks:
        if t.text in _OPEN:
            d += 1
        elif t.text in _CLOSE:
            d -= 1
        if t.text == ";" and d == 0:
            if cur:
                parts.append(Step(cur[0].col, cur))
            cur = []
            continue
        cur.append(t)
    if cur:
        parts.append(Step(cur[0].col, cur))
    return parts


def _tree(units: list[Step]) -> list[Step]:
    root = Step(-1, [])
    stack = [root]
    for u in units:
        while len(stack) > 1 and stack[-1].col >= u.col:
            stack.pop()
        stack[-1].children.append(u)
        stack.append(u)
        if u.bullet and u.children:  # inline content of the bullet sits deeper than the dot
            stack.append(Step(u.children[-1].col, [], ))
            stack.pop()
    return root.children


# ---------------------------------------------------------------- language
class _Math:
    def __init__(self, g: Glosser, tex: bool = False) -> None:
        self.g = g
        self.tex = tex  # formulas become \x01<latex>\x02 instead of plain text

    def raw(self, toks: list[Token]) -> str:
        s = _detok(toks).strip()
        return s if len(s) <= _MAX_MATH else s[: _MAX_MATH - 1] + "…"

    def code(self, toks: list[Token]) -> str:
        return f"`{self.raw(toks)}`" if toks else ""

    def prop(self, toks: list[Token]) -> str:
        if not toks:
            return ""
        try:
            n = parse(toks)
            if self.tex:
                return self._prop_tex(n) or self.code(toks)
            s = self.g.prop(n) if self.g.is_prop(n) else self.g.term(n)
        except Exception:
            return self.code(toks)
        if not s or len(s) > 2 * _MAX_MATH or "Raw" in s:
            return self.code(toks)
        return s

    def _prop_tex(self, n) -> str:
        from .gloss import _SLOT_RE
        from .latex import tex_math

        g = self.g
        g.masking, g.slots, g.slot_nodes = True, [], []
        try:
            s = g.prop(n) if g.is_prop(n) else g.term(n)
        finally:
            g.masking = False
        nodes = list(g.slot_nodes)
        return _SLOT_RE.sub(lambda mm: "\x01" + tex_math(nodes[int(mm.group(1))]) + "\x02", s)

    def formula(self, toks: list[Token]) -> str:
        """Pure math (calc steps): typeset in LaTeX mode, code otherwise."""
        if not self.tex or not toks:
            return self.code(toks)
        from .latex import tex_math
        try:
            s = tex_math(parse(toks), set(self.g.locals))
            if toks[0].text == "_" and s.startswith("\\cdot "):  # calc continuation `_ ≤ b`
                s = "\\qquad " + s[len("\\cdot "):]
            return "\x01" + s + "\x02"
        except Exception:
            return self.code(toks)

    def names(self, toks: list[Token]) -> list[str]:
        return [t.text for t in toks if t.kind == "ident" and t.text not in ("with", "at", "using", "only", "_")]


def _split_at(toks: list[Token], word: str) -> tuple[list[Token], list[Token] | None]:
    d = 0
    for i, t in enumerate(toks):
        if t.text in _OPEN:
            d += 1
        elif t.text in _CLOSE:
            d -= 1
        elif d == 0 and t.text == word:
            return toks[:i], toks[i + 1 :]
    return toks, None


def _list(items: list[str]) -> str:
    items = [x for x in items if x]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _bracket_items(toks: list[Token], m: _Math) -> list[str]:
    """`[a, ← b, c x]` -> ["`a`", "`b` (right to left)", "`c x`"]."""
    if not toks or toks[0].text != "[":
        return []
    inner, d, cur, out = toks[1:], 0, [], []
    for t in inner:
        if t.text in _OPEN:
            d += 1
        elif t.text in _CLOSE:
            if d == 0:
                break
            d -= 1
        if t.text == "," and d == 0:
            out.append(cur)
            cur = []
            continue
        cur.append(t)
    if cur:
        out.append(cur)
    res = []
    for c in out:
        if c and c[0].text in ("←", "<-"):
            res.append(f"{m.code(c[1:])} backwards")
        elif c:
            res.append(m.code(c))
    if len(res) > 6:
        res = res[:5] + [f"{len(res) - 5} more facts"]
    return res


def _loc(toks: list[Token], m: _Math) -> str:
    _, at = _split_at(toks, "at")
    if at is None:
        return ""
    names = [t.text for t in at if t.kind == "ident" or t.text == "⊢"]
    if not names or names == ["*"]:
        return " everywhere"
    return " in " + _list([("the goal" if n == "⊢" else f"`{n}`") for n in names])


_AUTO = {
    "linarith": "by linear arithmetic", "nlinarith": "by nonlinear arithmetic",
    "positivity": "since the expression is positive", "ring": "by algebra (both sides are equal as polynomials)",
    "ring_nf": "after normalising the algebra", "field_simp": "after clearing denominators",
    "omega": "by integer arithmetic", "decide": "by direct computation", "norm_num": "by numerical computation",
    "simp_arith": "by arithmetic simplification", "aesop": "by automatic proof search",
    "grind": "by automatic reasoning", "tauto": "by propositional logic", "trivial": "trivially",
    "rfl": "since both sides are equal by definition", "exact?": "by a library lemma",
    "assumption": "by one of our hypotheses", "contradiction": "since the hypotheses are contradictory",
    "continuity": "since the function is continuous", "fun_prop": "by the rules for continuity and differentiability",
    "measurability": "since everything is measurable", "infer_instance": "by the known structure",
    "gcongr": "by monotonicity (comparing term by term)", "bound": "by standard bounds",
    "polyrith": "by a polynomial identity", "abel": "by rearranging sums", "group": "by group algebra",
    "noncomm_ring": "by (noncommutative) algebra", "module": "by linear algebra", "linear_combination":
    "as a linear combination of the hypotheses", "norm_cast": "after moving the casts",
    "push_cast": "after pushing casts inward", "exact_mod_cast": "up to casts", "apply_mod_cast": "up to casts",
    "simp_all": "by simplifying everything", "subsingleton": "since there is only one such object",
    "compute_degree": "by computing the degree", "interval_cases": "by checking each value",
    "fin_cases": "by checking each case", "nlinarith!": "by nonlinear arithmetic",
}


def _sentence(toks: list[Token], m: _Math) -> str:
    if not toks:
        return ""
    head = toks[0].text.rstrip("!?")
    rest = toks[1:]
    if rest and rest[0].text in ("!", "?"):
        rest = rest[1:]
    code = m.code

    if head in ("intro", "intros", "rintro"):
        names = [n for n in m.names(rest)]
        m.g.locals.update(names)
        if not names:
            return "Introduce the hypotheses."
        hyps = [n for n in names if n.startswith("h") or n.startswith("H")]
        objs = [n for n in names if n not in hyps]
        parts = []
        if objs:
            parts.append(f"let {_list(objs)} be arbitrary")
        if hyps:
            parts.append(f"assume {_list([f'`{h}`' for h in hyps])}")
        return _cap(" and ".join(parts)) + "."
    if head in ("obtain", "rcases", "rcases'", "cases'", "choose"):
        pat, src = _split_at(rest, ":=")
        if src is None:
            pat, src = _split_at(rest, "with")
            src, pat = pat, src or []
        names = m.names(pat)
        m.g.locals.update(names)
        what = _list([f"`{n}`" for n in names]) or "the pieces"
        if pat and any(t.text == ":" for t in pat[:2]) is False and pat and pat[0].text == "⟨":
            pass
        return f"From {code(src)} obtain {what}." if src else f"Obtain {what}."
    if head in ("have", "haveI", "replace", "let", "set"):
        lhs, proof = _split_at(rest, ":=")
        name_toks, stmt = _split_at(lhs, ":")
        names = m.names(name_toks)
        label = f" (call it `{names[0]}`)" if names else ""
        m.g.locals.update(names)
        if head == "set" or head == "let":
            return f"Let {code(name_toks)} denote {m.prop(proof or [])}." if proof else f"Let {code(lhs)}."
        if stmt is None:  # `have h := foo x`
            return f"Note {code(proof or [])}{label}."
        how = ""
        if proof and proof[0].text != "by":
            how = f", by {code(proof)}"
        elif proof and len(proof) > 1:
            how = ", " + _how(proof[1:], m)
        return f"We show that {m.prop(stmt)}{label}{how}."
    if head in ("show", "change"):
        stmt, _ = _split_at(rest, "from")
        return f"We need to show that {m.prop(stmt)}." if head == "show" else f"Rewrite the goal as {m.prop(stmt)}."
    if head == "suffices":
        lhs, _ = _split_at(rest, "by")
        name_toks, stmt = _split_at(lhs, ":")
        return f"It suffices to show that {m.prop(stmt if stmt is not None else name_toks)}."
    if head in ("exact", "exacts"):
        return f"This follows from {code(rest)}."
    if head in ("apply", "refine", "refine'", "apply_rules"):
        if rest and rest[0].text == "⟨":
            return f"Build the required object piece by piece: {code(rest)}."
        return f"Apply {code(rest)}; it remains to check its hypotheses."
    if head in ("rw", "rewrite", "rwa", "nth_rewrite", "nth_rw", "erw", "simp_rw"):
        items = _bracket_items(rest if rest and rest[0].text == "[" else rest[1:], m)
        what = _list(items) or code(rest)
        tail = ", and conclude by a hypothesis" if head == "rwa" else ""
        return f"Rewrite using {what}{_loc(rest, m)}{tail}."
    if head in ("simp", "dsimp", "simpa", "simp_all", "norm_num", "field_simp", "simp?", "simpa?"):
        _, using = _split_at(rest, "using")
        args = [t for t in rest if True]
        if args and args[0].text == "only":
            args = args[1:]
        items = _bracket_items(args, m)
        verb = {"norm_num": "Compute", "field_simp": "Clear denominators", "dsimp": "Unfold definitions"}.get(head, "Simplify")
        s = verb + (f" using {_list(items)}" if items else "") + _loc(rest, m)
        if using is not None:
            s += f"; this gives {code(using)}"
        return s + "."
    if head in ("unfold", "delta"):
        return f"Unfold the definition of {_list([f'`{n}`' for n in m.names(rest)])}{_loc(rest, m)}."
    if head in ("constructor", "split_ands"):
        return "Prove each part separately."
    if head in ("use", "exists", "existsi"):
        return f"Take {code(rest)}."
    if head == "left":
        return "Prove the first alternative."
    if head == "right":
        return "Prove the second alternative."
    if head in ("induction", "induction'"):
        v, _ = _split_at(rest, "with")
        v, _ = _split_at(v, "using")
        return f"Argue by induction on {code(v)}."
    if head in ("cases", "rcases", "match", "split"):
        return f"Split into cases according to {code(rest)}." if rest else "Split into cases."
    if head in ("by_contra", "by_contra!", "by_contra'"):
        names = m.names(rest)
        return f"Suppose, for a contradiction, the opposite{f' (call it `{names[0]}`)' if names else ''}."
    if head == "by_cases":
        names, stmt = _split_at(rest, ":")
        return f"Consider separately whether {m.prop(stmt if stmt is not None else names)} or not."
    if head in ("contrapose", "contrapose!"):
        return "Prove the contrapositive instead."
    if head in ("push_neg",):
        return f"Push the negation inward{_loc(rest, m)}."
    if head in ("exfalso",):
        return "It is enough to reach a contradiction."
    if head in ("absurd",):
        return f"This contradicts {code(rest)}."
    if head in ("ext", "funext", "ext1", "Set.ext"):
        names = m.names(rest)
        return f"By extensionality it suffices to compare them at an arbitrary {_list([f'`{n}`' for n in names]) or 'point'}."
    if head in ("congr", "congr!", "congr_arg", "congr 1", "congrm"):
        return "Compare both sides piece by piece."
    if head == "specialize":
        return f"Specialise {code(rest)}."
    if head in ("calc",):
        rel, proof = _split_at(rest, ":=")
        if not rel:
            return "We compute:"
        how = (_how(proof[1:], m) if proof and proof[0].text == "by" else "by " + code(proof or [])) if proof else ""
        return f"We compute: {m.formula(rel)}" + (f" ({how})" if how else "")
    if head == "|":  # an alternative of induction / cases / match
        pat, tac = _split_at(rest, "=>")
        names = m.names(pat)
        m.g.locals.update(names[1:])
        label = f"Case `{_detok(pat).strip()}`"
        if tac:
            return f"{label}: {_lower(_sentence(tac, m))}"
        return f"{label}:"
    if head == "_" and m:
        rel, proof = _split_at(rest, ":=")
        return f"{m.formula(toks[:1] + rel)}" + (f" ({_how(proof[1:], m) if proof and proof[0].text == 'by' else 'by ' + code(proof or [])})" if proof else "")
    if head in ("symm", "symmetry"):
        return "By symmetry, swap the two sides."
    if head in ("trans", "transitivity"):
        return f"Go through the intermediate value {code(rest)}." if rest else "Use transitivity."
    if head == "subst":
        return f"Substitute {code(rest)}."
    if head in ("filter_upwards",):
        return f"It holds eventually, combining {_list(_bracket_items(rest, m)) or 'the hypotheses'}."
    if head in ("all_goals", "any_goals"):
        return f"In every remaining case: {_lower(_sentence(rest, m))}"
    if head in ("case", "next"):
        return f"Case {code(rest[:1])}:" if rest else "Next case:"
    if head in ("classical", "skip", "clear", "clear_value", "rename_i", "with_reducible", "exact_mod_cast?"):
        return ""
    if head in ("conv", "conv_lhs", "conv_rhs"):
        return "Rewrite inside a part of the expression:"
    if head in _AUTO:
        return _cap(_AUTO[head]) + "."
    # `foo <;> bar`
    left, right = _split_at(toks, "<;>")
    if right is not None:
        return f"{_sentence(left, m).rstrip('.')}, then on every resulting goal: {_lower(_sentence(right, m))}"
    UNKNOWN[head] += 1
    return f"Apply {code(toks)}."


def _how(toks: list[Token], m: _Math) -> str:
    """Inline `by tac` -> "by linear arithmetic" / "simplifying"."""
    if not toks:
        return ""
    head = toks[0].text.rstrip("!?")
    if head in _AUTO and len(toks) == 1:
        return _AUTO[head]
    s = _sentence(toks, m)
    return "since: " + _lower(s.rstrip("."))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _lower(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


# ---------------------------------------------------------------- public
def narrate(d: Decl, g: Glosser | None = None, tex: bool = False) -> list[tuple[int, str]]:
    """English steps of a theorem's proof as (depth, sentence).  With `tex`, formulas are
    marked as \x01<LaTeX>\x02 for the LaTeX writer."""
    toks = d.proof
    if not toks:
        return []
    g = g or Glosser()
    g.locals.update(x for b in d.binders + d.context for x in b.names)
    m = _Math(g, tex)
    head = toks[0].text
    if head == "|":
        return [(0, "By cases on the shape of the input (one equation per case).")]
    if head != ":=" or len(toks) < 2:
        return []
    body = toks[1:]
    if body[0].text != "by":
        return [(0, f"This follows directly from {m.code(body)}.")]
    tac = body[1:]
    if not tac:
        return []
    out: list[tuple[int, str]] = []

    def walk(steps: list[Step], depth: int) -> None:
        for s in steps:
            if s.bullet:
                out.append((depth, "Case:" if s.children else ""))
                walk(s.children, depth + 1)
                continue
            text = _sentence(s.toks, m)
            if text:
                out.append((depth, text))
            walk(s.children, depth + 1)

    walk(_tree(_units(tac)), 0)
    return [(dp, s) for dp, s in out if s]
