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
            or (first.col > cur.col and first.text in ("(", "[", "⟨", ",", "←"))  # argument wrapped over lines
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
        if not toks:
            return ""
        if not self.tex:
            return f"`{self.raw(toks)}`"
        # Paper style: never print raw Lean.  A local name stays (as a label); anything bigger is
        # described by the results it uses.
        if len(toks) == 1 and toks[0].kind == "ident" and toks[0].text in self.g.locals:
            return f"`{toks[0].text}`"
        c = self.cite(toks)
        if c:
            return c
        ids = [t.text for t in toks if t.kind == "ident" and t.text not in _NOT_NAMES]
        if ids and all(x in self.g.locals for x in ids):
            return _list([f"`{x}`" for x in dict.fromkeys(ids)])
        return "a direct computation"

    def cite(self, toks: list[Token]) -> str:
        """`foo (bar x) h` -> "Lemma 3.2 and Lemma 1.4" / "the standard result Nat.exists_prime_and_dvd"."""
        ours, lib = [], []
        for t in toks:
            if t.kind != "ident" or t.text in self.g.locals or t.text in _NOT_NAMES:
                continue
            name = t.text.lstrip("@")
            if not name or name[:1].isdigit() or name.startswith(".") or name.rsplit(".", 1)[-1].isdigit():
                continue
            full = REFS.get(name) or REFS.get(name.rsplit(".", 1)[-1])
            if full:
                if full not in ours:
                    ours.append(full)
            elif "." in name and name[:1].isupper() and name not in lib and not name.endswith((".mp", ".mpr", ".le")):
                lib.append(name)
            elif "_" in name and name not in lib and len(name) > 6:
                lib.append(name)
        parts = ["\x03" + f + "\x04" for f in ours[:4]]
        if len(ours) > 4:
            parts.append(f"{len(ours) - 4} further results")
        if lib:
            shown = ["\x05" + x + "\x06" for x in lib[:2]]
            parts.append(("the standard result " if len(lib) == 1 else "the standard results ") + _list(shown)
                         + (" and others" if len(lib) > 2 else ""))
        return _list(parts)

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
        return [t.text for t in toks if t.kind == "ident" and t.text not in _NOT_NAMES and t.text != "_"
                and not t.text[:1].isdigit() and not t.text.startswith(".")]


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


_NOT_NAMES = {"fun", "have", "show", "from", "by", "with", "at", "using", "only", "intro", "intros", "by_cases",
              "by_contra", "exact", "apply", "refine", "simp", "rw", "obtain", "let", "if", "then", "else", "match",
              "this", "rfl", "trivial", "id", "le_total", "le_rfl", "le_refl", "lt_irrefl", "absurd", "False.elim",
              "Or.inl", "Or.inr", "And.intro", "Exists.intro", "congrArg", "congr_arg", "funext", "mt", "fst", "snd"}
REFS: dict[str, str] = {}  # Lean name (full or short) -> full name of a result in the book (set by the writer)
STATEMENTS: dict[str, Decl] = {}  # full name -> declaration, to quote what a cited result gives


def _exists_conclusion(full: str, pattern_names: list[str], m: "_Math") -> str | None:
    """`obtain ⟨d, e, h⟩ := lemma ...` where lemma's statement ends in `∃ x y, P x y`:
    "there exist d and e such that P d e" (bound names replaced by the pattern's)."""
    d = STATEMENTS.get(full)
    if d is None or not d.type:
        return None
    toks = list(d.type)
    # skip leading ∀/→ hypotheses: take the part after the last top-level → if it starts with ∃
    k = next((i for i, t in enumerate(toks) if t.text == "∃"), None)
    if k is None:
        return None
    comma = next((i for i in range(k, len(toks)) if toks[i].text == ","), None)
    if comma is None:
        return None
    bound = [t.text for t in toks[k + 1:comma] if t.kind == "ident" and t.text not in ("in",)]
    colon = [i for i in range(k, comma) if toks[i].text == ":"]
    if colon:
        bound = [t.text for t in toks[k + 1:colon[0]] if t.kind == "ident"]
    body = toks[comma + 1:]
    if body and body[0].text == "∃":  # ∃ x, ∃ y, ...: take one level only
        return None
    new_names = [x for x in pattern_names if not x.startswith("h")][:len(bound)]
    if len(new_names) != len(bound):
        new_names = bound
    ren = dict(zip(bound, new_names))
    from .lexer import Token as _T
    body = [_T(t.kind, ren.get(t.text, t.text), t.line, t.col) if t.kind == "ident" else t for t in body]
    m.g.locals.update(new_names)
    what = m.prop(body)
    return f"there {'is' if len(new_names) == 1 else 'are'} {_list([f'`{x}`' for x in new_names])} such that {what}"


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
        return _obtain(src, names, what, m)
    if head in ("have", "haveI", "replace", "let", "set"):
        lhs, proof = _split_at(rest, ":=")
        name_toks, stmt = _split_at(lhs, ":")
        names = m.names(name_toks)
        label = f" (call it `{names[0]}`)" if names else ""
        m.g.locals.update(names)
        if (head in ("let", "obtain") or head == "have") and name_toks and name_toks[0].text == "⟨" and proof:
            return _obtain(proof, names, _list([f"`{n}`" for n in names]) or "the pieces", m)
        if head == "set" or head == "let":
            shown = _list([f"`{n}`" for n in names]) or code(name_toks)
            return f"Let {shown} denote {m.prop(proof or [])}." if proof else f"Let {shown} be as above."
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
        c = code(rest)
        ids = [t.text for t in rest if t.kind == "ident" and t.text not in _NOT_NAMES]
        if m.tex and len(ids) > 1 and all(x in m.g.locals for x in ids):
            return f"This follows by combining {c}."
        return f"This follows from {c}."
    if head in ("apply", "refine", "refine'", "apply_rules"):
        if rest and rest[0].text == "⟨":
            if m.tex:
                c = code(rest)
                return f"We build the required object from {c}." if c != "a direct computation" else \
                    "We build the required object piece by piece."
            return f"Build the required object piece by piece: {code(rest)}."
        if m.tex:
            return f"We apply {code(rest)}; it remains to verify its hypotheses."
        return f"Apply {code(rest)}; it remains to check its hypotheses."
    if head in ("grw", "nth_grw", "gcongr_rw"):
        items = _bracket_items(rest if rest and rest[0].text == "[" else rest[1:], m)
        return f"Rewrite with the inequalities {_list(items) or code(rest)}{_loc(rest, m)}."
    if head in ("convert", "convert_to"):
        e, using = _split_at(rest, "using")
        return f"This matches {code(e)} up to the differences that remain to be checked."
    if head in ("repeat", "repeat'", "iterate"):
        inner = rest[1:] if rest and rest[0].kind == "num" else rest
        return f"Repeatedly: {_lower(_sentence(inner, m))}" if inner else "Repeat."
    if head == "try":
        return f"If possible: {_lower(_sentence(rest, m))}" if rest else ""
    if head in ("lia", "cutsat"):
        return "By integer arithmetic."
    if head == "peel":
        return f"Strip the matching quantifiers from {code(rest)}."
    if head in ("generalize", "generalize'"):
        return f"Generalise: {code(rest)}."
    if head in ("split_ifs", "split_if"):
        return "Split into cases according to the conditions of the if-then-else expressions."
    if head == "field":
        return "By field arithmetic."
    if head in ("lift",):
        return f"Lift {code(rest)}."
    if head in ("rify", "qify", "zify", "push_cast"):
        target = {"rify": "real numbers", "qify": "rational numbers", "zify": "integers"}.get(head, "")
        return f"Move the statement to {target}." if target else "Push the casts inward."
    if head in ("swap", "rotate", "pick_goal"):
        return "Work on the next goal first."
    if head in ("ac_rfl",):
        return "By associativity and commutativity."
    if head == "apply_fun":
        return f"Apply {code(rest)} to both sides."
    if head == "wlog":
        names, stmt = _split_at(rest, ":")
        stmt, _ = _split_at(stmt or [], "with")
        return f"Without loss of generality, assume {m.prop(stmt) if stmt else code(rest)}."
    if head in ("algebraize",):
        return f"Turn the ring maps {code(rest)} into algebra structures."
    if head in ("aesop_cat", "simp_intro"):
        return "By automatic reasoning." if head == "aesop_cat" else "Introduce the hypotheses and simplify."
    if head in ("seal", "unseal"):
        return ""
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
        c = code(rest)
        if m.tex and ("\x03" in c or "standard" in c):
            return f"We take the witness provided by {c}."
        return f"Take {c}."
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
        first = pat[0].text if pat else ""
        if first in ("zero", "Nat.zero") and len(pat) == 1:
            label = "Base case"
        elif first in ("succ", "Nat.succ") and len(pat) >= 2:
            label = f"Inductive step (from `{pat[1].text}` to `{pat[1].text}` + 1)"
        else:
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
        if m.tex:
            phrase = _AUTO[head]
            return ("This follows " + phrase + ".") if phrase.startswith(("by", "since", "as", "after", "up")) \
                else _cap(phrase) + "."
        return _cap(_AUTO[head]) + "."
    # `foo <;> bar`
    left, right = _split_at(toks, "<;>")
    if right is not None:
        return f"{_sentence(left, m).rstrip('.')}, then on every resulting goal: {_lower(_sentence(right, m))}"
    UNKNOWN[head] += 1
    return f"Apply {code(toks)}."


def _obtain(src: list[Token] | None, names: list[str], what: str, m: "_Math") -> str:
    if not src:
        return f"Obtain {what}."
    if m.tex:
        head = next((t.text for t in src if t.kind == "ident"), "")
        full = REFS.get(head) or REFS.get(head.rsplit(".", 1)[-1])
        if full:
            concl = _exists_conclusion(full, names, m)
            hyp = [n for n in names if n.startswith("h")]
            tag = f" (call this `{hyp[-1]}`)" if hyp and concl else ""
            if concl:
                return f"By \x03{full}\x04, {concl}{tag}."
            return f"By \x03{full}\x04 we obtain {what}."
        c = m.cite(src)
        return f"By {c} we obtain {what}." if c else f"We obtain {what}."
    return f"From {m.code(src)} obtain {what}."


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
    intro: list[tuple[int, str]] = []
    if body[0].text in ("fun", "λ"):  # `fun x y ↦ by tac` / `fun x ↦ term`
        arrow = next((k for k, t in enumerate(body) if t.text in ("↦", "=>")), None)
        if arrow is not None and arrow + 1 < len(body):
            names = m.names(body[1:arrow])
            g.locals.update(names)
            if names:
                intro.append((0, f"Let {_list([f'`{n}`' for n in names])} be arbitrary."))
            body = body[arrow + 1:]
    if body[0].text != "by":
        return intro + [(0, f"This follows directly from {m.code(body)}.")]
    tac = body[1:]
    if not tac:
        return []
    out: list[tuple[int, str]] = []

    def walk(steps: list[Step], depth: int) -> None:
        n_bullets = sum(1 for s in steps if s.bullet)
        k = 0
        for s in steps:
            if s.bullet:
                k += 1
                if n_bullets > 1:
                    out.append((depth, f"Case {k}:" if s.children else ""))
                    walk(s.children, depth + 1)
                else:  # one focused goal: no case split, keep the steps at this level
                    walk(s.children, depth)
                continue
            text = _sentence(s.toks, m)
            if text:
                out.append((depth, text))
            walk(s.children, depth + 1)

    walk(_tree(_units(tac)), 0)
    return intro + [(dp, s) for dp, s in out if s]


# ---------------------------------------------------------------- named statements
KNOWN_PROPS = {
    "RiemannHypothesis": (
        "every zero s of the Riemann zeta function that is not a trivial zero (s ≠ -2, -4, -6, …) "
        "and is not s = 1 has real part 1/2.",
        r"every zero $s$ of the Riemann zeta function $\zeta$ that is not a trivial zero "
        r"($s \neq -2, -4, -6, \dots$) and is not $s = 1$ has real part $\operatorname{Re}(s) = \frac{1}{2}$."),
}


def prop_name(d: Decl) -> str | None:
    """If the statement is just a name (`theorem t : RiemannHypothesis`), that name."""
    if len(d.type) == 1 and d.type[0].kind == "ident":
        return d.type[0].text
    return None


def meanings(decls: list[Decl]) -> dict[str, Decl]:
    """Project definitions whose value is a statement, by full and short name."""
    out: dict[str, Decl] = {}
    for d in decls:
        if d.kind in ("def", "abbrev") and d.body and d.full_name and len(d.type) == 1 and d.type[0].text == "Prop":
            out[d.full_name] = d
            out.setdefault(d.full_name.rsplit(".", 1)[-1], d)
    return out
