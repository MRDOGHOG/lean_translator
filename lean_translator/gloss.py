"""Stage 1: Lean AST  ->  "Lean-English".

The output is deliberately literal: it keeps the mathematical terms
(``a * (b * c)``) but turns logic, quantifiers, relations, predicates,
type classes and hypotheses into English sentences.  It is correct by
construction (no guessing) but stilted; stage 2 (the neural refiner)
rewrites it into fluent prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import vocab
from .decls import Binder, Decl
from .lexer import Token
from .parser import (
    App, Ascribe, Bin, BinderSpec, Delim, Hole, Ite, Name, Node, Num, Post, Pre,
    Proj, Quant, Raw, SetLit, SetOf, Tuple, parse,
)

__all__ = ["Glosser", "gloss_decl", "gloss_name", "Translation"]

# --- precedence for re-parenthesising terms -------------------------------
_TERM_PREC = {
    "→": 25, "+": 65, "-": 65, "∪": 65, "++": 65, "⊔": 68, "*": 70, "/": 70, "%": 70,
    "∩": 70, "⊓": 69, "•": 73, "^": 75, "∘": 90, "⁻¹'": 80, "''": 80, "×": 35,
}
_ATOM = 1000

RELATIONS = {
    "=": "equals", "≠": "does not equal", "<": "is less than", ">": "is greater than",
    "≤": "is at most", "≥": "is at least", "<=": "is at most", ">=": "is at least",
    "∈": "is in", "∉": "is not in", "⊆": "is a subset of", "⊂": "is a proper subset of",
    "⊇": "is a superset of", "⊃": "is a proper superset of", "∣": "divides",
    "∤": "does not divide", "≡": "is congruent to", "≈": "is approximately",
    "≅": "is isomorphic to", "≃": "is equivalent to", "⋖": "is covered by", "∼": "is related to",
    "==": "equals", "!=": "does not equal", "≪": "is absolutely continuous with respect to",
}
_NEG_REL = {
    "=": "≠", "≠": "=", "∈": "∉", "∉": "∈", "∣": "∤", "∤": "∣",
}
_NOT_REL = {
    "<": "is not less than", ">": "is not greater than", "≤": "is not at most",
    "≥": "is not at least", "⊆": "is not a subset of",
}
LOGIC_OPS = {"↔", "<->", "∧", "∨", "&&", "||"}
_PROP_HEADS = re.compile(r"^(Is|Has)[A-Z]")
_ARROWS = set(vocab.TYPE_ARROWS)


def _last(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _lookup(table: dict[str, str], name: str) -> str | None:
    if name in table:
        return table[name]
    # Try progressively shorter namespace suffixes: `Mathlib.Foo.Nat.Prime` -> `Nat.Prime` -> `Prime`.
    parts = name.split(".")
    for k in range(1, len(parts)):
        cand = ".".join(parts[k:])
        if cand in table:
            return table[cand]
    return None


@dataclass
class Translation:
    name: str
    kind: str
    file: str
    line: int
    lean: str
    gloss: str
    name_gloss: str
    doc: str | None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "file": self.file, "line": self.line,
            "lean": self.lean, "gloss": self.gloss, "name_gloss": self.name_gloss, "doc": self.doc,
        }


class Glosser:
    def __init__(self) -> None:
        self.locals: set[str] = set()
        self.prop_vars: set[str] = set()
        self.type_vars: set[str] = set()

    # ------------------------------------------------------------------ names
    def _split_local(self, n: Name) -> Node:
        """`hp.two_le` / `f.Injective` / `s.card` where the head is a local."""
        ident = n.id
        if "." not in ident or ident.startswith("."):
            return n
        head, rest = ident.split(".", 1)
        if head not in self.locals:
            return n
        node: Node = Name(head)
        for fld in rest.split("."):
            node = Proj(node, fld)
        return node

    def name(self, ident: str) -> str:
        if ident in vocab.FUNCTIONS and "{" not in vocab.FUNCTIONS[ident]:
            return vocab.FUNCTIONS[ident]
        if ident in self.locals:
            return ident
        special = {"True": "true", "False": "false", "⊤": "⊤", "⊥": "⊥", "∅": "∅", "Real.pi": "π", "π": "π"}
        if ident in special:
            return special[ident]
        if ident in vocab.TYPE_NOUNS:
            return ident
        ident = ident.removeprefix("_root_.")
        # Drop namespaces for readability: `Polynomial.aeval` -> `aeval`.
        last = _last(ident)
        if last.isdigit() or last in ("mk", "symm", "toFun", "out", "val", "1", "2", "inv"):
            parts = ident.split(".")
            return ".".join(parts[-2:])
        return last

    # ------------------------------------------------------------------ terms
    def term(self, n: Node, prec: int = 0) -> str:
        s, p = self._term(n)
        return f"({s})" if p < prec else s

    def _term(self, n: Node) -> tuple[str, int]:
        if isinstance(n, Name):
            m = self._split_local(n)
            if m is not n:
                return self._term(m)
            return self.name(n.id), _ATOM
        if isinstance(n, Num):
            return n.value, _ATOM
        if isinstance(n, Hole):
            return "_", _ATOM
        if isinstance(n, Raw):
            return f"`{n.text}`", _ATOM
        if isinstance(n, Ascribe):  # casts: `(x : ℝ)` reads as just `x`
            if isinstance(n.expr, Name) and " " in n.expr.id:  # pi binder
                return f"for {n.expr.id}: {self.type_phrase(n.type)}", 0
            return self._term(n.expr)
        if isinstance(n, Pre):
            if n.op in ("↑", "⇑", "↥"):
                return self._term(n.arg)
            if n.op == "¬":
                return self.prop(n), 40
            if n.op == "#":
                return "#" + self.term(n.arg, _ATOM), _ATOM
            if n.op == "√":
                return "√" + self.term(n.arg, _ATOM), _ATOM
            return n.op + self.term(n.arg, 75), 75
        if isinstance(n, Post):
            op = "!" if n.op == "!" else n.op
            return self.term(n.arg, _ATOM) + op, _ATOM
        if isinstance(n, Bin):
            return self._bin_term(n)
        if isinstance(n, App):
            return self._app_term(n)
        if isinstance(n, Proj):
            if n.field.isdigit():
                return f"{self.term(n.expr, _ATOM)}.{n.field}", _ATOM
            return self._app_term(App(Name(n.field), [n.expr]), proj=True)
        if isinstance(n, Quant):
            return self._quant_term(n)
        if isinstance(n, Ite):
            return f"(if {self.prop(n.cond)} then {self.term(n.then)} else {self.term(n.other)})", _ATOM
        if isinstance(n, Tuple):
            return "(" + ", ".join(self.term(x) for x in n.items) + ")", _ATOM
        if isinstance(n, SetLit):
            return "{" + ", ".join(self.term(x) for x in n.items) + "}", _ATOM
        if isinstance(n, SetOf):
            var = self.term(n.var)
            with self._bind([var]):
                ty = f" in {self.type_phrase(n.type, plural=False)}" if n.type is not None else ""
                what = "the subtype of" if n.subtype else "the set of"
                return f"{what} {var}{ty} such that {self.prop(n.pred)}", 10
        if isinstance(n, Delim):
            inner = ", ".join(self.term(x) for x in n.items)
            return {
                "abs": f"|{inner}|", "norm": f"‖{inner}‖", "list": f"[{inner}]",
                "floor": f"⌊{inner}⌋", "ceil": f"⌈{inner}⌉", "inner": f"⟪{inner}⟫",
                "bracket": f"⁅{inner}⁆",
            }.get(n.kind, inner), _ATOM
        return "?", _ATOM

    def _bin_term(self, n: Bin) -> tuple[str, int]:
        op = n.op
        if op in RELATIONS or op in LOGIC_OPS or op in ("→", "->") and self.is_prop(n.lhs):
            return self.prop(n), 20
        if op == ":=":
            return f"{self.term(n.lhs)} := {self.term(n.rhs)}", 0
        p = _TERM_PREC.get(op.split("[")[0], 60)
        right_assoc = op in ("^", "→", "•", "×")
        lp = p + 1 if right_assoc else p
        rp = p if right_assoc else p + 1
        if op == "⁻¹'":
            return f"{self.term(n.lhs, _ATOM)}⁻¹({self.term(n.rhs)})", _ATOM
        if op == "''":
            return f"{self.term(n.lhs, _ATOM)}({self.term(n.rhs)})", _ATOM
        return f"{self.term(n.lhs, lp)} {op} {self.term(n.rhs, rp)}", p

    def _app_term(self, n: App, proj: bool = False) -> tuple[str, int]:
        fn = n.fn
        if isinstance(fn, Name):
            m = self._split_local(fn)
            if isinstance(m, Proj):  # `f.comp g x` -> comp(f, g, x)
                return self._app_term(App(Name(m.field), [m.expr] + n.args), proj=True)
        args = n.args
        if isinstance(fn, Name):
            ident = fn.id
            if ident in ("↑", "Nat.cast", "Int.cast", "Rat.cast", "ofReal", "Complex.ofReal") and len(args) == 1:
                return self._term(args[0])
            tmpl = _lookup(vocab.FUNCTIONS, ident)
            if tmpl is not None:
                k = _n_slots(tmpl)
                if k <= len(args):
                    head = tmpl.format(*[self.term(a, _ATOM) for a in args[:k]])
                    rest = args[k:]
                    if not rest:
                        if " " not in head:
                            return head, _ATOM
                        # "the real part of s" reads fine beside relations, not beside arithmetic
                        return head, 60 if head.startswith("the ") else 10
                    return f"({head})(" + ", ".join(self.term(a) for a in rest) + ")", _ATOM
            if ident in self.locals and len(ident) <= 2:
                return f"{ident}(" + ", ".join(self.term(a) for a in args) + ")", _ATOM
            head = self.name(ident)
        else:
            head = self.term(fn, _ATOM)
        return f"{head}(" + ", ".join(self.term(a) for a in args) + ")", _ATOM

    def _quant_term(self, n: Quant) -> tuple[str, int]:
        names = [x for b in n.binders for x in b.names]
        with self._bind(names):
            if n.q == "fun":
                return f"(the map {', '.join(names)} ↦ {self.term(n.body)})", _ATOM
            what = {
                "∑": "the sum", "∑'": "the infinite sum", "∏": "the product", "∏'": "the infinite product",
                "⋃": "the union", "⋂": "the intersection", "⨆": "the supremum", "⨅": "the infimum",
                "∫": "the integral", "∫⁻": "the lower integral", "⨍": "the average", "∮": "the contour integral",
                "⨁": "the direct sum", "∑ᶠ": "the sum", "∏ᶠ": "the product", "⨍⁻": "the average",
            }.get(n.q)
            if what is None and n.q[0] in "∑∏∫⨍⋃⋂⨆⨅":
                what = {"∑": "the sum", "∏": "the product", "∫": "the integral", "⨍": "the average",
                        "⋃": "the union", "⋂": "the intersection", "⨆": "the supremum", "⨅": "the infimum"}[n.q[0]]
            if what is None and n.q in ("Σ", "Π"):
                return f"(dependent {'pairs' if n.q == 'Σ' else 'functions'} over {self._binder_range(n.binders, n.extra)} into {self.term(n.body)})", _ATOM
            if what is None and n.q[0] in "∀∃":
                return self.prop(n), 0
            if what is None:
                return f"({n.q} {self._binder_range(n.binders, n.extra)}, {self.term(n.body)})", _ATOM
            rng = self._binder_range(n.binders, n.extra)
            body = self.term(n.body)
            measure = f" with respect to {self.term(n.extra['measure'])}" if "measure" in n.extra else ""
            return f"{what} of {body} over {rng}{measure}", 10

    def _binder_range(self, binders: list[BinderSpec], extra: dict) -> str:
        parts = []
        for b in binders:
            vs = ", ".join(b.names)
            if b.bound_op is not None:
                parts.append(f"{vs} {b.bound_op} {self.term(b.bound)}")
            elif b.type is not None:
                parts.append(f"{vs} in {self.type_phrase(b.type, plural=False, article_=False)}")
            else:
                parts.append(f"all {vs}")
        if "domain" in extra:
            d = extra["domain"]
            if isinstance(d, Bin) and d.op == "..":
                parts[-1] = f"{parts[-1].replace('all ', '')} from {self.term(d.lhs)} to {self.term(d.rhs)}"
            else:
                parts[-1] = f"{parts[-1].replace('all ', '')} in {self.term(d)}"
        return " and ".join(parts)

    # ------------------------------------------------------------------ types
    def type_noun(self, n: Node | None, plural: bool = False) -> str | None:
        """A noun phrase for a type, e.g. `natural numbers`, or None."""
        if n is None:
            return None
        if isinstance(n, Ascribe):
            n = n.expr
        if isinstance(n, Name):
            if n.id in vocab.TYPE_NOUNS:
                sing, plur = vocab.TYPE_NOUNS[n.id]
                return plur if plural else sing
            return None
        if isinstance(n, App) and isinstance(n.fn, Name):
            tmpl = _lookup(vocab.TYPE_CTORS, n.fn.id)
            if tmpl is not None:
                args = [self.type_noun(a, plural=True) or self.term(a, _ATOM) for a in n.args]
                if "{0}" in tmpl and not args:
                    return None
                noun = tmpl.format(*args) if "{0}" in tmpl else tmpl
                return vocab.pluralize(noun) if plural else noun
        if isinstance(n, Bin) and n.op.split("[")[0] in _ARROWS:
            op = n.op.split("[")[0]
            a = self.type_noun(n.lhs, plural=True) or self.term(n.lhs, 26)
            b = self.type_noun(n.rhs, plural=True) or self.term(n.rhs, 25)
            noun = vocab.TYPE_ARROWS[op].format(a, b)
            return vocab.pluralize(noun) if plural else noun
        return None

    def type_phrase(self, n: Node | None, plural: bool = False, article_: bool = True) -> str:
        noun = self.type_noun(n, plural)
        if noun is None:
            return self.term(n) if n is not None else "something"
        if plural or not article_:
            return noun
        return vocab.article(noun)

    # ------------------------------------------------------------------ props
    def is_prop(self, n: Node) -> bool:
        if isinstance(n, Bin):
            if n.op in RELATIONS or n.op in LOGIC_OPS or n.op.split("[")[0].rstrip("ᶠ") in RELATIONS:
                return True
            if n.op in ("→", "->"):
                return self.is_prop(n.rhs)
            return False
        if isinstance(n, Pre):
            return n.op == "¬"
        if isinstance(n, Quant):
            return n.q[0] in "∀∃" or n.q in ("Π",)
        if isinstance(n, Name):
            m = self._split_local(n)
            if m is not n:
                return self.is_prop(m)
            return n.id in self.prop_vars or n.id in ("True", "False")
        if isinstance(n, App) and isinstance(n.fn, Name):
            m = self._split_local(n.fn)
            ident = m.field if isinstance(m, Proj) else n.fn.id
            return _lookup(vocab.PREDICATES, ident) is not None or bool(_PROP_HEADS.match(_last(ident)))
        if isinstance(n, Proj):
            return _lookup(vocab.PREDICATES, n.field) is not None or bool(_PROP_HEADS.match(n.field))
        if isinstance(n, Ascribe):
            return self.is_prop(n.expr)
        return False

    def prop(self, n: Node) -> str:
        if isinstance(n, Bin):
            op = n.op
            if op in ("↔", "<->"):
                return f"{self.prop(n.lhs)} if and only if {self.prop(n.rhs)}"
            if op in ("∧", "&&"):
                return " and ".join(self.prop(x) for x in _flatten(n, op))
            if op in ("∨", "||"):
                return " or ".join(self.prop(x) for x in _flatten(n, op))
            if op in ("→", "->") and (self.is_prop(n.lhs) or isinstance(n.lhs, Name) and n.lhs.id in self.prop_vars):
                hyps = []
                cur: Node = n
                while isinstance(cur, Bin) and cur.op in ("→", "->") and self.is_prop(cur.lhs):
                    hyps.append(self.prop(cur.lhs))
                    cur = cur.rhs
                return f"if {_join_and(hyps)}, then {self.prop(cur)}"
            base = op.split("[")[0]
            if base in RELATIONS:
                return f"{self.term(n.lhs, 51)} {RELATIONS[base]} {self.term(n.rhs, 51)}"
            if base.endswith("ᶠ") and base[:-1] in RELATIONS:
                along = op[op.index("[") + 1 : -1] if "[" in op else "the filter"
                return f"{self.term(n.lhs, 51)} eventually {RELATIONS[base[:-1]]} {self.term(n.rhs, 51)} along {along}"
            return self.term(n)
        if isinstance(n, Pre) and n.op == "¬":
            return self._negate(n.arg)
        if isinstance(n, Quant):
            return self._quant_prop(n)
        if isinstance(n, (App, Proj)):
            s = self._pred(n)
            if s is not None:
                return s
            return self.term(n)
        if isinstance(n, Name):
            m = self._split_local(n)
            if isinstance(m, Proj):
                return self.prop(m)
            if n.id == "True":
                return "true"
            if n.id == "False":
                return "a contradiction (false)"
            return self.term(n)
        if isinstance(n, Ite):
            return f"if {self.prop(n.cond)} then {self.prop(n.then)}, otherwise {self.prop(n.other)}"
        if isinstance(n, Ascribe):
            return self.prop(n.expr)
        return self.term(n)

    def _pred(self, n: Node) -> str | None:
        if isinstance(n, Proj):
            n = App(Name(n.field), [n.expr])
        assert isinstance(n, App)
        fn = n.fn
        args = list(n.args)
        if isinstance(fn, Name):
            m = self._split_local(fn)
            if isinstance(m, Proj):
                fn = Name(m.field)
                args = [m.expr] + args
        if isinstance(fn, Proj):
            args = [fn.expr] + args
            fn = Name(fn.field)
        if not isinstance(fn, Name):
            return None
        tmpl = _lookup(vocab.PREDICATES, fn.id)
        if tmpl is not None:
            k = _n_slots(tmpl)
            if len(args) >= k:
                # Mathlib predicates often take extra leading implicit-ish args
                # (e.g. `Differentiable ℝ f`), so align from the template's needs.
                return tmpl.format(*[self.term(a, 51) for a in args[:k]])
        last = _last(fn.id)
        if _PROP_HEADS.match(last) and args:
            words = " ".join(vocab.camel_words(last[2:] if last.startswith("Is") else last))
            subj = self.term(args[-1] if len(args) > 1 and last.startswith("Is") else args[0], 51)
            rest = [a for a in args if self.term(a, 51) != subj]
            tail = f" (with respect to {', '.join(self.term(a) for a in rest)})" if rest else ""
            verb = "is" if last.startswith("Is") else ""
            return f"{subj} {verb} {vocab.article(words) if _is_noun(words) else words}{tail}".replace("  ", " ")
        return None

    def _negate(self, n: Node) -> str:
        if isinstance(n, Bin):
            base = n.op
            if base in _NEG_REL:
                return self.prop(Bin(_NEG_REL[base], n.lhs, n.rhs))
            if base in _NOT_REL:
                return f"{self.term(n.lhs, 51)} {_NOT_REL[base]} {self.term(n.rhs, 51)}"
        if isinstance(n, (App, Proj)):
            s = self._pred(n)
            if s is not None:
                for pos, neg in ((" is ", " is not "), (" are ", " are not "), (" has ", " does not have ")):
                    if pos in s:
                        return s.replace(pos, neg, 1)
        if isinstance(n, Quant) and n.q == "∃":
            rest = self._quant_prop(n)[len("there exists ") :]
            for art in ("a ", "an "):
                rest = rest.removeprefix(art)
            return "there is no " + rest if self._quant_prop(n).startswith("there exists ") else f"it is not the case that {self.prop(n)}"
        return f"it is not the case that {self.prop(n)}"

    def _quant_prop(self, n: Quant) -> str:
        names = [x for b in n.binders for x in b.names]
        q = n.q
        with self._bind(names):
            # `∀ (h : P), Q` is an implication.
            hyps: list[str] = []
            parts: list[str] = []
            for b in n.binders:
                if b.type is not None and self.is_prop(b.type) and not b.inst:
                    hyps.append(self.prop(b.type))
                    continue
                parts.append(self._binder_phrase(b, q))
            body = self.prop(n.body)
            if q == "fun":
                return self.term(n)
            if q.startswith("∀ᶠ"):
                where = self.term(n.extra["domain"]) if "domain" in n.extra else "the filter"
                return f"eventually (along {where}), {body}"
            if q.startswith("∃ᶠ"):
                where = self.term(n.extra["domain"]) if "domain" in n.extra else "the filter"
                return f"frequently (along {where}), {body}"
            if q == "∀":
                if hyps and not parts:
                    return f"if {_join_and(hyps)}, then {body}"
                pre = "for " + _join_and(parts)
                if hyps:
                    return f"{pre}, if {_join_and(hyps)}, then {body}"
                return f"{pre}, {body}"
            if q == "∃":
                cond = f"{_join_and(hyps)} and " if hyps else ""
                return f"there exists {_join_and(parts)} such that {cond}{body}"
            if q == "∃!":
                return f"there exists a unique {_join_and(parts)} such that {body}"
            if q[0] in "∀∃":
                return f"{q} {_join_and(parts)}, {body}"
            return self._quant_term(n)[0]

    def _binder_phrase(self, b: BinderSpec, q: str) -> str:
        names = _join_and(b.names)
        many = len(b.names) > 1
        if b.bound_op is not None:
            return f"{'all' if q == '∀' else 'some'} {names} {b.bound_op} {self.term(b.bound)}" if many else (
                f"{'every' if q == '∀' else 'some'} {names} {b.bound_op} {self.term(b.bound)}" if q == "∀"
                else f"{names} {b.bound_op} {self.term(b.bound)}"
            )
        if b.type is not None:
            noun = self.type_noun(b.type, plural=many)
            if noun is not None:
                if q == "∀":
                    return f"{'all' if many else 'every'} {noun} {names}"
                return f"{'' if many else vocab.article(noun.split()[0]).split()[0] + ' '}{noun} {names}".strip()
            ty = self.term(b.type)
            return f"{'all' if q == '∀' and many else 'every' if q == '∀' else ''} {names} in {ty}".strip()
        return f"{'all' if many else 'every'} {names}" if q == "∀" else names

    # ------------------------------------------------------------------ scope
    class _BindCtx:
        def __init__(self, g: "Glosser", names: list[str]):
            self.g = g
            self.new = [x for x in names if x not in g.locals]

        def __enter__(self):
            self.g.locals.update(self.new)

        def __exit__(self, *exc):
            self.g.locals.difference_update(self.new)

    def _bind(self, names: list[str]) -> "_BindCtx":
        return Glosser._BindCtx(self, names)


def _n_slots(tmpl: str) -> int:
    nums = [int(x) for x in re.findall(r"(?<!\{)\{(\d)\}", tmpl)]
    return max(nums) + 1 if nums else 0


def _flatten(n: Node, op: str) -> list[Node]:
    if isinstance(n, Bin) and n.op == op:
        return _flatten(n.lhs, op) + _flatten(n.rhs, op)
    return [n]


def _join_and(items: list[str]) -> str:
    items = [x for x in items if x]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _is_noun(words: str) -> bool:
    last = words.split()[-1] if words else ""
    return not last.endswith(("ed", "ive", "ble", "al", "ous", "ic", "ar", "ent", "ant", "ing", "ary", "ful"))


# ===========================================================================
# Name glossing: `mul_le_mul_of_nonneg_left` -> "mul ≤ mul, of nonnegative, left"

def gloss_name(name: str) -> str:
    last = _last(name)
    ns = name[: -len(last) - 1] if "." in name else ""
    pieces = []
    for chunk in re.split(r"_of_", last):
        words = [vocab.NAME_WORDS.get(w, w) for w in chunk.split("_") if w]
        pieces.append(" ".join(words))
    out = " — given ".join(pieces)
    out = out.replace(" iff ", " ⟺ ")
    if ns:
        out = f"{ns}: {out}"
    return out


# ===========================================================================
# Declarations

def _idents(toks: list[Token]) -> set[str]:
    out = set()
    for k, t in enumerate(toks):
        if t.kind == "ident" and not (k + 1 < len(toks) and toks[k + 1].text == ":="):
            out.add(t.text)
            out.add(t.text.split(".", 1)[0])
    return out


_BINDER_TOKENS = {"∀", "∃", "∃!", "∑", "∏", "∑'", "∏'", "fun", "λ", "⋃", "⋂", "⨆", "⨅", "∫", "⨍", "∀ᶠ", "∃ᶠ"}


def _quantified(toks: list[Token]) -> set[str]:
    """Names bound by quantifiers / lambdas inside the statement."""
    out: set[str] = set()
    for k, t in enumerate(toks):
        if t.text in _BINDER_TOKENS:
            j = k + 1
            while j < len(toks) and (toks[j].kind == "ident" or toks[j].text in ("(", "{", "⦃", "[")):
                if toks[j].kind == "ident":
                    out.add(toks[j].text)
                j += 1
    return out


def _relevant_context(d: Decl) -> list[Binder]:
    """Lean only includes section `variable`s that the statement mentions
    (plus instance arguments about them).  Mimic that."""
    if not d.context:
        return []
    rebound = {x for b in d.binders for x in b.names} | _quantified(d.type)
    used = _idents(d.type)
    for b in d.binders:
        used |= _idents(b.type)
    for _, ty in d.fields:
        used |= _idents(ty)
    used |= _idents(d.body) | _idents(d.extends)
    chosen: list[Binder] = []
    # Fixed point: a variable whose type mentions further variables pulls them in.
    changed = True
    picked = [False] * len(d.context)
    while changed:
        changed = False
        for k, b in enumerate(d.context):
            if picked[k] or b.kind == "inst":
                continue
            hit = [x for x in b.names if x in used and x not in rebound]
            if hit:
                picked[k] = True
                used |= _idents(b.type)
                changed = True
    var_names = {x for b in d.context if b.kind != "inst" for x in b.names}
    for k, b in enumerate(d.context):
        if b.kind == "inst":
            mentioned = _idents(b.type) & var_names
            if mentioned and mentioned <= used:
                picked[k] = True
    for k, b in enumerate(d.context):
        if picked[k]:
            if b.kind == "inst" or not b.names:
                chosen.append(b)
            else:
                names = [x for x in b.names if x in used and x not in rebound]
                if names:
                    chosen.append(Binder(b.kind, names, b.type))
    return chosen


_KIND_WORD = {
    "theorem": "Theorem", "lemma": "Lemma", "def": "Definition", "abbrev": "Definition",
    "instance": "Instance", "structure": "Structure", "class": "Class", "inductive": "Inductive type",
    "example": "Example", "axiom": "Axiom", "opaque": "Definition",
}


def gloss_decl(d: Decl) -> Translation:
    g = Glosser()
    binders = _relevant_context(d) + d.binders
    sentences: list[str] = []

    # --- classify binders ---------------------------------------------------
    type_vars: list[str] = []
    classes: list[tuple[str, Node]] = []  # (class name, parsed type)
    objects: list[tuple[list[str], Node | None]] = []
    hyps: list[Node] = []
    all_names = [x for b in binders for x in b.names]
    g.locals.update(all_names)
    parsed = [(b, parse(b.type) if b.type else None) for b in binders]
    for b, ty in parsed:
        if b.kind != "inst" and ty is not None and isinstance(ty, Name) and ty.id == "Prop":
            g.prop_vars.update(b.names)
    for b, ty in parsed:
        if b.kind == "inst":
            if ty is not None:
                classes.append(("", ty))
            continue
        if ty is None:
            objects.append((b.names, None))
        elif isinstance(ty, Name) and ty.id in ("Type", "Type*", "Sort", "Sort*"):
            type_vars.extend(b.names)
        elif g.is_prop(ty) or (all(x.startswith("h") for x in b.names) and not g.type_noun(ty) and _looks_prop(ty)):
            hyps.append(ty)
        else:
            objects.append((b.names, ty))
    g.type_vars = set(type_vars)

    # --- "Let G be a group." ------------------------------------------------
    described: dict[str, list[str]] = {}
    extra_facts: list[str] = []
    for _, ty in classes:
        fact = _class_fact(g, ty)
        if fact is None:
            continue
        carrier, phrase = fact
        if carrier is not None and carrier in type_vars:
            described.setdefault(carrier, []).append(phrase)
        elif carrier is not None:
            extra_facts.append(f"{carrier} {phrase}")
        else:
            extra_facts.append(phrase)
    plain = [v for v in type_vars if v not in described]
    if plain:
        sentences.append(f"Let {_join_and(plain)} be {'types' if len(plain) > 1 else 'a type'}.")
    extra_facts = list(dict.fromkeys(extra_facts))
    for v, phrases in described.items():
        phrases = list(dict.fromkeys(phrases))
        is_a = [p[3:] for p in phrases if p.startswith("is ")]
        others = [p for p in phrases if not p.startswith("is ")]
        if is_a:
            sentences.append(f"Let {v} be {_join_and(is_a)}.")
            if others:
                sentences.append(f"{v} {_join_and(others)}.")
        else:
            sentences.append(f"Let {v} be a type that {_join_and(others)}.")
    for fact in extra_facts:
        sentences.append(_cap_sentence(fact, g) + ".")

    # --- "Let a, b be elements of G." -------------------------------------
    for names, ty in _group_objects(objects, g):
        if ty is None:
            sentences.append(f"Let {_join_and(names)} be arbitrary.")
            continue
        many = len(names) > 1
        noun = g.type_noun(ty, plural=many)
        if noun is None:
            tyt = g.term(ty)
            if isinstance(ty, Name) and (ty.id in type_vars or ty.id in g.locals):
                noun = f"elements of {tyt}" if many else f"an element of {tyt}"
            elif _structure_head(ty):
                head, rest = _structure_head(ty)
                words = " ".join(vocab.camel_words(head))
                args = ", ".join(g.term(a, _ATOM) for a in rest)
                noun = (vocab.pluralize(words) if many else vocab.article(words)) + (f" ({args})" if args else "")
            else:
                noun = f"terms of type {tyt}" if many else f"a term of type {tyt}"
        elif not many:
            noun = vocab.article(noun)
        sentences.append(f"Let {_join_and(names)} be {noun}.")

    # --- hypotheses --------------------------------------------------------
    if hyps:
        sentences.append("Assume " + _join_and([f"that {g.prop(h)}" for h in hyps]) + ".")

    # --- the statement itself ------------------------------------------------
    stmt = parse(d.type) if d.type else None
    kind = d.kind
    shown = d.full_name or "(anonymous)"
    if kind in ("theorem", "lemma", "example", "axiom"):
        if stmt is not None:
            sentences.append(f"Then {_sent(g.prop(stmt))}")
    elif kind in ("def", "abbrev", "opaque"):
        what = g.type_phrase(stmt) if stmt is not None else None
        args = [x for names, _ in objects for x in names]
        call = f"{_last(shown)}({', '.join(args)})" if args else _last(shown)
        if stmt is not None and g.is_prop(stmt) or isinstance(stmt, Name) and stmt.id == "Prop":
            if d.body:
                sentences.append(f"{call} holds when {_sent(g.prop(parse(d.body)))}")
            else:
                sentences.append(f"{call} is a property{' of ' + _join_and(args) if args else ''}.")
        elif d.body:
            body = parse(d.body)
            sentences.append(f"Define {call} := {g.term(body)}" + (f", which is {what}." if what else "."))
        elif what:
            sentences.append(f"{call} is {what}.")
    elif kind == "instance":
        if stmt is not None:
            fact = _class_fact(g, stmt)
            sentences.append(_sent(fact[1] if fact and fact[0] is None else (f"{fact[0]} is {fact[1]}" if fact else g.term(stmt))))
    elif kind in ("structure", "class"):
        params = _join_and(type_vars + [x for names, _ in objects for x in names])
        head = f"{vocab.article(' '.join(vocab.camel_words(_last(shown))))}"
        if params:
            head += f" structure on {params}"
        parents = parse(d.extends) if d.extends else None
        line = f"{_cap(head)} consists of"
        if d.extends:
            parent_txt = _join_and([
                _class_fact(g, p)[1].removeprefix("is ") if _class_fact(g, p) else g.term(p)
                for p in _comma_split(d.extends)
            ])
            line = f"{_cap(head)} is {parent_txt}"
            line += ", together with" if d.fields else "."
        if d.fields:
            fl = []
            for fname, ftoks in d.fields:
                fty = parse(ftoks)
                if g.is_prop(fty):
                    fl.append(f"the property ({fname}) that {g.prop(fty)}")
                else:
                    fl.append(f"{fname}, {g.type_phrase(fty)}")
            line += " " + "; ".join(fl) + "."
        elif not d.extends:
            line += " no fields."
        del parents
        sentences.append(line)
    elif kind == "inductive":
        sentences.append(f"{_last(shown)} is an inductive type" + (f" of {g.type_phrase(stmt, plural=True)}." if stmt is not None else "."))

    text = " ".join(s for s in sentences if s)
    return Translation(
        name=shown, kind=kind, file=d.file, line=d.line, lean=d.signature,
        gloss=f"{_KIND_WORD.get(kind, kind)} {shown}: {text}", name_gloss=gloss_name(shown), doc=d.doc,
    )


def _comma_split(toks: list[Token]) -> list[Node]:
    out, cur, depth = [], [], 0
    for t in toks:
        if t.text in "([{⟨":
            depth += 1
        elif t.text in ")]}⟩":
            depth -= 1
        if t.text == "," and depth == 0:
            out.append(parse(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(parse(cur))
    return out


def _class_fact(g: Glosser, ty: Node) -> tuple[str | None, str] | None:
    """`Group G` -> ("G", "is a group");  `Module R M` -> (None, "M is a module over R").

    The first element is the carrier when the fact is about a single
    variable, in which case the second element is a predicate phrase
    ("is a group", "has zero morphisms"); otherwise it is a full sentence.
    """
    if isinstance(ty, Ascribe):
        ty = ty.expr
    if isinstance(ty, Name):
        return None, " ".join(vocab.camel_words(_last(ty.id)))
    if not isinstance(ty, App) or not isinstance(ty.fn, Name):
        return None, g.prop(ty) if g.is_prop(ty) else g.term(ty)
    cls = ty.fn.id
    if cls in vocab.SILENT_CLASSES or _last(cls) in vocab.SILENT_CLASSES:
        return None
    args = [a for a in ty.args if not (isinstance(a, Bin) and a.op == ":=")]
    tmpl = vocab.CLASS_TEMPLATES.get(cls) or vocab.CLASS_TEMPLATES.get(_last(cls))
    if tmpl is not None:
        k = _n_slots(tmpl)
        if len(args) >= k:
            rendered = [g.prop(a) if _last(cls) == "Fact" else g.term(a, _ATOM) for a in args[:k]]
            sentence = tmpl.format(*rendered)
            if k == 1 and tmpl.startswith("{0} ") and isinstance(args[0], Name):
                return args[0].id, sentence[len(rendered[0]) + 1 :]
            return None, sentence
    last = _last(cls)
    has = last.startswith("Has")
    words = " ".join(vocab.camel_words(last[3:] if has else last))
    pred = f"has {words}" if has else f"is {vocab.article(words)}"
    if len(args) == 1 and isinstance(args[0], Name):
        return args[0].id, pred
    if args:
        carrier = g.term(args[-1], _ATOM)
        others = ", ".join(g.term(a, _ATOM) for a in args[:-1])
        return None, f"{carrier} {pred}" + (f" over {others}" if others else "")
    return None, words


def _group_objects(objects, g: Glosser):
    """Merge consecutive binders of the same type: (a : G) (b : G) -> a, b : G."""
    out: list[tuple[list[str], Node | None]] = []
    for names, ty in objects:
        key = g.term(ty) if ty is not None else None
        if out and key is not None and out[-1][1] is not None and g.term(out[-1][1]) == key:
            out[-1] = (out[-1][0] + names, out[-1][1])
        else:
            out.append((list(names), ty))
    return out


def _looks_prop(ty: Node) -> bool:
    return isinstance(ty, (App, Bin, Pre, Quant, Proj))


def _structure_head(ty: Node) -> tuple[str, list[Node]] | None:
    """`Simplex ℝ P n` -> ("Simplex", [ℝ, P, n]) for capitalised type heads."""
    if isinstance(ty, App) and isinstance(ty.fn, Name):
        last = _last(ty.fn.id)
        if last[:1].isupper() and last.isidentifier():
            return last, ty.args
    if isinstance(ty, Name):
        last = _last(ty.id)
        if last[:1].isupper() and last.isidentifier() and len(last) > 2:
            return last, []
    return None


def _cap_sentence(s: str, g: Glosser) -> str:
    first = s.split(" ", 1)[0]
    if first in g.locals:
        return s
    return _cap(s)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _sent(s: str) -> str:
    s = s.strip()
    return s if s.endswith((".", "?", "!")) else s + "."
