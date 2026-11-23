"""A Pratt (precedence-climbing) parser for Lean 4 *statements*.

It covers the term language that shows up in theorem statements: logic,
quantifiers, relations, arithmetic, set-builder notation, big operators,
lambdas, coercions, anonymous constructors and function application.
Anything it cannot handle is wrapped in a ``Raw`` node so that translation
degrades gracefully instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Token

__all__ = [
    "Node", "Name", "Num", "App", "Bin", "Pre", "Post", "Quant", "BinderSpec",
    "Ite", "Tuple", "SetOf", "SetLit", "Ascribe", "Hole", "Delim", "Proj", "Raw",
    "parse", "ParseError",
]


class ParseError(Exception):
    pass


class Node:
    __slots__ = ()


@dataclass(slots=True)
class Name(Node):
    id: str


@dataclass(slots=True)
class Num(Node):
    value: str


@dataclass(slots=True)
class App(Node):
    fn: Node
    args: list[Node]


@dataclass(slots=True)
class Bin(Node):
    op: str
    lhs: Node
    rhs: Node


@dataclass(slots=True)
class Pre(Node):
    op: str
    arg: Node


@dataclass(slots=True)
class Post(Node):
    op: str
    arg: Node


@dataclass(slots=True)
class BinderSpec:
    names: list[str]
    type: Node | None = None
    bound_op: str | None = None  # `∀ x ∈ s` -> ("∈", s)
    bound: Node | None = None
    inst: bool = False


@dataclass(slots=True)
class Quant(Node):
    q: str  # ∀ ∃ ∃! fun ∑ ∏ ⋃ ⋂ ⨆ ⨅ ∫
    binders: list[BinderSpec]
    body: Node
    extra: dict = field(default_factory=dict)  # e.g. integral bounds


@dataclass(slots=True)
class Ite(Node):
    cond: Node
    then: Node
    other: Node
    hyp: str | None = None


@dataclass(slots=True)
class Tuple(Node):
    items: list[Node]
    anon: bool = False  # ⟨a, b⟩


@dataclass(slots=True)
class SetOf(Node):
    var: Node  # usually a Name, sometimes a pattern
    type: Node | None
    pred: Node
    subtype: bool = False  # {x // p}


@dataclass(slots=True)
class SetLit(Node):
    items: list[Node]


@dataclass(slots=True)
class Ascribe(Node):
    expr: Node
    type: Node


@dataclass(slots=True)
class Hole(Node):
    text: str  # `·` or `_`


@dataclass(slots=True)
class Delim(Node):
    kind: str  # "abs" | "norm" | "list" | "floor" | "ceil" | "inner"
    items: list[Node]


@dataclass(slots=True)
class Proj(Node):
    expr: Node
    field: str


@dataclass(slots=True)
class Raw(Node):
    text: str


# ---------------------------------------------------------------------------
# Operator tables (left-binding powers follow Lean 4 / Mathlib closely).

MAX = 1024
ARG = 1023

# op -> (lbp, right_assoc)
INFIX: dict[str, tuple[int, bool]] = {
    "↔": (20, False), "<->": (20, False),
    "→": (25, True), "->": (25, True),
    "≃": (25, True), "≃+*": (25, True), "≃*": (25, True), "≃+": (25, True),
    "≃ₗ": (25, True), "≃o": (25, True), "→+*": (25, True), "→*": (25, True),
    "→+": (25, True), "→ₗ": (25, True), "→o": (25, True), "↪": (25, True),
    "↠": (25, True), "→ₙ": (25, True),
    "∨": (30, True), "||": (30, True),
    "∧": (35, True), "&&": (35, True), "×": (35, True), "⧸": (35, False),
    "=": (50, False), "≠": (50, False), "<": (50, False), ">": (50, False),
    "≤": (50, False), "≥": (50, False), "<=": (50, False), ">=": (50, False),
    "∈": (50, False), "∉": (50, False), "⊆": (50, False), "⊂": (50, False),
    "⊇": (50, False), "⊃": (50, False), "∣": (50, False), "∤": (50, False),
    "≡": (50, False), "≈": (50, False), "≅": (50, False), "⋖": (50, False),
    "≪": (50, False), "∼": (50, False), "==": (50, False), "!=": (50, False),
    "≺": (50, False), "≼": (50, False), "⩿": (50, False),
    "+": (65, False), "-": (65, False), "∪": (65, False), "++": (65, False),
    "⊔": (68, False), "∆": (100, False),
    "*": (70, False), "/": (70, False), "%": (70, False), "∩": (70, False),
    "⊓": (69, False), "\\": (70, False), "⊗": (100, False), "⊕": (30, True),
    "•": (73, True), "^": (75, True), "∘": (90, False),
    "⁻¹'": (80, False), "''": (80, False), "∗": (70, False), "⋆": (70, False),
    "$": (10, True), "<|": (10, True), "|>": (10, False),
}
# Note: `-` as infix is subtraction; as prefix it is negation.
POSTFIX = {"⁻¹": MAX, "ᶜ": MAX, "!": 10000}
PREFIX = {"¬": 40, "-": 75, "↑": MAX, "⇑": MAX, "↥": MAX, "√": 100, "∑": 67, "∏": 67}
BIG_OPS = {"∑", "∏", "⋃", "⋂", "⨆", "⨅", "∫", "∮"}
QUANTS = {"∀", "∃", "∃!", "Π", "Σ", "λ", "fun"}
STOP = {
    ",", ")", "]", "}", "⟩", ":=", "then", "else", "=>", "↦", "|", ":", "with",
    "in", "∂", "⦄", "}}", "at", "from", "using", "..", "//", ";",
}


_MODS = set(
    chr(c) for r in ((0x02B0, 0x02FF), (0x1D2C, 0x1DBF), (0x2070, 0x207F), (0x208A, 0x209C))
    for c in range(r[0], r[1] + 1)
)
INFIX.update({"≫": (80, False), "⋙": (80, False), "⬝ᵥ": (72, False), "⊗ₜ": (100, False),
              "⟶": (10, True), "⥤": (26, True), "≌": (10, False), "⊣": (15, False),
              "▸": (75, True), "⇒": (25, True)})
BIG_OPS.update({"∑'", "∏'", "⨁", "⨍"})


def infix_info(s: str) -> tuple[int, bool] | None:
    """Binding power of an infix operator, allowing decorated variants like
    `*ᵥ`, `=ᶠ`, `×ˢ` (looked up via their undecorated base)."""
    if s in INFIX:
        return INFIX[s]
    base = s.rstrip("".join(_MODS))
    if base != s and base in INFIX:
        return INFIX[base]
    return None


def is_postfix_mod(s: str) -> bool:
    return bool(s) and all(ch in _MODS for ch in s) and s not in ("⁻¹",)


class Parser:
    def __init__(self, toks: list[Token]):
        self.toks = toks
        self.i = 0
        self.delims: list[str] = []  # open ‖ / | delimiters

    # -- helpers ---------------------------------------------------------
    def peek(self, k: int = 0) -> Token | None:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def peek_text(self, k: int = 0) -> str | None:
        t = self.peek(k)
        return t.text if t else None

    def next(self) -> Token:
        if self.i >= len(self.toks):
            raise ParseError("unexpected end of input")
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> Token:
        t = self.next()
        if t.text != text:
            raise ParseError(f"expected {text!r}, got {t.text!r}")
        return t

    def at_end(self) -> bool:
        return self.i >= len(self.toks)

    def adjacent(self, a: Token, b: Token) -> bool:
        return a.line == b.line and a.col + len(a.text) == b.col

    # -- core Pratt loop -------------------------------------------------
    def expr(self, rbp: int = 0) -> Node:
        left = self.nud()
        while True:
            t = self.peek()
            if t is None:
                return left
            s = t.text
            if t.kind == "sym" and s in STOP:
                return left
            if t.kind == "ident" and s in STOP:
                return left
            if t.kind == "sym" and (s in POSTFIX or is_postfix_mod(s)):
                if POSTFIX.get(s, MAX) <= rbp:
                    return left
                self.next()
                left = Post(s, left)
                continue
            if s == "." and t.kind == "sym":
                nxt = self.peek(1)
                if nxt is not None and nxt.kind == "ident" and self.adjacent(t, nxt):
                    self.i += 2
                    left = Proj(left, nxt.text)
                    continue
                return left
            info = infix_info(s) if t.kind == "sym" else None
            if info is not None and not (s.startswith("‖") and self.delims):
                lbp, right = info
                if lbp <= rbp:
                    return left
                self.next()
                nxt = self.peek()
                if nxt is not None and nxt.text == "[" and self.adjacent(t, nxt):
                    self.next()  # operator parameter, e.g. `⊗ₜ[R]`, `→ₗ[R]`
                    param = self.expr(0)
                    self.expect("]")
                    s = s + "[" + _short(param) + "]"
                rhs = self.expr(lbp - 1 if right else lbp)
                left = Bin(s, left, rhs)
                continue
            prev = self.toks[self.i - 1]
            if s == "[" and self.adjacent(prev, t):
                # `R[X]` polynomial rings, `f^[n]` iterates (after `^`) are handled here.
                if self.peek_text(1) == "X" and self.peek_text(2) == "]":
                    self.i += 3
                    left = App(Name("Polynomial"), [left])
                    continue
            if s == "^" and self.peek_text(1) == "[" and self.adjacent(t, self.peek(1)):
                self.i += 2
                k = self.expr(0)
                self.expect("]")
                left = App(Name("Nat.iterate"), [left, k])
                continue
            if self.starts_atom(t):
                if ARG <= rbp:
                    return left
                args = []
                while (t := self.peek()) is not None and self.starts_atom(t):
                    args.append(self.expr(ARG))
                if isinstance(left, App):
                    left = App(left.fn, left.args + args)
                else:
                    left = App(left, args)
                continue
            return left

    def starts_atom(self, t: Token) -> bool:
        s = t.text
        if t.kind in ("ident", "num", "str", "char"):
            return s not in STOP and s not in _NON_ATOM_WORDS
        if s in ("‖", "|"):
            return not self.delims and s == "‖"
        return s in _ATOM_SYMS

    def nud(self) -> Node:
        t = self.next()
        s = t.text
        if t.kind == "num":
            return Num(s)
        if t.kind in ("str", "char"):
            return Raw(s)
        if t.kind == "ident":
            if s in ("fun", "λ"):
                return self.quant("fun")
            if s == "if":
                return self.ite()
            if s in ("by", "match", "do", "have", "show", "let", "calc"):
                raise ParseError(f"unsupported construct {s}")
            if s in ("Type", "Sort"):
                nxt = self.peek()
                if nxt is not None and nxt.text in ("*", "_") and self.adjacent(t, nxt):
                    self.next()
                    return Name(s + "*")
                if nxt is not None and nxt.kind in ("ident", "num") and nxt.text not in STOP:
                    self.next()
                return Name(s)
            return Name(s)
        # symbols
        if s in QUANTS:
            return self.quant("fun" if s == "λ" else s)
        if s in BIG_OPS:
            return self.quant(s)
        if t.kind == "sym" and len(s) > 1 and s[0] in "∀∃∑∏∫⨍" and all(c in _MODS for c in s[1:].lstrip("⁻")):
            # decorated binders: `∀ᶠ x in l, p x`, `∃ᶠ`, `∫⁻`, `⨍⁻`
            return self.quant(s)
        if s == "@":
            return self.nud()
        if s == "(":
            return self.paren()
        if s == "⟨":
            items = self.comma_list("⟩")
            return Tuple(items, anon=True)
        if s == "[":
            return Delim("list", self.comma_list("]"))
        if s == "{":
            return self.brace()
        if s in ("‖", "|"):
            self.delims.append(s)
            try:
                inner = self.expr(0)
            finally:
                self.delims.pop()
            close = self.next()
            if not close.text.startswith(s):
                raise ParseError("unclosed norm/abs")
            return Delim("norm" if s == "‖" else "abs", [inner])
        if s == "⁅":
            items = self.comma_list("⁆")
            return Delim("bracket", items)
        if s == "⌊":
            inner = self.expr(0)
            self.expect("⌋")
            if self.peek_text() == "₊":
                self.next()
            return Delim("floor", [inner])
        if s == "⌈":
            inner = self.expr(0)
            self.expect("⌉")
            if self.peek_text() == "₊":
                self.next()
            return Delim("ceil", [inner])
        if s == "⟪":
            items = self.comma_list("⟫")
            if self.peek_text() and self.peek_text().startswith("_"):
                self.next()
            return Delim("inner", items)
        if s == "#":
            return Pre("#", self.expr(MAX))
        if s in ("·", "_"):
            return Hole(s)
        if s in ("⊤", "⊥", "∅", "∞", "π"):
            return Name(s)
        if s in PREFIX:
            if s in ("∑", "∏"):
                return self.quant(s)
            return Pre(s, self.expr(PREFIX[s]))
        if s == "." :
            nxt = self.next()
            return Name("." + nxt.text)
        raise ParseError(f"unexpected token {s!r}")

    def comma_list(self, close: str) -> list[Node]:
        items: list[Node] = []
        if self.peek_text() == close:
            self.next()
            return items
        while True:
            items.append(self.expr(0))
            t = self.next()
            if t.text == close:
                return items
            if t.text != ",":
                raise ParseError(f"expected , or {close}")

    def paren(self) -> Node:
        if self.peek_text() == ")":
            self.next()
            return Name("()")
        if self.peek_text() in ("↑", "⇑", "↥") and self.peek_text(1) == ")":
            self.i += 2
            return Name("↑")
        e = self.expr(0)
        t = self.next()
        if t.text == ")":
            return _section(e)
        if t.text == ":":
            ty = self.expr(0)
            self.expect(")")
            return Ascribe(e, ty)
        if t.text == ",":
            items = [e]
            while True:
                items.append(self.expr(0))
                t = self.next()
                if t.text == ")":
                    return Tuple(items)
                if t.text != ",":
                    raise ParseError("bad tuple")
        if t.text == ":=":  # named argument `(α := M)`
            val = self.expr(0)
            self.expect(")")
            return Bin(":=", e, val)
        raise ParseError(f"unexpected {t.text!r} in parens")

    def brace(self) -> Node:
        # {x | p x}, {x : T | p x}, {x // p}, {a, b, c}, {} , {x ∈ s | p}
        if self.peek_text() == "}":
            self.next()
            return SetLit([])
        start = self.i
        # `{a b : α} → β` : an implicit pi binder, treat like a Lean binder.
        j = self.i
        while j < len(self.toks) and self.toks[j].kind == "ident":
            j += 1
        if j > self.i and j < len(self.toks) and self.toks[j].text == ":":
            close = self._find_close(start - 1)
            if close + 1 < len(self.toks) and self.toks[close + 1].text in ("→", "->"):
                names = [t.text for t in self.toks[self.i:j]]
                self.i = j + 1
                ty = self.expr(0)
                self.expect("}")
                return Ascribe(Name(" ".join(names)), ty)
        try:
            var = self.expr(51)  # stop before relations so `{x ∈ s | p}` works
            ty = None
            bound = None
            if self.peek_text() == ":":
                self.next()
                ty = self.expr(0)
            elif self.peek_text() in ("∈", "≤", "<", "≥", ">", "⊆") :
                op = self.next().text
                bound = (op, self.expr(51))
            sep = self.peek_text()
            if sep in ("|", "//"):
                self.next()
                pred = self.expr(0)
                self.expect("}")
                if bound is not None:
                    pred = Bin("∧", Bin(bound[0], var, bound[1]), pred)
                return SetOf(var, ty, pred, subtype=(sep == "//"))
        except ParseError:
            pass
        self.i = start
        return SetLit(self.comma_list("}"))

    def _find_close(self, i: int) -> int:
        depth = 0
        opens, closes = {"(", "[", "{", "⟨", "⦃"}, {")", "]", "}", "⟩", "⦄"}
        for k in range(i, len(self.toks)):
            t = self.toks[k].text
            if t in opens:
                depth += 1
            elif t in closes:
                depth -= 1
                if depth == 0:
                    return k
        return len(self.toks)

    def ite(self) -> Node:
        hyp = None
        if self.peek() is not None and self.peek().kind == "ident" and self.peek_text(1) == ":":
            hyp = self.next().text
            self.next()
        cond = self.expr(0)
        self.expect("then")
        a = self.expr(0)
        self.expect("else")
        b = self.expr(0)
        return Ite(cond, a, b, hyp)

    # -- binders ---------------------------------------------------------
    def quant(self, q: str) -> Node:
        binders = self.binders()
        extra: dict = {}
        if q[0] in "∫∮⨍∀∃" and self.peek_text() == "in":
            self.next()
            extra["domain"] = self.expr(0)
            if self.peek_text() == "..":
                self.next()
                extra["domain"] = Bin("..", extra["domain"], self.expr(0))
        if q == "fun":
            t = self.next()
            if t.text not in ("=>", "↦"):
                raise ParseError("expected => in fun")
            body = self.expr(0)
            return Quant("fun", binders, body)
        self.expect(",")
        body = self.expr(67 if q in ("∑", "∏", "∑'", "∏'") else 0)
        if q[0] in "∫∮⨍" and self.peek_text() == "∂":
            self.next()
            extra["measure"] = self.expr(MAX)
        return Quant(q, binders, body, extra)

    def binders(self) -> list[BinderSpec]:
        out: list[BinderSpec] = []
        bracket = {"(": ")", "{": "}", "[": "]", "⦃": "⦄", "{{": "}}"}
        names: list[str] = []
        while True:
            t = self.peek()
            if t is None:
                raise ParseError("eof in binders")
            if t.text in bracket and not names:
                self.next()
                close = bracket[t.text]
                inner_names: list[str] = []
                while self.peek() is not None and self.peek().kind == "ident" and self.peek_text(1) not in (None,) and self.peek_text() not in STOP:
                    inner_names.append(self.next().text)
                    if self.peek_text() in (":", close):
                        break
                if t.text == "[":
                    if self.peek_text() == ":":
                        self.next()
                        ty = self.expr(0)
                    else:
                        # `[Group G]` – instance binder with no name
                        self.i -= len(inner_names)
                        inner_names = []
                        ty = self.expr(0)
                    self.expect(close)
                    out.append(BinderSpec(inner_names, ty, inst=True))
                    continue
                ty = None
                if self.peek_text() == ":":
                    self.next()
                    ty = self.expr(0)
                self.expect(close)
                out.append(BinderSpec(inner_names, ty))
                continue
            if t.kind == "ident" and t.text not in STOP or t.text in ("_",):
                names.append(self.next().text)
                continue
            if t.text == "⟨" and not names:  # destructuring pattern
                self.next()
                pat = self.comma_list("⟩")
                out.append(BinderSpec([_render_pattern(pat)]))
                continue
            break
        if names:
            t = self.peek()
            if t is not None and t.text == ":":
                self.next()
                ty = self.expr(0)
                out.append(BinderSpec(names, ty))
            elif t is not None and (infix_info(t.text) or (0,))[0] == 50:
                op = self.next().text
                bound = self.expr(51)
                out.append(BinderSpec(names, None, op, bound))
            else:
                out.append(BinderSpec(names))
        if not out:
            raise ParseError("empty binder list")
        return out


_NON_ATOM_WORDS = {"then", "else", "if", "by", "match", "do", "have", "show", "let",
                   "from", "at", "in", "with", "using", "extends", "where", "deriving"}
_ATOM_SYMS = {"(", "⟨", "[", "{", "·", "_", "↑", "⇑", "↥", "@", "⌊", "⌈", "√", "#",
              "⊤", "⊥", "∅", "∞", "⟪", "λ", "⁅"}


def _short(n: Node) -> str:
    return n.id if isinstance(n, Name) else "…"


def _render_pattern(items: list[Node]) -> str:
    parts = []
    for it in items:
        parts.append(it.id if isinstance(it, Name) else "_")
    return "⟨" + ", ".join(parts) + "⟩"


def _contains_hole(n: Node) -> bool:
    if isinstance(n, Hole):
        return n.text == "·"
    if isinstance(n, Bin):
        return _contains_hole(n.lhs) or _contains_hole(n.rhs)
    if isinstance(n, (Pre, Post)):
        return _contains_hole(n.arg)
    if isinstance(n, App):
        return _contains_hole(n.fn) or any(_contains_hole(a) for a in n.args)
    return False


def _fill_holes(n: Node, names: list[str]) -> Node:
    if isinstance(n, Hole) and n.text == "·":
        return Name(names.pop(0))
    if isinstance(n, Bin):
        lhs = _fill_holes(n.lhs, names)
        return Bin(n.op, lhs, _fill_holes(n.rhs, names))
    if isinstance(n, Pre):
        return Pre(n.op, _fill_holes(n.arg, names))
    if isinstance(n, Post):
        return Post(n.op, _fill_holes(n.arg, names))
    if isinstance(n, App):
        fn = _fill_holes(n.fn, names)
        return App(fn, [_fill_holes(a, names) for a in n.args])
    return n


def _section(e: Node) -> Node:
    """Turn `(· * x)` into `fun a => a * x`."""
    if not _contains_hole(e):
        return e
    pool = ["a", "b", "c", "d"]
    count = _count_holes(e)
    names = pool[:count] if count <= len(pool) else [f"x{k}" for k in range(count)]
    body = _fill_holes(e, list(names))
    return Quant("fun", [BinderSpec(list(names))], body)


def _count_holes(n: Node) -> int:
    if isinstance(n, Hole):
        return 1 if n.text == "·" else 0
    if isinstance(n, Bin):
        return _count_holes(n.lhs) + _count_holes(n.rhs)
    if isinstance(n, (Pre, Post)):
        return _count_holes(n.arg)
    if isinstance(n, App):
        return _count_holes(n.fn) + sum(_count_holes(a) for a in n.args)
    return 0


def parse(toks: list[Token]) -> Node:
    """Parse a full token list; on failure return a ``Raw`` node."""
    if not toks:
        return Raw("")
    p = Parser(toks)
    try:
        node = p.expr(0)
        if not p.at_end():
            raise ParseError(f"trailing tokens at {p.peek()!r}")
        return node
    except (ParseError, RecursionError, IndexError, AttributeError):
        return Raw(_detok(toks))


def _detok(toks: list[Token]) -> str:
    out = []
    prev = None
    for t in toks:
        if prev is not None and not (prev.line == t.line and prev.col + len(prev.text) == t.col):
            out.append(" ")
        out.append(t.text)
        prev = t
    return "".join(out)
