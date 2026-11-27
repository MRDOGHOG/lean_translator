"""LaTeX writer: a Lean project -> a typeset book, by rules (no AI).

* every formula is printed from its parse tree as real math
  (`∑ i ∈ s, f i ^ 2` -> $\\sum_{i \\in s} f(i)^{2}$),
* every statement is the logic-rule English with the math typeset,
* every proof is the narrated steps (see proofs.py) as nested lists,
* an overview part with figures drawn by TikZ / pgfplots: the import map
  between areas of the project, their sizes, the tactics used, the proof
  lengths, and a map of what the headline theorems are proved from.

    python -m lean_translator latex SRC_DIR -o book.tex --first Main.lean
    lualatex book.tex   (twice, for the table of contents)

Needs LuaLaTeX (for unicode-math and TikZ's automatic graph layout).
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path

from .decls import Decl, extract_decls
from .gloss import _SLOT_RE, Glosser, gloss_decl
from .parser import (App, Ascribe, Bin, Delim, Hole, Ite, Name, Node, Num, Post, Pre, Proj, Quant, Raw,
                     SetLit, SetOf, Tuple, infix_info)

# ---------------------------------------------------------------- formulas
_BB = {"ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}",
       "Real": r"\mathbb{R}", "Nat": r"\mathbb{N}", "Int": r"\mathbb{Z}", "Rat": r"\mathbb{Q}",
       "Complex": r"\mathbb{C}", "ℍ": r"\mathbb{H}"}
_CONST = {"π": r"\pi", "Real.pi": r"\pi", "∞": r"\infty", "⊤": r"\top", "⊥": r"\bot", "∅": r"\emptyset",
          "True": r"\mathrm{True}", "False": r"\mathrm{False}", "riemannZeta": r"\zeta", "Gamma": r"\Gamma",
          "Complex.Gamma": r"\Gamma", "Real.Gamma": r"\Gamma", "I": r"i", "Complex.I": r"i"}
_FUNC = {"exp": r"\exp", "log": r"\log", "sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "sinh": r"\sinh",
         "cosh": r"\cosh", "arctan": r"\arctan", "re": r"\operatorname{Re}", "im": r"\operatorname{Im}",
         "deriv": r"\operatorname{deriv}", "det": r"\det", "gcd": r"\gcd", "dim": r"\dim", "ker": r"\ker",
         "max": r"\max", "min": r"\min", "card": r"\#", "Nat.card": r"\#", "arg": r"\arg", "sup": r"\sup",
         "inf": r"\inf", "cexp": r"\exp", "rexp": r"\exp"}
_OPS = {"*": r"\cdot", "•": r"\cdot", "≠": r"\neq", "≤": r"\le", "≥": r"\ge", "<=": r"\le", ">=": r"\ge",
        "∈": r"\in", "∉": r"\notin", "⊆": r"\subseteq", "⊂": r"\subset", "⊇": r"\supseteq", "∩": r"\cap",
        "∪": r"\cup", "∧": r"\land", "∨": r"\lor", "→": r"\to", "->": r"\to", "↔": r"\iff", "<->": r"\iff",
        "∘": r"\circ", "∣": r"\mid", "∤": r"\nmid", "×": r"\times", "≡": r"\equiv", "\\": r"\setminus",
        "%": r"\bmod", "⊔": r"\sqcup", "⊓": r"\sqcap", "≃": r"\simeq", "≈": r"\approx", "⊗": r"\otimes",
        "⊕": r"\oplus", "^": "^", "''": r"''", "⁻¹'": r"^{-1}", "==": "=", "!=": r"\neq", "&&": r"\land",
        "||": r"\lor", "≪": r"\ll", "∼": r"\sim", "≅": r"\cong", "↪": r"\hookrightarrow", "$": r"\,", "<|": r"\,"}
_BIG = {"∑": r"\sum", "∑'": r"\sum", "∏": r"\prod", "∏'": r"\prod", "⋃": r"\bigcup", "⋂": r"\bigcap",
        "⨆": r"\sup", "⨅": r"\inf", "⨁": r"\bigoplus"}
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₙₘᵢⱼₖ", "0123456789nmijk")
_MAX_TEX = 400  # longer formulas are cut


def _ident(s: str, local: bool) -> str:
    """`x` -> x, `x₁` -> x_{1}, `foo_bar` -> \\mathrm{foo\\_bar}, `hx` (local) -> \\mathit{hx}."""
    base, sub = s, ""
    m = re.match(r"^(.*?)([₀-₉ₙₘᵢⱼₖ]+)$", s)
    if m and m.group(1):
        base, sub = m.group(1), m.group(2).translate(_SUB)
    primes = len(base) - len(base.rstrip("'"))
    base = base.rstrip("'")
    if len(base) == 1:
        out = base
    else:
        out = (r"\mathit{%s}" if local else r"\mathrm{%s}") % _esc_math_text(base)
    return out + "'" * primes + (f"_{{{sub}}}" if sub else "")


def _esc_math_text(s: str) -> str:
    return s.replace("\\", r"\backslash ").replace("_", r"\_").replace("#", r"\#").replace("%", r"\%") \
        .replace("&", r"\&").replace("$", r"\$").replace("{", r"\{").replace("}", r"\}").replace("^", r"\^{}")


class _Tex:
    def __init__(self, locals_: set[str]) -> None:
        self.locals = locals_

    def name(self, ident: str) -> str:
        ident = ident.removeprefix("_root_.")
        if ident == "_":  # placeholder, e.g. the left side of a calc step
            return r"\qquad"
        if ident in _BB:
            return _BB[ident]
        if ident in _CONST:
            return _CONST[ident]
        if ident in self.locals or "." not in ident:
            return _ident(ident, ident in self.locals and len(ident) > 1)
        head, field_ = ident.split(".", 1)
        if (head in self.locals or len(head) <= 2) and "." not in field_:
            return self.tex(Proj(Name(head), field_))  # `s.re` where s is a variable
        last = ident.rsplit(".", 1)[-1]
        if last in _FUNC:
            return _FUNC[last]
        if last in _CONST:
            return _CONST[last]
        return _ident(last, False)

    def atom(self, n: Node) -> bool:
        return isinstance(n, (Name, Num, Hole, Delim, Tuple, SetLit, SetOf)) or (
            isinstance(n, (App, Proj, Post)) and False)

    def wrap(self, n: Node, prec: int) -> str:
        s = self.tex(n)
        if isinstance(n, Bin):
            info = infix_info(n.op)
            if info and info[0] < prec:
                return rf"\left({s}\right)"
        if isinstance(n, (Quant, Ite)) and prec > 0:
            return rf"\left({s}\right)"
        if isinstance(n, Pre) and n.op == "-" and prec >= 70:
            return rf"\left({s}\right)"
        return s

    def arg(self, n: Node) -> str:
        """An argument in juxtaposition: atoms bare, everything else in parentheses."""
        s = self.tex(n)
        return s if self.atom(n) else rf"\left({s}\right)"

    def binders(self, q: Quant) -> str:
        parts = []
        for b in q.binders:
            names = ", ".join(self.name(x) for x in b.names)
            if b.bound_op and b.bound is not None:
                parts.append(f"{names} {_OPS.get(b.bound_op, _sym(b.bound_op))} {self.tex(b.bound)}")
            elif b.type is not None and q.q in ("∀", "∃", "∃!"):
                parts.append(f"{names} \\in {self.tex(b.type)}")
            else:
                parts.append(names)
        return ",\\ ".join(parts)

    def tex(self, n: Node) -> str:  # noqa: C901 - one case per node kind
        if isinstance(n, Name):
            return self.name(n.id)
        if isinstance(n, Num):
            return n.value
        if isinstance(n, Hole):
            return r"\cdot"
        if isinstance(n, Raw):
            return r"\text{\texttt{%s}}" % esc(n.text[:80])
        if isinstance(n, Ascribe):
            if isinstance(n.expr, Name) and " " in n.expr.id:
                return r"\text{\texttt{%s}}" % esc(n.expr.id[:80])
            return self.tex(n.expr)
        if isinstance(n, Bin):
            op = n.op
            base = op.split("[")[0]
            if base == "/":
                return rf"\frac{{{self.tex(n.lhs)}}}{{{self.tex(n.rhs)}}}"
            if base == "''":  # image f '' s
                return f"{self.wrap(n.lhs, 1000)}\\left[{self.tex(n.rhs)}\\right]"
            if base == "⁻¹'":  # preimage f ⁻¹' s
                return f"{{{self.wrap(n.lhs, 1000)}}}^{{-1}}\\left[{self.tex(n.rhs)}\\right]"
            if base == "^":
                return f"{{{self.wrap(n.lhs, 1000) if not self.atom(n.lhs) else self.tex(n.lhs)}}}^{{{self.tex(n.rhs)}}}"
            info = infix_info(op) or (50, False)
            p, right = info
            lhs = self.wrap(n.lhs, p + (1 if right else 0))
            rhs = self.wrap(n.rhs, p + (0 if right else 1))
            if base == "*" and isinstance(n.lhs, Num) and isinstance(n.rhs, (Name, App)):
                return f"{lhs}\\,{rhs}"
            sym = _OPS.get(base, _sym(base))
            if base.endswith("ᶠ"):
                sym = _OPS.get(base[:-1], base[:-1]) + r"^{\mathrm{ev}}"
            if base in ("..",):
                return f"[{lhs}, {rhs}]"
            return f"{lhs} {sym} {rhs}"
        if isinstance(n, Pre):
            if n.op in ("↑", "⇑", "↥"):
                return self.tex(n.arg)
            if n.op == "√":
                return rf"\sqrt{{{self.tex(n.arg)}}}"
            if n.op == "¬":
                return rf"\neg {self.wrap(n.arg, 40)}"
            if n.op == "#":  # cardinality
                return rf"\#{self.wrap(n.arg, 75)}"
            return f"{_sym(n.op)}{self.wrap(n.arg, 75)}"
        if isinstance(n, Post):
            sup = {"⁻¹": "^{-1}", "ᶜ": "^{c}", "!": "!"}.get(n.op, f"^{{{_sym(n.op)}}}")
            inner = self.tex(n.arg)
            return (inner if self.atom(n.arg) else rf"\left({inner}\right)") + sup
        if isinstance(n, App):
            fn, args = n.fn, n.args
            if isinstance(fn, Name):
                last = fn.id.rsplit(".", 1)[-1]
                if last in ("sqrt", "√") and len(args) == 1:
                    return rf"\sqrt{{{self.tex(args[0])}}}"
                if last in ("abs",) and len(args) == 1:
                    return rf"\left\lvert {self.tex(args[0])} \right\rvert"
                if last in ("norm",) and len(args) == 1:
                    return rf"\left\lVert {self.tex(args[0])} \right\rVert"
                if fn.id in ("Finset.range",) and len(args) == 1:
                    return rf"\{{0, \dots, {self.tex(args[0])} - 1\}}"
                if last in ("cast", "ofReal", "ofReal'", "toReal", "toNNReal", "ofNat", "coe") and len(args) == 1:
                    return self.tex(args[0])
            f = self.tex(fn) if isinstance(fn, (Name, Proj)) else rf"\left({self.tex(fn)}\right)"
            return f + r"\left(" + ", ".join(self.tex(a) for a in args) + r"\right)"
        if isinstance(n, Quant):
            body = self.tex(n.body)
            if n.q in ("fun", "λ"):
                return f"{self.binders(n)} \\mapsto {body}"
            if n.q in ("∀", "∃", "∃!"):
                cmd = {"∀": r"\forall", "∃": r"\exists", "∃!": r"\exists!"}[n.q]
                return f"{cmd} {self.binders(n)},\\ {body}"
            if n.q == "∫" or n.q.startswith("∫"):
                dom = n.extra.get("domain")
                lim = ""
                if isinstance(dom, Bin) and dom.op == "..":
                    lim = f"_{{{self.tex(dom.lhs)}}}^{{{self.tex(dom.rhs)}}}"
                elif dom is not None:
                    lim = f"_{{{self.tex(dom)}}}"
                var = ", ".join(self.name(x) for b in n.binders for x in b.names) or "x"
                meas = n.extra.get("measure")
                d = rf"\,d{self.tex(meas)}" if meas is not None else rf"\,d{var}"
                return rf"\int{lim} {body}{d}"
            if n.q in _BIG:
                return f"{_BIG[n.q]}_{{{self.binders(n)}}} {body}"
            return f"{_sym(n.q)} {self.binders(n)},\\ {body}"
        if isinstance(n, Ite):
            return (r"\begin{cases} %s & \text{if } %s,\\ %s & \text{otherwise} \end{cases}"
                    % (self.tex(n.then), self.tex(n.cond), self.tex(n.other)))
        if isinstance(n, Tuple):
            inner = ", ".join(self.tex(x) for x in n.items)
            return rf"\langle {inner} \rangle" if n.anon else rf"\left({inner}\right)"
        if isinstance(n, SetOf):
            return rf"\left\{{ {self.tex(n.var)} \mid {self.tex(n.pred)} \right\}}"
        if isinstance(n, SetLit):
            return r"\{" + ", ".join(self.tex(x) for x in n.items) + r"\}"
        if isinstance(n, Delim):
            inner = ", ".join(self.tex(x) for x in n.items)
            return {"abs": rf"\left\lvert {inner} \right\rvert", "norm": rf"\left\lVert {inner} \right\rVert",
                    "floor": rf"\left\lfloor {inner} \right\rfloor", "ceil": rf"\left\lceil {inner} \right\rceil",
                    "inner": rf"\left\langle {inner} \right\rangle", "list": rf"[{inner}]"}.get(n.kind, f"[{inner}]")
        if isinstance(n, Proj):
            inner = self.tex(n.expr)
            if n.field in _FUNC:
                return rf"{_FUNC[n.field]}\left({inner}\right)" if n.field != "card" else rf"\#{inner}"
            if n.field in ("symm", "inv"):
                return f"{{{inner}}}^{{-1}}"
            return rf"{inner}.\mathrm{{{_esc_math_text(n.field)}}}"
        return r"\text{?}"


def _sym(op: str) -> str:
    """An operator with no LaTeX name: printed as itself, LaTeX's special characters escaped."""
    return "".join({"#": r"\#", "%": r"\%", "&": r"\&", "$": r"\$", "_": r"\_", "~": r"\sim ",
                    "^": r"\hat{}", "\\": r"\setminus "}.get(ch, ch) for ch in op)


def safe_math(s: str) -> bool:
    r"""Cheap check that a math string can't break the document: balanced braces and
    \left/\right, no unescaped special characters."""
    depth = 0
    k = 0
    while k < len(s):
        ch = s[k]
        if ch == "\\":
            k += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        elif ch in "#%&$":
            return False
        k += 1
    return depth == 0 and s.count("\\left") == s.count("\\right") and s.count("\\begin{") == s.count("\\end{")


def tex_math(n: Node, locals_: set[str] | None = None) -> str:
    """A parsed Lean term as LaTeX math (without the surrounding $...$)."""
    try:
        s = _Tex(locals_ or set()).tex(n)
    except RecursionError:
        return r"\text{(formula too deep)}"
    if len(s) > _MAX_TEX:
        return r"\text{(long formula, see the Lean source)}"
    if not safe_math(s):
        return r"\text{(formula, see the Lean source)}"
    return s


# ---------------------------------------------------------------- text
_TEXT_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
             "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s: str) -> str:
    """Escape plain text for LaTeX; symbols the text font lacks go into math mode."""
    out = []
    for ch in s:
        if ch in _TEXT_ESC:
            out.append(_TEXT_ESC[ch])
        elif ord(ch) > 0x2000 and unicodedata.category(ch)[0] in "SP" and ch not in "–—‘’“”…":
            out.append(f"${ch}$")
        else:
            out.append(ch)
    return "".join(out)


def esc_code(s: str) -> str:
    return "".join(_TEXT_ESC.get(ch, ch) for ch in s)


_MARK = re.compile("\x01(.*?)\x02|`([^`]+)`", re.S)


def render_marked(s: str, local_words: set[str] = frozenset()) -> str:
    """Text with \\x01math\\x02 and `code` segments -> LaTeX."""
    out, pos = [], 0
    for m in _MARK.finditer(s):
        out.append(_words(s[pos:m.start()], local_words))
        out.append(f"${m.group(1)}$" if m.group(1) is not None else r"\leanc{%s}" % esc_code(m.group(2)))
        pos = m.end()
    out.append(_words(s[pos:], local_words))
    return "".join(out)


def _words(text: str, local_words: set[str]) -> str:
    """Escape text; variable names (`a`, `x₁`) become math."""
    parts = re.split(r"(?<![\w.])([^\W\d][\w'₀-₉]*)(?!\w|\.\w)", text)
    out = []
    for k, p in enumerate(parts):
        single = len(p) == 1 and p.isalpha() and p not in "aAI"  # one-letter words are variables
        # "a" followed by a word is the article ("be a matrix"), not the variable a
        article = (p == "a" and k + 2 < len(parts) and parts[k + 1] == " "
                   and parts[k + 2][:1].islower() and parts[k + 2] not in local_words)
        if k % 2 == 1 and not p.startswith("_") and not article and (p in local_words and len(p) <= 3 or single):
            out.append(f"${_ident(p, False)}$")
        else:
            out.append(esc(p))
    return "".join(out)


def statement_tex(d: Decl) -> str:
    """The English statement with every formula typeset."""
    tr = gloss_decl(d)
    locals_ = {x for b in d.binders + d.context for x in b.names}
    masked = re.sub(r"^\w+ [^:]+: ", "", tr.masked)  # the heading names it already
    out, pos = [], 0
    for m in _SLOT_RE.finditer(masked):
        k = int(m.group(1))
        out.append(_words(masked[pos:m.start()], locals_))
        node = tr.slot_nodes[k] if k < len(tr.slot_nodes) else None
        out.append(f"${tex_math(node, locals_)}$" if node is not None else esc(tr.slots[k]))
        pos = m.end()
    out.append(_words(masked[pos:], locals_))
    return "".join(out)


# ---------------------------------------------------------------- document
PREAMBLE = r"""\documentclass[10pt,openany]{book}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmonofont{DejaVu Sans Mono}[Scale=0.8]
\usepackage{adjustbox}
\usepackage{tikz}
\usetikzlibrary{graphs,graphdrawing,arrows.meta,quotes}
\usegdlibrary{layered,force}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{enumitem}
\setlistdepth{9}
\renewlist{itemize}{itemize}{9}
\setlist[itemize]{label=--,nosep,leftmargin=1.4em}
\usepackage[hidelinks]{hyperref}
\newcommand{\leanc}[1]{\texttt{#1}}
\newenvironment{leandecl}[3]{\par\medskip\noindent\hypertarget{#3}{}\textbf{#1} \leanc{#2}.\ }{\par}
\newenvironment{leanproof}{\par\noindent\textit{Proof.}\ }{\hfill\rule{0.55em}{0.55em}\par}
\sloppy\emergencystretch=4em
\setcounter{tocdepth}{0}
"""


def _area(rel: str, depth: int = 2) -> str:
    parts = Path(rel).with_suffix("").parts
    return "/".join(parts[:depth]) if len(parts) > depth else (parts[0] if len(parts) > 1 else "(top level)")


def _module(rel: str) -> str:
    return ".".join(Path(rel).with_suffix("").parts)


def _imports(src: str) -> list[str]:
    return re.findall(r"^\s*(?:public\s+)?import\s+([\w.]+)", src, re.M)


def _node_id(s: str) -> str:
    return "n" + re.sub(r"[^A-Za-z0-9]", "", s.title()) + str(abs(hash(s)) % 997)


def _short_label(s: str, width: int = 22) -> str:
    return esc(s if len(s) <= width else s[: width - 1] + "…")


def import_map_tex(files: dict[str, str], max_areas: int = 28) -> str:
    """Areas of the project (first two folders) and which import which, laid out in layers."""
    mod_area = {_module(rel): _area(rel) for rel in files}
    size = Counter()
    edges = Counter()
    for rel, src in files.items():
        a = _area(rel)
        size[a] += src.count("\n")
        for imp in _imports(src):
            b = mod_area.get(imp)
            if b and b != a:
                edges[(a, b)] += 1
    keep = {a for a, _ in size.most_common(max_areas)}
    lines = [r"\begin{tikzpicture}",
             r"\graph[layered layout, grow=down, level distance=1.6cm, sibling distance=0.4cm,"
             r" nodes={draw, rounded corners, font=\scriptsize, fill=blue!6}, edges={-{Stealth[length=1.6mm]}, gray}] {"]
    for a in sorted(keep):
        lines.append(f'  {_node_id(a)}/"{_short_label(a)}";')
    for (a, b), _ in edges.items():
        if a in keep and b in keep:
            lines.append(f"  {_node_id(a)} -> {_node_id(b)};")
    lines += ["};", r"\end{tikzpicture}"]
    return "\n".join(lines)


def bar_chart_tex(data: list[tuple[str, float]], xlabel: str) -> str:
    if not data:
        return ""
    syms = ",".join("{" + _short_label(k, 30) + "}" for k, _ in data)
    coords = " ".join(f"({v},{{{_short_label(k, 30)}}})" for k, v in data)
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[xbar, width=0.8\textwidth, height={0.45 * len(data) + 1.5:.1f}cm, xmin=0,
  symbolic y coords={{{syms}}}, ytick=data, y dir=reverse, xlabel={{{xlabel}}},
  yticklabel style={{font=\scriptsize}}, nodes near coords, nodes near coords style={{font=\tiny}},
  enlarge y limits=0.03, bar width=6pt]
\addplot[fill=blue!40, draw=blue!70] coordinates {{{coords}}};
\end{{axis}}
\end{{tikzpicture}}"""


def histogram_tex(values: list[int], xlabel: str, bins: list[int]) -> str:
    counts = [0] * len(bins)
    for v in values:
        for k in range(len(bins) - 1, -1, -1):
            if v >= bins[k]:
                counts[k] += 1
                break
    labels = [f"{bins[k]}-{bins[k + 1] - 1}" if k + 1 < len(bins) else f"{bins[k]}+" for k in range(len(bins))]
    syms = ",".join(labels)
    coords = " ".join(f"({lab},{c})" for lab, c in zip(labels, counts))
    return rf"""\begin{{tikzpicture}}
\begin{{axis}}[ybar, width=0.85\textwidth, height=6cm, ymin=0, symbolic x coords={{{syms}}}, xtick=data,
  xlabel={{{xlabel}}}, ylabel={{proofs}}, x tick label style={{font=\scriptsize, rotate=45, anchor=east}},
  nodes near coords, nodes near coords style={{font=\tiny}}, bar width=10pt]
\addplot[fill=orange!50, draw=orange!80] coordinates {{{coords}}};
\end{{axis}}
\end{{tikzpicture}}"""


def proof_map_tex(roots: list[str], uses: dict[str, list[str]], max_nodes: int = 30, depth: int = 3) -> str:
    """What the headline theorems are proved from: names their proofs mention, a few levels deep."""
    seen, order, q = set(), [], deque((r, 0) for r in roots if r in uses)
    edges = []
    while q and len(order) < max_nodes:
        n, d = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        if d >= depth:
            continue
        for m in uses.get(n, [])[:6]:
            edges.append((n, m))
            if m not in seen:
                q.append((m, d + 1))
    shown = set(order)
    lines = [r"\begin{tikzpicture}",
             r"\graph[layered layout, grow=down, level distance=1.5cm, sibling distance=0.3cm,"
             r" nodes={draw, rounded corners, font=\tiny, fill=green!6}, edges={-{Stealth[length=1.5mm]}, gray}] {"]
    for n in order:
        fill = ", fill=red!12" if n in roots else ""
        lines.append(f'  {_node_id(n)}/"{_short_label(n.rsplit(".", 1)[-1], 26)}"[{fill.lstrip(", ")}];')
    for a, b in edges:
        if a in shown and b in shown:
            lines.append(f"  {_node_id(a)} -> {_node_id(b)};")
    lines += ["};", r"\end{tikzpicture}"]
    return "\n".join(lines)


def decl_tex(d: Decl, tactic_counts: Counter, proof_lengths: list[int], means: dict[str, str] | None = None) -> str:
    from .proofs import narrate

    kind = {"theorem": "Theorem", "lemma": "Lemma", "def": "Definition", "abbrev": "Definition",
            "structure": "Structure", "class": "Class", "inductive": "Type", "instance": "Instance",
            "example": "Example", "axiom": "Axiom"}.get(d.kind, d.kind.title())
    name = d.full_name or "(unnamed)"
    try:
        stmt = statement_tex(d)
    except Exception:
        stmt = r"\leanc{%s}" % esc_code(d.signature[:400])
    from .proofs import prop_name

    pn = prop_name(d)
    if pn and means:
        meaning = means.get(pn) or means.get(pn.rsplit(".", 1)[-1])
        if meaning:
            stmt += r" Here, \emph{%s} means: %s" % (esc(pn), meaning)
    out = [r"\begin{leandecl}{%s}{%s}{%s}" % (kind, esc_code(name), esc_code(name)), stmt, r"\end{leandecl}"]
    try:
        steps = narrate(d, Glosser(), tex=True)
    except Exception:
        steps = []
    if steps:
        proof_lengths.append(len(steps))
        for t in d.proof[2:]:
            if t.kind == "ident" and t.col > 0:
                pass
        locals_ = {x for b in d.binders + d.context for x in b.names}
        out.append(r"\begin{leanproof}")
        out.append(r"\begin{itemize}")
        level = 0
        for depth, s in steps:
            depth = min(depth, 8)
            while level < depth:
                out.append(r"\begin{itemize}")
                level += 1
            while level > depth:
                out.append(r"\end{itemize}")
                level -= 1
            out.append(r"\item " + render_marked(s, locals_))
        while level > 0:
            out.append(r"\end{itemize}")
            level -= 1
        out.append(r"\end{itemize}")
        out.append(r"\end{leanproof}")
    return "\n".join(out)


def _tactic_heads(d: Decl) -> list[str]:
    toks = d.proof
    if len(toks) < 2 or toks[1].text != "by":
        return []
    heads = []
    last_line = toks[1].line
    for t in toks[2:]:
        if t.line != last_line and t.kind in ("ident", "sym") and t.text not in ("·", ".", "|", "_", ")", "]", "⟩"):
            heads.append(t.text)
        last_line = t.line
    return heads


def _proof_mentions(d: Decl, project_names: dict[str, str]) -> list[str]:
    out = []
    for t in d.proof:
        if t.kind == "ident":
            full = project_names.get(t.text) or project_names.get(t.text.rsplit(".", 1)[-1])
            if full and full != d.full_name and full not in out:
                out.append(full)
    return out


def write_book(src_dir: str, output: str, title: str, first: list[str], headline: list[str],
               max_files: int = 0) -> dict:
    root = Path(src_dir)
    paths = sorted(p for p in root.rglob("*.lean") if ".lake" not in p.parts)
    files = {str(p.relative_to(root)): p.read_text(encoding="utf-8", errors="replace") for p in paths}
    # order: start at the headline files and walk down the imports (top-down reading order)
    mod_file = {_module(rel): rel for rel in files}
    starts = [rel for rel in files if any(rel.endswith(f) for f in first)] or sorted(files)[:1]
    order, seen, q = [], set(), deque(starts)
    while q:
        rel = q.popleft()
        if rel in seen:
            continue
        seen.add(rel)
        order.append(rel)
        for imp in _imports(files[rel]):
            if imp in mod_file and mod_file[imp] not in seen:
                q.append(mod_file[imp])
    order += [rel for rel in sorted(files) if rel not in seen]
    if max_files:
        order = order[:max_files]

    sys.setrecursionlimit(20000)
    decls = {rel: extract_decls(files[rel], rel) for rel in order}
    names: dict[str, str] = {}
    for ds in decls.values():
        for d in ds:
            if d.full_name:
                names.setdefault(d.full_name, d.full_name)
                names.setdefault(d.full_name.rsplit(".", 1)[-1], d.full_name)
    from .proofs import KNOWN_PROPS, meanings

    means = {k: v[1] for k, v in KNOWN_PROPS.items()}
    for mname, md in meanings([d for ds in decls.values() for d in ds]).items():
        try:
            means[mname] = statement_tex(md)
        except Exception:
            pass
    tactic_counts: Counter = Counter()
    proof_lengths: list[int] = []
    uses: dict[str, list[str]] = {}
    body_parts: list[str] = []
    current_area = None
    n_decls = 0
    for rel in order:
        area = _area(rel)
        if area != current_area:
            body_parts.append(r"\chapter{%s}" % esc(area))
            current_area = area
        body_parts.append(r"\section*{%s}\addcontentsline{toc}{section}{%s}" % (esc(rel), esc(rel)))
        for d in decls[rel]:
            n_decls += 1
            tactic_counts.update(_tactic_heads(d))
            if d.full_name:
                uses[d.full_name] = _proof_mentions(d, names)
            body_parts.append(decl_tex(d, tactic_counts, proof_lengths, means))

    roots = [names[h] for h in headline if h in names]
    lines_per_area = Counter()
    for rel in order:
        lines_per_area[_area(rel, 1) if "/" in rel else "(top level)"] += files[rel].count("\n")
    finer = Counter()
    for rel in order:
        finer[_area(rel)] += files[rel].count("\n")
    overview = [
        r"\chapter{Overview}",
        rf"This book was produced from the Lean source by fixed translation rules, with no AI: "
        rf"{len(order):,} files, {sum(files[r].count(chr(10)) for r in order):,} lines and {n_decls:,} "
        rf"declarations. Each statement is rendered in English with its formulas typeset; each proof is "
        rf"narrated step by step from its tactics. Names in \leanc{{this font}} are Lean names or Lean "
        rf"code that no rule covers yet.",
        r"\begin{figure}[p]\centering\begin{adjustbox}{max width=\textwidth, max totalheight=0.8\textheight}",
        import_map_tex({r: files[r] for r in order}),
        r"\end{adjustbox}\caption{Which parts of the project build on which (an arrow $A \to B$ means a file in $A$ "
        r"imports one in $B$; the largest areas are shown).}\end{figure}",
        r"\begin{figure}[p]\centering",
        bar_chart_tex(finer.most_common(25), "lines of Lean"),
        r"\caption{The largest areas of the project.}\end{figure}",
        r"\begin{figure}[p]\centering",
        bar_chart_tex(tactic_counts.most_common(20), "uses"),
        r"\caption{The proof steps (tactics) used most often.}\end{figure}",
        r"\begin{figure}[p]\centering",
        histogram_tex(proof_lengths, "steps per proof", [1, 2, 3, 5, 10, 20, 40, 80, 160]),
        r"\caption{How long the proofs are, in narrated steps.}\end{figure}",
    ]
    if roots:
        overview += [
            r"\begin{figure}[p]\centering\begin{adjustbox}{max width=\textwidth, max totalheight=0.8\textheight}",
            proof_map_tex(roots, uses),
            r"\end{adjustbox}\caption{What the main theorems (red) are proved from: the results each proof mentions, "
            r"a few levels down.}\end{figure}",
        ]
    doc = [PREAMBLE, r"\title{%s}\author{Translated from Lean by rules}\date{}" % esc(title),
           r"\begin{document}", r"\maketitle", r"\tableofcontents"] + overview + body_parts + [r"\end{document}"]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(doc) + "\n")
    return {"files": len(order), "declarations": n_decls, "proofs": len(proof_lengths),
            "steps": sum(proof_lengths), "headline_found": roots}
