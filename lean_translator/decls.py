"""Split a Lean file into declarations and keep track of their context.

Key speed trick: we never look inside proofs.  A declaration's *statement*
is everything between the declaration keyword and the first top-level
``:=`` / ``where`` / ``|``; the proof is skipped by jumping to the next
command that starts in column 0 (Mathlib style guarantees this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Token, tokenize

__all__ = ["Decl", "extract_decls", "split_binders"]

DECL_KEYWORDS = {
    "theorem", "lemma", "def", "abbrev", "instance", "structure", "class",
    "inductive", "example", "axiom", "opaque",
}
MODIFIERS = {
    "private", "protected", "noncomputable", "nonrec", "partial", "unsafe",
    "public", "scoped", "local", "irreducible_def", "@[", "meta",
}
# Every other column-0 word that starts a command; used to find where a
# proof ends.
OTHER_COMMANDS = {
    "variable", "variables", "section", "namespace", "end", "open", "import",
    "universe", "universes", "set_option", "attribute", "notation", "infix",
    "infixl", "infixr", "prefix", "postfix", "macro", "macro_rules", "syntax",
    "elab", "deriving", "#check", "#eval", "#print", "#guard", "#align",
    "alias", "initialize", "mutual", "add_decl_doc", "assert_not_exists",
    "noncomputable", "module", "omit", "include", "suppress_compilation",
    "library_note", "register_simp_attr", "run_cmd", "compile_inductive",
    "export", "termination_by", "decreasing_by", "@[", "#adaptation_note",
}
COMMAND_STARTS = DECL_KEYWORDS | MODIFIERS | OTHER_COMMANDS

_OPEN = {"(": ")", "[": "]", "{": "}", "⦃": "⦄", "⟨": "⟩", "{{": "}}", "@[": "]"}
_CLOSE = set(_OPEN.values())


@dataclass
class Binder:
    kind: str  # explicit | implicit | inst | strict
    names: list[str]
    type: list[Token]


@dataclass
class Decl:
    kind: str
    name: str
    binders: list[Binder]
    type: list[Token]  # the statement after the final top-level ':'
    doc: str | None
    file: str
    line: int
    namespace: str
    context: list[Binder] = field(default_factory=list)  # `variable`s in scope
    signature: str = ""  # original source text of the statement
    body: list[Token] = field(default_factory=list)  # short `:= term` bodies of defs
    fields: list[tuple[str, list[Token]]] = field(default_factory=list)  # structure fields
    extends: list[Token] = field(default_factory=list)
    proof: list[Token] = field(default_factory=list)  # tokens after the statement (theorems only)

    @property
    def full_name(self) -> str:
        if self.namespace and self.name and not self.name.startswith("_root_."):
            return f"{self.namespace}.{self.name}"
        return self.name.removeprefix("_root_.")


def _is_cmd_start(tok: Token) -> bool:
    return tok.col == 0 and (tok.text in COMMAND_STARTS or tok.kind in ("doc", "moddoc"))


def _match_close(toks: list[Token], i: int) -> int:
    """Index of the token closing the bracket at ``toks[i]`` (or len)."""
    depth = 0
    n = len(toks)
    while i < n:
        t = toks[i].text
        if toks[i].kind == "sym" or t in _OPEN or t in _CLOSE:
            if t in _OPEN:
                depth += 1
            elif t in _CLOSE:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return n


def split_binders(toks: list[Token]) -> tuple[list[Binder], list[Token]]:
    """Split ``(a b : T) [C a] {x} : stmt`` into binders and ``stmt``."""
    binders: list[Binder] = []
    i = 0
    n = len(toks)
    kinds = {"(": "explicit", "{": "implicit", "[": "inst", "⦃": "strict", "{{": "strict"}
    while i < n:
        t = toks[i].text
        if t in kinds and toks[i].kind == "sym":
            j = _match_close(toks, i)
            inner = toks[i + 1 : j]
            colon = _top_level_index(inner, ":")
            if colon is None:
                if t == "[":
                    binders.append(Binder("inst", [], inner))
                else:
                    binders.append(Binder(kinds[t], [x.text for x in inner], []))
            else:
                names = [x.text for x in inner[:colon]]
                binders.append(Binder(kinds[t], names, inner[colon + 1 :]))
            i = j + 1
        elif t == ":":
            return binders, toks[i + 1 :]
        elif toks[i].kind == "ident":  # untyped explicit binder: `def f x y : ...`
            binders.append(Binder("explicit", [t], []))
            i += 1
        else:
            return binders, toks[i:]
    return binders, []


def _top_level_index(toks: list[Token], text: str, start: int = 0) -> int | None:
    depth = 0
    for k in range(start, len(toks)):
        t = toks[k].text
        if t in _OPEN:
            depth += 1
        elif t in _CLOSE:
            depth -= 1
        elif depth == 0 and t == text:
            return k
    return None


def _statement_end(toks: list[Token], i: int, kind: str) -> int:
    """Index of the token ending the statement that starts at ``toks[i]``."""
    depth = 0
    n = len(toks)
    seen_colon = False
    while i < n:
        tok = toks[i]
        t = tok.text
        if tok.col == 0 and tok.text in COMMAND_STARTS:
            return i
        if t in _OPEN:
            depth += 1
        elif t in _CLOSE:
            depth -= 1
        elif depth <= 0:
            if t in (":=", "where"):
                return i
            if t == ":":
                seen_colon = True
            # `| pat => ...` starts a line; `|x|` (absolute value) sits inside one
            starts_line = i == 0 or toks[i - 1].line < tok.line
            if t == "|" and (kind in ("inductive",) or seen_colon and starts_line):
                return i
            if t in ("extends", "deriving", "with") and kind in ("structure", "class", "inductive"):
                return i
        i += 1
    return n


class _Scope:
    __slots__ = ("kind", "name", "variables")

    def __init__(self, kind: str, name: str):
        self.kind = kind
        self.name = name
        self.variables: list[Binder] = []


def extract_decls(src: str, file: str = "<input>") -> list[Decl]:
    toks = tokenize(src)
    lines = src.split("\n")
    out: list[Decl] = []
    scopes: list[_Scope] = [_Scope("file", "")]
    pending_doc: str | None = None
    i = 0
    n = len(toks)

    def namespace() -> str:
        return ".".join(s.name for s in scopes if s.kind == "namespace" and s.name)

    def skip_to_next_command(j: int) -> int:
        j += 1
        while j < n and not _is_cmd_start(toks[j]):
            j += 1
        return j

    while i < n:
        tok = toks[i]
        if tok.col != 0 and tok.kind != "doc":
            i += 1
            continue
        t = tok.text
        if tok.kind == "doc":
            pending_doc = tok.text
            i += 1
            continue
        if tok.kind == "moddoc":
            i += 1
            continue
        if t in ("namespace", "section", "noncomputable") and i + 1 < n:
            if t == "noncomputable" and toks[i + 1].text != "section":
                pass  # `noncomputable def ...` – handled below as a modifier
            else:
                if t == "noncomputable":
                    i += 1
                    t = "section"
                name = ""
                if i + 1 < n and toks[i + 1].kind == "ident" and toks[i + 1].line == toks[i].line:
                    name = toks[i + 1].text
                scopes.append(_Scope(t, name))
                i = skip_to_next_command(i)
                continue
        if t == "end":
            if len(scopes) > 1:
                scopes.pop()
            i = skip_to_next_command(i)
            continue
        if t in ("variable", "variables"):
            j = skip_to_next_command(i)
            binders, _ = split_binders(toks[i + 1 : j])
            scopes[-1].variables.extend(b for b in binders if b.type or b.kind == "inst")
            i = j
            continue
        if t in DECL_KEYWORDS or t in MODIFIERS:
            # Skip attributes and modifiers.
            j = i
            while j < n and toks[j].kind != "doc" and toks[j].text not in DECL_KEYWORDS:
                if toks[j].text == "@[":
                    j = _match_close(toks, j) + 1
                    continue
                if toks[j].text in MODIFIERS or toks[j].kind == "ident" and toks[j].text in ("public",):
                    j += 1
                    continue
                break
            if j < n and toks[j].kind == "doc":  # doc comment after attributes
                pending_doc = toks[j].text
                i = j + 1
                continue
            if j >= n or toks[j].text not in DECL_KEYWORDS:
                pending_doc = None
                i = skip_to_next_command(i)
                continue
            kind = toks[j].text
            k = j + 1
            name = ""
            if kind == "instance" and k < n and toks[k].text == "(" and k + 1 < n and toks[k + 1].text == "priority":
                k = _match_close(toks, k) + 1
            if kind != "example" and k < n and toks[k].kind == "ident" and toks[k].text not in ("extends",):
                # `instance foo : C` has a name; `instance [Foo] : C` does not.
                name = toks[k].text
                k += 1
            if kind == "instance" and k < n and toks[k].text == "(" and k + 1 < n and toks[k + 1].text == "priority":
                k = _match_close(toks, k) + 1
            end = _statement_end(toks, k, kind)
            stmt = toks[k:end]
            binders, typ = split_binders(stmt)
            first_line = toks[j].line
            last_line = toks[end - 1].line if end > k else first_line
            sig = "\n".join(lines[first_line - 1 : last_line])
            if end < n and toks[end].line == last_line:
                sig = sig[: _byte_col_cut(lines[last_line - 1], toks[end].col, sig)]
            ctx = [b for s in scopes for b in s.variables]
            decl = Decl(kind, name, binders, typ, pending_doc, file, toks[j].line, namespace(), ctx, sig.strip())
            nxt = skip_to_next_command(end - 1 if end > k else k)
            _attach_tail(decl, toks[end:nxt])
            if kind in ("theorem", "lemma", "example"):
                decl.proof = toks[end:nxt]
            out.append(decl)
            pending_doc = None
            i = nxt
            continue
        pending_doc = None
        i = skip_to_next_command(i)
    return out


def _byte_col_cut(last_line: str, col: int, sig: str) -> int:
    """Length of ``sig`` with the last line truncated at ``col``."""
    return len(sig) - (len(last_line) - col)


_MAX_BODY_TOKENS = 120


def _attach_tail(decl: Decl, tail: list[Token]) -> None:
    """Record short definition bodies and structure fields / parents."""
    if not tail:
        return
    head = tail[0].text
    if decl.kind in ("def", "abbrev") and head == ":=":
        body = tail[1:]
        if 0 < len(body) <= _MAX_BODY_TOKENS and body[0].text not in ("by", "match", "fun", "λ", "do", "{", "⟨", "where"):
            if not any(t.text in ("by", "match", "let", "have", "do", "where", "|") for t in body):
                decl.body = body
    elif decl.kind in ("structure", "class"):
        i = 0
        if head == "extends":
            j = 1
            while j < len(tail) and tail[j].text not in ("where", ":=", "with") and not (
                tail[j].kind == "ident" and j + 1 < len(tail) and tail[j + 1].text == "::"
            ):
                j += 1
            decl.extends = tail[1:j]
            i = j
        if i < len(tail) and tail[i].text in ("where", ":="):
            i += 1
        # Fields: `name : type` where `name` starts a line (doc comments skipped).
        indent = None
        k = i
        while k < len(tail):
            t = tail[k]
            line_start = k == 0 or tail[k - 1].line != t.line or tail[k - 1].kind == "doc"
            if t.kind == "ident" and line_start and (indent is None or t.col == indent):
                # collect names until ':'
                m = k
                names = []
                while m < len(tail) and tail[m].text in ("protected", "private"):
                    m += 1
                while m < len(tail) and tail[m].kind == "ident" and tail[m].line == t.line:
                    names.append(tail[m].text)
                    m += 1
                while m < len(tail) and tail[m].text in ("(", "{", "[") and tail[m].line == t.line:
                    m = _match_close(tail, m) + 1  # field binders, skip
                if m < len(tail) and tail[m].text == ":" and names:
                    if indent is None:
                        indent = t.col
                    e = m + 1
                    while e < len(tail) and not (tail[e].col <= indent and tail[e].line != tail[e - 1].line) and tail[e].text != ":=":
                        e += 1
                    for nm in names:
                        decl.fields.append((nm, tail[m + 1 : e]))
                    k = e
                    while k < len(tail) and not (tail[k].col <= indent and tail[k].line != tail[k - 1].line):
                        k += 1
                    continue
            k += 1
