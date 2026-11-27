"""Static mode: no compiler, no build. Just follow the names.

Starting from the theorems you want to keep, read every name their source
mentions, work out which declarations those names can mean, read *their*
source, and so on.  Anything never reached is deleted.  Pure text analysis,
so it scales to millions of lines.

Names are resolved the way Lean resolves them: a bare `foo` can only mean
`foo` in the current namespace, in a namespace opened with `open`, in the
namespace of the declaration being defined, or at the top level.  Local
variables and tactic names therefore match nothing.

Two things can't be decided from the text, so they're handled conservatively:

* **Dot notation** (`h.le`, `(f x).comp g`): the meaning depends on the type of
  `h`, which only the compiler knows.  `X.le` is kept if `X` comes from outside
  the project (core Lean) or `X` itself is kept.
* **Invisible uses**: `p + q` silently uses an `instance`, `simp` silently uses
  `@[simp]` lemmas, `@[to_additive]` generates declarations whose names never
  appear in the source.  Such declarations (see INVISIBLE_ATTRS) are kept as
  soon as anything their *statement* mentions is kept, or always if their
  statement mentions nothing from the project.  A simp lemma about matrices
  can't fire in a proof that never touches matrices.

So it may keep more than strictly needed; it never deletes something a kept
declaration mentions.  It can't prove the result still builds: run one build
afterwards (or use graph mode, which reads Lean's own dependency records).
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

from lean_translator.decls import extract_decls

from .spans import split_commands

# Attributes that make a declaration usable without its name being written.
INVISIBLE_ATTRS = {
    "simp", "ext", "instance", "coe", "norm_cast", "push_cast", "norm_num", "positivity", "aesop",
    "gcongr", "fun_prop", "continuity", "measurability", "mono", "refl", "symm", "trans",
    "induction_eliminator", "cases_eliminator", "default_instance", "macro", "tactic", "term_elab",
    "command_elab", "builtin_tactic", "simproc", "simprocs", "csimp", "implemented_by", "extern",
    "export", "app_unexpander", "delab", "to_additive", "to_dual", "simps", "reassoc", "elementwise",
    "bound", "field_simps", "grind", "env_linter", "match_pattern", "init", "builtin_init",
    "nontriviality", "zify_simps", "qify_simps", "rify_simps", "mfld_simps", "functor_norm",
    "monotonicity", "algebraize", "local", "scoped",
}
INVISIBLE_KINDS = {"instance", "class"}
SCOPE_WORDS = {"in", "hiding", "renaming", "scoped", "→", "(", ")"}
# Words the lexer sees as identifiers but that never name a declaration.
NOT_NAMES = {
    "_", "fun", "by", "at", "with", "only", "in", "let", "have", "show", "from", "calc", "match", "do",
    "then", "else", "if", "return", "where", "deriving", "extends", "using", "generalizing", "says",
    "def", "theorem", "lemma", "instance", "structure", "class", "abbrev", "inductive", "example",
    "variable", "alias", "open", "namespace", "section", "end", "private", "protected", "noncomputable",
    "Type", "Sort", "Prop", "this", "sorry", "termination_by", "decreasing_by", "omit", "include",
}


_I = sys.intern


def _shrink(items: list["Item"]) -> list["Item"]:
    """Intern every string and share identical tuples: a 17M-line project has
    millions of repeated identifiers, and this is what keeps memory in check."""
    ctxs: dict[tuple, tuple] = {}
    for it in items:
        it.names = [_I(n) for n in it.names]
        it.refs = [(_I(r), d) for r, d in it.refs]
        it.sig_refs = [_I(r) for r in it.sig_refs]
        it.ctx = ctxs.setdefault(it.ctx, tuple(_I(c) for c in it.ctx))
        it.head = _I(it.head)
        it.kind = _I(it.kind)
    return items


@dataclass(slots=True)
class Item:
    file: Path
    start: int
    end: int
    kind: str
    head: str
    names: list[str]  # full names this span declares
    refs: list[tuple[str, bool]]  # (identifier, is dot-notation)
    sig_refs: list[str]  # identifiers in the statement (before :=)
    ctx: tuple[str, ...]  # namespaces a bare name may live in ("" = top level)
    root: bool  # always kept
    invisible: bool = False


def _attrs(tokens) -> set[str]:
    out, depth = set(), 0
    for t in tokens:
        if t.text == "@[":
            depth = 1
            continue
        if depth:
            if t.text in ("[", "("):
                depth += 1
            elif t.text in ("]", ")"):
                depth -= 1
            elif t.kind == "ident" and depth == 1:
                out.add(t.text.split(".")[0])
        elif t.text in ("theorem", "lemma", "def", "instance", "structure", "class", "abbrev", "inductive"):
            break
    return out


def _prefixes(ns: str) -> list[str]:
    parts = ns.split(".") if ns else []
    return [".".join(parts[:k]) for k in range(len(parts), -1, -1)]


def _opened(tokens) -> list[str]:
    out = []
    for t in tokens[1:]:
        if t.text in ("in",):
            break
        if t.kind == "ident" and t.text not in SCOPE_WORDS:
            out.append(t.text)
    return out


def scan_file(path_str: str) -> list[Item]:
    path = Path(path_str)
    src = path.read_text(encoding="utf-8", errors="replace")
    decls = {d.line: d for d in extract_decls(src, path_str)}
    items: list[Item] = []
    # scope stack: (namespace string, opens, refs made by `variable` lines)
    stack: list[tuple[str, list[str], list[tuple[str, bool]]]] = [("", [], [])]
    for s in split_commands(src):
        ns = stack[-1][0]
        if s.head == "namespace" and len(s.tokens) > 1:
            name = s.tokens[1].text
            stack.append((f"{ns}.{name}" if ns else name, [], []))
        elif s.head in ("section", "noncomputable") or (s.head == "public" and "section" in [t.text for t in s.tokens[:3]]):
            stack.append((ns, [], []))
        elif s.head == "end" and len(stack) > 1:
            stack.pop()
        elif s.head == "open" and s.kind == "other":
            stack[-1][1].extend(_opened(s.tokens))

        names: list[str] = []
        decl_ns = ""
        if s.head == "alias":  # `alias foo := bar`, `alias ⟨a, b⟩ := bar`: declares names
            depth = 0
            for t in s.tokens[1:]:
                if t.text == ":=":
                    break
                if t.kind == "ident":
                    names.append(f"{ns}.{t.text}" if ns else t.text)
        if s.kind == "decl":
            for ln in range(s.start, s.end + 1):
                d = decls.get(ln)
                if d is not None:
                    full = d.full_name or f"<anonymous {d.kind} {path.name}:{d.line}>"
                    names.append(full)
                    names += [f"{full}.{f}" for f, _ in d.fields]
                    decl_ns = full.rsplit(".", 1)[0] if "." in full else ""
                    break

        # namespaces a bare name may live in, innermost first
        bases = _prefixes(ns)
        if decl_ns:
            bases = _prefixes(decl_ns) + bases
        opens = [o for _, os_, _ in stack for o in os_]
        # `open X in` written right before this command
        for t_i, t in enumerate(s.tokens):
            if t.col == 0 and t.text == "open" and any(x.text == "in" and x.line == t.line for x in s.tokens[t_i:]):
                opens += _opened([x for x in s.tokens[t_i:] if x.line == t.line])
        ctx = list(dict.fromkeys(bases + [f"{b}.{o}" if b else o for o in opens for b in bases]))

        refs: list[tuple[str, bool]] = []
        sig_refs: list[str] = []
        # names bound by this declaration's binders (and section variables) are locals
        local_names: set[str] = set()
        if s.kind == "decl":
            d0 = next((decls[ln] for ln in range(s.start, s.end + 1) if ln in decls), None)
            if d0 is not None:
                for b in d0.binders + d0.context:
                    local_names.update(b.names)
        # locals introduced inside the command: fun x / ∀ x / intro h / obtain ⟨a, ha⟩ / have h : ...
        binder_kw = {"fun", "λ", "∀", "∃", "∃!", "Π", "Σ", "∑", "∏", "⋃", "⋂", "intro", "intros", "rintro",
                     "obtain", "rcases", "rename_i", "next", "case", "have", "let", "set", "generalize",
                     "with", "refine", "choose", "use"}
        stop = {",", "=>", "↦", ":", ":=", "∈", "|", "←", "→"}
        toks_ = s.tokens
        for k, t in enumerate(toks_):
            if t.text in binder_kw:
                j = k + 1
                while j < len(toks_) and toks_[j].text not in stop and toks_[j].line == t.line:
                    if toks_[j].kind == "ident" and "." not in toks_[j].text and t.text != "use":
                        local_names.add(toks_[j].text)
                    j += 1
        if s.head in ("open", "namespace", "section", "end", "import", "module", "set_option", "assert_not_exists"):
            items.append(Item(path, s.start, s.end, s.kind, s.head, [], [], [], (), True))
            continue
        in_sig, depth, attr_depth = True, 0, 0
        prev = None
        toks = s.tokens
        option_lines = {t.line for t in toks if t.col == 0 and t.text == "set_option"}
        attr_name = ""
        for k, t in enumerate(toks):
            nxt = toks[k + 1].text if k + 1 < len(toks) else ""
            if t.line in option_lines:
                prev = t
                continue
            if t.text == "@[":
                attr_depth = 1
                prev = t
                continue
            if attr_depth:
                if t.text in ("[", "("):
                    attr_depth += 1
                elif t.text in ("]", ")"):
                    attr_depth -= 1
            if t.text in "([{⟨":
                depth += 1
            elif t.text in ")]}⟩":
                depth -= 1
            elif depth == 0 and t.text in (":=", "where", "|"):
                in_sig = False
            if t.kind == "ident":
                ident = t.text.lstrip("@")
                is_attr_name = attr_depth == 1 and prev is not None and prev.text in ("@[", ",")
                if is_attr_name:
                    attr_name = ident
                # inside @[...] only `implemented_by foo` is a real use of foo
                in_attr_meta = attr_depth > 0 and attr_name not in ("implemented_by", "extern")
                named_arg = nxt == ":=" and prev is not None and prev.text == "("
                head = ident.split(".", 1)[0]
                if not (ident in NOT_NAMES or is_attr_name or in_attr_meta or named_arg or ident in local_names):
                    dotted = prev is not None and prev.text == "." and prev.line == t.line and prev.col + 1 == t.col
                    # `h.foo` with h a local: dot notation on h
                    if head in local_names and "." in ident:
                        dotted = True
                        ident = ident.split(".", 1)[1]
                    refs.append((ident, dotted))
                    if in_sig:
                        sig_refs.append(ident)
            prev = t
        if s.head == "variable":
            # applies to later declarations in this scope, not a reason to keep anything by itself
            stack[-1][2].extend(refs)
            items.append(Item(path, s.start, s.end, s.kind, s.head, [], [], [], (), True))
            continue
        if s.kind == "decl":
            for _, _, vrefs in stack:
                refs.extend(vrefs)
        attrs = _attrs(s.tokens)
        invisible = s.kind == "decl" and bool(s.head in INVISIBLE_KINDS or attrs & INVISIBLE_ATTRS)
        if s.kind == "attribute":
            # `attribute [simp] foo bar`: kept with (and keeping) foo and bar; see analyse()
            items.append(Item(path, s.start, s.end, "attribute", s.head, [], list(dict.fromkeys(refs)),
                              [], tuple(ctx), False))
            continue
        root = (s.kind in ("structural", "other") and s.head != "alias") or (s.kind == "decl" and not names)
        items.append(Item(path, s.start, s.end, "alias" if s.head == "alias" else s.kind, s.head, names,
                          list(dict.fromkeys(refs)), list(dict.fromkeys(sig_refs)), tuple(ctx), root, invisible))
    return _shrink(items)


@dataclass
class Result:
    items: list[Item]
    keep: set[int]
    deleted: list[Item]
    seconds: float


class Resolver:
    def __init__(self, items: list[Item]):
        self.items = items
        self.by_full: dict[str, list[int]] = defaultdict(list)
        self.by_last: dict[str, list[int]] = defaultdict(list)
        for k, it in enumerate(items):
            for n in it.names:
                self.by_full[n].append(k)
                self.by_last[n.rsplit(".", 1)[-1]].append(k)
        self.cache: dict[tuple, tuple[list[int], list[tuple[int, list[int]]]]] = {}
        self.cache_limit = 2_000_000
        self.dot_cache: dict[str, tuple] = {}

    def lookup(self, full: str, generated: bool) -> list[int]:
        """Exact name, or (if `generated`) a name generated from a declaration
        (Foo.mk, foo.eq_1, Foo.ext_iff, ...) -> that declaration."""
        hit = self.by_full.get(full)
        if hit:
            return hit
        if generated and "." in full:
            return self.by_full.get(full.rsplit(".", 1)[0], [])
        return []

    def _dot(self, p: str) -> tuple[tuple[int, ...], tuple[tuple[int, tuple[int, ...]], ...]]:
        """Candidates for dot notation `_.p`, built once per field name and shared."""
        hit = self.dot_cache.get(p)
        if hit is not None:
            return hit
        now, later = [], []
        for i in self.by_last.get(p, ()):
            for n in self.items[i].names:
                if "." not in n or n.rsplit(".", 1)[-1] != p:
                    continue
                owner = self.by_full.get(n.rsplit(".", 1)[0])
                if owner:
                    later.append((i, tuple(owner)))
                else:
                    now.append(i)  # namespace from core Lean: can't rule it out
        res = (tuple(now), tuple(later))
        self.dot_cache[p] = res
        return res

    def resolve(self, ident: str, dotted: bool, ctx: tuple[str, ...]):
        """Returns (lists of declarations to keep now,
                    lists of (candidate, namespace decls it waits on)).
        Lists are shared, never copied: iterate, don't mutate."""
        key = (ident, dotted, ctx)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        nows: list = []
        laters: list = []
        multi = "." in ident.removeprefix("_root_.")
        if ident.startswith("_root_."):
            nows.append(self.lookup(ident[len("_root_."):], multi))
        elif not dotted:
            for base in ctx:
                found = self.lookup(f"{base}.{ident}" if base else ident, multi)
                if found:
                    nows.append(found)
        parts = [p for p in ident.removeprefix("_root_.").split(".") if p]
        # Dot notation: `h.foo.bar`, `(e).foo`, `.foo` -> fields of unknown types.
        # A capitalised head (`Prod.mk`, `List.map`) is a namespace, not a local.
        local_head = bool(parts) and not parts[0][:1].isupper()
        if dotted or (not nows and len(parts) > 1 and local_head):
            for p in (parts if dotted else parts[1:]):
                dn, dl = self._dot(p)
                if dn:
                    nows.append(dn)
                if dl:
                    laters.append(dl)
        res = (tuple(nows), tuple(laters))
        if len(self.cache) >= self.cache_limit:
            self.cache.clear()  # bounded memory; entries are cheap to recompute
        self.cache[key] = res
        return res


def analyse(files: list[Path], targets: list[str], jobs: int = 0) -> Result:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs or os.cpu_count()) as ex:
        per_file = list(ex.map(scan_file, [str(f) for f in files], chunksize=32))
    items = [it for chunk in per_file for it in chunk]
    del per_file
    items = _shrink(items)  # strings from different worker processes are separate copies
    R = Resolver(items)

    seeds: list[int] = []
    for t in targets:
        hit = R.by_full.get(t) or [i for i in R.by_last.get(t.rsplit(".", 1)[-1], [])
                                   if any(n.endswith("." + t) for n in items[i].names)]
        if not hit:
            sys.exit(f"target not found: {t}")
        seeds += hit
    seeds += [k for k, it in enumerate(items) if it.root]

    waiting: dict[int, list[int]] = defaultdict(list)
    for k, it in enumerate(items):
        if it.kind == "attribute":
            attr_names, depth = [], 0
            for ident, dotted in it.refs:
                nows, _ = R.resolve(ident, dotted, it.ctx)
                for i in chain.from_iterable(nows):
                    waiting[i].append(k)  # kept as soon as any target is kept...
                    if not items[i].invisible:
                        items[i].invisible = True  # ...and its targets become invisibly usable
    for k, it in enumerate(items):
        if not it.invisible:
            continue
        mentioned = set()
        for ident in it.sig_refs:
            nows, _ = R.resolve(ident, False, it.ctx)
            mentioned.update(i for i in chain.from_iterable(nows) if i != k)
        if not mentioned:
            seeds.append(k)  # mentions nothing from the project: could fire anywhere
        for i in mentioned:
            waiting[i].append(k)

    keep: set[int] = set()
    stack = list(seeds)
    while stack:
        k = stack.pop()
        if k in keep:
            continue
        keep.add(k)
        stack.extend(w for w in waiting.pop(k, ()) if w not in keep)
        it = items[k]
        for ident, dotted in it.refs:
            nows, laters = R.resolve(ident, dotted, it.ctx)
            stack.extend(i for i in chain.from_iterable(nows) if i not in keep)
            for cand, owners in chain.from_iterable(laters):
                if cand in keep:
                    continue
                if any(o in keep for o in owners):
                    stack.append(cand)
                else:
                    for o in owners:
                        waiting[o].append(cand)
    deleted = [it for k, it in enumerate(items) if k not in keep and it.kind in ("decl", "debug", "alias", "attribute")]
    return Result(items, keep, deleted, time.time() - t0)


def render(src: str, dels: list[Item]) -> str:
    lines = src.split("\n")
    drop = {k for it in dels for k in range(it.start, it.end + 1)}
    return re.sub(r"\n{3,}", "\n\n", "\n".join(ln for k, ln in enumerate(lines, 1) if k not in drop))


def run(args) -> None:
    root = Path(args.root).resolve()
    dirs = [root / d for d in args.include] if args.include else [root]
    files = sorted(p for d in dirs for p in ([d] if d.is_file() else d.rglob("*.lean"))
                   if ".lake" not in p.parts and not p.name.startswith(".slop_"))
    res = analyse(files, args.keep, args.jobs)
    decls = [it for it in res.items if it.kind in ("decl", "alias")]
    del_decls = [it for it in res.deleted if it.kind in ("decl", "alias")]
    del_lines = sum(it.end - it.start + 1 for it in res.deleted)
    total_lines = sum(len(f.read_text(encoding="utf-8", errors="replace").split("\n")) for f in files)
    touched = defaultdict(list)
    for it in res.deleted:
        touched[it.file].append(it)
    print(f"scanned {len(files):,} files ({total_lines:,} lines, {len(decls):,} declarations) "
          f"in {res.seconds:.1f}s, no compiler")
    print(f"kept (needed, or possibly used invisibly): {len(decls) - len(del_decls):,} declarations")
    print(f"deletable: {len(del_decls):,} declarations + {len(res.deleted) - len(del_decls):,} "
          f"other commands (#check/#eval/example/attribute) = {del_lines:,} lines ({100 * del_lines / max(total_lines, 1):.0f}%) "
          f"in {len(touched):,} files")
    if args.show:
        for it in res.deleted[: args.show]:
            print(f"  - {it.file.relative_to(root)}:{it.start}  {it.head} {', '.join(it.names[:1])}")
        if len(res.deleted) > args.show:
            print(f"  ... and {len(res.deleted) - args.show:,} more")
    if getattr(args, "report", None):
        write_report(Path(args.report), root, files, res, total_lines, args.keep)
        print(f"report written to {args.report}/")
    if not args.write:
        print("dry run: nothing written (use --write; commit first so you can review and revert)")
        return
    for path, dels in touched.items():
        path.write_text(render(path.read_text(encoding="utf-8"), dels), encoding="utf-8")
    print(f"wrote {len(touched):,} files. Run one build (`lake build`) to confirm.")


def write_report(out: Path, root: Path, files: list[Path], res: Result, total_lines: int, targets: list[str]) -> None:
    """report.md (human summary) + deletions.tsv (every deletion) + kept.tsv (every kept declaration)."""
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(p.relative_to(root))
    decls = [it for it in res.items if it.kind in ("decl", "alias")]
    deleted_ids = {id(it) for it in res.deleted}
    with open(out / "deletions.tsv", "w", encoding="utf-8") as f:
        f.write("file\tstart_line\tend_line\tkind\tname\n")
        for it in res.deleted:
            f.write(f"{rel(it.file)}\t{it.start}\t{it.end}\t{it.head}\t{(it.names or [''])[0]}\n")
    with open(out / "kept.tsv", "w", encoding="utf-8") as f:
        f.write("file\tstart_line\tkind\tname\n")
        for it in decls:
            if id(it) not in deleted_ids:
                f.write(f"{rel(it.file)}\t{it.start}\t{it.head}\t{(it.names or [''])[0]}\n")

    del_lines = sum(it.end - it.start + 1 for it in res.deleted)
    by_dir: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # decls, deleted decls, deleted lines
    by_file: dict[str, int] = defaultdict(int)
    for it in decls:
        by_dir[rel(it.file).split("/")[0]][0] += 1
    for it in res.deleted:
        d = by_dir[rel(it.file).split("/")[0]]
        d[1] += it.kind in ("decl", "alias")
        d[2] += it.end - it.start + 1
        by_file[rel(it.file)] += it.end - it.start + 1
    files_with_kept = {it.file for it in decls if id(it) not in deleted_ids}
    files_fully_dead = len({it.file for it in res.deleted} - files_with_kept)
    lines = [
        "# SLOP_DESTROYER report (static mode, no compiler)",
        "",
        f"**Kept theorem(s):** {', '.join(f'`{t}`' for t in targets)}",
        "",
        "| | |",
        "|---|---|",
        f"| files scanned | {len(files):,} |",
        f"| lines scanned | {total_lines:,} |",
        f"| declarations | {len(decls):,} |",
        f"| deletable declarations | {sum(1 for it in res.deleted if it.kind in ('decl', 'alias')):,} |",
        f"| deletable other commands (#check, #eval, example, attribute) | "
        f"{sum(1 for it in res.deleted if it.kind not in ('decl', 'alias')):,} |",
        f"| deletable lines | {del_lines:,} ({100 * del_lines / max(total_lines, 1):.1f}%) |",
        f"| files left with no declarations | {files_fully_dead:,} |",
        f"| analysis time | {res.seconds:.0f}s |",
        "",
        "Static mode is conservative: it keeps anything that *might* be used invisibly "
        "(instances, `@[simp]` lemmas, ... whose statement mentions something kept). "
        "Nothing here has been checked by compiling; run one build on the result to confirm.",
        "",
        "## By top-level directory",
        "",
        "| directory | declarations | deletable | deletable lines |",
        "|---|---:|---:|---:|",
    ]
    for d, (n, nd, nl) in sorted(by_dir.items(), key=lambda kv: -kv[1][2]):
        lines.append(f"| `{d}` | {n:,} | {nd:,} | {nl:,} |")
    lines += ["", "## Files with the most deletable lines", "", "| file | deletable lines |", "|---|---:|"]
    for f_, nl in sorted(by_file.items(), key=lambda kv: -kv[1])[:40]:
        lines.append(f"| `{f_}` | {nl:,} |")
    lines += ["", "Full lists: `deletions.tsv`, `kept.tsv`."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
