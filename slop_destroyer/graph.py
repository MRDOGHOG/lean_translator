"""Graph mode: delete everything a set of target theorems doesn't need,
across a whole (already built) project, with no compile per deletion.

1. Read the dependency graph Lean recorded when it built the project
   (DumpDeps.lean: one load of the .olean files).
2. Walk it from the targets.  Everything reachable is needed, exactly:
   proof terms record every constant used, including instances and simp
   lemmas that never appear by name in the source.
3. Delete the declarations nothing reaches.  Kept regardless: tactic and
   elaborator machinery (it runs during checking but never appears in the
   finished proofs), notation/macros/options, and anything Lean couldn't
   place in the source.
4. Optionally run ONE build to confirm; if a deletion broke something the
   graph couldn't see, restore just the declarations the errors name and
   rebuild.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .spans import Span, split_commands

DUMP_SCRIPT = Path(__file__).with_name("DumpDeps.lean")


@dataclass
class Const:
    name: str
    module: str
    start: int
    end: int
    meta: bool
    deps: list[str]


@dataclass
class Plan:
    keep: set[str]
    deleted_spans: dict[Path, list[Span]] = field(default_factory=dict)
    names_by_span: dict[tuple[Path, int], list[str]] = field(default_factory=dict)


def dump_graph(root: Path, modules: list[str], prefixes: list[str], lean_cmd: list[str], out: Path) -> None:
    cmd = lean_cmd + ["--run", str(DUMP_SCRIPT), str(out), *prefixes, "--", *modules]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"dumping the dependency graph failed:\n{r.stderr[-3000:]}\n{r.stdout[-2000:]}")
    print(f"[graph] {r.stderr.strip()} in {time.time() - t0:.1f}s", file=sys.stderr)


def load_graph(path: Path) -> dict[str, Const]:
    g: dict[str, Const] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                parts += [""] * (6 - len(parts))
            name, mod, s, e, meta, deps = parts[:6]
            g[name] = Const(name, mod, int(s or 0), int(e or 0), meta == "true", deps.split() if deps else [])
    return g


def reachable(g: dict[str, Const], targets: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = [t for t in targets if t in g]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(d for d in g[n].deps if d in g and d not in seen)
    return seen


def module_path(root: Path, module: str) -> Path:
    return root / (module.replace(".", "/") + ".lean")


def plan(root: Path, g: dict[str, Const], targets: list[str], keep_modules: list[str]) -> Plan:
    missing = [t for t in targets if t not in g]
    if missing:
        sys.exit(f"not found in the built project: {', '.join(missing)} (use full names, e.g. Foo.bar)")
    roots = list(targets)
    # Machinery that runs during checking but leaves no trace in proof terms.
    roots += [c.name for c in g.values() if c.meta or any(c.module.startswith(m) for m in keep_modules)]
    keep = reachable(g, roots)

    by_module: dict[str, list[Const]] = defaultdict(list)
    for c in g.values():
        by_module[c.module].append(c)

    p = Plan(keep)
    for mod, consts in by_module.items():
        path = module_path(root, mod)
        if not path.exists():
            continue
        spans = split_commands(path.read_text(encoding="utf-8"))
        dels: list[Span] = []
        for s in spans:
            if s.kind == "debug":
                dels.append(s)  # #check / #eval / example: never needed by anything
                continue
            if s.kind != "decl":
                continue
            inside = [c for c in consts if c.start and s.start <= c.start <= s.end]
            if not inside:
                continue  # Lean placed nothing here; leave it alone
            if any(c.name in keep for c in inside):
                continue
            dels.append(s)
            p.names_by_span[(path, s.start)] = [c.name for c in inside]
        if dels:
            p.deleted_spans[path] = dels
    return p


def render(src: str, dels: list[Span]) -> str:
    lines = src.split("\n")
    drop = set()
    for s in dels:
        drop.update(range(s.start, s.end + 1))
    kept = [ln for k, ln in enumerate(lines, 1) if k not in drop]
    text = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", text)


def drop_attributes(src: str, deleted_names: set[str]) -> str:
    """Remove `attribute [..] foo bar` commands whose every target was deleted."""
    spans = [s for s in split_commands(src) if s.kind == "attribute"]
    if not spans:
        return src
    dead = []
    short = {n.rsplit(".", 1)[-1] for n in deleted_names} | deleted_names
    for s in spans:
        depth, names = 0, []
        for t in s.tokens[1:]:
            if t.text == "[":
                depth += 1
            elif t.text == "]":
                depth -= 1
            elif depth == 0 and t.kind == "ident":
                names.append(t.text)
        if names and all(n in short or n.rsplit(".", 1)[-1] in short for n in names):
            dead.append(s)
    return render(src, dead) if dead else src


def apply(p: Plan) -> dict[Path, str]:
    """New contents for every changed file (nothing is written here)."""
    deleted_names = {n for names in p.names_by_span.values() for n in names}
    out = {}
    for path, dels in p.deleted_spans.items():
        src = path.read_text(encoding="utf-8")
        out[path] = drop_attributes(render(src, dels), deleted_names)
    return out


_UNKNOWN = re.compile(r"unknown (?:identifier|constant|declaration) '([^']+)'")


def verify_and_repair(root: Path, p: Plan, originals: dict[Path, str], build_cmd: list[str], rounds: int) -> bool:
    """Build once; if errors name deleted declarations, restore those and rebuild."""
    for k in range(rounds):
        t0 = time.time()
        r = subprocess.run(build_cmd, cwd=root, capture_output=True, text=True)
        log = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"[verify] build OK ({time.time() - t0:.0f}s)", file=sys.stderr)
            return True
        wanted = set(_UNKNOWN.findall(log))
        restored = 0
        for path, dels in list(p.deleted_spans.items()):
            keep_back = [s for s in dels
                         if any(n in wanted or n.rsplit(".", 1)[-1] in {w.rsplit(".", 1)[-1] for w in wanted}
                                for n in p.names_by_span.get((path, s.start), []))]
            if keep_back:
                p.deleted_spans[path] = [s for s in dels if s not in keep_back]
                restored += len(keep_back)
        print(f"[verify] build failed (round {k + 1}); restoring {restored} declaration(s)", file=sys.stderr)
        if not restored:
            print(log[-4000:], file=sys.stderr)
            return False
        for path, text in originals.items():
            path.write_text(text, encoding="utf-8")
        for path, text in apply(p).items():
            path.write_text(text, encoding="utf-8")
    return False


def run(args) -> None:
    root = Path(args.root).resolve()
    lean_cmd = args.lean.split()
    dump = Path(args.graph) if args.graph else root / ".slop_destroyer_graph.tsv"
    if not args.graph:
        dump_graph(root, args.module, args.prefix or [m.split(".")[0] for m in args.module], lean_cmd, dump)
    g = load_graph(dump)
    p = plan(root, g, args.keep, args.keep_module)
    changes = apply(p)

    n_decl = sum(len(v) for v in p.names_by_span.values())
    n_spans = sum(len(v) for v in p.deleted_spans.values())
    before = sum(len(path.read_text(encoding="utf-8").split("\n")) for path in changes)
    after = sum(len(t.split("\n")) for t in changes.values())
    print(f"constants in project: {len(g):,}   needed: {len(p.keep):,}   "
          f"deletable: {n_decl:,} constants in {n_spans:,} commands across {len(changes):,} files")
    print(f"lines in touched files: {before:,} -> {after:,}")
    if args.show:
        for path, dels in sorted(p.deleted_spans.items()):
            for s in dels:
                print(f"  - {path.relative_to(root)}:{s.start}  {s.head} {s.name}")
    if not args.write:
        print("dry run: nothing written (use --write; commit first so you can review and revert)")
        return
    originals = {path: path.read_text(encoding="utf-8") for path in changes}
    for path, text in changes.items():
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(changes):,} files")
    if args.verify:
        ok = verify_and_repair(root, p, originals, args.build.split(), args.rounds)
        if not ok:
            for path, text in originals.items():
                path.write_text(text, encoding="utf-8")
            sys.exit("could not get a clean build; all files restored")
