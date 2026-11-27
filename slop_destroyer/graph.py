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
    type_deps: list[str] = field(default_factory=list)


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
            if len(parts) < 7:
                parts += [""] * (7 - len(parts))
            name, mod, s, e, meta, deps, tdeps = parts[:7]
            g[name] = Const(name, mod, int(s or 0), int(e or 0), meta == "true", deps.split(), tdeps.split())
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


SRC_DIRS = [".", "src", "Src", "lib"]


def module_path(root: Path, module: str) -> Path:
    """Source file of a module, trying the usual lake `srcDir`s."""
    rel = module.replace(".", "/") + ".lean"
    for d in SRC_DIRS:
        p = (root / d / rel).resolve()
        if p.exists():
            return p
    return root / rel


def plan(root: Path, g: dict[str, Const], targets: list[str], keep_modules: list[str]) -> Plan:
    missing = [t for t in targets if t not in g]
    if missing:
        sys.exit(f"not found in the built project: {', '.join(missing)} (use full names, e.g. Foo.bar)")
    roots = list(targets)
    # Machinery that runs during checking but leaves no trace in proof terms.
    roots += [c.name for c in g.values() if c.meta or any(c.module.startswith(m) for m in keep_modules)]
    keep = reachable(g, roots)

    # The finished proof isn't the whole story: the *source* of a kept proof may
    # name things its final term doesn't use (`simp [helper]` where helper turned
    # out unnecessary).  Deleting those would leave "unknown identifier" errors,
    # so also keep whatever kept source names, and repeat until nothing changes.
    from .static import Resolver, scan_file

    files = sorted({module_path(root, c.module) for c in g.values()})
    items = [it for f in files if f.exists() for it in scan_file(str(f))]
    R = Resolver(items)
    by_name = {n: k for k, it in enumerate(items) for n in it.names}
    # Declarations usable without a trace even in the proof term (rfl simp lemmas
    # used by dsimp, instances, ext lemmas, ...): keep them whenever their
    # statement mentions something kept, like static mode does.
    invisible = [n for n, c in g.items() if n in by_name and items[by_name[n]].invisible]
    while True:
        woken = [n for n in invisible if n not in keep
                 and (not any(d in g for d in g[n].type_deps) or any(d in keep for d in g[n].type_deps))]
        if woken:
            keep |= reachable(g, woken)
            continue
        kept_items = {by_name[n] for n in keep if n in by_name}
        named: set[str] = set()
        for k in kept_items:
            it = items[k]
            for ident, dotted in it.refs:
                nows, _ = R.resolve(ident, dotted, it.ctx)
                for found in nows:
                    for i in found:
                        named.update(n for n in items[i].names if n in g)
        new = named - keep
        if not new:
            break
        keep |= reachable(g, list(new))

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
            if s.head in ("instance", "class"):
                # Instances can be needed while proofs are *checked* even when no finished
                # proof term mentions them (seen on a real 114k-line project): never delete.
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
    if rounds <= 0:  # auto: enough rounds to bisect the deletions
        rounds = 2 + sum(len(v) for v in p.deleted_spans.values()).bit_length()
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
        if not wanted:
            # The error doesn't say what's missing: put back half of the deletions and retry.
            spans = [(path, s) for path, dels in p.deleted_spans.items() for s in dels]
            half = set(id(s) for _, s in spans[: (len(spans) + 1) // 2])
            for path, dels in list(p.deleted_spans.items()):
                p.deleted_spans[path] = [s for s in dels if id(s) not in half]
            restored = len(half)
            print(f"[verify] build failed without naming a declaration; restoring {restored} of "
                  f"{len(spans)} deletions", file=sys.stderr)
            for path, text in originals.items():
                path.write_text(text, encoding="utf-8")
            for path, text in apply(p).items():
                path.write_text(text, encoding="utf-8")
            continue
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
    SRC_DIRS[:0] = list(getattr(args, "src_dir", []) or [])
    lean_cmd = args.lean.split()
    import tempfile

    dump = Path(args.graph) if args.graph else Path(tempfile.gettempdir()) / f"slop_destroyer_graph_{os.getpid()}.tsv"
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
            print("no deletion survived the rebuild; all files restored, rebuilding the original", file=sys.stderr)
            r = subprocess.run(args.build.split(), cwd=root, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit("the restored original does not build either:\n" + (r.stdout + r.stderr)[-3000:])
            print("result: nothing could be safely deleted (the project is already minimal at this level)")
