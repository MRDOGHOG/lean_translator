"""Redundancy mode: theorems proved more than once.

Two theorems with the same statement (up to renaming bound variables and
universes) are interchangeable: one proof is enough.  Lean decides "same
statement" exactly; DumpDeps.lean writes a structural fingerprint of every
theorem's statement, including every Mathlib / core theorem in scope, so
this finds both

* duplicates inside the project, and
* project lemmas that re-prove something the libraries already have.

Each duplicate X of a kept theorem Y becomes `alias X := Y`: X keeps its
name (nothing that uses it changes) but loses its proof.  A rebuild restores
any file where that doesn't compile (e.g. Y isn't imported there).

    python -m slop_destroyer redundant --root ~/proj --module MyProj --keep MyProj.main --write --verify
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .graph import DUMP_SCRIPT, load_graph, module_path, reachable
from .spans import split_commands

# Attributes that generate other declarations from this one: aliasing would lose them.
GENERATING = {"to_additive", "to_dual", "simps", "reassoc", "elementwise", "ext", "mk_iff", "simps!"}
# Names Lean generates on demand (equation lemmas, auxiliary proofs): not real sources to alias to.
_AUTO = re.compile(r"\.(eq_\d+|eq_def|eq_unfold|_sunfold|_unfold|proof_\d+|match_\d+|_simp_\d+|_proof_\d+)$|\._")


def dump(root: Path, modules: list[str], prefixes: list[str], lean_cmd: list[str]) -> tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="slop_redundant_"))
    g, fp = tmp / "graph.tsv", tmp / "fp.tsv"
    cmd = lean_cmd + ["--run", str(DUMP_SCRIPT), str(g), "--fp", str(fp), *prefixes, "--", *modules]
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"dump failed:\n{r.stderr[-3000:]}")
    print(f"[redundant] {r.stderr.strip()}", file=sys.stderr)
    return g, fp


def groups(fp: Path, project_prefixes: list[str]) -> list[list[tuple[str, str]]]:
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in open(fp, encoding="utf-8"):
        h, name, mod = line.rstrip("\n").split("\t")
        by_hash[h].append((name, mod))
    in_project = lambda mod: any(mod == p or mod.startswith(p + ".") for p in project_prefixes)
    return [v for v in by_hash.values() if len(v) > 1 and any(in_project(m) for _, m in v)]


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
                out.add(t.text)
    return out


def _apply_form(path: Path, s, survivor: str):
    """`theorem X binders : stmt := @Y b1 b2 ...` (header kept; returns (start, end, text))."""
    lines = path.read_text(encoding="utf-8").split("\n")
    toks = s.tokens
    # find the top-level `:=` that starts the proof
    depth, assign = 0, None
    for t in toks:
        if t.text in "([{⟨⦃":
            depth += 1
        elif t.text in ")]}⟩⦄":
            depth -= 1
        elif depth == 0 and t.text == ":=":
            assign = t
            break
    if assign is None:
        return None
    # binder names in order; instance / anonymous binders become `_`
    args, depth, i = [], 0, 0
    head_i = next(k for k, t in enumerate(toks) if t.text == s.head)
    k = head_i + 2
    while k < len(toks) and toks[k] is not assign:
        t = toks[k]
        if depth == 0 and t.text == ":":
            break
        if t.text in ("(", "{", "⦃", "{{") and depth == 0:
            j, names = k + 1, []
            while j < len(toks) and toks[j].kind == "ident":
                names.append(toks[j].text)
                j += 1
            args += names if j < len(toks) and toks[j].text == ":" else []
        if t.text == "[" and depth == 0:
            args.append("_")
        if t.text in "([{⟨⦃":
            depth += 1
        elif t.text in ")]}⟩⦄":
            depth -= 1
        k += 1
    line = lines[assign.line - 1]
    new_last = line[: assign.col] + ":= @" + survivor + "".join(" " + a for a in args)
    # everything from the `:=` line to the end of the declaration collapses to that one line
    head_lines = lines[s.start - 1 : assign.line - 1]
    return (s.start, s.end, "\n".join(head_lines + [new_last]))


def run(args) -> None:
    root = Path(args.root).resolve()
    lean_cmd = args.lean.split()
    prefixes = args.prefix or [m.split(".")[0] for m in args.module]
    gpath, fp = dump(root, args.module, prefixes, lean_cmd)
    graph = load_graph(gpath)
    _closure: dict[str, set[str]] = {}

    def depends_on(a: str, b: str) -> bool:
        """Does a's statement or proof (transitively) use b?  Then `b := a` would be circular."""
        if a not in graph:
            return False  # library theorem: can't depend on project code
        if a not in _closure:
            _closure[a] = reachable(graph, [a])
        return b in _closure[a]
    in_project = lambda mod: any(mod == p or mod.startswith(p + ".") for p in prefixes)
    gs = groups(fp, prefixes)
    keep = set(args.keep or [])

    # For each group pick the survivor: a library theorem if there is one, else a project one.
    plan: dict[str, str] = {}  # duplicate full name -> survivor full name
    stats = Counter()
    for g in gs:
        g = [(n, m) for n, m in g if not _AUTO.search(n)]
        lib = sorted((n for n, m in g if not in_project(m)), key=len)
        proj = sorted((n, m) for n, m in g if in_project(m))
        if not proj or len(g) < 2:
            stats["auto-generated only"] += 1
            continue
        stats["library re-proofs" if lib else "project duplicates"] += len(proj) - (0 if lib else 1)
        # Survivors: library theorems first, then project theorems that aren't built on the others.
        candidates = lib + [n for n, _ in proj if not any(depends_on(n, o) for o, _ in proj if o != n)]
        for n, _ in proj:
            if n in keep or "_private" in n:
                continue
            survivor = next((c for c in candidates if c != n and not depends_on(c, n)), None)
            if survivor is None:
                stats["skipped (circular)"] += 1
                continue
            if survivor in plan:  # don't alias to something that is itself becoming an alias
                continue
            plan[n] = survivor
    print(f"duplicate statements: {len(gs):,} groups; "
          + ", ".join(f"{k}: {v:,}" for k, v in stats.items()))

    # Locate each duplicate's source and turn it into an alias.
    mod_of = {}
    for line in open(fp, encoding="utf-8"):
        _, name, mod = line.rstrip("\n").split("\t")
        if name in plan:
            mod_of[name] = mod
    edits: dict[Path, list[tuple[int, int, str, str]]] = defaultdict(list)  # file -> (start, end, new, desc)
    manifest = root / "lake-manifest.json"
    has_alias = manifest.exists() and re.search(r'"(batteries|mathlib)"', manifest.read_text(), re.I) is not None
    shown = 0
    for name, survivor in sorted(plan.items()):
        path = module_path(root, mod_of[name])
        if not path.exists():
            continue
        short = name.rsplit(".", 1)[-1]
        for s in split_commands(path.read_text(encoding="utf-8")):
            if s.kind != "decl" or s.head not in ("theorem", "lemma") or not s.name:
                continue
            if s.name != short and not name.endswith("." + s.name) and s.name != name:
                continue
            if _attrs(s.tokens) & GENERATING or any(t.text == "private" for t in s.tokens[:6]):
                break
            head = next(t for t in s.tokens if t.text == s.head and t.kind == "ident")
            if has_alias:
                edits[path].append((head.line, s.end, f"alias {s.name} := _root_.{survivor}", f"{name} = {survivor}"))
            else:
                # core Lean: keep the statement, prove it by applying the survivor to the binders
                body = _apply_form(path, s, survivor)
                if body is None:
                    break
                edits[path].append(body + (f"{name} = {survivor}",))
            if shown < args.show:
                print(f"  {path.relative_to(root)}:{head.line}  {name}  ==  {survivor}")
                shown += 1
            break

    originals: dict[Path, str] = {}
    saved = 0
    for path, es in edits.items():
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        for start, end, new, _ in sorted(es, reverse=True):
            if new.startswith("alias "):
                # keep anything before the keyword on its line (`protected `, attributes)
                first = lines[start - 1]
                kw = re.search(r"\b(theorem|lemma)\b", first)
                new = (first[: kw.start()] if kw else "") + new
            saved += (end - start + 1) - new.count("\n") - 1
            lines[start - 1 : end] = new.split("\n")
        originals[path] = text
        if args.write:
            path.write_text("\n".join(lines), encoding="utf-8")
    print(f"aliasable duplicates: {sum(len(v) for v in edits.values()):,} in {len(edits):,} files, "
          f"~{saved:,} lines of proof")
    if not args.write:
        print("dry run: nothing written (use --write)")
        return
    if args.verify:
        _verify(root, originals, args.build.split(), args.rounds)


_ERR_FILE = re.compile(r"(?:^|\s)error: ([^\s:][^:\n]*\.lean):\d+:\d+:|^([^\s:][^:\n]*\.lean):\d+:\d+: error", re.M)


def error_files(log: str) -> set[str]:
    return {a or b for a, b in _ERR_FILE.findall(log)}


def _verify(root: Path, originals: dict[Path, str], build: list[str], rounds: int) -> None:
    for _ in range(rounds):
        r = subprocess.run(build, cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[verify] build OK; {len(originals):,} changed files kept")
            return
        bad = {(root / m).resolve() for m in error_files(r.stdout + r.stderr)} & set(originals)
        if not bad:
            print("[verify] build failed outside the edited files; reverting everything", file=sys.stderr)
            print((r.stdout + r.stderr)[-3000:], file=sys.stderr)
            bad = set(originals)
        print(f"[verify] restoring {len(bad)} file(s) that no longer compile")
        errs = [l for l in (r.stdout + r.stderr).split("\n") if "error:" in l and ".lean:" in l]
        for l in errs[:8]:
            print("    " + l.strip()[:300])
        for f in bad:
            f.write_text(originals.pop(f), encoding="utf-8")
        if not originals:
            break
    r = subprocess.run(build, cwd=root, capture_output=True, text=True)
    if r.returncode != 0 and originals:
        # out of rounds: put the rest back so the project builds again
        print(f"[verify] no clean build after {rounds} rounds; restoring the other {len(originals)} file(s)")
        for f, text in originals.items():
            f.write_text(text, encoding="utf-8")
        originals.clear()
        r = subprocess.run(build, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("could not get a clean build")
    print(f"[verify] build OK; {len(originals):,} changed files kept")
