"""Speed mode: find what's slow, and make the standard fix.

1. Profile: compile every file once with Lean's profiler on and add up the
   time per declaration and per tactic.  The report lists the slowest.
2. `--simp-only`: in the slowest declarations, every bare `simp` / `simp_all`
   / `simpa` / `dsimp` (which searches the whole simp library every time) is
   replaced by the exact `simp only [...]` call Lean itself suggests via
   `simp?`.  That's the usual Mathlib advice: same proof, a fraction of the
   search, and it doesn't break when the simp library changes.
3. Each changed file is recompiled; files that fail or don't get faster
   are restored, and a final build confirms the project.

    python -m slop_destroyer speed --root ~/proj --top 50 --simp-only --write --verify
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .spans import split_commands

_TOOK = re.compile(r"^(?P<what>.+?) took (?P<num>[\d.]+)(?P<unit>ms|s)$")
_BARE_SIMP = re.compile(r"(?<![\w.?'])(simp_all|simpa|simp|dsimp)(?![\w?'!])(?!\s+only\b)")


def _ms(num: str, unit: str) -> float:
    return float(num) * (1000.0 if unit == "s" else 1.0)


def lean_json(path: Path, root: Path, lean_cmd: list[str], extra: list[str]) -> tuple[int, list[dict]]:
    r = subprocess.run(lean_cmd + ["--json", *extra, str(path)], cwd=root, capture_output=True, text=True)
    msgs = []
    for line in r.stdout.split("\n"):
        if line.startswith("{"):
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return r.returncode, msgs


def profile(path: Path, root: Path, lean_cmd: list[str], threshold: int) -> dict[int, tuple[float, dict[str, float]]]:
    """line of declaration -> (total ms, {what: ms})"""
    _, msgs = lean_json(path, root, lean_cmd, ["-Dprofiler=true", f"-Dprofiler.threshold={threshold}"])
    out: dict[int, list] = {}
    for m in msgs:
        if m.get("severity") != "information" or not m.get("pos"):
            continue
        for part in m.get("data", "").strip().split("\n"):
            t = _TOOK.match(part.strip())
            if not t:
                continue
            ln = m["pos"]["line"]
            tot, by = out.setdefault(ln, [0.0, defaultdict(float)])
            ms = _ms(t.group("num"), t.group("unit"))
            out[ln][0] = tot + ms
            by[t.group("what")] += ms
    return {ln: (v[0], dict(v[1])) for ln, v in out.items()}


def decl_times(path: Path, prof: dict[int, tuple[float, dict]]) -> list[tuple[float, int, int, str, dict]]:
    spans = [s for s in split_commands(path.read_text(encoding="utf-8")) if s.kind == "decl"]
    res = []
    for s in spans:
        tot, by = 0.0, defaultdict(float)
        for ln, (t, b) in prof.items():
            if s.start <= ln <= s.end:
                tot += t
                for k, v in b.items():
                    by[k] += v
        if tot:
            res.append((tot, s.start, s.end, f"{s.head} {s.name}".strip(), dict(by)))
    return res


def simp_only(path: Path, root: Path, lean_cmd: list[str], ranges: list[tuple[int, int]]) -> tuple[str, int] | None:
    """Rewrite bare simp calls inside `ranges` (line spans) to `simp only [...]` as suggested by `simp?`."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    calls = []  # (line, start col, end col of keyword)
    for a, b in ranges:
        for ln in range(a, b + 1):
            code = lines[ln - 1].split("--", 1)[0].rstrip()
            found = list(_BARE_SIMP.finditer(code))
            # only lines that are exactly one simp call (the suggestion replaces the whole call,
            # arguments and `at`/`using` included, so nothing else may share the line)
            if len(found) != 1:
                continue
            m = found[0]
            if code[: m.start()].strip() not in ("", "·") or ";" in code[m.end():] or "<;>" in code:
                continue
            calls.append((ln, m.start(), m.end(), len(code)))
    if not calls:
        return None
    probe = [list(l) for l in lines]
    for ln, s, e, _ in sorted(calls, reverse=True):
        probe[ln - 1].insert(e, "?")
    tmp = path.with_name(f".slop_speed_{path.name}")
    tmp.write_text("\n".join("".join(l) for l in probe), encoding="utf-8")
    try:
        code, msgs = lean_json(tmp, root, lean_cmd, [])
    finally:
        tmp.unlink()
    if code != 0:
        return None
    suggestions: dict[tuple[int, int], tuple[int, int, str]] = {}
    for m in msgs:
        data = m.get("data", "")
        if "Try this" in data and "[apply]" in data and m.get("pos") and m.get("endPos"):
            sug = " ".join(data.split("[apply]", 1)[1].split())
            p, e = m["pos"], m["endPos"]
            suggestions[(p["line"], p["column"])] = (e["line"], e["column"], sug)
    new_lines = list(lines)
    done = 0
    for ln, s, e, call_end in sorted(calls, reverse=True):
        hit = suggestions.get((ln, s))
        if not hit or not hit[2].startswith(("simp", "dsimp")) or "✝" in hit[2] or "↓" in hit[2]:
            continue  # `✝` names are inaccessible and `↓` doesn't parse back: can't write them into source
        sug = hit[2]
        # Lean's range for the suggestion covers only the `simp?` keyword, so find the end of the
        # whole call (its arguments may continue on later lines) ourselves.
        end_line, end_col = call_extent(lines, ln, s)
        if apply_suggestion(new_lines, ln, s, end_line, end_col, sug):
            done += 1
    return ("\n".join(new_lines), done) if done else None


def call_extent(lines: list[str], ln: int, col: int) -> tuple[int, int]:
    """End (line, column) of the tactic call starting at (ln, col): the rest of its line, plus
    following lines while brackets are open; a trailing `-- comment` is not part of the call."""
    depth = 0
    cur = ln
    text = lines[ln - 1][col:]
    start = col
    while True:
        code = text.split("--", 1)[0]
        for ch in code:
            if ch in "([{⟨":
                depth += 1
            elif ch in ")]}⟩":
                depth -= 1
        end = start + len(code.rstrip())
        if depth <= 0 or cur >= len(lines):
            return cur, end
        cur += 1
        text, start = lines[cur - 1], 0


def apply_suggestion(lines: list[str], ln: int, col: int, end_line: int, end_col: int, sug: str) -> bool:
    """Replace the source range (ln, col)..(end_line, end_col) (1-based lines, 0-based columns)
    with `sug`.  The range is the whole tactic call as Lean parsed it, so arguments that continue
    on later lines are replaced too, not left behind."""
    if end_line < ln or end_line > len(lines) or (end_line == ln and end_col < col):
        return False
    tail = lines[end_line - 1][end_col:]
    lines[ln - 1 : end_line] = [lines[ln - 1][:col] + sug + tail]
    return True


def run(args) -> None:
    root = Path(args.root).resolve()
    lean_cmd = args.lean.split()
    files = sorted(p for p in root.rglob("*.lean")
                   if ".lake" not in p.parts and p.name != "lakefile.lean" and not p.name.startswith(".slop_"))
    if args.include:
        files = [f for f in files if any(str(f.relative_to(root)).startswith(i) for i in args.include)]
    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        profs = list(ex.map(lambda f: profile(f, root, lean_cmd, args.threshold), files))
    rows = []
    tactic_total: dict[str, float] = defaultdict(float)
    for f, prof in zip(files, profs):
        for tot, a, b, desc, by in decl_times(f, prof):
            rows.append((tot, f, a, b, desc, by))
            for k, v in by.items():
                tactic_total[k] += v
    rows.sort(key=lambda r: -r[0])
    total = sum(r[0] for r in rows)
    print(f"profiled {len(files):,} files in {time.time() - t0:.0f}s; {total / 1000:,.0f}s of measured work")
    print("where the time goes:")
    for k, v in sorted(tactic_total.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v / 1000:9.1f}s  {k}")
    print(f"slowest {min(args.top, len(rows))} declarations:")
    for tot, f, a, b, desc, by in rows[: args.top]:
        top = max(by.items(), key=lambda kv: kv[1])[0] if by else ""
        print(f"  {tot / 1000:8.2f}s  {f.relative_to(root)}:{a}  {desc}   (mostly: {top})")
    if not args.simp_only:
        return
    targets: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for tot, f, a, b, desc, by in rows[: args.top]:
        targets[f].append((a, b))
    originals: dict[Path, str] = {}
    n = 0
    for f, ranges in targets.items():
        res = simp_only(f, root, lean_cmd, ranges)
        if res is None:
            continue
        new, done = res
        originals[f] = f.read_text(encoding="utf-8")
        if args.write:
            f.write_text(new, encoding="utf-8")
        n += done
        print(f"  {f.relative_to(root)}: {done} simp call(s) -> simp only")
    print(f"simp -> simp only: {n} call(s) in {len(originals)} file(s)")
    if not args.write:
        print("dry run: nothing written (use --write)")
        return
    if args.verify:
        from .redundant import _verify

        _verify(root, originals, args.build.split(), args.rounds)
        # report the speed-up on the files that kept their changes
        after = 0.0
        before = 0.0
        for f in originals:
            before += sum(t for t, *_ in decl_times(f, profs[files.index(f)])) if f in files else 0
        with ThreadPoolExecutor(args.jobs) as ex:
            new_profs = list(ex.map(lambda f: profile(f, root, lean_cmd, args.threshold), list(originals)))
        for f, prof in zip(originals, new_profs):
            after += sum(t for t, *_ in decl_times(f, prof))
        if before:
            print(f"measured time in changed files: {before / 1000:.1f}s -> {after / 1000:.1f}s")
