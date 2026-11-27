"""Lint-fix mode: slop *inside* proofs, found by Lean's own linters.

One compile of every file with the "unused" linters switched on reports, at
exact positions:

* simp arguments the simp call never used  -> the argument is removed
* `have h : ... := ...` whose h is never used -> the `have` block is removed
* tactics that do nothing (Mathlib's linter)  -> the tactic line is removed

The edits are applied all at once, then the project is rebuilt.  Any file
that no longer compiles gets its original back, and the build is repeated
until it's clean.  Several passes can be run: removing a `have` can make
other things unused.

    python -m slop_destroyer lintfix --root ~/proj --write --verify
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

LINT_OPTS = [
    "-Dweak.linter.unusedSimpArgs=true",
    "-Dweak.linter.unusedVariables=true",
    "-Dweak.linter.unusedVariables.analyzeTactics=true",
    "-Dweak.linter.unusedTactic=true",
]
_HEAD = re.compile(r"^(?P<file>[^\s:][^:]*\.lean):(?P<line>\d+):(?P<col>\d+): (?P<sev>warning|error)(?:\([^)]*\))?: (?P<msg>.*)$")


@dataclass
class Warning_:
    file: Path
    line: int  # 1-based
    col: int  # 0-based, in characters
    kind: str  # simp_arg | unused_have | unused_tactic | other
    msg: str


def classify(msg: str) -> str:
    m = msg.lower()
    if "simp argument is unused" in m:
        return "simp_arg"
    if "tactic does nothing" in m or "unused tactic" in m:
        return "unused_tactic"
    if "is not explicitly referenced" in m or m.startswith("unused variable"):
        return "unused_var"  # becomes unused_have if it sits on a `have` line
    return "other"


def lint_file(path: Path, root: Path, lean_cmd: list[str]) -> list[Warning_]:
    r = subprocess.run(lean_cmd + LINT_OPTS + [str(path)], cwd=root, capture_output=True, text=True)
    out: list[Warning_] = []
    for line in (r.stdout + r.stderr).split("\n"):
        m = _HEAD.match(line)
        if m and m.group("sev") == "warning":
            out.append(Warning_(path, int(m.group("line")), int(m.group("col")), classify(m.group("msg")),
                                m.group("msg")))
    return out


def _block_end(lines: list[str], ln: int) -> int:
    """Last line (1-based) of the tactic starting on line `ln`: it plus deeper-indented lines."""
    indent = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
    end = ln
    while end < len(lines):
        nxt = lines[end]
        if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
            break
        end += 1
    while end > ln and not lines[end - 1].strip():
        end -= 1
    return end


def _remove_simp_arg(line: str, col: int) -> str | None:
    """Remove the simp argument starting at `col`, plus one adjacent comma."""
    if col >= len(line):
        return None
    depth, end = 0, col
    while end < len(line):
        ch = line[end]
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            if depth == 0:
                break
            depth -= 1
        elif ch == "," and depth == 0:
            break
        end += 1
    start = col
    if end < len(line) and line[end] == ",":  # `a, b` -> drop "a, "
        end += 1
        while end < len(line) and line[end] == " ":
            end += 1
    else:  # last argument: drop the preceding ", "
        k = start
        while k > 0 and line[k - 1] == " ":
            k -= 1
        if k > 0 and line[k - 1] == ",":
            start = k - 1
    new = line[:start] + line[end:]
    # `simp [x]` with x removed -> `simp`
    new = re.sub(r"\s*\[\s*\]", "", new)
    return new


def plan_edits(text: str, warns: list[Warning_]) -> tuple[str, Counter]:
    lines = text.split("\n")
    done: Counter = Counter()
    drop: set[int] = set()
    # bottom-up / right-to-left so earlier positions stay valid
    for w in sorted(warns, key=lambda w: (w.line, w.col), reverse=True):
        if w.line < 1 or w.line > len(lines) or w.line in drop:
            continue
        line = lines[w.line - 1]
        if w.kind == "simp_arg":
            new = _remove_simp_arg(line, w.col)
            if new is not None and new != line:
                lines[w.line - 1] = new
                done["simp_arg"] += 1
        elif w.kind == "unused_var":
            # only `have name ...` / `have name : ...` lines that start the tactic
            if re.match(r"^\s*have\s+$", line[: w.col]):
                end = _block_end(lines, w.line)
                drop.update(range(w.line, end + 1))
                done["unused_have"] += 1
        elif w.kind == "unused_tactic":
            if not line[: w.col].strip():  # the tactic starts the line
                end = _block_end(lines, w.line)
                drop.update(range(w.line, end + 1))
                done["unused_tactic"] += 1
    kept = [ln for k, ln in enumerate(lines, 1) if k not in drop]
    return "\n".join(kept), done


_ERR_FILE = re.compile(r"(?:^|\s)error: ([^\s:][^:\n]*\.lean):\d+:\d+:|^([^\s:][^:\n]*\.lean):\d+:\d+: error", re.M)


def error_files(log: str) -> set[str]:
    return {a or b for a, b in _ERR_FILE.findall(log)}


def run(args) -> None:
    root = Path(args.root).resolve()
    lean_cmd = args.lean.split()
    files = sorted(p for p in root.rglob("*.lean")
                   if ".lake" not in p.parts and p.name not in ("lakefile.lean",) and not p.name.startswith(".slop_"))
    if args.include:
        files = [f for f in files if any(str(f.relative_to(root)).startswith(i) for i in args.include)]
    total = Counter()
    for pas in range(1, args.passes + 1):
        t0 = time.time()
        with ThreadPoolExecutor(args.jobs) as ex:
            per = list(ex.map(lambda f: lint_file(f, root, lean_cmd), files))
        warns = [w for ws in per for w in ws]
        kinds = Counter(w.kind for w in warns)
        print(f"[pass {pas}] linted {len(files):,} files in {time.time() - t0:.0f}s: "
              + ", ".join(f"{k} {v:,}" for k, v in kinds.most_common()), flush=True)
        by_file: dict[Path, list[Warning_]] = defaultdict(list)
        for w in warns:
            # Lean reports paths relative to cwd
            by_file[(root / w.file).resolve() if not w.file.is_absolute() else w.file].append(w)
        originals: dict[Path, str] = {}
        changed: dict[Path, str] = {}
        per_file: dict[Path, Counter] = {}
        for f, ws in by_file.items():
            if not f.exists():
                continue
            text = f.read_text(encoding="utf-8")
            new, d = plan_edits(text, ws)
            if new != text:
                originals[f] = text
                changed[f] = new
                per_file[f] = d
        done = sum(per_file.values(), Counter())
        print(f"[pass {pas}] fixes: " + (", ".join(f"{k} {v:,}" for k, v in done.items()) or "none"), flush=True)
        if not changed or not args.write:
            if not args.write:
                print("dry run: nothing written (use --write)")
            break
        for f, new in changed.items():
            f.write_text(new, encoding="utf-8")
        if args.verify:
            for rnd in range(args.rounds):
                r = subprocess.run(args.build.split(), cwd=root, capture_output=True, text=True)
                if r.returncode == 0:
                    print(f"[pass {pas}] build OK", flush=True)
                    break
                bad = {(root / m).resolve() for m in error_files(r.stdout + r.stderr)}
                bad &= set(originals)
                if not bad:
                    print(f"[pass {pas}] build failed and the errors name no edited file; reverting this pass",
                          file=sys.stderr)
                    print((r.stdout + r.stderr)[-3000:], file=sys.stderr)
                    for f, text in originals.items():
                        f.write_text(text, encoding="utf-8")
                    originals.clear()
                    r = subprocess.run(args.build.split(), cwd=root, capture_output=True, text=True)  # rebuild originals
                    break
                print(f"[pass {pas}] build failed in {len(bad)} edited file(s); restoring them", flush=True)
                errs = [l for l in (r.stdout + r.stderr).split("\n") if "error:" in l and ".lean:" in l]
                for l in errs[:6]:
                    print("    " + l.strip()[:300], flush=True)
                for f in bad:
                    f.write_text(originals.pop(f), encoding="utf-8")
            else:
                r = subprocess.run(args.build.split(), cwd=root, capture_output=True, text=True)
            if r.returncode != 0:
                # Out of rounds: never leave a broken project behind.
                print(f"[pass {pas}] no clean build after {args.rounds} rounds; reverting the rest of this pass",
                      flush=True)
                for f, text in originals.items():
                    f.write_text(text, encoding="utf-8")
                originals.clear()
                r = subprocess.run(args.build.split(), cwd=root, capture_output=True, text=True)
                if r.returncode != 0:
                    sys.exit("the restored original does not build:\n" + (r.stdout + r.stderr)[-3000:])
            done = sum((per_file[f] for f in originals), Counter())  # fixes that survived the build
            print(f"[pass {pas}] kept fixes: " + (", ".join(f"{k} {v:,}" for k, v in done.items()) or "none"))
        total += done
    print("total fixes: " + (", ".join(f"{k} {v:,}" for k, v in total.items()) or "none"))
