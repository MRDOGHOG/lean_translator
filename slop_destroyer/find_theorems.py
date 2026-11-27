"""Find theorems whose statement is exactly a given proposition (e.g. RiemannHypothesis).

    python -m slop_destroyer.find_theorems ROOT RiemannHypothesis
prints the full names, one per line.  Only files mentioning the name are parsed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lean_translator.decls import extract_decls


def find_with_locations(root: Path, prop: str) -> list[tuple[str, str, int, str]]:
    """(full name, file, line, source text of the declaration) for each match."""
    r = subprocess.run(["grep", "-rlF", "--include=*.lean", "--exclude-dir=.lake", prop, str(root)],
                       capture_output=True, text=True)
    out = []
    for path in r.stdout.split():
        src = Path(path).read_text(encoding="utf-8", errors="replace")
        lines = src.split("\n")
        for d in extract_decls(src, path):
            stmt = " ".join(t.text for t in d.type)
            if d.kind in ("theorem", "lemma") and stmt in (prop, f"_root_.{prop}"):
                # the declaration runs until the next line that starts in column 0
                end = d.line
                while end < len(lines) and (end == d.line or not lines[end] or lines[end][0] in " \t"):
                    end += 1
                out.append((d.full_name, str(Path(path).relative_to(root)), d.line,
                            "\n".join(lines[d.line - 1 : end]).rstrip()))
    return out


def find(root: Path, prop: str) -> list[str]:
    r = subprocess.run(["grep", "-rlF", "--include=*.lean", "--exclude-dir=.lake", prop, str(root)],
                       capture_output=True, text=True)
    out = []
    for path in r.stdout.split():
        src = Path(path).read_text(encoding="utf-8", errors="replace")
        for d in extract_decls(src, path):
            stmt = " ".join(t.text for t in d.type)
            if d.kind in ("theorem", "lemma") and stmt in (prop, f"_root_.{prop}"):
                out.append(d.full_name)
    return sorted(set(out))


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "--where":
        base = sys.argv[4] if len(sys.argv) > 4 else ""
        found = find_with_locations(Path(sys.argv[1]), sys.argv[2])
        for name, file, line, text in found:
            print(f"=== {name}  at  {file}:{line}")
            if base:
                print(f"    {base}/{file}#L{line}")
            print(text[:6000])
            print()
        sys.exit(0 if found else 1)
    names = find(Path(sys.argv[1]), sys.argv[2])
    print("\n".join(names))
    sys.exit(0 if names else 1)
