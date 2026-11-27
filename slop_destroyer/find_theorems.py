"""Find theorems whose statement is exactly a given proposition (e.g. RiemannHypothesis).

    python -m slop_destroyer.find_theorems ROOT RiemannHypothesis
prints the full names, one per line.  Only files mentioning the name are parsed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lean_translator.decls import extract_decls


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
    names = find(Path(sys.argv[1]), sys.argv[2])
    print("\n".join(names))
    sys.exit(0 if names else 1)
