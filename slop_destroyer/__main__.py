"""SLOP_DESTROYER: delete everything your theorem doesn't need, with the Lean
compiler as the judge (no AI anywhere).

Whole project, one pass (for millions of lines; needs the project built):
    SLOP_DESTROYER graph --root ~/myproj --module MyProj --keep MyProj.main_theorem
    SLOP_DESTROYER graph ... --write --verify

Whole project, no compiler at all (follows names through the source):
    SLOP_DESTROYER static --root ~/myproj --keep MyProj.main_theorem

One file, down to the tactic line (one compile per attempt):
    SLOP_DESTROYER probe Foo.lean --keep main_theorem
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

BANNER = "SLOP_DESTROYER"


def cmd_probe(args) -> None:
    from .probe import destroy

    path = Path(args.file).resolve()
    cwd = Path(args.root).resolve() if args.root else path.parent
    before = path.read_text(encoding="utf-8")
    text, log, oracle = destroy(path, args.keep, args.lean.split(), cwd, tactics=not args.no_tactics,
                                max_compiles=args.max_compiles, keep_debug=args.keep_debug)
    for line in log:
        print(f"  {line}")
    nb, na = len(before.rstrip().split("\n")), len(text.rstrip().split("\n"))
    print(f"{BANNER}: {len(log)} deletions, {nb} -> {na} lines "
          f"({oracle.calls} compiles, {oracle.seconds:.0f}s); statements and sorry-freedom unchanged")
    out = Path(args.output) if args.output else (path if args.in_place else path.with_suffix(".destroyed.lean"))
    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                                   str(path), str(out)))
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


def cmd_graph(args) -> None:
    from .graph import run

    run(args)


def cmd_static(args) -> None:
    from .static import run

    run(args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog=BANNER, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="one file, every deletion checked by the compiler")
    p.add_argument("file")
    p.add_argument("--keep", action="append", required=True, help="theorem to keep (repeatable)")
    p.add_argument("--lean", default="lean", help='how to run Lean, e.g. "lake env lean"')
    p.add_argument("--root", help="directory to run Lean in (project root for lake)")
    p.add_argument("--no-tactics", action="store_true", help="only delete whole commands")
    p.add_argument("--keep-debug", action="store_true", help="keep #check/#eval/example")
    p.add_argument("--max-compiles", type=int, default=500)
    p.add_argument("-o", "--output")
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--diff", action="store_true", help="print a unified diff")
    p.set_defaults(func=cmd_probe)

    g = sub.add_parser("graph", help="whole project in one pass, using Lean's recorded dependencies")
    g.add_argument("--root", required=True, help="project root (built with `lake build`)")
    g.add_argument("--module", action="append", required=True, help="module(s) to load, e.g. MyProj or Mathlib")
    g.add_argument("--prefix", action="append", help="only consider modules starting with this (default: --module roots)")
    g.add_argument("--keep", action="append", required=True, help="fully-qualified theorem to keep (repeatable)")
    g.add_argument("--keep-module", action="append", default=[], help="never delete from modules starting with this")
    g.add_argument("--lean", default="lake env lean")
    g.add_argument("--graph", help="reuse a dump from an earlier run")
    g.add_argument("--show", action="store_true", help="list every deletion")
    g.add_argument("--write", action="store_true", help="actually edit the files")
    g.add_argument("--verify", action="store_true", help="after writing, build once and repair if needed")
    g.add_argument("--build", default="lake build")
    g.add_argument("--rounds", type=int, default=3)
    g.set_defaults(func=cmd_graph)

    s = sub.add_parser("static", help="whole project, no compiler at all: follow names through the source")
    s.add_argument("--root", required=True, help="project directory")
    s.add_argument("--keep", action="append", required=True, help="theorem to keep (repeatable)")
    s.add_argument("--include", action="append", help="only these subdirectories/files of --root (e.g. Mathlib)")
    s.add_argument("--show", type=int, default=0, help="list the first N deletions")
    s.add_argument("--write", action="store_true", help="actually edit the files")
    s.add_argument("--report", help="directory for report.md, deletions.tsv and kept.tsv")
    s.add_argument("-j", "--jobs", type=int, default=0)
    s.set_defaults(func=cmd_static)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
