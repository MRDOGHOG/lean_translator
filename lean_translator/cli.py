"""Command-line entry point.

    python -m lean_translator gloss  PATH...   [-o out.jsonl] [-j WORKERS]
    python -m lean_translator gloss  -          (read Lean from stdin, print text)
    python -m lean_translator pairs  PATH... -o pairs.jsonl   (training data for stage 2)
    python -m lean_translator refine IN.jsonl -o OUT.jsonl --model ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .decls import extract_decls
from .gloss import gloss_decl


def _iter_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.lean")))
        else:
            out.append(path)
    return out


def translate_source(src: str, file: str = "<input>") -> list[dict]:
    rows = []
    for d in extract_decls(src, file):
        try:
            rows.append(gloss_decl(d).to_dict())
        except Exception as e:  # never let one weird declaration kill a run
            rows.append({"name": d.full_name, "kind": d.kind, "file": file, "line": d.line,
                         "lean": d.signature, "gloss": None, "error": repr(e), "doc": d.doc})
    return rows


def _work(path: str, root: str | None) -> tuple[list[dict], int]:
    sys.setrecursionlimit(20000)
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    rel = os.path.relpath(path, root) if root else path
    return translate_source(src, rel), src.count("\n") + 1


def _chunks(files: list[Path], size: int):
    for i in range(0, len(files), size):
        yield files[i : i + size]


def _work_many(paths: list[str], root: str | None) -> tuple[list[dict], int]:
    rows: list[dict] = []
    lines = 0
    for p in paths:
        r, n = _work(p, root)
        rows.extend(r)
        lines += n
    return rows, lines


def run_gloss(args: argparse.Namespace) -> None:
    if args.paths == ["-"]:
        for row in translate_source(sys.stdin.read()):
            print(row["gloss"] if row.get("gloss") else f"[failed: {row.get('error')}]")
            print()
        return
    files = _iter_files(args.paths)
    root = os.path.commonpath([str(p.parent) for p in files]) if files else None
    t0 = time.perf_counter()
    total_lines = 0
    total_decls = 0
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    workers = args.jobs or os.cpu_count() or 1
    batches = [[str(p) for p in b] for b in _chunks(files, 32)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows, n in ex.map(_work_many, batches, [root] * len(batches)):
            total_lines += n
            total_decls += len(rows)
            for row in rows:
                if args.format == "text":
                    out.write(f"{row['file']}:{row['line']}  {row.get('gloss')}\n")
                else:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    dt = time.perf_counter() - t0
    if out is not sys.stdout:
        out.close()
    print(
        f"translated {total_decls:,} declarations from {len(files):,} files "
        f"({total_lines:,} lines) in {dt:.1f}s  ->  {total_lines / dt:,.0f} lines/s on {workers} workers",
        file=sys.stderr,
    )


_KIND = {"theorem": "Theorem", "lemma": "Lemma", "def": "Definition", "abbrev": "Definition",
         "structure": "Structure", "class": "Class", "inductive": "Type", "instance": "Instance",
         "example": "Example", "axiom": "Axiom", "opaque": "Definition"}


def _book_file(path: str, root: str | None) -> tuple[str, int, dict]:
    """One file as Markdown: every declaration's statement in English, then its proof steps."""
    from collections import Counter

    from .gloss import Glosser
    from .proofs import UNKNOWN, narrate

    sys.setrecursionlimit(20000)
    UNKNOWN.clear()
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    rel = os.path.relpath(path, root) if root else path
    stats: Counter = Counter()
    parts = [f"## {rel}\n"]
    for d in extract_decls(src, rel):
        try:
            tr = gloss_decl(d)
            statement = tr.gloss
            short = d.full_name.rsplit(".", 1)[-1] if d.full_name else ""
            if statement and short:  # the title already names it
                statement = re.sub(rf"^\w+ {re.escape(short)}: ", "", statement)
        except Exception:
            statement = None
        stats["decls"] += 1
        title = f"**{_KIND.get(d.kind, d.kind.title())} {d.full_name or '(unnamed)'}**"
        parts.append(f"{title} (line {d.line}). {statement or '`' + d.signature[:300] + '`'}\n")
        if statement:
            stats["statements"] += 1
        try:
            steps = narrate(d, Glosser())
        except Exception:
            steps = []
        if steps:
            stats["proofs"] += 1
            stats["steps"] += len(steps)
            parts.append("*Proof.*")
            for depth, s in steps:
                parts.append("  " * depth + "- " + s)
            parts.append("∎\n")
    stats.update({f"unknown:{k}": v for k, v in UNKNOWN.items()})
    return "\n".join(parts) + "\n", src.count("\n") + 1, dict(stats)


def _book_many(paths: list[str], root: str | None):
    return [_book_file(p, root) for p in paths]


def run_book(args: argparse.Namespace) -> None:
    from collections import Counter

    files = _iter_files(args.paths)
    if args.first:  # put these files first (e.g. the file with the main theorem)
        first = [f for f in files if any(str(f).endswith(x) for x in args.first)]
        files = first + [f for f in files if f not in first]
    root = os.path.commonpath([str(p.parent) for p in files]) if files else None
    t0 = time.perf_counter()
    total = Counter()
    lines = 0
    batches = [[str(p) for p in b] for b in _chunks(files, 16)]
    with ProcessPoolExecutor(max_workers=args.jobs or os.cpu_count() or 1) as ex, \
            open(args.output, "w", encoding="utf-8") as out:
        out.write(f"# {args.title}\n\nTranslated from Lean by rules (no AI): every statement, and every proof "
                  "step by step.\n\n")
        for res in ex.map(_book_many, batches, [root] * len(batches)):
            for md, n, st in res:
                out.write(md)
                lines += n
                total.update(st)
    dt = time.perf_counter() - t0
    unknown = Counter({k[8:]: v for k, v in total.items() if k.startswith("unknown:")})
    print(f"book: {len(files):,} files, {lines:,} lines, {total['decls']:,} declarations "
          f"({total['statements']:,} statements in English), {total['proofs']:,} proofs narrated in "
          f"{total['steps']:,} steps, in {dt:.1f}s", file=sys.stderr)
    if unknown:
        print("tactics without a template (shown as code): "
              + ", ".join(f"{k} {v}" for k, v in unknown.most_common(25)), file=sys.stderr)


def run_latex(args: argparse.Namespace) -> None:
    from .latex import write_book

    t0 = time.perf_counter()
    st = write_book(args.path, args.output, args.title, args.first, args.headline, args.max_files)
    print(f"latex: {st['files']:,} files, {st['declarations']:,} declarations, {st['proofs']:,} proofs "
          f"({st['steps']:,} steps) -> {args.output} in {time.perf_counter() - t0:.1f}s; "
          f"headline theorems found: {', '.join(st['headline_found']) or 'none'}", file=sys.stderr)


def run_pairs(args: argparse.Namespace) -> None:
    """Build (gloss -> docstring) training pairs for the neural refiner."""
    from .refiner.data import make_pairs

    files = _iter_files(args.paths)
    root = os.path.commonpath([str(p.parent) for p in files]) if files else None
    batches = [[str(p) for p in b] for b in _chunks(files, 32)]
    n = 0
    with ProcessPoolExecutor(max_workers=args.jobs or os.cpu_count()) as ex, open(args.output, "w", encoding="utf-8") as out:
        for rows, _ in ex.map(_work_many, batches, [root] * len(batches)):
            for pair in make_pairs(rows):
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n:,} training pairs to {args.output}", file=sys.stderr)


def run_refine(args: argparse.Namespace) -> None:
    from .refiner.translate import refine_file

    refine_file(args.input, args.output, args.model, batch_size=args.batch_size, device=args.device,
                check=args.check, min_score=args.min_score)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lean_translator", description="Lean 4 -> English translator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gloss", help="stage 1: rule-based Lean -> Lean-English")
    g.add_argument("paths", nargs="+", help=".lean files or directories, or - for stdin")
    g.add_argument("-o", "--output")
    g.add_argument("-j", "--jobs", type=int, default=0)
    g.add_argument("--format", choices=["jsonl", "text"], default="jsonl")
    g.set_defaults(func=run_gloss)

    b = sub.add_parser("book", help="whole project -> one English Markdown book (statements + proofs)")
    b.add_argument("paths", nargs="+")
    b.add_argument("-o", "--output", required=True)
    b.add_argument("-j", "--jobs", type=int, default=0)
    b.add_argument("--title", default="The proof in English")
    b.add_argument("--first", nargs="*", default=[], help="files to put at the start")
    b.set_defaults(func=run_book)

    lx = sub.add_parser("latex", help="whole project -> a LaTeX book with typeset math and figures")
    lx.add_argument("path", help="source directory")
    lx.add_argument("-o", "--output", required=True)
    lx.add_argument("--title", default="The proof in English")
    lx.add_argument("--first", nargs="*", default=[], help="files to start from (reading order follows imports)")
    lx.add_argument("--headline", nargs="*", default=[], help="main theorems, for the proof map figure")
    lx.add_argument("--max-files", type=int, default=0)
    lx.set_defaults(func=run_latex)

    p = sub.add_parser("pairs", help="build stage-2 training pairs (gloss -> docstring)")
    p.add_argument("paths", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-j", "--jobs", type=int, default=0)
    p.set_defaults(func=run_pairs)

    r = sub.add_parser("refine", help="stage 2: neural Lean-English -> English")
    r.add_argument("input", help="jsonl produced by `gloss`")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--model", required=True, help="refiner.pt (from train) or a directory (from train_hf)")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--check", help="reverse model for the round-trip check (outputs that fail fall back to the gloss)")
    r.add_argument("--min-score", type=float, default=0.6)
    r.add_argument("--device", default=None)
    r.set_defaults(func=run_refine)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
