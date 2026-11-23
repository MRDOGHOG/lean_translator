"""Round-trip (cycle-consistency) checking.

    gloss ──forward──► English ──reverse──► gloss'      score = similarity(gloss, gloss')

If the English dropped a hypothesis, invented a condition, or swapped which
slot goes where, the reverse model can't reconstruct the original gloss, and
the score falls.  Used two ways:

1. Filtering training data (e.g. LLM-distilled pairs): keep pairs whose
   English round-trips.

       python -m lean_translator.refiner.roundtrip filter distilled.jsonl \\
           --reverse reverse-t5/ --min 0.6 -o distilled.clean.jsonl

2. At translation time (`lean_translator refine --check reverse-t5/`):
   outputs that don't round-trip fall back to the stage-1 gloss.

Limitation: it catches *loss* and *invention*, but if both directions share
the same misreading they can agree on something wrong.  That's why the
exact slot check (`gloss.slot_check`) is always applied too.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics

from ..gloss import _SLOT_RE
from .backends import Translator, load_translator

_WORD = re.compile(r"\[T\d+\]|[^\W_]+|[^\w\s]", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def similarity(original: str, reconstructed: str) -> float:
    """Token-level F1 between two glosses, zeroed if the slot sets differ.

    Glosses come from a fixed rule set, so a faithful round trip should
    reproduce most of the wording, and must reproduce the same slots."""
    if sorted(_SLOT_RE.findall(original)) != sorted(_SLOT_RE.findall(reconstructed)):
        return 0.0
    a, b = _tokens(original), _tokens(reconstructed)
    if not a or not b:
        return float(a == b)
    common: dict[str, int] = {}
    for t in a:
        common[t] = common.get(t, 0) + 1
    hit = 0
    for t in b:
        if common.get(t, 0) > 0:
            common[t] -= 1
            hit += 1
    if hit == 0:
        return 0.0
    p, r = hit / len(b), hit / len(a)
    return 2 * p * r / (p + r)


def round_trip_scores(reverse: Translator, glosses: list[str], english: list[str], batch_size: int = 64) -> list[float]:
    back = reverse.translate(english, batch_size)
    return [similarity(g, b) for g, b in zip(glosses, back)]


def cmd_filter(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    rev = load_translator(args.reverse, args.device)
    if rev.meta.get("direction") != "reverse":
        print(f"warning: {args.reverse} was not trained with --reverse")
    scores = round_trip_scores(rev, [r["src"] for r in rows], [r["tgt"] for r in rows], args.batch_size)
    kept = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            r["rt_score"] = round(s, 4)
            if s >= args.min:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    qs = statistics.quantiles(scores, n=4) if len(scores) > 1 else scores
    print(f"round-trip scores: median {statistics.median(scores):.3f}, quartiles {[round(q, 3) for q in qs]}")
    print(f"kept {kept:,} / {len(rows):,} pairs with score >= {args.min} -> {args.output}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="roundtrip")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter", help="score src/tgt pairs by round trip; keep the good ones")
    f.add_argument("input")
    f.add_argument("--reverse", required=True, help="model trained with --reverse")
    f.add_argument("--min", type=float, default=0.6)
    f.add_argument("-o", "--output", required=True)
    f.add_argument("--batch-size", type=int, default=64)
    f.add_argument("--device", default=None)
    f.set_defaults(func=cmd_filter)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
