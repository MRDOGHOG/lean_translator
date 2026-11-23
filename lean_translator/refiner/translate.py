"""Run the stage-2 refiner over the JSONL produced by stage 1.

For masked models (trained on `[Tk]` slots) every output goes through:
  1. the slot check: each slot exactly once, none invented (exact, free);
  2. optionally the round-trip check with a reverse model (`--check`);
and only then are the slots filled back in.  Anything that fails falls back to
the stage-1 gloss, which is always correct, so the worst case is "robotic",
never "wrong".
"""

from __future__ import annotations

import json
import sys
import time

from ..gloss import slot_check, unmask
from .backends import load_translator
from .data import source_text
from .roundtrip import round_trip_scores


def refine_file(inp: str, out: str, model_path: str, batch_size: int = 64, device: str | None = None,
                check: str | None = None, min_score: float = 0.6) -> None:
    fwd = load_translator(model_path, device)
    masked = fwd.meta.get("masked", False)
    with open(inp, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("gloss")]
    srcs = [r["masked"] if masked else source_text(r) for r in rows]

    t0 = time.time()
    order = sorted(range(len(rows)), key=lambda i: len(srcs[i]))  # tight batches
    outs: list[str] = [""] * len(rows)
    for k in range(0, len(order), batch_size):
        idx = order[k : k + batch_size]
        for i, text in zip(idx, fwd.translate([srcs[i] for i in idx], batch_size)):
            outs[i] = text
        print(f"\r{min(k + batch_size, len(order)):,}/{len(order):,} declarations [{time.time() - t0:.0f}s]",
              end="", file=sys.stderr)
    print(file=sys.stderr)

    ok = [True] * len(rows)
    if masked:
        ok = [slot_check(o, len(r["slots"])) for o, r in zip(outs, rows)]
    scores: list[float | None] = [None] * len(rows)
    if check:
        rev = load_translator(check, device)
        idx = [i for i in range(len(rows)) if ok[i]]
        for i, s in zip(idx, round_trip_scores(rev, [srcs[i] for i in idx], [outs[i] for i in idx], batch_size)):
            scores[i] = s
            ok[i] = s >= min_score

    stats = {"refined": 0, "fallback": 0}
    with open(out, "w", encoding="utf-8") as f:
        for r, o, good, s in zip(rows, outs, ok, scores):
            if good:
                r["english"] = unmask(o, r["slots"]) if masked else o
                stats["refined"] += 1
            else:
                r["english"] = r["gloss"]  # stage 1 is always faithful
                stats["fallback"] += 1
            r["english_source"] = "refiner" if good else "gloss"
            r["refiner_raw"] = o
            if s is not None:
                r["rt_score"] = round(s, 4)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats), file=sys.stderr)
