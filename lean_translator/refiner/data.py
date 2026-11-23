"""Training data for the stage-2 refiner.

Every Mathlib declaration that has a docstring gives a free parallel pair:

    source  = stage-1 gloss  (literal "Lean-English", produced by rules)
    target  = the human-written docstring (fluent English)

So the refiner never has to learn Lean; it only learns to rewrite stilted,
but already-correct, English into the house style of mathematicians.
"""

from __future__ import annotations

import json
import random
import re
from typing import Iterable, Iterator

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"\s+")


def source_text(row: dict) -> str:
    """What the refiner sees: the gloss, prefixed with the name reading."""
    gloss = row.get("gloss") or ""
    name = row.get("name_gloss") or ""
    return f"{gloss} ⟦{name}⟧" if name else gloss


def clean_doc(doc: str) -> str | None:
    """Keep the first paragraph of a docstring, normalised."""
    para = doc.strip().split("\n\n", 1)[0]
    para = _MD_LINK.sub(r"\1", para)
    para = _WS.sub(" ", para).strip()
    if len(para) < 12 or len(para) > 400:
        return None
    if para.lower().startswith(("see ", "todo", "deprecated", "an auxiliary", "auxiliary", "implementation detail")):
        return None
    return para


def make_pairs(rows: Iterable[dict]) -> Iterator[dict]:
    for row in rows:
        doc = row.get("doc")
        if not doc or not row.get("gloss"):
            continue
        tgt = clean_doc(doc)
        if tgt is None:
            continue
        src = source_text(row)
        if len(src) > 1200:
            continue
        yield {"src": src, "tgt": tgt, "name": row.get("name"), "file": row.get("file")}


def load_pairs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_by_file(pairs: list[dict], valid_frac: float = 0.03, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out whole files so validation measures generalisation, not memory."""
    files = sorted({p["file"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(files)
    held = set(files[: max(1, int(len(files) * valid_frac))])
    train = [p for p in pairs if p["file"] not in held]
    valid = [p for p in pairs if p["file"] in held]
    return train, valid
