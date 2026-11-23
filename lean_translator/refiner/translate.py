"""Run the stage-2 refiner over the JSONL produced by stage 1."""

from __future__ import annotations

import json
import sys
import time

import torch
from tokenizers import Tokenizer

from .data import source_text
from .model import Seq2Seq
from .tokenizer import decode, encode
from .train import _pad


def refine_file(inp: str, out: str, model_path: str, batch_size: int = 256, device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok_json = Seq2Seq.load(model_path, device)
    tok = Tokenizer.from_str(tok_json)
    with open(inp, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("gloss")]
    # Sort by length so batches are tight; restore order at the end.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["gloss"]))
    english: list[str | None] = [None] * len(rows)
    t0 = time.time()
    for k in range(0, len(order), batch_size):
        idx = order[k : k + batch_size]
        src = _pad(encode(tok, [source_text(rows[i]) for i in idx], model.cfg.max_len)).to(device)
        ids = model.greedy(src, max_len=96)
        for i, text in zip(idx, decode(tok, ids.tolist())):
            english[i] = text
        done = min(k + batch_size, len(order))
        print(f"\r{done:,}/{len(order):,} declarations  [{time.time() - t0:.0f}s]", end="", file=sys.stderr)
    print(file=sys.stderr)
    with open(out, "w", encoding="utf-8") as f:
        for r, e in zip(rows, english):
            r["english"] = e
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
