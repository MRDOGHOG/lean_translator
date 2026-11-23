"""Fine-tune a *pretrained* seq2seq model (default: flan-t5-small, ~77M params)
as the refiner.  Starting from a model that already knows English is much
stronger than training from scratch on ~100k pairs.

    python -m lean_translator.refiner.train_hf distilled.jsonl -o refiner-t5/ --epochs 5
    python -m lean_translator.refiner.train_hf distilled.jsonl -o reverse-t5/ --reverse

Works on CPU, but meant for a GPU (a few hours on a consumer card, much less
on a datacenter one).  Needs `pip install transformers sentencepiece`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .backends import META_FILE, HFTranslator
from .data import load_pairs, oriented, split_by_file

N_SLOT_TOKENS = 64


def prepare_tokenizer(tok, texts: list[str]) -> int:
    """Add slot markers as whole tokens, plus any character the pretrained
    vocabulary can't represent (T5's vocabulary lacks e.g. `{`, `ℕ`, `α`)."""
    added = tok.add_tokens([f"[T{k}]" for k in range(N_SLOT_TOKENS)])
    unk = tok.unk_token_id
    chars = sorted({c for t in texts for c in t if not c.isspace()})
    missing = [c for c in chars if unk in tok(c, add_special_tokens=False).input_ids]
    added += tok.add_tokens(missing)
    return added


def make_batches(enc_src, enc_tgt, batch_size: int, shuffle: bool):
    order = sorted(range(len(enc_src)), key=lambda i: len(enc_src[i]))
    groups = [order[k : k + batch_size] for k in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(groups)
    return groups


def collate(tok, enc_src, enc_tgt, idx, device):
    pad = tok.pad_token_id
    L = max(len(enc_src[i]) for i in idx)
    T = max(len(enc_tgt[i]) for i in idx)
    src = torch.tensor([enc_src[i] + [pad] * (L - len(enc_src[i])) for i in idx], device=device)
    labels = torch.tensor([enc_tgt[i] + [-100] * (T - len(enc_tgt[i])) for i in idx], device=device)
    return src, (src != pad).long(), labels


@torch.no_grad()
def evaluate(model, tok, S, T, device, batch_size) -> float:
    model.eval()
    total, count = 0.0, 0
    for idx in make_batches(S, T, batch_size, shuffle=False):
        src, mask, labels = collate(tok, S, T, idx, device)
        n = (labels != -100).sum().item()
        total += model(input_ids=src, attention_mask=mask, labels=labels).loss.item() * n
        count += n
    model.train()
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", help="jsonl with src/tgt (from `distill collect` or `lean_translator pairs`)")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    ap.add_argument("--base", default="google/flan-t5-small", help="pretrained model name or local path")
    ap.add_argument("--reverse", action="store_true", help="train English -> gloss (for round-trip checks)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=0, help="optional wall-clock budget (0 = no limit)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (recent GPUs)")
    args = ap.parse_args(argv)
    random.seed(0)
    torch.manual_seed(0)

    pairs = load_pairs(args.pairs)
    meta = {"masked": "slots" in pairs[0], "direction": "reverse" if args.reverse else "forward",
            "base": args.base, "max_len": args.max_len}
    pairs = oriented(pairs, args.reverse)
    train, valid = split_by_file(pairs)
    print(f"{len(train):,} train / {len(valid):,} valid pairs")

    tok = AutoTokenizer.from_pretrained(args.base)
    n_added = prepare_tokenizer(tok, [p["src"] for p in train] + [p["tgt"] for p in train])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    model.resize_token_embeddings(len(tok))
    model.to(args.device).train()
    print(f"{args.base}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
          f"{n_added} tokens added, on {args.device}")

    def enc(texts):
        return tok(texts, truncation=True, max_length=args.max_len).input_ids

    S, T = enc([p["src"] for p in train]), enc([p["tgt"] for p in train])
    VS, VT = enc([p["src"] for p in valid]), enc([p["tgt"] for p in valid])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_total = args.epochs * math.ceil(len(S) / args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * max(0.05, 1 - s / max(steps_total, 1)))
    out = Path(args.output)
    best = math.inf
    step = 0
    t0 = time.time()
    out_of_time = False
    for epoch in range(args.epochs):
        for idx in make_batches(S, T, args.batch_size, shuffle=True):
            src, mask, labels = collate(tok, S, T, idx, args.device)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(input_ids=src, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps_total} loss {loss.item():.3f} [{time.time() - t0:.0f}s]", flush=True)
            if args.minutes and time.time() - t0 > args.minutes * 60:
                out_of_time = True
                break
        vl = evaluate(model, tok, VS, VT, args.device, args.batch_size)
        print(f"== epoch {epoch} valid loss {vl:.3f} (ppl {math.exp(vl):.1f})", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(out)
            tok.save_pretrained(out)
            (out / META_FILE).write_text(json.dumps(meta, indent=2))
            print(f"   saved {out}", flush=True)
        if out_of_time:
            break

    tr = HFTranslator(str(out), args.device)
    sample = valid[:8]
    for p, o in zip(sample, tr.translate([p["src"] for p in sample])):
        print("\nSRC :", p["src"][:300])
        print("REF :", p["tgt"])
        print("OUT :", o)


if __name__ == "__main__":
    main()
