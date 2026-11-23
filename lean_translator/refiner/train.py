"""Train the stage-2 refiner.

    python -m lean_translator.refiner.train pairs.jsonl -o refiner.pt --minutes 30
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
from torch import nn

from .data import load_pairs, split_by_file
from .model import PAD, Config, Seq2Seq
from .tokenizer import decode, encode, train_tokenizer


def batches(src: list[list[int]], tgt: list[list[int]], max_tokens: int, shuffle: bool = True):
    """Length-bucketed batches with a token budget (standard NMT practice)."""
    order = sorted(range(len(src)), key=lambda i: (len(src[i]), len(tgt[i])))
    groups, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, len(src[i]), len(tgt[i]))
        if cur and m * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur, m = [], max(len(src[i]), len(tgt[i]))
        cur.append(i)
        cur_max = m
    if cur:
        groups.append(cur)
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        yield _pad([src[i] for i in g]), _pad([tgt[i] for i in g])


def _pad(seqs: list[list[int]]) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs], dtype=torch.long)


def evaluate(model: Seq2Seq, src, tgt, device, max_tokens: int) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    total, count = 0.0, 0
    with torch.no_grad():
        for s, t in batches(src, tgt, max_tokens, shuffle=False):
            s, t = s.to(device), t.to(device)
            logits = model(s, t[:, :-1])
            total += loss_fn(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1)).item()
            count += (t[:, 1:] != PAD).sum().item()
    model.train()
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("-o", "--output", default="refiner.pt")
    ap.add_argument("--minutes", type=float, default=30.0, help="wall-clock training budget")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    random.seed(0)
    torch.manual_seed(0)

    pairs = load_pairs(args.pairs)
    train, valid = split_by_file(pairs)
    print(f"{len(train):,} train / {len(valid):,} valid pairs")
    tok = train_tokenizer(
        [p["src"] for p in train] + [p["tgt"] for p in train], args.vocab
    )
    S = encode(tok, [p["src"] for p in train], args.max_len)
    T = encode(tok, [p["tgt"] for p in train], args.max_len)
    VS = encode(tok, [p["src"] for p in valid], args.max_len)
    VT = encode(tok, [p["tgt"] for p in valid], args.max_len)

    cfg = Config(vocab_size=tok.get_vocab_size(), d_model=args.d_model, n_heads=max(4, args.d_model // 64),
                 enc_layers=args.enc_layers, dec_layers=args.dec_layers, ff=4 * args.d_model, max_len=args.max_len)
    model = Seq2Seq(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M parameters on {args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    warmup = 500
    step = 0
    best = math.inf
    t0 = time.time()
    budget = args.minutes * 60
    for epoch in range(args.epochs):
        for s, t in batches(S, T, args.max_tokens):
            s, t = s.to(args.device), t.to(args.device)
            lr = args.lr * min(1.0, (step + 1) / warmup) * min(1.0, math.sqrt(warmup / (step + 1)))
            for g in opt.param_groups:
                g["lr"] = lr
            logits = model(s, t[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.3f} lr {lr:.2e} [{time.time() - t0:.0f}s]", flush=True)
            if time.time() - t0 > budget:
                break
        vl = evaluate(model, VS, VT, args.device, args.max_tokens)
        print(f"== epoch {epoch} valid loss {vl:.3f} (ppl {math.exp(vl):.1f})", flush=True)
        if vl < best:
            best = vl
            model.save(args.output, tok.to_str())
            print(f"   saved {args.output}", flush=True)
        if time.time() - t0 > budget:
            break

    # Show a few held-out translations.
    model, _ = Seq2Seq.load(args.output, args.device)
    sample = valid[:8]
    ids = model.greedy(_pad(encode(tok, [p["src"] for p in sample], args.max_len)).to(args.device))
    for p, out in zip(sample, decode(tok, ids.tolist())):
        print("\nSRC :", p["src"][:300])
        print("REF :", p["tgt"])
        print("OUT :", out)


if __name__ == "__main__":
    main()
