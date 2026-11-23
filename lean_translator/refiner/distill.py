"""Make clean training targets with a large model, once, offline.

The refiner is only as good as its targets, and Mathlib docstrings are loose
(they often don't say what the statement says).  Here Claude writes a
faithful English sentence for each *masked* gloss.  It sees the math in the
slots but must write [T0], [T1], ... back verbatim, so the targets have the
same slots as the sources, and the small model learns prose only.

The big model is used only to create data; the shipped translator stays tiny.

    # 1. preview prompts + cost estimate (no API calls)
    python -m lean_translator.refiner.distill prepare mathlib.jsonl -n 1000 -o pilot/

    # 2. submit to the Message Batches API (50% price; usually done within an hour)
    python -m lean_translator.refiner.distill submit pilot/

    # 3. download, validate slots, write training pairs
    python -m lean_translator.refiner.distill collect pilot/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from ..gloss import slot_check

DEFAULT_MODEL = "claude-opus-5-5"

# $ per million tokens: (input, output, cache read). Batch API halves all of them.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}

SYSTEM = """You turn formal Lean 4 statements into one or two sentences of plain mathematical English, in the style of a good textbook or a Mathlib docstring.

You are given:
- LEAN: the original Lean source of the statement.
- GLOSS: a literal, machine-made English reading of it, in which compound math terms are replaced by slot markers [T0], [T1], ...
- SLOTS: what each slot marker stands for.

Rules:
1. Be exactly faithful to the statement. Do not add, drop, strengthen or weaken any hypothesis or conclusion. If GLOSS and LEAN disagree, follow LEAN.
2. Write every slot marker from GLOSS exactly once, verbatim (for example [T0]), and no others. Never write out a slot's contents; write the marker. You may reorder the slots and the words around them.
3. Keep variable names as they are. Leave out type-class boilerplate that a mathematician would leave implicit ("Let G be a type"), but keep structure that matters ("in a commutative ring R").
4. Output only the English, inside <english></english> tags.

Examples:

LEAN: theorem mul_left_comm (a b c : G) : a * (b * c) = b * (a * c)
GLOSS: Theorem mul_left_comm: Let G be a commutative semigroup. Let a, b and c be elements of G. Then [T0] equals [T1].
SLOTS: [T0] = a * (b * c); [T1] = b * (a * c)
<english>In a commutative semigroup, [T0] = [T1] for all a, b and c.</english>

LEAN: theorem Nat.exists_dvd_of_not_prime {n : ℕ} (n2 : 2 ≤ n) (np : ¬Prime n) : ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n
GLOSS: Theorem Nat.exists_dvd_of_not_prime: Let n be a natural number. Assume that 2 is at most n and that n is not prime. Then there exists m such that m divides n and m does not equal 1 and m does not equal n.
SLOTS: (none)
<english>If a natural number n ≥ 2 is not prime, then it has a divisor m other than 1 and n.</english>

LEAN: theorem Finset.sum_range_id_mul_two (n : ℕ) : (∑ i ∈ range n, i) * 2 = n * (n - 1)
GLOSS: Theorem Finset.sum_range_id_mul_two: Let n be a natural number. Then [T0] equals [T1].
SLOTS: [T0] = (the sum of i over i ∈ {0, …, n - 1}) * 2; [T1] = n * (n - 1)
<english>Twice the sum of the first n natural numbers satisfies [T0] = [T1].</english>

LEAN: def RiemannHypothesis : Prop := ∀ (s : ℂ) (_ : riemannZeta s = 0) (_ : ¬∃ n : ℕ, s = -2 * (n + 1)) (_ : s ≠ 1), s.re = 1 / 2
GLOSS: Definition RiemannHypothesis: RiemannHypothesis holds when for every complex number s, if [T0] equals 0, there is no natural number n such that s equals [T1] and s does not equal 1, then [T2] equals [T3].
SLOTS: [T0] = ζ(s); [T1] = -2 * (n + 1); [T2] = the real part of s; [T3] = 1 / 2
<english>The Riemann hypothesis: every zero s of the Riemann zeta function, meaning [T0] = 0, that is not a trivial zero (s is not of the form [T1] for a natural number n) and is not s = 1, has [T2] equal to [T3].</english>"""

KINDS = {"theorem", "lemma", "def", "abbrev", "instance", "structure", "class"}
_ENGLISH = re.compile(r"<english>(.*?)</english>", re.S)


def user_prompt(row: dict) -> str:
    slots = "; ".join(f"[T{k}] = {v}" for k, v in enumerate(row["slots"])) or "(none)"
    return f"LEAN: {row['lean']}\nGLOSS: {row['masked']}\nSLOTS: {slots}"


def _request_params(row: dict, model: str) -> dict:
    return {
        "model": model,
        # Thinking can't be turned off on current models and counts toward
        # max_tokens; low effort keeps it short for a task this simple.
        "max_tokens": 2048,
        "output_config": {"effort": "low"},
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_prompt(row)}],
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("gloss") and r["kind"] in KINDS and len(r["masked"]) < 1500 and len(r["lean"]) < 1500:
                rows.append(r)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            keep = {x: r[x] for x in ("name", "kind", "file", "lean", "gloss", "masked", "slots", "name_gloss")}
            f.write(json.dumps({"id": f"d{k}", "model": args.model, **keep}, ensure_ascii=False) + "\n")

    # Rough cost estimate (Lean is token-dense: ~3 chars/token).
    pin, pout, pcache = PRICES.get(args.model, PRICES[DEFAULT_MODEL])
    sys_tok = len(SYSTEM) / 3
    in_tok = sum(len(user_prompt(r)) / 3 for r in rows)
    out_tok = len(rows) * args.est_output_tokens
    cost = 0.5 * (in_tok * pin + len(rows) * sys_tok * pcache + out_tok * pout) / 1e6
    print(f"{len(rows):,} items -> {out / 'items.jsonl'}")
    print(f"model {args.model}: ~{in_tok / 1e6:.2f}M input + ~{out_tok / 1e6:.2f}M output tokens "
          f"(assuming {args.est_output_tokens} output tokens/item incl. thinking)")
    print(f"estimated batch cost: ~${cost:,.2f}  (rough; the pilot's `collect` prints the real usage)")
    print("\nfirst prompt:\n" + user_prompt(rows[0]) if rows else "")


def cmd_submit(args: argparse.Namespace) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    d = Path(args.dir)
    items = [json.loads(line) for line in open(d / "items.jsonl", encoding="utf-8")]
    client = anthropic.Anthropic()
    ids = []
    for k in range(0, len(items), 100_000):  # batch limit: 100k requests
        chunk = items[k : k + 100_000]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=it["id"], params=MessageCreateParamsNonStreaming(**_request_params(it, it["model"])))
            for it in chunk
        ])
        ids.append(batch.id)
        print(f"submitted {len(chunk):,} requests as batch {batch.id}")
    (d / "batches.json").write_text(json.dumps(ids))
    print("check back with `collect` (most batches finish within an hour; max 24h)")


def cmd_collect(args: argparse.Namespace) -> None:
    import anthropic

    d = Path(args.dir)
    items = {it["id"]: it for it in (json.loads(line) for line in open(d / "items.jsonl", encoding="utf-8"))}
    client = anthropic.Anthropic()
    stats = {"ok": 0, "bad_slots": 0, "no_tags": 0, "refusal": 0, "errored": 0, "pending": 0}
    usage = {"input": 0, "output": 0, "cache_read": 0}
    pairs = []
    for bid in json.loads((d / "batches.json").read_text()):
        batch = client.messages.batches.retrieve(bid)
        if batch.processing_status != "ended":
            print(f"batch {bid} still {batch.processing_status}: {batch.request_counts}")
            stats["pending"] += 1
            continue
        for res in client.messages.batches.results(bid):  # results come in any order: key by custom_id
            it = items[res.custom_id]
            if res.result.type != "succeeded":
                stats["errored"] += 1
                continue
            msg = res.result.message
            usage["input"] += msg.usage.input_tokens
            usage["output"] += msg.usage.output_tokens
            usage["cache_read"] += msg.usage.cache_read_input_tokens or 0
            if msg.stop_reason == "refusal":
                stats["refusal"] += 1
                continue
            text = "".join(b.text for b in msg.content if b.type == "text")
            m = _ENGLISH.search(text)
            if not m:
                stats["no_tags"] += 1
                continue
            english = " ".join(m.group(1).split())
            # The free, exact check: every slot exactly once, none invented.
            if not slot_check(english, len(it["slots"])):
                stats["bad_slots"] += 1
                continue
            stats["ok"] += 1
            pairs.append({"src": it["masked"], "tgt": english, "slots": it["slots"], "name": it["name"],
                          "file": it["file"], "name_gloss": it["name_gloss"]})
    with open(d / "distilled.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    model = next(iter(items.values()))["model"] if items else DEFAULT_MODEL
    pin, pout, pcache = PRICES.get(model, PRICES[DEFAULT_MODEL])
    cost = 0.5 * (usage["input"] * pin + usage["output"] * pout + usage["cache_read"] * pcache) / 1e6
    print(json.dumps(stats))
    print(f"usage: {usage}  ->  ~${cost:,.2f} at batch prices (cache writes not included)")
    print(f"wrote {len(pairs):,} pairs to {d / 'distilled.jsonl'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="distill")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("input", help="jsonl from `lean_translator gloss`")
    p.add_argument("-o", "--output", required=True, help="working directory")
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--est-output-tokens", type=int, default=250)
    p.set_defaults(func=cmd_prepare)
    s = sub.add_parser("submit")
    s.add_argument("dir")
    s.set_defaults(func=cmd_submit)
    c = sub.add_parser("collect")
    c.add_argument("dir")
    c.set_defaults(func=cmd_collect)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
