# lean_translator

A lightweight, **multi-stage** Lean 4 → English translator that is *not* an LLM.

```
 Lean 4 source
      │
      ▼
┌─────────────────────────────┐   deterministic, ~140k lines/s on 4 cores
│ Stage 1: parse + gloss      │   all of Mathlib (2.4M lines) in ~17 s
│ lexer → decl splitter →     │   faithful by construction (no guessing)
│ Pratt parser → rule glosser │
└─────────────────────────────┘
      │  "Lean-English": correct but stilted
      ▼
┌─────────────────────────────┐   ~8M-parameter encoder–decoder transformer
│ Stage 2: neural refiner     │   (the same architecture as classic NMT,
│ Lean-English → English      │    e.g. Marian / OPUS-MT, scaled way down)
└─────────────────────────────┘
      │
      ▼
 Fluent English
```

The core idea: **don't make a neural net learn Lean.** Lean's syntax is
completely regular, so a parser can turn it into correct (if robotic)
English for free, with no hallucinations. The neural part then only has to do
English → English style transfer. That's a much easier job, so a tiny model
can handle it.

## Example (stage 1)

```lean
theorem exists_dvd_of_not_prime {n : ℕ} (n2 : 2 ≤ n) (np : ¬Prime n) :
    ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n
```
→
> Theorem Nat.exists_dvd_of_not_prime: Let n be a natural number. Assume that
> 2 is at most n and that n is not prime. Then there exists m such that m
> divides n and m does not equal 1 and m does not equal n.

```lean
variable [CommSemigroup G]
theorem mul_left_comm (a b c : G) : a * (b * c) = b * (a * c)
```
→
> Theorem mul_left_comm: Let G be a commutative semigroup. Let a, b and c be
> elements of G. Then a * (b * c) equals b * (a * c).

## Usage

```bash
pip install -e .                 # stage 1 has no dependencies
pip install -e '.[refiner]'      # stage 2 needs torch + tokenizers
```

Stage 1, a whole directory, multiprocess, JSONL out:
```bash
python -m lean_translator gloss path/to/Mathlib -o mathlib.jsonl
```

Stage 1, quick look from stdin:
```bash
echo 'theorem t (a b : ℕ) : a + b = b + a := by omega' | python -m lean_translator gloss -
```

Stage 2, build training pairs, train, run:
```bash
python -m lean_translator pairs path/to/Mathlib -o pairs.jsonl
python -m lean_translator.refiner.train pairs.jsonl -o refiner.pt --minutes 30
python -m lean_translator refine mathlib.jsonl -o mathlib.english.jsonl --model refiner.pt
```

Each output row has `name`, `kind`, `file`, `line`, `lean` (the original
statement), `gloss` (stage 1), `name_gloss`, `doc`, and after stage 2
`english`.

## How each stage works

### Stage 1 — `lean_translator/`
| file | job |
|---|---|
| `lexer.py` | Unicode-aware Lean tokenizer. Keeps `/-- docstrings -/`, drops comments, and handles decorated operators (`‖x‖₊`, `*ᵥ`, `⊗ₜ[R]`, `∑'`). |
| `decls.py` | Splits a file into declarations and **skips proofs entirely**: a statement ends at the top-level `:=`, and the proof is jumped over to the next column-0 command. This is the main speed trick. It also tracks `namespace`/`section`/`variable` so `[Group G]` from a section header is still known, and keeps only the variables the statement actually uses. |
| `parser.py` | Pratt (precedence-climbing) parser for the term language: logic, `∀/∃/∃!`, relations, arithmetic, `∑ i ∈ s`, `∫ x in a..b`, set-builder, `fun`, `(· * x)` sections, coercions, `R[X]`, `f^[n]`, anonymous constructors… Anything unsupported becomes a `Raw` node, so it never crashes. |
| `gloss.py` | AST → English. Quantifiers, connectives and relations become words. Type classes become "Let G be a group". Hypotheses become "Assume that…". Predicates become "p is prime" / "f is continuous on s". Terms stay mathematical (`a * (b * c)`), coercions and casts are dropped, and `x.foo` dot-notation is resolved. |
| `vocab.py` | Plain-data dictionaries (types, type classes, predicates, functions, name words). **This is where to add coverage.** |

### Stage 2 — `lean_translator/refiner/`

```
 Lean ─► stage 1 ─► masked gloss ───────► small seq2seq ───► English with [T0] [T1]
                    "Then [T0] equals [T1]."                      │
                    slots: n * 2, n + n                           ▼
                                           ① slot check: every [Tk] exactly once?
                                           ② round trip: reverse model gets back the gloss?
                                                  │ yes                     │ no
                                                  ▼                         ▼
                                   fill slots: "… n * 2 = n + n"   use the stage-1 gloss
```

Four safeguards stop the neural layer from saying wrong things:

1. **Masking (`gloss.py`).** Compound math terms become slots `[T0]`, `[T1]`, …
   The neural model only rewrites the prose *around* the math; it never sees
   or produces the math, so it can't get it wrong. Bare variables and numbers
   stay inline. Across all of Mathlib, filling the slots back in reproduces
   the stage-1 gloss exactly.
2. **A pretrained start (`train_hf.py`).** Fine-tune `google/flan-t5-small`
   (~77M parameters), which already knows English, instead of training from
   scratch. The slot markers, plus any character T5's vocabulary lacks
   (`{`, `ℕ`, `α`, …), are added as tokens.
3. **Clean targets (`distill.py`).** A large model writes faithful English for
   each masked gloss, once and offline: either Claude through the Message
   Batches API, or a local model through Ollama, free. It must
   copy every slot back verbatim, and replies that don't are discarded. The
   big model only makes training data; the shipped translator stays small.
4. **Round-trip check (`roundtrip.py`).** A reverse model (English → gloss)
   checks that the English still says the same thing. This is used both to
   filter the training data and at translation time. It catches dropped or
   invented conditions; it can miss a misreading that both directions share,
   which is why the exact slot check always runs as well.

If any check fails, the output falls back to the stage-1 gloss. The worst
case is robotic English, never wrong English.

Other files: `model.py` and `train.py` are the original from-scratch
transformer, for use without a download; `backends.py` puts both model
kinds behind one interface; and `data.py` builds the free (gloss → Mathlib
docstring) pairs.

### Recipe (on a GPU machine)

```bash
pip install -e '.[refiner,distill]'
python -m lean_translator gloss path/to/Mathlib -o mathlib.jsonl
```

1. Make the targets. Start with a small pilot to check quality and the real cost:
```bash
python -m lean_translator.refiner.distill prepare mathlib.jsonl -n 1000 -o pilot/
```
```bash
python -m lean_translator.refiner.distill submit pilot/
```
```bash
python -m lean_translator.refiner.distill collect pilot/
```

Or do it free and locally with any model you've pulled into
[Ollama](https://ollama.com) instead of `submit`/`collect`. Quality depends on
the model; bigger is better if your GPU fits it. Every reply goes through
the same slot check, and if you stop it, rerunning resumes where it left off.
```bash
python -m lean_translator.refiner.distill ollama pilot/ --ollama-model qwen2.5:14b
```

2. Train the reverse model, filter the targets by round trip, then train the forward model:
```bash
python -m lean_translator.refiner.train_hf pilot/distilled.jsonl -o reverse-t5/ --reverse --bf16
```
```bash
python -m lean_translator.refiner.roundtrip filter pilot/distilled.jsonl --reverse reverse-t5/ -o clean.jsonl
```
```bash
python -m lean_translator.refiner.train_hf clean.jsonl -o refiner-t5/ --bf16
```

3. Translate, with every check on:
```bash
python -m lean_translator refine mathlib.jsonl -o mathlib.english.jsonl --model refiner-t5/ --check reverse-t5/
```

Rough distillation cost for 250k statements at batch prices (`prepare`
prints an estimate; `collect` reports the actual usage): Opus 5.5 ~$700,
Sonnet 5.5 ~$360, Haiku 5.5 ~$20.

## Current results

**Stage 1** runs on all of Mathlib (8,733 files, 2.39M lines, 263,823
declarations) in about 17 s on 4 CPU cores, with zero crashes.

**Stage 2**, first run, before any of the safeguards above: a 3.7M-parameter
from-scratch model, trained for 40 minutes on CPU on docstring pairs. It
learned the docstring style but regularly said false things:

| Lean | stage 1 (faithful) | first stage 2 (fluent, wrong) |
|---|---|---|
| `trace (diagonal d) = ∑ i, d i` | Then trace(diagonal(d)) equals (the sum of d(i) over all i). | The trace of a matrix is diagonal. |
| `(hx : x ∈ I) : x * y ∈ I` | Assume that x is in I. Then x * y is in I. | If `x` is a non-zero ideal, then … |

The safeguarded pipeline has been smoke-tested end-to-end on CPU, with a
tiny randomly-initialized T5 and synthetic pairs, because this sandbox can't
download pretrained models or call the API. The plumbing works, and with
deliberately bad models the checks rejected 100% of the outputs and fell
back to the gloss. Its *quality* is untested until the GPU run above.

## Honest notes / limits

- **Speed:** stage 1 really is "millions of lines in seconds" (it's pure
  Python; a Rust/C port would be 20–50× faster still). Stage 2 is the slow
  part. On CPU, plan on minutes to an hour for all of Mathlib. On a GPU with
  big batches it's minutes. Using a smaller model or skipping stage 2 for
  simple statements helps.
- **Faithfulness:** stage 1 says only what the code says. Stage 2 is a neural
  net, so it *can* drift or hallucinate, especially since docstrings are often
  looser than the formal statement. That's why the gloss is always kept next
  to the English: the gloss is the audit trail.
- **Coverage:** about 89% of Mathlib statements parse fully. The rest have
  pieces rendered as raw `code` (exotic notation like `𝓝[<] a`,
  `C^n⟮I, M⟯`, category-theory infix). Add those to `parser.py`/`vocab.py`
  as needed.
- **No elaboration:** this reads *surface syntax*, not Lean's elaborated
  terms, so implicit arguments and overloaded notation are interpreted
  heuristically. A future version could read `.olean`/`.ilean` export data
  for exact types.
- **English → Lean** (the reverse direction) is a much harder problem,
  because the output has to type-check. The same stage design could work,
  with a neural model producing Lean-English and a deterministic
  Lean-English → Lean compiler, but that compiler isn't built yet.

## Tests

```bash
python -m pytest -q
```
