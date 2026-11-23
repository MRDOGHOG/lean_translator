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
- **Free parallel data.** Every Mathlib declaration with a docstring gives a
  pair (stage-1 gloss → human docstring). That's about 72k pairs, with no labelling needed.
- `model.py`: a standard pre-norm Transformer encoder–decoder with tied
  embeddings and a shared byte-level BPE vocabulary (8k).
- `train.py`: token-budget length bucketing, label smoothing, inverse-sqrt
  LR schedule, validation on **held-out files**, and a wall-clock budget.
- `translate.py`: batched greedy decoding over stage-1 JSONL.

## Current results

**Stage 1** runs on all of Mathlib (8,733 files, 2.39M lines, 263,823
declarations) in about 17 s on 4 CPU cores, with zero crashes.

**Stage 2** (first run): a 3.7M-parameter model, trained for 40 minutes on CPU
over 4 epochs, reached a held-out-file validation perplexity of 19.8.
*It is not good enough yet.* It learned the docstring style, but it
regularly says false things:

| Lean | stage 1 (faithful) | stage 2 (fluent, wrong) |
|---|---|---|
| `trace (diagonal d) = ∑ i, d i` | Then trace(diagonal(d)) equals (the sum of d(i) over all i). | The trace of a matrix is diagonal. |
| `(hx : x ∈ I) : x * y ∈ I` | Assume that x is in I. Then x * y is in I. | If `x` is a non-zero ideal, then … |

So for now, **use the stage-1 output.** To make stage 2 trustworthy, the
likely levers, roughly in order of impact, are:
1. **Scale:** around 30–60M parameters trained for hours on a GPU, which is
   still tiny next to an LLM.
2. **A copy/pointer mechanism**, so variables and terms are copied from the
   gloss instead of being regenerated.
3. **Better targets:** docstrings often aren't paraphrases of the statement.
   Filter pairs by overlap, or bootstrap cleaner targets.
4. **A faithfulness check:** the identifiers and relations in the output
   must match the gloss, and if they don't, fall back to the gloss.

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
