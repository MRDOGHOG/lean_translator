# proofsat: proof search as a puzzle

A working, end-to-end version of "turn *is there a proof of X?* into a
puzzle, solve the puzzle, and get the proof back". It uses a formal system
small enough to actually run: Hofstadter's MIU system.

```
"is there a proof of T in <= K steps?"
        │  encode: the proof is the unknown; the clauses say "this is a valid proof of T"
        ▼
  SAT formula (DIMACS) ──► any solver ──► satisfying assignment
                                                 │  decode
                                                 ▼
                                   proof ──► independent checker (plain Python) ──► VALID / INVALID
```

The same construction works for any formal system, Lean included. For Lean,
though, the "valid proof" clauses would have to contain Lean's entire proof
checker plus every Mathlib definition the statement uses, which would put
the formula in the trillions of clauses. That's why this demo uses MIU.

## MIU in one paragraph

Strings start from the axiom `MI`. There are four rules: `xI → xIU`,
`Mx → Mxx`, `xIIIy → xUy`, and `xUUy → xy`. Hofstadter's famous question is
whether you can reach `MU`. You can't: the number of `I`s is never a
multiple of 3. The solver can only ever say "no proof within K steps". The
*reason* none exists is an insight, not a search result, which makes it a
small stand-in for how something like RH resists brute force.

## Use it

```bash
pip install pycosat
```
```bash
python -m proofsat prove MUIU
```
```
<= 3 steps: no proof   (240 vars, 7,681 clauses, 1 ms)
<= 4 steps: FOUND      (309 vars, 10,216 clauses, 2 ms)
proof: MI  ->  MII  ->  MIIII  ->  MIIIIU  ->  MUIU
independent check: VALID
```

Plug in your own solver. `export` writes standard DIMACS, and `check`
reads standard solver output. Nothing the solver returns is trusted: every
clause is re-checked, and the decoded proof goes through the independent
checker.

```bash
python -m proofsat export MUIU --steps 4 -o muiu.cnf
```
```bash
python -m proofsat check MUIU --steps 4 --solution muiu.sol
```

## What's verified

- The encoding is checked against brute-force search for **every** target
  up to length 7. It is satisfiable exactly when a proof exists, and every
  solution decodes to a real proof (`tests/test_proofsat.py`).
- A solution with a single flipped bit is rejected.

## Size

| bounds (max length, steps) | variables | clauses |
|---|---|---|
| 12, 8 | 585 | 20,356 |
| 20, 15 | 1,872 | 118,887 |
| 30, 25 | 4,737 | 471,502 |
| 40, 40 | 10,157 | 1,378,752 |

## Not here: the Sudoku step

SAT → Sudoku is a known reduction: it goes through Latin-square completion,
and Yato and Seta (2003) showed Sudoku is NP-complete that way. It isn't
implemented here; I didn't want to write it from memory without the
published construction to check against. Any SAT → Sudoku tool can take the
exported `.cnf` as input, and any answer it produces can be mapped back to
SAT and verified with `check`.
