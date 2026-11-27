# SLOP_DESTROYER

Give it the theorem(s) you care about. It deletes everything in the Lean
project they don't need. No AI is involved: it's either pure text analysis
or Lean's own compiler.

| mode | needs Lean? | speed | precision | use it for |
|---|---|---|---|---|
| `static` | **no**, never runs Lean | all of Mathlib (2.4M lines) in ~2 min | conservative (keeps extra) | huge projects, no toolchain at hand |
| `graph` | reads the built project once | one dependency dump + one optional build | exact, from Lean's own dependency records | whole projects, millions of lines |
| `probe` | one compile per attempt | seconds to minutes per file | down to single tactic lines and `simp` arguments | polishing one file |

Each mode prints its plan and touches nothing unless you pass `--write`
(probe writes `*.destroyed.lean` unless you pass `--in-place`). Commit
first so you can review the diff and revert.

## static: no compiler at all

```bash
python -m slop_destroyer static --root ~/myproj --include MyProj --keep MyProj.main_theorem --show 50
```

It follows names through the source, resolving them the way Lean does:
the current `namespace`, `open`ed namespaces, the declaration's own
namespace, and the top level. Local variables (`fun x`, `intro h`,
binders) shadow global names. Some uses never appear by name, so those
are handled conservatively:

- **Invisible uses:** instances, `@[simp]`/`@[ext]`/`@[norm_num]`/… lemmas,
  and `@[to_additive]` generated names. These are kept as soon as their
  statement mentions anything that's kept.
- **Dot notation** (`h.le`): only the compiler knows `h`'s type, so
  `X.le` is kept if `X` is kept or comes from outside the project.

The result is that static mode keeps more than needed, never deletes
anything a kept declaration mentions by name, and runs no build. Run one
build afterwards to confirm.

## graph: Lean's own dependency records

```bash
lake build                                    # the project must be built
python -m slop_destroyer graph --root ~/myproj --module MyProj --keep MyProj.main_theorem --show
python -m slop_destroyer graph --root ~/myproj --module MyProj --keep MyProj.main_theorem --write --verify
```

1. `DumpDeps.lean` loads the compiled project once and dumps, for every
   declaration, every constant its statement and proof term use. That
   includes instances and lemmas that never appear by name.
2. It walks that graph from your theorems, and anything unreachable is
   deletable.
3. It also keeps what kept *source* names (`simp [foo]` needs `foo` to
   exist even if the final proof didn't use it), and "invisible"
   declarations whose statement mentions something kept. `rfl` simp lemmas
   used by `simp` leave no trace even in the proof term; the test fixture
   has one.
4. `--verify` runs one build. If it fails, the declarations the errors
   name are restored. If the errors don't name anything, half the
   deletions are restored per round until it builds. If nothing works,
   every file is put back.

## probe: down to the tactic line

```bash
python -m slop_destroyer probe Foo.lean --keep main_theorem --lean "lake env lean" --root ~/myproj --diff
```

It tries deleting, one at a time, from the bottom up:
1. whole commands;
2. single tactic lines;
3. single `simp [...]` / `linarith [...]` arguments;
4. whole commands again, since step 3 can free more.

A deletion is kept only if all three checks pass:
- the file still compiles;
- every `--keep` theorem has **exactly** the same statement, compared with
  `pp.all` so hidden instances count;
- no theorem picked up a `sorry` (checked with `#print axioms`).

## lintfix: slop inside proofs, from Lean's own linters

```bash
python -m slop_destroyer lintfix --root ~/myproj --write --verify
```

One compile per file, with Lean's "unused" linters switched on, reports at
exact positions:
- **unused `simp` arguments**, which get removed;
- **unused `have`s**, whose block gets deleted;
- **tactics that do nothing** (a Mathlib linter), whose line gets deleted.

All the fixes are applied, then the project is rebuilt, and any file that
no longer compiles gets its original back. Run graph mode again afterwards:
removing `simp [..., helper]` can make `helper` itself unused.

## redundant: theorems proved twice

```bash
python -m slop_destroyer redundant --root ~/myproj --module MyProj --keep MyProj.main --write --verify
```

`DumpDeps.lean --fp` fingerprints every theorem's statement, including every
Mathlib and core theorem in scope. The fingerprint is a structural hash with
universes renamed (variable names already don't matter, because Lean stores
bound variables as positions). Theorems with equal fingerprints state the
same thing, so one proof is enough. Each duplicate becomes `alias X := Y` on
projects with Batteries or Mathlib, or `theorem X … := @Y args` on plain
Lean. Its name stays, its proof goes, and a rebuild restores anything that
breaks. It finds both duplicates inside the project and project lemmas that
re-prove something already in Mathlib.

## speed: what's slow, and the standard fix

```bash
python -m slop_destroyer speed --root ~/myproj --top 50                          # report only
python -m slop_destroyer speed --root ~/myproj --top 50 --simp-only --write --verify
```

Compiles every file with Lean's profiler on, then reports time per
declaration and per tactic, slowest first. With `--simp-only`, each bare
`simp` / `simp_all` / `simpa` / `dsimp` in the slowest declarations is
replaced by the exact `simp only [...]` call that Lean's `simp?` suggests.
That's the same proof with far less searching, and it can't break when the
simp library changes. The changed files are rebuilt and re-profiled.

## Measured on a real project (lsernau/grh-lean, `grh-compact`, 114k lines)

| | |
|---|---|
| baseline `lake build` on GitHub's 4-core runner | 2 h 14 min |
| constants Lean recorded for the project | 51,804 |
| needed by the 5 headline theorems | 37,911 |
| used by no proof term | 27 (26 anonymous instances + 1 lemma) |

Deleting those 27 broke the build: instances can be needed while proofs are
*checked* without appearing in any finished proof term. Graph mode now never
deletes instances or classes. **Static mode's estimate for this project, about
27,000 deletable lines, was far too high:** on dense code, text alone can't
see most of the uses. Treat static numbers as an upper bound, and rely on
graph mode for real deletions.

## Not handled

- **Shrinking imports.** For that, use Mathlib's `shake`, which removes
  unused imports.
- **Statement changes.** Hypotheses are never removed from a kept
  theorem, because that changes what it says.
- **Static mode is conservative:** for a small target in Mathlib it keeps
  ~200k of 270k declarations, because any simp lemma about a type the
  proof touches *could* fire. Graph mode doesn't have that problem.
