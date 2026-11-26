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

## Not handled

- **Shrinking imports.** For that, use Mathlib's `shake`, which removes
  unused imports.
- **Statement changes.** Hypotheses are never removed from a kept
  theorem, because that changes what it says.
- **Static mode is conservative:** for a small target in Mathlib it keeps
  ~200k of 270k declarations, because any simp lemma about a type the
  proof touches *could* fire. Graph mode doesn't have that problem.
