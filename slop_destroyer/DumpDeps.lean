import Lean
open Lean

/-!
Dump the dependency graph that Lean already recorded in a *built* project.

    lake env lean --run DumpDeps.lean OUT.tsv PREFIX... -- MODULE...

Loads MODULE... (e.g. `Mathlib`) from their compiled .olean files and writes
one line per constant whose module name starts with one of PREFIX...:

    name <TAB> module <TAB> startLine <TAB> endLine <TAB> isMeta <TAB> deps... <TAB> statement deps...

`dep`s are every constant mentioned in the constant's type or value (its
proof term, for theorems), i.e. exactly what Lean used, including instances
and simp lemmas that never appear by name in the source.

`isMeta` marks tactic/elaborator machinery (anything whose type mentions the
`Lean` namespace): it runs while proofs are checked but never appears inside
the finished proofs, so the graph alone can't tell it's needed.
-/

def splitArgs (args : List String) : List String × List String :=
  let rec go (acc : List String) : List String → List String × List String
    | [] => (acc.reverse, [])
    | "--" :: rest => (acc.reverse, rest)
    | a :: rest => go (a :: acc) rest
  go [] args

/-- Fingerprint of a theorem's statement: structural hash with universe
parameters renamed to u0, u1, ... (bound variables are already nameless de
Bruijn indices), so two theorems stating the same thing get the same hash. -/
def fingerprint (ci : ConstantInfo) : UInt64 :=
  let us := ci.levelParams.mapIdx fun i _ => Level.param (Name.mkSimple s!"u{i}")
  (ci.type.instantiateLevelParams ci.levelParams us).hash

def isMetaType (e : Expr) : Bool :=
  e.getUsedConstants.any fun c => (`Lean).isPrefixOf c || (`IO).isPrefixOf c

unsafe def main (args : List String) : IO UInt32 := do
  let out :: rest := args | do IO.eprintln "usage: DumpDeps OUT [--fp FP.tsv] PREFIX... -- MODULE..."; return 1
  let (fpOut, rest) := match rest with
    | "--fp" :: f :: r => (some f, r)
    | r => (none, r)
  let (prefixes, mods) := splitArgs rest
  if mods.isEmpty then
    IO.eprintln "no modules given"; return 1
  let prefixes := prefixes.map String.toName
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let env ← importModules (mods.toArray.map fun m => { module := m.toName }) {} (trustLevel := 1024) (loadExts := true)
  let modNames := env.header.moduleNames
  let h ← IO.FS.Handle.mk out .write
  let mut n := 0
  for (name, ci) in env.constants.map₁.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let mod := modNames[idx.toNat]!
    unless prefixes.any (·.isPrefixOf mod) do continue
    let (s, e) := match declRangeExt.find? env name with
      | some r => (r.range.pos.line, r.range.endPos.line)
      | none => (0, 0)
    let deps := ci.getUsedConstantsAsSet.foldl (fun acc c => acc.push (toString c)) (#[] : Array String)
    let tdeps := ci.type.getUsedConstants.map toString
    h.putStrLn s!"{name}\t{mod}\t{s}\t{e}\t{isMetaType ci.type}\t{" ".intercalate deps.toList}\t{" ".intercalate tdeps.toList}"
    n := n + 1
  IO.eprintln s!"dumped {n} constants"
  -- Statement fingerprints of *every* theorem in scope (dependencies included),
  -- so re-proofs of library lemmas can be found too.
  if let some fp := fpOut then
    let hf ← IO.FS.Handle.mk fp .write
    let mut m := 0
    for (name, ci) in env.constants.map₁.toList do
      if name.isInternal then continue
      let .thmInfo _ := ci | continue
      let some idx := env.getModuleIdxFor? name | continue
      hf.putStrLn s!"{fingerprint ci}\t{name}\t{modNames[idx.toNat]!}"
      m := m + 1
    IO.eprintln s!"fingerprinted {m} theorems"
  return 0
