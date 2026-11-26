"""Proof search through SAT, end to end.

Find a proof (iterative deepening; uses pycosat if installed):
    python -m proofsat prove MUIU --max-len 10 --max-steps 8

Export the puzzle for any external solver (DIMACS CNF):
    python -m proofsat export MUIU --steps 4 --max-len 10 -o muiu.cnf

Feed back a solver's answer; it's decoded and independently checked:
    python -m proofsat check MUIU --steps 4 --max-len 10 --solution muiu.sol
(the solution file is standard solver output: lines like "v 1 -2 3 ... 0",
or just whitespace-separated literals.)
"""

from __future__ import annotations

import argparse
import sys
import time

from .miu import check_proof, encode


def _solve(cnf):
    try:
        import pycosat
    except ImportError:
        sys.exit("pycosat is not installed: `pip install pycosat`, or use `export` with your own solver")
    sol = pycosat.solve(cnf.clauses)
    return None if sol == "UNSAT" else {v for v in sol if v > 0}


def _report(enc, true_vars) -> bool:
    proof = enc.decode(true_vars)
    ok = check_proof(proof, enc.target)
    print("proof:", "  ->  ".join(proof))
    print("independent check:", "VALID" if ok else "INVALID (the solution doesn't decode to a real proof)")
    return ok


def cmd_prove(args) -> None:
    for k in range(1, args.max_steps + 1):
        enc = encode(args.target, k, args.max_len)
        t0 = time.perf_counter()
        sol = _solve(enc.cnf)
        dt = time.perf_counter() - t0
        size = f"{enc.cnf.n_vars:,} vars, {len(enc.cnf.clauses):,} clauses"
        if sol is None:
            print(f"<= {k} steps: no proof   ({size}, {dt * 1000:.0f} ms)")
            continue
        print(f"<= {k} steps: FOUND      ({size}, {dt * 1000:.0f} ms)")
        sys.exit(0 if _report(enc, sol) else 1)
    print(f"no proof of {args.target} within {args.max_steps} steps and length {args.max_len}"
          " (search can't tell whether a longer proof exists)")


def cmd_export(args) -> None:
    enc = encode(args.target, args.steps, args.max_len)
    with open(args.output, "w") as f:
        f.write(f"c MIU proof search: target {args.target}, steps <= {args.steps}, max length {args.max_len}\n")
        f.write(enc.cnf.to_dimacs())
    print(f"wrote {args.output}: {enc.cnf.n_vars:,} variables, {len(enc.cnf.clauses):,} clauses")


def cmd_check(args) -> None:
    enc = encode(args.target, args.steps, args.max_len)
    lits = []
    for line in open(args.solution):
        line = line.strip()
        if not line or line[0] in "cs":  # comments, "s SATISFIABLE"
            continue
        lits += [int(x) for x in line.lstrip("v").split()]
    true_vars = {v for v in lits if v > 0}
    # Don't trust the solver: confirm every clause is satisfied.
    assign = {abs(v): v > 0 for v in lits if v != 0}
    broken = sum(1 for c in enc.cnf.clauses if not any(assign.get(abs(l), False) == (l > 0) for l in c))
    print(f"clauses violated by the solution: {broken}")
    sys.exit(0 if _report(enc, true_vars) and broken == 0 else 1)


def main() -> None:
    ap = argparse.ArgumentParser(prog="proofsat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove")
    p.add_argument("target")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-len", type=int, default=10)
    p.set_defaults(func=cmd_prove)
    e = sub.add_parser("export")
    e.add_argument("target")
    e.add_argument("--steps", type=int, required=True)
    e.add_argument("--max-len", type=int, default=10)
    e.add_argument("-o", "--output", required=True)
    e.set_defaults(func=cmd_export)
    c = sub.add_parser("check")
    c.add_argument("target")
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--max-len", type=int, default=10)
    c.add_argument("--solution", required=True)
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
