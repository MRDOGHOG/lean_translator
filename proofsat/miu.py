"""Proof search as SAT, on Hofstadter's MIU system.

The MIU system (Gödel, Escher, Bach): strings over {M, I, U}, axiom "MI",
and four rules (x, y are any strings):

    1. xI   -> xIU        (append U after a trailing I)
    2. Mx   -> Mxx        (double everything after M)
    3. xIIIy -> xUy       (replace III by U)
    4. xUUy -> xy         (delete UU)

A *proof* of a string T is a sequence MI = s0, s1, ..., sK = T where each
step applies one rule.  Checking a proof is easy.  Finding one is search.

This module turns "is there a proof of T with at most K steps, never longer
than L symbols?" into a SAT formula, exactly the construction you'd use for
any formal system (Lean included): the proof is the unknown, and the clauses
say "this is a valid proof of T".  A satisfying assignment decodes to a proof,
which `check_proof` (plain Python, no SAT involved) then verifies.

Famous fact: MU has no proof at any length: the number of I's is never a
multiple of 3.  The solver can only report "no proof within K steps"; the
reason it never succeeds takes a mathematical insight, not search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

SYMS = ("I", "U", "_")  # tail alphabet; "_" is blank padding


# ---------------------------------------------------------------------------
# Plain-Python reference checker (independent of the SAT encoding).

def successors(s: str) -> set[str]:
    out = set()
    if s.endswith("I"):
        out.add(s + "U")
    if s.startswith("M"):
        out.add("M" + s[1:] * 2)
    for p in range(len(s) - 2):
        if s[p : p + 3] == "III":
            out.add(s[:p] + "U" + s[p + 3 :])
    for p in range(len(s) - 1):
        if s[p : p + 2] == "UU":
            out.add(s[:p] + s[p + 2 :])
    return out


def check_proof(proof: list[str], target: str) -> bool:
    if not proof or proof[0] != "MI" or proof[-1] != target:
        return False
    return all(b in successors(a) for a, b in zip(proof, proof[1:]))


# ---------------------------------------------------------------------------
# SAT encoding.

@dataclass
class CNF:
    n_vars: int = 0
    clauses: list[list[int]] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)

    def var(self, name: str) -> int:
        self.n_vars += 1
        self.names[self.n_vars] = name
        return self.n_vars

    def add(self, *lits: int) -> None:
        self.clauses.append(list(lits))

    def exactly_one(self, lits: list[int]) -> None:
        self.add(*lits)
        for a, b in combinations(lits, 2):
            self.add(-a, -b)

    def to_dimacs(self) -> str:
        head = [f"p cnf {self.n_vars} {len(self.clauses)}"]
        return "\n".join(head + [" ".join(map(str, c)) + " 0" for c in self.clauses]) + "\n"


@dataclass
class MIUEncoding:
    target: str
    max_steps: int
    max_len: int
    cnf: CNF
    cell: list[list[dict[str, int]]]  # cell[k][i][sym] -> var: step k, tail position i
    selectors: list[list[tuple[str, int]]]  # per step: (rule description, var)

    def decode(self, true_vars: set[int]) -> list[str]:
        strings = []
        for k in range(self.max_steps + 1):
            tail = ""
            for pos in self.cell[k]:
                sym = next(s for s, v in pos.items() if v in true_vars)
                if sym != "_":
                    tail += sym
            strings.append("M" + tail)
        # collapse "stay" steps so the proof reads naturally
        proof = [strings[0]]
        for s in strings[1:]:
            if s != proof[-1]:
                proof.append(s)
        return proof


def encode(target: str, max_steps: int, max_len: int) -> MIUEncoding:
    """Clauses that are satisfiable iff `target` has a proof of <= max_steps steps
    in which no string exceeds max_len symbols (counting the leading M)."""
    if not target.startswith("M") or any(c not in "IU" for c in target[1:]):
        raise ValueError("MIU strings look like M followed by I's and U's")
    N = max_len - 1  # tail length
    if len(target) - 1 > N:
        raise ValueError("target longer than max_len")
    cnf = CNF()
    cell = [[{s: cnf.var(f"s{k}[{i}]={s}") for s in SYMS} for i in range(N)] for k in range(max_steps + 1)]

    # Each tail position holds exactly one symbol; blanks only at the end.
    for k in range(max_steps + 1):
        for i in range(N):
            cnf.exactly_one(list(cell[k][i].values()))
            if i + 1 < N:
                cnf.add(-cell[k][i]["_"], cell[k][i + 1]["_"])

    def lit(k: int, i: int, s: str) -> int | bool:
        """Literal for "tail k has symbol s at position i" (beyond the end: blank)."""
        if i >= N:
            return s == "_"
        return cell[k][i][s]

    def require(sel: int, k: int, i: int, s: str) -> None:
        l = lit(k, i, s)
        if l is True:
            return
        if l is False:
            cnf.add(-sel)  # this rule choice is impossible
            return
        cnf.add(-sel, l)

    def same(sel: int, k: int, i: int, j: int) -> None:
        """sel -> tail_{k+1}[i] == tail_k[j]."""
        for s in SYMS:
            a, b = lit(k, j, s), lit(k + 1, i, s)
            for x, y in ((a, b), (b, a)):  # x -> y, under sel
                if x is False or y is True:
                    continue
                if x is True and y is False:
                    cnf.add(-sel)
                elif x is True:
                    cnf.add(-sel, y)
                elif y is False:
                    cnf.add(-sel, -x)
                else:
                    cnf.add(-sel, -x, y)

    def has_length(sel: int, k: int, m: int) -> None:
        if m > 0:
            cnf.add(-sel, -cell[k][m - 1]["_"])
        if m < N:
            require(sel, k, m, "_")

    selectors: list[list[tuple[str, int]]] = []
    for k in range(max_steps):
        sels: list[tuple[str, int]] = []

        stay = cnf.var(f"step{k}:stay")  # lets proofs be shorter than max_steps
        for i in range(N):
            same(stay, k, i, i)
        sels.append(("stay", stay))

        for m in range(1, N):  # rule 1: tail of length m ends in I, append U
            v = cnf.var(f"step{k}:r1@{m}")
            has_length(v, k, m)
            require(v, k, m - 1, "I")
            for i in range(m):
                same(v, k, i, i)
            require(v, k + 1, m, "U")
            for i in range(m + 1, N):
                require(v, k + 1, i, "_")
            sels.append((f"rule 1 (append U), length {m}", v))

        for m in range(0, N // 2 + 1):  # rule 2: double the tail
            v = cnf.var(f"step{k}:r2@{m}")
            has_length(v, k, m)
            for i in range(N):
                if i < m:
                    same(v, k, i, i)
                elif i < 2 * m:
                    same(v, k, i, i - m)
                else:
                    require(v, k + 1, i, "_")
            sels.append((f"rule 2 (double), length {m}", v))

        for p in range(N - 2):  # rule 3: III -> U at p
            v = cnf.var(f"step{k}:r3@{p}")
            for d in range(3):
                require(v, k, p + d, "I")
            for i in range(N):
                if i < p:
                    same(v, k, i, i)
                elif i == p:
                    require(v, k + 1, i, "U")
                else:
                    same(v, k, i, i + 2)
            sels.append((f"rule 3 (III->U) at {p}", v))

        for p in range(N - 1):  # rule 4: delete UU at p
            v = cnf.var(f"step{k}:r4@{p}")
            for d in range(2):
                require(v, k, p + d, "U")
            for i in range(N):
                same(v, k, i, i if i < p else i + 2)
            sels.append((f"rule 4 (delete UU) at {p}", v))

        cnf.exactly_one([v for _, v in sels])
        selectors.append(sels)

    # Start at the axiom MI, end at the target.
    for k, word in ((0, "MI"), (max_steps, target)):
        tail = word[1:]
        for i in range(N):
            cnf.add(cell[k][i][tail[i] if i < len(tail) else "_"])

    return MIUEncoding(target, max_steps, max_len, cnf, cell, selectors)
