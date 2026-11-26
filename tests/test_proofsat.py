from itertools import product

import pytest

from proofsat.miu import check_proof, encode, successors

pycosat = pytest.importorskip("pycosat")


def reachable(max_steps: int, max_len: int) -> set[str]:
    """Brute force: every string provable in <= max_steps, never exceeding max_len."""
    seen = frontier = {"MI"}
    for _ in range(max_steps):
        frontier = {t for s in frontier for t in successors(s) if len(t) <= max_len} - seen
        seen = seen | frontier
    return seen


def test_reference_checker():
    assert check_proof(["MI", "MII", "MIIII", "MIIIIU", "MUIU"], "MUIU")
    assert not check_proof(["MI", "MIIII"], "MIIII")  # not one rule step
    assert not check_proof(["MI", "MU"], "MU")


def test_sat_encoding_matches_brute_force_exactly():
    """Satisfiable <=> a proof exists, for every target up to length 7 (sound and complete)."""
    max_len, max_steps = 7, 4
    truth = reachable(max_steps, max_len)
    for n in range(0, max_len):
        for tail in product("IU", repeat=n):
            target = "M" + "".join(tail)
            enc = encode(target, max_steps, max_len)
            sol = pycosat.solve(enc.cnf.clauses)
            found = sol != "UNSAT"
            assert found == (target in truth), target
            if found:
                proof = enc.decode({v for v in sol if v > 0})
                assert check_proof(proof, target), (target, proof)


def test_mu_has_no_proof():
    # Every reachable string's I-count is not a multiple of 3 (Hofstadter's invariant),
    # so no bound ever finds MU.
    enc = encode("MU", 10, 12)
    assert pycosat.solve(enc.cnf.clauses) == "UNSAT"
