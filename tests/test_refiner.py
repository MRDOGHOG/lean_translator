import json

import pytest

pytest.importorskip("torch")

from lean_translator.refiner import translate as tr  # noqa: E402
from lean_translator.refiner.backends import Translator, _fix_slot_spacing  # noqa: E402
from lean_translator.refiner.roundtrip import similarity  # noqa: E402

GLOSS = "Theorem t: Let n be a natural number. Then [T0] equals [T1]."
ROW = {
    "name": "t", "kind": "theorem", "file": "A.lean", "line": 1, "lean": "...", "doc": None, "name_gloss": "t",
    "gloss": "Theorem t: Let n be a natural number. Then n * 2 equals n + n.",
    "masked": GLOSS, "slots": ["n * 2", "n + n"],
}


class Fake(Translator):
    def __init__(self, mapping, meta):
        self.mapping, self.meta = mapping, meta

    def translate(self, texts, batch_size=64):
        return [self.mapping(t) for t in texts]


def test_similarity_identical_and_slot_mismatch():
    assert similarity(GLOSS, GLOSS) == 1.0
    assert similarity(GLOSS, GLOSS.replace("[T1]", "[T2]")) == 0.0  # different slots -> 0
    assert 0 < similarity(GLOSS, "Let n be a natural number. [T0] equals [T1].") < 1


def test_fix_slot_spacing():
    assert _fix_slot_spacing("then[T0]equals [T1].") == "then [T0] equals [T1]."


def _run(tmp_path, monkeypatch, forward, reverse=None):
    models = {"fwd": Fake(forward, {"masked": True}), "rev": Fake(reverse or (lambda t: t), {"direction": "reverse"})}
    monkeypatch.setattr(tr, "load_translator", lambda path, device=None: models[path])
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text(json.dumps(ROW) + "\n")
    tr.refine_file(str(inp), str(out), "fwd", check="rev" if reverse else None)
    return json.loads(out.read_text())


def test_refine_fills_slots_back_in(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, lambda t: "For every natural number n, [T0] = [T1].")
    assert r["english_source"] == "refiner"
    assert r["english"] == "For every natural number n, n * 2 = n + n."


def test_refine_falls_back_when_a_slot_is_dropped(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, lambda t: "For every n, [T0] holds.")
    assert r["english_source"] == "gloss" and r["english"] == ROW["gloss"]


def test_refine_falls_back_when_round_trip_fails(tmp_path, monkeypatch):
    # Slots are intact, but the reverse model can't recover the gloss: reject.
    r = _run(tmp_path, monkeypatch, lambda t: "[T0] and [T1] are unrelated.", reverse=lambda t: "[T0] [T1]")
    assert r["english_source"] == "gloss"
    assert r["rt_score"] < 0.6


def test_refine_accepts_when_round_trip_succeeds(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, lambda t: "For every natural number n, [T0] = [T1].", reverse=lambda t: GLOSS)
    assert r["english_source"] == "refiner" and r["rt_score"] == 1.0
