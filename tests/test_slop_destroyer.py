import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slop_destroyer.spans import lemma_args, split_commands, tactic_units
from slop_destroyer.static import analyse

FIXTURE = Path(__file__).parent / "fixtures" / "slop"
HAVE_LAKE = shutil.which("lake") is not None


def test_split_commands_keeps_prefixes_attached():
    src = (
        "set_option maxHeartbeats 400 in\ntheorem foo : True := trivial\n\n"
        "open Nat in\n/-- doc -/\n@[simp] lemma bar : 1 = 1 := rfl\n\n"
        "open Nat\n-- note\ntheorem baz : True := by\n  trivial\n#check baz\n"
    )
    got = [(s.kind, s.head, s.name, s.start, s.end) for s in split_commands(src)]
    assert got == [
        ("decl", "theorem", "foo", 1, 2),
        ("decl", "lemma", "bar", 4, 6),
        ("other", "open", "", 8, 8),
        ("decl", "theorem", "baz", 9, 11),  # claims the comment above it
        ("debug", "#check", "", 12, 12),
    ]


def test_tactic_units_and_lemma_args():
    src = "theorem t : True := by\n  have h : 1 = 1 := by\n    rfl\n  trivial\n"
    (s,) = split_commands(src)
    assert tactic_units(s, src.split("\n")) == [(2, 3), (4, 4)]
    line = "  simp only [mul_assoc, foo (bar x), baz]"
    assert [line[a:b] for a, b in lemma_args(line)] == ["mul_assoc,", " foo (bar x),", ", baz"]


def test_static_mode_on_fixture():
    files = sorted(FIXTURE.rglob("*.lean"))
    res = analyse(files, ["Pt.add_x_comm"], jobs=1)
    deleted = {(it.file.name, it.head, (it.names or [""])[0]) for it in res.deleted}
    assert deleted == {
        ("Basic.lean", "def", "Pt.norm1"),
        ("Basic.lean", "theorem", "Pt.norm1_zero"),
        ("Basic.lean", "theorem", "Pt.norm1_zero'"),
        ("Basic.lean", "#eval", ""),
        ("Main.lean", "example", ""),
        ("Main.lean", "#check", ""),
    }
    kept = {n for k in res.keep for n in res.items[k].names}
    # used invisibly (instance, @[simp] lemma) or named in a kept proof: must survive
    assert {"Pt.add_x", "helper", "Pt.add_x_comm"} <= kept


@pytest.mark.skipif(not HAVE_LAKE, reason="needs a Lean toolchain (lake) on PATH")
def test_graph_then_probe_end_to_end(tmp_path):
    proj = tmp_path / "slop"
    shutil.copytree(FIXTURE, proj)
    run = lambda *a: subprocess.run([sys.executable, "-m", "slop_destroyer", *a], capture_output=True, text=True,
                                    cwd=Path(__file__).parents[1])
    assert subprocess.run(["lake", "build"], cwd=proj, capture_output=True).returncode == 0
    r = run("graph", "--root", str(proj), "--module", "Slop", "--keep", "Pt.add_x_comm", "--write", "--verify")
    assert r.returncode == 0, r.stdout + r.stderr
    basic = (proj / "Slop" / "Basic.lean").read_text()
    assert "norm1" not in basic and "Pt.add_x " in basic and "instance" in basic
    r = run("probe", str(proj / "Slop" / "Main.lean"), "--keep", "Pt.add_x_comm", "--lean", "lake env lean",
            "--root", str(proj), "--in-place")
    assert r.returncode == 0, r.stdout + r.stderr
    main = (proj / "Slop" / "Main.lean").read_text()
    assert "helper" not in main and "unused" not in main and "simp [Nat.add_comm]" in main
    assert subprocess.run(["lake", "build"], cwd=proj, capture_output=True).returncode == 0


def test_lintfix_edits():
    from pathlib import Path as P

    from slop_destroyer.lintfix import Warning_, _remove_simp_arg, plan_edits

    assert _remove_simp_arg("  simp [a, foo (b c), d]", 11) == "  simp [a, d]"
    assert _remove_simp_arg("  simp [a, d]", 11) == "  simp [a]"
    assert _remove_simp_arg("  simp only [a] at h", 13) == "  simp only at h"
    src = (
        "theorem t : True := by\n"
        "  have unused : 1 = 1 := by\n"
        "    rfl\n"
        "  have used : True := trivial\n"
        "  skip\n"
        "  exact used\n"
    )
    f = P("x.lean")
    warns = [
        Warning_(f, 2, 7, "unused_var", "Variable name `unused` is not explicitly referenced."),
        Warning_(f, 1, 10, "unused_var", "binder in the statement: must not be touched"),
        Warning_(f, 5, 2, "unused_tactic", "'skip' tactic does nothing"),
    ]
    new, done = plan_edits(src, warns)
    assert new == "theorem t : True := by\n  have used : True := trivial\n  exact used\n"
    assert done == {"unused_have": 1, "unused_tactic": 1}


def test_error_formats_of_current_lean():
    from slop_destroyer import graph, lintfix, redundant

    log = ("✖ [9840/9843] Building New.Main (12s)\n"
           "error: src/New/Main.lean:174:17: Unknown constant `_root_.LFunction_ne_zero_of_half_lt_re`\n"
           "Foo/Bar.lean:3:4: error: unknown identifier 'helper'\n")
    for mod in (lintfix, redundant):
        assert mod.error_files(log) == {"src/New/Main.lean", "Foo/Bar.lean"}
    assert set(graph._UNKNOWN.findall(log)) == {"LFunction_ne_zero_of_half_lt_re", "helper"}


def test_redundant_never_aliases_to_a_dependent_or_generated_name():
    from slop_destroyer.redundant import _AUTO

    assert _AUTO.search("boxIndicator.eq_1") and _AUTO.search("foo.proof_2")
    assert not _AUTO.search("boxIndicator_apply")
