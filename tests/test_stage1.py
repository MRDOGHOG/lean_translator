from lean_translator import translate
from lean_translator.decls import extract_decls
from lean_translator.gloss import gloss_name
from lean_translator.lexer import tokenize
from lean_translator.parser import Bin, Quant, Raw, parse


def one(src: str) -> str:
    out = translate(src)
    assert len(out) == 1, out
    return out[0]


def test_lexer_decorated_operators():
    texts = [t.text for t in tokenize("‖x‖₊ * A *ᵥ v (1 - ‖z‖)⁻¹ ∑' n, f n")]
    assert "‖₊" in texts and "*ᵥ" in texts and "∑'" in texts
    assert ")" in texts and "⁻¹" in texts  # brackets never absorb modifiers


def test_lexer_keeps_docstrings_drops_comments():
    toks = tokenize("/-- hello -/ -- comment\n/- block /- nested -/ -/ theorem")
    assert [(t.kind, t.text) for t in toks] == [("doc", "hello"), ("ident", "theorem")]


def test_parser_precedence():
    n = parse(tokenize("a + b * c = d ∧ p ↔ q"))
    assert isinstance(n, Bin) and n.op == "↔"
    assert n.lhs.op == "∧" and n.lhs.lhs.op == "="


def test_parser_binders():
    n = parse(tokenize("∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x| < ε"))
    assert isinstance(n, Quant) and n.binders[0].bound_op == ">"


def test_parser_falls_back_to_raw():
    assert isinstance(parse(tokenize("foo := by simp")), Raw)


def test_decls_skip_proofs_and_track_variables():
    src = """
variable {G : Type*} [Group G]

/-- doc -/
theorem mul_inv (a : G) : a * a⁻¹ = 1 := by
  simp
  -- theorem fake : False := sorry  (inside a proof, must be ignored)

lemma two : 1 + 1 = 2 := rfl
"""
    ds = extract_decls(src)
    assert [d.name for d in ds] == ["mul_inv", "two"]
    assert ds[0].doc == "doc"
    assert len(ds[0].context) == 2


def test_gloss_group_theorem():
    g = one("variable {G : Type*} [Group G]\ntheorem mul_inv (a : G) : a * a⁻¹ = 1 := sorry")
    assert g == "Theorem mul_inv: Let G be a group. Let a be an element of G. Then a * a⁻¹ equals 1."


def test_gloss_hypotheses_predicates_negation():
    g = one("theorem foo {n : ℕ} (h : 2 ≤ n) (np : ¬Nat.Prime n) : ∃ m, m ∣ n ∧ m ≠ 1 := sorry")
    assert "Let n be a natural number." in g
    assert "Assume that 2 is at most n and that n is not prime." in g
    assert "there exists m such that m divides n and m does not equal 1" in g


def test_gloss_quantifiers_with_types():
    g = one("theorem t : ∀ a b : ℕ, a + b = b + a := sorry")
    assert "for all natural numbers a and b, a + b equals b + a" in g


def test_gloss_dot_notation():
    g = one("theorem t {p : ℕ} (hp : p.Prime) : 2 ≤ p := sorry")
    assert "Assume that p is prime." in g


def test_gloss_big_operators():
    g = one("theorem t (n : ℕ) : ∑ i ∈ Finset.range n, i = n * (n - 1) / 2 := sorry")
    assert "the sum of i over i ∈ {0, …, n - 1}" in g


def test_gloss_definition_body():
    g = one("def double (n : ℕ) : ℕ := 2 * n")
    assert g == "Definition double: Let n be a natural number. Define double(n) := 2 * n, which is a natural number."


def test_gloss_class():
    g = one("class Foo (α : Type*) extends Bar α where\n  op : α → α\n  op_self : ∀ a, op (op a) = a")
    assert "is a bar, together with op, a function from α to α" in g
    assert "the property (op_self) that for every a, op(op(a)) equals a" in g


def test_unused_section_variables_are_dropped():
    g = one("variable {α β : Type*} [Group α] [Ring β]\ntheorem t (a : α) : a = a := sorry")
    assert "β" not in g and "ring" not in g


def test_name_gloss():
    assert gloss_name("Nat.mul_le_mul_of_nonneg_left") == "Nat: mul ≤ mul — given nonnegative left"


def test_riemann_hypothesis():
    g = one(
        "def RiemannHypothesis : Prop :=\n"
        "  ∀ (s : ℂ) (_ : riemannZeta s = 0) (_ : ¬∃ n : ℕ, s = -2 * (n + 1)) (_ : s ≠ 1), s.re = 1 / 2\n"
    )
    assert g == (
        "Definition RiemannHypothesis: RiemannHypothesis holds when for every complex number s, "
        "if ζ(s) equals 0, there is no natural number n such that s equals -2 * (n + 1) and "
        "s does not equal 1, then the real part of s equals 1 / 2."
    )


def _masked(src: str):
    from lean_translator.gloss import gloss_decl

    (d,) = extract_decls(src)
    return gloss_decl(d)


def test_masking_hides_compound_terms_only():
    t = _masked("theorem t (n : ℕ) : ∑ i ∈ Finset.range n, i = n * (n - 1) / 2 := sorry")
    assert t.masked == "Theorem t: Let n be a natural number. Then [T0] equals [T1]."
    assert t.slots == ["(the sum of i over i ∈ {0, …, n - 1})", "n * (n - 1) / 2"]


def test_masking_round_trips_exactly():
    from lean_translator.gloss import slot_check, unmask

    t = _masked(
        "def RiemannHypothesis : Prop :=\n"
        "  ∀ (s : ℂ) (_ : riemannZeta s = 0) (_ : ¬∃ n : ℕ, s = -2 * (n + 1)) (_ : s ≠ 1), s.re = 1 / 2\n"
    )
    assert t.slots == ["ζ(s)", "-2 * (n + 1)", "the real part of s", "1 / 2"]
    assert slot_check(t.masked, len(t.slots))
    assert unmask(t.masked, t.slots) == t.gloss


def test_slot_check_rejects_dropped_or_duplicated_slots():
    from lean_translator.gloss import slot_check

    assert slot_check("[T1] then [T0]", 2)  # reordering is fine
    assert not slot_check("[T0] only", 2)  # dropped
    assert not slot_check("[T0] [T0] [T1]", 2)  # duplicated
    assert not slot_check("[T0] [T1] [T2]", 2)  # invented


def test_proof_narration_keeps_structure():
    from lean_translator.decls import extract_decls
    from lean_translator.proofs import narrate

    src = """theorem t (x : ℝ) : |x| ≥ 0 ∨ x < 0 := by
  by_cases hx : 0 ≤ x
  · left
    simp [abs_nonneg]
  · right
    linarith
"""
    (d,) = extract_decls(src)
    steps = narrate(d)
    assert steps[0] == (0, "Consider separately whether 0 is at most x or not.")
    assert (1, "Simplify using `abs_nonneg`.") in steps
    assert (1, "By linear arithmetic.") in steps


def test_latex_formulas():
    from lean_translator.latex import tex_math
    from lean_translator.lexer import tokenize
    from lean_translator.parser import parse

    def t(s):
        return tex_math(parse(tokenize(s)))

    assert t("a / b") == r"\frac{a}{b}"
    assert t("x ^ 2 ≤ ‖y‖") == r"{x}^{2} \le \left\lVert y \right\rVert"
    assert r"\sum_{i \in s}" in t("∑ i ∈ s, f i")
    assert t("Real.sqrt x") == r"\sqrt{x}"


def test_polish_keeps_math_and_rejects_tampering():
    from lean_translator.polish import Polisher, acceptable, mask, unmask

    marked = "Let \x01n\x02 be a natural number. Assume that 2 is at most \x01n\x02. Then \x01n \\le n^2\x02."
    masked, table = mask(marked)
    assert masked == "Let [M1] be a natural number. Assume that 2 is at most [M2]. Then [M3]."
    assert unmask(masked, table) == marked

    good = Polisher("fake", chat=lambda s, u: "For every natural number [M1] with 2 at most [M2], we have [M3].")
    out = good.statement(marked)
    assert out == "For every natural number \x01n\x02 with 2 at most \x01n\x02, we have \x01n \\le n^2\x02."
    dropped = Polisher("fake", chat=lambda s, u: "For every natural number [M1], we have [M3].")
    assert dropped.statement(marked) is None and dropped.stats["rejected"] == 1
    invented = Polisher("fake", chat=lambda s, u: "Here is [M1] [M2] [M3] [M4].")
    assert invented.statement(marked) is None
    assert not acceptable(masked, "Let $n$ be [M1] [M2] [M3]", table)  # raw math smuggled in
