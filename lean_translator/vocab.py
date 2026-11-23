"""Hand-written vocabulary used by the glosser.

These tables are the "dictionary" layer of the rule-based translator.
They are deliberately plain data so they can be extended without touching
the translation logic.
"""

from __future__ import annotations

import re

# --- types ---------------------------------------------------------------
# name -> (singular, plural)
TYPE_NOUNS: dict[str, tuple[str, str]] = {
    "ℕ": ("natural number", "natural numbers"),
    "Nat": ("natural number", "natural numbers"),
    "ℤ": ("integer", "integers"),
    "Int": ("integer", "integers"),
    "ℚ": ("rational number", "rational numbers"),
    "Rat": ("rational number", "rational numbers"),
    "ℝ": ("real number", "real numbers"),
    "Real": ("real number", "real numbers"),
    "ℂ": ("complex number", "complex numbers"),
    "Complex": ("complex number", "complex numbers"),
    "ℝ≥0": ("nonnegative real", "nonnegative reals"),
    "NNReal": ("nonnegative real", "nonnegative reals"),
    "ℝ≥0∞": ("extended nonnegative real", "extended nonnegative reals"),
    "ENNReal": ("extended nonnegative real", "extended nonnegative reals"),
    "ℕ+": ("positive natural number", "positive natural numbers"),
    "PNat": ("positive natural number", "positive natural numbers"),
    "Prop": ("proposition", "propositions"),
    "Bool": ("boolean", "booleans"),
    "Type": ("type", "types"),
    "Type*": ("type", "types"),
    "Sort": ("sort", "sorts"),
    "Sort*": ("sort", "sorts"),
    "String": ("string", "strings"),
    "Char": ("character", "characters"),
    "Unit": ("unit value", "unit values"),
    "EReal": ("extended real", "extended reals"),
    "ℍ": ("quaternion", "quaternions"),
}

# Unary type constructors: name -> template using {0} for the argument
# (already rendered as a plural noun phrase or a name).
TYPE_CTORS: dict[str, str] = {
    "Set": "set of {0}",
    "Finset": "finite set of {0}",
    "List": "list of {0}",
    "Multiset": "multiset of {0}",
    "Array": "array of {0}",
    "Option": "optional {0}",
    "Polynomial": "polynomial over {0}",
    "Polynomial.Polynomial": "polynomial over {0}",
    "Fin": "index less than {0}",
    "ZMod": "integer modulo {0}",
    "Subgroup": "subgroup of {0}",
    "AddSubgroup": "additive subgroup of {0}",
    "Submonoid": "submonoid of {0}",
    "Subring": "subring of {0}",
    "Subfield": "subfield of {0}",
    "Ideal": "ideal of {0}",
    "Filter": "filter on {0}",
    "Equiv.Perm": "permutation of {0}",
    "Perm": "permutation of {0}",
    "MvPolynomial": "multivariate polynomial",
    "PowerSeries": "power series over {0}",
    "Matrix": "matrix",
    "Measure": "measure on {0}",
    "MeasureTheory.Measure": "measure on {0}",
    "Units": "unit of {0}",
    "Sylow": "Sylow subgroup",
    "TopologicalSpace": "topology on {0}",
    "Opens": "open set of {0}",
    "Seq": "sequence of {0}",
    "Stream'": "stream of {0}",
    "Sym2": "unordered pair of {0}",
    "SimpleGraph": "simple graph on {0}",
    "Cardinal": "cardinal",
    "Ordinal": "ordinal",
}

# Binary type operators.
TYPE_ARROWS: dict[str, str] = {
    "→": "function from {0} to {1}",
    "->": "function from {0} to {1}",
    "→+*": "ring homomorphism from {0} to {1}",
    "→*": "monoid homomorphism from {0} to {1}",
    "→+": "additive monoid homomorphism from {0} to {1}",
    "≃": "equivalence between {0} and {1}",
    "≃+*": "ring isomorphism between {0} and {1}",
    "≃*": "multiplicative isomorphism between {0} and {1}",
    "≃+": "additive isomorphism between {0} and {1}",
    "→ₗ": "linear map from {0} to {1}",
    "≃ₗ": "linear equivalence between {0} and {1}",
    "→o": "order-preserving map from {0} to {1}",
    "≃o": "order isomorphism between {0} and {1}",
    "↪": "embedding of {0} into {1}",
    "×": "pair of an element of {0} and an element of {1}",
    "⟶": "morphism from {0} to {1}",
    "⥤": "functor from {0} to {1}",
}

# --- type classes --------------------------------------------------------
# Rendered as "<carrier> is a/an <phrase>".  Multi-parameter classes use a
# template over the arguments.
CLASS_TEMPLATES: dict[str, str] = {
    "Module": "{1} is a module over {0}",
    "Algebra": "{1} is an algebra over {0}",
    "NormedSpace": "{1} is a normed vector space over {0}",
    "InnerProductSpace": "{1} is an inner product space over {0}",
    "MulAction": "{0} acts on {1}",
    "AddAction": "{0} acts additively on {1}",
    "DistribMulAction": "{0} acts distributively on {1}",
    "SMul": "{0} has a scalar action on {1}",
    "Pow": "{0} can be raised to powers in {1}",
    "IsScalarTower": "the actions of {0}, {1} and {2} are compatible",
    "CharP": "{0} has characteristic {1}",
    "Fact": "{0}",
    "Membership": "{1} has a membership relation for {0}",
    "Coe": "{0} coerces to {1}",
    "FunLike": "{0} is a type of functions from {1} to {2}",
    "IsDomain": "{0} is an integral domain",
    "Field": "{0} is a field",
    "Fintype": "{0} is finite",
    "Finite": "{0} is finite",
    "Infinite": "{0} is infinite",
    "Nonempty": "{0} is nonempty",
    "Countable": "{0} is countable",
    "Inhabited": "{0} is inhabited",
    "Subsingleton": "{0} has at most one element",
    "Nontrivial": "{0} has at least two elements",
    "NeZero": "{0} is nonzero",
    "Invertible": "{0} is invertible",
    "MeasurableSpace": "{0} is a measurable space",
    "Category": "{0} is a category",
    "T2Space": "{0} is a Hausdorff space",
    "CompactSpace": "{0} is compact",
}

# Instance arguments that add nothing for a human reader.
SILENT_CLASSES = {
    "Decidable", "DecidableEq", "DecidablePred", "DecidableRel", "Inhabited",
    "BEq", "LawfulBEq", "Hashable", "Repr", "ToString", "Lean.ToExpr",
    "DecidableLE", "DecidableLT", "Encodable", "Primcodable", "MeasurableSingletonClass",
    "BorelSpace", "OpensMeasurableSpace", "SecondCountableTopology", "IsCountablyGenerated",
    "Std.Associative", "Std.Commutative", "Fintype.Decidable",
}

# CamelCase fragments with special readings.
CAMEL_WORDS: dict[str, str] = {
    "Comm": "commutative",
    "Add": "additive",
    "Mul": "multiplicative",
    "Div": "division",
    "Inv": "inverse",
    "Ordered": "ordered",
    "Normed": "normed",
    "Topological": "topological",
    "Lin": "linear",
    "Semi": "semi",
    "Zero": "zero",
    "One": "one",
    "Neg": "negation",
    "Sub": "sub",
    "Is": "",
    "Has": "has",
    "Conditionally": "conditionally",
    "Complete": "complete",
    "Cancel": "cancellative",
    "Monoid": "monoid",
    "With": "with",
}

# --- predicates ----------------------------------------------------------
# Applied predicate -> template ({0}, {1} ... are the rendered arguments).
PREDICATES: dict[str, str] = {
    "Prime": "{0} is prime",
    "Nat.Prime": "{0} is prime",
    "Irreducible": "{0} is irreducible",
    "Squarefree": "{0} is squarefree",
    "Even": "{0} is even",
    "Odd": "{0} is odd",
    "IsUnit": "{0} is a unit",
    "IsSquare": "{0} is a perfect square",
    "Nat.Coprime": "{0} and {1} are coprime",
    "Coprime": "{0} and {1} are coprime",
    "IsCoprime": "{0} and {1} are coprime",
    "Injective": "{0} is injective",
    "Function.Injective": "{0} is injective",
    "Surjective": "{0} is surjective",
    "Function.Surjective": "{0} is surjective",
    "Bijective": "{0} is bijective",
    "Function.Bijective": "{0} is bijective",
    "Monotone": "{0} is monotone",
    "Antitone": "{0} is antitone",
    "StrictMono": "{0} is strictly increasing",
    "StrictAnti": "{0} is strictly decreasing",
    "MonotoneOn": "{0} is monotone on {1}",
    "StrictMonoOn": "{0} is strictly increasing on {1}",
    "Continuous": "{0} is continuous",
    "ContinuousOn": "{0} is continuous on {1}",
    "ContinuousAt": "{0} is continuous at {1}",
    "UniformContinuous": "{0} is uniformly continuous",
    "Differentiable": "{1} is differentiable (over {0})",
    "DifferentiableAt": "{1} is differentiable at {2}",
    "DifferentiableOn": "{1} is differentiable on {2}",
    "HasDerivAt": "{0} has derivative {1} at {2}",
    "Measurable": "{0} is measurable",
    "Integrable": "{0} is integrable",
    "IntervalIntegrable": "{0} is integrable on the interval from {2} to {3}",
    "Summable": "{0} is summable",
    "HasSum": "{0} sums to {1}",
    "IsOpen": "{0} is open",
    "IsClosed": "{0} is closed",
    "IsCompact": "{0} is compact",
    "IsConnected": "{0} is connected",
    "Dense": "{0} is dense",
    "Set.Nonempty": "{0} is nonempty",
    "Finset.Nonempty": "{0} is nonempty",
    "Nonempty": "{0} is nonempty",
    "Set.Finite": "{0} is finite",
    "Set.Infinite": "{0} is infinite",
    "Set.Countable": "{0} is countable",
    "Disjoint": "{0} and {1} are disjoint",
    "Commute": "{0} and {1} commute",
    "IsLUB": "{1} is a least upper bound of {0}",
    "IsGLB": "{1} is a greatest lower bound of {0}",
    "IsLeast": "{1} is the least element of {0}",
    "IsGreatest": "{1} is the greatest element of {0}",
    "BddAbove": "{0} is bounded above",
    "BddBelow": "{0} is bounded below",
    "Filter.Tendsto": "{0} tends to {2} along {1}",
    "Tendsto": "{0} tends to {2} along {1}",
    "Nat.ModEq": "{1} ≡ {2} (mod {0})",
    "Int.ModEq": "{1} ≡ {2} (mod {0})",
    "Periodic": "{0} is periodic with period {1}",
    "Function.Periodic": "{0} is periodic with period {1}",
    "IsMaximal": "{0} is a maximal ideal",
    "Ideal.IsPrime": "{0} is a prime ideal",
    "IsPrime": "{0} is prime",
    "Normal": "{0} is a normal subgroup",
    "Subgroup.Normal": "{0} is a normal subgroup",
    "ConvexOn": "{2} is convex on {1}",
    "ConcaveOn": "{2} is concave on {1}",
    "Convex": "{1} is convex",
    "Pairwise": "{0} holds pairwise",
    "Set.InjOn": "{0} is injective on {1}",
    "InjOn": "{0} is injective on {1}",
    "Set.SurjOn": "{0} maps {1} onto {2}",
    "Set.MapsTo": "{0} maps {1} into {2}",
    "Symmetric": "{0} is symmetric",
    "Reflexive": "{0} is reflexive",
    "Transitive": "{0} is transitive",
    "Equivalence": "{0} is an equivalence relation",
    "WellFounded": "{0} is well-founded",
    "IsEmpty": "{0} is empty",
    "Algebra.IsAlgebraic": "{1} is algebraic over {0}",
    "IsAlgebraic": "{1} is algebraic over {0}",
    "IsIntegral": "{1} is integral over {0}",
    "Irrational": "{0} is irrational",
    "Transcendental": "{1} is transcendental over {0}",
    "Polynomial.Monic": "{0} is monic",
    "Monic": "{0} is monic",
    "Splits": "{1} splits under {0}",
}

# Applied functions with readable names: name -> template.
FUNCTIONS: dict[str, str] = {
    "Finset.card": "#{0}",
    "Fintype.card": "|{0}|",
    "Nat.card": "|{0}|",
    "Set.ncard": "#{0}",
    "Cardinal.mk": "|{0}|",
    "Nat.gcd": "gcd({0}, {1})",
    "gcd": "gcd({0}, {1})",
    "Nat.lcm": "lcm({0}, {1})",
    "lcm": "lcm({0}, {1})",
    "Real.sqrt": "√{0}",
    "Nat.sqrt": "⌊√{0}⌋",
    "Real.exp": "exp({0})",
    "Complex.exp": "exp({0})",
    "Real.log": "log({0})",
    "Complex.log": "log({0})",
    "Real.sin": "sin({0})",
    "Real.cos": "cos({0})",
    "Real.tan": "tan({0})",
    "Real.pi": "π",
    "Nat.factorial": "{0}!",
    "Nat.iterate": "{0}^[{1}]",
    "Nat.choose": "C({0}, {1})",
    "Nat.succ": "{0} + 1",
    "Nat.pred": "{0} - 1",
    "Nat.totient": "φ({0})",
    "deriv": "{0}′",
    "derivWithin": "{0}′ (within {1})",
    "Finset.range": "{{0, …, {0} - 1}}",
    "Finset.Icc": "[{0}, {1}]",
    "Set.Icc": "[{0}, {1}]",
    "Set.Ico": "[{0}, {1})",
    "Set.Ioc": "({0}, {1}]",
    "Set.Ioo": "({0}, {1})",
    "Set.Ici": "[{0}, ∞)",
    "Set.Ioi": "({0}, ∞)",
    "Set.Iic": "(-∞, {0}]",
    "Set.Iio": "(-∞, {0})",
    "Set.univ": "the whole space",
    "Set.range": "the range of {0}",
    "Set.image": "{0}({1})",
    "Set.preimage": "{0}⁻¹({1})",
    "Polynomial.eval": "{1}({0})",
    "Polynomial.degree": "deg {0}",
    "Polynomial.natDegree": "deg {0}",
    "Polynomial.derivative": "{0}′",
    "Polynomial.roots": "the roots of {0}",
    "Polynomial.X": "X",
    "orderOf": "the order of {0}",
    "Nat.digits": "the base-{0} digits of {1}",
    "Int.natAbs": "|{0}|",
    "abs": "|{0}|",
    "norm": "‖{0}‖",
    "dist": "dist({0}, {1})",
    "max": "max({0}, {1})",
    "min": "min({0}, {1})",
    "Prod.fst": "{0}.1",
    "Prod.snd": "{0}.2",
    "id": "the identity function",
    "Nat.floor": "⌊{0}⌋",
    "Int.floor": "⌊{0}⌋",
    "Nat.ceil": "⌈{0}⌉",
    "Int.ceil": "⌈{0}⌉",
    "Int.fract": "frac({0})",
    "closure": "the closure of {0}",
    "interior": "the interior of {0}",
    "frontier": "the boundary of {0}",
    "nhds": "neighborhoods of {0}",
    "𝓝": "neighborhoods of {0}",
    "Filter.atTop": "infinity",
    "atTop": "infinity",
    "Filter.atBot": "negative infinity",
    "atBot": "negative infinity",
    "MeasureTheory.volume": "Lebesgue measure",
    "volume": "Lebesgue measure",
}

# Words used to read Mathlib names like `mul_le_mul_of_nonneg_left`.
NAME_WORDS: dict[str, str] = {
    "add": "add", "mul": "mul", "sub": "sub", "div": "div", "neg": "neg",
    "inv": "inverse", "comm": "commutative", "assoc": "associative",
    "le": "≤", "lt": "<", "ge": "≥", "gt": ">", "eq": "=", "ne": "≠",
    "pos": "positive", "nonneg": "nonnegative", "nonpos": "nonpositive",
    "iff": "iff", "of": "of", "self": "self", "zero": "zero", "one": "one",
    "two": "two", "pow": "power", "sq": "square", "sqrt": "square root",
    "dvd": "divides", "mem": "membership", "subset": "subset", "inter": "intersection",
    "union": "union", "compl": "complement", "card": "cardinality",
    "comp": "composition", "id": "identity", "inj": "injective",
    "surj": "surjective", "bij": "bijective", "symm": "symmetric", "trans": "transitive",
    "refl": "reflexive", "mono": "monotone", "cast": "cast", "coe": "coercion",
    "smul": "scalar multiplication", "nsmul": "ℕ-scalar multiplication",
    "zsmul": "ℤ-scalar multiplication", "ext": "extensionality", "prod": "product",
    "sum": "sum", "deriv": "derivative", "tendsto": "tends to", "nhds": "neighborhood",
    "sup": "supremum", "inf": "infimum", "bot": "bottom", "top": "top", "abs": "absolute value",
    "exp": "exp", "log": "log", "succ": "successor", "pred": "predecessor",
    "lcm": "lcm", "gcd": "gcd", "fst": "first component", "snd": "second component",
    "nat": "ℕ", "int": "ℤ", "rat": "ℚ", "real": "ℝ", "cancel": "cancel", "map": "map",
    "image": "image", "preimage": "preimage", "range": "range", "apply": "apply",
    "injective": "injective", "surjective": "surjective", "continuous": "continuous",
    "measurable": "measurable", "integral": "integral", "max": "max", "min": "min",
    "def": "definition", "succ'": "successor", "irrefl": "irreflexive",
    "antisymm": "antisymmetric", "left": "left", "right": "right", "not": "not",
    "and": "and", "or": "or", "imp": "implies", "forall": "for all", "exists": "exists",
    "empty": "empty", "univ": "universe", "singleton": "singleton", "insert": "insert",
    "erase": "erase", "filter": "filter", "length": "length", "append": "append",
    "cons": "cons", "nil": "nil", "factorial": "factorial", "choose": "choose",
    "floor": "floor", "ceil": "ceiling", "norm": "norm", "dist": "distance",
}

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9']*|[a-z0-9']+")


def camel_words(name: str) -> list[str]:
    """Split ``LinearOrderedCommRing`` into readable lowercase words."""
    out = []
    for part in _CAMEL_RE.findall(name):
        w = CAMEL_WORDS.get(part, part if part.isupper() and len(part) > 1 else part.lower())
        if w:
            out.append(w)
    return out


def article(phrase: str) -> str:
    if not phrase:
        return phrase
    first = phrase.split()[0].lower()
    if first in ("one", "unit", "uniform", "uniformly", "union", "unique", "euclidean"):
        return "a " + phrase
    if first[0] in "aeiou" or first in ("hausdorff",) and False:
        return "an " + phrase
    return "a " + phrase


def pluralize(noun: str) -> str:
    words = noun.split(" ")
    # pluralize the head noun: first word before "of/over/from/on/between/modulo/less"
    head = len(words) - 1
    for k, w in enumerate(words):
        if w in ("of", "over", "from", "on", "between", "modulo", "less", "into", "in"):
            head = k - 1
            break
    w = words[head]
    if w.endswith(("s", "x", "sh", "ch")):
        w += "es"
    elif w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        w = w[:-1] + "ies"
    else:
        w += "s"
    words[head] = w
    return " ".join(words)
