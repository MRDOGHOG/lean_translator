/-- A tiny number type. -/
structure Pt where
  x : Nat
  y : Nat

-- used invisibly: `p + q` in the target elaborates through this instance
instance : Add Pt := ⟨fun p q => ⟨p.x + q.x, p.y + q.y⟩⟩

-- used invisibly: `simp` picks it up from the simp set
@[simp] theorem Pt.add_x (p q : Pt) : (p + q).x = p.x + q.x := rfl

-- slop: nothing uses it
def Pt.norm1 (p : Pt) : Nat := p.x + p.y

-- slop: only used by other slop
theorem Pt.norm1_zero : Pt.norm1 ⟨0, 0⟩ = 0 := rfl
theorem Pt.norm1_zero' : Pt.norm1 ⟨0, 0⟩ = 0 ∧ True := ⟨Pt.norm1_zero, trivial⟩

#eval Pt.norm1 ⟨1, 2⟩
