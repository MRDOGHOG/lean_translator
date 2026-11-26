import Slop.Basic

open Nat

-- slop: a helper lemma the final proof doesn't need
theorem helper (n : Nat) : n + 0 = n := by simp

theorem Pt.add_x_comm (p q : Pt) : (p + q).x = (q + p).x := by
  have unused : 1 + 1 = 2 := rfl
  simp [Nat.add_comm, helper]

example : 2 + 2 = 4 := rfl
#check Pt.add_x_comm
