# FNO Mathematical Derivations

**Author:** Navin C Chacko
**Started:** April 2026
**Reference:** Li et al. 2021 — arXiv 2010.08895

These are my personal derivations written while reading the FNO paper.
Each derivation is worked from scratch — not copied from any source.

---

## D1 — Fourier Basis Orthogonality

**Why this matters:** The Fourier coefficient formula works because
different modes are orthogonal. This justifies why R[k] can learn
mode k independently without contamination from other modes.

**Claim:**
∫₀¹ cos(2πkx)·cos(2πlx) dx = 0 when k ≠ l, = 1/2 when k = l

**Proof:**
Using product-to-sum: cos(A)cos(B) = ½[cos(A−B) + cos(A+B)]

For k ≠ l:
∫₀¹ cos(2πkx)cos(2πlx) dx
= ½ ∫₀¹ [cos(2π(k−l)x) + cos(2π(k+l)x)] dx
= ½ [sin(2π(k−l)x)/(2π(k−l)) + sin(2π(k+l)x)/(2π(k+l))]₀¹
= 0  (since sin(2πm) = 0 for any integer m ≠ 0)

For k = l:
∫₀¹ cos²(2πkx) dx = ½ ∫₀¹ [1 + cos(4πkx)] dx = ½

**Intuition:** Each Fourier mode is a basis vector perpendicular
to all others. Learning R[k] only affects mode k.

**Connection to FNO code:**
In SpectralConv1D, R has shape (k_max, dv, dv).
R[k] operates on mode k completely independently of R[l] for l≠k.
This independence is valid because the Fourier basis is orthogonal.

---

## D2 — Differentiation Becomes Multiplication

*(To be completed — notebook derivation done Apr 2026)*

---

## D3 — Convolution Theorem

*(To be completed)*

---

## D4 — FFT Complexity O(n log n)

*(To be completed)*

---

## D5 — FNO Equation 4 From Equation 3

*(To be completed)*
