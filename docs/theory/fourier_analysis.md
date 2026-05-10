# Fourier Analysis — Foundation Notes

**Author:** Navin C Chacko
**Context:** M.Tech AI/ML BITS Pilani WILP — neuralforge learning laboratory
**Reference:** Li et al. 2021 (arXiv 2010.08895)

---

## 1. What Is a Fourier Series?

Any periodic function f(x) defined on [0,1] can be written as an
infinite sum of sine and cosine waves:

```
f(x) = a₀/2 + Σₖ₌₁^∞ [ aₖ cos(2πkx) + bₖ sin(2πkx) ]
```

Each term is a **basis function** — a pure oscillation at frequency k.
The coefficients aₖ and bₖ tell you how much of each frequency is present.

Using complex exponentials (Euler's formula: e^{iθ} = cos θ + i sin θ):

```
f(x) = Σₖ₌₋∞^∞ ĉₖ · e^{2πikx}

where ĉₖ = ∫₀¹ f(x) · e^{-2πikx} dx   (Fourier coefficient)
```

---

## 2. Basis Orthogonality — Why Fourier Works

**Claim:**
```
∫₀¹ cos(2πkx) · cos(2πlx) dx = 0    when k ≠ l
                               = 1/2   when k = l
```

**Proof (using product-to-sum identity):**
```
cos(A)cos(B) = ½[cos(A−B) + cos(A+B)]

For k ≠ l:
∫₀¹ cos(2πkx)cos(2πlx) dx
= ½ ∫₀¹ [cos(2π(k−l)x) + cos(2π(k+l)x)] dx
= ½ [sin(2π(k−l)x)/(2π(k−l)) + ...]₀¹
= 0   (since sin(2πm) = 0 for any integer m ≠ 0)
```

**Why this matters for FNO:**
Each Fourier mode is a basis vector perpendicular to all others.
R[k] operates on mode k completely independently of R[l] for l ≠ k.
This independence is valid because the Fourier basis is orthogonal.

---

## 3. Differentiation Becomes Multiplication

**Claim:** In Fourier space, differentiation = multiplication by 2πik.

**Proof:**
```
Let f(x) = Σₖ ĉₖ · e^{2πikx}

df/dx = Σₖ ĉₖ · d/dx[e^{2πikx}]
      = Σₖ ĉₖ · (2πik) · e^{2πikx}
      = Σₖ (2πik · ĉₖ) · e^{2πikx}
```

The Fourier coefficient of df/dx is (2πik) times the coefficient of f.
Differentiation = multiply each mode by 2πik.

**Why this matters:**
PDEs containing ∂u/∂x or ∂²u/∂x² become algebraic equations
in Fourier space. No finite difference approximation needed.
This is why Fourier methods are exact for smooth periodic problems.

---

## 4. The Discrete Fourier Transform (DFT)

For a function stored at n equally-spaced grid points:

```
(F̂f)ₗ(k) = Σₓ₌₀^{n-1} fₗ(x) · e^{-2πi·xk/n}

(F̂⁻¹f)ₗ(x) = Σₖ₌₀^{n-1} fₗ(k) · e^{+2πi·xk/n}
```

- k = 0,1,...,n-1 are the frequency mode indices
- x = 0,1,...,n-1 are the grid point indices
- l = 1,...,dv are the channel indices

**Naive DFT complexity: O(n²)**
For each of n output modes, sum over n inputs = n² operations.

---

## 5. The Fast Fourier Transform (FFT) — O(n log n)

The FFT uses divide-and-conquer:

```
DFT of f[0,1,2,...,n-1]
= DFT of f[0,2,4,...,n-2]   (even-indexed, size n/2)
+ DFT of f[1,3,5,...,n-1]   (odd-indexed, size n/2)
```

**Recurrence relation:**
```
T(n) = 2·T(n/2) + n

Solving (Master theorem): T(n) = O(n log₂ n)
```

**Why log n appears:**
- Recursion depth = log₂(n) levels
- Each level costs O(n) to combine sub-results
- Total = O(n) × log₂(n) = O(n log n)

**Requirement:** n must be a power of 2 (Cooley-Tukey algorithm).

**Speedup at n=1024:**
```
Naive DFT: 1,048,576 operations
FFT:           10,240 operations
Speedup:          102×
```

---

## 6. rfft vs fft — Why FNO Uses rfft

For a **real-valued** input of length n:
- Full FFT: n complex outputs (redundant — negative frequencies are conjugate)
- rfft: n//2+1 complex outputs (exploits conjugate symmetry)

```python
x = torch.randn(batch, n, dv)      # real input
x_ft = torch.fft.rfft(x, dim=1)   # shape: (batch, n//2+1, dv)
# NOT n outputs — only n//2+1 (roughly half)
```

This halves the computation and memory for the FFT step.
Always specify `dim=1` explicitly — default is last dim, which is wrong
for FNO's (batch, n, dv) tensor convention.

---

## 7. Mode Truncation — The Key FNO Operation

After FFT: F̂(v_t) has n//2+1 complex modes.
FNO keeps only the k_max lowest frequency modes:

```
F̂(v_t)[:, :k_max, :]   — keep low frequencies (global structure)
F̂(v_t)[:, k_max:, :]   — discard high frequencies (fine details)
```

**Why truncation works:**
Physical PDE solutions are smooth — energy concentrated in low modes.
Burgers at t=1: smooth enough that k_max=16 modes capture 99%+ energy.

**Nyquist condition:** Need n ≥ 2·k_max to resolve all modes.
For k_max=16: need n ≥ 32 minimum. Paper uses n=1024.

---

## 8. Resolution Invariance via Fourier Basis

The Fourier basis function φₖ(x) = e^{2πikx} is defined at EVERY
real-valued x ∈ ℝ — not just at grid points.

To evaluate FNO output u(x*) at any arbitrary point x*:
```
u(x*) = Σₖ ûₖ · e^{2πik·x*}
       = Σₖ ûₖ · [cos(2πkx*) + i·sin(2πkx*)]
```

This is just arithmetic — computable at any x*, on any grid.

**Consequence:** Same trained R works at any resolution n.
Train at n=64, evaluate at n=4096 — same weights, same error.
