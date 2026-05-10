# FNO Mathematical Derivations

**Author:** Navin C Chacko
**Reference:** Li et al. 2021 — Fourier Neural Operator (arXiv 2010.08895)
**Goal:** Derive every equation in the paper from first principles

---

## D1 — From Universal Approximation to Operator Learning

**Classical neural networks** approximate functions:
```
G: ℝⁿ → ℝᵐ   (finite-dimensional to finite-dimensional)
```

**Neural operators** approximate operators:
```
G†: A → U   (function space to function space)
             (infinite-dimensional to infinite-dimensional)
```

A function space contains infinite objects — every continuous function
on [0,1] is one element. The goal is to learn the mapping between spaces.

**Why this matters for PDEs:**
A PDE like Burgers' equation defines a mapping:
```
G†: {initial conditions} → {solutions at t=1}
    u₀(x)               ↦  u(x,1)
```

This mapping is infinite-dimensional. Classical ML cannot learn it
directly — it would need to retrain for every new resolution.

---

## D2 — The General Neural Operator (Equation 3)

```
v_{t+1}(x) = σ( W·v_t(x) + (K(a; φ) v_t)(x) )
```

Where the kernel integral operator K is:

```
(K(a;φ) v_t)(x) = ∫_D κ_φ(x, y, a(x), a(y)) · v_t(y) dy
```

**Four arguments of κ:**
- x: query point (where we want output)
- y: source point (where information comes from)
- a(x): input function value at query point
- a(y): input function value at source point

The full kernel is input-dependent: different a(x) → different κ.

---

## D3 — FNO Simplification to Convolution (Equation 4)

FNO makes two simplifications:

**Simplification 1 — Remove input dependence:**
```
κ(x, y, a(x), a(y)) → κ(x, y)
```
The kernel no longer depends on the input function values.
This is a modelling choice — not physics.

**Simplification 2 — Translation invariance:**
```
κ(x, y) → κ(x − y)
```
The kernel depends only on the distance between points, not their position.

**Result:** The kernel integral becomes a convolution:
```
(K v_t)(x) = ∫_D κ(x−y) · v_t(y) dy = (κ * v_t)(x)
```

Applying the convolution theorem (see convolution_theorem.md):
```
(K v_t)(x) = F⁻¹( R_φ · F(v_t) )(x)   ← Equation 4
```

Where R_φ = F(κ) is the learned Fourier transform of the kernel.

---

## D4 — Equation 5: The Mode Multiplication

For the discrete case with k_max modes:

```
(R · F̂v_t)_{k,l} = Σⱼ₌₁^dv R_{k,l,j} · (F̂v_t)_{k,j}

Indices:
  k = Fourier mode index (k = 1,...,k_max)
  l = output channel index (l = 1,...,dv)
  j = input channel index (j = 1,...,dv)
```

This is k_max separate dv×dv matrix-vector multiplications.
For each mode k, R[k] mixes all dv input channels → dv output channels.

**PyTorch implementation:**
```python
# R stored as (k_max, dv, dv, 2) real tensor
R_complex = torch.view_as_complex(R.contiguous())
# R_complex shape: (k_max, dv, dv) complex

out = torch.einsum("bki,koi->bko", ft_trunc, R_complex)
# b=batch, k=mode, i=input channel, o=output channel
```

---

## D5 — The Complete FNO Layer (Equation 2)

```
v_{t+1}(x) = σ( W·v_t(x)  +  (K·v_t)(x) )
                  ↑                ↑
              W branch         K branch
              (local)          (global)
```

**W branch — local linear transform:**
- Implemented as Conv1d(dv, dv, kernel_size=1)
- kernel_size=1 means NO spatial communication
- Position xᵢ only uses its own values v_t(xᵢ)
- Equivalent to dv×dv matrix multiply at every point
- Conv1d is 14× faster than nn.Linear in a loop (vectorised)

**Why NOT nn.Linear for W branch:**
Conv1d expects (batch, channels, length) → needs transpose.
But nn.Linear works on last dim → no transpose needed.
For P and Q (lifting/projection), Linear is cleaner.
For W branch (processing all n points simultaneously), Conv1d is faster.

**Transpose pattern for W branch:**
```python
# x shape: (batch, n, dv)
w_out = self.w(x.transpose(1,2)).transpose(1,2)
#             (batch, dv, n) → Conv1d → (batch, dv, n)
#                                    → transpose → (batch, n, dv)
```

**Why σ(a+b) not σ(a)+σ(b):**
σ(a+b) ≠ σ(a)+σ(b). Applying activation after the sum preserves
the interaction between both branches. Applying before would
treat them as independent linear maps, limiting expressiveness.

---

## D6 — The Full FNO Architecture (Figure 2a)

```
a(x) → P → [FourierLayer × T] → Q → u(x)

P: lifting network
   nn.Linear(da, dv)
   Maps input channels da → hidden channels dv
   Applied pointwise: broadcasts over (batch, n)

FourierLayer × T:
   T=4 stacked layers
   Each: (batch, n, dv) → (batch, n, dv)
   Shape unchanged throughout

Q: projection network
   nn.Sequential(Linear(dv,128), ReLU, Linear(128,du))
   Maps hidden channels dv → output channels du
   Two layers for nonlinear projection

Why P uses Linear not Conv1d:
   Linear operates on last dimension automatically
   For (batch, n, da), transforms da → dv
   Broadcasts over (batch, n) — no transpose needed

Why Q has two layers (dv → 128 → du):
   One linear layer: only affine map f(x) = Wx + b
   Two layers + ReLU: can learn nonlinear relationships
   Especially important when du=1 (scalar output)
```

---

## D7 — Discretisation Invariance

**Claim:** R has shape (k_max, dv, dv) — contains NO n.

**Proof that this gives resolution invariance:**

Step 1: Encode (any n_in)
```
FFT(v_t) → keep k_max modes
Result: (k_max, dv) — independent of n_in
```

Step 2: Process (resolution-free)
```
R · F(v_t) — no n anywhere
Same R regardless of input resolution
```

Step 3: Decode (any n_out)
```
iFFT at n_out points
Evaluates Σₖ ûₖ · e^{2πikxⱼ} for j=0,...,n_out-1
Valid for any n_out — e^{2πikx} defined everywhere on ℝ
```

**Zero-shot super-resolution:**
Train at n=64, evaluate at n=256.
Same R, same error, 4× higher resolution output.
No retraining needed.

---

## D8 — Why Uniform Grid Is Required

The FFT requires equally-spaced grid points:
```
xᵢ = i/n   for i = 0,1,...,n-1
```

The Cooley-Tukey even/odd split only works when grid points
are uniformly spaced. Non-uniform grids break the recursion.

**Consequence for SCF problem:**
K-joint ANSYS mesh is NOT uniform (refined near weld toe).
Must interpolate from ANSYS mesh to uniform grid before FNO.
This is a preprocessing step not required by GNN/GINO.
