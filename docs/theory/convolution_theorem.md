# Convolution Theorem — Complete Derivation

**Author:** Navin C Chacko
**Context:** Why FNO uses FFT for kernel integral evaluation
**Reference:** Li et al. 2021 (arXiv 2010.08895), Equation 3 → Equation 4

---

## 1. The Kernel Integral (What We Want To Compute)

The general neural operator kernel integral is:

```
(K(φ) v_t)(x) = ∫_D κ_φ(x, y) · v_t(y) dy
```

For n grid points, naive discrete approximation:

```
(K v_t)(xᵢ) ≈ Σⱼ κ(xᵢ, xⱼ) · v_t(xⱼ)
```

This is O(n²) — for n=1024: 1,048,576 operations per channel.
Too slow for training with N=1000 samples × 500 epochs.

---

## 2. The Convolution Simplification

FNO restricts κ to be **translation invariant**:

```
κ(x, y) = κ(x − y)   (depends only on distance, not position)
```

This is a modelling choice — not derived from physics.
It means the kernel integral becomes a **convolution**:

```
(K v_t)(x) = ∫_D κ(x − y) · v_t(y) dy  =  (κ * v_t)(x)
```

---

## 3. The Convolution Theorem

**Statement:**
```
F(f * g)(k) = F(f)(k) · F(g)(k)
```

Convolution in physical space = pointwise multiplication in Fourier space.

**Proof:**
```
F(f * g)(k) = ∫ (f * g)(x) · e^{-2πikx} dx

            = ∫ [∫ f(y) · g(x−y) dy] · e^{-2πikx} dx

Swap order of integration:
            = ∫ f(y) [∫ g(x−y) · e^{-2πikx} dx] dy

Let z = x−y (so x = z+y, dx = dz):
            = ∫ f(y) [∫ g(z) · e^{-2πik(z+y)} dz] dy

            = ∫ f(y) · e^{-2πiky} dy · ∫ g(z) · e^{-2πikz} dz

            = F(f)(k) · F(g)(k)   ∎
```

---

## 4. Applying The Theorem to FNO

Define R(k) = F(κ)(k) — the Fourier transform of the kernel.

Then by the convolution theorem:

```
F(κ * v_t)(k) = F(κ)(k) · F(v_t)(k)
              = R(k) · F(v_t)(k)
```

Taking the inverse Fourier transform:

```
(K v_t)(x) = κ * v_t(x) = F⁻¹( R · F(v_t) )(x)
```

**This is Equation 4 of the paper.**

---

## 5. Complexity Reduction

```
Naive kernel integral:   O(n²)    per channel
After convolution theorem:

  FFT forward:  O(n log n)
  Multiply R:   O(k_max)    ← only k_max modes kept
  FFT inverse:  O(n log n)

Total: O(n log n) instead of O(n²)

At n=1024, k_max=16:
  Naive: 1,048,576 operations
  FNO:       10,240 operations
  Speedup:      102×
```

---

## 6. What R Represents

R_φ ∈ ℂ^{k_max × dv × dv} is the learnable Fourier transform
of the kernel. Shape: (k_max, dv, dv) complex.

For each Fourier mode k:
- R[k] is a dv×dv complex matrix
- It mixes all dv input channels → dv output channels
- Equation 5: (R · F̂v_t)_{k,l} = Σⱼ R_{k,l,j} · (F̂v_t)_{k,j}

In PyTorch (one-liner):
```python
out = torch.einsum("bki,koi->bko", ft_truncated, R_complex)
# b=batch, k=mode, i=input channel, o=output channel
```

---

## 7. The Complete Forward Pass Shape Trace

```
Input v_t:          (batch, n, dv)    real
rfft(dim=1):        (batch, n//2+1, dv)   complex
truncate:           (batch, k_max, dv)    complex
R multiply:         (batch, k_max, dv)    complex
zero-pad:           (batch, n//2+1, dv)   complex
irfft(n=n, dim=1):  (batch, n, dv)    real
```

The `dim=1` argument is critical — without it, PyTorch applies
FFT along the last dimension (dv) instead of the spatial dimension (n).
