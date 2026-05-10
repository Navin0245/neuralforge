# FNO Architecture Design Notes

**Author:** Navin C Chacko
**Context:** Design decisions and why they were made
**Reference:** Li et al. 2021, Sections 3-4

---

## 1. The Four Dimension Variables

| Variable | Meaning | Who Sets It | Typical Values |
|----------|---------|-------------|----------------|
| batch | Samples processed simultaneously | You (memory constraint) | 8–20 |
| n | Grid points in spatial domain | Problem resolution | 64–8192 |
| da | Input channels per grid point | Problem (fixed) | 1–5 |
| dv | Hidden channels inside FNO | You (hyperparameter) | 32–64 |
| du | Output channels per grid point | Problem (fixed) | 1–2 |

**Key rule:**
```
da and du are fixed by your problem.
dv is the only dimension you freely choose.
dv only exists between P and Q — never in input or output.
```

---

## 2. The Parameterisation of R

Three options tested in the paper:

**Type A — Direct (paper uses this):**
```
R_φ ∈ ℂ^{k_max × dv × dv}
Fixed tensor — same R for all inputs
Parameters: k_max × dv² × 2 real
Performance: BEST
```

**Type B — Linear input-dependent:**
```
R(k, F̂(a)(k)) = R₀(k) + W(k)·F̂(a)(k)
Parameters: k_max × dv² × 2 + k_max × dv³ × 2
Performance: similar to Type A
Reason: W(k) → 0 during training (redundant — input already in v_t)
```

**Type C — Neural network:**
```
R(k, F̂(a)(k)) = NN_φ(k, F̂(a)(k))
Performance: WORST
Reason: k ∈ Z^d is discrete — NN interpolation between modes is wrong
        Mode k=1 and k=2 are independent — no smooth relationship
```

**Why Type A wins:**
The input a is already encoded in v_t through P and W branches.
R does not need to see a directly — it learns a universal spectral kernel.

---

## 3. Why kernel_size=1 for W Branch

**Goal:** Apply dv×dv matrix multiply at every grid point independently.

**Option 1 — Python loop (wrong):**
```python
for i in range(n):
    out[:, i, :] = linear(x[:, i, :])   # slow — Python loop
```

**Option 2 — Conv1d kernel_size=1 (correct):**
```python
out = conv(x.transpose(1,2)).transpose(1,2)   # fast — vectorised
```

Conv1d with kernel_size=1 processes all n positions simultaneously.
Benchmark (n=1024, dv=32): Conv1d is ~14× faster than loop.
Results are mathematically identical.

**Transpose pattern explained:**
```
x shape: (batch, n, dv)        FNO convention — spatial first
transpose(1,2): (batch, dv, n) Conv1d convention — channels first
Conv1d output:  (batch, dv, n)
transpose(1,2): (batch, n, dv) back to FNO convention
```

---

## 4. Why P Uses nn.Linear

**Contrast with W branch:**

W branch uses Conv1d because it needs to process all n positions
simultaneously — Conv1d is vectorised, a loop is not.

P uses nn.Linear because:
```
nn.Linear(da, dv) operates on the LAST dimension automatically.
For input (batch, n, da), it transforms da → dv and
PyTorch broadcasts over (batch, n) — no extra steps.

Using Conv1d for P would require:
  x.transpose(1,2)           ← extra step
  conv(...)                  ← Conv1d
  .transpose(1,2)            ← extra step back
Two unnecessary transposes for no benefit.
```

**Rule:**
- Operations that must process all n points simultaneously: Conv1d
- Operations that just transform the last dimension: Linear

---

## 5. Why Q Has Two Layers

```
Option A — one layer:
  nn.Linear(dv, du)
  f(x) = Wx + b
  Only learns LINEAR relationships between features and output

Option B — two layers (what FNO uses):
  nn.Linear(dv, 128) → ReLU → nn.Linear(128, du)
  f(x) = W₂·σ(W₁·x + b₁) + b₂
  Can learn NONLINEAR relationships
```

For SCF (du=1), the relationship between 32 hidden features and
the scalar SCF value is not necessarily linear. Two layers
gives the model capacity to learn that nonlinearity.

---

## 6. The Padding Mechanism

Fourier basis assumes periodic domain. Real domains are not periodic.

```
Without padding — non-periodic domain causes artifacts:
  FNO assumes f(0) = f(1) — incorrect for Darcy/NS

With padding=8:
  Spatial dim extended: (batch, n, dv) → (batch, n+8, dv)
  Fourier layers run on padded domain
  Padding removed after layers: (batch, n+8, dv) → (batch, n, dv)

Code:
  if self.padding > 0:
      x = F.pad(x, [0, 0, 0, self.padding])
  # ... fourier layers ...
  if self.padding > 0:
      x = x[:, :-self.padding, :]
```

Burgers is periodic — padding=0.
Darcy and NS are non-periodic — padding needed.

---

## 7. Quasi-Linear Complexity

**Per FNO layer, all operations:**

| Operation | Formula | n=1024, k_max=16, dv=64 |
|-----------|---------|--------------------------|
| FFT forward | n × dv × log₂n | 655,360 |
| Truncate | k_max × dv | 1,024 (negligible) |
| R multiply | k_max × dv² | 65,536 |
| iFFT inverse | n × dv × log₂n | 655,360 |
| W branch | n × dv² | 4,194,304 |

**Dominant term:** FFT + iFFT = O(n × dv × log n)

**The paper's claim "O(n log n)"** treats dv as constant.
Full expression: O(n · dv · log n).

**R multiply is O(k_max) — constant in n.**
Same computational cost at n=64 or n=65536.
This is why FNO has consistent error at any resolution.

---

## 8. Hyperparameter Summary

**Burgers equation (1D, Section 5.1):**
```
da=2  (u₀(x) + grid x), du=1, dv=64, k_max=16, T=4
N=1000, batch=20, epochs=500
lr=0.001, StepLR: step=100, gamma=0.5
Target: 1.49% relative L2
```

**Darcy flow (2D, Section 5.2):**
```
da=1, du=1, dv=32, k_max=12, T=4
Resolution: 64×64
Target: 1.08% relative L2
```

**SCF K-joint (your paper):**
```
da=5  (β, γ, τ, θ, ζ), du=1, dv=32, k_max=12, T=4
N=400 (train), N=50 (test)
epochs=1000, lr=0.001, StepLR: step=100, gamma=0.5
Target: <3% relative L2
```

---

## 9. The Burgers Operator Notation Decoded

```
G† : L²_per((0,1); ℝ) → H^r_per((0,1); ℝ)   defined by u₀ ↦ u(·,1)
```

| Symbol | Meaning |
|--------|---------|
| L²_per | Space of periodic, square-integrable functions |
| H^r_per | Sobolev space (r derivatives exist) — smoother than L² |
| (0,1) | Spatial domain |
| ; ℝ | Real-valued |
| u₀ ↦ u(·,1) | Maps initial condition to solution at t=1 |

**Why output is in H^r (smoother than L²):**
Viscosity ν > 0 smooths the solution over time.
Sharp initial features diffuse by t=1.
Smooth output → well-approximated by k_max=16 Fourier modes.

**Why da=2 for Burgers:**
```
Input at each grid point: [u₀(xᵢ), xᵢ]
                            ↑         ↑
                         velocity   position

Without xᵢ: FNO cannot break translation invariance.
With xᵢ:   Model knows WHERE it is evaluating.
Similar to positional encoding in Transformers.
```

**Why du=1 (not 2):**
Output is just the velocity u(x,1) — a scalar.
The positions are already known from the input grid.
You never need to predict position — only the value at each position.
