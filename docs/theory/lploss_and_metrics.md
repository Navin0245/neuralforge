# LpLoss and Evaluation Metrics

**Author:** Navin C Chacko
**Reference:** Li et al. 2021, Section 5 (used for all benchmark results)

---

## 1. What Is LpLoss?

LpLoss is a family of loss functions based on the Lp norm — a mathematical
measure of the "size" of a vector or function.

```
p=1: L1 norm — sum of absolute values (Mean Absolute Error)
p=2: L2 norm — Euclidean distance (most common for PDEs)
p=∞: L∞ norm — maximum absolute value
```

The L2 norm aligns with physical energy:
```
||u||₂ = √(∫|u(x)|² dx) ≈ √(energy of the field)
```

---

## 2. Relative L2 Loss — The Formula

**For a single sample:**
```
Relative L2 = ||u_pred - u_true||₂ / ||u_true||₂

Discrete form:
           = √(Σᵢ(u_pred(xᵢ) - u_true(xᵢ))²)
             ─────────────────────────────────
                    √(Σᵢ u_true(xᵢ)²)
```

**For a batch (mean reduction):**
```
Loss = (1/B) × Σᵢ₌₁ᴮ [||u_pred_i - u_true_i||₂ / ||u_true_i||₂]
```

**In PyTorch:**
```python
def forward(self, u_pred, u_true):
    u_pred_flat = torch.flatten(u_pred, start_dim=1)   # (batch, n*dv)
    u_true_flat = torch.flatten(u_true, start_dim=1)

    numerator   = torch.linalg.norm(u_pred_flat - u_true_flat, ord=2, dim=1)
    denominator = torch.linalg.norm(u_true_flat, ord=2, dim=1).clamp(min=1e-8)

    return (numerator / denominator).mean()
```

---

## 3. Why Relative and Not Absolute (MSE)?

**The scale problem:**
```
Sample A (SCF = 2.5): absolute error 0.1 = 4% error  ← bad
Sample B (pressure = 1e6): absolute error 0.1 = tiny  ← irrelevant
```

**Relative loss normalises:**
```
Sample A: error/2.5 = 4%
Sample B: error/1e6 = negligible
Both measured in physically meaningful percentages
```

**Three reasons relative beats MSE:**

1. Scale independence: 5% error means the same regardless of magnitude
2. Training stability: prevents high-magnitude samples dominating gradients
3. Physical meaning: engineers think in percentages, not absolute values

---

## 4. Five Contracts of LpLoss

| Contract | Test | Why |
|---------|------|-----|
| Identity | loss(u,u) = 0 | Perfect prediction → zero error |
| Non-negativity | loss ≥ 0 always | L2 norm is always non-negative |
| Scale invariance | loss(kA,kB) = loss(A,B) | k cancels in numerator/denominator |
| Asymmetry | loss(A,B) ≠ loss(B,A) | Denominator uses u_true — swapping changes denominator |
| Batch independence | mean of individuals = batch mean | Per-sample before reducing |

**The critical implementation detail — batch independence:**
```
WRONG (sum before divide):
  total_num = Σᵢ ||u_pred_i - u_true_i||₂
  total_den = Σᵢ ||u_true_i||₂
  loss = total_num / total_den   ← high-magnitude samples dominate

CORRECT (divide then reduce):
  rel_errors = [||u_pred_i - u_true_i||₂ / ||u_true_i||₂ for each i]
  loss = mean(rel_errors)        ← each sample contributes equally
```

**The asymmetry test must use controlled tensors:**
```python
# WRONG — random tensors can accidentally be symmetric (flaky test)
y_true = torch.rand(32, 64, 1)
y_pred = torch.rand(32, 64, 1)

# CORRECT — mathematically guaranteed asymmetry
y_true = torch.ones(4, 64, 1) * 10.0   # large magnitude
y_pred = torch.ones(4, 64, 1) * 1.0    # small magnitude
# loss(pred, true) = 9/10 = 0.9
# loss(true, pred) = 9/1  = 9.0
# Always different
```

---

## 5. Why Not MSE for PDEs

**MSE is mesh-dependent:**
```
MSE = (1/N) Σᵢ (u_pred(xᵢ) - u_true(xᵢ))²

Depends on N (number of grid points).
At n=64: MSE is divided by 64.
At n=256: same total error but divided by 256 → MSE is 4× smaller.
Resolution-dependent metric for a resolution-invariant operator = wrong.
```

**Relative L2 approximates the continuous L2 norm:**
```
||u||_L2 = √(∫_D |u(x)|² dx)

This is independent of how you discretise D.
More points → better approximation of the integral,
but the limit is the same.
```

**Physical energy interpretation:**
The L2 norm of many physical fields = energy of the system.
Minimising L2 error = minimising "phantom energy" introduced by the model.
MSE is just a statistical average — no physical interpretation.

---

## 6. Using the Metric in Your Paper

**In your paper's results section:**
```
All errors reported as relative L2 error:
  error = ||u_pred - u_true||₂ / ||u_true||₂

For SCF prediction: u values range from 1 to 10.
Relative L2 error of 2.4% means:
  On average, predicted SCF is within 2.4% of FEA value.
  For SCF=5.0: prediction is within ±0.12
```

**Reporting standard:**
- Report the BEST validation relative L2 error achieved
- Report at multiple resolutions to demonstrate invariance
- Compare against Efthymiou (parametric) and MLP baselines
