# Complexity Analysis — FNO vs Classical Methods

**Author:** Navin C Chacko
**Reference:** Li et al. 2021, Section 4 ("Quasi-linear complexity")

---

## 1. What Complexity Measures

O-notation measures how the number of arithmetic operations
grows as the problem size n grows. It ignores constants.

```
O(n²)     — quadratic: double n → 4× more work
O(n log n) — quasi-linear: double n → slightly more than 2× work
O(n)       — linear: double n → 2× more work
O(k_max)   — constant in n: any n → same work
```

---

## 2. Naive DFT — O(n²) Derivation

The DFT formula for one output mode k:
```
F̂(k) = Σₓ₌₀^{n-1} f(x) · e^{-2πi·xk/n}
```

**Operation count:**
- n multiplications (f(x) × exponential)
- n-1 additions (summing n products)
- Total: ~2n operations per output mode

You need this for ALL n output modes → n × 2n = **O(n²)**

At n=1024: ~2,000,000 operations per channel per layer.

---

## 3. FFT — O(n log n) Derivation

**Cooley-Tukey algorithm (divide and conquer):**
```
DFT(n) = DFT(even indices, n/2) + DFT(odd indices, n/2)
```

**Recurrence:**
```
T(n) = 2·T(n/2) + n

Expanding:
T(n) = 2·T(n/2) + n
     = 2·[2·T(n/4) + n/2] + n = 4·T(n/4) + 2n
     = 4·[2·T(n/8) + n/4] + 2n = 8·T(n/8) + 3n
     ...
     = n·T(1) + n·log₂(n)
     = O(n log₂ n)
```

**Where log n comes from:**
```
n=8 recursion tree:
Level 0: 1 problem of size 8
Level 1: 2 problems of size 4
Level 2: 4 problems of size 2
Level 3: 8 problems of size 1

Depth = log₂(8) = 3 levels
Work per level = O(n) to combine
Total = O(n) × log₂(n) = O(n log n)
```

---

## 4. Truncated DFT — O(n · k_max)

FNO only computes k_max modes (not all n modes):
```
For each of k_max modes:
  F̂(k) = Σₓ₌₀^{n-1} f(x) · e^{-2πi·xk/n}
  Cost: 2n operations

Total: k_max × 2n = O(n · k_max)
```

But FNO uses FFT (computing all n modes first, then truncating):
```
FFT:      O(n log n)    ← compute all modes
Truncate: O(k_max)      ← slice array, negligible
```

FFT + truncate is faster than truncated DFT when:
```
n log n < n · k_max
log n < k_max
```

For n=1024, k_max=16: log₂(1024)=10 < 16. FFT wins.

---

## 5. Complete FNO Layer Cost

| Operation | Formula | n=1024, k_max=16, dv=64 |
|-----------|---------|--------------------------|
| FFT forward | n · dv · log₂n | 655,360 |
| Truncate | k_max · dv | 1,024 |
| **R multiply** | **k_max · dv²** | **65,536** |
| Zero-pad | k_max · dv | 1,024 |
| iFFT inverse | n · dv · log₂n | 655,360 |
| W branch | n · dv² | 4,194,304 |
| **Total** | **O(n · dv · log n)** | **5,573,648** |

**The R multiply (65,536 ops) is constant in n.**
At n=65536: R multiply is STILL 65,536 ops.
FFT scales: 65536 × 64 × 16 = 67,108,864 ops.

---

## 6. Speedup vs Classical Solver

**FEM/FDM for one PDE solve:**
```
Time complexity: O(n^α) where α ≥ 1
For 3D: O(n^(4/3)) or worse
At n=1024: ~millions of operations, minutes of compute
```

**FNO inference (one forward pass):**
```
Time complexity: O(n log n)
At n=1024: ~5.5M operations, milliseconds
```

**Practical speedup (Burgers experiment):**
```
Traditional solver: ~2 minutes per sample
FNO inference:      ~0.5ms per sample
Speedup:            ~240,000×
```

This is the speedup that makes surrogate modelling valuable.
For 10,000 design evaluations: solver = 140 days, FNO = 5 seconds.

---

## 7. The Uniform Grid Requirement

**Why FFT needs uniform grid:**
The Cooley-Tukey even/odd split requires:
```
xᵢ = i/n   for i = 0,1,...,n-1
```

The exponential e^{-2πixk/n} must form a regular pattern
for the recursion to work. Non-uniform spacing breaks the pattern.

**What happens without uniform grid:**
- FFT cannot be applied
- Must use NFFT (Non-Uniform FFT) — slower
- Or GNN/GINO which have no grid requirement

**For your SCF problem:**
ANSYS mesh is non-uniform (refined near weld toe).
Options:
1. Interpolate ANSYS results → uniform grid → FNO (preprocessing step)
2. Use GINO directly on unstructured mesh (no interpolation needed)

This difference justifies the GNN/GINO comparison in your paper.

---

## 8. Summary Table

| Method | Complexity | n=1024 ops | Resolution invariant |
|--------|-----------|-----------|---------------------|
| Naive DFT | O(n²) | 1,048,576 | No |
| FFT | O(n log n) | 10,240 | Yes |
| Truncated DFT | O(n·k_max) | 16,384 | Yes |
| R multiply | O(k_max) | 16 | Yes (constant!) |
| FEM solve | O(n^α) | millions | N/A |
| FNO inference | O(n log n) | ~10K | Yes |
