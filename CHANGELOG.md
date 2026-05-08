# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- `LpLoss` — relative Lp loss function for operator learning benchmarks

---

## [0.3.0] — 2026-05-08

### Added
- `FNO1D` — complete 1D Fourier Neural Operator (Figure 2a, Li et al. 2021)
  - Pointwise lifting layer P: `nn.Linear(da, dv)`
  - T stacked `FourierLayer1D` blocks
  - Nonlinear projection Q: `dv → 128 → du`
  - Optional padding for non-periodic domains
  - `NeuralOperator` abstract base class with `count_parameters()`
- `FourierLayer1D` — equation 2 of Li et al. 2021

---

## [0.2.0] — 2026-05-08

### Added
- `SpectralConv1D` — 1D Fourier integral operator layer (Li et al. 2021, Definition 3, Equations 4–5)
  - Complex weight tensor R of shape `(k_max, dv, dv)`
  - Test suite: 11 tests covering shape, resolution invariance, gradients, NaN safety, batch independence

---

## [0.1.0] — 2026-05-08

### Added
- Repository structure and packaging (`pyproject.toml`)
- GitHub Actions CI pipeline (test and lint)
- Pre-commit hooks (`ruff`, `black`, `isort`)
- Theory documentation: FNO derivations (D1 complete)
- References: Li 2021, Lu 2021, Anandh 2024
- Updated dependencies and `requirements.txt`
