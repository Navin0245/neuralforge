# Changelog

Format: [version] — date
Types: Added | Changed | Fixed | Removed

---

## [Unreleased]

### Added
- Repository structure and packaging (pyproject.toml)
- GitHub Actions CI pipeline (test + lint)
- Pre-commit hooks (ruff, black, isort)
- Theory documentation: FNO derivations (D1 complete)
- References: Li 2021, Lu 2021, Anandh 2024

### In Progress
- SpectralConv1D — Fourier integral operator layer
- FNO1D — full 1D Fourier Neural Operator

## [Unreleased] — ongoing

### Added
- SpectralConv1D — 1D Fourier integral operator layer
  - Implements Li et al. 2021, Definition 3, Equations 4-5
  - R tensor: (k_max, dv, dv) complex weight
  - Full test suite: 11 tests covering shape, resolution
    invariance, gradients, NaN, batch independence

### Learned
- TDD workflow: write failing test first, then implement
- torch.fft.rfft vs fft: rfft halves computation for real input
- torch.einsum: 'bki,koi->bko' for batched matrix multiply
- torch.view_as_complex: real (k,dv,dv,2) → complex (k,dv,dv)
- Git: feature branch → push → PR → review → merge → delete

git push origin feature/fourier-layer-1d
