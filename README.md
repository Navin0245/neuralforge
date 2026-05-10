# neuralforge
---

Personal learning laboratory for neural operators, production ML
engineering, Git workflows, and HPC.

---

## What This Repo Is

Not a general-purpose library. A learning laboratory where every
neural operator is implemented from scratch to build deep
understanding of the mathematics, the code, and the industrial
engineering workflow.

**Learning goals:**
- Implement FNO, DeepONet, WNO, GINO from papers
- Production code quality: tests, CI/CD, type hints, docstrings
- Git workflow: branches, PRs, conventional commits
- MLOps: MLflow, DVC, Docker
- HPC: SLURM, MPI distributed training, CUDA kernels

---

## Implementations

| Model     | Paper                        | Status      | Benchmark             |
|-----------|------------------------------|-------------|-----------------------|
| FNO 1D    | Li et al. 2021 (arXiv 2010.08895) | Done | 0.34% rel. L2 (paper: 1.49%) |
| FNO 2D    | Li et al. 2021               | Planned  | Target: <1.5% L2     |
| DeepONet  | Lu et al. 2021               | Planned  | —                    |
| GNN       | -                            | Planned  | —                    |
| GINO      | -                            | Planned  | —                    |

### Burgers 1D — Resolution Invariance

Trained at n=1024, evaluated on held-out test set at multiple resolutions using the same weights.

| Resolution | Rel. L2 Error | Notes |
|-----------|---------------|-------|
| 64        | 0.322%        | 16× coarser than training |
| 128       | 0.319%        | |
| 256       | 0.318%        | |
| 512       | 0.317%        | |
| 1024      | 0.317%        | training resolution |
| 2048      | 0.317%        | |
| 4096      | 3.785%        | subsampling artifacts in ground truth data† |

Error varies by 0.005% across a 32× resolution range (n=64 to n=2048).

†At n=4096 the dataset's own resolution limit (~8192 points) is approached. The error increase reflects ground truth data quality, not model degradation.
---

## Structure
neuralforge/        ← implementations
hpc/                ← SLURM, MPI, CUDA scripts
experiments/        ← one script per paper result
configs/            ← YAML hyperparameters
docs/theory/        ← derivations and learning notes
tests/              ← unit and integration tests

---

## Setup

```bash
git clone https://github.com/Navin0245/neuralforge
cd neuralforge
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
pytest tests/ -v
```

---

## References

- Li et al. 2021 — Fourier Neural Operator (arXiv 2010.08895)
- Lu et al. 2021 — DeepONet (Nature Machine Intelligence)
- Tripura & Chakraborty 2022 — WNO (arXiv 2205.02191)
- Li et al. 2023 — GINO (arXiv 2309.00024)
