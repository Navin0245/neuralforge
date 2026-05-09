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
| FNO 1D    | Li et al. 2021 (arXiv 2010.08895) | 🔄 Building | Target:<2% L2   |
| FNO 2D    | Li et al. 2021               | 📋 Planned  | Target: <1.5% L2     |
| DeepONet  | Lu et al. 2021               | 📋 Planned  | —                    |
| GNN       | -                            | 📋 Planned  | —                    |
| GINO      | -                            | 📋 Planned  | —                    |
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
