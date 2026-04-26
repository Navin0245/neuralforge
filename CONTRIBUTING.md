# Contributing

This is a personal learning repository.
The standards below are self-imposed — to build good habits.

## Code Standards

- Type hints on every function signature
- Docstring on every class and function
- Reference the paper equation in the docstring
- Test written before the implementation (TDD)

## Commit Convention

Format: type(scope): description

Types:
  feat     — new implementation
  fix      — bug fix
  docs     — documentation
  test     — adding tests
  refactor — restructuring without behaviour change
  hpc      — HPC scripts
  chore    — setup, config, dependencies

Examples:
  feat(layers): add SpectralConv1D with complex weight tensor
  docs(theory): add convolution theorem derivation
  hpc(slurm): add array job script for k_max sweep
  test(models): add FNO1D resolution invariance test

## Branch Naming

  feature/what-you-are-building
  fix/what-bug-you-are-fixing
  docs/what-documentation
  hpc/what-hpc-feature
  experiment/what-experiment
