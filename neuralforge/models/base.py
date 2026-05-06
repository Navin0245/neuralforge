"""Abstract base class for all neural operators."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NeuralOperator(ABC, nn.Module):
    """
    Abstract base class for all neural operators in neuralforge.

    Every model must implement forward().
    Provides count_parameters() for free.

    Reference: Li et al. 2021 — operator learning framework
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\nTrainable parameters: {self.count_parameters():,}"
