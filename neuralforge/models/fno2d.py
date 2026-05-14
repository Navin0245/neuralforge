"""
Fourier Neural Operator — 2D.

Full architecture from Li et al. 2021, Figure 2(a) extended to 2D:

    a(x,y) → P → [FourierLayer2D × T] → Q → u(x,y)

    P: pointwise lifting — nn.Linear(da, d_v)
    FourierLayer2D × T: stacked Fourier layers, each v_{t+1} = σ(W·v_t + K·v_t)
    Q: nonlinear projection — nn.Sequential(d_v → 128 → du)

Reference:
    Li et al. 2021 — FNO (arXiv 2010.08895)
    Figure 2(a), Section 4, Section 5.2 (Darcy flow)
"""

import torch
import torch.nn as nn

from neuralforge.layers.fourier_layer import FourierLayer2D
from neuralforge.models.base import NeuralOperator


class FNO2D(NeuralOperator):
    """
    Fourier Neural Operator for 2D problems.

    Learns the operator G†: A → U mapping input functions
    to output functions on 2D domains.

    Args:
        da:       Input function channels. Default 1 (scalar field).
        du:       Output function channels. Default 1 (scalar field).
        d_v:      Hidden channel dimension. Paper: 32 for 2D.
        k_max1:   Max Fourier modes along dimension 1. Paper: 12 (Darcy).
        k_max2:   Max Fourier modes along dimension 2. Paper: 12 (Darcy).
        n_layers: Number of Fourier layers T. Paper: 4.
        padding:  Zero-padding for non-periodic domains. Default 0.

    Input shape:  (batch, s1, s2, da)
    Output shape: (batch, s1, s2, du)

    Paper hyperparameters (Darcy flow, Section 5.2):
        da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4
        N_train=1000, epochs=500, lr=0.001

    Example:
        >>> model = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        >>> x = torch.randn(4, 64, 64, 1)
        >>> u = model(x)
        >>> u.shape
        torch.Size([4, 64, 64, 1])
    """

    def __init__(
        self,
        da: int = 1,
        du: int = 1,
        d_v: int = 32,
        k_max1: int = 12,
        k_max2: int = 12,
        n_layers: int = 4,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.da = da
        self.du = du
        self.d_v = d_v
        self.k_max1 = k_max1
        self.k_max2 = k_max2
        self.n_layers = n_layers
        self.padding = padding

        self.p = nn.Linear(da, d_v)
        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer2D(d_v=d_v, k_max1=k_max1, k_max2=k_max2)
                for _ in range(n_layers)
            ]
        )
        self.q = nn.Sequential(
            nn.Linear(d_v, 128),
            nn.ReLU(),
            nn.Linear(128, du),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input function, shape (batch, s1, s2, da).

        Returns:
            u: Output function, shape (batch, s1, s2, du).
        """
        x = self.p(x)

        if self.padding > 0:
            x = nn.functional.pad(x, [0, 0, 0, self.padding, 0, self.padding])

        for layer in self.fourier_layers:
            x = layer(x)

        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, :]

        return self.q(x)
