"""
Fourier Neural Operator — 1D.

Full architecture from Li et al. 2021, Figure 2(a):

    a(x) → P → [FourierLayer1D × T] → Q → u(x)

    P: lifting network
       nn.Linear(da, d_v) applied pointwise
       Lifts low-dimensional input (da) to
       high-dimensional representation (d_v)

    FourierLayer1D × T:
       T=4 stacked Fourier layers
       Each: v_{t+1} = σ(W·v_t + K·v_t)
       Mixes information locally (W) and globally (K)

    Q: projection network
       nn.Sequential: d_v → 128 → du
       Two-layer MLP for expressiveness
       Projects back to output dimension

Why P uses nn.Linear not Conv1d:
    Input a(x) has shape (batch, n, da)
    P maps each point independently: da → d_v
    nn.Linear(da, d_v) applied to last dimension
    PyTorch broadcasts automatically over (batch, n)
    No transpose needed — Linear works on last dim

Why Q has two layers (d_v → 128 → du) not one:
    One linear layer: d_v → du
    This is a single affine map — limited expressiveness
    Two layers with ReLU: learns nonlinear projection
    Especially important when du=1 (scalar output)
    The hidden dim 128 is a standard choice

Why padding exists:
    FNO assumes periodic domain (Fourier basis is periodic)
    Real domains are non-periodic (Darcy, Navier-Stokes)
    Padding adds zeros at domain edge → creates artificial period
    The W branch (local) corrects boundary behaviour
    Padding is removed after Fourier layers before Q

Reference:
    Li et al. 2021 — FNO (arXiv 2010.08895)
    Figure 2(a), Section 4, Section 5.1
"""

import torch
import torch.nn as nn

from neuralforge.layers.fourier_layer import FourierLayer1D
from neuralforge.models.base import NeuralOperator


class FNO1D(NeuralOperator):
    """
    Fourier Neural Operator for 1D problems.

    Learns the operator G†: A → U mapping input functions
    to output functions on 1D domains.

    Args:
        da:       Input function channels. Default 1 (scalar field).
        du:       Output function channels. Default 1 (scalar field).
        d_v:      Hidden channel dimension. Paper: 64 for 1D.
        k_max:    Max Fourier modes. Paper: 16 for Burgers.
        n_layers: Number of Fourier layers T. Paper: 4.
        padding:  Zero-padding for non-periodic domains. Default 0.

    Input shape:  (batch, n, da)
    Output shape: (batch, n, du)

    Paper hyperparameters (Burgers equation, Section 5.1):
        da=1, du=1, d_v=64, k_max=16, n_layers=4
        N_train=1000, epochs=500, lr=0.001

    Example:
        >>> model = FNO1D(da=1, du=1, d_v=64, k_max=16, n_layers=4)
        >>> x = torch.randn(8, 1024, 1)
        >>> u = model(x)
        >>> u.shape
        torch.Size([8, 1024, 1])
    """

    def __init__(
        self,
        da: int = 1,
        du: int = 1,
        d_v: int = 64,
        k_max: int = 16,
        n_layers: int = 4,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.da = da
        self.du = du
        self.d_v = d_v
        self.k_max = k_max
        self.n_layers = n_layers
        self.padding = padding

        # ── P: lifting network ────────────────────────────────────────
        # Maps input channels da → hidden channels d_v
        # Applied pointwise: nn.Linear broadcasts over (batch, n)
        # No spatial communication at this stage
        self.p = nn.Linear(da, d_v)

        # ── T Fourier layers ──────────────────────────────────────────
        # Each layer: v_{t+1} = σ(W·v_t + K·v_t)
        # ModuleList so PyTorch registers all parameters
        self.fourier_layers = nn.ModuleList(
            [FourierLayer1D(d_v=d_v, k_max=k_max) for _ in range(n_layers)]
        )

        # ── Q: projection network ─────────────────────────────────────
        # Maps hidden channels d_v → output channels du
        # Two-layer MLP: d_v → 128 → du
        # Applied pointwise: Linear broadcasts over (batch, n)
        self.q = nn.Sequential(
            nn.Linear(d_v, 128),
            nn.ReLU(),
            nn.Linear(128, du),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full FNO1D forward pass.

        Args:
            x: Input function, shape (batch, n, da).
               For Burgers: initial condition u0(x), da=1.
               For SCF: geometry parameters, da=5.

        Returns:
            u: Output function, shape (batch, n, du).
               For Burgers: solution u(x,1) at t=1, du=1.
               For SCF: stress concentration field, du=1.

        Shape trace (batch=8, n=64, da=1, d_v=32, du=1):
            x                (8, 64,  1)   input
            p(x)             (8, 64, 32)   lifted
            [+ padding]      (8, 74, 32)   if padding=10
            fourier_layers   (8, 74, 32)   T times
            [- padding]      (8, 64, 32)   remove padding
            q(x)             (8, 64,  1)   projected to output
        """
        # ── Step 1: Lift input to hidden dimension ─────────────────────
        # nn.Linear(da, d_v) applied to last dimension
        # PyTorch broadcasts: (batch, n, da) → (batch, n, d_v)
        x = self.p(x)

        # ── Step 2: Optional padding for non-periodic domains ──────────
        # Adds zeros along spatial dimension (dim=1)
        # Creates artificial periodicity at domain boundaries
        if self.padding > 0:
            x = nn.functional.pad(x, [0, 0, 0, self.padding])
            # pad format: (last_dim_left, last_dim_right,
            #              second_last_left, second_last_right)
            # [0, 0, 0, padding] pads spatial dim on the right

        # ── Step 3: T Fourier layers ───────────────────────────────────
        # Each layer: v_{t+1}(x) = σ(W·v_t(x) + K·v_t(x))
        # Shape stays (batch, n_padded, d_v) throughout
        for layer in self.fourier_layers:
            x = layer(x)

        # ── Step 4: Remove padding ─────────────────────────────────────
        if self.padding > 0:
            x = x[:, : -self.padding, :]

        # ── Step 5: Project to output dimension ───────────────────────
        # nn.Sequential with two Linear layers + ReLU
        # Applied to last dimension: (batch, n, d_v) → (batch, n, du)
        x = self.q(x)

        return x
