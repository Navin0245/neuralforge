"""
Fourier Layer — one complete FNO layer.

Implements equation 2 from Li et al. 2021:

    v_{t+1}(x) = σ( W·v_t(x)  +  (K·v_t)(x) )
                      ↑                ↑
                  local branch    global branch
                  (W matrix)      (SpectralConv1D)

The two branches are complementary:

    W branch (local):
        - Matrix multiply at each point independently
        - No communication across spatial locations
        - Handles non-periodic boundaries
        - Implemented as Conv1d(dv, dv, kernel_size=1)
        - Cost: O(n · dv²)

    K branch (global):
        - Fourier integral operator (SpectralConv1D)
        - Mixes information across ALL spatial locations
        - Captures long-range dependencies
        - Cost: O(n log n)

Neither branch alone is sufficient:
    - K alone: cannot handle non-periodic boundaries
    - W alone: cannot capture global structure (like a 1×1 CNN)
    - Together: complete spatial mixing, fast and expressive

Reference:
    Li et al. 2021 — Fourier Neural Operator for Parametric PDEs
    arXiv 2010.08895, Definition 1, Equation 2
"""

import torch
import torch.nn as nn

from neuralforge.layers.spectral_conv import SpectralConv1D, SpectralConv2D


class FourierLayer1D(nn.Module):
    """
    One complete Fourier layer for 1D problems.

    Implements equation 2 of Li et al. 2021:
        v_{t+1}(x) = σ( W·v_t(x) + (K·v_t)(x) )

    This layer is the fundamental building block of FNO1D.
    Four of these are stacked to form the complete FNO architecture.

    Args:
        d_v:   Hidden channel dimension.
               Paper: 64 (1D problems), 32 (2D problems).
        k_max: Maximum Fourier modes for the K branch.
               Paper: 16 (Burgers), 12 (Darcy).

    Input shape:  (batch, n, d_v)   real
    Output shape: (batch, n, d_v)   real

    Parameters:
        spectral_conv: SpectralConv1D — the K (global) branch
        w:             Conv1d         — the W (local) branch
        activation:    ReLU           — applied after combining branches

    Example:
        >>> layer = FourierLayer1D(d_v=64, k_max=16)
        >>> x = torch.randn(8, 64, 64)   # (batch, n, dv)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 64, 64])

    Stacking (FNO uses T=4 layers):
        >>> layers = nn.ModuleList([FourierLayer1D(32, 12) for _ in range(4)])
        >>> for layer in layers:
        ...     x = layer(x)
    """

    def __init__(self, d_v: int, k_max: int) -> None:
        super().__init__()

        self.d_v = d_v
        self.k_max = k_max

        # ── K branch: global Fourier integral operator ────────────────
        # SpectralConv1D implements equation 4:
        #   (K v_t)(x) = F^{-1}( R · F(v_t) )(x)
        # Learnable parameter: R of shape (k_max, dv, dv, 2)
        self.spectral_conv = SpectralConv1D(d_v=d_v, k_max=k_max)

        # ── W branch: local pointwise linear transform ────────────────
        # Implements W·v_t(x) applied at each x independently.
        # Conv1d with kernel_size=1 is equivalent to a matrix multiply
        # applied at every position simultaneously.
        #
        # Why Conv1d and not nn.Linear?
        #   nn.Linear expects (batch, features) — would need a loop over n
        #   Conv1d(dv, dv, kernel_size=1) processes all n positions at once
        #   BUT Conv1d expects (batch, channels, length) not (batch, length, channels)
        #   Solution: transpose before Conv1d, transpose back after
        self.w = nn.Conv1d(
            in_channels=d_v,
            out_channels=d_v,
            kernel_size=1,
        )

        # ── Activation ───────────────────────────────────────────────
        # ReLU applied after combining both branches.
        # Applied ONCE to the sum, not to each branch separately.
        # σ(a + b) ≠ σ(a) + σ(b) — applying separately would be wrong.
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing equation 2.

        Args:
            x: Hidden representation, shape (batch, n, d_v).
               Real-valued. n can be any resolution.

        Returns:
            Updated representation, shape (batch, n, d_v).
            Real-valued.

        Shape trace:
            x                   (batch, n, dv)   real
            ├─ K branch:
            │   spectral_conv(x) (batch, n, dv)   real
            └─ W branch:
                x.transpose(1,2) (batch, dv, n)   real   ← Conv1d needs channels first
                w(...)           (batch, dv, n)   real
                .transpose(1,2)  (batch, n, dv)   real   ← restore original layout
            sum + activation     (batch, n, dv)   real
        """
        # ── K branch: global Fourier integral operator ────────────────
        # SpectralConv1D handles the FFT → truncate → R multiply → iFFT
        # Input:  (batch, n, dv)
        # Output: (batch, n, dv)
        k_out = self.spectral_conv(x)

        # ── W branch: local pointwise linear transform ────────────────
        # Conv1d expects (batch, channels, length)
        # Our x is (batch, length, channels) = (batch, n, dv)
        # Step 1: transpose to (batch, dv, n)
        # Step 2: apply Conv1d → (batch, dv, n)
        # Step 3: transpose back to (batch, n, dv)
        w_out = self.w(x.transpose(1, 2)).transpose(1, 2)

        # ── Combine and activate ──────────────────────────────────────
        # Add both branches (same shape: (batch, n, dv))
        # Apply ReLU: σ(x) = max(0, x) elementwise
        return self.activation(k_out + w_out)


class FourierLayer2D(nn.Module):
    """
    One complete 2D Fourier layer.

    Implements equation 2 of Li et al. 2021 for 2D problems:
        v_{t+1}(x) = σ( W·v_t(x) + (K·v_t)(x) )

    Key differences from FourierLayer1D:
        - SpectralConv2D instead of SpectralConv1D
        - Conv2d instead of Conv1d for W branch
        - permute(0,3,1,2) instead of transpose(1,2)

    Args:
        d_v:    Hidden channel dimension
        k_max1: Max Fourier modes in dim 1
        k_max2: Max Fourier modes in dim 2

    Input shape:  (batch, s1, s2, d_v)
    Output shape: (batch, s1, s2, d_v)
    """

    def __init__(self, d_v: int, k_max1: int, k_max2: int) -> None:
        super().__init__()
        self.k_max1 = k_max1
        self.k_max2 = k_max2
        # ── K branch: global Fourier integral operator ────────────────
        self.spectral_conv = SpectralConv2D(d_v=d_v, k_max1=k_max1, k_max2=k_max2)
        # W branch: Conv2d(dv, dv, kernel_size=1)
        self.w = nn.Conv2d(
            in_channels=d_v,
            out_channels=d_v,
            kernel_size=1,
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # K branch: spectral_conv(x)
        # W branch: permute → Conv2d → permute back
        # return: activation(k_out + w_out)
        k_out = self.spectral_conv(x)
        w_out = self.w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.activation(k_out + w_out)
