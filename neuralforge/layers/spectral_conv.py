"""
Spectral Convolution layers — the core computational unit of FNO.

Implements the Fourier integral operator from Definition 3, Equation 4:

    (K v_t)(x) = F^{-1}( R_phi · F(v_t) )(x)

where:
    F       = Fast Fourier Transform (rfft for real-valued input)
    R_phi   = learnable complex weight tensor, shape (k_max, dv, dv)
    F^{-1}  = Inverse FFT (irfft)

The multiplication R_phi · F(v_t) is Equation 5:
    (R · F̂v_t)_{k,l} = Σ_j R_{k,l,j} · (F̂v_t)_{k,j}

This is a matrix-vector product per Fourier mode k:
    for each mode k: mix all dv input channels → dv output channels

Reference:
    Li et al. 2021 — Fourier Neural Operator for Parametric PDEs
    arXiv 2010.08895, Definition 3, Equations 4 and 5
"""

import torch
import torch.nn as nn


class SpectralConv1D(nn.Module):
    """
    1D Fourier integral operator layer.

    Implements equation 4 of Li et al. 2021 for 1D problems.

    Forward pass:
        1. FFT:      v_t (batch, n, dv) → F̂(v_t) (batch, n//2+1, dv)
        2. Truncate: keep only k_max lowest modes
                     F̂(v_t) (batch, k_max, dv)
        3. Multiply: apply learned R per mode (equation 5)
                     out_ft (batch, k_max, dv)
        4. Pad:      zero-pad back to n//2+1 modes
        5. iFFT:     back to physical space (batch, n, dv)

    Args:
        d_v:   Hidden channel dimension. Paper: 64 (1D), 32 (2D).
        k_max: Max Fourier modes kept. Paper: 16 (Burgers), 12 (Darcy).

    Learnable parameters:
        R: complex weight tensor stored as real (k_max, dv, dv, 2)
           Shape: (k_max, dv, dv) complex = (k_max, dv, dv, 2) real
           Initialised with scale 1/sqrt(dv * k_max) for stable training

    Input shape:  (batch, n, dv)   — real
    Output shape: (batch, n, dv)   — real

    Example:
        >>> layer = SpectralConv1D(d_v=64, k_max=16)
        >>> x = torch.randn(8, 64, 64)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 64, 64])
    """

    def __init__(self, d_v: int, k_max: int) -> None:
        super().__init__()

        self.d_v = d_v
        self.k_max = k_max

        # Learnable complex weight tensor R
        # Stored as real tensor (k_max, dv, dv, 2)
        # Last dim: [real_part, imaginary_part]
        # Scale: 1/sqrt(dv * k_max) prevents exploding activations
        scale = 1.0 / (d_v * k_max) ** 0.5
        self.R = nn.Parameter(scale * torch.randn(k_max, d_v, d_v, 2))

    def _multiply_modes(
        self,
        ft: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply equation 5: multiply Fourier modes by weight tensor R.

        For each mode k, performs a dv×dv matrix-vector multiply:
            out[b, k, l] = Σ_j R[k, l, j] · ft[b, k, j]

        Using einsum notation:
            b = batch dimension
            k = Fourier mode index (0 to k_max-1)
            i = input channel index (0 to dv-1)
            o = output channel index (0 to dv-1)

        Args:
            ft: Truncated Fourier coefficients, shape (batch, k_max, dv)
                dtype: complex64
            R:  Weight tensor, shape (k_max, dv, dv)
                dtype: complex64

        Returns:
            Output Fourier coefficients, shape (batch, k_max, dv)
            dtype: complex64
        """
        return torch.einsum("bki,koi->bko", ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Fourier integral operator.

        Args:
            x: Input hidden representation.
               Shape: (batch, n, dv) — real-valued.
               n = number of grid points (any value, uniform spacing).

        Returns:
            Output of same shape as input: (batch, n, dv) — real-valued.

        Note on rfft vs fft:
            rfft exploits that x is real-valued.
            For real x of length n, rfft gives n//2+1 complex modes
            instead of n. The negative frequencies are redundant
            (conjugate symmetry). This halves the FFT computation.
        """
        batch, n, d_v = x.shape

        # ── Step 1: FFT along spatial dimension ──────────────────────
        # rfft exploits real-valued input: n points → n//2+1 modes
        # Shape: (batch, n, dv) real → (batch, n//2+1, dv) complex
        x_ft = torch.fft.rfft(x, dim=1, norm="ortho")

        # ── Step 2: Truncate to k_max lowest frequency modes ─────────
        # Only keep modes 0, 1, ..., k_max-1
        # These are the low-frequency modes carrying global structure
        # High-frequency modes (k > k_max) are discarded
        k = min(self.k_max, x_ft.shape[1])
        x_ft_trunc = x_ft[:, :k, :]
        # Shape: (batch, k_max, dv) complex

        # ── Step 3: Multiply by learned weight tensor R (equation 5) ──
        # Convert R from real storage (k_max, dv, dv, 2)
        # to complex (k_max, dv, dv)
        weights = torch.view_as_complex(self.R.contiguous())
        out_ft_trunc = self._multiply_modes(x_ft_trunc, weights)
        # Shape: (batch, k_max, dv) complex

        # ── Step 4: Zero-pad back to n//2+1 modes ────────────────────
        # Modes k > k_max were discarded — set them to zero
        # This preserves the signal length for irfft
        out_ft = torch.zeros(
            batch,
            n // 2 + 1,
            d_v,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :k, :] = out_ft_trunc
        # Shape: (batch, n//2+1, dv) complex

        # ── Step 5: Inverse FFT back to physical space ────────────────
        # n= argument ensures output length matches input length exactly
        # Shape: (batch, n//2+1, dv) complex → (batch, n, dv) real
        out = torch.fft.irfft(out_ft, n=n, dim=1, norm="ortho")

        return out
