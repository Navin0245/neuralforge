"""
Spectral convolution layers — core computational unit of FNO.

Implements the Fourier integral operator (Li et al. 2021, Def. 3, Eq. 4–5):

    (K v_t)(x) = F^{-1}( R_phi · F(v_t) )(x)

    R_phi: learnable complex weight tensor (k_max, dv, dv)
    F / F^{-1}: rfft / irfft (real input → conjugate symmetry halves cost)

Reference: arXiv 2010.08895
"""

import torch
import torch.nn as nn


class SpectralConv1D(nn.Module):
    """
    1D Fourier integral operator layer (Li et al. 2021, Eq. 4–5).

    Args:
        d_v:   Hidden channel dimension.
        k_max: Number of Fourier modes to retain.

    Input/Output shape: (batch, n, d_v) — real.
    """

    def __init__(self, d_v: int, k_max: int) -> None:
        super().__init__()
        self.d_v = d_v
        self.k_max = k_max
        scale = 1.0 / (d_v * k_max) ** 0.5
        self.R = nn.Parameter(scale * torch.randn(k_max, d_v, d_v, 2))

    def _multiply_modes(self, ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bki,koi->bko", ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n, d_v = x.shape

        x_ft = torch.fft.rfft(x, dim=1, norm="ortho")

        k = min(self.k_max, x_ft.shape[1])
        weights = torch.view_as_complex(self.R.contiguous())
        out_ft_trunc = self._multiply_modes(x_ft[:, :k, :], weights)

        out_ft = torch.zeros(
            batch, n // 2 + 1, d_v, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :k, :] = out_ft_trunc

        return torch.fft.irfft(out_ft, n=n, dim=1, norm="ortho")


class SpectralConv2D(nn.Module):
    """
    2D Fourier integral operator layer (Li et al. 2021, Eq. 4–5).

    Uses rfft2 and mixes two spectral corners to cover the full 2D frequency
    support after the redundant conjugate half is dropped.

    Args:
        d_v:    Hidden channel dimension.
        k_max1: Fourier modes retained along dimension 1.
        k_max2: Fourier modes retained along dimension 2.

    Input/Output shape: (batch, s1, s2, d_v) — real.
    """

    def __init__(self, d_v: int, k_max1: int, k_max2: int) -> None:
        super().__init__()
        self.d_v = d_v
        self.k_max1 = k_max1
        self.k_max2 = k_max2
        scale = 1.0 / (d_v * k_max1 * k_max2) ** 0.5
        self.R1 = nn.Parameter(scale * torch.randn(k_max1, k_max2, d_v, d_v, 2))
        self.R2 = nn.Parameter(scale * torch.randn(k_max1, k_max2, d_v, d_v, 2))

    def _multiply_modes(self, ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bkli,kloi->bklo", ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, s1, s2, d_v = x.shape

        x_ft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        k1 = min(self.k_max1, x_ft.shape[1])
        k2 = min(self.k_max2, x_ft.shape[2])

        weights1 = torch.view_as_complex(self.R1.contiguous())
        weights2 = torch.view_as_complex(self.R2.contiguous())

        out_ft = torch.zeros(
            batch, s1, s2 // 2 + 1, d_v, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :k1, :k2, :] = self._multiply_modes(x_ft[:, :k1, :k2, :], weights1)
        out_ft[:, -k1:, :k2, :] = self._multiply_modes(x_ft[:, -k1:, :k2, :], weights2)

        return torch.fft.irfft2(out_ft, s=(s1, s2), dim=(1, 2), norm="ortho")
