"""
Unit tests for SpectralConv2D.

Contracts:
    1. Output shape matches input shape
    2. Resolution invariance — same weights at any s
    3. No NaN or Inf in output
    4. Gradients flow to BOTH R1 and R2
    5. Output is real-valued (not complex)
    6. Both corners contribute independently
"""

import pytest
import torch

from neuralforge.layers.spectral_conv import SpectralConv2D


class TestSpectralConv2D:
    def test_output_shape_matches_input(self):
        """
        Output shape must equal input shape (batch, s1, s2, dv).
        Input convention: channels LAST — (batch, s1, s2, dv).
        """
        layer = SpectralConv2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(8, 32, 32, 16)  # (batch=8, s1=32, s2=32, dv=16)
        y = layer(x)
        assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"

    def test_resolution_invariance(self):
        """
        Same layer must work at any grid resolution.
        Train at s=64, evaluate at s=32 or s=128.
        """
        layer = SpectralConv2D(d_v=16, k_max1=4, k_max2=4)
        layer.eval()
        for s in [32, 64, 128]:
            x = torch.randn(4, s, s, 16)
            y = layer(x)
            assert y.shape == x.shape, (
                f"Resolution invariance failed at s={s}: "
                f"input {x.shape}, output {y.shape}"
            )

    def test_no_nan_in_output(self):
        layer = SpectralConv2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16)
        y = layer(x)
        assert not torch.isnan(y).any(), "Output contains NaN values"

    def test_gradient_flows_to_both_r_tensors(self):
        """
        Gradients must reach BOTH R1 and R2.
        R1 handles corner 1 (top-left).
        R2 handles corner 2 (bottom-left).
        If either receives no gradient it cannot learn.
        """
        layer = SpectralConv2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.R1.grad is not None, "Gradient did not flow to R1"
        assert layer.R2.grad is not None, "Gradient did not flow to R2"

    def test_output_is_real_valued(self):
        """
        irfft2 must return real output.
        Complex output would crash downstream layers.
        """
        layer = SpectralConv2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16)
        y = layer(x)
        assert torch.isreal(y).all(), "Output contains complex values"

    def test_both_corners_contribute(self):
        """
        R1 and R2 must both contribute to the output.
        Zero out R2 — output must differ from full output.
        """
        layer = SpectralConv2D(d_v=16, k_max1=4, k_max2=4)
        layer.eval()
        x = torch.randn(2, 32, 32, 16)

        with torch.no_grad():
            y_full = layer(x)
            layer.R2.fill_(0)
            y_r1_only = layer(x)

        assert not torch.allclose(
            y_full, y_r1_only
        ), "R2 contributes nothing — both corners must be active"

    @pytest.mark.parametrize("k_max", [4, 8, 12])
    def test_various_k_max_values(self, k_max):
        layer = SpectralConv2D(d_v=16, k_max1=k_max, k_max2=k_max)
        x = torch.randn(2, 64, 64, 16)
        y = layer(x)
        assert y.shape == x.shape
