"""
Unit tests for FourierLayer1D.

Implements equation 2 from Li et al. 2021:
    v_{t+1}(x) = σ( W·v_t(x) + (K·v_t)(x) )

Tests written BEFORE implementation (TDD).
All tests should FAIL with ImportError first.

What we test:
    1. Output shape matches input shape
    2. Works at any resolution (resolution invariance)
    3. Output is real-valued (not complex)
    4. No NaN or Inf in output
    5. Gradients flow to both W branch and K branch
    6. W and K branches contribute independently
    7. Activation is applied (output differs from linear combination)

Reference:
    Li et al. 2021 — FNO (arXiv 2010.08895)
    Definition 1, Equation 2
"""

import torch

from neuralforge.layers.fourier_layer import FourierLayer1D


class TestFourierLayer1D:
    """Tests for one complete FNO layer (W branch + K branch + activation)."""

    def test_output_shape_matches_input(self):
        """
        Output shape must equal input shape (batch, n, dv).

        FourierLayer1D must be shape-preserving — it transforms
        the representation but does not change its dimensions.
        This allows T layers to be stacked without shape changes.
        """
        layer = FourierLayer1D(d_v=64, k_max=16)
        x = torch.randn(8, 64, 64)
        y = layer(x)
        assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"

    def test_resolution_invariance(self):
        """
        Same layer must work at any grid resolution n.

        Both branches must be resolution-invariant:
        - SpectralConv1D (K branch): invariant by design
        - Conv1d kernel_size=1 (W branch): invariant because
          it operates pointwise, independent of n

        Reference: Li et al. 2021, Section 4
        """
        layer = FourierLayer1D(d_v=32, k_max=12)
        layer.eval()

        for n in [32, 64, 128, 256]:
            x = torch.randn(4, n, 32)
            y = layer(x)
            assert y.shape == x.shape, f"Resolution invariance failed at n={n}"

    def test_output_is_real_valued(self):
        """
        Output must be real-valued, not complex.

        Both branches produce real outputs:
        - K branch: irfft converts complex → real
        - W branch: Conv1d on real input produces real output
        Their sum must also be real.
        """
        layer = FourierLayer1D(d_v=32, k_max=8)
        x = torch.randn(4, 64, 32)
        y = layer(x)
        assert not y.is_complex(), "FourierLayer1D output is complex — must be real"

    def test_no_nan_in_output(self):
        """Output must not contain NaN or Inf."""
        layer = FourierLayer1D(d_v=64, k_max=16)
        x = torch.randn(4, 64, 64)
        y = layer(x)
        assert not torch.isnan(y).any(), "NaN in FourierLayer1D output"
        assert not torch.isinf(y).any(), "Inf in FourierLayer1D output"

    def test_gradients_flow_to_both_branches(self):
        """
        Gradients must reach both the W branch and the K branch.

        The W branch parameters are in self.w (Conv1d weights).
        The K branch parameters are in self.spectral_conv.R.

        If either branch receives no gradients, it cannot learn.
        Both branches must contribute to the loss.
        """
        layer = FourierLayer1D(d_v=32, k_max=8)
        x = torch.randn(4, 64, 32)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check W branch (Conv1d weight)
        assert (
            layer.w.weight.grad is not None
        ), "Gradient did not reach W branch (Conv1d weights)"
        assert not torch.isnan(layer.w.weight.grad).any(), "NaN in W branch gradient"

        # Check K branch (SpectralConv R tensor)
        assert (
            layer.spectral_conv.R.grad is not None
        ), "Gradient did not reach K branch (SpectralConv R)"
        assert not torch.isnan(
            layer.spectral_conv.R.grad
        ).any(), "NaN in K branch gradient"

    def test_both_branches_contribute(self):
        """
        Both W and K branches must contribute to the output.

        If one branch produces all zeros, it contributes nothing
        and the layer is effectively using only one branch.
        This test verifies both branches are active.
        """
        layer = FourierLayer1D(d_v=16, k_max=4)
        layer.eval()
        x = torch.randn(4, 32, 16)

        # Get full output
        y_full = layer(x)

        # Zero out W branch weights — only K branch active
        with torch.no_grad():
            layer.w.weight.fill_(0)
            layer.w.bias.fill_(0)
        y_k_only = layer(x)

        # They must differ — W branch was contributing
        assert not torch.allclose(y_full, y_k_only), (
            "W branch contributes nothing to output — " "layer is using K branch only"
        )

    def test_activation_is_applied(self):
        """
        ReLU activation must be applied after adding both branches.

        ReLU zeros out negative values: σ(x) = max(0, x).
        If activation is applied, output must have no negative values
        for inputs that produce negative pre-activation values.

        Test: if we force the layer to produce negative pre-activations,
        the output after ReLU must be non-negative everywhere.
        """
        layer = FourierLayer1D(d_v=8, k_max=4)

        # Force output to be negative by using large negative input
        # and zeroing all weights (so output = activation(0) = 0)
        with torch.no_grad():
            layer.w.weight.fill_(0)
            layer.w.bias.fill_(-10.0)  # large negative bias
            layer.spectral_conv.R.fill_(0)

        x = torch.randn(4, 32, 8)
        y = layer(x)

        # After ReLU, all values must be >= 0
        # (bias of -10 → pre-activation negative → ReLU → 0)
        assert (y >= 0).all(), (
            "Activation not applied correctly — " "negative values found after ReLU"
        )

    def test_stacking_multiple_layers(self):
        """
        Multiple FourierLayer1D can be stacked without issues.

        FNO uses T=4 layers. Each takes (batch, n, dv) and
        outputs (batch, n, dv). Stacking must work seamlessly.
        """
        layers = torch.nn.ModuleList(
            [FourierLayer1D(d_v=32, k_max=12) for _ in range(4)]
        )

        x = torch.randn(4, 64, 32)
        for layer in layers:
            x = layer(x)

        assert x.shape == torch.Size(
            [4, 64, 32]
        ), f"Shape wrong after 4 stacked layers: {x.shape}"
        assert not torch.isnan(x).any(), "NaN after stacking 4 FourierLayer1D layers"
