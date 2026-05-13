import torch

from neuralforge.layers.fourier_layer import FourierLayer2D


class TestFourierLayer2D:
    """Test for complete 2D FNO layer (W branch + K branch + activation)."""

    def test_output_shape_matches_input(self):
        """
        Output shape must equal input shape (batch, s1, s2, dv).

        FourierLayer2D must be shape-preserving — it transforms
        the representation but does not change its dimensions.
        This allows T layers to be stacked without shape changes.
        """
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(8, 32, 32, 16)  # (batch=8, s1=32, s2=32, dv=16)
        y = layer(x)
        assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"

    def test_resolution_invariance(self):
        """
        Same layer must work at any grid resolution (s1, s2).

        Both branches must be resolution-invariant:
        - SpectralConv2D (K branch): invariant by design
        - Conv2d kernel_size=1 (W branch): invariant because
          it operates pointwise, independent of s1 and s2

        Reference: Li et al. 2021, Section 4
        """
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        layer.eval()

        for s in [32, 64, 128]:
            x = torch.randn(4, s, s, 16)
            y = layer(x)
            assert y.shape == x.shape, f"Resolution invariance failed at s={s}"

    def test_output_is_real_valued(self):
        """
        Output must be real-valued, not complex.

        Both branches produce real outputs:
        - K branch: irfft2 converts complex → real
        - W branch: Conv2d on real input produces real output
        Their sum must also be real.
        """
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16)
        y = layer(x)
        assert not y.is_complex(), "FourierLayer2D output is complex — must be real"

    def test_no_nan_in_output(self):
        """Output must not contain NaN or Inf."""
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16)
        y = layer(x)
        assert not torch.isnan(y).any(), "Output contains NaN values"
        assert not torch.isinf(y).any(), "Output contains Inf values"

    def test_gradient_flows_to_both_r_tensors(self):
        """
        Gradients must reach BOTH R1 and R2.
        R1 handles corner 1 (top-left).
        R2 handles corner 2 (bottom-left).
        If either receives no gradient it cannot learn.
        """
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.spectral_conv.R1.grad is not None
        assert layer.spectral_conv.R2.grad is not None
        assert layer.w.weight.grad is not None, "Gradient did not flow to W branch"

    def test_both_corners_contribute(self):
        """
        R1 and R2 must both contribute to the output.
        Zero out R2 — output must differ from full output.
        """
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        layer.eval()

        x = torch.randn(4, 32, 32, 16)
        full_output = layer(x)

        # Zero out R2 weights
        with torch.no_grad():
            layer.spectral_conv.R2.zero_()

        r1_only_output = layer(x)

        assert not torch.allclose(
            full_output, r1_only_output
        ), "R2 has no effect on output — check if it contributes"

    def test_both_branches_contribute(self):
        """
        Both W and K branches must contribute to the output.

        If one branch produces all zeros, it contributes nothing
        and the layer is effectively using only one branch.
        """
        layer = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        layer.eval()

        x = torch.randn(4, 32, 32, 16)
        full_output = layer(x)

        # Zero out W branch weights
        with torch.no_grad():
            for param in layer.w.parameters():
                param.zero_()

        k_branch_only_output = layer(x)

        assert not torch.allclose(
            full_output, k_branch_only_output
        ), "W branch has no effect on output — check if it contributes"

    def test_stacking_multiple_layers(self):
        """
        Multiple FourierLayer2D layers must be stackable without shape issues.

        Each layer is shape-preserving, so stacking T layers should not change
        the shape from the original (batch, s1, s2, d_v).
        """
        layer1 = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        layer2 = FourierLayer2D(d_v=16, k_max1=4, k_max2=4)
        x = torch.randn(4, 32, 32, 16)
        y1 = layer1(x)
        y2 = layer2(y1)
        assert y1.shape == x.shape, "First layer changed shape"
        assert y2.shape == x.shape, "Second layer changed shape"
