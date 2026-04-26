"""
Unit tests for SpectralConv1D.

Tests are written BEFORE the implementation (TDD).
Run these first — they should all FAIL with ImportError.
After implementing SpectralConv1D, all should PASS.

What we are testing:
    1. Output shape matches input shape
    2. Works at any resolution (resolution invariance)
    3. Parameter count matches theory: k_max × dv × dv × 2
    4. No NaN or Inf in output
    5. Gradients flow to learnable parameter R
    6. Each sample in batch processed independently

Reference:
    Li et al. 2021 — FNO (arXiv 2010.08895)
    Definition 3, Equation 4-5
"""

import pytest
import torch

from neuralforge.layers.spectral_conv import SpectralConv1D


class TestSpectralConv1D:
    """Tests for the 1D Fourier integral operator layer."""

    def test_output_shape_matches_input(self):
        """
        Output shape must equal input shape (batch, n, dv).

        SpectralConv1D should not change the tensor shape.
        FFT → truncate → multiply R → pad → iFFT must
        return exactly the same shape as the input.
        """
        layer = SpectralConv1D(d_v=64, k_max=16)
        x = torch.randn(8, 64, 64)  # (batch=8, n=64, dv=64)
        y = layer(x)
        assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"

    def test_resolution_invariance_same_weights(self):
        """
        Same learned weights must work at any grid resolution.

        This is the core property of FNO — R is in Fourier mode
        space, not grid space. Train at n=64, evaluate at n=256.
        Both must work with identical weights.

        Reference: Li et al. 2021, Section 4 — Invariance to
        discretization.
        """
        layer = SpectralConv1D(d_v=32, k_max=12)
        layer.eval()

        for n in [32, 64, 128, 256, 512]:
            x = torch.randn(4, n, 32)
            y = layer(x)
            assert y.shape == x.shape, (
                f"Resolution invariance failed at n={n}: "
                f"input {x.shape}, output {y.shape}"
            )

    def test_parameter_count_matches_theory(self):
        """
        R must have exactly k_max × dv × dv × 2 real parameters.

        R is stored as real tensor of shape (k_max, dv, dv, 2)
        where the last dimension stores [real_part, imag_part].

        From equation 5: R_{k,l,j} is complex-valued.
        Shape: (k_max, dv, dv) complex = (k_max, dv, dv, 2) real.

        Reference: Li et al. 2021, Equation 5
        """
        k_max, d_v = 12, 64
        layer = SpectralConv1D(d_v=d_v, k_max=k_max)

        expected = k_max * d_v * d_v * 2  # ×2 for real + imag
        actual = layer.R.numel()

        assert actual == expected, (
            f"Parameter count wrong: "
            f"expected {expected} (={k_max}×{d_v}×{d_v}×2), "
            f"got {actual}"
        )

    def test_no_nan_in_output(self):
        """
        Output must not contain NaN or Inf.

        NaN propagates silently through training and causes
        loss = nan with no useful error message.
        Catching it here at the layer level is critical.
        """
        layer = SpectralConv1D(d_v=64, k_max=16)
        x = torch.randn(4, 64, 64)
        y = layer(x)

        assert not torch.isnan(y).any(), "NaN detected in SpectralConv1D output"
        assert not torch.isinf(y).any(), "Inf detected in SpectralConv1D output"

    def test_gradient_flows_to_weight_r(self):
        """
        Gradients must flow back to the learnable weight R.

        If gradients do not reach R, the layer cannot learn.
        This test verifies the backward pass works correctly
        through: output → iFFT → R multiply → FFT → input.
        """
        layer = SpectralConv1D(d_v=32, k_max=8)
        x = torch.randn(4, 64, 32, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Gradient did not flow to input x"
        assert (
            layer.R.grad is not None
        ), "Gradient did not flow to learnable parameter self.R"
        assert not torch.isnan(layer.R.grad).any(), "NaN in gradient of self.R"

    def test_batch_independence(self):
        """
        Each sample in the batch must be processed independently.

        Processing sample 0 must give the same result whether
        sample 0 is processed alone or as part of a batch.
        This verifies no cross-contamination between samples.
        """
        layer = SpectralConv1D(d_v=16, k_max=4)
        layer.eval()

        x = torch.randn(4, 32, 16)

        y_batch = layer(x)
        y_single = layer(x[0:1])

        assert torch.allclose(y_batch[0:1], y_single, atol=1e-5), (
            "Batch processing gives different result from single sample. "
            "Samples are not being processed independently."
        )

    def test_output_is_real_valued(self):
        """
        Output must be real-valued, not complex.

        Even though FFT produces complex numbers internally,
        the final irfft must return real values.
        PyTorch will raise an error if complex tensors are
        passed to downstream real-valued layers.
        """
        layer = SpectralConv1D(d_v=32, k_max=8)
        x = torch.randn(4, 64, 32)
        y = layer(x)

        assert not y.is_complex(), (
            "SpectralConv1D output is complex — must be real. "
            "Check that irfft is applied correctly."
        )

    @pytest.mark.parametrize("k_max", [4, 8, 12, 16, 20])
    def test_various_k_max_values(self, k_max):
        """
        Layer must work for any k_max value.

        k_max is a hyperparameter tuned per problem.
        Burgers uses k_max=16, Darcy uses k_max=12.
        All must work without errors.
        """
        layer = SpectralConv1D(d_v=32, k_max=k_max)
        x = torch.randn(4, 64, 32)
        y = layer(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("d_v", [16, 32, 64, 128])
    def test_various_dv_values(self, d_v):
        """
        Layer must work for any hidden channel dimension dv.

        dv is a hyperparameter: paper uses dv=64 (1D),
        dv=32 (2D). All must work.
        """
        layer = SpectralConv1D(d_v=d_v, k_max=12)
        x = torch.randn(4, 64, d_v)
        y = layer(x)
        assert y.shape == x.shape
