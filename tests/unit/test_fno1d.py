"""
Unit tests for FNO1D — complete 1D Fourier Neural Operator.

Implements the full architecture from Li et al. 2021, Figure 2(a):
    a(x) → P → [FourierLayer × T] → Q → u(x)

Tests written BEFORE implementation (TDD).

What we test:
    1. Output shape: (batch, n, du)
    2. Resolution invariance
    3. No NaN in output
    4. Gradients flow end to end
    5. Different k_max values work
    6. Batch independence
    7. T layers stacked correctly

Reference:
    Li et al. 2021 — FNO (arXiv 2010.08895), Figure 2(a)
"""

import pytest
import torch

from neuralforge.models.fno1d import FNO1D


class TestFNO1D:
    def test_output_shape_1d_to_1d(self):
        """
        Standard 1D case: scalar input → scalar output.
        Paper: Burgers equation, da=1, du=1.
        Input:  (batch, n, da=1)
        Output: (batch, n, du=1)
        """
        model = FNO1D(da=1, du=1, d_v=32, k_max=12, n_layers=4)
        x = torch.randn(8, 64, 1)
        u = model(x)
        assert u.shape == torch.Size([8, 64, 1]), f"Expected (8,64,1), got {u.shape}"

    def test_output_shape_multi_channel(self):
        """
        Multi-channel input and output.
        da=5 (e.g. 5 geometric parameters for SCF problem)
        du=1 (e.g. scalar SCF field)
        """
        model = FNO1D(da=5, du=1, d_v=32, k_max=12, n_layers=4)
        x = torch.randn(4, 64, 5)
        u = model(x)
        assert u.shape == torch.Size([4, 64, 1]), f"Expected (4,64,1), got {u.shape}"

    def test_resolution_invariance(self):
        """
        Same trained model must work at any grid resolution.
        Train at n=64, evaluate at n=256 — no retraining.
        Reference: Li et al. 2021, Section 4
        """
        model = FNO1D(da=1, du=1, d_v=32, k_max=12, n_layers=4)
        model.eval()

        for n in [32, 64, 128, 256, 512]:
            x = torch.randn(2, n, 1)
            u = model(x)
            assert u.shape == torch.Size(
                [2, n, 1]
            ), f"Resolution invariance failed at n={n}: got {u.shape}"

    def test_no_nan_in_output(self):
        """Output must not contain NaN or Inf."""
        model = FNO1D(da=1, du=1, d_v=64, k_max=16, n_layers=4)
        x = torch.randn(4, 64, 1)
        u = model(x)
        assert not torch.isnan(u).any(), "NaN in FNO1D output"
        assert not torch.isinf(u).any(), "Inf in FNO1D output"

    def test_gradients_flow_end_to_end(self):
        """
        Gradients must flow from output loss back to:
        - P network (lifting layer)
        - FourierLayer weights (R and W branch)
        - Q network (projection layer)
        """
        model = FNO1D(da=1, du=1, d_v=32, k_max=8, n_layers=4)
        x = torch.randn(4, 64, 1)

        u = model(x)
        loss = u.sum()
        loss.backward()

        # P network
        assert model.p.weight.grad is not None, "No gradient in P (lifting) network"

        # FourierLayer K branch (spectral conv R)
        assert (
            model.fourier_layers[0].spectral_conv.R.grad is not None
        ), "No gradient in FourierLayer K branch"

        # FourierLayer W branch
        assert (
            model.fourier_layers[0].w.weight.grad is not None
        ), "No gradient in FourierLayer W branch"

        # Q network
        assert (
            model.q[0].weight.grad is not None
        ), "No gradient in Q (projection) network"

    def test_paper_hyperparameters_burgers(self):
        """
        Exact hyperparameters from paper for Burgers equation.
        Section 5.1: da=1, du=1, d_v=64, k_max=16, T=4
        """
        model = FNO1D(da=1, du=1, d_v=64, k_max=16, n_layers=4)
        x = torch.randn(8, 1024, 1)
        u = model(x)
        assert u.shape == torch.Size([8, 1024, 1])
        assert not torch.isnan(u).any()

    def test_batch_independence(self):
        """Each sample in batch processed independently."""
        model = FNO1D(da=1, du=1, d_v=16, k_max=4, n_layers=2)
        model.eval()

        x = torch.randn(4, 32, 1)
        u_batch = model(x)
        u_single = model(x[0:1])

        assert torch.allclose(
            u_batch[0:1], u_single, atol=1e-5
        ), "Batch processing differs from single sample"

    def test_parameter_count_scales_correctly(self):
        """
        Total parameters must scale with d_v and k_max.
        Larger d_v or k_max → more parameters.
        """
        model_small = FNO1D(da=1, du=1, d_v=16, k_max=4, n_layers=4)
        model_large = FNO1D(da=1, du=1, d_v=64, k_max=16, n_layers=4)

        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        assert params_large > params_small, "Larger model must have more parameters"

    @pytest.mark.parametrize("n_layers", [1, 2, 4, 6])
    def test_various_layer_counts(self, n_layers):
        """Model must work with any number of Fourier layers."""
        model = FNO1D(da=1, du=1, d_v=16, k_max=4, n_layers=n_layers)
        x = torch.randn(2, 32, 1)
        u = model(x)
        assert u.shape == torch.Size([2, 32, 1])
        assert not torch.isnan(u).any()

    def test_no_activation_after_final_fourier_layer(self):
        """
        The reference implementation applies no activation after the
        last Fourier layer — ReLU there would clip the hidden state
        to non-negative values before Q.
        """
        model = FNO1D(da=1, du=1, d_v=16, k_max=4, n_layers=4)
        assert isinstance(model.fourier_layers[-1].activation, torch.nn.Identity)
        for layer in model.fourier_layers[:-1]:
            assert isinstance(layer.activation, torch.nn.ReLU)
