"""
Unit tests for FNO2D.

Contracts:
    1. Output shape: (batch, s1, s2, du)
    2. Resolution invariance — s=32, 64, 128
    3. No NaN in output
    4. Gradients flow end to end (P, K branch, W branch, Q)
    5. Paper hyperparameters work (da=1,du=1,dv=32,k_max=12,T=4)
    6. Batch independence
    7. Parameter count scales with dv and k_max
"""

import torch

from neuralforge.models.fno2d import FNO2D


class TestFNO2D:
    def test_output_shape(self):
        # input (batch, s1, s2, da) → output (batch, s1, s2, du)
        # da=1, du=1 for Darcy flow example in paper
        model = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        x = torch.randn(8, 64, 64, 1)
        u = model(x)
        assert u.shape == torch.Size(
            [8, 64, 64, 1]
        ), f"Expected (8,64,64,1), got {u.shape}"

    def test_resolution_invariance(self):
        # Train at s=64, evaluate at s=32 and s=128
        model = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        model.eval()
        for s in [32, 64, 128]:
            x = torch.randn(4, s, s, 1)
            u = model(x)
            assert u.shape == torch.Size(
                [4, s, s, 1]
            ), f"Resolution invariance failed at s={s}: got {u.shape}"

    def test_no_nan_in_output(self):
        model = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        x = torch.randn(4, 64, 64, 1)
        u = model(x)
        assert not torch.isnan(u).any(), "Output contains NaN values"

    def test_gradients_flow_end_to_end(self):
        model = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        x = torch.randn(2, 64, 64, 1, requires_grad=True)
        u = model(x)
        loss = u.mean()
        loss.backward()
        assert x.grad is not None, "Gradients did not flow back to input"

    def test_paper_hyperparameters_darcy(self):
        # Test that paper hyperparameters run without error
        model = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        x = torch.randn(4, 64, 64, 1)
        u = model(x)
        assert u.shape == torch.Size(
            [4, 64, 64, 1]
        ), f"Expected (4,64,64,1), got {u.shape}"

    def test_batch_independence(self):
        model = FNO2D(da=1, du=1, d_v=16, k_max1=4, k_max2=4, n_layers=2)
        model.eval()

        x = torch.randn(4, 32, 32, 1)
        u_batch = model(x)
        u_single = model(x[0:1])

        assert torch.allclose(
            u_batch[0:1], u_single, atol=1e-5
        ), "Batch processing gives different result from single sample"

    def test_parameter_count_scales(self):
        model_small = FNO2D(da=1, du=1, d_v=32, k_max1=12, k_max2=12, n_layers=4)
        model_large = FNO2D(da=1, du=1, d_v=64, k_max1=12, k_max2=12, n_layers=4)

        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        assert params_large > params_small, "Larger dv must produce more parameters"
