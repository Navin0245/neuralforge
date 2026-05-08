"""
Tests for LpLoss — relative Lp loss function.

Five contracts verified:
    1. Identity: loss(u, u) = 0
    2. Non-negativity: loss >= 0 always
    3. Scale invariance: loss(k*A, k*B) = loss(A, B)
    4. Asymmetry: loss(A, B) != loss(B, A)
    5. Batch independence: mean of individuals = batch mean
"""

import torch

from neuralforge.training.losses import LpLoss


class TestLpLoss:
    def test_identity_zero_loss_on_perfect_prediction(self):
        y_true = torch.rand(32, 1, 64, 64)  # Batch of 32, 64x64 grid
        y_pred = y_true.clone()

        lossfn = LpLoss(p=2, reduction="mean")
        loss = lossfn(y_pred, y_true)
        assert torch.allclose(
            loss, torch.tensor(0.0)
        ), "Loss must be 0 for perfect predictions"
        print("Loss is Zero. Thus identity case is preserved.")

    def test_non_negativity(self):
        y_true = torch.randn(32, 1, 64, 64)
        y_pred = torch.randn(32, 1, 64, 64)

        loss_fn = LpLoss(reduction="mean")
        loss = loss_fn(y_pred, y_true)
        assert loss >= 0.0, "Relative L2 loss can never be negative"
        print("Relative Loss is Positive.")

    def test_scale_invariance(self):
        y_true = torch.rand(32, 1, 64, 64)
        y_pred = torch.rand(32, 1, 64, 64)

        loss_fn = LpLoss(reduction="mean")

        loss_standard = loss_fn(y_pred, y_true)
        loss_scaled = loss_fn(y_pred * 1000.0, y_true * 1000.0)

        assert torch.allclose(
            loss_standard, loss_scaled
        ), "Loss must be invariant to global scaling"
        print("Loss is invariant to global scaling.")

    def test_asymmetry(self):
        y_true = torch.rand(32, 1, 64, 64)
        y_pred = torch.rand(32, 1, 64, 64)

        loss = LpLoss(p=2, reduction="mean")

        loss_correct_order = loss(y_pred, y_true)
        loss_reversed = loss(y_true, y_pred)  # Swapped!

        assert not torch.allclose(
            loss_correct_order, loss_reversed
        ), "Relative L2 loss must be asymmetric"
        print("Not Symmetric!")

    def test_batch_independence(self):
        y_true_1 = torch.rand(1, 1, 64, 64)
        y_pred_1 = torch.rand(1, 1, 64, 64)

        y_true_2 = torch.rand(1, 1, 64, 64)
        y_pred_2 = torch.rand(1, 1, 64, 64)

        loss_fn = LpLoss(reduction="mean")

        # Calculate individually
        loss_1 = loss_fn(y_pred_1, y_true_1)
        loss_2 = loss_fn(y_pred_2, y_true_2)
        expected_batch_loss = (loss_1 + loss_2) / 2.0

        # Calculate together
        y_true_batch = torch.cat([y_true_1, y_true_2], dim=0)
        y_pred_batch = torch.cat([y_pred_1, y_pred_2], dim=0)
        actual_batch_loss = loss_fn(y_pred_batch, y_true_batch)

        assert torch.allclose(
            expected_batch_loss, actual_batch_loss
        ), "Batch loss must equal the mean of individual sample losses"
        print("Batch Loss is the Mean of Individual Sample Losses.")

    def test_sum_reduction(self):
        """sum reduction returns sum not mean."""
        loss_fn_mean = LpLoss(p=2, reduction="mean")
        loss_fn_sum = LpLoss(p=2, reduction="sum")
        u_pred = torch.randn(4, 64, 1)
        u_true = torch.randn(4, 64, 1)
        loss_mean = loss_fn_mean(u_pred, u_true)
        loss_sum = loss_fn_sum(u_pred, u_true)
        assert torch.allclose(
            loss_sum, loss_mean * 4
        ), "sum reduction must equal mean * batch_size"
