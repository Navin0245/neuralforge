import torch
import torch.nn as nn


class LpLoss(nn.Module):
    """
    Computes the Relative Lp Loss between predicted and true tensors.

    This metric is the standard for evaluating Neural Operators (like FNOs)
    because it provides a scale-invariant measure of error, approximating
    the continuous error between two physical functions.

    Mathematical Formulation:
        For a single sample, the relative error is defined as:
            Relative Error = || u_pred - u_true ||_p / || u_true ||_p

        For a batch of size B (with reduction='mean'):
            Loss = (1 / B) * SUM_{i=1}^{B} [ || u_pred_i - u_true_i ||_p / || u_true_i ||_p ]

    Args:
        p (int or float): The order of the norm (default: 2, for L2 norm / energy).
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default): The batch loss is the average of the individual sample losses.
            'sum': The batch loss is the total sum of the individual sample losses.

    Inputs:
        u_pred (torch.Tensor): The prediction from the Neural Operator.
            Shape: (batch_size, channels, spatial_dims...)
        u_true (torch.Tensor): The ground truth data.
            Shape: (batch_size, channels, spatial_dims...) - MUST match u_pred.

    Returns:
        torch.Tensor: A 0-dimensional tensor (scalar) representing the computed loss.

    Reference:
        Li et al. 2021 — Fourier Neural Operator for Parametric PDEs
        arXiv 2010.08895, Section 5 (used for all benchmark results)

    Example:
        >>> loss_fn = LpLoss(p=2, reduction="mean")
        >>> u_pred = torch.randn(8, 64, 1)
        >>> u_true = torch.randn(8, 64, 1)
        >>> loss = loss_fn(u_pred, u_true)
        >>> loss.shape
        torch.Size([])
    """

    def __init__(self, p=2, reduction="mean", eps=1e-8):
        super().__init__()
        assert reduction in (
            "mean",
            "sum",
        ), f"reduction must be 'mean' or 'sum', got '{reduction}'"
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, u_pred, u_true):
        # Step 1: Flatten spatial/channel dims per sample.
        # Transforms shape from (batch, channels, x, y...) to (batch, num_elements).
        # This preserves the batch dimension so we can compute norms per-sample.
        u_pred_flat = torch.flatten(u_pred, start_dim=1)
        u_true_flat = torch.flatten(u_true, start_dim=1)

        # Step 2: Compute numerator per sample.
        # Calculates the absolute error magnitude: || u_pred - u_true ||_p
        # dim=1 applies the norm across the flattened spatial/channel elements.
        numerator = torch.linalg.norm(u_pred_flat - u_true_flat, ord=self.p, dim=1)

        # Step 3: Compute denominator per sample.
        # Calculates the true physical magnitude: || u_true ||_p
        # .clamp(min=1e-8) is critical! It prevents ZeroDivisionError if the
        # ground truth happens to be completely zero (e.g., fluid at rest).
        denominator = torch.linalg.norm(u_true_flat, ord=self.p, dim=1).clamp(
            min=self.eps
        )

        # Step 4: Compute relative error per sample.
        # This ensures scale invariance.
        # Shape of relative_error: (batch_size,)
        relative_error = numerator / denominator

        # Step 5: Reduce across the batch to output a single scalar value.
        if self.reduction == "mean":
            return torch.mean(relative_error)
        elif self.reduction == "sum":
            return torch.sum(relative_error)
        else:
            raise NotImplementedError(f"Reduction {self.reduction} not implemented")
