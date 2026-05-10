import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralforge.training.losses import LpLoss


def evaluate_at_resolution(
    model: nn.Module,
    a_test: np.ndarray,
    u_test: np.ndarray,
    resolution: int,
    loss_fn: LpLoss,
    device: torch.device,
    batch_size: int = 20,
) -> float:
    """
    Evaluate the Fourier Neural Operator at a specific spatial resolution.

    Subsamples from the full high-resolution dataset to test the zero-shot
    super-resolution capability of the FNO.

    Reference: Li et al. 2021, Section 5.1 (Zero-shot super-resolution)

    Args:
        model (nn.Module): The trained FNO1D model.
        a_test (np.ndarray): Raw test initial conditions, shape (N_test, 8192).
        u_test (np.ndarray): Raw test ground truth solutions, shape (N_test, 8192).
        resolution (int): Target spatial resolution (n) to evaluate at.
        loss_fn (LpLoss): The relative Lp loss function.
        device (torch.device): Compute device (cpu/cuda).
        batch_size (int, optional): Evaluation batch size. Defaults to 20.

    Returns:
        float: The average relative L2 error across the test set.
    """
    model.eval()

    # 1. Subsample the raw 8192-point dataset
    subsample_step = 8192 // resolution
    a_sub = a_test[:, ::subsample_step]
    u_sub = u_test[:, ::subsample_step]

    # 2. Convert to float32 tensors and add channel dimension
    a_tensor = torch.tensor(a_sub, dtype=torch.float32).unsqueeze(-1)
    u_tensor = torch.tensor(u_sub, dtype=torch.float32).unsqueeze(-1)

    # 3. Build the spatial grid [0, 1] for this specific resolution
    num_samples = a_test.shape[0]
    grid = torch.linspace(0, 1, resolution, dtype=torch.float32)
    grid = grid.reshape(1, resolution, 1).expand(num_samples, -1, -1)

    # Concatenate initial condition and grid -> shape (batch, resolution, 2)
    inputs = torch.cat([a_tensor, grid], dim=-1)

    # 4. Create DataLoader
    dataset = TensorDataset(inputs, u_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_samples = 0

    # 5. Run inference safely without gradients
    with torch.no_grad():
        for x, y_true in loader:
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)

            # Accurate averaging
            current_batch_size = x.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

    return total_loss / total_samples
