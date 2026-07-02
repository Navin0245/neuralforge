"""
Burgers Equation — FNO1D Experiment
=====================================
Reproduces Table 1 from Li et al. 2021:
    "Fourier Neural Operator for Parametric PDEs"
    arXiv 2010.08895, Section 5.1

Config: configs/burgers_fno.yaml

Usage:
    python experiments/burgers_fno1d.py
    python experiments/burgers_fno1d.py --config configs/burgers_fno.yaml

Results are logged to MLflow. View with:
    mlflow ui
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import urllib.request
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import mlflow
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

# ── neuralforge ───────────────────────────────────────────────────────────────
from neuralforge.models.fno1d import FNO1D
from neuralforge.training.losses import LpLoss
from neuralforge.training.trainer import Trainer
from neuralforge.utils.visualization import plot_loss

# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    """
    Load YAML config file.

    Args:
        path: Path to .yaml config file.

    Returns:
        Nested dict of config values.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {path}")
    return cfg


# ── Data ──────────────────────────────────────────────────────────────────────


def download_dataset(url: str, path: str) -> None:
    """Download Burgers dataset with retry logic."""
    if Path(path).exists():
        size_mb = Path(path).stat().st_size / 1024 / 1024
        print(f"Dataset found: {path} ({size_mb:.1f} MB)")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n  {url}")
    print("Note: file is ~500MB, this may take several minutes.")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, path)
            size_mb = Path(path).stat().st_size / 1024 / 1024
            print(f"Downloaded: {size_mb:.1f} MB → {path}")
            return
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            # Remove partial file before retry
            if Path(path).exists():
                Path(path).unlink()
            if attempt == max_retries:
                raise RuntimeError(
                    f"Download failed after {max_retries} attempts.\n"
                    f"Download manually from:\n  {url}\n"
                    f"Place file at: {path}"
                ) from e


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Burgers dataset from .mat file.

    Returns:
        a: Initial conditions u0(x), shape (2048, 8192)
        u: Solutions at t=1,   shape (2048, 8192)
    """
    data = scipy.io.loadmat(path)
    a = data["a"].astype(np.float32)
    u = data["u"].astype(np.float32)
    print(f"Loaded dataset: a={a.shape}, u={u.shape}")
    return a, u


def prepare_loaders(
    a: np.ndarray,
    u: np.ndarray,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """
    Prepare train and test DataLoaders.

    Subsamples from 8192 → resolution grid points.
    Concatenates spatial grid to input: [u0(x), x] → da=2.

    Args:
        a:   Initial conditions (N, 8192)
        u:   Solutions at t=1  (N, 8192)
        cfg: data section of config dict

    Returns:
        train_loader, test_loader
    """
    resolution = cfg["resolution"]
    n_train = cfg["n_train"]
    n_test = cfg["n_test"]
    batch_size = cfg["batch_size"]

    # Subsample from 8192 → resolution
    subsample = a.shape[1] // resolution
    a_sub = a[:, ::subsample]  # (N, resolution)
    u_sub = u[:, ::subsample]  # (N, resolution)

    # Convert to tensors and add channel dimension
    # (N, resolution) → (N, resolution, 1)
    a_tensor = torch.tensor(a_sub).unsqueeze(-1)
    u_tensor = torch.tensor(u_sub).unsqueeze(-1)

    # Spatial grid x ∈ [0, 1], broadcast to (N, resolution, 1)
    total = a_tensor.shape[0]
    grid = (
        torch.linspace(0, 1, resolution).reshape(1, resolution, 1).expand(total, -1, -1)
    )

    # Concatenate [u0(x), x] along channel dim → (N, resolution, 2)
    # da=2: position information breaks translation invariance
    inputs = torch.cat([a_tensor, grid], dim=-1)

    # Split: train = first n_train, test = last n_test
    x_train = inputs[:n_train]
    y_train = u_tensor[:n_train]
    x_test = inputs[-n_test:]
    y_test = u_tensor[-n_test:]

    print(f"x_train: {tuple(x_train.shape)}  " f"y_train: {tuple(y_train.shape)}")
    print(f"x_test:  {tuple(x_test.shape)}   " f"y_test:  {tuple(y_test.shape)}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LpLoss,
    device: torch.device,
) -> float:
    """
    Compute average relative L2 error on a DataLoader.

    Returns:
        Scalar relative L2 error averaged over all samples.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y_true in loader:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)

            batch = x.size(0)
            total_loss += loss.item() * batch
            total_samples += batch

    return total_loss / total_samples


# ── Main ──────────────────────────────────────────────────────────────────────


def main(config_path: str = "configs/burgers_fno.yaml") -> None:
    # Load config
    cfg = load_config(config_path)

    # Convenience aliases for config sections
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    mlflow_cfg = cfg["mlflow"]
    bench_cfg = cfg["benchmark"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    download_dataset(data_cfg["url"], data_cfg["path"])
    a, u = load_dataset(data_cfg["path"])
    train_loader, test_loader = prepare_loaders(a, u, data_cfg)

    # Model
    model = FNO1D(
        da=model_cfg["da"],
        du=model_cfg["du"],
        d_v=model_cfg["d_v"],
        k_max=model_cfg["k_max"],
        n_layers=model_cfg["n_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimiser and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg["scheduler_step"],
        gamma=train_cfg["scheduler_gamma"],
    )

    # Loss
    loss_fn = LpLoss(p=2, reduction="mean")

    # MLflow — log all config params before training
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    mlflow.log_params(
        {
            "da": model_cfg["da"],
            "du": model_cfg["du"],
            "d_v": model_cfg["d_v"],
            "k_max": model_cfg["k_max"],
            "n_layers": model_cfg["n_layers"],
            "n_train": data_cfg["n_train"],
            "resolution": data_cfg["resolution"],
            "batch_size": data_cfg["batch_size"],
            "epochs": train_cfg["epochs"],
            "lr": train_cfg["lr"],
            "scheduler_step": train_cfg["scheduler_step"],
            "device": str(device),
            "config_path": config_path,
        }
    )

    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        scheduler=scheduler,
        device=device,
        checkpoint_path="checkpoints",
    )

    # Train
    print(f"\nTraining for {train_cfg['epochs']} epochs...")
    trainer.fit(
        epochs=train_cfg["epochs"],
        run_name=mlflow_cfg["run_name"],
    )

    # Final evaluation on test set
    final_l2 = evaluate(model, test_loader, loss_fn, device)
    mlflow.log_metric("final_test_l2", final_l2)

    # Report
    lo, hi = bench_cfg["acceptable_range"]
    target = bench_cfg["target_l2"]

    print(f"\n{'='*50}")
    print(f"Final test L2 error: {final_l2:.6f} ({final_l2*100:.3f}%)")
    print(f"Paper target:        {target:.4f}  (1.49%)")
    print(f"Acceptable range:    ({lo:.3f}, {hi:.3f})")
    print(f"{'='*50}")

    if lo <= final_l2 <= hi:
        print("PASS — within acceptable reproducibility range")
    elif final_l2 < lo:
        print("EXCELLENT — beat the paper result")
    else:
        print("FAIL — outside acceptable range")
        print("Try: more epochs, different seed, or check hyperparameters")

    plot_loss(
        trainer.history,
        title="FNO1D on Burgers — Li et al. 2021 Reproduction",
        log_scale=True,
        show_lr_drops=[100, 200, 300, 400],
        save_path="docs/benchmarks/figures/burgers_loss.png",
    )

    import json

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/results.json").write_text(
        json.dumps({"test_l2_error": final_l2, "test_l2_pct": round(final_l2 * 100, 4)})
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNO1D on Burgers equation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/burgers_fno.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
