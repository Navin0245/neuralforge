# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import urllib.request
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

# ── neuralforge ───────────────────────────────────────────────────────────────
from neuralforge.models.fno2d import FNO2D
from neuralforge.training.losses import LpLoss
from neuralforge.training.trainer import Trainer
from neuralforge.utils.plotter import FieldComparePlotter
from neuralforge.utils.visualization import plot_loss

# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {path}")
    return cfg


# ── Data ──────────────────────────────────────────────────────────────────────


def maybe_download(path: str, url: str | None) -> None:
    p = Path(path)
    if p.exists():
        return
    if not url:
        raise FileNotFoundError(
            f"Dataset missing and no download URL configured: {path}\n"
            "Download it from the FNO datasets folder "
            "(https://github.com/neuraloperator/neuraloperator) "
            f"and place it at: {path}"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    # Google Drive large files redirect to a virus-scan confirmation page;
    # appending confirm=t bypasses it so urlretrieve gets the actual file.
    if "drive.google.com" in url and "confirm=" not in url:
        url = url + "&confirm=t"
    print(f"Downloading {url} → {path}")
    urllib.request.urlretrieve(url, str(p))
    # Validate: a .mat file starts with "MATL" or the HDF5 magic bytes.
    with open(p, "rb") as f:
        header = f.read(8)
    if not (header[:4] == b"MATL" or header[:8] == b"\x89HDF\r\n\x1a\n"):
        p.unlink()
        raise RuntimeError(
            "Download produced an HTML page instead of a .mat file "
            "(Google Drive confirmation redirect). "
            f"Download the file manually and place it at: {path}"
        )
    print("Download complete.")


def load_darcy_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load real Darcy dataset from .mat file.

    Returns:
        a: Permeability field, shape (1024, 241, 241)
        u: Pressure field,     shape (1024, 241, 241)
    """
    import scipy.io

    data = scipy.io.loadmat(path)
    a = data["coeff"].astype(np.float32)  # (1024, 241, 241)
    u = data["sol"].astype(np.float32)  # (1024, 241, 241)
    print(f"Loaded: a={a.shape}, u={u.shape}")
    return a, u


def subsample_grid(field: np.ndarray, resolution: int) -> np.ndarray:
    """
    Subsample (N, n, n) fields to (N, resolution, resolution) covering
    the FULL domain: the stride must land exactly on the last grid point,
    i.e. (n-1) must be divisible by (resolution-1). Truncating a
    non-dividing stride would silently crop the domain — the model would
    then have to predict u from a permeability field it only partially
    sees, which puts a hard floor on the achievable error.
    """
    n = field.shape[1]
    if (n - 1) % (resolution - 1) != 0:
        valid = [d + 1 for d in range(1, n) if (n - 1) % d == 0]
        raise ValueError(
            f"resolution={resolution} does not evenly subsample an n={n} grid: "
            f"(n-1) % (resolution-1) != 0. Valid resolutions: {valid}"
        )
    step = (n - 1) // (resolution - 1)
    return field[:, ::step, ::step]


def make_inputs(a_norm: np.ndarray, resolution: int) -> torch.Tensor:
    """Stack [a(x,y), x, y] into (N, resolution, resolution, 3)."""
    a_tensor = torch.tensor(a_norm).unsqueeze(-1)
    total = a_tensor.shape[0]
    gx, gy = torch.meshgrid(
        torch.linspace(0, 1, resolution),
        torch.linspace(0, 1, resolution),
        indexing="ij",
    )
    gx = gx.unsqueeze(0).unsqueeze(-1).expand(total, -1, -1, -1)
    gy = gy.unsqueeze(0).unsqueeze(-1).expand(total, -1, -1, -1)
    return torch.cat([a_tensor, gx, gy], dim=-1)


def prepare_loaders(a_train, u_train, a_test, u_test, cfg):
    resolution = cfg["resolution"]
    n_train = cfg["n_train"]
    n_test = cfg["n_test"]
    batch_size = cfg["batch_size"]

    if n_train > a_train.shape[0]:
        raise ValueError(f"n_train={n_train} > train samples={a_train.shape[0]}")
    if n_test > a_test.shape[0]:
        raise ValueError(f"n_test={n_test} > test samples={a_test.shape[0]}")

    # Subsample — full-domain, no cropping
    a_tr = subsample_grid(a_train[:n_train], resolution)
    u_tr = subsample_grid(u_train[:n_train], resolution)
    a_te = subsample_grid(a_test[:n_test], resolution)
    u_te = subsample_grid(u_test[:n_test], resolution)

    # Normalise — stats from the TRAINING set only, applied to both splits
    a_min, a_max = a_tr.min(), a_tr.max()
    u_max = u_tr.max()

    a_tr_norm = (a_tr - a_min) / (a_max - a_min)
    a_te_norm = (a_te - a_min) / (a_max - a_min)
    u_tr_norm = u_tr / u_max
    u_te_norm = u_te / u_max

    print(
        f"Normalisation (train stats): a=[{a_tr_norm.min():.3f},{a_tr_norm.max():.3f}]"
        f" u=[{u_tr_norm.min():.4f},{u_tr_norm.max():.4f}]"
    )

    x_train = make_inputs(a_tr_norm, resolution)
    y_train = torch.tensor(u_tr_norm).unsqueeze(-1)
    x_test = make_inputs(a_te_norm, resolution)
    y_test = torch.tensor(u_te_norm).unsqueeze(-1)

    print(f"x_train: {tuple(x_train.shape)}")
    print(f"y_train: {tuple(y_train.shape)}")

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
    return train_loader, test_loader, x_test, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LpLoss,
    device: torch.device,
) -> float:
    """Compute average relative L2 error on a DataLoader."""
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


def main(config_path: str = "configs/darcy_fno.yaml") -> None:
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    mlflow_cfg = cfg["mlflow"]
    bench_cfg = cfg["benchmark"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data — paper protocol: train on smooth1, test on smooth2
    maybe_download(data_cfg["train_path"], data_cfg.get("train_url"))
    maybe_download(data_cfg["test_path"], data_cfg.get("test_url"))
    a_train, u_train = load_darcy_data(data_cfg["train_path"])
    a_test, u_test = load_darcy_data(data_cfg["test_path"])
    train_loader, test_loader, x_test, y_test = prepare_loaders(
        a_train, u_train, a_test, u_test, data_cfg
    )

    # Model
    model = FNO2D(
        da=model_cfg["da"],
        du=model_cfg["du"],
        d_v=model_cfg["d_v"],
        k_max1=model_cfg["k_max1"],
        k_max2=model_cfg["k_max2"],
        n_layers=model_cfg["n_layers"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg["scheduler_step"],
        gamma=train_cfg["scheduler_gamma"],
    )

    loss_fn = LpLoss(p=2, reduction="mean")

    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    run_params = {
        "da": model_cfg["da"],
        "du": model_cfg["du"],
        "d_v": model_cfg["d_v"],
        "k_max1": model_cfg["k_max1"],
        "k_max2": model_cfg["k_max2"],
        "n_layers": model_cfg["n_layers"],
        "n_train": data_cfg["n_train"],
        "n_test": data_cfg["n_test"],
        "resolution": data_cfg["resolution"],
        "batch_size": data_cfg["batch_size"],
        "epochs": train_cfg["epochs"],
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "scheduler_step": train_cfg["scheduler_step"],
        "device": str(device),
        "config_path": config_path,
    }

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

    print(f"\nTraining for {train_cfg['epochs']} epochs...")
    trainer.fit(
        epochs=train_cfg["epochs"],
        run_name=mlflow_cfg["run_name"],
        params=run_params,
    )

    final_l2 = evaluate(model, test_loader, loss_fn, device)
    mlflow.log_metric("final_test_l2", final_l2)

    lo, hi = bench_cfg["acceptable_range"]
    target = bench_cfg["target_l2"]

    print(f"\n{'='*50}")
    print(f"Final test L2 error: {final_l2:.6f} ({final_l2*100:.3f}%)")
    print(f"Paper target:        {target:.4f}  (1.08%)")
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
        title="FNO2D on Darcy Flow — Li et al. 2021 Reproduction",
        log_scale=True,
        show_lr_drops=[100, 200, 300, 400],
        save_path="docs/benchmarks/figures/darcy_loss.png",
    )

    field_plotter = FieldComparePlotter()
    field_plotter.plot_predictions(
        model,
        x_test,
        y_test,
        device,
        n_samples=4,
        title="FNO2D on Darcy Flow — Prediction vs Ground Truth",
        save_path="docs/benchmarks/figures/darcy_predictions.png",
    )
    field_plotter.plot_error_field(
        model,
        x_test,
        y_test,
        device,
        n_samples=6,
        title="FNO2D Error Field |û(x) - u(x)| — Darcy Flow",
        save_path="docs/benchmarks/figures/darcy_error_field.png",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNO2D on Darcy flow")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/darcy_fno.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
