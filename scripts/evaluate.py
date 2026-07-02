"""
Stage 3: evaluate
Loads best checkpoint, runs inference on test set,
writes metrics/eval_report.json and metrics/predictions.csv
for DVC metrics tracking.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch
import yaml

from neuralforge.models.fno1d import FNO1D
from neuralforge.training.losses import LpLoss


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = scipy.io.loadmat(cfg["data"]["path"])
    a = torch.tensor(data["a"].astype(np.float32))
    u = torch.tensor(data["u"].astype(np.float32))

    res = cfg["data"]["resolution"]
    sub = a.shape[1] // res
    a = a[:, ::sub].unsqueeze(-1)
    u = u[:, ::sub].unsqueeze(-1)
    grid = torch.linspace(0, 1, res).reshape(1, res, 1).expand(a.shape[0], -1, -1)
    inputs = torch.cat([a, grid], dim=-1)

    n_test = cfg["data"]["n_test"]
    x_test = inputs[-n_test:].to(device)
    y_test = u[-n_test:].to(device)

    # Load model
    model_cfg = cfg["model"]
    model = FNO1D(
        da=model_cfg["da"],
        du=model_cfg["du"],
        d_v=model_cfg["d_v"],
        k_max=model_cfg["k_max"],
        n_layers=model_cfg["n_layers"],
    ).to(device)

    ckpt = sorted(Path("checkpoints").glob("*.pt"))[-1]
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loss_fn = LpLoss(p=2, reduction="mean")
    with torch.no_grad():
        preds = model(x_test)
        l2 = loss_fn(preds, y_test).item()

    Path("metrics").mkdir(exist_ok=True)

    report = {
        "test_l2_error": round(l2, 6),
        "test_l2_pct": round(l2 * 100, 4),
        "paper_benchmark_pct": 1.49,
        "improvement_factor": round(1.49 / (l2 * 100), 2),
        "checkpoint": str(ckpt),
        "n_test": n_test,
    }
    Path("metrics/eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Predictions CSV for DVC plots
    sample = preds[0].squeeze().cpu().numpy()
    truth = y_test[0].squeeze().cpu().numpy()
    x_grid = np.linspace(0, 1, res)
    lines = ["x,predicted,ground_truth"]
    for xi, pi, ti in zip(x_grid, sample, truth):
        lines.append(f"{xi:.4f},{pi:.6f},{ti:.6f}")
    Path("metrics/predictions.csv").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/burgers_fno.yaml")
    args = parser.parse_args()
    main(args.config)
