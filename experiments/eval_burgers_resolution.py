import os

import scipy.io
import torch
import yaml

from neuralforge.models.fno1d import FNO1D
from neuralforge.training.losses import LpLoss
from neuralforge.utils.eval import evaluate_at_resolution
from neuralforge.utils.plotter import (
    DataSeries,
    GenericLinePlotter,
    PlotConfig,
    PlotTheme,
)


def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the configuration
    config_path = "configs/burgers_fno.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 3. Build the EMPTY model architecture using config parameters
    model = FNO1D(
        da=cfg["model"]["da"],
        du=cfg["model"]["du"],
        d_v=cfg["model"]["d_v"],
        k_max=cfg["model"]["k_max"],
        n_layers=cfg["model"]["n_layers"],
    ).to(device)

    # 4. Load your saved weights!
    # Update this path to point to wherever your Trainer saved the best model.
    checkpoint_path = r"checkpoints\best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Cannot find {checkpoint_path}. Did you train the model yet?"
        )

    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Put the model in evaluation mode
    model.eval()

    # 5. Load the raw data
    data_path = cfg["data"][
        "path"
    ]  # Assuming this is in your yaml, e.g., "data/Burgers_R10.mat"
    print(f"Loading data from {data_path}...")
    data = scipy.io.loadmat(data_path)
    a = data["a"]
    u = data["u"]

    n_test = cfg["data"]["n_test"]
    batch_size = cfg["data"]["batch_size"]

    a_test_raw = a[-n_test:]
    u_test_raw = u[-n_test:]

    # 6. Run the Resolution Invariance Test
    loss_fn = LpLoss(p=2, reduction="mean")
    resolutions = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print("\n" + "=" * 50)
    print("🚀 RUNNING RESOLUTION INVARIANCE TEST 🚀")
    print("=" * 50)

    # --- DYNAMIC DATA COLLECTION ---
    res_clean, err_clean = [], []
    res_outlier, err_outlier = [], []

    for n in resolutions:
        error = evaluate_at_resolution(
            model=model,
            a_test=a_test_raw,
            u_test=u_test_raw,
            resolution=n,
            loss_fn=loss_fn,
            device=device,
            batch_size=batch_size,
        )
        print(f"n={n:5d} | Relative L2 error: {error:.6f} ({error*100:.3f}%)")

        # Dynamically separate clean points vs outliers (aliasing threshold)
        if n <= 2048:
            res_clean.append(n)
            err_clean.append(error * 100)  # percentage
        else:
            res_outlier.append(n)
            err_outlier.append(error * 100)

    # --- GENERIC PLOTTING INVOCATION ---
    theme = PlotTheme()

    # Define the canvas properties
    config = PlotConfig(
        title="Resolution Invariance — FNO1D on Burgers Equation",
        xlabel="Grid resolution $n$",
        ylabel="Relative L2 Error (%)",
        xscale="log",
        xscale_base=2,
        ylim_bottom=0.0,
    )

    plotter = GenericLinePlotter(config)

    # 1. Add the main model line
    plotter.add_series(
        DataSeries(
            x=res_clean, y=err_clean, label="FNO1D (neuralforge)", color=theme.primary
        )
    )

    # 2. Add outlier scatter points dynamically (if they exist)
    if res_outlier:
        plotter.add_series(
            DataSeries(
                x=res_outlier,
                y=err_outlier,
                label="n=4096 (Data quality limit)",
                color=theme.danger,
                marker="^",
                markersize=9,
                is_scatter_only=True,
            )
        )

    # 3. Add context lines
    plotter.add_vline(x=1024, text="training\nresolution", color=theme.muted)
    plotter.add_hline(
        y=1.49, label="Paper target (Li et al.): 1.49%", color=theme.warning
    )

    # Generate and save
    plotter.generate()
    plotter.export("docs/benchmarks/figures/burgers_resolution_invariance.png")


if __name__ == "__main__":
    main()
