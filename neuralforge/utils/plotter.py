from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# 1. THEME MANAGER
# ==========================================
class PlotTheme:
    """Centralized styling to ensure consistent 'neuralforge' branding."""

    primary = "#2196F3"  # Blue
    secondary = "#4CAF50"  # Green
    danger = "#F44336"  # Red
    warning = "#FF9800"  # Orange
    muted = "#888888"  # Gray

    title_size = 13
    label_size = 12
    legend_size = 9
    dpi = 150
    grid_alpha = 0.25  # <-- FIX: Added the missing attribute!


# ==========================================
# 2. DATA STRUCTURES
# ==========================================
@dataclass
class DataSeries:
    """Defines a single line or scatter series on the plot."""

    x: List[Union[float, int]]
    y: List[Union[float, int]]
    label: str
    color: Optional[str] = None
    marker: str = "o"
    linestyle: str = "-"
    linewidth: int = 2
    markersize: int = 7
    is_scatter_only: bool = False  # Set to True for isolated points (like outliers)


@dataclass
class PlotConfig:
    """Configuration for the overall plot canvas."""

    title: str
    xlabel: str
    ylabel: str
    xscale: str = "linear"  # e.g., 'linear', 'log'
    yscale: str = "linear"
    xscale_base: int = 10  # e.g., 2 for resolutions
    ylim_bottom: Optional[float] = None
    show_grid: bool = True
    legend_loc: str = "best"


# ==========================================
# 3. GENERIC PLOTTER CLASS
# ==========================================
class GenericLinePlotter:
    """A generic class capable of plotting multiple data series."""

    def __init__(self, config: PlotConfig):
        self.config = config
        self.series_list: List[DataSeries] = []
        self.hlines = []
        self.vlines = []
        self.theme = PlotTheme()
        self.fig, self.ax = plt.subplots(figsize=(9, 5))

    def add_series(self, series: DataSeries):
        """Adds a line/scatter series to the plot."""
        self.series_list.append(series)

    def add_hline(self, y: float, label: str, color: str, linestyle: str = "--"):
        """Adds a horizontal benchmark line."""
        self.hlines.append((y, label, color, linestyle))

    def add_vline(self, x: float, text: str, color: str, linestyle: str = "--"):
        """Adds a vertical marker line with text."""
        self.vlines.append((x, text, color, linestyle))

    def generate(self) -> tuple:
        """Draws the plot based on the provided configuration and series."""
        self.ax.set_title(
            self.config.title, fontsize=self.theme.title_size, fontweight="bold"
        )
        self.ax.set_xlabel(self.config.xlabel, fontsize=self.theme.label_size)
        self.ax.set_ylabel(self.config.ylabel, fontsize=self.theme.label_size)

        # Scale logic
        if self.config.xscale == "log":
            self.ax.set_xscale("log", base=self.config.xscale_base)
            self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            self.ax.set_xscale(self.config.xscale)

        if self.config.ylim_bottom is not None:
            self.ax.set_ylim(bottom=self.config.ylim_bottom)

        all_x_ticks = set()

        # Plot all data series dynamically
        for s in self.series_list:
            all_x_ticks.update(s.x)
            if s.is_scatter_only:
                self.ax.plot(
                    s.x,
                    s.y,
                    marker=s.marker,
                    color=s.color or self.theme.primary,
                    linestyle="",
                    markersize=s.markersize,
                    label=s.label,
                    zorder=3,
                )
            else:
                self.ax.plot(
                    s.x,
                    s.y,
                    marker=s.marker,
                    color=s.color or self.theme.primary,
                    linestyle=s.linestyle,
                    linewidth=s.linewidth,
                    markersize=s.markersize,
                    label=s.label,
                    zorder=3,
                )

        # Ensure ticks show up correctly for log plots
        if all_x_ticks and self.config.xscale == "log":
            self.ax.set_xticks(sorted(list(all_x_ticks)))

        # Plot structural lines
        for y, label, color, ls in self.hlines:
            self.ax.axhline(
                y, color=color, linestyle=ls, label=label, alpha=0.8, linewidth=1.5
            )

        for x, text, color, ls in self.vlines:
            self.ax.axvline(
                x, color=color, linestyle=ls, alpha=0.6, linewidth=1, zorder=1
            )
            # Offset text by 2% of the x-range; place near the top (90% of y-range).
            # Both limits are accurate here because all series have already been plotted.
            x_offset = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.02
            self.ax.text(
                x + x_offset,
                self.ax.get_ylim()[1] * 0.9,
                text,
                fontsize=8,
                color=color,
                va="top",
            )

        if self.config.show_grid:
            self.ax.grid(True, alpha=self.theme.grid_alpha, which="both")

        self.ax.legend(fontsize=self.theme.legend_size, loc=self.config.legend_loc)
        plt.tight_layout()
        return self.fig, self.ax

    def export(self, filepath: str):
        """Safely saves the figure and cleans up memory."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=self.theme.dpi, bbox_inches="tight")
        print(f"✅ Figure exported to: {path.absolute()}")
        plt.close(self.fig)


# ==========================================
# 4. 2D FIELD COMPARISON PLOTTER
# ==========================================
class FieldComparePlotter:
    """
    Visualises 2D field predictions for any spatial neural operator.

    Works with any model mapping (batch, s1, s2, da) → (batch, s1, s2, du).
    Channel 0 of x_test is treated as the physical input field for display.

    Usage (resolution invariance via GenericLinePlotter):
        plotter = GenericLinePlotter(PlotConfig(
            title="Resolution Invariance", xlabel="Grid s×s", ylabel="Rel L2 Error (%)",
        ))
        plotter.add_series(DataSeries(x=resolutions, y=errors, label="FNO2D"))
        plotter.add_hline(y=target_pct, label="Paper target", color=PlotTheme.warning)
        plotter.add_vline(x=train_res, text="train\nres", color=PlotTheme.muted)
        fig, ax = plotter.generate()
        plotter.export("figures/resolution_invariance.png")
    """

    def __init__(self, theme: Optional[PlotTheme] = None) -> None:
        self.theme = theme or PlotTheme()

    def plot_predictions(
        self,
        model: nn.Module,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        device: torch.device,
        n_samples: int = 4,
        title: str = "Prediction vs Ground Truth",
        save_path: Optional[str] = None,
    ) -> tuple:
        """
        Side-by-side grid: Input | Ground Truth | Prediction | Error.

        Args:
            model:     Trained model (batch, s, s, da) → (batch, s, s, du).
            x_test:    Test inputs  (N, s, s, da).
            y_test:    Test targets (N, s, s, 1).
            device:    Compute device.
            n_samples: Rows to display.
            title:     Figure suptitle.
            save_path: If set, saves to this path.

        Returns:
            fig, axes
        """
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test[:n_samples].to(device)).cpu()

        y_true = y_test[:n_samples]
        a = x_test[:n_samples, :, :, 0]

        fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.5 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for col, t in enumerate(
            ["Input a(x)", "Ground Truth u(x)", "Prediction û(x)", "Error |u - û|"]
        ):
            axes[0, col].set_title(t, fontsize=11, fontweight="bold", pad=8)

        for i in range(n_samples):
            a_np = a[i].numpy()
            u_np = y_true[i, :, :, 0].numpy()
            pred_np = y_pred[i, :, :, 0].numpy()
            error_np = np.abs(u_np - pred_np)
            rel_err = np.linalg.norm(pred_np - u_np) / (np.linalg.norm(u_np) + 1e-8)
            vmin_u = min(u_np.min(), pred_np.min())
            vmax_u = max(u_np.max(), pred_np.max())

            im0 = axes[i, 0].imshow(a_np, cmap="viridis", origin="lower")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            axes[i, 0].set_ylabel(f"Sample {i + 1}", fontsize=9)

            im1 = axes[i, 1].imshow(
                u_np, cmap="RdBu_r", origin="lower", vmin=vmin_u, vmax=vmax_u
            )
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            im2 = axes[i, 2].imshow(
                pred_np, cmap="rainbow", origin="lower", vmin=vmin_u, vmax=vmax_u
            )
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            axes[i, 2].set_xlabel(f"Rel L2: {rel_err * 100:.2f}%", fontsize=8)

            im3 = axes[i, 3].imshow(error_np, cmap="rainbow", origin="lower")
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(title, fontsize=self.theme.title_size, fontweight="bold", y=1.01)
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")
            plt.close(fig)

        return fig, axes

    def plot_error_field(
        self,
        model: nn.Module,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        device: torch.device,
        n_samples: int = 6,
        title: str = "Error Field |û(x) - u(x)|",
        save_path: Optional[str] = None,
    ) -> tuple:
        """
        Grid of error heatmaps |pred - true| for n_samples test inputs.

        Bright regions highlight where the model makes the largest errors.

        Args:
            model:     Trained model.
            x_test:    Test inputs  (N, s, s, da).
            y_test:    Test targets (N, s, s, 1).
            device:    Compute device.
            n_samples: Number of samples (laid out in rows of 3).
            title:     Figure suptitle.
            save_path: If set, saves to this path.

        Returns:
            fig, axes
        """
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test[:n_samples].to(device)).cpu()

        y_true = y_test[:n_samples]
        ncols = min(3, n_samples)
        nrows = (n_samples + ncols - 1) // ncols

        # squeeze=False guarantees a 2D array of Axes regardless of nrows/ncols,
        # avoiding the bare-Axes case when both dimensions equal 1.
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False
        )

        for idx in range(n_samples):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            u_np = y_true[idx, :, :, 0].numpy()
            pred_np = y_pred[idx, :, :, 0].numpy()
            error = np.abs(u_np - pred_np)
            rel_err = np.linalg.norm(pred_np - u_np) / (np.linalg.norm(u_np) + 1e-8)

            im = ax.imshow(error, cmap="rainbow", origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, label="|error|")
            ax.set_title(
                f"Sample {idx + 1}\nRel L2: {rel_err * 100:.2f}%  |  Max: {error.max():.4f}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n_samples, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(title, fontsize=self.theme.title_size, fontweight="bold")
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.theme.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")
            plt.close(fig)

        return fig, axes
