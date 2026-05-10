from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
            self.ax.text(
                x * 1.05, self.ax.get_ylim()[1] * 0.1, text, fontsize=8, color=color
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
