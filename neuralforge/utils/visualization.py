"""
Loss visualisation utilities for neuralforge.

Provides a single, flexible function for plotting training and
validation loss curves for one or multiple models.

The history format used throughout neuralforge:
    {
        "train_loss": [0.91, 0.45, 0.23, ...],   # one float per epoch
        "val_loss":   [0.88, 0.47, 0.21, ...],   # one float per epoch
    }

Usage — single model:
    from neuralforge.utils.visualisation import plot_loss

    trainer.fit(epochs=500)
    plot_loss(trainer.history, title="Burgers FNO1D")

Usage — multiple models (ablation comparison):
    plot_loss(
        histories={
            "k_max=8":  trainer_kmax8.history,
            "k_max=12": trainer_kmax12.history,
            "k_max=16": trainer_kmax16.history,
        },
        title="k_max Ablation — Burgers FNO1D",
        save_path="docs/benchmarks/figures/kmax_ablation.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Type aliases ──────────────────────────────────────────────────────────────

# A single training history dict
History = dict[str, list[float]]

# Either one history or a named dict of histories
HistoryInput = Union[History, dict[str, History]]


# ── Colour palette — consistent across all neuralforge plots ─────────────────

_PALETTE = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]

_TRAIN_ALPHA = 1.0
_VAL_ALPHA = 0.55
_LINEWIDTH = 1.8
_GRID_ALPHA = 0.25


# ── Helper functions ──────────────────────────────────────────────────────────


def _is_single_history(h: HistoryInput) -> bool:
    """
    Detect whether the input is one history or a dict of histories.

    Single history:  {"train_loss": [...], "val_loss": [...]}
    Multiple:        {"model_a": {"train_loss": ...}, "model_b": {...}}
    """
    if not isinstance(h, dict):
        raise TypeError(f"Expected dict, got {type(h)}")
    first_val = next(iter(h.values()))
    return isinstance(first_val, list)


def _normalise_input(h: HistoryInput) -> dict[str, History]:
    """
    Convert any valid input to {label: history} format.

    Single history {"train_loss": [...]} becomes {"Model": {...}}.
    Multiple histories are returned as-is.
    """
    if _is_single_history(h):
        return {"Model": h}
    return h  # type: ignore[return-value]


def _smooth(values: list[float], window: int) -> np.ndarray:
    """
    Apply a simple moving average for visual clarity.

    Args:
        values: Raw loss values per epoch.
        window: Number of epochs to average over.

    Returns:
        Smoothed array of the same length as values.
    """
    if window <= 1 or len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    # Use 'valid' convolution then pad edges with raw values
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = len(values) - len(smoothed)
    left_pad = pad // 2
    right_pad = pad - left_pad
    return np.concatenate(
        [
            np.array(values[:left_pad]),
            smoothed,
            np.array(values[len(values) - right_pad :]) if right_pad else [],
        ]
    )


# ── Main plotting function ────────────────────────────────────────────────────


def plot_loss(
    histories: HistoryInput,
    *,
    title: str = "Training Loss",
    train_key: str = "train_loss",
    val_key: str = "val_loss",
    log_scale: bool = True,
    smooth_window: int = 1,
    show_best_val: bool = True,
    show_lr_drops: list[int] | None = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: str | None = None,
    dpi: int = 150,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training and validation loss curves for one or more models.

    Works with any model that stores a history dict during training.
    Single model or ablation comparison — same function handles both.

    Args:
        histories:
            Single history dict:
                {"train_loss": [...], "val_loss": [...]}
            Or named dict of histories (ablation comparison):
                {
                    "k_max=8":  {"train_loss": [...], "val_loss": [...]},
                    "k_max=16": {"train_loss": [...], "val_loss": [...]},
                }

        title:
            Plot title string.

        train_key:
            Key name for training loss in each history dict.
            Default "train_loss".

        val_key:
            Key name for validation loss in each history dict.
            Default "val_loss".

        log_scale:
            If True, use log scale on y-axis. Recommended for loss curves
            that drop orders of magnitude (e.g. 0.9 → 0.015 over 500 epochs).

        smooth_window:
            Moving average window in epochs. 1 = no smoothing.
            Useful for noisy loss curves. Smoothed line is semi-transparent
            overlay; original raw values are always shown faint underneath.

        show_best_val:
            If True, mark the epoch of best validation loss with a
            vertical dashed line and annotate the value.

        show_lr_drops:
            Optional list of epoch numbers where learning rate was halved.
            Example: [100, 200, 300, 400] for StepLR(step=100).
            Draws vertical grey lines to explain loss curve changes.

        figsize:
            Matplotlib figure size (width, height) in inches.

        save_path:
            If provided, save the figure to this path.
            Parent directories are created automatically.
            Supports .png, .pdf, .svg.

        dpi:
            Resolution for saved figure. 150 is good for papers.

        show:
            If True, call plt.show(). Set False when running in scripts.

    Returns:
        matplotlib Figure object (can be further customised by caller).

    Examples:
        # Single model
        plot_loss(trainer.history, title="Burgers FNO1D — 500 epochs")

        # Ablation study
        plot_loss(
            {
                "k_max=8":  t1.history,
                "k_max=12": t2.history,
                "k_max=16": t3.history,
            },
            title="k_max Ablation",
            log_scale=True,
            show_lr_drops=[100, 200, 300, 400],
            save_path="docs/benchmarks/figures/kmax_ablation.png",
        )

        # Smooth noisy curves
        plot_loss(trainer.history, smooth_window=10)

        # Save for paper (high DPI, PDF vector format)
        plot_loss(trainer.history, save_path="paper/fig2_loss.pdf", dpi=300)
    """
    named_histories = _normalise_input(histories)
    n_models = len(named_histories)
    is_single = n_models == 1

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (label, history) in enumerate(named_histories.items()):
        color = _PALETTE[idx % len(_PALETTE)]

        # Extract loss arrays
        train_vals = history.get(train_key, [])
        val_vals = history.get(val_key, [])

        if not train_vals and not val_vals:
            raise ValueError(
                f"History for '{label}' has neither '{train_key}' "
                f"nor '{val_key}'. Check key names."
            )

        # ── Training loss ─────────────────────────────────────
        if train_vals:
            raw = np.array(train_vals)
            smoothed = _smooth(train_vals, smooth_window)

            # Raw values — faint background
            if smooth_window > 1:
                ax.plot(
                    list(range(1, len(raw) + 1)),
                    raw,
                    color=color,
                    alpha=0.15,
                    linewidth=0.8,
                    zorder=1,
                )

            # Smoothed (or raw if no smoothing)
            train_label = f"{label} — train" if not is_single else "Train loss"
            ax.plot(
                list(range(1, len(smoothed) + 1)),
                smoothed,
                color=color,
                alpha=_TRAIN_ALPHA,
                linewidth=_LINEWIDTH,
                label=train_label,
                zorder=3,
            )

        # ── Validation loss ───────────────────────────────────
        if val_vals:
            raw_val = np.array(val_vals)
            smoothed_val = _smooth(val_vals, smooth_window)

            if smooth_window > 1:
                ax.plot(
                    list(range(1, len(raw_val) + 1)),
                    raw_val,
                    color=color,
                    alpha=0.15,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=1,
                )

            val_label = f"{label} — val" if not is_single else "Val loss"
            ax.plot(
                list(range(1, len(smoothed_val) + 1)),
                smoothed_val,
                color=color,
                alpha=_VAL_ALPHA,
                linewidth=_LINEWIDTH,
                linestyle="--",
                label=val_label,
                zorder=3,
            )

            # Mark best validation epoch
            if show_best_val and val_vals:
                best_epoch = int(np.argmin(raw_val)) + 1
                best_loss = float(np.min(raw_val))
                ax.axvline(
                    best_epoch,
                    color=color,
                    alpha=0.4,
                    linewidth=1.0,
                    linestyle=":",
                    zorder=2,
                )
                ax.annotate(
                    f"  best {best_loss:.4f}\n  @ep{best_epoch}",
                    xy=(best_epoch, best_loss),
                    fontsize=7,
                    color=color,
                    alpha=0.8,
                )

    # ── LR drop markers ───────────────────────────────────────────────────────
    if show_lr_drops:
        for ep in show_lr_drops:
            ax.axvline(
                ep,
                color="#888888",
                alpha=0.35,
                linewidth=1.0,
                linestyle="-.",
                zorder=1,
            )
        # Add a single legend entry for all LR drop lines
        ax.axvline(
            show_lr_drops[0],
            color="#888888",
            alpha=0.35,
            linewidth=1.0,
            linestyle="-.",
            label="LR halved",
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Relative L2 Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=_GRID_ALPHA, which="both")
    ax.set_xlim(left=1)

    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()

    return fig


# ── Convenience wrapper for paper-quality output ──────────────────────────────


def plot_loss_paper(
    histories: HistoryInput,
    title: str,
    save_path: str,
    **kwargs,
) -> plt.Figure:
    """
    Wrapper for paper-quality figures.

    Uses high DPI, PDF format, no interactive display.
    All other arguments passed through to plot_loss().

    Args:
        histories: Same as plot_loss().
        title:     Figure title.
        save_path: Output path — use .pdf for vector quality.

    Example:
        plot_loss_paper(
            trainer.history,
            title="FNO1D on Burgers — Li et al. 2021 Reproduction",
            save_path="paper/figures/fig_loss_burgers.pdf",
        )
    """
    return plot_loss(
        histories,
        title=title,
        save_path=save_path,
        dpi=300,
        show=False,
        **kwargs,
    )
