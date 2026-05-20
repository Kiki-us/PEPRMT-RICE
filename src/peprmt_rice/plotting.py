"""
Minimal diagnostic plotting.

This module is intentionally light — the goal is one sanity-check figure per
run so the user can see at a glance that nothing is on fire. Detailed
publication-quality plots belong in user notebooks downstream.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the CLI can run headless.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_fluxes(timeseries: pd.DataFrame, out_path: Path) -> Path:
    """Plot GPP / Reco / NEE / CH4 as a 4-panel figure and save to ``out_path``.

    Returns the path written.
    """
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(timeseries["Date"], timeseries["GPP"], color="tab:green", lw=1.0)
    axes[0].set_ylabel("GPP\n(g C m$^{-2}$ d$^{-1}$)")

    axes[1].plot(timeseries["Date"], timeseries["Reco"], color="tab:orange", lw=1.0)
    axes[1].set_ylabel("Reco\n(g C m$^{-2}$ d$^{-1}$)")

    axes[2].plot(timeseries["Date"], timeseries["NEE"], color="tab:blue", lw=1.0)
    axes[2].axhline(0, color="grey", lw=0.5)
    axes[2].set_ylabel("NEE\n(g C m$^{-2}$ d$^{-1}$)")

    axes[3].plot(timeseries["Date"], timeseries["CH4"], color="tab:red", lw=1.0)
    axes[3].set_ylabel("CH$_4$\n(g C m$^{-2}$ d$^{-1}$)")
    axes[3].set_xlabel("Date")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
