"""Matplotlib figures used by the Streamlit dashboard."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson


COLORS = {
    "bg": "#111820",
    "grid": "#2b3943",
    "text": "#d7e2e8",
    "muted": "#84949d",
    "teal": "#41d6c3",
    "amber": "#f2b84b",
    "red": "#ff6172",
}


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["bg"])
    axis.tick_params(colors=COLORS["muted"], labelsize=9)
    axis.xaxis.label.set_color(COLORS["muted"])
    axis.yaxis.label.set_color(COLORS["muted"])
    axis.title.set_color(COLORS["text"])
    for spine in axis.spines.values():
        spine.set_color(COLORS["grid"])
    axis.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.7)


def traffic_figure(events: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 3.35), facecolor=COLORS["bg"])
    _style_axis(axis)
    if events.empty:
        axis.text(0.5, 0.5, "Waiting for the first observation", color=COLORS["muted"], ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        x = events["window_number"].astype(int).to_numpy()
        counts = events["observed_count"].astype(float).to_numpy()
        expected = events["expected_count"].astype(float).to_numpy()
        threshold = events["threshold"].astype(float).to_numpy()
        lower = np.maximum(0, expected - (threshold - expected))
        axis.plot(x, counts, color=COLORS["teal"], marker="o", markersize=3.8, linewidth=1.8, label="Observed packets", zorder=3)
        axis.plot(x, expected, color=COLORS["amber"], linewidth=1.35, linestyle="--", label="Expected λ")
        axis.plot(x, threshold, color=COLORS["red"], linewidth=1.25, linestyle=":", label="Upper burst threshold")
        axis.plot(x, lower, color=COLORS["muted"], linewidth=1, linestyle=":", alpha=0.65, label="Lower guide")
        burst_rows = events[events["status"] == "BURST DETECTED"]
        if not burst_rows.empty:
            axis.scatter(burst_rows["window_number"], burst_rows["observed_count"], color=COLORS["red"], edgecolors="#fff0f0", linewidths=0.5, s=66, zorder=5, label="Detected burst")
        axis.set_xlabel("Window number")
        axis.set_ylabel("Packets per window")
        axis.set_title("Packet arrival counts", loc="left", fontsize=12, pad=12)
        axis.legend(frameon=False, ncol=4, fontsize=8, labelcolor=COLORS["muted"], loc="upper left")
        if len(x) > 12:
            axis.set_xlim(max(0, x[-1] - 59), x[-1] + 1)
    figure.tight_layout()
    return figure


def poisson_figure(expected_count: float, observed_count: int | None = None) -> plt.Figure:
    upper = min(500, max(1, int(math.ceil(expected_count + 5 * math.sqrt(expected_count) + 2))))
    if observed_count is not None:
        upper = min(500, max(upper, observed_count + 2))
    x = np.arange(upper + 1)
    probabilities = poisson.pmf(x, expected_count)
    figure, axis = plt.subplots(figsize=(5, 3), facecolor=COLORS["bg"])
    _style_axis(axis)
    axis.bar(x, probabilities, color=COLORS["teal"], width=0.82, alpha=0.8)
    if observed_count is not None and observed_count <= upper:
        axis.axvline(observed_count, color=COLORS["red"], linestyle="--", linewidth=1.5, label=f"Observed: {observed_count}")
        axis.legend(frameon=False, labelcolor=COLORS["muted"])
    axis.set_title("Poisson probability mass", loc="left", fontsize=12, pad=12)
    axis.set_xlabel("Packets in one window")
    axis.set_ylabel("Probability")
    figure.tight_layout()
    return figure


def interarrival_figure(intervals: list[float]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5, 3), facecolor=COLORS["bg"])
    _style_axis(axis)
    if intervals:
        axis.hist(intervals, bins=min(24, max(6, int(math.sqrt(len(intervals))))), density=True, color=COLORS["amber"], alpha=0.78, edgecolor=COLORS["bg"])
        mean = float(np.mean(intervals))
        x = np.linspace(0, max(intervals), 160)
        axis.plot(x, np.exp(-x / mean) / mean, color=COLORS["teal"], linewidth=1.8, label="Fitted exponential")
        axis.legend(frameon=False, labelcolor=COLORS["muted"])
    else:
        axis.text(0.5, 0.5, "Normal traffic arrivals will populate this plot", color=COLORS["muted"], ha="center", va="center", transform=axis.transAxes)
    axis.set_title("Inter-arrival times", loc="left", fontsize=12, pad=12)
    axis.set_xlabel("Seconds between arrivals")
    axis.set_ylabel("Density")
    figure.tight_layout()
    return figure
