"""
src/plots.py
────────────
Publication-grade plotting routines for LearnMate research papers:
1. Risk-Coverage trade-off curves with target risk reference lines (5%, 10%).
2. Calibration Reliability Diagrams (before vs after calibration) with ECE and Brier annotations.
3. Ablation study comparison bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.pyplot as plt
import numpy as np

from src.metrics import compute_calibration_metrics
from src.utils import ensure_dir, get_logger

logger = get_logger("plots")

# Clean, professional style configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
})


def plot_risk_coverage_curve(
    systems_data: dict[str, dict[str, Any]],
    out_path: str | Path,
    title: str = "Risk-Coverage Trade-off",
    target_risks: Sequence[float] = (0.05, 0.10),
) -> Path:
    """
    Generate overlaid risk-coverage curve comparing multiple systems.

    Args:
        systems_data: Dict mapping system_name -> {
            'coverages': np.ndarray,
            'risks': np.ndarray,
            'aurc': float,
            'color': optional str,
            'linestyle': optional str,
        }
        out_path: Path to save figure (e.g. results/figures/risk_coverage.png).
        title: Plot title.
        target_risks: List of target risk thresholds to annotate with horizontal lines.
    """
    out_file = Path(out_path)
    ensure_dir(out_file.parent)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (sys_name, data) in enumerate(systems_data.items()):
        covs = data["coverages"]
        risks = data["risks"]
        aurc = data.get("aurc", None)

        color = data.get("color", palette[i % len(palette)])
        linestyle = data.get("linestyle", "-" if "proposed" in sys_name.lower() else "--")

        label = f"{sys_name}"
        if aurc is not None:
            label += f" (AURC={aurc:.3f})"

        ax.plot(covs, risks, label=label, color=color, linestyle=linestyle, linewidth=2.2)

    # Reference lines for target risk levels
    for tr in target_risks:
        ax.axhline(
            y=tr,
            color="#555555",
            linestyle=":",
            alpha=0.75,
            linewidth=1.2,
        )
        ax.text(
            0.02,
            tr + 0.008,
            f"Target Risk = {tr*100:.0f}%",
            color="#333333",
            fontsize=9,
            style="italic",
        )

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.02, max(0.5, ax.get_ylim()[1]))
    ax.set_xlabel("Coverage (Fraction of Questions Answered)")
    ax.set_ylabel("Empirical Risk (Error Rate on Answered Questions)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"Saved risk-coverage plot to {out_file}")
    return out_file


def plot_reliability_diagram(
    y_true: Sequence[bool],
    y_prob: Sequence[float],
    out_path: str | Path,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
    prob_uncalibrated: Sequence[float] | None = None,
) -> Path:
    """
    Generate reliability diagram (calibration curve). If prob_uncalibrated is provided,
    plots side-by-side 'Before Calibration' vs 'After Calibration' panels.
    """
    out_file = Path(out_path)
    ensure_dir(out_file.parent)

    two_panels = prob_uncalibrated is not None
    ncols = 2 if two_panels else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 5.0), dpi=300, sharey=True)
    if not two_panels:
        axes = [axes]

    configs = []
    if two_panels and prob_uncalibrated is not None:
        configs.append(("Raw / Uncalibrated", prob_uncalibrated, "#d62728"))
        configs.append(("Calibrated (LearnMate)", y_prob, "#1f77b4"))
    else:
        configs.append(("Calibrated", y_prob, "#1f77b4"))

    y_arr = np.asarray(y_true, dtype=float)

    for ax, (subtitle, p_scores, color) in zip(axes, configs):
        p_arr = np.clip(np.asarray(p_scores, dtype=float), 0.0, 1.0)
        cal_stats = compute_calibration_metrics(p_arr, y_arr, n_bins=n_bins)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_confs = []
        bin_accs = []
        bin_counts = []

        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (p_arr >= low) & (p_arr <= high)
            else:
                mask = (p_arr >= low) & (p_arr < high)

            count = np.sum(mask)
            bin_counts.append(count)
            if count > 0:
                bin_accs.append(float(np.mean(y_arr[mask])))
                bin_confs.append(float(np.mean(p_arr[mask])))
            else:
                bin_accs.append(np.nan)
                bin_confs.append((low + high) / 2.0)

        # Diagonal perfect calibration
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")

        # Bin bars
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = 1.0 / n_bins * 0.8
        valid = [not np.isnan(acc) for acc in bin_accs]

        ax.bar(
            np.array(bin_centers)[valid],
            np.array(bin_accs)[valid],
            width=width,
            alpha=0.6,
            color=color,
            edgecolor="black",
            label="Empirical accuracy",
        )

        ax.plot(
            np.array(bin_confs)[valid],
            np.array(bin_accs)[valid],
            marker="o",
            color=color,
            linewidth=2,
        )

        # Annotations (ECE and Brier)
        info_text = f"ECE: {cal_stats['ece']*100:.2f}%\nBrier: {cal_stats['brier_score']:.4f}\nMCE: {cal_stats['mce']*100:.2f}%"
        ax.text(
            0.05, 0.82, info_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            fontsize=9.5,
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy (P(correct))")
        ax.set_title(f"{subtitle}", fontweight="bold", pad=8)
        ax.grid(True)
        ax.legend(loc="lower right")

    fig.suptitle(title, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"Saved reliability diagram to {out_file}")
    return out_file


def plot_ablation_bar_chart(
    ablation_scores: dict[str, dict[str, float]],
    metric_key: str,
    metric_label: str,
    out_path: str | Path,
    title: str = "Ablation Study Comparison",
) -> Path:
    """
    Generate a bar chart comparing models on a specific metric (e.g. Unsupported-Answer Rate, AURC).
    """
    out_file = Path(out_path)
    ensure_dir(out_file.parent)

    systems = list(ablation_scores.keys())
    values = [ablation_scores[s].get(metric_key, 0.0) for s in systems]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)
    colors = ["#1f77b4" if "proposed" in s.lower() else "#aec7e8" for s in systems]

    bars = ax.barh(systems, values, color=colors, edgecolor="#333333", height=0.55)

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (max(values) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}",
            va="center",
            ha="left",
            fontsize=9.5,
        )

    ax.set_xlabel(metric_label)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(axis="x")
    ax.invert_yaxis()  # Highest on top if sorted

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"Saved ablation bar chart to {out_file}")
    return out_file
