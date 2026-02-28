"""IMAGES/generate_plots.py

Generate illustration plots from VALIDATION_ANALYSIS.md simulation data.
Run from repository root: python IMAGES/generate_plots.py
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# N = 90 (30 seeds × 3 tasks) per condition
# CI method: bootstrap percentile, B = 2000 resamples, LCG seed 77, α = 0.05
CONDITIONS = [
    "Baseline LLM",
    "LLM + RAG",
    "LLM + Governor",
    "Ablation A\n(δ=0)",
    "Ablation B\n(always-exec)",
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# ── Data from VALIDATION_ANALYSIS.md ──────────────────────────────────────────

# H_T: mean [95% CI lower, upper]
H_T_MEAN = [0.4595, 0.3679, 0.2200, 0.2422, 0.4369]
H_T_CI_LO = [0.4491, 0.3585, 0.2139, 0.2350, 0.4270]
H_T_CI_HI = [0.4702, 0.3775, 0.2266, 0.2497, 0.4473]

# Violations: mean [95% CI lower, upper]
VIOLATIONS_MEAN = [13.24, 12.56, 12.59, 13.79, 24.68]
VIOLATIONS_CI_LO = [12.59, 11.94, 11.82, 13.03, 23.80]
VIOLATIONS_CI_HI = [13.96, 13.18, 13.38, 14.49, 25.57]

# Success %: mean [95% CI lower, upper]
SUCCESS_MEAN = [34.4, 38.9, 41.1, 30.0, 0.0]
SUCCESS_CI_LO = [24.4, 28.9, 31.1, 21.1, 0.0]
SUCCESS_CI_HI = [44.4, 48.9, 51.1, 38.9, 0.0]

# Action Variance: mean [95% CI lower, upper]
ACTION_VAR_MEAN = [0.0472, 0.0470, 0.0474, 0.0475, 0.0477]
ACTION_VAR_CI_LO = [0.0465, 0.0463, 0.0466, 0.0468, 0.0470]
ACTION_VAR_CI_HI = [0.0480, 0.0477, 0.0481, 0.0482, 0.0485]

# Performance overhead data
OVERHEAD_COMPONENTS = [
    "Claim\nverif. ΔI",
    "Constraint\nchecks ΔR",
    "Contradiction\nscore ΔC",
    "Governor\nupdate",
    "Logging /\naudit trail",
]
OVERHEAD_LATENCY = [1.2, 0.9, 2.4, 0.3, 0.6]
OVERHEAD_LATENCY_ERR = [0.2, 0.1, 0.4, 0.1, 0.1]
OVERHEAD_CPU = [2.1, 1.8, 3.5, 0.4, 0.9]
OVERHEAD_MEM = [4.2, 3.1, 8.6, 1.1, 2.3]

# ── Helper ─────────────────────────────────────────────────────────────────────


def _savefig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _ci_errors(means, lo, hi):
    """Convert CI bounds to (lower_error, upper_error) arrays for errorbar."""
    return (
        np.array(means) - np.array(lo),
        np.array(hi) - np.array(means),
    )


# ── Plot 1: H_T by condition with 95% CI ──────────────────────────────────────


def plot_hallucination_rate() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CONDITIONS))
    yerr = _ci_errors(H_T_MEAN, H_T_CI_LO, H_T_CI_HI)
    bars = ax.bar(x, H_T_MEAN, color=COLORS, width=0.55, edgecolor="white",
                  linewidth=0.8)
    ax.errorbar(x, H_T_MEAN, yerr=yerr, fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt="%.4f", padding=12, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_ylabel("H_T (↓ better)")
    ax.set_title(
        "Oracle Hallucination Rate H_T by Condition\n"
        "(N=90, mean ± 95% bootstrap CI, B=2000, LCG seed 77)"
    )
    ax.set_ylim(0, 0.56)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig1_hallucination_rate.png")


# ── Plot 2: Violations by condition with 95% CI ───────────────────────────────


def plot_violations() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CONDITIONS))
    yerr = _ci_errors(VIOLATIONS_MEAN, VIOLATIONS_CI_LO, VIOLATIONS_CI_HI)
    bars = ax.bar(x, VIOLATIONS_MEAN, color=COLORS, width=0.55, edgecolor="white",
                  linewidth=0.8)
    ax.errorbar(x, VIOLATIONS_MEAN, yerr=yerr, fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt="%.2f", padding=12, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_ylabel("Mean violations per episode (↓ better)")
    ax.set_title(
        "Mean Unsafe-Action Violations by Condition\n"
        "(N=90, mean ± 95% bootstrap CI, B=2000, LCG seed 77)"
    )
    ax.set_ylim(0, 30)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig2_violations_by_condition.png")


# ── Plot 3: Success % by condition with 95% CI ────────────────────────────────


def plot_success_rate() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CONDITIONS))
    yerr = _ci_errors(SUCCESS_MEAN, SUCCESS_CI_LO, SUCCESS_CI_HI)
    bars = ax.bar(x, SUCCESS_MEAN, color=COLORS, width=0.55, edgecolor="white",
                  linewidth=0.8)
    ax.errorbar(x, SUCCESS_MEAN, yerr=yerr, fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in SUCCESS_MEAN], padding=12,
                 fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_ylabel("Success rate % (↑ better)")
    ax.set_title(
        "Task Success Rate by Condition\n"
        "(N=90, mean ± 95% bootstrap CI, B=2000, LCG seed 77)"
    )
    ax.set_ylim(0, 60)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig3_success_rate.png")


# ── Plot 4: Action Variance by condition with 95% CI ─────────────────────────


def plot_action_variance() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CONDITIONS))
    yerr = _ci_errors(ACTION_VAR_MEAN, ACTION_VAR_CI_LO, ACTION_VAR_CI_HI)
    bars = ax.bar(x, ACTION_VAR_MEAN, color=COLORS, width=0.55, edgecolor="white",
                  linewidth=0.8)
    ax.errorbar(x, ACTION_VAR_MEAN, yerr=yerr, fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt="%.4f", padding=12, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_ylabel("Action Variance (↓ better)")
    ax.set_title(
        "Action Variance by Condition\n"
        "(N=90, mean ± 95% bootstrap CI, B=2000, LCG seed 77)"
    )
    ax.set_ylim(0.045, 0.050)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig4_action_variance.png")


# ── Plot 5: Performance overhead ──────────────────────────────────────────────


def plot_performance_overhead() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(OVERHEAD_COMPONENTS))
    w = 0.6

    # Latency
    ax = axes[0]
    bars = ax.bar(x, OVERHEAD_LATENCY, w, color="#4C72B0", edgecolor="white",
                  linewidth=0.8)
    ax.errorbar(x, OVERHEAD_LATENCY, yerr=OVERHEAD_LATENCY_ERR, fmt="none",
                color="black", capsize=4, linewidth=1.2)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in OVERHEAD_LATENCY],
                 padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(OVERHEAD_COMPONENTS, fontsize=7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency per Component\n(TOTAL = 5.4 ± 0.7 ms)")
    ax.set_ylim(0, 3.5)
    ax.spines[["top", "right"]].set_visible(False)

    # CPU
    ax = axes[1]
    bars = ax.bar(x, OVERHEAD_CPU, w, color="#DD8452", edgecolor="white",
                  linewidth=0.8)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in OVERHEAD_CPU],
                 padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(OVERHEAD_COMPONENTS, fontsize=7)
    ax.set_ylabel("CPU (%)")
    ax.set_title("CPU Usage per Component\n(TOTAL = 8.7 %)")
    ax.set_ylim(0, 5)
    ax.spines[["top", "right"]].set_visible(False)

    # Memory
    ax = axes[2]
    bars = ax.bar(x, OVERHEAD_MEM, w, color="#55A868", edgecolor="white",
                  linewidth=0.8)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in OVERHEAD_MEM],
                 padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(OVERHEAD_COMPONENTS, fontsize=7)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage per Component\n(TOTAL = 19.3 MB)")
    ax.set_ylim(0, 12)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(fig, "fig5_performance_overhead.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_hallucination_rate()
    plot_violations()
    plot_success_rate()
    plot_action_variance()
    plot_performance_overhead()
    print("All plots generated successfully.")
