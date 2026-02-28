"""IMAGES/generate_plots.py

Generate illustration plots from VALIDATION_ANALYSIS.md simulation data.
Run from repository root: python IMAGES/generate_plots.py
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["baseline", "governor", "ablation_a", "rag"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Data from VALIDATION_ANALYSIS.md ──────────────────────────────────────────

VIOLATIONS = {"baseline": 12.4, "governor": 6.2, "ablation_a": 9.8, "rag": 12.4}

H_T = {"baseline": 0.561, "governor": 0.561, "ablation_a": 0.561, "rag": 0.589}

MEAN_PHI = {
    "baseline": 0.7132,
    "governor": 0.7132,
    "ablation_a": 0.7132,
    "rag": 0.7031,
}

MEAN_DELTA_PHI = {
    "baseline": 0.2731,
    "governor": 0.2731,
    "ablation_a": 0.2483,
    "rag": 0.2815,
}

# Mode distribution (% of 600 steps per condition)
MODE_EXECUTE = {"baseline": 100.0, "governor": 43.0, "ablation_a": 49.5, "rag": 100.0}
MODE_VERIFY = {"baseline": 0.0, "governor": 51.8, "ablation_a": 48.3, "rag": 0.0}
MODE_SAFE = {"baseline": 0.0, "governor": 5.2, "ablation_a": 2.2, "rag": 0.0}

# Per-episode violations (5 seeds each)
PER_SEED_VIOLATIONS = {
    "baseline": [15, 12, 9, 11, 15],
    "governor": [9, 6, 6, 4, 6],
    "ablation_a": [12, 10, 8, 8, 11],
    "rag": [15, 12, 9, 11, 15],
}

# ── Helper ─────────────────────────────────────────────────────────────────────


def _savefig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 1: Violations by condition ───────────────────────────────────────────


def plot_violations() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CONDITIONS))
    vals = [VIOLATIONS[c] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Mean violations per episode")
    ax.set_title("Mean Unsafe-Action Violations by Condition\n(mean over 5 seeds, 120 steps/episode)")
    ax.set_ylim(0, 16)
    ax.axhline(VIOLATIONS["baseline"], color="grey", linestyle="--", linewidth=0.8,
               label=f"baseline = {VIOLATIONS['baseline']}")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig1_violations_by_condition.png")


# ── Plot 2: Mode distribution stacked bar ─────────────────────────────────────


def plot_mode_distribution() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CONDITIONS))
    w = 0.55
    execute = [MODE_EXECUTE[c] for c in CONDITIONS]
    verify = [MODE_VERIFY[c] for c in CONDITIONS]
    safe = [MODE_SAFE[c] for c in CONDITIONS]

    p1 = ax.bar(x, execute, w, label="EXECUTE", color="#4C72B0")
    p2 = ax.bar(x, verify, w, bottom=execute, label="VERIFY", color="#55A868")
    execute_verify_sum = [e + v for e, v in zip(execute, verify)]
    p3 = ax.bar(x, safe, w, bottom=execute_verify_sum, label="SAFE", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Percentage of steps (%)")
    ax.set_title("Mode Distribution by Condition\n(600 steps per condition, 5 seeds)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig2_mode_distribution.png")


# ── Plot 3: H_T hallucination rate by condition ───────────────────────────────


def plot_hallucination_rate() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CONDITIONS))
    vals = [H_T[c] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("H_T (oracle-verification failure rate)")
    ax.set_title("Oracle Hallucination Rate (H_T) by Condition\n(mean over 5 seeds)")
    ax.set_ylim(0.54, 0.61)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig3_hallucination_rate.png")


# ── Plot 4: Signal statistics (mean φ and mean ΔΦ) ───────────────────────────


def plot_signal_statistics() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CONDITIONS))
    w = 0.35
    phi_vals = [MEAN_PHI[c] for c in CONDITIONS]
    delta_phi_vals = [MEAN_DELTA_PHI[c] for c in CONDITIONS]

    bars1 = ax.bar(x - w / 2, phi_vals, w, label="mean φ", color="#4C72B0",
                   edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w / 2, delta_phi_vals, w, label="mean ΔΦ", color="#DD8452",
                   edgecolor="white", linewidth=0.8)
    ax.bar_label(bars1, fmt="%.4f", padding=3, fontsize=7)
    ax.bar_label(bars2, fmt="%.4f", padding=3, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("Signal value")
    ax.set_title("Mean Coherence (φ) and Step Deviation (ΔΦ) by Condition\n(mean over 600 steps, 5 seeds)")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig4_signal_statistics.png")


# ── Plot 5: Per-seed violations heatmap ───────────────────────────────────────


def plot_per_seed_violations() -> None:
    seeds = [0, 1, 2, 3, 4]
    data = np.array([PER_SEED_VIOLATIONS[c] for c in CONDITIONS], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=16)

    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_yticks(np.arange(len(CONDITIONS)))
    ax.set_yticklabels(CONDITIONS)
    ax.set_title("Per-Seed Violations Heatmap\n(number of unsafe-action violations per episode)")

    for i in range(len(CONDITIONS)):
        for j in range(len(seeds)):
            ax.text(j, i, str(int(data[i, j])), ha="center", va="center",
                    fontsize=10, color="black" if data[i, j] < 10 else "white")

    plt.colorbar(im, ax=ax, label="Violations")
    fig.tight_layout()
    _savefig(fig, "fig5_per_seed_violations_heatmap.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_violations()
    plot_mode_distribution()
    plot_hallucination_rate()
    plot_signal_statistics()
    plot_per_seed_violations()
    print("All plots generated successfully.")
