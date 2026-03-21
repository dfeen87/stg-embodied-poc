"""IMAGES/generate_plots.py

Generate illustration plots from VALIDATION_ANALYSIS.md simulation data.
Run from repository root: python images/generate_plots.py

Pass --fig8 to regenerate only Figure 8:
    python images/generate_plots.py --fig8
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["baseline", "llm_rag", "governor", "ablation_a", "ablation_b"]
COND_LABELS = [
    "Baseline LLM", "LLM + RAG", "LLM + Governor",
    "Ablation A\n(δ=0)", "Ablation B\n(always-exec)",
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# ── Data from VALIDATION_ANALYSIS.md ──────────────────────────────────────────

# Extended condition comparison (bootstrap 95 % CI mid-points)
VIOLATIONS = {
    "baseline": 13.24,
    "llm_rag":  12.56,
    "governor": 12.59,
    "ablation_a": 13.79,
    "ablation_b": 24.68,
}
VIOLATIONS_CI_LO = {
    "baseline": 12.59, "llm_rag": 11.94, "governor": 11.82,
    "ablation_a": 13.03, "ablation_b": 23.80,
}
VIOLATIONS_CI_HI = {
    "baseline": 13.96, "llm_rag": 13.18, "governor": 13.38,
    "ablation_a": 14.49, "ablation_b": 25.57,
}

H_T = {
    "baseline": 0.4595,
    "llm_rag":  0.3679,
    "governor": 0.2200,
    "ablation_a": 0.2422,
    "ablation_b": 0.4369,
}
H_T_CI_LO = {
    "baseline": 0.4491, "llm_rag": 0.3585, "governor": 0.2139,
    "ablation_a": 0.2350, "ablation_b": 0.4270,
}
H_T_CI_HI = {
    "baseline": 0.4702, "llm_rag": 0.3775, "governor": 0.2266,
    "ablation_a": 0.2497, "ablation_b": 0.4473,
}

SUCCESS_RATE = {
    "baseline": 34.4,
    "llm_rag":  38.9,
    "governor": 41.1,
    "ablation_a": 30.0,
    "ablation_b": 0.0,
}
SUCCESS_CI_LO = {
    "baseline": 24.4, "llm_rag": 28.9, "governor": 31.1,
    "ablation_a": 21.1, "ablation_b": 0.0,
}
SUCCESS_CI_HI = {
    "baseline": 44.4, "llm_rag": 48.9, "governor": 51.1,
    "ablation_a": 38.9, "ablation_b": 0.0,
}

MEAN_PHI = {
    "baseline": 0.7132,
    "llm_rag":  0.7031,
    "governor": 0.7132,
    "ablation_a": 0.7132,
    "ablation_b": 0.7132,
}

MEAN_DELTA_PHI = {
    "baseline": 0.2731,
    "llm_rag":  0.2815,
    "governor": 0.2731,
    "ablation_a": 0.2483,
    "ablation_b": 0.2731,
}

# Mode distribution (% of 600 steps per condition) — original 4-condition data
MODE_CONDITIONS = ["baseline", "governor", "ablation_a", "rag"]
MODE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
MODE_EXECUTE = {"baseline": 100.0, "governor": 43.0, "ablation_a": 49.5, "rag": 100.0}
MODE_VERIFY = {"baseline": 0.0, "governor": 51.8, "ablation_a": 48.3, "rag": 0.0}
MODE_SAFE = {"baseline": 0.0, "governor": 5.2, "ablation_a": 2.2, "rag": 0.0}

# Per-episode violations (5 seeds each) — original 4-condition data
PER_SEED_CONDITIONS = ["baseline", "governor", "ablation_a", "rag"]
PER_SEED_VIOLATIONS = {
    "baseline": [15, 12, 9, 11, 15],
    "governor": [9, 6, 6, 4, 6],
    "ablation_a": [12, 10, 8, 8, 11],
    "rag": [15, 12, 9, 11, 15],
}

# Performance overhead data
OVERHEAD_COMPONENTS = [
    "Claim verif.\n(ΔI)",
    "Constraint\nchecks (ΔR)",
    "Contradiction\nscore (ΔC)",
    "Governor\nupdate (ΔΦ)",
    "Logging /\naudit trail",
]
OVERHEAD_LATENCY = [1.2, 0.9, 2.4, 0.3, 0.6]
OVERHEAD_LATENCY_ERR = [0.2, 0.1, 0.4, 0.1, 0.1]
OVERHEAD_CPU = [2.1, 1.8, 3.5, 0.4, 0.9]
OVERHEAD_MEM = [4.2, 3.1, 8.6, 1.1, 2.3]
OVERHEAD_ENERGY = [0.8, 0.6, 1.4, 0.2, 0.3]

# ── Helper ─────────────────────────────────────────────────────────────────────


def _savefig(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 1: Violations by condition ───────────────────────────────────────────


def plot_violations() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS))
    vals = [VIOLATIONS[c] for c in CONDITIONS]
    yerr_lo = [VIOLATIONS[c] - VIOLATIONS_CI_LO[c] for c in CONDITIONS]
    yerr_hi = [VIOLATIONS_CI_HI[c] - VIOLATIONS[c] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi], fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt="%.2f", padding=6, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=8)
    ax.set_ylabel("Mean violations per episode")
    ax.set_title("Mean Unsafe-Action Violations by Condition\n(bootstrap 95 % CI)")
    ax.set_ylim(0, 30)
    ax.axhline(VIOLATIONS["baseline"], color="grey", linestyle="--", linewidth=0.8,
               label=f"baseline = {VIOLATIONS['baseline']:.2f}")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig1_violations_by_condition.png")


# ── Plot 2: Mode distribution stacked bar ─────────────────────────────────────


def plot_mode_distribution() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(MODE_CONDITIONS))
    w = 0.55
    execute = [MODE_EXECUTE[c] for c in MODE_CONDITIONS]
    verify = [MODE_VERIFY[c] for c in MODE_CONDITIONS]
    safe = [MODE_SAFE[c] for c in MODE_CONDITIONS]

    ax.bar(x, execute, w, label="EXECUTE", color="#4C72B0")
    ax.bar(x, verify, w, bottom=execute, label="VERIFY", color="#55A868")
    execute_verify_sum = [e + v for e, v in zip(execute, verify)]
    ax.bar(x, safe, w, bottom=execute_verify_sum, label="SAFE", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_CONDITIONS)
    ax.set_ylabel("Percentage of steps (%)")
    ax.set_title("Mode Distribution by Condition\n(600 steps per condition, 5 seeds)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig2_mode_distribution.png")


# ── Plot 3: H_T hallucination rate by condition ───────────────────────────────


def plot_hallucination_rate() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS))
    vals = [H_T[c] for c in CONDITIONS]
    yerr_lo = [H_T[c] - H_T_CI_LO[c] for c in CONDITIONS]
    yerr_hi = [H_T_CI_HI[c] - H_T[c] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi], fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt="%.4f", padding=6, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=8)
    ax.set_ylabel("H_T (oracle-verification failure rate)")
    ax.set_title("Oracle Hallucination Rate (H_T) by Condition\n(bootstrap 95 % CI)")
    ax.set_ylim(0, 0.58)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig3_hallucination_rate.png")


# ── Plot 4: Signal statistics (mean φ and mean ΔΦ) ───────────────────────────


def plot_signal_statistics() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
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
    ax.set_xticklabels(COND_LABELS, fontsize=8)
    ax.set_ylabel("Signal value")
    ax.set_title(
        "Mean Coherence (φ) and Step Deviation (ΔΦ) by Condition\n"
        "(mean over 600 steps, 5 seeds)"
    )
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig4_signal_statistics.png")


# ── Plot 5: Per-seed violations heatmap ───────────────────────────────────────


def plot_per_seed_violations() -> None:
    seeds = [0, 1, 2, 3, 4]
    data = np.array([PER_SEED_VIOLATIONS[c] for c in PER_SEED_CONDITIONS], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=16)

    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_yticks(np.arange(len(PER_SEED_CONDITIONS)))
    ax.set_yticklabels(PER_SEED_CONDITIONS)
    ax.set_title("Per-Seed Violations Heatmap\n(number of unsafe-action violations per episode)")

    for i in range(len(PER_SEED_CONDITIONS)):
        for j in range(len(seeds)):
            ax.text(j, i, str(int(data[i, j])), ha="center", va="center",
                    fontsize=10, color="black" if data[i, j] < 10 else "white")

    plt.colorbar(im, ax=ax, label="Violations")
    fig.tight_layout()
    _savefig(fig, "fig5_per_seed_violations_heatmap.png")


# ── Plot 6: Success rate by condition ─────────────────────────────────────────


def plot_success_rate() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS))
    vals = [SUCCESS_RATE[c] for c in CONDITIONS]
    yerr_lo = [SUCCESS_RATE[c] - SUCCESS_CI_LO[c] for c in CONDITIONS]
    yerr_hi = [SUCCESS_CI_HI[c] - SUCCESS_RATE[c] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi], fmt="none", color="black",
                capsize=4, linewidth=1.2)
    ax.bar_label(bars, fmt="%.1f%%", padding=6, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=8)
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Task Success Rate by Condition\n(bootstrap 95 % CI)")
    ax.set_ylim(0, 60)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig(fig, "fig6_success_rate.png")


# ── Plot 7: Governor computational overhead ───────────────────────────────────


def plot_performance_overhead() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(OVERHEAD_COMPONENTS))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # Left: latency breakdown
    ax = axes[0]
    bars = ax.bar(x, OVERHEAD_LATENCY, yerr=OVERHEAD_LATENCY_ERR, color=colors,
                  width=0.55, edgecolor="white", linewidth=0.8,
                  capsize=4, error_kw={"linewidth": 1.2})
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(OVERHEAD_COMPONENTS, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Component Latency\n(mean ± std)")
    ax.set_ylim(0, 3.5)
    ax.axhline(sum(OVERHEAD_LATENCY), color="grey", linestyle="--", linewidth=0.8,
               label=f"total = {sum(OVERHEAD_LATENCY):.1f} ms")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: CPU / Mem / Energy grouped
    ax2 = axes[1]
    width = 0.22
    x2 = np.arange(len(OVERHEAD_COMPONENTS))
    ax2.bar(x2 - width, OVERHEAD_CPU, width, label="CPU (%)", color="#4C72B0")
    ax2.bar(x2, OVERHEAD_MEM, width, label="Mem (MB)", color="#DD8452")
    ax2.bar(x2 + width, OVERHEAD_ENERGY, width, label="Energy (mJ)", color="#55A868")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(OVERHEAD_COMPONENTS, fontsize=8)
    ax2.set_ylabel("Resource usage")
    ax2.set_title("Per-Component CPU / Memory / Energy")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(fig, "fig7_performance_overhead.png")


# ── Figure 8 helpers ──────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    return pd.read_csv(path)


def _group_mean_std(
    df: pd.DataFrame, group_col: str, value_col: str
) -> tuple:
    """Return (means, stds) Series indexed by group_col for value_col."""
    grouped = df.groupby(group_col)[value_col]
    return grouped.mean(), grouped.std(ddof=1).fillna(0)


# ── Plot 8: Robustness summary (4-panel) ──────────────────────────────────────


def plot_figure_8() -> None:
    """Generate a 2×2 panel figure summarising robustness experiments."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Figure 8 — Robustness Summary", fontsize=13, fontweight="bold")

    # ── Panel A: Violation Rate vs Hallucination Rate ─────────────────────────
    # Dual Y-axes: left = violation rate (unsafe actions per episode, absolute
    # count), right = hallucination rate H_T (fraction of claims, normalised
    # [0, 1]).  The governor does NOT modify the LLM; in the primary MuJoCo
    # experiment baseline and governed conditions show identical H_T.
    ax_a = axes[0, 0]
    metrics_df = _load_csv("metrics.csv")
    panel_a_conds = ["baseline", "governor", "real_llm", "noisy_oracle"]
    panel_a_labels = ["Baseline", "Governor", "Real LLM", "Noisy Oracle"]
    viol_rates = [
        metrics_df.loc[metrics_df["condition"] == c, "violation_rate"].values[0]
        for c in panel_a_conds
    ]
    hall_rates = [
        metrics_df.loc[metrics_df["condition"] == c, "H_T"].values[0]
        for c in panel_a_conds
    ]
    x_a = np.arange(len(panel_a_conds))
    w_a = 0.35
    bars_av = ax_a.bar(x_a - w_a / 2, viol_rates, w_a,
                       label="Violation rate (unsafe actions per episode)",
                       color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax_a.set_ylabel("Violation rate (unsafe actions per episode)")
    ax_a.set_ylim(0, max(viol_rates) * 1.35)
    ax_a2 = ax_a.twinx()
    bars_ah = ax_a2.bar(x_a + w_a / 2, hall_rates, w_a,
                        label="Hallucination rate H_T (fraction of claims)",
                        color="#DD8452", edgecolor="white", linewidth=0.8)
    ax_a2.set_ylabel("Hallucination rate H_T (fraction of claims)")
    ax_a2.set_ylim(0, 1)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(panel_a_labels, fontsize=8)
    ax_a.set_title("(A) Violation Rate vs Hallucination Rate H_T\n"
                   "(governor ↓ violations; H_T identical in MuJoCo)")
    handles = [bars_av, bars_ah]
    labels_leg = [h.get_label() for h in handles]
    ax_a.legend(handles, labels_leg, fontsize=7, loc="upper right")
    ax_a.spines["top"].set_visible(False)

    # ── Panel B: Ablation Study ───────────────────────────────────────────────
    ax_b = axes[0, 1]
    ablations_df = _load_csv("ablations.csv")
    abl_conds = ["full_governor", "delta_0", "remove_I", "remove_C"]
    abl_labels = ["Full Governor", "δ=0", "remove_I", "remove_C"]
    abl_mean_series, _ = _group_mean_std(ablations_df, "condition", "violation_rate")
    abl_std_series, _ = _group_mean_std(ablations_df, "condition", "std")
    abl_means = [abl_mean_series[c] for c in abl_conds]
    abl_stds = [abl_std_series[c] for c in abl_conds]
    x_b = np.arange(len(abl_conds))
    abl_colors = ["#55A868", "#C44E52", "#8172B2", "#CCB974"]
    ax_b.bar(x_b, abl_means, yerr=abl_stds, color=abl_colors, width=0.5,
             edgecolor="white", linewidth=0.8, capsize=5,
             error_kw={"linewidth": 1.2})
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(abl_labels, fontsize=8)
    ax_b.set_ylabel("Violation rate (unsafe actions per episode)")
    ax_b.set_title("(B) Ablation Study\n(necessity of all ΔΦ components)")
    ax_b.axhline(abl_means[0], color="grey", linestyle="--", linewidth=0.8,
                 label=f"full governor = {abl_means[0]:.2f}")
    ax_b.legend(fontsize=8)
    ax_b.spines[["top", "right"]].set_visible(False)

    # ── Panel C: Sensitivity Sweep Stability ──────────────────────────────────
    # Success rate (%) and violation rate (unsafe actions per episode) share the
    # left Y-axis.  ΔΦ stability (unitless index, derived from variance of the
    # composite trust signal) uses the right Y-axis.  Values are NOT converted
    # to percent.
    ax_c = axes[1, 0]
    sens_df = _load_csv("sensitivity_delta_phi.csv")
    wf = sens_df["weight_factor"].values
    ax_c.plot(wf, sens_df["success_rate"].values, "o-", color="#4C72B0",
              label="Success rate (%)", linewidth=1.6, markersize=6)
    ax_c.plot(wf, sens_df["violation_rate"].values, "s-", color="#C44E52",
              label="Violation rate (unsafe actions per episode)",
              linewidth=1.6, markersize=6)
    ax_c.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, label="nominal (1.0)")
    ax_c.set_xlabel("Weight Factor")
    ax_c.set_ylabel("Success rate (%) / Violation rate (unsafe actions per episode)")
    ax_c.set_title("(C) Sensitivity Sweep Stability\n(±30% weight variation)")
    ax_c.spines["top"].set_visible(False)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(wf, sens_df["delta_phi_stability"].values, "^-", color="#55A868",
               label="ΔΦ stability (unitless index)", linewidth=1.6, markersize=6)
    ax_c2.set_ylabel("ΔΦ stability (unitless index)")
    ax_c2.spines["top"].set_visible(False)
    handles_c = (ax_c.get_lines() + ax_c2.get_lines())
    labels_c = [h.get_label() for h in handles_c]
    ax_c.legend(handles_c, labels_c, fontsize=7, loc="lower right")

    # ── Panel D: Real LLM vs Mock LLM ─────────────────────────────────────────
    # Dual Y-axes: left = success rate (%), right = violation rate (unsafe
    # actions per episode, absolute count).
    ax_d = axes[1, 1]
    llm_conds = ["mock_llm", "real_llm"]
    llm_labels = ["Mock LLM", "Real LLM"]
    llm_success = [
        metrics_df.loc[metrics_df["condition"] == c, "success_rate"].values[0]
        for c in llm_conds
    ]
    llm_viol = [
        metrics_df.loc[metrics_df["condition"] == c, "violation_rate"].values[0]
        for c in llm_conds
    ]
    x_d = np.arange(len(llm_conds))
    w_d = 0.35
    bars_ds = ax_d.bar(x_d - w_d / 2, llm_success, w_d,
                       label="Success rate (%)",
                       color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax_d.set_ylabel("Success rate (%)")
    ax_d.set_ylim(0, max(llm_success) * 1.45)
    ax_d2 = ax_d.twinx()
    bars_dv = ax_d2.bar(x_d + w_d / 2, llm_viol, w_d,
                        label="Violation rate (unsafe actions per episode)",
                        color="#DD8452", edgecolor="white", linewidth=0.8)
    ax_d2.set_ylabel("Violation rate (unsafe actions per episode)")
    ax_d2.set_ylim(0, max(llm_viol) * 1.45)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(llm_labels, fontsize=9)
    ax_d.set_title("(D) Real LLM vs Mock LLM\n(STG works under non-deterministic LLM)")
    handles_d = [bars_ds, bars_dv]
    labels_d = [h.get_label() for h in handles_d]
    ax_d.legend(handles_d, labels_d, fontsize=7, loc="upper right")
    ax_d.spines["top"].set_visible(False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(
        0.5, 0.03,
        "Figure 8. Robustness summary across noise conditions, ablations, and ΔΦ sensitivity. "
        "All metrics use explicit units: success rate (%), violation rate (unsafe actions per episode), "
        "hallucination rate H_T (fraction), and ΔΦ stability (unitless index). "
        "The governor reduces unsafe actions without modifying the underlying LLM, "
        "as confirmed by identical hallucination rates in MuJoCo.",
        ha="center", va="bottom", fontsize=7.5, wrap=True,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc",
                  alpha=0.8),
    )
    _savefig(fig, "fig8_robustness_summary.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots for the robotics PoC.")
    parser.add_argument("--fig8", action="store_true",
                        help="Regenerate only Figure 8 (robustness summary).")
    args = parser.parse_args()

    if args.fig8:
        plot_figure_8()
        print("Figure 8 generated successfully.")
    else:
        plot_violations()
        plot_mode_distribution()
        plot_hallucination_rate()
        plot_signal_statistics()
        plot_per_seed_violations()
        plot_success_rate()
        plot_performance_overhead()
        plot_figure_8()
        print("All plots generated successfully.")
