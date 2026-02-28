#!/usr/bin/env bash
# scripts/run_robustness.sh — Run the robustness check with held-out seeds.
#
# Usage:
#   bash scripts/run_robustness.sh
#
# Steps:
#   1. Run the governor condition only on seeds 40–49 (never seen during
#      hyperparameter tuning) to verify out-of-distribution generalisation.
#   2. Compute and display summary metrics from the saved results.

set -euo pipefail

echo "=== STG MuJoCo PoC — Robustness Check (held-out seeds 40–49) ==="
echo

echo "Step 1/2: Running governor condition on seeds 40 41 42 43 44 45 46 47 48 49 ..."
python run_experiment.py --seeds 40 41 42 43 44 45 46 47 48 49 --conditions governor

echo
echo "Step 2/2: Computing summary metrics from results/ ..."
python analysis/compute_metrics.py --results_dir results/

echo
echo "=== Robustness check complete. Results are in results/ ==="
