#!/usr/bin/env bash
# scripts/run_all.sh — Run the full STG experiment pipeline.
#
# Usage:
#   bash scripts/run_all.sh
#
# Steps:
#   1. Run all four conditions (baseline, governor, ablation_a, rag) with
#      seeds 0–9 (excluding seed 5, which is reserved for robustness checks).
#   2. Compute and display summary metrics from the saved results.

set -euo pipefail

echo "=== STG MuJoCo PoC — Full Experiment Pipeline ==="
echo

echo "Step 1/2: Running experiments (conditions: all, seeds: 0 1 2 3 4 6 7 8 9) ..."
python run_experiment.py --seeds 0 1 2 3 4 6 7 8 9 --conditions all

echo
echo "Step 2/2: Computing summary metrics from results/ ..."
python analysis/compute_metrics.py --results_dir results/

echo
echo "=== Pipeline complete. Results are in results/ ==="
