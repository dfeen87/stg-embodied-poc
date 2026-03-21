"""analysis/compute_metrics.py

Metric computation from episode logs produced by run_experiment.py.

Metric definitions
------------------
Three primary metrics are tracked explicitly to separate the effects of the
Spiral-Time Governor (STG) from the LLM's intrinsic properties:

  H_T             — hallucination rate: fraction of LLM claims per episode
                    that fail oracle verification.  STG does *not* reduce
                    this; it is a fixed property of the underlying LLM.
  violation_rate  — fraction of steps on which an unsafe proposed action
                    passes through to the actuators.  STG *does* reduce this
                    by switching to SAFE mode before the action is applied.
  success_rate    — fraction of episodes in which the final reward > 0.5.

Keeping H_T and violation_rate as separate columns in the output CSV makes
this distinction explicit and prevents misleading aggregations.

CLI usage
---------
    python analysis/compute_metrics.py --results_dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Columns written to metrics.csv — the three primary metrics that explicitly
# separate hallucination rate (H_T) from violation rate (V_T).
METRICS_CSV_COLS = ("condition", "seed", "H_T", "violation_rate", "success")


def compute_metrics(episode_result: Dict[str, Any], results_dir: Path | None = None) -> Dict[str, Any]:
    """Compute summary metrics from a single episode result dict.

    Parameters
    ----------
    episode_result:
        Dict as returned by ``run_experiment.run_episode``.
    results_dir:
        Optional Path to the results directory. If provided and step_logs
        are missing, it will attempt to load them from `{condition}_steps.csv`.

    Returns
    -------
    dict
        Summary metrics for the episode.
    """
    logs = episode_result.get("step_logs", [])
    condition = episode_result.get("condition", "unknown")
    seed = episode_result.get("seed", -1)

    if not logs and results_dir is not None:
        csv_path = results_dir / f"{condition}_steps.csv"
        if csv_path.exists():
            try:
                # Load the CSV and filter for the specific seed
                steps_df = pd.read_csv(csv_path)
                seed_df = steps_df[steps_df["seed"] == seed]
                # Drop condition and seed columns to match original step_logs format
                seed_df = seed_df.drop(columns=["condition", "seed"], errors="ignore")
                logs = seed_df.to_dict(orient="records")
            except Exception:
                pass

    if not logs:
        return {
            "condition": condition,
            "seed": seed,
            "n_steps": 0,
            "mean_phi": float("nan"),
            "mean_delta_phi": float("nan"),
            "H_T": episode_result.get("H_T", float("nan")),
            "violation_rate": float("nan"),
            "violations": episode_result.get("violations", 0),
            "success": episode_result.get("success", False),
            "total_reward": float("nan"),
            "mode_execute_frac": float("nan"),
            "mode_verify_frac": float("nan"),
            "mode_safe_frac": float("nan"),
        }

    df = pd.DataFrame(logs)

    n_steps = len(df)
    mean_phi = float(df["phi"].mean()) if "phi" in df.columns else float("nan")
    mean_delta_phi = (
        float(df["delta_phi"].mean()) if "delta_phi" in df.columns else float("nan")
    )
    total_reward = (
        float(df["reward"].sum()) if "reward" in df.columns else float("nan")
    )

    mode_counts = df["mode"].value_counts(normalize=True) if "mode" in df.columns else {}
    execute_frac = float(mode_counts.get("EXECUTE", 0.0))
    verify_frac = float(mode_counts.get("VERIFY", 0.0))
    safe_frac = float(mode_counts.get("SAFE", 0.0))

    return {
        "condition": episode_result.get("condition", "unknown"),
        "seed": episode_result.get("seed", -1),
        "n_steps": n_steps,
        "mean_phi": mean_phi,
        "mean_delta_phi": mean_delta_phi,
        # Hallucination rate H_T: fraction of LLM claims that fail oracle
        # verification.  STG does NOT reduce this; it is a property of the LLM.
        "H_T": episode_result.get("H_T", float("nan")),
        # violation_rate: fraction of steps on which an unsafe proposed action
        # reached the actuators.  STG DOES reduce this via SAFE-mode gating.
        "violation_rate": episode_result.get("violations", 0) / max(n_steps, 1),
        "violations": episode_result.get("violations", 0),
        "success": episode_result.get("success", False),
        "total_reward": total_reward,
        "mode_execute_frac": execute_frac,
        "mode_verify_frac": verify_frac,
        "mode_safe_frac": safe_frac,
    }


def summarise_runs(results: List[Dict[str, Any]], results_dir: Path | None = None) -> pd.DataFrame:
    """Aggregate metrics across multiple episode results.

    Parameters
    ----------
    results:
        List of dicts as returned by ``run_experiment.run_episode``.
    results_dir:
        Optional Path to the results directory.

    Returns
    -------
    pd.DataFrame
        One row per episode with all computed metrics.
    """
    rows = [compute_metrics(r, results_dir) for r in results]
    return pd.DataFrame(rows)


def main(argv: list | None = None) -> None:
    """CLI entry-point: load results.json and print a metrics summary.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="Compute and display metrics from experiment results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Directory containing results.json (default: results/)",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    json_path = results_dir / "results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found. Run run_experiment.py first.", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)

    summary_df = summarise_runs(results, results_dir)
    print("\n=== Per-episode metrics ===")
    print(summary_df.to_string(index=False))

    print("\n=== Condition averages ===")
    cols = [c for c in ("H_T", "violation_rate", "success", "mean_phi") if c in summary_df.columns]
    condition_avg = summary_df.groupby("condition")[cols].mean()
    print(condition_avg.to_string())

    out_path = results_dir / "summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSummary written to {out_path}")

    # Export the three primary metrics (H_T, violation_rate, success_rate) to
    # metrics.csv so that hallucination rate and violation rate are explicitly
    # separated and easy to compare across conditions.
    metrics_cols = [c for c in METRICS_CSV_COLS if c in summary_df.columns]
    metrics_df = summary_df[metrics_cols].copy()
    metrics_df = metrics_df.rename(columns={"success": "success_rate"})
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
