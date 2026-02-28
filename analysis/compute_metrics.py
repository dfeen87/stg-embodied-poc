"""analysis/compute_metrics.py

Metric computation from episode logs produced by run_experiment.py.

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


def compute_metrics(episode_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary metrics from a single episode result dict.

    Parameters
    ----------
    episode_result:
        Dict as returned by ``run_experiment.run_episode``.

    Returns
    -------
    dict
        Summary metrics for the episode.
    """
    logs = episode_result.get("step_logs", [])
    if not logs:
        return {
            "condition": episode_result.get("condition", "unknown"),
            "seed": episode_result.get("seed", -1),
            "n_steps": 0,
            "mean_phi": float("nan"),
            "mean_delta_phi": float("nan"),
            "H_T": episode_result.get("H_T", float("nan")),
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
        "H_T": episode_result.get("H_T", float("nan")),
        "violations": episode_result.get("violations", 0),
        "success": episode_result.get("success", False),
        "total_reward": total_reward,
        "mode_execute_frac": execute_frac,
        "mode_verify_frac": verify_frac,
        "mode_safe_frac": safe_frac,
    }


def summarise_runs(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate metrics across multiple episode results.

    Parameters
    ----------
    results:
        List of dicts as returned by ``run_experiment.run_episode``.

    Returns
    -------
    pd.DataFrame
        One row per episode with all computed metrics.
    """
    rows = [compute_metrics(r) for r in results]
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

    summary_df = summarise_runs(results)
    print("\n=== Per-episode metrics ===")
    print(summary_df.to_string(index=False))

    print("\n=== Condition averages ===")
    cols = [c for c in ("H_T", "violations", "success", "mean_phi") if c in summary_df.columns]
    print(summary_df.groupby("condition")[cols].mean().to_string())

    out_path = results_dir / "summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
