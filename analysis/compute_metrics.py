"""analysis/compute_metrics.py

Metric computation from episode logs produced by run_experiment.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
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
