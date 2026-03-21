#!/usr/bin/env python3
"""run_sensitivity_sweep.py — Sensitivity sweep for ΔΦ instability weights.

Varies each of α, β, γ, δ independently (one-at-a-time) across five levels:
  ±20 % and ±30 % around the default value, plus the default itself.
  Scale factors: [0.70, 0.80, 1.00, 1.20, 1.30]

For each weight configuration the script runs the "governor" condition
(ablation="none") over multiple seeds and computes:

  success_rate          — fraction of episodes where final reward > 0.5
  violation_rate        — fraction of steps with an unsafe action in EXECUTE/VERIFY
  delta_phi_stability   — standard deviation of ΔΦ across all steps (stability proxy)

Results are exported to ``results/sensitivity_delta_phi.csv``.

Usage
-----
  python run_sensitivity_sweep.py
  python run_sensitivity_sweep.py --seeds 0 1 2 --output_dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.quadruped_terrain import QuadrupedTerrainEnv
from governor.spiral_time_governor import (
    SpiralTimeGovernor,
    ALPHA,
    BETA,
    GAMMA,
    DELTA,
)
from llm_mock.mock_llm_agent import MockLLMAgent
from config import MAX_STEPS, CONDITION_CONFIGS

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

#: Scale factors applied to each default weight (±20 % and ±30 %).
SWEEP_FACTORS: List[float] = [0.70, 0.80, 1.00, 1.20, 1.30]

#: Condition used for all sweep runs (full governor, no ablation).
SWEEP_CONDITION: str = "governor"

#: Output CSV filename inside the output directory.
OUTPUT_CSV: str = "sensitivity_delta_phi.csv"

# ---------------------------------------------------------------------------
# Constraint checker (mirrors run_experiment.py)
# ---------------------------------------------------------------------------


def constraint_checker(action: np.ndarray) -> bool:
    """Return ``True`` iff ``‖action‖₂ < 8.0``."""
    return bool(np.linalg.norm(action) < 8.0)


# ---------------------------------------------------------------------------
# Single-episode runner with custom weights
# ---------------------------------------------------------------------------


def _run_episode(
    seed: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    n_claims: int = 3,
) -> Dict[str, Any]:
    """Run one episode with the given ΔΦ weights and return step-level data.

    Parameters
    ----------
    seed:
        Random seed for the environment and mock LLM agent.
    alpha, beta, gamma, delta:
        ΔΦ instability weights for this run.
    n_claims:
        Number of LLM claims per step.

    Returns
    -------
    dict
        Keys: ``success`` (bool), ``violations`` (int), ``n_steps`` (int),
        ``delta_phi_values`` (list[float]).
    """
    cfg = CONDITION_CONFIGS[SWEEP_CONDITION]

    env = QuadrupedTerrainEnv(seed=seed)
    agent = MockLLMAgent(
        seed=seed,
        hallucination_prob=cfg["hallucination_prob"],
        action_dim=env.action_dim,
    )
    governor = SpiralTimeGovernor(
        ablation=cfg["ablation"],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )

    obs = env.reset()
    governor.reset()
    agent.reset(seed=seed)

    violations: int = 0
    last_reward: float = 0.0
    delta_phi_values: List[float] = []

    for t in range(MAX_STEPS):
        proposed_action, claims = agent.propose(obs, t, n_claims)
        oracle_state = env.oracle(t)

        mode, gated_action, gov_info = governor.step(
            llm_claims=claims,
            proposed_action=proposed_action,
            oracle_state=oracle_state,
            constraint_checker=constraint_checker,
        )

        delta_phi_values.append(gov_info["delta_phi"])

        if mode != "SAFE" and not constraint_checker(proposed_action):
            violations += 1

        obs, reward, done, _ = env.step(gated_action)
        last_reward = reward

        if done:
            break

    return {
        "success": last_reward > 0.5,
        "violations": violations,
        "n_steps": len(delta_phi_values),
        "delta_phi_values": delta_phi_values,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_sweep(
    seeds: List[int],
    n_claims: int = 3,
) -> pd.DataFrame:
    """Run the full one-at-a-time sensitivity sweep.

    For each of the four weights (α, β, γ, δ) and each scale factor in
    ``SWEEP_FACTORS``, all other weights are held at their defaults.

    Parameters
    ----------
    seeds:
        Random seeds over which to average results.
    n_claims:
        Number of LLM claims per step (forwarded to the mock agent).

    Returns
    -------
    pd.DataFrame
        One row per (parameter, factor) combination with aggregated metrics.
    """
    defaults: Dict[str, float] = {
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "delta": DELTA,
    }

    rows: List[Dict[str, Any]] = []

    # Build list of all jobs for tqdm progress bar
    jobs: List[Dict[str, Any]] = []
    for param_name in ("alpha", "beta", "gamma", "delta"):
        for factor in SWEEP_FACTORS:
            weights = dict(defaults)
            weights[param_name] = defaults[param_name] * factor
            jobs.append(
                {
                    "param": param_name,
                    "factor": factor,
                    "default_value": defaults[param_name],
                    "swept_value": weights[param_name],
                    "weights": weights,
                }
            )

    for job in tqdm(jobs, desc="Sweep configs", unit="cfg"):
        weights = job["weights"]
        successes: List[bool] = []
        total_violations: int = 0
        total_steps: int = 0
        all_delta_phi: List[float] = []

        for seed in seeds:
            ep = _run_episode(
                seed=seed,
                alpha=weights["alpha"],
                beta=weights["beta"],
                gamma=weights["gamma"],
                delta=weights["delta"],
                n_claims=n_claims,
            )
            successes.append(ep["success"])
            total_violations += ep["violations"]
            total_steps += ep["n_steps"]
            all_delta_phi.extend(ep["delta_phi_values"])

        success_rate = float(np.mean(successes))
        violation_rate = total_violations / max(total_steps, 1)
        delta_phi_std = float(np.std(all_delta_phi)) if all_delta_phi else float("nan")

        rows.append(
            {
                "param": job["param"],
                "factor": job["factor"],
                "default_value": job["default_value"],
                "swept_value": round(job["swept_value"], 6),
                "alpha": round(weights["alpha"], 6),
                "beta": round(weights["beta"], 6),
                "gamma": round(weights["gamma"], 6),
                "delta": round(weights["delta"], 6),
                "success_rate": round(success_rate, 4),
                "violation_rate": round(violation_rate, 4),
                "delta_phi_stability": round(delta_phi_std, 6),
                "n_seeds": len(seeds),
                "n_steps_total": total_steps,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and run the ΔΦ weight sensitivity sweep."""
    parser = argparse.ArgumentParser(
        description="Sensitivity sweep for ΔΦ instability weights (α, β, γ, δ)."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--n_claims",
        type=int,
        default=3,
        help="LLM claims per step (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory (default: results/)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running sensitivity sweep: 4 params × {len(SWEEP_FACTORS)} factors "
        f"× {len(args.seeds)} seeds = "
        f"{4 * len(SWEEP_FACTORS) * len(args.seeds)} episodes total"
    )

    results_df = run_sweep(seeds=args.seeds, n_claims=args.n_claims)

    csv_path = output_dir / OUTPUT_CSV
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
