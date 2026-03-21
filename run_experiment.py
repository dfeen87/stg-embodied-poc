#!/usr/bin/env python3
"""run_experiment.py — Main experiment runner for the STG MuJoCo PoC.

Supported conditions
--------------------
  baseline          : ablation="always_execute", hallucination_prob=0.45
  governor          : ablation="none",           hallucination_prob=0.45
  ablation_a        : ablation="no_delta",       hallucination_prob=0.45
  ablation_remove_I : ablation="remove_I",       hallucination_prob=0.45
  ablation_remove_C : ablation="remove_C",       hallucination_prob=0.45
  rag               : ablation="always_execute", hallucination_prob=0.30

Usage
-----
  python run_experiment.py
  python run_experiment.py --conditions governor --seeds 0 1 2 --verbose
  python run_experiment.py --conditions all --output_dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.quadruped_terrain import QuadrupedTerrainEnv
from governor.spiral_time_governor import SpiralTimeGovernor
from llm_mock.mock_llm_agent import MockLLMAgent
from llm_mock.real_llm_agent import RealLLMAgent
from analysis.compute_metrics import summarise_runs, METRICS_CSV_COLS
from config import MAX_STEPS, CONDITION_CONFIGS

ALL_CONDITIONS = list(CONDITION_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Constraint checker
# ---------------------------------------------------------------------------

def constraint_checker(action: np.ndarray) -> bool:
    """Return True if the action satisfies the norm constraint.

    Parameters
    ----------
    action:
        Proposed action vector.

    Returns
    -------
    bool
        ``True`` iff ``‖action‖₂ < 8.0``.
    """
    return bool(np.linalg.norm(action) < 8.0)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    condition: str,
    seed: int,
    n_claims: int = 3,
    verbose: bool = False,
    use_real_llm: bool = False,
) -> Dict[str, Any]:
    """Run a single episode for the given condition and seed.

    Parameters
    ----------
    condition:
        One of the keys in ``CONDITION_CONFIGS``.
    seed:
        Random seed for environment and agent.
    n_claims:
        Number of LLM claims per step.
    verbose:
        If ``True``, print per-step information.
    use_real_llm:
        If ``True``, use ``RealLLMAgent`` (real OpenAI API) instead of
        ``MockLLMAgent``.  Requires the ``OPENAI_API_KEY`` environment
        variable and the ``openai`` package.

    Returns
    -------
    dict
        Full episode result including step logs, metrics, and metadata.
    """
    cfg = CONDITION_CONFIGS[condition]

    env = QuadrupedTerrainEnv(seed=seed)
    if use_real_llm:
        agent: MockLLMAgent | RealLLMAgent = RealLLMAgent(
            seed=seed,
            action_dim=env.action_dim,
        )
    else:
        agent = MockLLMAgent(
            seed=seed,
            hallucination_prob=cfg["hallucination_prob"],
            action_dim=env.action_dim,
        )
    governor = SpiralTimeGovernor(ablation=cfg["ablation"])

    obs = env.reset()
    governor.reset()
    agent.reset(seed=seed)

    step_logs: List[Dict[str, Any]] = []
    n_hallucinated: float = 0.0
    total_claims: int = 0
    violations: int = 0
    last_reward: float = 0.0

    for t in range(MAX_STEPS):
        proposed_action, claims = agent.propose(obs, t, n_claims)
        oracle_state = env.oracle(t)

        mode, gated_action, gov_info = governor.step(
            llm_claims=claims,
            proposed_action=proposed_action,
            oracle_state=oracle_state,
            constraint_checker=constraint_checker,
        )

        # Track hallucination (accumulate as float to avoid rounding errors).
        # NOTE: H_T measures how often the LLM makes false claims about the world.
        # The STG does NOT reduce H_T — hallucinations still occur at the same
        # rate regardless of the ablation condition.  H_T is tracked here only as
        # an independent baseline measure of LLM grounding quality.
        delta_I = gov_info["delta_I"]
        n_hallucinated += delta_I * len(claims)
        total_claims += len(claims)

        # Track violations: unsafe action proposed in EXECUTE/VERIFY mode.
        # NOTE: Reducing violations (V_T) is the *primary safety objective* of
        # the STG.  When ΔΦ(t) ≥ TAU2, the governor switches to SAFE mode and
        # replaces the proposed action with a zero-torque fallback, preventing
        # the unsafe action from reaching the actuators.
        # (gated_action is always safe after gating, so we must check proposed_action)
        if mode != "SAFE" and not constraint_checker(proposed_action):
            violations += 1

        obs, reward, done, info = env.step(gated_action)
        last_reward = reward

        log_entry: Dict[str, Any] = {
            **gov_info,
            "reward": reward,
            "oracle": oracle_state,
        }
        if use_real_llm:
            # Log the extra detail that real-LLM runs produce:
            # raw claims from the model, action norm, and per-claim oracle
            # verification results (derived from delta_I already in gov_info).
            log_entry["real_llm_claims"] = claims
            log_entry["proposed_action_norm"] = float(
                np.linalg.norm(proposed_action)
            )
            log_entry["oracle_verified_claim_count"] = int(
                round((1.0 - delta_I) * len(claims))
            )
        step_logs.append(log_entry)

        if verbose:
            print(
                f"  t={t:03d} | mode={mode:7s} | φ={gov_info['phi']:.3f} "
                f"| ΔΦ={gov_info['delta_phi']:.3f} | r={reward:.3f}"
            )

        if done:
            break

    H_T = n_hallucinated / max(total_claims, 1)
    success = last_reward > 0.5

    return {
        "condition": condition,
        "seed": seed,
        "H_T": H_T,
        "violations": violations,
        "success": success,
        "n_steps": len(step_logs),
        "step_logs": step_logs,
        "governor_log": governor.log,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and run all requested experiments."""
    parser = argparse.ArgumentParser(
        description="STG MuJoCo PoC experiment runner."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["all"],
        choices=ALL_CONDITIONS + ["all"],
        help="Conditions to run (default: all)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="terrain",
        help="Task name (reserved for future multi-task, default: terrain)",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-step output",
    )
    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        dest="use_real_llm",
        help=(
            "Use RealLLMAgent (real OpenAI API) instead of MockLLMAgent. "
            "Requires the OPENAI_API_KEY environment variable and the "
            "'openai' package."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve conditions
    if "all" in args.conditions:
        conditions = ALL_CONDITIONS
    else:
        conditions = args.conditions

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    jobs = [(c, s) for c in conditions for s in args.seeds]

    print(
        f"Running {len(jobs)} episodes "
        f"({len(conditions)} conditions × {len(args.seeds)} seeds) ..."
    )

    for condition, seed in tqdm(jobs, desc="Episodes", unit="ep"):
        if args.verbose:
            print(f"\n[{condition}] seed={seed}")
        result = run_episode(
            condition=condition,
            seed=seed,
            n_claims=args.n_claims,
            verbose=args.verbose,
            use_real_llm=args.use_real_llm,
        )
        all_results.append(result)

    # Save per-condition CSV of step logs
    for condition in conditions:
        cond_results = [r for r in all_results if r["condition"] == condition]
        rows = []
        for r in cond_results:
            for entry in r["step_logs"]:
                rows.append({"condition": condition, "seed": r["seed"], **entry})
        if rows:
            df = pd.DataFrame(rows)
            csv_path = output_dir / f"{condition}_steps.csv"
            df.to_csv(csv_path, index=False)
            if args.verbose:
                print(f"  Saved step log: {csv_path}")

    # Save summary metrics
    summary_df = summarise_runs(all_results)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    # Display condition averages for the three primary metrics:
    #   H_T            — hallucination rate (LLM property; STG does NOT change this)
    #   violation_rate — unsafe-action rate (STG DOES reduce this via SAFE-mode gating)
    #   success        — episode success rate
    cols = ["H_T", "violation_rate", "violations", "success", "mean_phi"]
    cols = [c for c in cols if c in summary_df.columns]
    print(summary_df.groupby("condition")[cols].mean().to_string())

    # Export the three explicit metrics to results/metrics.csv so that
    # hallucination rate and violation rate are clearly separated and easy
    # to compare across conditions.
    metrics_cols = [c for c in METRICS_CSV_COLS if c in summary_df.columns]
    metrics_df = summary_df[metrics_cols].copy()
    metrics_df = metrics_df.rename(columns={"success": "success_rate"})
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Save full results as JSON (step logs excluded to keep file manageable)
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k not in ("step_logs", "governor_log")}
        json_results.append(jr)
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    if args.verbose:
        print(f"Results JSON saved to {json_path}")

    # Export ablations.csv comparing ablation conditions vs full ΔΦ and δ=0.
    # Includes: governor (full ΔΦ), ablation_a (δ=0), ablation_remove_I,
    # ablation_remove_C — whichever of these were actually run.
    ablation_conditions = ["governor", "ablation_a", "ablation_remove_I", "ablation_remove_C"]
    ran_ablation_conditions = [c for c in ablation_conditions if c in conditions]
    if ran_ablation_conditions:
        ablation_rows = [
            r for r in all_results if r["condition"] in ran_ablation_conditions
        ]
        if ablation_rows:
            ablation_summary = summarise_runs(ablation_rows)
            ablation_cols = [
                c for c in
                ["condition", "seed", "H_T", "violation_rate", "violations",
                 "success", "mean_phi", "mean_delta_phi"]
                if c in ablation_summary.columns
            ]
            ablations_path = output_dir / "ablations.csv"
            ablation_summary[ablation_cols].to_csv(ablations_path, index=False)
            print(f"Ablations comparison saved to {ablations_path}")


if __name__ == "__main__":
    main()
