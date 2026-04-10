#!/usr/bin/env python3
"""run_replay.py — CLI for the deterministic counterfactual replay system.

Runs paired rollouts that compare raw (unfiltered) vs. STG-filtered actions
under identical seeds and identical initial environment state.

Usage
-----
    python run_replay.py
    python run_replay.py --seeds 0 1 2 --condition governor --verbose
    python run_replay.py --seeds 0 --output_dir results/replay/

Output
------
One JSON file per seed is written to ``results/replay/`` (or ``--output_dir``).
Each file contains per-step logs for both tracks and a summary comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from config import CONDITION_CONFIGS
from replay.counterfactual_replay import (
    DEFAULT_REPLAY_DIR,
    run_counterfactual_replays,
)

ALL_CONDITIONS = list(CONDITION_CONFIGS.keys())


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and run counterfactual replays."""
    parser = argparse.ArgumentParser(
        description="STG counterfactual replay: compare raw vs. filtered actions."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="governor",
        choices=ALL_CONDITIONS,
        help="Governor condition for Track B filtering (default: governor)",
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
        default=str(DEFAULT_REPLAY_DIR),
        help=f"Output directory (default: {DEFAULT_REPLAY_DIR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-replay output",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {len(args.seeds)} replay(s) "
        f"[condition={args.condition}, n_claims={args.n_claims}] ..."
    )

    results = run_counterfactual_replays(
        seeds=args.seeds,
        condition=args.condition,
        n_claims=args.n_claims,
        output_dir=output_dir,
        verbose=args.verbose,
    )

    # Print aggregate summary across all seeds
    print(f"\n{'Seed':>6}  {'TrackA viol':>12}  {'TrackB viol':>12}  "
          f"{'Δ viol':>8}  {'STG intvn':>10}  {'Δ reward':>10}")
    print("-" * 68)
    for r in results:
        s = r.summary
        print(
            f"{r.seed:>6}  {s['track_a_violations']:>12}  "
            f"{s['track_b_violations']:>12}  "
            f"{s['delta_violations']:>8}  "
            f"{s['stg_interventions']:>10}  "
            f"{s['delta_total_reward']:>10.4f}"
        )

    # Aggregate totals
    total_a = sum(r.summary["track_a_violations"] for r in results)
    total_b = sum(r.summary["track_b_violations"] for r in results)
    total_delta_r = sum(r.summary["delta_total_reward"] for r in results)
    total_intvn = sum(r.summary["stg_interventions"] for r in results)
    print("-" * 68)
    print(
        f"{'TOTAL':>6}  {total_a:>12}  {total_b:>12}  "
        f"{total_a - total_b:>8}  {total_intvn:>10}  "
        f"{total_delta_r:>10.4f}"
    )

    # Write aggregate summary JSON
    aggregate = {
        "seeds": args.seeds,
        "condition": args.condition,
        "n_claims": args.n_claims,
        "per_seed": [
            {"seed": r.seed, **r.summary} for r in results
        ],
        "aggregate": {
            "track_a_total_violations": total_a,
            "track_b_total_violations": total_b,
            "delta_violations": total_a - total_b,
            "stg_interventions": total_intvn,
            "delta_total_reward": total_delta_r,
        },
    }
    agg_path = output_dir / f"replay_aggregate_{args.condition}.json"
    with open(agg_path, "w") as fh:
        json.dump(aggregate, fh, indent=2)
    print(f"\nAggregate summary saved to {agg_path}")


if __name__ == "__main__":
    main()
