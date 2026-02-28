"""config.py — Single source of truth for all experimental parameters.

Governor weights and thresholds are imported from
``governor.spiral_time_governor`` (where they are defined once) and
re-exported here for convenience.  All other experimental hyper-parameters
are defined in this file.

Usage
-----
    from config import MAX_STEPS, CONDITION_CONFIGS, DEFAULT_SEEDS
"""

from __future__ import annotations

from typing import Any

from governor.spiral_time_governor import (
    WR,
    WI,
    WC,
    ALPHA,
    BETA,
    GAMMA,
    DELTA,
    TAU1,
    TAU2,
    PHI0,
)

# ---------------------------------------------------------------------------
# Re-export governor parameters (single source of truth is spiral_time_governor)
# ---------------------------------------------------------------------------

__all__ = [
    # Governor params
    "WR", "WI", "WC",
    "ALPHA", "BETA", "GAMMA", "DELTA",
    "TAU1", "TAU2",
    # Experiment params
    "MAX_STEPS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "SIGNIFICANCE_LEVEL",
    "DEFAULT_SEEDS",
    "INITIAL_PHI",
    "CONDITION_CONFIGS",
]

# ---------------------------------------------------------------------------
# Experiment hyper-parameters
# ---------------------------------------------------------------------------

MAX_STEPS: int = 120
"""Maximum episode steps (same value used in envs and run_experiment)."""

BOOTSTRAP_RESAMPLES: int = 2000
"""Number of bootstrap resamples for confidence interval estimation."""

BOOTSTRAP_SEED: int = 77
"""Random seed for bootstrap resampling (ensures reproducibility)."""

SIGNIFICANCE_LEVEL: float = 0.05
"""α-level for hypothesis tests (two-sided, 95 % confidence intervals)."""

DEFAULT_SEEDS: list[int] = list(range(10))
"""Default random seeds (0–9) used in the main experiment."""

INITIAL_PHI: float = PHI0
"""Initial coherence score φ₀ (re-exported from governor module)."""

# ---------------------------------------------------------------------------
# Condition configurations
# ---------------------------------------------------------------------------

CONDITION_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "ablation": "always_execute",
        "hallucination_prob": 0.45,
    },
    "governor": {
        "ablation": "none",
        "hallucination_prob": 0.45,
    },
    "ablation_a": {
        "ablation": "no_delta",
        "hallucination_prob": 0.45,
    },
    "rag": {
        "ablation": "always_execute",
        "hallucination_prob": 0.30,
    },
}
"""
Experimental conditions.

Each entry maps a condition name to the kwargs passed to
``SpiralTimeGovernor`` (``ablation``) and ``MockLLMAgent``
(``hallucination_prob``).
"""
