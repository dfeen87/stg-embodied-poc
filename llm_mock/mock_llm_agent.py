"""llm_mock/mock_llm_agent.py

Deterministic mock LLM agent for the STG PoC.

Produces deterministic claims + actions from obs + seed.
Simulates LLM planning behaviour including controlled hallucination injection.
The SAME agent is used for all conditions — the governor handles suppression
externally.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Claim pools
# ---------------------------------------------------------------------------

_CORRECT_CLAIMS: List[str] = [
    "The robot is feasible and stable on current terrain.",
    "All four feet are in contact with the ground.",
    "The torso is upright and balanced.",
    "Terrain is flat ahead, safe to proceed.",
    "Current gait is stable, continuing forward motion.",
    "Contact set is complete, locomotion is safe.",
]

_HALLUCINATED_CLAIMS: List[str] = [
    "The robot is feasible and stable on current terrain.",  # may contradict oracle
    "All four feet are firmly grounded.",                    # may be false mid-air
    "Incline detected ahead, adjusting gait.",              # may be wrong terrain class
    "The torso is upright and balanced.",                   # may be false if falling
    "Gap crossing complete, terrain is flat.",              # likely false
    "Contact with all limbs confirmed.",                    # may be false
]


class MockLLMAgent:
    """Deterministic mock LLM agent with controlled hallucination injection.

    Parameters
    ----------
    seed : int
        Random seed.  Same seed + same obs + same t → identical output.
    hallucination_prob : float
        Probability that each claim is drawn from the hallucinated pool.
    action_dim : int
        Dimensionality of the action vector.

    Class Constants
    ---------------
    BASELINE_HALLUCINATION_PROB : float
        Default hallucination probability for baseline / governor conditions.
    RAG_HALLUCINATION_PROB : float
        Reduced hallucination probability simulating RAG noise reduction.
    """

    BASELINE_HALLUCINATION_PROB: float = 0.45
    RAG_HALLUCINATION_PROB: float = 0.30
    BAD_ACTION_PROB: float = 0.10       # probability of injecting a high-norm action
    BAD_ACTION_NORM: float = 8.5        # target norm for injected bad actions (> threshold 8.0)

    def __init__(
        self,
        seed: int,
        hallucination_prob: float = 0.45,
        action_dim: int = 12,
    ) -> None:
        """Initialise the mock LLM agent.

        Parameters
        ----------
        seed:
            Random seed for the internal RNG.
        hallucination_prob:
            Per-claim probability of drawing from the hallucinated pool.
        action_dim:
            Dimensionality of the proposed action.
        """
        self._seed = seed
        self._hallucination_prob = hallucination_prob
        self._action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        obs: np.ndarray,
        t: int,
        n_claims: int = 3,
    ) -> Tuple[np.ndarray, List[str]]:
        """Propose an action and a list of claims for the current timestep.

        Parameters
        ----------
        obs:
            Current environment observation (used for conditioning; not
            currently modelled but kept for interface compatibility).
        t:
            Current timestep index; drives the gait phase.
        n_claims:
            Number of claims to generate.

        Returns
        -------
        action : np.ndarray
            Proposed action vector.
        claims : List[str]
            List of ``n_claims`` claim strings.
        """
        action = self._generate_action(t)
        claims = self._generate_claims(n_claims)
        return action, claims

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's RNG state.

        Parameters
        ----------
        seed:
            New seed.  If ``None``, reinitialises with the original seed.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_action(self, t: int) -> np.ndarray:
        """Generate a sinusoidal gait action with noise.

        Occasionally injects a bad action (norm > 8.0) with probability 0.1
        to test constraint gating in the governor.

        Parameters
        ----------
        t:
            Current timestep index.

        Returns
        -------
        np.ndarray
            Proposed action of shape ``(action_dim,)``.
        """
        phase = t * 0.1
        offsets = np.linspace(0.0, 2.0 * math.pi, self._action_dim)
        base = 0.3 * np.sin(phase + offsets)
        noise = self._rng.normal(0.0, 0.05, self._action_dim)
        action = base + noise

        # Occasionally inject a high-norm action to exercise gating
        if self._rng.random() < self.BAD_ACTION_PROB:
            # Scale to norm slightly above 8.0
            current_norm = float(np.linalg.norm(action))
            if current_norm > 0:
                action = action * (self.BAD_ACTION_NORM / current_norm)

        return action.astype(np.float64)

    def _generate_claims(self, n_claims: int) -> List[str]:
        """Generate a list of claims with random hallucination injection.

        Parameters
        ----------
        n_claims:
            Number of claims to generate.

        Returns
        -------
        List[str]
            Claim strings (duplicates allowed, matching realistic LLM output).
        """
        claims: List[str] = []
        for _ in range(n_claims):
            if self._rng.random() < self._hallucination_prob:
                idx = int(self._rng.integers(0, len(_HALLUCINATED_CLAIMS)))
                claims.append(_HALLUCINATED_CLAIMS[idx])
            else:
                idx = int(self._rng.integers(0, len(_CORRECT_CLAIMS)))
                claims.append(_CORRECT_CLAIMS[idx])
        return claims
