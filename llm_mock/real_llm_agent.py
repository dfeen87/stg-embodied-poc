"""llm_mock/real_llm_agent.py

Real LLM agent for the STG PoC.

Calls a real LLM API (OpenAI) to generate claims about the current robot
state.  Action generation uses the same deterministic sinusoidal gait as
``MockLLMAgent`` to keep the policy constrained — no additional autonomy is
added beyond what the mock agent provides.

Environment variable
--------------------
OPENAI_API_KEY:
    Must be set before using ``RealLLMAgent`` or ``query_real_llm``.

Usage
-----
    from llm_mock.real_llm_agent import RealLLMAgent

    agent = RealLLMAgent(seed=0, action_dim=12)
    action, claims = agent.propose(obs, t=0, n_claims=3)
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_CLAIM_PROMPT_TEMPLATE = (
    "You are a robot locomotion monitor. "
    "Given a brief robot-state summary, generate exactly {n_claims} short "
    "factual claims about the current state of the robot. "
    "Output only the claims, one per line, with no numbering or bullets.\n\n"
    "Robot state: {obs_summary}"
)


# ---------------------------------------------------------------------------
# Public API function
# ---------------------------------------------------------------------------

def query_real_llm(claim_text: str, model: str = "gpt-3.5-turbo") -> str:
    """Query the OpenAI chat-completion endpoint and return the raw response.

    Parameters
    ----------
    claim_text:
        The full prompt to send to the model.
    model:
        OpenAI model name (default ``"gpt-3.5-turbo"``).

    Returns
    -------
    str
        Raw text response from the model.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    RuntimeError
        If the ``OPENAI_API_KEY`` environment variable is not set.
    """
    try:
        import openai  # noqa: PLC0415 — lazy import to keep the mock path free of the dep
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for RealLLMAgent.  "
            "Install it with: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set.  "
            "Export it before running with --use-real-llm."
        )

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": claim_text}],
        max_tokens=256,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Helper: parse raw LLM response into individual claim strings
# ---------------------------------------------------------------------------

def _parse_claims(raw_response: str, n_claims: int) -> List[str]:
    """Parse newline-separated claims from a raw LLM response.

    Strips common leading markers (numbers, dashes, bullets) and pads or
    truncates the result to exactly ``n_claims`` entries.

    Parameters
    ----------
    raw_response:
        Raw text returned by the LLM.
    n_claims:
        Desired number of claims.

    Returns
    -------
    List[str]
        Exactly ``n_claims`` claim strings.
    """
    lines = [ln.strip() for ln in raw_response.strip().splitlines() if ln.strip()]

    cleaned: List[str] = []
    for line in lines:
        # Remove leading "1. ", "- ", "* ", "• " style markers
        for marker in ("•", "*", "-"):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break
        # Remove leading numeric markers like "1." or "1)"
        if len(line) >= 2 and line[0].isdigit() and line[1] in (".", ")"):
            line = line[2:].strip()
        if line:
            cleaned.append(line)

    if not cleaned:
        cleaned = ["State uncertain."]

    # Truncate or repeat-pad to exactly n_claims
    if len(cleaned) >= n_claims:
        return cleaned[:n_claims]
    while len(cleaned) < n_claims:
        cleaned.append(cleaned[-1])
    return cleaned


# ---------------------------------------------------------------------------
# RealLLMAgent
# ---------------------------------------------------------------------------

class RealLLMAgent:
    """LLM agent backed by a real API for claim generation.

    Action generation uses the same deterministic sinusoidal gait as
    ``MockLLMAgent`` so the policy remains constrained.  Only claim generation
    delegates to the real LLM.

    Parameters
    ----------
    seed:
        Random seed for the internal action-generation RNG.
    action_dim:
        Dimensionality of the action vector (default 12).
    model:
        OpenAI model name (default ``"gpt-3.5-turbo"``).

    Class Constants
    ---------------
    BAD_ACTION_PROB : float
        Probability of injecting a high-norm action to exercise constraint
        gating (mirrors ``MockLLMAgent``).
    BAD_ACTION_NORM : float
        Target norm for injected bad actions (> constraint threshold 8.0).
    """

    BAD_ACTION_PROB: float = 0.10
    BAD_ACTION_NORM: float = 8.5

    def __init__(
        self,
        seed: int,
        action_dim: int = 12,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        self._seed = seed
        self._action_dim = action_dim
        self._model = model
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API  (mirrors MockLLMAgent)
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
            Current environment observation (used to build the LLM prompt).
        t:
            Current timestep index; drives the gait phase.
        n_claims:
            Number of claims to request from the LLM.

        Returns
        -------
        action : np.ndarray
            Proposed action vector (deterministic sinusoidal gait).
        claims : List[str]
            ``n_claims`` claim strings returned by the real LLM.
        """
        action = self._generate_action(t)
        claims, raw_response = self._generate_claims(obs, t, n_claims)

        logger.info(
            "t=%d | raw_llm_response=%r | parsed_claims=%s | action_norm=%.4f",
            t,
            raw_response,
            claims,
            float(np.linalg.norm(action)),
        )

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
        """Generate a sinusoidal gait action (identical to MockLLMAgent)."""
        phase = t * 0.1
        offsets = np.linspace(0.0, 2.0 * math.pi, self._action_dim)
        base = 0.3 * np.sin(phase + offsets)
        noise = self._rng.normal(0.0, 0.05, self._action_dim)
        action = base + noise

        # Occasionally inject a high-norm action to exercise gating
        if self._rng.random() < self.BAD_ACTION_PROB:
            current_norm = float(np.linalg.norm(action))
            if current_norm > 0:
                action = action * (self.BAD_ACTION_NORM / current_norm)

        return action.astype(np.float64)

    def _generate_claims(
        self,
        obs: np.ndarray,
        t: int,
        n_claims: int,
    ) -> Tuple[List[str], str]:
        """Call the real LLM and parse claims from the response.

        Returns
        -------
        claims : List[str]
            Parsed claim strings.
        raw_response : str
            Unmodified text returned by the LLM (for logging).
        """
        obs_summary = (
            f"timestep={t}, "
            f"obs_mean={float(np.mean(obs)):.3f}, "
            f"obs_norm={float(np.linalg.norm(obs)):.3f}"
        )
        prompt = _CLAIM_PROMPT_TEMPLATE.format(
            n_claims=n_claims,
            obs_summary=obs_summary,
        )
        raw_response = query_real_llm(prompt, model=self._model)
        logger.debug("t=%d | raw_llm_response=%r", t, raw_response)

        claims = _parse_claims(raw_response, n_claims)
        return claims, raw_response
