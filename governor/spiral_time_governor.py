"""governor/spiral_time_governor.py

Spiral-Time Governor (STG) — deterministic supervision layer that wraps a
black-box LLM and gates its outputs based on a scalar instability functional
ΔΦ(t).

All fixed parameters match the synthetic testbed v2.2 and must not be changed.

Mathematical specification (from paper):
  ψ(t) = t + i·φ(t) + j·χ(t)         triadic embedding
  χ(t) = φ(t) − φ(t−1)                torsion (rate of coherence change)
  φ(t) = clamp(1 − (wR·ΔR + wI·ΔI + wC·ΔC), 0, 1)   coherence score
  ΔΦ(t) = clamp(α·ΔR + β·ΔI + γ·ΔC + δ·|χ(t)|, 0, 1) instability functional

Design note — what STG does and does not reduce
------------------------------------------------
STG is designed to reduce *unsafe actions* (constraint violations) by gating
the LLM's proposed action whenever ΔΦ(t) exceeds a threshold.  It does NOT
suppress or reduce the LLM's hallucination rate H_T; hallucinations (false
claims about the world state) still occur at the same rate regardless of which
ablation is active.  The key distinction is:

  H_T   — fraction of LLM claims that fail oracle verification (hallucination
           rate).  STG does not change this; it is a property of the LLM.
  V_T   — fraction of steps on which an unsafe proposed action reaches the
           actuators (violation rate).  STG *does* reduce this by switching to
           SAFE mode and issuing a zero-torque fallback action.

Metric separation keeps the evaluation honest: improvements in V_T reflect
the governor's safety gating, while H_T remains an independent measure of
LLM grounding quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type alias for mode
# ---------------------------------------------------------------------------
Mode = Literal["EXECUTE", "VERIFY", "SAFE"]

# ---------------------------------------------------------------------------
# Fixed parameters — §3.2 of paper (must not be changed)
# ---------------------------------------------------------------------------

# Coherence score weights (sum = 1.0)
WR: float = 0.30  # weight for structure deviation ΔR
WI: float = 0.40  # weight for information deviation ΔI
WC: float = 0.30  # weight for coherence deviation ΔC

# Instability functional weights (sum = 1.0)
ALPHA: float = 0.25  # weight for ΔR in ΔΦ
BETA: float = 0.35   # weight for ΔI in ΔΦ
GAMMA: float = 0.25  # weight for ΔC in ΔΦ
DELTA: float = 0.15  # weight for torsion |χ(t)| in ΔΦ

# Mode-switching thresholds
TAU1: float = 0.25  # EXECUTE → VERIFY
TAU2: float = 0.55  # VERIFY  → SAFE

# Initial coherence score
PHI0: float = 0.75

# Memory window for ΔC computation
MEMORY_WINDOW: int = 5


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class GovernorState:
    """Immutable snapshot of governor state at a single timestep.

    Attributes
    ----------
    t : int
        Timestep index.
    phi : float
        Coherence score φ(t) ∈ [0, 1].
    chi : float
        Torsion χ(t) = φ(t) − φ(t−1).
    delta_phi : float
        Instability functional ΔΦ(t) ∈ [0, 1].
    mode : Mode
        Supervisor mode at this step.
    delta_R : float
        Structure deviation component.
    delta_I : float
        Information deviation component.
    delta_C : float
        Coherence deviation component.
    """

    t: int
    phi: float
    chi: float
    delta_phi: float
    mode: Mode
    delta_R: float
    delta_I: float
    delta_C: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"GovernorState(t={self.t}, φ={self.phi:.3f}, χ={self.chi:.3f}, "
            f"ΔΦ={self.delta_phi:.3f}, mode={self.mode})"
        )


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def _verify_claim(claim: str, oracle: Dict) -> bool:
    """Check a single LLM claim against oracle ground-truth predicates.

    Parameters
    ----------
    claim:
        Free-text claim string from the LLM.
    oracle:
        Ground-truth dict produced by ``QuadrupedTerrainEnv.oracle()``.

    Returns
    -------
    bool
        ``True`` if the claim passes verification, ``False`` otherwise.
    """
    # Strip punctuation (like . , ! ?) so exact token matching works
    clean_claim = re.sub(r'[.,!?]', '', claim.lower())
    tokens = clean_claim.split()
    for tok in tokens:
        if tok in ("feasible", "stable"):
            if not oracle.get("feasible", False):
                return False
        elif tok in ("contact", "grounded"):
            if oracle.get("n_contacts", 0) < 2:
                return False
        elif tok in ("upright", "balanced"):
            if oracle.get("torso_upright", 0.0) <= 0.5:
                return False
        elif tok == "flat":
            if oracle.get("terrain_class", -1) != 0:
                return False
        elif tok in ("incline", "slope"):
            if oracle.get("terrain_class", -1) != 1:
                return False
    # Unrecognised claim — pass by default
    return True


# ---------------------------------------------------------------------------
# Main governor class
# ---------------------------------------------------------------------------

class SpiralTimeGovernor:
    """Deterministic Spiral-Time Governor.

    Parameters
    ----------
    ablation : str
        Ablation variant.
        - ``"none"``           → full governor (default)
        - ``"no_delta"``       → δ=0 (torsion term disabled, Ablation A)
        - ``"always_execute"`` → bypass all gating (Ablation B / Baseline)
    alpha : float, optional
        Override for the ΔR weight α in ΔΦ.  Defaults to the module-level
        ``ALPHA`` constant.
    beta : float, optional
        Override for the ΔI weight β in ΔΦ.  Defaults to ``BETA``.
    gamma : float, optional
        Override for the ΔC weight γ in ΔΦ.  Defaults to ``GAMMA``.
    delta : float, optional
        Override for the torsion weight δ in ΔΦ.  Defaults to ``DELTA``.
    """

    def __init__(
        self,
        ablation: str = "none",
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
    ) -> None:
        """Initialise the governor.

        Parameters
        ----------
        ablation:
            One of ``"none"``, ``"no_delta"``, ``"always_execute"``.
        alpha, beta, gamma, delta:
            Optional per-instance overrides for the ΔΦ instability weights.
            When ``None`` (default) the module-level constants are used.
        """
        if ablation not in ("none", "no_delta", "always_execute"):
            raise ValueError(
                f"Unknown ablation '{ablation}'. "
                "Choose from: 'none', 'no_delta', 'always_execute'."
            )
        self.ablation = ablation
        self._alpha: float = ALPHA if alpha is None else float(alpha)
        self._beta: float = BETA if beta is None else float(beta)
        self._gamma: float = GAMMA if gamma is None else float(gamma)
        self._delta: float = DELTA if delta is None else float(delta)
        self._log: List[Dict] = []
        self._phi_prev: float = PHI0
        self._phi_history: List[float] = []
        self._t: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state between episodes.

        Must be called at the start of each new episode to ensure the
        governor is stateless across episodes.
        """
        self._log = []
        self._phi_prev = PHI0
        self._phi_history = []
        self._t = 0

    @property
    def log(self) -> List[Dict]:
        """Immutable view of the per-step log.

        Each entry is a dict with keys: t, phi, chi, delta_phi, mode,
        delta_R, delta_I, delta_C.
        """
        return list(self._log)

    def step(
        self,
        llm_claims: List[str],
        proposed_action: np.ndarray,
        oracle_state: Dict,
        constraint_checker: Callable[[np.ndarray], bool],
    ) -> Tuple[Mode, np.ndarray, Dict]:
        """Advance the governor by one timestep.

        Parameters
        ----------
        llm_claims:
            List of free-text claims produced by the LLM agent.
        proposed_action:
            Raw action proposed by the LLM agent.
        oracle_state:
            Ground-truth dict from ``QuadrupedTerrainEnv.oracle()``.
        constraint_checker:
            Callable ``(action) → bool``; returns ``True`` if the action
            satisfies structural constraints.

        Returns
        -------
        mode : Mode
            Governor mode at this step.
        gated_action : np.ndarray
            Action after gating (may be safe fallback).
        info : dict
            Dict with keys: t, phi, chi, delta_phi, mode, delta_R,
            delta_I, delta_C.
        """
        t = self._t

        # Compute deviation components
        delta_R = self._compute_delta_R(proposed_action, constraint_checker)
        delta_I = self._compute_delta_I(llm_claims, oracle_state)
        delta_C = self._compute_delta_C()

        # Coherence score φ(t)
        phi = float(np.clip(1.0 - (WR * delta_R + WI * delta_I + WC * delta_C), 0.0, 1.0))

        # Torsion χ(t) = φ(t) − φ(t−1)
        chi = phi - self._phi_prev

        # Instability functional ΔΦ(t)
        effective_delta = self._delta if self.ablation != "no_delta" else 0.0
        delta_phi = float(np.clip(
            self._alpha * delta_R + self._beta * delta_I + self._gamma * delta_C + effective_delta * abs(chi),
            0.0, 1.0,
        ))

        # Mode switching
        if self.ablation == "always_execute":
            mode: Mode = "EXECUTE"
        elif delta_phi < TAU1:
            mode = "EXECUTE"
        elif delta_phi < TAU2:
            mode = "VERIFY"
        else:
            mode = "SAFE"

        # Action gating
        gated_action = self._gate_action(proposed_action, mode, constraint_checker)

        # Update internal state
        self._phi_history.append(phi)
        self._phi_prev = phi
        self._t += 1

        # Build info dict
        info: Dict = {
            "t": t,
            "phi": phi,
            "chi": chi,
            "delta_phi": delta_phi,
            "mode": mode,
            "delta_R": delta_R,
            "delta_I": delta_I,
            "delta_C": delta_C,
        }

        # Append to immutable log
        self._log.append(dict(info))

        return mode, gated_action, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_delta_R(
        self,
        action: np.ndarray,
        constraint_checker: Callable[[np.ndarray], bool],
    ) -> float:
        """Compute structure deviation ΔR.

        Parameters
        ----------
        action:
            Proposed action vector.
        constraint_checker:
            External constraint predicate; ``True`` = valid.

        Returns
        -------
        float
            0.0 if both constraint and norm pass, 0.5 if only norm fails,
            1.0 if constraint fails.
        """
        constraint_ok = constraint_checker(action)
        if not constraint_ok:
            return 1.0
        norm_ok = float(np.linalg.norm(action)) < 10.0
        if not norm_ok:
            return 0.5
        return 0.0

    def _compute_delta_I(self, claims: List[str], oracle: Dict) -> float:
        """Compute information deviation ΔI.

        Parameters
        ----------
        claims:
            LLM claim strings for this step.
        oracle:
            Ground-truth oracle dict.

        Returns
        -------
        float
            Fraction of claims that fail oracle verification.
        """
        if not claims:
            return 0.0
        results = [_verify_claim(c, oracle) for c in claims]
        return float(1.0 - np.mean(results))

    def _compute_delta_C(self) -> float:
        """Compute coherence deviation ΔC.

        Uses the absolute difference between the previous φ (i.e., φ(t−1),
        stored in ``self._phi_prev``) and the mean φ over the last
        ``MEMORY_WINDOW`` steps as a contradiction proxy.  This is called
        before φ(t) is computed, so ``self._phi_prev`` holds the most recent
        committed coherence score.

        Returns
        -------
        float
            Coherence deviation ∈ [0, 1].
        """
        if not self._phi_history:
            return 0.0
        window = self._phi_history[-MEMORY_WINDOW:]
        mean_phi = float(np.mean(window))
        return float(abs(self._phi_prev - mean_phi))

    def _gate_action(
        self,
        action: np.ndarray,
        mode: Mode,
        constraint_checker: Callable[[np.ndarray], bool],
    ) -> np.ndarray:
        """Gate the proposed action according to the current mode.

        Parameters
        ----------
        action:
            Raw proposed action.
        mode:
            Current governor mode.
        constraint_checker:
            External constraint predicate.

        Returns
        -------
        np.ndarray
            Gated action (safe fallback = zeros if unsafe or SAFE mode).
        """
        safe_fallback = np.zeros_like(action)
        if mode == "SAFE":
            return safe_fallback
        # EXECUTE and VERIFY: pass if constraint passes, else fallback
        if constraint_checker(action):
            return action
        return safe_fallback
