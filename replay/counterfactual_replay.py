"""replay/counterfactual_replay.py

Deterministic "Counterfactual Replay" system that compares raw vs. STG-filtered
actions under identical seeds and identical initial environment state.

Algorithm
---------
1. **Record phase** — run one episode with the mock LLM agent and capture, at
   every timestep, the *proposed* action, the LLM claims, and the oracle state.
   The STG math is *not* executed during recording; this phase only produces
   the raw action sequence.

2. **Track A** — reset the environment with the identical seed, then replay the
   recorded raw proposed actions **without any filtering**.  Metrics (reward,
   constraint violations) are collected to characterise unfiltered behaviour.

3. **Track B** — reset the environment with the identical seed, then replay the
   same recorded (proposed_action, claims, oracle_state) tuples through a fresh
   SpiralTimeGovernor.  The governor gates unsafe actions exactly as in a live
   episode.  Metrics are collected to quantify the safety improvement from STG.

Guarantee of fairness
---------------------
* Identical seed → identical dm_control physics initialisation.
* Identical proposed actions / claims / oracle states → only the *filtering*
  step differs between the two tracks.
* The STG math (ΔΦ computation, mode switching) is left entirely unchanged.

Output
------
Per-replay JSON files are written to ``results/replay/`` (or a custom
``output_dir``).  Each file contains the full step-level logs for both tracks
plus a summary comparison table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import MAX_STEPS, CONDITION_CONFIGS
from envs.quadruped_terrain import QuadrupedTerrainEnv
from governor.spiral_time_governor import SpiralTimeGovernor
from llm_mock.mock_llm_agent import MockLLMAgent

# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------

DEFAULT_REPLAY_DIR: Path = Path("results/replay")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StepRecord = Dict[str, Any]


# ---------------------------------------------------------------------------
# Constraint checker (same as run_experiment.py)
# ---------------------------------------------------------------------------

def _constraint_checker(action: np.ndarray) -> bool:
    """Return ``True`` iff ``‖action‖₂ < 8.0``."""
    return bool(np.linalg.norm(action) < 8.0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RecordedStep:
    """One timestep captured during the record phase.

    Attributes
    ----------
    t : int
        Zero-based timestep index.
    proposed_action : np.ndarray
        Raw action proposed by the agent before any filtering.
    claims : List[str]
        LLM claims generated at this timestep.
    oracle_state : Dict[str, Any]
        Ground-truth oracle state returned by the environment.
    """

    t: int
    proposed_action: np.ndarray
    claims: List[str]
    oracle_state: Dict[str, Any]


@dataclass
class ReplayResult:
    """Full result of a counterfactual replay run.

    Attributes
    ----------
    seed : int
        Random seed used for the replay.
    condition : str
        Governor condition used for Track B filtering.
    n_claims : int
        Number of LLM claims per step.
    record_steps : int
        Number of steps captured in the record phase.
    track_a : List[StepRecord]
        Per-step logs for Track A (raw, unfiltered).
    track_b : List[StepRecord]
        Per-step logs for Track B (STG-filtered).
    summary : Dict[str, Any]
        Scalar comparison metrics.
    """

    seed: int
    condition: str
    n_claims: int
    record_steps: int
    track_a: List[StepRecord]
    track_b: List[StepRecord]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Record phase
# ---------------------------------------------------------------------------

def _record_phase(
    seed: int,
    hallucination_prob: float,
    n_claims: int,
) -> Tuple[List[RecordedStep], np.ndarray]:
    """Run one episode and capture the raw action sequence.

    The environment is reset with *seed* and the mock LLM agent proposes
    actions deterministically.  No governor filtering is applied; the
    environment is stepped with the raw proposed actions so that the oracle
    state reflects the unfiltered trajectory.

    Parameters
    ----------
    seed :
        Random seed for both the environment and the agent.
    hallucination_prob :
        Hallucination probability forwarded to :class:`MockLLMAgent`.
    n_claims :
        Number of claims per step.

    Returns
    -------
    recorded : List[RecordedStep]
        Ordered sequence of per-step records.
    initial_obs : np.ndarray
        Observation returned by ``env.reset()`` (i.e., the starting
        observation for both replay tracks).
    """
    env = QuadrupedTerrainEnv(seed=seed)
    agent = MockLLMAgent(
        seed=seed,
        hallucination_prob=hallucination_prob,
        action_dim=env.action_dim,
    )

    initial_obs = env.reset()
    agent.reset(seed=seed)

    recorded: List[RecordedStep] = []

    obs = initial_obs
    for t in range(MAX_STEPS):
        proposed_action, claims = agent.propose(obs, t, n_claims)
        oracle_state = env.oracle(t)

        recorded.append(
            RecordedStep(
                t=t,
                proposed_action=proposed_action.copy(),
                claims=list(claims),
                oracle_state=dict(oracle_state),
            )
        )

        # Step with raw action so oracle state matches the unfiltered trajectory
        obs, _reward, done, _info = env.step(proposed_action)
        if done:
            break

    return recorded, initial_obs.copy()


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _step_log(
    t: int,
    proposed_action: np.ndarray,
    executed_action: np.ndarray,
    reward: float,
    done: bool,
    gov_info: Optional[Dict[str, Any]] = None,
) -> StepRecord:
    """Build a per-step log dict for one track."""
    is_violation = not _constraint_checker(proposed_action)
    log: StepRecord = {
        "t": t,
        "reward": reward,
        "done": done,
        "proposed_action_norm": float(np.linalg.norm(proposed_action)),
        "executed_action_norm": float(np.linalg.norm(executed_action)),
        "constraint_violation": is_violation,
        "action_delta_norm": float(
            np.linalg.norm(executed_action - proposed_action)
        ),
    }
    if gov_info is not None:
        log.update(gov_info)
    return log


def _replay_track_a(
    seed: int,
    recorded: List[RecordedStep],
) -> List[StepRecord]:
    """Replay Track A: raw actions, no filtering.

    Parameters
    ----------
    seed :
        Seed used to reset the environment to the identical initial state.
    recorded :
        Step records captured during the record phase.

    Returns
    -------
    List[StepRecord]
        Per-step logs.
    """
    env = QuadrupedTerrainEnv(seed=seed)
    env.reset()  # identical initial state

    logs: List[StepRecord] = []
    for rec in recorded:
        action = rec.proposed_action
        obs, reward, done, _info = env.step(action)
        log = _step_log(
            t=rec.t,
            proposed_action=action,
            executed_action=action,
            reward=reward,
            done=done,
            gov_info=None,
        )
        logs.append(log)
        if done:
            break

    return logs


def _replay_track_b(
    seed: int,
    recorded: List[RecordedStep],
    ablation: str,
) -> List[StepRecord]:
    """Replay Track B: STG-filtered actions.

    The governor receives the **recorded** (proposed_action, claims,
    oracle_state) tuples — exactly the same inputs that Track A used —
    and may replace unsafe proposed actions with a zero-torque fallback.
    The environment is stepped with the gated action.

    Parameters
    ----------
    seed :
        Seed used to reset the environment to the identical initial state.
    recorded :
        Step records captured during the record phase.
    ablation :
        Governor ablation mode (forwarded to :class:`SpiralTimeGovernor`).

    Returns
    -------
    List[StepRecord]
        Per-step logs including governor diagnostics.
    """
    env = QuadrupedTerrainEnv(seed=seed)
    env.reset()  # identical initial state

    governor = SpiralTimeGovernor(ablation=ablation)
    governor.reset()

    logs: List[StepRecord] = []
    for rec in recorded:
        mode, gated_action, gov_info = governor.step(
            llm_claims=rec.claims,
            proposed_action=rec.proposed_action,
            oracle_state=rec.oracle_state,
            constraint_checker=_constraint_checker,
        )

        obs, reward, done, _info = env.step(gated_action)
        log = _step_log(
            t=rec.t,
            proposed_action=rec.proposed_action,
            executed_action=gated_action,
            reward=reward,
            done=done,
            gov_info=gov_info,
        )
        log["mode"] = mode
        logs.append(log)
        if done:
            break

    return logs


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_summary(
    track_a: List[StepRecord],
    track_b: List[StepRecord],
) -> Dict[str, Any]:
    """Compute scalar comparison metrics between the two tracks.

    Parameters
    ----------
    track_a :
        Track A step logs (raw).
    track_b :
        Track B step logs (filtered).

    Returns
    -------
    Dict[str, Any]
        Keys: ``{track}_total_reward``, ``{track}_violations``,
        ``{track}_violation_rate``, ``{track}_n_steps``,
        and ``delta_*`` difference fields.
    """
    def _agg(logs: List[StepRecord]) -> Dict[str, Any]:
        n = len(logs)
        total_reward = sum(s["reward"] for s in logs)
        violations = sum(1 for s in logs if s["constraint_violation"])
        return {
            "n_steps": n,
            "total_reward": float(total_reward),
            "violations": int(violations),
            "violation_rate": float(violations / n) if n > 0 else 0.0,
        }

    a = _agg(track_a)
    b = _agg(track_b)

    # Count steps where governor actually intervened (action was replaced)
    interventions = sum(
        1 for s in track_b if s.get("action_delta_norm", 0.0) > 1e-9
    )

    return {
        "track_a_n_steps": a["n_steps"],
        "track_a_total_reward": a["total_reward"],
        "track_a_violations": a["violations"],
        "track_a_violation_rate": a["violation_rate"],
        "track_b_n_steps": b["n_steps"],
        "track_b_total_reward": b["total_reward"],
        "track_b_violations": b["violations"],
        "track_b_violation_rate": b["violation_rate"],
        "delta_violations": a["violations"] - b["violations"],
        "delta_violation_rate": a["violation_rate"] - b["violation_rate"],
        "delta_total_reward": b["total_reward"] - a["total_reward"],
        "stg_interventions": interventions,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_counterfactual_replay(
    seed: int,
    condition: str = "governor",
    n_claims: int = 3,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> ReplayResult:
    """Run a full counterfactual replay for one seed.

    Executes the three phases (record → Track A → Track B) and saves
    the result as a JSON file under *output_dir*.

    Parameters
    ----------
    seed :
        Random seed.  Determines the initial environment state and the
        agent's action sequence.
    condition :
        One of the keys in ``config.CONDITION_CONFIGS``.  Selects the
        governor ablation and hallucination probability.
    n_claims :
        Number of LLM claims generated per step.
    output_dir :
        Directory where the JSON result file will be written.
        Defaults to :data:`DEFAULT_REPLAY_DIR`.
    verbose :
        If ``True``, print progress information.

    Returns
    -------
    ReplayResult
        Full replay result (also persisted to disk).
    """
    if output_dir is None:
        output_dir = DEFAULT_REPLAY_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CONDITION_CONFIGS[condition]
    hallucination_prob: float = cfg["hallucination_prob"]
    ablation: str = cfg["ablation"]

    if verbose:
        print(f"[replay] seed={seed} condition={condition}")
        print(f"  Record phase ...")

    # ------------------------------------------------------------------
    # 1. Record phase
    # ------------------------------------------------------------------
    recorded, initial_obs = _record_phase(
        seed=seed,
        hallucination_prob=hallucination_prob,
        n_claims=n_claims,
    )
    if verbose:
        print(f"  Captured {len(recorded)} steps.")

    # ------------------------------------------------------------------
    # 2. Track A — raw actions
    # ------------------------------------------------------------------
    if verbose:
        print("  Track A (raw) ...")
    track_a = _replay_track_a(seed=seed, recorded=recorded)

    # ------------------------------------------------------------------
    # 3. Track B — STG-filtered actions
    # ------------------------------------------------------------------
    if verbose:
        print("  Track B (STG-filtered) ...")
    track_b = _replay_track_b(seed=seed, recorded=recorded, ablation=ablation)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    summary = _compute_summary(track_a, track_b)

    result = ReplayResult(
        seed=seed,
        condition=condition,
        n_claims=n_claims,
        record_steps=len(recorded),
        track_a=track_a,
        track_b=track_b,
        summary=summary,
    )

    # ------------------------------------------------------------------
    # 5. Persist to disk
    # ------------------------------------------------------------------
    _save_replay(result, output_dir)

    if verbose:
        _print_summary(summary)

    return result


def run_counterfactual_replays(
    seeds: List[int],
    condition: str = "governor",
    n_claims: int = 3,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> List[ReplayResult]:
    """Run counterfactual replays for multiple seeds.

    Parameters
    ----------
    seeds :
        List of random seeds to replay.
    condition :
        Governor condition (see :func:`run_counterfactual_replay`).
    n_claims :
        Claims per step.
    output_dir :
        Output directory.
    verbose :
        Verbose output.

    Returns
    -------
    List[ReplayResult]
        One result per seed.
    """
    results: List[ReplayResult] = []
    for seed in seeds:
        result = run_counterfactual_replay(
            seed=seed,
            condition=condition,
            n_claims=n_claims,
            output_dir=output_dir,
            verbose=verbose,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


def _save_replay(result: ReplayResult, output_dir: Path) -> Path:
    """Serialise and write *result* to a JSON file.

    Parameters
    ----------
    result :
        Replay result to save.
    output_dir :
        Destination directory.

    Returns
    -------
    Path
        Path of the written file.
    """
    payload = {
        "seed": result.seed,
        "condition": result.condition,
        "n_claims": result.n_claims,
        "record_steps": result.record_steps,
        "summary": _serialise(result.summary),
        "track_a": _serialise(result.track_a),
        "track_b": _serialise(result.track_b),
    }
    fname = output_dir / f"replay_seed{result.seed}_{result.condition}.json"
    with open(fname, "w") as fh:
        json.dump(payload, fh, indent=2)
    return fname


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print(
        f"\n  {'Metric':<30} {'Track A (raw)':>14} {'Track B (STG)':>14}"
    )
    print("  " + "-" * 60)
    pairs = [
        ("n_steps", "track_a_n_steps", "track_b_n_steps"),
        ("total_reward", "track_a_total_reward", "track_b_total_reward"),
        ("violations", "track_a_violations", "track_b_violations"),
        ("violation_rate", "track_a_violation_rate", "track_b_violation_rate"),
    ]
    for label, key_a, key_b in pairs:
        va = summary[key_a]
        vb = summary[key_b]
        if isinstance(va, float):
            print(f"  {label:<30} {va:>14.4f} {vb:>14.4f}")
        else:
            print(f"  {label:<30} {va:>14} {vb:>14}")
    print(
        f"  {'stg_interventions':<30} {'':>14} "
        f"{summary['stg_interventions']:>14}"
    )
    print(
        f"  {'delta_violations (A−B)':<30} "
        f"{summary['delta_violations']:>14}"
    )
    print(
        f"  {'delta_reward (B−A)':<30} "
        f"{summary['delta_total_reward']:>14.4f}"
    )
