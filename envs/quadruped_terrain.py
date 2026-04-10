"""envs/quadruped_terrain.py

MuJoCo environment wrapper for the STG PoC, built on dm_control's built-in
quadruped domain with the "escape" task.

Uses:
    from dm_control import suite
    env = suite.load(domain_name="quadruped", task_name="escape",
                     task_kwargs={"random": seed})
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from dm_control import suite
    _DM_CONTROL_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError):  # pragma: no cover
    suite = None  # type: ignore[assignment]
    _DM_CONTROL_AVAILABLE = False

from config import (
    MAX_STEPS as _MAX_STEPS,
    ORACLE_POSITION_NOISE_SCALE,
    ORACLE_CONTACT_NOISE_PROB,
    ORACLE_DELAY_STEPS,
    ORACLE_MISCLASSIFICATION_PROB,
)

# Terrain classification thresholds (torso x position)
_TERRAIN_FLAT_MAX: float = 2.0    # x < 2.0  → class 0 (flat)
_TERRAIN_INCLINE_MAX: float = 5.0  # 2.0 ≤ x < 5.0 → class 1 (incline)
#                                    x ≥ 5.0 → class 2 (gap/complex)

# Foot geom names in dm_control quadruped model
_FOOT_GEOM_NAMES: List[str] = ["foot_front_left", "foot_front_right", "foot_back_left", "foot_back_right"]


def _classify_terrain(x: float) -> int:
    """Classify terrain from torso x position.

    Parameters
    ----------
    x:
        Torso x-coordinate in world frame.

    Returns
    -------
    int
        0 = flat, 1 = incline, 2 = gap/complex.
    """
    if x < _TERRAIN_FLAT_MAX:
        return 0
    if x < _TERRAIN_INCLINE_MAX:
        return 1
    return 2


class QuadrupedTerrainEnv:
    """dm_control quadruped terrain environment wrapper for the STG PoC.

    Parameters
    ----------
    seed : int
        Random seed for the environment and observation noise.
    noise_scale : float
        Standard deviation of Gaussian noise added to observations.
    oracle_pos_noise_scale : float
        Standard deviation of Gaussian noise added to the torso position
        returned by :meth:`oracle`.  Defaults to
        ``config.ORACLE_POSITION_NOISE_SCALE`` (0.0 = disabled).
    oracle_contact_noise_prob : float
        Per-flag probability of flipping each boolean contact observation in
        :meth:`oracle`.  Defaults to ``config.ORACLE_CONTACT_NOISE_PROB``
        (0.0 = disabled).
    oracle_delay_steps : int
        Number of timesteps by which oracle observations are delayed (0, 1,
        or 2).  Defaults to ``config.ORACLE_DELAY_STEPS`` (0 = disabled).
    oracle_misclassification_prob : float
        Probability of returning a randomly wrong terrain class from
        :meth:`oracle`.  Defaults to ``config.ORACLE_MISCLASSIFICATION_PROB``
        (0.0 = disabled).
    """

    def __init__(
        self,
        seed: int,
        noise_scale: float = 0.01,
        oracle_pos_noise_scale: float = ORACLE_POSITION_NOISE_SCALE,
        oracle_contact_noise_prob: float = ORACLE_CONTACT_NOISE_PROB,
        oracle_delay_steps: int = ORACLE_DELAY_STEPS,
        oracle_misclassification_prob: float = ORACLE_MISCLASSIFICATION_PROB,
    ) -> None:
        """Initialise the environment.

        Parameters
        ----------
        seed:
            Random seed passed to dm_control task and numpy RNG.
        noise_scale:
            Gaussian noise added to flattened observations.
        oracle_pos_noise_scale:
            Std dev of Gaussian noise on torso position in oracle output.
        oracle_contact_noise_prob:
            Per-flag probability of flipping contact booleans in oracle output.
        oracle_delay_steps:
            Delay the oracle output by this many steps (0–2).
        oracle_misclassification_prob:
            Probability of returning a wrong terrain class from oracle.
        """
        if not _DM_CONTROL_AVAILABLE:
            raise ImportError(
                "dm_control is required. Install with: pip install dm_control"
            )
        self._seed = seed
        self._noise_scale = noise_scale
        self._oracle_pos_noise_scale = float(oracle_pos_noise_scale)
        self._oracle_contact_noise_prob = float(oracle_contact_noise_prob)
        self._oracle_delay_steps = max(0, int(oracle_delay_steps))
        self._oracle_misclassification_prob = float(oracle_misclassification_prob)
        self._rng = np.random.default_rng(seed)
        self._env = suite.load(
            domain_name="quadruped",
            task_name="escape",
            task_kwargs={"random": seed},
        )
        self._step_count: int = 0
        self._last_time_step = None

        # Delay buffer: holds up to oracle_delay_steps+1 states so that the
        # oldest entry in a full buffer is exactly oracle_delay_steps steps old.
        # When delay is disabled (0) we skip the buffer entirely.
        self._oracle_buffer: Deque[Dict] = (
            deque(maxlen=self._oracle_delay_steps + 1)
            if self._oracle_delay_steps > 0
            else deque()
        )

        # Cache spec info
        action_spec = self._env.action_spec()
        self._action_low = action_spec.minimum.copy()
        self._action_high = action_spec.maximum.copy()
        self._action_dim_val: int = int(action_spec.shape[0])

        # Compute obs dim from observation_spec (avoids triggering a rendering reset)
        obs_spec = self._env.observation_spec()
        self._obs_dim_val: int = int(sum(np.prod(spec.shape) for spec in obs_spec.values()))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Dimensionality of the flattened observation vector."""
        return self._obs_dim_val

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""
        return self._action_dim_val

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation.

        Returns
        -------
        np.ndarray
            Flattened, noise-augmented observation vector.
        """
        self._rng = np.random.default_rng(self._seed)
        self._step_count = 0
        self._oracle_buffer.clear()
        time_step = self._env.reset()
        self._last_time_step = time_step
        return self._obs_with_noise(time_step.observation)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance the environment by one control step.

        Parameters
        ----------
        action:
            Control action, clipped to action spec bounds before
            passing to dm_control.

        Returns
        -------
        obs : np.ndarray
            Noisy flattened observation.
        reward : float
            Task reward from dm_control.
        done : bool
            ``True`` if episode should terminate.
        info : dict
            Additional diagnostic information.
        """
        clipped = np.clip(action, self._action_low, self._action_high)
        time_step = self._env.step(clipped)
        self._last_time_step = time_step
        self._step_count += 1

        obs = self._obs_with_noise(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        done = bool(time_step.last()) or self._step_count >= _MAX_STEPS
        info: Dict = {
            "step_count": self._step_count,
            "discount": time_step.discount,
        }
        return obs, reward, done, info

    def oracle(self, t: int) -> Dict:
        """Return ground-truth predicates for the governor's ΔI computation.

        Applies controlled observation noise according to the parameters set
        at construction time (or their defaults from ``config``):

        * **Gaussian position noise** – zero-mean noise with std dev
          ``oracle_pos_noise_scale`` is added to each component of
          ``torso_pos``.
        * **Contact flip noise** – each contact boolean is independently
          flipped with probability ``oracle_contact_noise_prob``.
        * **Observation delay** – when ``oracle_delay_steps > 0`` the method
          returns the oracle state from that many steps ago; the current
          (noisy) state is always buffered first.
        * **Terrain misclassification** – with probability
          ``oracle_misclassification_prob`` the ``terrain_class`` field is
          replaced by a uniformly-random class different from the true one.

        Both the *baseline* and the *governor* conditions call this method on
        the same environment instance, so they observe identical noisy data.

        Parameters
        ----------
        t:
            Current timestep index (stored in the returned dict).

        Returns
        -------
        dict with keys:
            t, torso_pos, torso_upright, contact_flags, n_contacts,
            feasible, terrain_class.
        """
        physics = self._env.physics
        try:
            torso_pos = physics.named.data.xpos["torso"].tolist()
        except Exception:
            torso_pos = [0.0, 0.0, 0.0]

        # torso_upright: dot-product of torso z-axis with world z-axis ∈ [-1, 1].
        # Uses the standard dm_control helper physics.torso_upright() which
        # returns xmat['torso', 'zz'] — 1.0 = perfectly upright, -1.0 = inverted.
        try:
            torso_upright = float(physics.torso_upright())
        except Exception:
            torso_upright = 1.0

        # Per-foot contact detection
        contact_flags = self._detect_foot_contacts(physics)

        # --- Apply position noise ---
        if self._oracle_pos_noise_scale > 0.0:
            torso_pos = [
                v + float(self._rng.normal(0.0, self._oracle_pos_noise_scale))
                for v in torso_pos
            ]

        # --- Apply contact flip noise ---
        if self._oracle_contact_noise_prob > 0.0:
            contact_flags = [
                (not flag) if self._rng.random() < self._oracle_contact_noise_prob else flag
                for flag in contact_flags
            ]

        n_contacts = int(sum(contact_flags))
        feasible = bool(torso_upright > 0.5 and n_contacts >= 2)
        terrain_class = _classify_terrain(torso_pos[0])

        # --- Apply terrain misclassification ---
        if self._oracle_misclassification_prob > 0.0:
            if self._rng.random() < self._oracle_misclassification_prob:
                other_classes = [c for c in (0, 1, 2) if c != terrain_class]
                terrain_class = int(self._rng.choice(other_classes))

        state: Dict = {
            "t": t,
            "torso_pos": torso_pos,
            "torso_upright": torso_upright,
            "contact_flags": contact_flags,
            "n_contacts": n_contacts,
            "feasible": feasible,
            "terrain_class": terrain_class,
        }

        # --- Apply observation delay ---
        if self._oracle_delay_steps > 0:
            self._oracle_buffer.append(state)
            if len(self._oracle_buffer) > self._oracle_delay_steps:
                # Buffer is full: oldest entry is exactly oracle_delay_steps old.
                delayed = self._oracle_buffer[0]
                # Keep the current timestep index so callers can still
                # correlate oracle output with the environment step.
                return {**delayed, "t": t}
            # Buffer not yet full: return current state (no delay available yet).
            return state

        return state

    def action_spec(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return action bounds.

        Returns
        -------
        (low, high) : Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for each action dimension.
        """
        return self._action_low.copy(), self._action_high.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, observation) -> np.ndarray:
        """Concatenate all values in the observation OrderedDict.

        Parameters
        ----------
        observation:
            Ordered dict from dm_control time_step.observation.

        Returns
        -------
        np.ndarray
            1-D flattened array.
        """
        parts = [np.asarray(v).ravel() for v in observation.values()]
        return np.concatenate(parts).astype(np.float64)

    def _obs_with_noise(self, observation) -> np.ndarray:
        """Flatten observation and add Gaussian noise.

        Parameters
        ----------
        observation:
            Raw observation dict from dm_control.

        Returns
        -------
        np.ndarray
            Noisy flattened observation.
        """
        obs = self._flatten_obs(observation)
        noise = self._rng.normal(0.0, self._noise_scale, obs.shape)
        return obs + noise

    def _detect_foot_contacts(self, physics) -> List[bool]:
        """Detect per-foot ground contacts from physics data.

        Iterates the contact array and maps geom IDs to names.  Falls back
        to checking whether any geom name contains ``"foot"`` or ``"toe"``.
        Returns a conservative all-False estimate on any exception.

        Parameters
        ----------
        physics:
            dm_control Physics object.

        Returns
        -------
        List[bool]
            Four booleans: [lf_foot, rf_foot, lh_foot, rh_foot].
        """
        active_geoms: set = set()
        try:
            for contact in physics.data.contact:
                for geom_id in (contact.geom1, contact.geom2):
                    try:
                        name = physics.model.id2name(int(geom_id), "geom")
                        if name:
                            active_geoms.add(name)
                    except Exception:
                        pass
        except Exception:
            return [False, False, False, False]

        flags: List[bool] = []
        for foot in _FOOT_GEOM_NAMES:
            if foot in active_geoms:
                flags.append(True)
            else:
                # Fallback: check for partial name match ("foot" or "toe")
                # and matching leg identifier (e.g., "front_left")
                leg_id = foot.replace("foot_", "")
                matched = any(
                    ("foot" in g or "toe" in g)
                    for g in active_geoms
                    if leg_id in g
                )
                flags.append(matched)
        return flags
