"""envs/quadruped_terrain.py

MuJoCo environment wrapper for the STG PoC, built on dm_control's built-in
quadruped domain with the "escape" task.

Uses:
    from dm_control import suite
    env = suite.load(domain_name="quadruped", task_name="escape",
                     task_kwargs={"random": seed})
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from dm_control import suite
    _DM_CONTROL_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError):  # pragma: no cover
    _DM_CONTROL_AVAILABLE = False

# Terrain classification thresholds (torso x position)
_TERRAIN_FLAT_MAX: float = 2.0    # x < 2.0  → class 0 (flat)
_TERRAIN_INCLINE_MAX: float = 5.0  # 2.0 ≤ x < 5.0 → class 1 (incline)
#                                    x ≥ 5.0 → class 2 (gap/complex)

# Foot geom names in dm_control quadruped model
_FOOT_GEOM_NAMES: List[str] = ["lf_foot", "rf_foot", "lh_foot", "rh_foot"]

# Maximum episode steps
_MAX_STEPS: int = 120


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
    """

    def __init__(self, seed: int, noise_scale: float = 0.01) -> None:
        """Initialise the environment.

        Parameters
        ----------
        seed:
            Random seed passed to dm_control task and numpy RNG.
        noise_scale:
            Gaussian noise added to flattened observations.
        """
        if not _DM_CONTROL_AVAILABLE:
            raise ImportError(
                "dm_control is required. Install with: pip install dm_control"
            )
        self._seed = seed
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
        self._env = suite.load(
            domain_name="quadruped",
            task_name="escape",
            task_kwargs={"random": seed},
        )
        self._step_count: int = 0
        self._last_time_step = None

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

        # torso_upright: w-component of torso quaternion ∈ [0, 1]
        try:
            quat = physics.named.data.xquat["torso"]
            # w is the first element in MuJoCo quaternion (w, x, y, z)
            torso_upright = float(abs(quat[0]))
        except Exception:
            torso_upright = 1.0

        # Per-foot contact detection
        contact_flags = self._detect_foot_contacts(physics)
        n_contacts = int(sum(contact_flags))

        feasible = bool(torso_upright > 0.5 and n_contacts >= 2)
        terrain_class = _classify_terrain(torso_pos[0])

        return {
            "t": t,
            "torso_pos": torso_pos,
            "torso_upright": torso_upright,
            "contact_flags": contact_flags,
            "n_contacts": n_contacts,
            "feasible": feasible,
            "terrain_class": terrain_class,
        }

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
                matched = any(
                    ("foot" in g or "toe" in g)
                    for g in active_geoms
                    if foot.split("_")[0] in g  # e.g. "lf" in geom name
                )
                flags.append(matched)
        return flags
