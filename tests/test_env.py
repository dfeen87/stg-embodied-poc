"""tests/test_env.py — Tests for QuadrupedTerrainEnv (mocked dm_control)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Build a minimal dm_control mock so tests can run without the real library
# ---------------------------------------------------------------------------

def _make_dm_control_mock():
    """Construct a minimal mock of dm_control.suite that satisfies the env."""
    # Build action_spec mock
    action_spec = MagicMock()
    action_spec.shape = (12,)
    action_spec.minimum = np.full(12, -1.0)
    action_spec.maximum = np.full(12, 1.0)

    # Build observation dict mock
    obs_dict = {
        "egocentric_state": np.zeros(18),
        "velocimeter": np.zeros(3),
        "imu": np.zeros(6),
        "force_torque": np.zeros(24),
    }

    # Build time_step mocks
    def make_time_step(last=False, reward=0.0):
        ts = MagicMock()
        ts.observation = obs_dict
        ts.reward = reward
        ts.discount = 1.0 if not last else 0.0
        ts.last.return_value = last
        return ts

    # Build physics mock
    physics = MagicMock()
    physics.named.data.xpos.__getitem__ = lambda self, key: np.array([0.1, 0.0, 0.5])
    physics.named.data.xquat.__getitem__ = lambda self, key: np.array([0.98, 0.0, 0.0, 0.2])

    # No contacts by default
    physics.data.contact = []
    physics.model.id2name = lambda gid, kind: f"geom_{gid}"

    # Build observation spec mock (mirrors obs_dict shapes)
    obs_spec = {
        "egocentric_state": MagicMock(shape=(18,)),
        "velocimeter": MagicMock(shape=(3,)),
        "imu": MagicMock(shape=(6,)),
        "force_torque": MagicMock(shape=(24,)),
    }

    # Build env mock
    env_mock = MagicMock()
    env_mock.action_spec.return_value = action_spec
    env_mock.observation_spec.return_value = obs_spec
    env_mock.reset.return_value = make_time_step(last=False, reward=0.0)
    env_mock.step.return_value = make_time_step(last=False, reward=0.1)
    env_mock.physics = physics

    # Build suite mock
    suite_mock = MagicMock()
    suite_mock.load.return_value = env_mock

    return suite_mock, env_mock, make_time_step


@pytest.fixture
def mock_suite():
    """Patch dm_control.suite for all env tests."""
    suite_mock, env_mock, make_time_step = _make_dm_control_mock()

    dm_control_mod = types.ModuleType("dm_control")
    dm_control_mod.suite = suite_mock

    with patch.dict(sys.modules, {"dm_control": dm_control_mod, "dm_control.suite": suite_mock}):
        # Also patch the _DM_CONTROL_AVAILABLE flag in the module
        import envs.quadruped_terrain as qt_mod
        original = qt_mod._DM_CONTROL_AVAILABLE
        qt_mod._DM_CONTROL_AVAILABLE = True
        qt_mod.suite = suite_mock
        yield env_mock, make_time_step
        qt_mod._DM_CONTROL_AVAILABLE = original


# ---------------------------------------------------------------------------
# QuadrupedTerrainEnv — basic interface
# ---------------------------------------------------------------------------

class TestQuadrupedTerrainEnv:
    def test_obs_dim_positive(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        assert env.obs_dim > 0

    def test_action_dim_positive(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        assert env.action_dim == 12

    def test_reset_returns_array(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert len(obs) == env.obs_dim

    def test_step_returns_correct_structure(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        action = np.zeros(12)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_action_spec_returns_bounds(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        low, high = env.action_spec()
        assert low.shape == (12,)
        assert high.shape == (12,)
        assert np.all(low <= high)

    def test_done_after_max_steps(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        done = False
        for _ in range(120):
            _, _, done, _ = env.step(np.zeros(12))
            if done:
                break
        assert done


# ---------------------------------------------------------------------------
# QuadrupedTerrainEnv — oracle
# ---------------------------------------------------------------------------

class TestOracle:
    def test_oracle_has_required_keys(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        o = env.oracle(0)
        for key in ("t", "torso_pos", "torso_upright", "contact_flags",
                    "n_contacts", "feasible", "terrain_class"):
            assert key in o, f"Missing oracle key: {key}"

    def test_oracle_t_matches_argument(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        for t in (0, 5, 42):
            o = env.oracle(t)
            assert o["t"] == t

    def test_oracle_terrain_class_valid(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        o = env.oracle(0)
        assert o["terrain_class"] in (0, 1, 2)

    def test_oracle_contact_flags_length(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        o = env.oracle(0)
        assert len(o["contact_flags"]) == 4

    def test_oracle_n_contacts_consistent(self, mock_suite):
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0)
        env.reset()
        o = env.oracle(0)
        assert o["n_contacts"] == sum(o["contact_flags"])


# ---------------------------------------------------------------------------
# Oracle noise tests
# ---------------------------------------------------------------------------

class TestOracleNoise:
    """Tests for controlled oracle observation noise features."""

    def test_position_noise_changes_torso_pos(self, mock_suite):
        """Gaussian position noise should perturb torso_pos."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env_clean = QuadrupedTerrainEnv(seed=0, oracle_pos_noise_scale=0.0)
        env_noisy = QuadrupedTerrainEnv(seed=0, oracle_pos_noise_scale=1.0)
        env_clean.reset()
        env_noisy.reset()
        o_clean = env_clean.oracle(0)
        o_noisy = env_noisy.oracle(0)
        # With scale=1.0 the noisy position should differ from the clean one.
        assert not np.allclose(o_clean["torso_pos"], o_noisy["torso_pos"])

    def test_position_noise_disabled_keeps_original(self, mock_suite):
        """With scale=0.0, torso_pos should be identical across calls."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=42, oracle_pos_noise_scale=0.0)
        env.reset()
        o1 = env.oracle(0)
        o2 = env.oracle(0)
        assert np.allclose(o1["torso_pos"], o2["torso_pos"])

    def test_contact_noise_can_flip_flags(self, mock_suite):
        """With flip prob=1.0 every contact flag should be flipped."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        # Default mock has no contacts → all False; flipping all gives all True.
        env = QuadrupedTerrainEnv(seed=0, oracle_contact_noise_prob=1.0)
        env.reset()
        o = env.oracle(0)
        assert all(o["contact_flags"])
        assert o["n_contacts"] == 4

    def test_contact_noise_disabled_leaves_flags(self, mock_suite):
        """With flip prob=0.0 contact flags should match physics contacts."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0, oracle_contact_noise_prob=0.0)
        env.reset()
        o = env.oracle(0)
        # Default mock has no active contacts → all False.
        assert not any(o["contact_flags"])

    def test_misclassification_prob_1_always_wrong(self, mock_suite):
        """With misclassification_prob=1.0, terrain_class should never equal
        the true class (x=0.1 → true class 0)."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=7, oracle_misclassification_prob=1.0)
        env.reset()
        for t in range(20):
            o = env.oracle(t)
            assert o["terrain_class"] != 0, (
                f"Expected misclassified class at t={t}, got 0"
            )

    def test_misclassification_prob_0_always_correct(self, mock_suite):
        """With misclassification_prob=0.0, terrain_class matches the true
        classification (x=0.1 → class 0)."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0, oracle_misclassification_prob=0.0)
        env.reset()
        o = env.oracle(0)
        assert o["terrain_class"] == 0

    def test_delay_1_returns_previous_state(self, mock_suite):
        """With delay=1, oracle at step t should return state buffered at t-1."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0, oracle_delay_steps=1)
        env.reset()
        # First call fills the buffer but returns current state (no prior state).
        o0 = env.oracle(0)
        # Second call should return the state buffered from step 0.
        o1 = env.oracle(1)
        # The delayed result at t=1 should carry the data from t=0.
        assert np.allclose(o1["torso_pos"], o0["torso_pos"])
        # But the timestep index should reflect the current step.
        assert o1["t"] == 1

    def test_delay_0_no_buffering(self, mock_suite):
        """With delay=0, oracle should return current state immediately."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0, oracle_delay_steps=0)
        env.reset()
        o = env.oracle(5)
        assert o["t"] == 5

    def test_oracle_buffer_cleared_on_reset(self, mock_suite):
        """Calling reset() should clear the delay buffer."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=0, oracle_delay_steps=2)
        env.reset()
        env.oracle(0)
        env.oracle(1)
        env.reset()
        assert len(env._oracle_buffer) == 0

    def test_noisy_oracle_has_required_keys(self, mock_suite):
        """Noisy oracle output must still contain all required keys."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(
            seed=0,
            oracle_pos_noise_scale=0.1,
            oracle_contact_noise_prob=0.1,
            oracle_delay_steps=1,
            oracle_misclassification_prob=0.1,
        )
        env.reset()
        env.oracle(0)  # warm up buffer
        o = env.oracle(1)
        for key in ("t", "torso_pos", "torso_upright", "contact_flags",
                    "n_contacts", "feasible", "terrain_class"):
            assert key in o, f"Missing oracle key: {key}"

    def test_n_contacts_consistent_with_noisy_flags(self, mock_suite):
        """n_contacts should equal sum(contact_flags) even after noise."""
        from envs.quadruped_terrain import QuadrupedTerrainEnv
        env = QuadrupedTerrainEnv(seed=3, oracle_contact_noise_prob=0.5)
        env.reset()
        for t in range(10):
            o = env.oracle(t)
            assert o["n_contacts"] == sum(o["contact_flags"])


# ---------------------------------------------------------------------------
# Terrain classification helper
# ---------------------------------------------------------------------------

class TestTerrainClassification:
    def test_flat_below_2(self):
        from envs.quadruped_terrain import _classify_terrain
        assert _classify_terrain(1.0) == 0
        assert _classify_terrain(0.0) == 0
        assert _classify_terrain(-5.0) == 0

    def test_incline_2_to_5(self):
        from envs.quadruped_terrain import _classify_terrain
        assert _classify_terrain(2.0) == 1
        assert _classify_terrain(3.5) == 1
        assert _classify_terrain(4.99) == 1

    def test_gap_at_5_and_beyond(self):
        from envs.quadruped_terrain import _classify_terrain
        assert _classify_terrain(5.0) == 2
        assert _classify_terrain(10.0) == 2
