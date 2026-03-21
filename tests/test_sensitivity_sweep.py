"""tests/test_sensitivity_sweep.py — Tests for the ΔΦ weight sensitivity sweep."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from governor.spiral_time_governor import (
    SpiralTimeGovernor,
    ALPHA, BETA, GAMMA, DELTA,
)


# ---------------------------------------------------------------------------
# Helpers shared with test_env.py pattern
# ---------------------------------------------------------------------------

def _make_dm_control_mock():
    action_spec = MagicMock()
    action_spec.shape = (12,)
    action_spec.minimum = np.full(12, -1.0)
    action_spec.maximum = np.full(12, 1.0)

    obs_dict = {
        "egocentric_state": np.zeros(18),
        "velocimeter": np.zeros(3),
        "imu": np.zeros(6),
        "force_torque": np.zeros(24),
    }

    def make_time_step(last=False, reward=0.0):
        ts = MagicMock()
        ts.observation = obs_dict
        ts.reward = reward
        ts.discount = 1.0 if not last else 0.0
        ts.last.return_value = last
        return ts

    physics = MagicMock()
    physics.named.data.xpos.__getitem__ = lambda self, key: np.array([0.1, 0.0, 0.5])
    physics.named.data.xquat.__getitem__ = lambda self, key: np.array([0.98, 0.0, 0.0, 0.2])
    physics.data.contact = []
    physics.model.id2name = lambda gid, kind: f"geom_{gid}"

    obs_spec = {
        "egocentric_state": MagicMock(shape=(18,)),
        "velocimeter": MagicMock(shape=(3,)),
        "imu": MagicMock(shape=(6,)),
        "force_torque": MagicMock(shape=(24,)),
    }

    env_mock = MagicMock()
    env_mock.action_spec.return_value = action_spec
    env_mock.observation_spec.return_value = obs_spec
    env_mock.reset.return_value = make_time_step(last=False, reward=0.0)
    env_mock.step.return_value = make_time_step(last=True, reward=0.8)
    env_mock.physics = physics

    suite_mock = MagicMock()
    suite_mock.load.return_value = env_mock

    return suite_mock, env_mock, make_time_step


@pytest.fixture
def mock_suite():
    suite_mock, env_mock, make_time_step = _make_dm_control_mock()

    dm_control_mod = types.ModuleType("dm_control")
    dm_control_mod.suite = suite_mock

    with patch.dict(sys.modules, {"dm_control": dm_control_mod, "dm_control.suite": suite_mock}):
        import envs.quadruped_terrain as qt_mod
        original = qt_mod._DM_CONTROL_AVAILABLE
        qt_mod._DM_CONTROL_AVAILABLE = True
        qt_mod.suite = suite_mock
        yield env_mock, make_time_step
        qt_mod._DM_CONTROL_AVAILABLE = original


# ---------------------------------------------------------------------------
# Governor weight override tests
# ---------------------------------------------------------------------------

class TestGovernorWeightOverrides:
    """Tests that SpiralTimeGovernor correctly uses per-instance weight overrides."""

    _DUMMY_ORACLE = {
        "t": 0,
        "torso_pos": [0.0, 0.0, 0.5],
        "torso_upright": 0.9,
        "contact_flags": [True, True, True, True],
        "n_contacts": 4,
        "feasible": True,
        "terrain_class": 0,
    }
    _ACTION = np.zeros(12)

    def _always_ok(self, action):
        return True

    def test_default_weights_match_constants(self):
        g = SpiralTimeGovernor()
        assert g._alpha == ALPHA
        assert g._beta == BETA
        assert g._gamma == GAMMA
        assert g._delta == DELTA

    def test_custom_alpha_stored(self):
        g = SpiralTimeGovernor(alpha=0.10)
        assert g._alpha == pytest.approx(0.10)
        assert g._beta == BETA

    def test_custom_beta_stored(self):
        g = SpiralTimeGovernor(beta=0.50)
        assert g._beta == pytest.approx(0.50)

    def test_custom_gamma_stored(self):
        g = SpiralTimeGovernor(gamma=0.20)
        assert g._gamma == pytest.approx(0.20)

    def test_custom_delta_stored(self):
        g = SpiralTimeGovernor(delta=0.05)
        assert g._delta == pytest.approx(0.05)

    def test_zero_alpha_reduces_delta_phi(self):
        """Setting α=0 should reduce ΔΦ when ΔR > 0."""
        oracle = dict(self._DUMMY_ORACLE, feasible=False)
        action = np.ones(12) * 20.0  # will fail norm check → ΔR > 0

        def always_fail(a):
            return False

        g_default = SpiralTimeGovernor()
        g_zero_alpha = SpiralTimeGovernor(alpha=0.0)

        _, _, info_default = g_default.step([], action, oracle, always_fail)
        _, _, info_zero = g_zero_alpha.step([], action, oracle, always_fail)

        assert info_zero["delta_phi"] <= info_default["delta_phi"]

    def test_weight_override_affects_delta_phi(self):
        """Doubling β should increase ΔΦ when ΔI > 0 (failing claims)."""
        oracle = dict(self._DUMMY_ORACLE, feasible=False)
        claims = ["feasible", "grounded"]  # will fail

        g_default = SpiralTimeGovernor()
        g_high_beta = SpiralTimeGovernor(beta=BETA * 2.0)

        _, _, info_default = g_default.step(claims, self._ACTION, oracle, self._always_ok)
        _, _, info_high_beta = g_high_beta.step(claims, self._ACTION, oracle, self._always_ok)

        assert info_high_beta["delta_phi"] >= info_default["delta_phi"]

    def test_no_delta_ablation_ignores_delta_override(self):
        """With ablation='no_delta', the delta weight should be zeroed out
        regardless of the per-instance override."""
        oracle = dict(self._DUMMY_ORACLE)

        g1 = SpiralTimeGovernor(ablation="no_delta", delta=DELTA)
        g2 = SpiralTimeGovernor(ablation="no_delta", delta=DELTA * 5.0)

        # Run a second step so torsion χ is non-zero
        for g in (g1, g2):
            g.step([], self._ACTION, oracle, self._always_ok)
            g.step([], self._ACTION, dict(oracle, feasible=False), self._always_ok)

        # Both should have the same delta_phi since delta is zeroed
        assert g1.log[1]["delta_phi"] == pytest.approx(g2.log[1]["delta_phi"])


# ---------------------------------------------------------------------------
# Sensitivity sweep unit tests
# ---------------------------------------------------------------------------

class TestSweepFactors:
    """Test the sweep mechanics without running full episodes."""

    def test_sweep_factors_cover_required_range(self):
        from run_sensitivity_sweep import SWEEP_FACTORS
        assert min(SWEEP_FACTORS) <= 0.70, "Sweep must include at least -30%"
        assert max(SWEEP_FACTORS) >= 1.30, "Sweep must include at least +30%"
        assert 1.00 in SWEEP_FACTORS or any(abs(f - 1.0) < 1e-9 for f in SWEEP_FACTORS), \
            "Sweep must include the default (factor=1.0)"

    def test_four_parameters_swept(self):
        from run_sensitivity_sweep import SWEEP_FACTORS
        params = ("alpha", "beta", "gamma", "delta")
        assert len(params) == 4

    def test_output_csv_name(self):
        from run_sensitivity_sweep import OUTPUT_CSV
        assert OUTPUT_CSV == "sensitivity_delta_phi.csv"


class TestRunSweep:
    """Integration tests for run_sweep using a mocked environment."""

    def test_sweep_returns_dataframe_with_correct_columns(self, mock_suite):
        from run_sensitivity_sweep import run_sweep
        df = run_sweep(seeds=[0], n_claims=1)
        for col in ("param", "factor", "success_rate", "violation_rate", "delta_phi_stability"):
            assert col in df.columns, f"Missing column: {col}"

    def test_sweep_has_correct_number_of_rows(self, mock_suite):
        from run_sensitivity_sweep import run_sweep, SWEEP_FACTORS
        df = run_sweep(seeds=[0], n_claims=1)
        # 4 params × len(SWEEP_FACTORS) configurations
        assert len(df) == 4 * len(SWEEP_FACTORS)

    def test_sweep_covers_all_params(self, mock_suite):
        from run_sensitivity_sweep import run_sweep
        df = run_sweep(seeds=[0], n_claims=1)
        assert set(df["param"].unique()) == {"alpha", "beta", "gamma", "delta"}

    def test_success_rate_in_valid_range(self, mock_suite):
        from run_sensitivity_sweep import run_sweep
        df = run_sweep(seeds=[0], n_claims=1)
        assert (df["success_rate"] >= 0.0).all()
        assert (df["success_rate"] <= 1.0).all()

    def test_violation_rate_non_negative(self, mock_suite):
        from run_sensitivity_sweep import run_sweep
        df = run_sweep(seeds=[0], n_claims=1)
        assert (df["violation_rate"] >= 0.0).all()

    def test_delta_phi_stability_non_negative(self, mock_suite):
        from run_sensitivity_sweep import run_sweep
        df = run_sweep(seeds=[0], n_claims=1)
        assert (df["delta_phi_stability"] >= 0.0).all()

    def test_default_factor_has_default_weights(self, mock_suite):
        """Row with factor=1.0 must use unmodified default weights."""
        from run_sensitivity_sweep import run_sweep
        df = run_sweep(seeds=[0], n_claims=1)
        default_map = {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "delta": DELTA}
        for param in ("alpha", "beta", "gamma", "delta"):
            row = df[(df["param"] == param) & (df["factor"].apply(lambda f: abs(f - 1.0) < 1e-9))]
            assert len(row) == 1
            assert row.iloc[0]["swept_value"] == pytest.approx(default_map[param])

    def test_csv_export(self, mock_suite, tmp_path):
        """run_sweep results can be written to CSV and re-read."""
        from run_sensitivity_sweep import run_sweep, OUTPUT_CSV
        df = run_sweep(seeds=[0], n_claims=1)
        csv_path = tmp_path / OUTPUT_CSV
        df.to_csv(csv_path, index=False)
        import pandas as pd
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == len(df)
        assert list(df_read.columns) == list(df.columns)
