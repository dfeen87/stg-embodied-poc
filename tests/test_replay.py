"""tests/test_replay.py — Tests for the counterfactual replay module."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared dm_control mock (mirrors test_env.py pattern)
# ---------------------------------------------------------------------------

def _make_dm_control_mock():
    """Build a minimal dm_control mock that satisfies QuadrupedTerrainEnv."""
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

    def make_time_step(last=False, reward=0.1):
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
    physics.torso_upright = lambda: 0.95

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
    env_mock.step.return_value = make_time_step(last=False, reward=0.1)
    env_mock.physics = physics

    suite_mock = MagicMock()
    suite_mock.load.return_value = env_mock

    return suite_mock, env_mock, make_time_step


@pytest.fixture
def mock_dm_control():
    """Patch dm_control.suite for replay tests."""
    suite_mock, env_mock, make_time_step = _make_dm_control_mock()

    dm_control_mod = types.ModuleType("dm_control")
    dm_control_mod.suite = suite_mock

    with patch.dict(
        sys.modules,
        {"dm_control": dm_control_mod, "dm_control.suite": suite_mock},
    ):
        import envs.quadruped_terrain as qt_mod

        original_flag = qt_mod._DM_CONTROL_AVAILABLE
        original_suite = qt_mod.suite
        qt_mod._DM_CONTROL_AVAILABLE = True
        qt_mod.suite = suite_mock
        yield suite_mock, env_mock, make_time_step
        qt_mod._DM_CONTROL_AVAILABLE = original_flag
        qt_mod.suite = original_suite


# ---------------------------------------------------------------------------
# _record_phase
# ---------------------------------------------------------------------------

class TestRecordPhase:
    def test_returns_nonempty_records(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import _record_phase

        recorded, initial_obs = _record_phase(
            seed=0, hallucination_prob=0.45, n_claims=3
        )
        assert len(recorded) > 0
        assert isinstance(initial_obs, np.ndarray)

    def test_records_have_correct_fields(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import _record_phase, RecordedStep

        recorded, _ = _record_phase(seed=1, hallucination_prob=0.45, n_claims=2)
        step = recorded[0]
        assert isinstance(step, RecordedStep)
        assert isinstance(step.proposed_action, np.ndarray)
        assert isinstance(step.claims, list)
        assert len(step.claims) == 2
        assert isinstance(step.oracle_state, dict)

    def test_deterministic_across_calls(self, mock_dm_control):
        from replay.counterfactual_replay import _record_phase

        rec1, obs1 = _record_phase(seed=42, hallucination_prob=0.45, n_claims=3)
        rec2, obs2 = _record_phase(seed=42, hallucination_prob=0.45, n_claims=3)

        assert len(rec1) == len(rec2)
        for a, b in zip(rec1, rec2):
            np.testing.assert_array_equal(a.proposed_action, b.proposed_action)
            assert a.claims == b.claims

    def test_different_seeds_give_different_actions(self, mock_dm_control):
        from replay.counterfactual_replay import _record_phase

        rec0, _ = _record_phase(seed=0, hallucination_prob=0.45, n_claims=3)
        rec1, _ = _record_phase(seed=1, hallucination_prob=0.45, n_claims=3)

        # At least one action must differ between seeds
        any_diff = any(
            not np.array_equal(a.proposed_action, b.proposed_action)
            for a, b in zip(rec0, rec1)
        )
        assert any_diff


# ---------------------------------------------------------------------------
# _replay_track_a
# ---------------------------------------------------------------------------

class TestReplayTrackA:
    def test_track_a_returns_step_logs(self, mock_dm_control):
        from replay.counterfactual_replay import _record_phase, _replay_track_a

        recorded, _ = _record_phase(seed=0, hallucination_prob=0.45, n_claims=3)
        logs = _replay_track_a(seed=0, recorded=recorded)

        assert len(logs) > 0
        log = logs[0]
        assert "reward" in log
        assert "constraint_violation" in log
        assert "proposed_action_norm" in log
        assert "executed_action_norm" in log
        # Track A: no filtering → action_delta_norm must be 0
        assert log["action_delta_norm"] == pytest.approx(0.0)

    def test_track_a_no_gov_info(self, mock_dm_control):
        from replay.counterfactual_replay import _record_phase, _replay_track_a

        recorded, _ = _record_phase(seed=0, hallucination_prob=0.45, n_claims=3)
        logs = _replay_track_a(seed=0, recorded=recorded)

        # Governor keys should NOT be present in Track A logs
        for log in logs:
            assert "mode" not in log
            assert "phi" not in log


# ---------------------------------------------------------------------------
# _replay_track_b
# ---------------------------------------------------------------------------

class TestReplayTrackB:
    def test_track_b_returns_step_logs_with_gov_info(self, mock_dm_control):
        from replay.counterfactual_replay import _record_phase, _replay_track_b

        recorded, _ = _record_phase(seed=0, hallucination_prob=0.45, n_claims=3)
        logs = _replay_track_b(seed=0, recorded=recorded, ablation="none")

        assert len(logs) > 0
        log = logs[0]
        assert "mode" in log
        assert "phi" in log
        assert "delta_phi" in log
        assert "constraint_violation" in log

    def test_track_b_always_execute_ablation_passes_through(self, mock_dm_control):
        """With always_execute ablation the governor never enters SAFE mode."""
        from replay.counterfactual_replay import _record_phase, _replay_track_b

        recorded, _ = _record_phase(seed=0, hallucination_prob=0.0, n_claims=3)
        logs = _replay_track_b(seed=0, recorded=recorded, ablation="always_execute")

        # always_execute → all modes should be EXECUTE, no interventions
        for log in logs:
            assert log["mode"] == "EXECUTE"


# ---------------------------------------------------------------------------
# _compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def _make_logs(self, n: int, violation: bool, reward: float):
        return [
            {
                "t": i,
                "reward": reward,
                "constraint_violation": violation,
                "action_delta_norm": 0.0,
            }
            for i in range(n)
        ]

    def test_no_violations(self):
        from replay.counterfactual_replay import _compute_summary

        a = self._make_logs(10, False, 0.5)
        b = self._make_logs(10, False, 0.5)
        s = _compute_summary(a, b)
        assert s["track_a_violations"] == 0
        assert s["track_b_violations"] == 0
        assert s["delta_violations"] == 0

    def test_delta_violations_positive_when_track_a_worse(self):
        from replay.counterfactual_replay import _compute_summary

        a = self._make_logs(10, True, 0.0)   # all violating
        b = self._make_logs(10, False, 0.5)  # none violating
        s = _compute_summary(a, b)
        assert s["track_a_violations"] == 10
        assert s["track_b_violations"] == 0
        assert s["delta_violations"] == 10

    def test_stg_interventions_counted(self):
        from replay.counterfactual_replay import _compute_summary

        a = self._make_logs(5, False, 0.1)
        b = [
            {"t": i, "reward": 0.1, "constraint_violation": False, "action_delta_norm": 0.5}
            for i in range(5)
        ]
        s = _compute_summary(a, b)
        assert s["stg_interventions"] == 5


# ---------------------------------------------------------------------------
# run_counterfactual_replay (integration)
# ---------------------------------------------------------------------------

class TestRunCounterfactualReplay:
    def test_produces_json_file(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replay

        result = run_counterfactual_replay(
            seed=0,
            condition="governor",
            n_claims=3,
            output_dir=tmp_path,
        )
        json_file = tmp_path / "replay_seed0_governor.json"
        assert json_file.exists()

        with open(json_file) as fh:
            data = json.load(fh)

        assert data["seed"] == 0
        assert data["condition"] == "governor"
        assert "track_a" in data
        assert "track_b" in data
        assert "summary" in data

    def test_result_fields(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replay, ReplayResult

        result = run_counterfactual_replay(
            seed=1, condition="baseline", n_claims=2, output_dir=tmp_path
        )
        assert isinstance(result, ReplayResult)
        assert result.seed == 1
        assert result.condition == "baseline"
        assert result.record_steps > 0
        assert len(result.track_a) > 0
        assert len(result.track_b) > 0

    def test_track_a_and_b_same_length(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replay

        result = run_counterfactual_replay(
            seed=0, condition="governor", n_claims=3, output_dir=tmp_path
        )
        # Both tracks replay the same recorded sequence so must have same length
        assert len(result.track_a) == len(result.track_b)

    def test_track_a_identical_proposed_and_executed(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replay

        result = run_counterfactual_replay(
            seed=0, condition="governor", n_claims=3, output_dir=tmp_path
        )
        for log in result.track_a:
            assert log["action_delta_norm"] == pytest.approx(0.0), (
                "Track A must pass raw actions through unchanged"
            )

    def test_deterministic_across_calls(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replay

        r1 = run_counterfactual_replay(
            seed=7, condition="governor", n_claims=3, output_dir=tmp_path
        )
        r2 = run_counterfactual_replay(
            seed=7, condition="governor", n_claims=3, output_dir=tmp_path
        )
        assert r1.summary == r2.summary

    def test_summary_keys_present(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replay

        result = run_counterfactual_replay(
            seed=0, condition="governor", n_claims=3, output_dir=tmp_path
        )
        required = {
            "track_a_violations", "track_b_violations",
            "track_a_violation_rate", "track_b_violation_rate",
            "delta_violations", "delta_violation_rate",
            "delta_total_reward", "stg_interventions",
        }
        assert required.issubset(set(result.summary.keys()))


# ---------------------------------------------------------------------------
# run_counterfactual_replays (multi-seed)
# ---------------------------------------------------------------------------

class TestRunCounterfactualReplays:
    def test_multi_seed(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replays

        results = run_counterfactual_replays(
            seeds=[0, 1],
            condition="governor",
            n_claims=3,
            output_dir=tmp_path,
        )
        assert len(results) == 2
        assert results[0].seed == 0
        assert results[1].seed == 1

    def test_output_files_created(self, mock_dm_control, tmp_path):
        from replay.counterfactual_replay import run_counterfactual_replays

        run_counterfactual_replays(
            seeds=[0, 2],
            condition="ablation_a",
            n_claims=3,
            output_dir=tmp_path,
        )
        assert (tmp_path / "replay_seed0_ablation_a.json").exists()
        assert (tmp_path / "replay_seed2_ablation_a.json").exists()
