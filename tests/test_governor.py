"""tests/test_governor.py — Unit tests for the Spiral-Time Governor."""

from __future__ import annotations

import numpy as np
import pytest

from governor.spiral_time_governor import (
    GovernorState,
    SpiralTimeGovernor,
    PHI0,
    TAU1,
    TAU2,
    WR, WI, WC,
    ALPHA, BETA, GAMMA, DELTA,
    _verify_claim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_ok(action: np.ndarray) -> bool:
    return True


def _always_fail(action: np.ndarray) -> bool:
    return False


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


# ---------------------------------------------------------------------------
# GovernorState
# ---------------------------------------------------------------------------

class TestGovernorState:
    def test_repr_contains_key_fields(self):
        state = GovernorState(
            t=5, phi=0.8, chi=0.02, delta_phi=0.1,
            mode="EXECUTE", delta_R=0.0, delta_I=0.0, delta_C=0.0,
        )
        r = repr(state)
        assert "t=5" in r
        assert "EXECUTE" in r
        assert "φ=" in r or "phi" in r.lower()


# ---------------------------------------------------------------------------
# Parameter constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_weights_sum_to_one(self):
        assert abs(WR + WI + WC - 1.0) < 1e-9

    def test_instability_weights_sum_to_one(self):
        assert abs(ALPHA + BETA + GAMMA + DELTA - 1.0) < 1e-9

    def test_tau_ordering(self):
        assert TAU1 < TAU2

    def test_phi0_in_range(self):
        assert 0.0 <= PHI0 <= 1.0


# ---------------------------------------------------------------------------
# Verify claim
# ---------------------------------------------------------------------------

class TestVerifyClaim:
    def test_feasible_passes_when_oracle_feasible(self):
        assert _verify_claim("feasible", {"feasible": True}) is True

    def test_feasible_fails_when_oracle_not_feasible(self):
        assert _verify_claim("The robot is feasible and stable.", {"feasible": False}) is False

    def test_contact_grounded_passes(self):
        assert _verify_claim("grounded", {"n_contacts": 3}) is True

    def test_contact_grounded_fails(self):
        assert _verify_claim("grounded", {"n_contacts": 1}) is False

    def test_upright_passes(self):
        assert _verify_claim("upright", {"torso_upright": 0.8}) is True

    def test_upright_fails(self):
        assert _verify_claim("upright", {"torso_upright": 0.3}) is False

    def test_flat_passes(self):
        assert _verify_claim("flat", {"terrain_class": 0}) is True

    def test_flat_fails(self):
        assert _verify_claim("flat", {"terrain_class": 1}) is False

    def test_incline_passes(self):
        assert _verify_claim("incline", {"terrain_class": 1}) is True

    def test_incline_fails(self):
        assert _verify_claim("incline", {"terrain_class": 0}) is False

    def test_unknown_claim_passes(self):
        assert _verify_claim("something unrecognised", {}) is True

    def test_empty_claim_passes(self):
        assert _verify_claim("", {}) is True


# ---------------------------------------------------------------------------
# SpiralTimeGovernor — construction and reset
# ---------------------------------------------------------------------------

class TestGovernorInit:
    def test_invalid_ablation_raises(self):
        with pytest.raises(ValueError):
            SpiralTimeGovernor(ablation="invalid")

    def test_valid_ablations(self):
        for abl in ("none", "no_delta", "always_execute"):
            g = SpiralTimeGovernor(ablation=abl)
            assert g.ablation == abl

    def test_initial_log_empty(self):
        g = SpiralTimeGovernor()
        assert g.log == []

    def test_reset_clears_log(self):
        g = SpiralTimeGovernor()
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        g.reset()
        assert g.log == []


# ---------------------------------------------------------------------------
# SpiralTimeGovernor — mode switching
# ---------------------------------------------------------------------------

class TestModeSwitching:
    def test_execute_mode_on_low_instability(self):
        """With all-correct claims and passing constraints → EXECUTE."""
        g = SpiralTimeGovernor(ablation="none")
        oracle = dict(_DUMMY_ORACLE, feasible=True, n_contacts=4, torso_upright=0.9)
        mode, _, _ = g.step(
            ["The robot is feasible and stable."],
            _ACTION, oracle, _always_ok,
        )
        assert mode == "EXECUTE"

    def test_safe_mode_on_high_instability(self):
        """With all-failing claims and failing constraints → SAFE."""
        g = SpiralTimeGovernor(ablation="none")
        oracle = dict(
            _DUMMY_ORACLE, feasible=False, n_contacts=0,
            torso_upright=0.1, terrain_class=0,
        )
        # Claims that all fail + constraint failure → high ΔΦ
        claims = [
            "feasible", "grounded", "upright", "feasible", "grounded", "upright",
        ]
        mode, _, _ = g.step(claims, _ACTION * 0 + 100.0, oracle, _always_fail)
        assert mode == "SAFE"

    def test_always_execute_ablation(self):
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(
            _DUMMY_ORACLE, feasible=False, n_contacts=0, torso_upright=0.1,
        )
        claims = ["feasible", "grounded", "upright"]
        mode, _, _ = g.step(claims, _ACTION, oracle, _always_fail)
        assert mode == "EXECUTE"


# ---------------------------------------------------------------------------
# SpiralTimeGovernor — action gating
# ---------------------------------------------------------------------------

class TestActionGating:
    def test_safe_mode_returns_zeros(self):
        g = SpiralTimeGovernor(ablation="none")
        oracle = dict(
            _DUMMY_ORACLE, feasible=False, n_contacts=0,
            torso_upright=0.1, terrain_class=0,
        )
        claims = ["feasible"] * 10
        action = np.ones(12) * 5.0
        mode, gated, _ = g.step(claims, action, oracle, _always_fail)
        if mode == "SAFE":
            np.testing.assert_array_equal(gated, np.zeros_like(action))

    def test_execute_passes_valid_action(self):
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE)
        action = np.ones(12) * 0.1
        mode, gated, _ = g.step([], action, oracle, _always_ok)
        assert mode == "EXECUTE"
        np.testing.assert_array_equal(gated, action)

    def test_execute_falls_back_on_constraint_fail(self):
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE)
        action = np.ones(12) * 5.0
        mode, gated, _ = g.step([], action, oracle, _always_fail)
        np.testing.assert_array_equal(gated, np.zeros_like(action))


# ---------------------------------------------------------------------------
# SpiralTimeGovernor — logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_grows_each_step(self):
        g = SpiralTimeGovernor()
        for i in range(5):
            g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        assert len(g.log) == 5

    def test_log_entry_has_required_keys(self):
        g = SpiralTimeGovernor()
        g.step(["The torso is upright and balanced."], _ACTION, _DUMMY_ORACLE, _always_ok)
        entry = g.log[0]
        for key in ("t", "phi", "chi", "delta_phi", "mode", "delta_R", "delta_I", "delta_C"):
            assert key in entry, f"Missing key: {key}"

    def test_log_is_copy(self):
        """Modifying the returned log should not affect internal state."""
        g = SpiralTimeGovernor()
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        log = g.log
        log.clear()
        assert len(g.log) == 1

    def test_t_increments(self):
        g = SpiralTimeGovernor()
        for i in range(3):
            g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        ts = [e["t"] for e in g.log]
        assert ts == [0, 1, 2]


# ---------------------------------------------------------------------------
# SpiralTimeGovernor — ablations
# ---------------------------------------------------------------------------

class TestAblations:
    def test_no_delta_gives_different_delta_phi(self):
        """Ablation A (no_delta) should generally differ from full governor."""
        oracle = dict(_DUMMY_ORACLE, feasible=True, torso_upright=0.6)
        claims = ["The torso is upright.", "Terrain is flat ahead."]

        g_full = SpiralTimeGovernor(ablation="none")
        g_full.step(claims, _ACTION, oracle, _always_ok)
        # Inject some chi by running a second step with different oracle
        oracle2 = dict(_DUMMY_ORACLE, feasible=False, n_contacts=1, torso_upright=0.3)
        g_full.step(claims, _ACTION, oracle2, _always_ok)
        chi_nonzero = abs(g_full.log[1]["chi"]) > 0

        g_nd = SpiralTimeGovernor(ablation="no_delta")
        g_nd.step(claims, _ACTION, oracle, _always_ok)
        g_nd.step(claims, _ACTION, oracle2, _always_ok)

        if chi_nonzero:
            assert g_full.log[1]["delta_phi"] != g_nd.log[1]["delta_phi"]

    def test_phi_in_range(self):
        g = SpiralTimeGovernor()
        for _ in range(10):
            g.step(["feasible"], _ACTION, _DUMMY_ORACLE, _always_ok)
        for entry in g.log:
            assert 0.0 <= entry["phi"] <= 1.0, f"phi={entry['phi']} out of range"

    def test_delta_phi_in_range(self):
        g = SpiralTimeGovernor()
        for _ in range(10):
            g.step(["feasible"], _ACTION, _DUMMY_ORACLE, _always_ok)
        for entry in g.log:
            assert 0.0 <= entry["delta_phi"] <= 1.0
