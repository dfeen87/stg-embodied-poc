"""tests/test_governor.py — Unit tests for the Spiral-Time Governor."""

from __future__ import annotations

import numpy as np
import pytest

from governor.spiral_time_governor import (
    GovernorState,
    SafetyFilterReport,
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

    def test_punctuation_handled_correctly(self):
        assert _verify_claim("terrain is flat.", {"terrain_class": 0}) is True
        assert _verify_claim("terrain is flat.", {"terrain_class": 1}) is False
        assert _verify_claim("feasible, stable!", {"feasible": True}) is True
        assert _verify_claim("feasible, stable!", {"feasible": False}) is False


# ---------------------------------------------------------------------------
# SpiralTimeGovernor — construction and reset
# ---------------------------------------------------------------------------

class TestGovernorInit:
    def test_invalid_ablation_raises(self):
        with pytest.raises(ValueError):
            SpiralTimeGovernor(ablation="invalid")

    def test_valid_ablations(self):
        for abl in ("none", "no_delta", "always_execute", "remove_I", "remove_C"):
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

    def test_remove_I_zeroes_delta_I_contribution(self):
        """remove_I ablation: ΔI is excluded from ΔΦ computation."""
        # Use an oracle that causes hallucinations (delta_I > 0)
        oracle = dict(_DUMMY_ORACLE, feasible=False, n_contacts=0, torso_upright=0.1)
        claims = ["feasible", "grounded", "upright"]  # all will fail → delta_I = 1.0

        g_full = SpiralTimeGovernor(ablation="none")
        _, _, info_full = g_full.step(claims, _ACTION, oracle, _always_ok)

        g_ri = SpiralTimeGovernor(ablation="remove_I")
        _, _, info_ri = g_ri.step(claims, _ACTION, oracle, _always_ok)

        # delta_I is still recorded (for logging), but excluded from delta_phi
        assert info_ri["delta_I"] > 0.0, "delta_I should still be computed for logging"
        # Without the beta*delta_I term, delta_phi must be lower than full governor
        assert info_ri["delta_phi"] < info_full["delta_phi"]

    def test_remove_C_zeroes_delta_C_contribution(self):
        """remove_C ablation: ΔC is excluded from ΔΦ computation."""
        oracle = dict(_DUMMY_ORACLE, feasible=True, torso_upright=0.9)
        claims = ["feasible"]

        # Run several steps to build up coherence history (so delta_C becomes non-zero)
        g_full = SpiralTimeGovernor(ablation="none")
        g_rc = SpiralTimeGovernor(ablation="remove_C")
        oracle2 = dict(_DUMMY_ORACLE, feasible=False, n_contacts=0, torso_upright=0.1)
        for _ in range(3):
            g_full.step(claims, _ACTION, oracle, _always_ok)
            g_rc.step(claims, _ACTION, oracle, _always_ok)
        # Trigger a coherence shift
        g_full.step(claims, _ACTION, oracle2, _always_ok)
        g_rc.step(claims, _ACTION, oracle2, _always_ok)

        # Find a step where delta_C is non-zero in the full governor
        delta_C_values = [e["delta_C"] for e in g_full.log]
        if any(dc > 0 for dc in delta_C_values):
            # At least one step has non-zero delta_C; verify remove_C differs from full
            for i, (e_full, e_rc) in enumerate(zip(g_full.log, g_rc.log)):
                if e_full["delta_C"] > 0:
                    assert e_rc["delta_phi"] < e_full["delta_phi"], (
                        f"step {i}: remove_C should give lower delta_phi when delta_C>0"
                    )
                    break

    def test_remove_I_valid_ablation(self):
        g = SpiralTimeGovernor(ablation="remove_I")
        assert g.ablation == "remove_I"

    def test_remove_C_valid_ablation(self):
        g = SpiralTimeGovernor(ablation="remove_C")
        assert g.ablation == "remove_C"

    def test_invalid_ablation_still_raises(self):
        with pytest.raises(ValueError, match="remove_R"):
            SpiralTimeGovernor(ablation="remove_R")

    def test_remove_I_delta_phi_in_range(self):
        g = SpiralTimeGovernor(ablation="remove_I")
        oracle = dict(_DUMMY_ORACLE, feasible=False, n_contacts=0, torso_upright=0.1)
        for _ in range(10):
            g.step(["feasible"], _ACTION, oracle, _always_ok)
        for entry in g.log:
            assert 0.0 <= entry["delta_phi"] <= 1.0

    def test_remove_C_delta_phi_in_range(self):
        g = SpiralTimeGovernor(ablation="remove_C")
        for _ in range(10):
            g.step(["feasible"], _ACTION, _DUMMY_ORACLE, _always_ok)
        for entry in g.log:
            assert 0.0 <= entry["delta_phi"] <= 1.0


# ---------------------------------------------------------------------------
# SafetyFilterReport — log fields
# ---------------------------------------------------------------------------

class TestSafetyFilterReport:
    """Tests for the structured Safety Filter Report fields added to each log entry."""

    # Required keys in every log entry (extends existing keys)
    _REPORT_KEYS = (
        "intervention_flag",
        "reason_codes",
        "constraint_margins",
        "clamped_action_delta",
    )
    _MARGIN_KEYS = ("joint_limit", "torque", "orientation", "contact_impulse")

    def test_log_entry_has_safety_report_keys(self):
        """Every log entry must contain all four new Safety Filter Report fields."""
        g = SpiralTimeGovernor()
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        entry = g.log[0]
        for key in self._REPORT_KEYS:
            assert key in entry, f"Missing Safety Filter Report key: {key}"

    def test_constraint_margins_has_all_keys(self):
        """constraint_margins must have exactly the four required sub-keys."""
        g = SpiralTimeGovernor()
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        margins = g.log[0]["constraint_margins"]
        for key in self._MARGIN_KEYS:
            assert key in margins, f"Missing constraint_margins key: {key}"

    def test_constraint_margins_are_floats(self):
        """All constraint margin values must be Python floats."""
        g = SpiralTimeGovernor()
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        for k, v in g.log[0]["constraint_margins"].items():
            assert isinstance(v, float), f"constraint_margins[{k!r}] is not float: {type(v)}"

    def test_clamped_action_delta_is_list(self):
        """clamped_action_delta must be a plain list (JSON-serialisable)."""
        g = SpiralTimeGovernor()
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        delta = g.log[0]["clamped_action_delta"]
        assert isinstance(delta, list), f"Expected list, got {type(delta)}"

    def test_intervention_flag_false_on_passthrough(self):
        """No intervention when EXECUTE mode and constraint passes."""
        g = SpiralTimeGovernor(ablation="always_execute")
        action = np.ones(12) * 0.1
        g.step([], action, _DUMMY_ORACLE, _always_ok)
        assert g.log[0]["intervention_flag"] is False

    def test_intervention_flag_true_on_safe_mode(self):
        """Governor must set intervention_flag=True when it uses the zero fallback."""
        g = SpiralTimeGovernor(ablation="none")
        oracle = dict(
            _DUMMY_ORACLE, feasible=False, n_contacts=0,
            torso_upright=0.1, terrain_class=0,
        )
        claims = ["feasible"] * 10
        action = np.ones(12) * 5.0
        mode, gated, _ = g.step(claims, action, oracle, _always_fail)
        if mode == "SAFE":
            assert g.log[0]["intervention_flag"] is True

    def test_intervention_flag_true_on_constraint_fail_execute(self):
        """Intervention when constraint fails even in EXECUTE mode (always_execute ablation)."""
        g = SpiralTimeGovernor(ablation="always_execute")
        action = np.ones(12) * 5.0
        g.step([], action, _DUMMY_ORACLE, _always_fail)
        assert g.log[0]["intervention_flag"] is True

    def test_clamped_action_delta_zero_on_passthrough(self):
        """Delta is zero vector when no intervention occurs."""
        g = SpiralTimeGovernor(ablation="always_execute")
        action = np.ones(12) * 0.1
        g.step([], action, _DUMMY_ORACLE, _always_ok)
        delta = g.log[0]["clamped_action_delta"]
        np.testing.assert_array_equal(delta, [0.0] * 12)

    def test_clamped_action_delta_equals_neg_action_on_safe(self):
        """Delta equals -proposed_action when gated to zeros."""
        g = SpiralTimeGovernor(ablation="always_execute")
        action = np.ones(12) * 5.0
        g.step([], action, _DUMMY_ORACLE, _always_fail)
        delta = g.log[0]["clamped_action_delta"]
        np.testing.assert_allclose(delta, (-action).tolist())

    def test_reason_codes_empty_on_clean_execute(self):
        """No reason codes when everything is safe in EXECUTE mode."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, torso_upright=0.9, n_contacts=4)
        g.step([], _ACTION, oracle, _always_ok)
        assert g.log[0]["reason_codes"] == []

    def test_reason_codes_contain_high_instability_on_safe(self):
        """HIGH_INSTABILITY must appear in reason_codes when mode is SAFE."""
        g = SpiralTimeGovernor(ablation="none")
        oracle = dict(
            _DUMMY_ORACLE, feasible=False, n_contacts=0,
            torso_upright=0.1, terrain_class=0,
        )
        claims = ["feasible"] * 10
        mode, _, _ = g.step(claims, _ACTION * 0 + 100.0, oracle, _always_fail)
        if mode == "SAFE":
            assert "HIGH_INSTABILITY" in g.log[0]["reason_codes"]

    def test_reason_codes_contain_constraint_violation(self):
        """CONSTRAINT_VIOLATION must appear when constraint_checker returns False."""
        g = SpiralTimeGovernor(ablation="always_execute")
        g.step([], _ACTION, _DUMMY_ORACLE, _always_fail)
        assert "CONSTRAINT_VIOLATION" in g.log[0]["reason_codes"]

    def test_reason_codes_contain_norm_exceeded(self):
        """NORM_EXCEEDED must appear when action norm ≥ 10.0."""
        g = SpiralTimeGovernor(ablation="always_execute")
        big_action = np.ones(12) * 5.0  # norm ≈ 17.3 > 10.0
        # Constraint passes but norm gate fires
        g.step([], big_action, _DUMMY_ORACLE, _always_ok)
        assert "NORM_EXCEEDED" in g.log[0]["reason_codes"]

    def test_reason_codes_contain_low_orientation(self):
        """LOW_ORIENTATION must appear when torso_upright ≤ 0.5."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, torso_upright=0.3)
        g.step([], _ACTION, oracle, _always_ok)
        assert "LOW_ORIENTATION" in g.log[0]["reason_codes"]

    def test_reason_codes_contain_insufficient_contacts(self):
        """INSUFFICIENT_CONTACTS must appear when n_contacts < 2."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, n_contacts=1)
        g.step([], _ACTION, oracle, _always_ok)
        assert "INSUFFICIENT_CONTACTS" in g.log[0]["reason_codes"]

    def test_joint_limit_margin_positive_for_small_action(self):
        """joint_limit margin should be positive when action norm < 10.0."""
        g = SpiralTimeGovernor(ablation="always_execute")
        action = np.zeros(12)
        g.step([], action, _DUMMY_ORACLE, _always_ok)
        assert g.log[0]["constraint_margins"]["joint_limit"] > 0.0

    def test_joint_limit_margin_negative_for_large_action(self):
        """joint_limit margin should be negative when action norm ≥ 10.0."""
        g = SpiralTimeGovernor(ablation="always_execute")
        action = np.ones(12) * 5.0  # norm ≈ 17.3 > 10.0
        g.step([], action, _DUMMY_ORACLE, _always_ok)
        assert g.log[0]["constraint_margins"]["joint_limit"] < 0.0

    def test_torque_margin_positive_when_constraint_passes(self):
        """torque margin is +1.0 when external constraint passes."""
        g = SpiralTimeGovernor(ablation="always_execute")
        g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        assert g.log[0]["constraint_margins"]["torque"] == 1.0

    def test_torque_margin_negative_when_constraint_fails(self):
        """torque margin is -1.0 when external constraint fails."""
        g = SpiralTimeGovernor(ablation="always_execute")
        g.step([], _ACTION, _DUMMY_ORACLE, _always_fail)
        assert g.log[0]["constraint_margins"]["torque"] == -1.0

    def test_orientation_margin_positive_when_upright(self):
        """orientation margin is positive when torso_upright > 0.5."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, torso_upright=0.9)
        g.step([], _ACTION, oracle, _always_ok)
        assert g.log[0]["constraint_margins"]["orientation"] > 0.0

    def test_orientation_margin_negative_when_fallen(self):
        """orientation margin is negative when torso_upright ≤ 0.5."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, torso_upright=0.2)
        g.step([], _ACTION, oracle, _always_ok)
        assert g.log[0]["constraint_margins"]["orientation"] < 0.0

    def test_contact_impulse_margin_positive_with_enough_contacts(self):
        """contact_impulse margin is positive when n_contacts >= 2."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, n_contacts=4)
        g.step([], _ACTION, oracle, _always_ok)
        assert g.log[0]["constraint_margins"]["contact_impulse"] > 0.0

    def test_contact_impulse_margin_negative_with_few_contacts(self):
        """contact_impulse margin is negative when n_contacts < 2."""
        g = SpiralTimeGovernor(ablation="always_execute")
        oracle = dict(_DUMMY_ORACLE, n_contacts=1)
        g.step([], _ACTION, oracle, _always_ok)
        assert g.log[0]["constraint_margins"]["contact_impulse"] < 0.0

    def test_report_fields_logged_every_step(self):
        """All Safety Filter Report fields must appear in every log entry."""
        g = SpiralTimeGovernor()
        for _ in range(5):
            g.step([], _ACTION, _DUMMY_ORACLE, _always_ok)
        for i, entry in enumerate(g.log):
            for key in self._REPORT_KEYS:
                assert key in entry, f"Step {i}: missing key {key!r}"

    def test_deterministic_reason_codes(self):
        """Same inputs must produce identical reason_codes across two runs."""
        oracle = dict(_DUMMY_ORACLE, torso_upright=0.3, n_contacts=1)
        g1 = SpiralTimeGovernor()
        g2 = SpiralTimeGovernor()
        g1.step([], _ACTION, oracle, _always_fail)
        g2.step([], _ACTION, oracle, _always_fail)
        assert g1.log[0]["reason_codes"] == g2.log[0]["reason_codes"]

    def test_deterministic_constraint_margins(self):
        """Same inputs must produce identical constraint_margins across two runs."""
        oracle = dict(_DUMMY_ORACLE, torso_upright=0.7, n_contacts=3)
        action = np.ones(12) * 0.5
        g1 = SpiralTimeGovernor()
        g2 = SpiralTimeGovernor()
        g1.step([], action, oracle, _always_ok)
        g2.step([], action, oracle, _always_ok)
        assert g1.log[0]["constraint_margins"] == g2.log[0]["constraint_margins"]

    def test_safety_filter_report_dataclass_importable(self):
        """SafetyFilterReport must be importable from the governor module."""
        assert SafetyFilterReport is not None
