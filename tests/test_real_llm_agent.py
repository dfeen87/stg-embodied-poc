"""tests/test_real_llm_agent.py — Unit tests for the Real LLM agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_mock.real_llm_agent import RealLLMAgent, _parse_claims, query_real_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_OBS = np.zeros(32, dtype=np.float64)

_FAKE_RESPONSE = (
    "The robot is stable on flat terrain.\n"
    "All four feet are in contact with the ground.\n"
    "Torso orientation is upright."
)


# ---------------------------------------------------------------------------
# _parse_claims
# ---------------------------------------------------------------------------

class TestParseClaims:
    def test_plain_lines(self):
        raw = "Claim one.\nClaim two.\nClaim three."
        result = _parse_claims(raw, 3)
        assert result == ["Claim one.", "Claim two.", "Claim three."]

    def test_numbered_markers_stripped(self):
        raw = "1. First claim.\n2. Second claim.\n3. Third claim."
        result = _parse_claims(raw, 3)
        assert result == ["First claim.", "Second claim.", "Third claim."]

    def test_dash_bullets_stripped(self):
        raw = "- Claim A\n- Claim B\n- Claim C"
        result = _parse_claims(raw, 3)
        assert result == ["Claim A", "Claim B", "Claim C"]

    def test_star_bullets_stripped(self):
        raw = "* Claim X\n* Claim Y"
        result = _parse_claims(raw, 2)
        assert result == ["Claim X", "Claim Y"]

    def test_truncates_to_n_claims(self):
        raw = "A\nB\nC\nD\nE"
        result = _parse_claims(raw, 2)
        assert len(result) == 2

    def test_pads_when_too_few_lines(self):
        raw = "Only one claim."
        result = _parse_claims(raw, 3)
        assert len(result) == 3
        assert result[1] == result[2] == "Only one claim."

    def test_empty_response_gives_fallback(self):
        result = _parse_claims("", 2)
        assert len(result) == 2
        assert result[0] == "State uncertain."

    def test_blank_lines_ignored(self):
        raw = "\nClaim one.\n\nClaim two.\n\nClaim three.\n"
        result = _parse_claims(raw, 3)
        assert result == ["Claim one.", "Claim two.", "Claim three."]


# ---------------------------------------------------------------------------
# query_real_llm
# ---------------------------------------------------------------------------

class TestQueryRealLlm:
    def test_raises_import_error_if_openai_missing(self, monkeypatch):
        """query_real_llm raises ImportError when openai is not installed."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="openai"):
            query_real_llm("test prompt")

    def test_raises_runtime_error_if_no_api_key(self, monkeypatch):
        """query_real_llm raises RuntimeError when OPENAI_API_KEY is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": fake_openai}):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                query_real_llm("test prompt")

    def test_returns_model_response(self, monkeypatch):
        """query_real_llm returns the text from the API response."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        fake_message = MagicMock()
        fake_message.content = "Mocked claim."
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_completion = MagicMock()
        fake_completion.choices = [fake_choice]

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_completion

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = query_real_llm("test prompt", model="gpt-4")

        assert result == "Mocked claim."
        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# RealLLMAgent
# ---------------------------------------------------------------------------

class TestRealLLMAgentInit:
    def test_default_construction(self):
        agent = RealLLMAgent(seed=0)
        assert agent._action_dim == 12
        assert agent._seed == 0

    def test_custom_action_dim(self):
        agent = RealLLMAgent(seed=1, action_dim=8)
        assert agent._action_dim == 8


class TestRealLLMAgentPropose:
    """Tests for ``RealLLMAgent.propose`` using a mocked ``query_real_llm``."""

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_returns_action_and_claims(self, _mock_llm):
        agent = RealLLMAgent(seed=42, action_dim=12)
        action, claims = agent.propose(_DUMMY_OBS, t=0, n_claims=3)
        assert isinstance(action, np.ndarray)
        assert action.shape == (12,)
        assert isinstance(claims, list)
        assert len(claims) == 3

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_action_dtype_float64(self, _mock_llm):
        agent = RealLLMAgent(seed=42, action_dim=12)
        action, _ = agent.propose(_DUMMY_OBS, t=5)
        assert action.dtype == np.float64

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_action_is_deterministic_for_same_seed_and_t(self, _mock_llm):
        """Same seed and timestep should yield the same action."""
        for t in (0, 10, 50):
            a1, _ = RealLLMAgent(seed=42, action_dim=12).propose(_DUMMY_OBS, t=t)
            a2, _ = RealLLMAgent(seed=42, action_dim=12).propose(_DUMMY_OBS, t=t)
            np.testing.assert_array_equal(a1, a2)

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_claims_have_correct_count(self, _mock_llm):
        for n in (1, 2, 5):
            agent = RealLLMAgent(seed=42, action_dim=12)
            _, claims = agent.propose(_DUMMY_OBS, t=0, n_claims=n)
            assert len(claims) == n

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_bad_action_norm_possible(self, _mock_llm):
        """Some timesteps produce actions with norm > 8 (bad action injection)."""
        agent = RealLLMAgent(seed=0, action_dim=12)
        norms = [
            float(np.linalg.norm(agent.propose(_DUMMY_OBS, t=t)[0]))
            for t in range(200)
        ]
        assert any(n > 8.0 for n in norms), "Expected at least one bad action"

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_most_actions_within_normal_range(self, _mock_llm):
        """The vast majority of actions should be in the normal gait range."""
        agent = RealLLMAgent(seed=7, action_dim=12)
        norms = [
            float(np.linalg.norm(agent.propose(_DUMMY_OBS, t=t)[0]))
            for t in range(200)
        ]
        normal = sum(1 for n in norms if n < 8.0)
        assert normal / len(norms) > 0.85

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value=_FAKE_RESPONSE)
    def test_model_forwarded_to_query(self, mock_llm):
        """RealLLMAgent passes its model name to query_real_llm."""
        agent = RealLLMAgent(seed=0, action_dim=12, model="gpt-4")
        agent.propose(_DUMMY_OBS, t=0, n_claims=3)
        _, kwargs = mock_llm.call_args
        assert kwargs.get("model") == "gpt-4"


class TestRealLLMAgentReset:
    @patch("llm_mock.real_llm_agent.query_real_llm", return_value="claim")
    def test_reset_restores_rng(self, _mock_llm):
        """After reset, the agent produces the same sequence as fresh init."""
        agent = RealLLMAgent(seed=3, action_dim=12)

        a1, _ = agent.propose(_DUMMY_OBS, t=0)
        agent.reset()
        a2, _ = agent.propose(_DUMMY_OBS, t=0)
        np.testing.assert_array_equal(a1, a2)

    @patch("llm_mock.real_llm_agent.query_real_llm", return_value="claim")
    def test_reset_with_new_seed(self, _mock_llm):
        """Resetting with a different seed changes the output."""
        agent = RealLLMAgent(seed=0, action_dim=12)

        a1, _ = agent.propose(_DUMMY_OBS, t=0)
        agent.reset(seed=99)
        a2, _ = agent.propose(_DUMMY_OBS, t=0)
        # Different seeds should (almost certainly) give different actions
        assert not np.array_equal(a1, a2)
