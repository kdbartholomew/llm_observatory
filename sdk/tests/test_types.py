"""Tests for LLM Observatory type definitions and cost calculation."""

import pytest
from datetime import datetime, timezone

from llm_observatory.types import LLMMetric, calculate_cost, MODEL_PRICING


class TestCalculateCost:
    """Tests for the calculate_cost function."""

    def test_exact_match_openai(self):
        """Cost calculation with exact model name match."""
        # gpt-5-mini: (0.10, 0.40) per 1M tokens
        cost = calculate_cost("gpt-5-mini", tokens_in=1_000_000, tokens_out=1_000_000)
        assert cost == pytest.approx(0.50, abs=1e-6)  # 0.10 + 0.40

    def test_exact_match_anthropic(self):
        """Cost calculation for Anthropic model."""
        # claude-opus-4-6: (5.00, 25.00) per 1M tokens
        cost = calculate_cost("claude-opus-4-6", tokens_in=1000, tokens_out=500)
        expected = (1000 * 5.00 / 1_000_000) + (500 * 25.00 / 1_000_000)
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_exact_match_gemini(self):
        """Cost calculation for Gemini model."""
        # gemini-2.5-pro: (1.25, 10.00)
        cost = calculate_cost("gemini-2.5-pro", tokens_in=2000, tokens_out=1000)
        expected = (2000 * 1.25 / 1_000_000) + (1000 * 10.00 / 1_000_000)
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_prefix_match_versioned_model(self):
        """Versioned model names should match via prefix."""
        # gpt-5-mini-2025-08-07 should match gpt-5-mini pricing
        cost = calculate_cost("gpt-5-mini-2025-08-07", tokens_in=1_000_000, tokens_out=0)
        assert cost == pytest.approx(0.10, abs=1e-6)

    def test_unknown_model_returns_zero(self):
        """Unknown models should return 0 cost."""
        cost = calculate_cost("unknown-model-xyz", tokens_in=1000, tokens_out=1000)
        assert cost == 0.0

    def test_zero_tokens(self):
        """Zero tokens should return zero cost."""
        cost = calculate_cost("gpt-5", tokens_in=0, tokens_out=0)
        assert cost == 0.0

    def test_case_insensitive_prefix_match(self):
        """Prefix matching should be case-insensitive."""
        cost = calculate_cost("GPT-5-mini", tokens_in=1_000_000, tokens_out=0)
        assert cost == pytest.approx(0.10, abs=1e-6)

    def test_all_models_have_valid_pricing(self):
        """Every model in the pricing dict should have positive prices."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            assert input_price >= 0, f"{model} has negative input price"
            assert output_price >= 0, f"{model} has negative output price"


class TestLLMMetric:
    """Tests for the LLMMetric dataclass."""

    def test_default_timestamp_is_utc(self):
        """Default timestamp should be timezone-aware UTC."""
        metric = LLMMetric(
            model="gpt-5",
            tokens_in=10,
            tokens_out=20,
            latency_ms=100.0,
            cost=0.001,
        )
        assert metric.timestamp.tzinfo is not None

    def test_to_dict(self):
        """to_dict should serialize all fields correctly."""
        metric = LLMMetric(
            model="gpt-5",
            tokens_in=100,
            tokens_out=200,
            latency_ms=150.5,
            cost=0.001,
            endpoint="chat",
            project="test",
        )
        d = metric.to_dict()
        assert d["model"] == "gpt-5"
        assert d["tokens_in"] == 100
        assert d["tokens_out"] == 200
        assert d["latency_ms"] == 150.5
        assert d["cost"] == 0.001
        assert d["endpoint"] == "chat"
        assert d["project"] == "test"
        assert d["error"] is None
        # timestamp should be ISO format string
        assert isinstance(d["timestamp"], str)

    def test_to_dict_with_error(self):
        """to_dict should include error when set."""
        metric = LLMMetric(
            model="gpt-5",
            tokens_in=0,
            tokens_out=0,
            latency_ms=50.0,
            cost=0.0,
            error="RateLimitError: too many requests",
        )
        d = metric.to_dict()
        assert d["error"] == "RateLimitError: too many requests"
