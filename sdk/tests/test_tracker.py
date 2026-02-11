"""Tests for the metric extraction logic in tracker.py."""

import pytest
from unittest.mock import MagicMock

from llm_observatory.tracker import _extract_metric


class MockOpenAIUsage:
    """Mock OpenAI usage object."""
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockOpenAIResponse:
    """Mock OpenAI ChatCompletion response."""
    def __init__(self, model="gpt-5-mini", prompt_tokens=100, completion_tokens=50):
        self.model = model
        self.usage = MockOpenAIUsage(prompt_tokens, completion_tokens)


class MockAnthropicUsage:
    """Mock Anthropic usage object."""
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicResponse:
    """Mock Anthropic Message response."""
    def __init__(self, model="claude-sonnet-4-5-20250929", input_tokens=100, output_tokens=50):
        self.model = model
        self.usage = MockAnthropicUsage(input_tokens, output_tokens)


class MockGeminiUsage:
    """Mock Gemini usage (via wrapper)."""
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockGeminiResponse:
    """Mock Gemini wrapped response (as returned by GeminiResponseWrapper)."""
    def __init__(self, model="gemini-2.5-flash", prompt_tokens=100, completion_tokens=50):
        self.model = model
        self.usage = MockGeminiUsage(prompt_tokens, completion_tokens)


class TestExtractMetric:
    """Tests for _extract_metric."""

    def test_openai_response(self):
        """Should extract tokens from OpenAI response format."""
        result = MockOpenAIResponse(model="gpt-5-mini", prompt_tokens=200, completion_tokens=100)
        metric = _extract_metric(result=result, latency_ms=150.0, error=None, endpoint_tag="chat")

        assert metric is not None
        assert metric.model == "gpt-5-mini"
        assert metric.tokens_in == 200
        assert metric.tokens_out == 100
        assert metric.latency_ms == 150.0
        assert metric.endpoint == "chat"

    def test_anthropic_response(self):
        """Should extract tokens from Anthropic response format."""
        result = MockAnthropicResponse(model="claude-sonnet-4-5-20250929", input_tokens=300, output_tokens=150)
        metric = _extract_metric(result=result, latency_ms=500.0, error=None, endpoint_tag=None)

        assert metric is not None
        assert metric.model == "claude-sonnet-4-5-20250929"
        assert metric.tokens_in == 300
        assert metric.tokens_out == 150

    def test_gemini_response(self):
        """Should extract tokens from Gemini wrapped response."""
        result = MockGeminiResponse(model="gemini-2.5-flash", prompt_tokens=150, completion_tokens=75)
        metric = _extract_metric(result=result, latency_ms=200.0, error=None, endpoint_tag=None)

        assert metric is not None
        assert metric.model == "gemini-2.5-flash"
        assert metric.tokens_in == 150
        assert metric.tokens_out == 75

    def test_error_without_result(self):
        """Should create metric with error info when result is None."""
        metric = _extract_metric(
            result=None,
            latency_ms=50.0,
            error="RateLimitError: too many requests",
            endpoint_tag="chat",
        )

        assert metric is not None
        assert metric.model == "unknown"
        assert metric.error == "RateLimitError: too many requests"
        assert metric.tokens_in == 0
        assert metric.tokens_out == 0

    def test_none_result_no_error_returns_none(self):
        """Should return None when both result and error are None."""
        metric = _extract_metric(result=None, latency_ms=0, error=None, endpoint_tag=None)
        assert metric is None

    def test_cost_is_calculated(self):
        """Metric should include calculated cost."""
        result = MockOpenAIResponse(model="gpt-5-mini", prompt_tokens=1000, completion_tokens=500)
        metric = _extract_metric(result=result, latency_ms=100.0, error=None, endpoint_tag=None)

        assert metric is not None
        assert metric.cost > 0
