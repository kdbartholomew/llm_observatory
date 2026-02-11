"""Shared test fixtures for SDK tests."""

import pytest

from llm_observatory.client import ObservatoryClient
from llm_observatory.types import LLMMetric


@pytest.fixture
def mock_client():
    """Create a client that doesn't actually start a background thread."""
    client = ObservatoryClient(
        endpoint="http://localhost:8000",
        api_key="test-key",
        batch_size=5,
        flush_interval=1.0,
        project="test-project",
    )
    return client


@pytest.fixture
def sample_metric():
    """A sample LLMMetric for testing."""
    return LLMMetric(
        model="gpt-5-mini",
        tokens_in=100,
        tokens_out=50,
        latency_ms=250.0,
        cost=0.00003,
    )
