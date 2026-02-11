"""Tests for the ObservatoryClient batching and queue logic."""

import pytest
import threading
import time
from unittest.mock import AsyncMock, patch, MagicMock

from llm_observatory.client import ObservatoryClient, get_client, set_client
from llm_observatory.types import LLMMetric


class TestObservatoryClient:
    """Tests for ObservatoryClient."""

    def test_record_adds_to_queue(self, mock_client):
        """record() should add metrics to the internal queue."""
        metric = LLMMetric(
            model="gpt-5", tokens_in=10, tokens_out=20,
            latency_ms=100, cost=0.001,
        )
        # Don't start the background thread for this test
        mock_client._started = True  # Pretend started to skip auto-start
        mock_client.record(metric)
        assert mock_client._queue.qsize() == 1

    def test_record_sets_project(self, mock_client):
        """record() should set project from client config if not on metric."""
        metric = LLMMetric(
            model="gpt-5", tokens_in=10, tokens_out=20,
            latency_ms=100, cost=0.001,
        )
        assert metric.project is None
        mock_client._started = True
        mock_client.record(metric)
        assert metric.project == "test-project"

    def test_record_preserves_metric_project(self, mock_client):
        """record() should not override metric's own project."""
        metric = LLMMetric(
            model="gpt-5", tokens_in=10, tokens_out=20,
            latency_ms=100, cost=0.001, project="custom-project",
        )
        mock_client._started = True
        mock_client.record(metric)
        assert metric.project == "custom-project"

    def test_shutdown_without_start(self, mock_client):
        """shutdown() should be safe to call without start()."""
        mock_client.shutdown()  # Should not raise

    def test_start_is_idempotent(self, mock_client):
        """Multiple start() calls should not create multiple threads."""
        mock_client.start()
        thread1 = mock_client._thread
        mock_client.start()
        thread2 = mock_client._thread
        assert thread1 is thread2
        mock_client.shutdown()

    def test_queue_multiple_metrics(self, mock_client):
        """Multiple metrics should all be queued."""
        mock_client._started = True
        for i in range(10):
            metric = LLMMetric(
                model=f"model-{i}", tokens_in=i, tokens_out=i,
                latency_ms=float(i), cost=0.0,
            )
            mock_client.record(metric)
        assert mock_client._queue.qsize() == 10


class TestGlobalClient:
    """Tests for get_client/set_client."""

    def test_set_and_get_client(self):
        """set_client/get_client should work as a pair."""
        client = ObservatoryClient(
            endpoint="http://test:8000",
            api_key="key",
        )
        set_client(client)
        assert get_client() is client
        # Cleanup
        set_client(None)

    def test_default_client_is_none(self):
        """Default global client should be None."""
        set_client(None)
        assert get_client() is None

    def test_thread_safety(self):
        """set_client/get_client should be thread-safe."""
        results = []

        def setter():
            client = ObservatoryClient(endpoint="http://test:8000", api_key="key")
            set_client(client)
            results.append(get_client())

        threads = [threading.Thread(target=setter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be non-None ObservatoryClient instances
        assert all(isinstance(r, ObservatoryClient) for r in results)
        # Cleanup
        set_client(None)
