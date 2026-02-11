"""Tests for metrics ingestion and query endpoints."""

import os
import pytest
from unittest.mock import MagicMock

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OBSERVATORY_API_KEY", "test-api-key")

from fastapi.testclient import TestClient
from main import app, API_KEY, get_db


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


class TestIngestMetrics:
    """Tests for POST /metrics."""

    def test_empty_batch(self, client, auth_headers):
        """Empty batch should return inserted: 0."""
        response = client.post(
            "/metrics",
            json={"metrics": []},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["inserted"] == 0

    def test_valid_batch(self, client, auth_headers, mock_db):
        """Valid metrics batch should be inserted."""
        mock_db.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": "1"}, {"id": "2"}]
        )

        response = client.post(
            "/metrics",
            json={
                "metrics": [
                    {
                        "model": "gpt-5-mini",
                        "tokens_in": 100,
                        "tokens_out": 50,
                        "latency_ms": 200.0,
                        "cost": 0.00003,
                        "timestamp": "2026-01-15T10:00:00Z",
                    },
                    {
                        "model": "claude-sonnet-4-5-20250929",
                        "tokens_in": 200,
                        "tokens_out": 100,
                        "latency_ms": 500.0,
                        "cost": 0.002,
                        "timestamp": "2026-01-15T10:01:00Z",
                    },
                ]
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_negative_tokens_rejected(self, client, auth_headers):
        """Negative token counts should be rejected by validation."""
        response = client.post(
            "/metrics",
            json={
                "metrics": [
                    {
                        "model": "gpt-5",
                        "tokens_in": -1,
                        "tokens_out": 50,
                        "latency_ms": 100.0,
                        "cost": 0.001,
                        "timestamp": "2026-01-15T10:00:00Z",
                    }
                ]
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_empty_model_rejected(self, client, auth_headers):
        """Empty model name should be rejected by validation."""
        response = client.post(
            "/metrics",
            json={
                "metrics": [
                    {
                        "model": "",
                        "tokens_in": 100,
                        "tokens_out": 50,
                        "latency_ms": 100.0,
                        "cost": 0.001,
                        "timestamp": "2026-01-15T10:00:00Z",
                    }
                ]
            },
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestGetMetrics:
    """Tests for GET /metrics."""

    def test_get_metrics_with_filters(self, client, auth_headers, mock_db):
        """GET /metrics should accept filter params."""
        mock_result = MagicMock()
        mock_result.data = []
        # Chain: table().select().order().limit().eq().execute()
        mock_chain = MagicMock()
        mock_chain.execute.return_value = mock_result
        mock_chain.order.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        mock_chain.gte.return_value = mock_chain
        mock_chain.lte.return_value = mock_chain
        mock_chain.eq.return_value = mock_chain
        mock_db.table.return_value.select.return_value = mock_chain

        response = client.get(
            "/metrics?model=gpt-5&limit=10",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "total" in data
