"""Tests for API authentication."""

import os
import pytest

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OBSERVATORY_API_KEY", "test-api-key")

from fastapi.testclient import TestClient
from main import app, API_KEY


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestVerifyApiKey:
    """Tests for the verify_api_key dependency."""

    def test_missing_authorization_header(self, client):
        """Request without Authorization header should return 401."""
        response = client.get("/projects")
        assert response.status_code == 401
        assert "Missing" in response.json()["detail"]

    def test_invalid_format_no_bearer(self, client):
        """Authorization without 'Bearer' prefix should return 401."""
        response = client.get("/projects", headers={"Authorization": "Token abc123"})
        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]

    def test_invalid_format_no_space(self, client):
        """Authorization with no space should return 401."""
        response = client.get("/projects", headers={"Authorization": "Bearertoken"})
        assert response.status_code == 401

    def test_wrong_api_key(self, client):
        """Incorrect API key should return 403."""
        response = client.get("/projects", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_correct_api_key(self, client):
        """Correct API key should pass authentication (may fail on DB)."""
        response = client.get("/projects", headers={"Authorization": f"Bearer {API_KEY}"})
        # Should not be 401 or 403 — DB errors are 500
        assert response.status_code not in (401, 403)
