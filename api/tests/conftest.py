"""Shared test fixtures for API tests."""

import os
import pytest
from unittest.mock import MagicMock

# Set required env vars before importing the app
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OBSERVATORY_API_KEY", "test-api-key")

from fastapi.testclient import TestClient
from main import app, API_KEY


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    """Valid authorization headers."""
    return {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def mock_db():
    """Mocked Supabase client."""
    db = MagicMock()
    return db
