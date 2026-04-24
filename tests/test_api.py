#tests/test_api.py

"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_endpoint(client):
    response = client.post(
        "/recommendations",
        json={"user_id": 1, "n": 10}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 10


def test_recommendations_invalid_user(client):
    response = client.post(
        "/recommendations",
        json={"user_id": 99999, "n": 10}
    )
    assert response.status_code == 404


def test_recommendations_correct_fields(client):
    response = client.post(
        "/recommendations",
        json={"user_id": 1, "n": 5}
    )
    assert response.status_code == 200
    data = response.json()
    first = data["recommendations"][0]
    assert "movie_id" in first
    assert "title" in first
    assert "predicted_score" in first