import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def dummy_users_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": 1,
                "name": "Target",
                "age": 25,
                "location": "Osaka",
                "instrument": "guitar",
                "genres": "rock,pop",
                "skill_level": "intermediate",
                "experience_years": 3,
                "purpose": "cover_band",
                "frequency": "weekly",
                "available_days": "saturday,sunday",
                "communication_style": "chat_mainly",
                "bio": "大阪でロックやポップを演奏したいギターです。",
            },
            {
                "user_id": 2,
                "name": "GoodMatch",
                "age": 27,
                "location": "Osaka",
                "instrument": "bass",
                "genres": "rock,pop",
                "skill_level": "intermediate",
                "experience_years": 4,
                "purpose": "cover_band",
                "frequency": "weekly",
                "available_days": "saturday",
                "communication_style": "chat_mainly",
                "bio": "大阪でロックバンドをやりたいベースです。",
            },
            {
                "user_id": 3,
                "name": "DifferentLocation",
                "age": 24,
                "location": "Tokyo",
                "instrument": "vocal",
                "genres": "rock,pop",
                "skill_level": "intermediate",
                "experience_years": 2,
                "purpose": "cover_band",
                "frequency": "weekly",
                "available_days": "saturday,sunday",
                "communication_style": "chat_mainly",
                "bio": "東京でロックやポップを歌いたいです。",
            },
        ]
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, dummy_users_df: pd.DataFrame) -> TestClient:
    monkeypatch.setattr(main, "users", dummy_users_df)
    return TestClient(main.app)


def test_get_recommendations_returns_200(client: TestClient):
    response = client.get("/users/1/recommendations")

    assert response.status_code == 200


def test_get_recommendations_returns_list_response(client: TestClient):
    response = client.get("/users/1/recommendations")

    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0


def test_get_recommendations_returns_expected_fields(client: TestClient):
    response = client.get("/users/1/recommendations")

    item = response.json()[0]

    assert item.keys() >= {
        "user_id",
        "name",
        "age",
        "location",
        "instrument",
        "genres",
        "skill_level",
        "purpose",
        "frequency",
        "available_days",
        "score",
        "reasons",
    }


def test_get_recommendations_respects_top_k(client: TestClient):
    response = client.get("/users/1/recommendations?top_k=1")

    data = response.json()

    assert response.status_code == 200
    assert len(data) == 1


def test_get_recommendations_filters_same_location_by_default(client: TestClient):
    response = client.get("/users/1/recommendations")

    data = response.json()

    assert response.status_code == 200
    assert all(item["location"] == "Osaka" for item in data)


def test_get_recommendations_can_include_different_location_when_filter_is_disabled(
    client: TestClient,
):
    response = client.get("/users/1/recommendations?same_location_only=false")

    data = response.json()
    user_ids = [item["user_id"] for item in data]

    assert response.status_code == 200
    assert 3 in user_ids


def test_get_recommendations_returns_404_when_user_does_not_exist(client: TestClient):
    response = client.get("/users/999/recommendations")

    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}


def test_get_recommendations_returns_422_when_top_k_is_too_small(client: TestClient):
    response = client.get("/users/1/recommendations?top_k=0")

    assert response.status_code == 422


def test_get_recommendations_returns_422_when_top_k_is_too_large(client: TestClient):
    response = client.get("/users/1/recommendations?top_k=21")

    assert response.status_code == 422
