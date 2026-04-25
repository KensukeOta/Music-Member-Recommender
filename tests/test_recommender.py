import pandas as pd
import pytest

from app.recommender import recommend_users


@pytest.fixture
def users_df() -> pd.DataFrame:
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
                "name": "WeakMatch",
                "age": 30,
                "location": "Osaka",
                "instrument": "drums",
                "genres": "jazz",
                "skill_level": "advanced",
                "experience_years": 10,
                "purpose": "professional",
                "frequency": "monthly",
                "available_days": "weekday_night",
                "communication_style": "voice_call_ok",
                "bio": "ジャズを中心に活動しています。",
            },
            {
                "user_id": 4,
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


def test_recommend_users_returns_top_k_users(users_df: pd.DataFrame):
    recommendations = recommend_users(
        user_id=1,
        users=users_df,
        top_k=1,
        same_location_only=True,
    )

    assert len(recommendations) == 1
    assert recommendations.iloc[0]["user_id"] == 2


def test_recommend_users_excludes_target_user(users_df: pd.DataFrame):
    recommendations = recommend_users(
        user_id=1,
        users=users_df,
        top_k=10,
        same_location_only=False,
    )

    recommended_user_ids = recommendations["user_id"].tolist()

    assert 1 not in recommended_user_ids


def test_recommend_users_filters_by_same_location(users_df: pd.DataFrame):
    recommendations = recommend_users(
        user_id=1,
        users=users_df,
        top_k=10,
        same_location_only=True,
    )

    recommended_locations = recommendations["location"].unique().tolist()

    assert recommended_locations == ["Osaka"]


def test_recommend_users_can_include_different_location_when_filter_is_disabled(
    users_df: pd.DataFrame,
):
    recommendations = recommend_users(
        user_id=1,
        users=users_df,
        top_k=10,
        same_location_only=False,
    )

    recommended_user_ids = recommendations["user_id"].tolist()

    assert 4 in recommended_user_ids


def test_recommend_users_returns_recommendation_reasons(users_df: pd.DataFrame):
    recommendations = recommend_users(
        user_id=1,
        users=users_df,
        top_k=1,
        same_location_only=True,
    )

    reasons = recommendations.iloc[0]["reasons"]

    assert "活動頻度が一致" in reasons
    assert "活動目的が一致" in reasons
    assert "スキルレベルが近い" in reasons
    assert "活動地域が一致" in reasons
    assert any(reason.startswith("共通ジャンル:") for reason in reasons)


def test_recommend_users_raises_value_error_when_user_does_not_exist(
    users_df: pd.DataFrame,
):
    with pytest.raises(ValueError, match="user_id=999 not found"):
        recommend_users(
            user_id=999,
            users=users_df,
            top_k=5,
            same_location_only=True,
        )


def test_recommend_users_returns_empty_dataframe_when_no_candidate_exists(
    users_df: pd.DataFrame,
):
    only_target_df = users_df[users_df["user_id"] == 1]

    recommendations = recommend_users(
        user_id=1,
        users=only_target_df,
        top_k=5,
        same_location_only=True,
    )

    assert recommendations.empty
