from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RecommendWeights:
    genre: float = 0.35
    frequency: float = 0.20
    purpose: float = 0.20
    skill_level: float = 0.10
    available_days: float = 0.10
    bio: float = 0.05


def split_values(value: str) -> set[str]:
    if pd.isna(value):
        return set()

    return {item.strip() for item in str(value).split(",") if item.strip()}


def jaccard_similarity(a: str, b: str) -> float:
    set_a = split_values(a)
    set_b = split_values(b)

    if not set_a and not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    return len(intersection) / len(union)


def exact_match_score(a: str, b: str) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0

    return 1.0 if str(a) == str(b) else 0.0


def bio_similarity(target_bio: str, candidate_bios: pd.Series) -> list[float]:
    texts = [str(target_bio)] + candidate_bios.fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    return scores.tolist()


def build_recommendation_reasons(
    target_user: pd.Series,
    candidate: pd.Series,
) -> list[str]:
    reasons: list[str] = []

    common_genres = split_values(target_user["genres"]) & split_values(
        candidate["genres"]
    )
    if common_genres:
        reasons.append(f"共通ジャンル: {', '.join(sorted(common_genres))}")

    if target_user["frequency"] == candidate["frequency"]:
        reasons.append("活動頻度が一致")

    if target_user["purpose"] == candidate["purpose"]:
        reasons.append("活動目的が一致")

    if target_user["skill_level"] == candidate["skill_level"]:
        reasons.append("スキルレベルが近い")

    common_days = split_values(target_user["available_days"]) & split_values(
        candidate["available_days"]
    )
    if common_days:
        reasons.append(f"活動可能日が一致: {', '.join(sorted(common_days))}")

    if target_user["location"] == candidate["location"]:
        reasons.append("活動地域が一致")

    return reasons


def recommend_users(
    user_id: int,
    users: pd.DataFrame,
    top_k: int = 5,
    same_location_only: bool = True,
    weights: RecommendWeights = RecommendWeights(),
) -> pd.DataFrame:
    if user_id not in users["user_id"].values:
        raise ValueError(f"user_id={user_id} not found")

    target_user = users.loc[users["user_id"] == user_id].iloc[0]

    candidates = users[users["user_id"] != user_id].copy()

    # 地域は「似ている度」ではなく、まず会いやすさの条件としてフィルタする
    if same_location_only:
        candidates = candidates[
            candidates["location"] == target_user["location"]
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates["genre_score"] = candidates["genres"].apply(
        lambda value: jaccard_similarity(target_user["genres"], value)
    )

    candidates["frequency_score"] = candidates["frequency"].apply(
        lambda value: exact_match_score(target_user["frequency"], value)
    )

    candidates["purpose_score"] = candidates["purpose"].apply(
        lambda value: exact_match_score(target_user["purpose"], value)
    )

    candidates["skill_level_score"] = candidates["skill_level"].apply(
        lambda value: exact_match_score(target_user["skill_level"], value)
    )

    candidates["available_days_score"] = candidates["available_days"].apply(
        lambda value: jaccard_similarity(target_user["available_days"], value)
    )

    candidates["bio_score"] = bio_similarity(
        target_user["bio"],
        candidates["bio"],
    )

    candidates["score"] = (
        candidates["genre_score"] * weights.genre
        + candidates["frequency_score"] * weights.frequency
        + candidates["purpose_score"] * weights.purpose
        + candidates["skill_level_score"] * weights.skill_level
        + candidates["available_days_score"] * weights.available_days
        + candidates["bio_score"] * weights.bio
    )

    candidates["reasons"] = candidates.apply(
        lambda row: build_recommendation_reasons(target_user, row),
        axis=1,
    )

    return candidates.sort_values("score", ascending=False).head(top_k)
