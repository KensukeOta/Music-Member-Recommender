from fastapi import FastAPI, HTTPException, Query
import pandas as pd

from app.recommender import recommend_users
from app.schemas import RecommendationResponse

app = FastAPI()

users = pd.read_csv("app/data/users.csv")


@app.get(
    "/users/{user_id}/recommendations",
    response_model=list[RecommendationResponse],
)
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=5, ge=1, le=20),
    same_location_only: bool = True,
):
    try:
        recommendations = recommend_users(
            user_id=user_id,
            users=users,
            top_k=top_k,
            same_location_only=same_location_only,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")

    return recommendations[
        [
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
        ]
    ].to_dict(orient="records")
