from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    user_id: int
    name: str
    age: int
    location: str
    instrument: str
    genres: str
    skill_level: str
    purpose: str
    frequency: str
    available_days: str
    score: float
    reasons: list[str]
