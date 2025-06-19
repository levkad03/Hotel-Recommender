from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    query: str
    k: int = 3
