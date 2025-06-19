from pydantic import BaseModel


class HotelRecommendation(BaseModel):
    name: str
    location: str
    description: str
