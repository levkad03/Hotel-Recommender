from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
)

from .base import Base


class Hotel(Base):
    __tablename__ = "hotels"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    price_per_night = Column(Float, nullable=False)
    star_rating = Column(Float, nullable=False)
    review_score = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    description_embedding = Column(Vector(384))
