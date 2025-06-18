import os

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Float, Integer, String, Text, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))
Base = declarative_base()


class Hotel(Base):
    __tablename__ = "hotels"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    price_per_night = Column(Float, nullable=False)
    star_rating = Column(Float, nullable=False)
    review_score = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    description_embedding = Vector(768)


Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
