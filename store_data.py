import os

from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from db_setup import Hotel, engine
from mini_lm_embeddings import MiniLMEmbeddings

load_dotenv()

# Create embeddings instance
embeddings = MiniLMEmbeddings()

# Example hotel data
hotel_data = {
    "name": "Hotel Sunshine",
    "location": "Paris",
    "price_per_night": 120.0,
    "star_rating": 4.0,
    "review_score": 8.7,
    "description": "A cozy 4-star hotel in the heart of Paris with free breakfast.",
}

# Generate embedding for the description
embedding_vector = embeddings.embed_query(hotel_data["description"])

# Insert into database
Session = sessionmaker(bind=engine)
session = Session()

hotel = Hotel(
    name=hotel_data["name"],
    location=hotel_data["location"],
    price_per_night=hotel_data["price_per_night"],
    star_rating=hotel_data["star_rating"],
    review_score=hotel_data["review_score"],
    description=hotel_data["description"],
    description_embedding=embedding_vector,  # This works if using pgvector's Vector type
)

session.add(hotel)
session.commit()
session.close()

print("Hotel inserted with embedding.")
