import json

from db_setup import Hotel, engine
from dotenv import load_dotenv
from mini_lm_embeddings import MiniLMEmbeddings
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Create embeddings instance
embeddings = MiniLMEmbeddings()

# Example hotel data
with open("data/hotel_data.json", "r", encoding="utf-8") as f:
    hotels = json.load(f)

Session = sessionmaker(bind=engine)
session = Session()

for hotel in hotels:
    # Generate embedding for the description
    embedding_vector = embeddings.embed_query(hotel["description"])

    # Create Hotel instance
    hotel_obj = Hotel(
        name=hotel["name"],
        location=hotel["location"],
        price_per_night=hotel["price_per_night"],
        star_rating=hotel["star_rating"],
        review_score=hotel["review_score"],
        description=hotel["description"],
        description_embedding=embedding_vector,
    )
    session.add(hotel_obj)


session.commit()
session.close()

print("Hotel inserted with embedding.")
