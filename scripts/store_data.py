import json
import os

from dotenv import load_dotenv
from langchain_postgres import PGVector

from models.mini_lm_embeddings import MiniLMEmbeddings

load_dotenv()

# Create embeddings instance
embeddings = MiniLMEmbeddings()
connection_string = os.getenv("DATABASE_URL")

vectorstore = PGVector(
    connection=connection_string,
    embeddings=embeddings,
    collection_name="hotels",
)

# Example hotel data
with open("hotel_data.json", "r", encoding="utf-8") as f:
    hotels = json.load(f)

texts = [hotel["description"] for hotel in hotels]
metadatas = hotels  # or add more metadata fields if you have them

vectorstore.delete_collection()
vectorstore.create_collection()
vectorstore.add_texts(texts, metadatas=metadatas)
print("Hotels ingested into LangChain vectorstore.")
