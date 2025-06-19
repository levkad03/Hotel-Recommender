import os
from typing import List

import uvicorn
from fastapi import FastAPI
from langchain_postgres import PGVector

from data.mini_lm_embeddings import MiniLMEmbeddings
from schemas.hotel_recommendation import HotelRecommendation
from schemas.recommendation_request import RecommendationRequest

app = FastAPI(
    title="Hotel Recommender API",
    description="API for hotel recommendation bot",
    version="1.0.0",
)

embeddings = MiniLMEmbeddings()
connection_string = os.getenv("DATABASE_URL")
vectorstore = PGVector(
    connection=connection_string,
    embeddings=embeddings,
    collection_name="hotels",
)


@app.post("/recommend", response_model=List[HotelRecommendation])
def recommend_hotels(request: RecommendationRequest):
    results = vectorstore.similarity_search(request.query, k=request.k)
    recommendations = []

    for doc in results:
        meta = doc.metadata
        recommendations.append(
            HotelRecommendation(
                name=meta.get("name", "Unknown"),
                location=meta.get("location", "Unknown"),
                description=doc.page_content,
            )
        )

    return recommendations


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
