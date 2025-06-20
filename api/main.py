import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_postgres import PGVector

from models.mini_lm_embeddings import MiniLMEmbeddings
from schemas.hotel_recommendation import HotelRecommendation
from schemas.recommendation_request import RecommendationRequest
from services.recommender import recommend_hotels, recommend_hotels_llm

app = FastAPI(
    title="Hotel Recommender API",
    description="API for hotel recommendation bot",
    version="1.0.0",
)


@app.post("/recommend", response_model=List[HotelRecommendation])
def recommend_hotels_endpoint(request: RecommendationRequest):
    return recommend_hotels(request.query, request.k)


@app.post("/recommend_llm")
def recommend_hotels_llm_endpoint(request: RecommendationRequest):
    return recommend_hotels_llm(request.query, request.k)
