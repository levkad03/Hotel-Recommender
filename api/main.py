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

load_dotenv()

llm = ChatGroq(
    temperature=0, model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")
)


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


@tool
def search_hotels(query: str, k: int = 3) -> list:
    """Search for hotels matching the user's query."""
    # Convert k to int if it's a string (from LLM/tool call)
    if isinstance(k, str):
        k = int(k)
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "name": doc.metadata.get("name", "Unknown"),
            "location": doc.metadata.get("location", "Unknown"),
            "description": doc.page_content,
        }
        for doc in results
    ]


llm_with_tools = llm.bind_tools([search_hotels])


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


@app.post("/recommend_llm")
def recommend_hotels_llm(request: RecommendationRequest):
    user_input = request.query

    # Create a prompt that encourages the LLM to use the search tool
    prompt = f"""You are a hotel recommendation assistant. Help the user find hotels based on their query: "{user_input}"
    
    Use the search_hotels tool to find relevant hotels and provide a helpful response with recommendations."""

    # Invoke the LLM with tools
    response = llm_with_tools.invoke(prompt)

    # Handle tool calls if the LLM decides to use them
    if hasattr(response, "tool_calls") and response.tool_calls:
        # Execute tool calls and get results
        tool_results = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_hotels":
                args = tool_call["args"]
                result = search_hotels.invoke(args)
                tool_results.extend(result)

        # Create a follow-up prompt with the tool results
        follow_up_prompt = f"""Based on the hotel search results: {tool_results}
        
        Please provide a helpful recommendation response for the user's query: "{user_input}"
        Include details about the recommended hotels."""

        final_response = llm.invoke(follow_up_prompt)
        return {"response": final_response.content, "hotels": tool_results}

    return {"response": response.content, "hotels": []}
