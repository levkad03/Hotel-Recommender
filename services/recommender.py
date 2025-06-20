import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_postgres import PGVector

from models.mini_lm_embeddings import MiniLMEmbeddings

load_dotenv()

llm = ChatGroq(
    temperature=0, model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")
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


def recommend_hotels(query: str, k: int):
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "name": doc.metadata.get("name", "Unknown"),
            "location": doc.metadata.get("location", "Unknown"),
            "description": doc.page_content,
        }
        for doc in results
    ]


def recommend_hotels_llm(query: str, k: int = 3):
    prompt = (
        "You are a hotel recommendation assistant. "
        f'Help the user find hotels based on their query: "{query}"\n'
        "Use the search_hotels tool to find relevant hotels and provide a helpful "
        f"Always call the search_hotels tool with k={k}."
        "response with recommendations."
    )
    response = llm_with_tools.invoke(prompt)
    print(response)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_hotels":
                args = tool_call["args"]
                result = search_hotels.invoke(args)
                tool_results.extend(result)
        follow_up_prompt = f"""Based on the hotel search results: {tool_results}
        Please provide a helpful recommendation response for the user's query: "{query}"
        Include details about the recommended hotels."""
        final_response = llm.invoke(follow_up_prompt)
        return {"response": final_response.content, "hotels": tool_results}
    return {"response": response.content, "hotels": []}
