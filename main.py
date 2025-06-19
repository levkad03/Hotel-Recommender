import os

from langchain_postgres import PGVector

from data.mini_lm_embeddings import MiniLMEmbeddings

embeddings = MiniLMEmbeddings()
connection_string = os.getenv("DATABASE_URL")

vectorestore = PGVector(
    connection=connection_string,
    embeddings=embeddings,
    collection_name="hotels",
)

user_query = "I want a nice hotel in New York"

# TODO: Does not work, need to remake it
results = vectorestore.similarity_search(user_query, k=3)
print(results)

print("\nTop hotel recommendations:")
for doc in results:
    print(doc)
    meta = doc.metadata
    print(f"- {meta['name']} ({meta['location']}): {doc.page_content}")
