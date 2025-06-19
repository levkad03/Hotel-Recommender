import os

from langchain_community.vectorstores import PGVector

from data.mini_lm_embeddings import MiniLMEmbeddings

embeddings = MiniLMEmbeddings()
connection_string = os.getenv("DATABASE_URL")

vectorestore = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name="hotels",
)

user_query = "I want a nice hotel in New York"

results = vectorestore.similarity_search(user_query, k=3)
print(results)

print("\nTop hotel recommendations:")
for doc in results:
    print(doc)
    meta = doc.metadata
    print(f"- {meta['name']} ({meta['location']}): {doc.page_content}")
