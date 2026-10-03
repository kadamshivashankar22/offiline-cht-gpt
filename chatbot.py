from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Simple chat loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() in ["exit", "quit", "q"]:
        break

    results = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content[:500])  # show first 500 chars
