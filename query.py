from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

# Embeddings must match indexer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vectorstore
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Example query
query = "What text was in the PDF, DOCX, and Image?"
results = vectorstore.similarity_search(query, k=5)

print("\n🔍 Query:", query)
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:\n{doc.page_content}")
