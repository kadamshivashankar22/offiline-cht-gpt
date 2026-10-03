from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredImageLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Loaders
pdf_loader = PyPDFLoader("sample.pdf")
docx_loader = UnstructuredWordDocumentLoader("sample.docx")
img_loader = UnstructuredImageLoader("sample.png")

pdf_docs = pdf_loader.load()
docx_docs = docx_loader.load()
img_docs = img_loader.load()

all_docs = pdf_docs + docx_docs + img_docs

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store (Chroma auto-persists now)
vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory="./chroma_db")

print("✅ Documents indexed successfully into ChromaDB!")
