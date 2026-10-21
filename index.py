# File responsible for indexing the data
# import os
from dotenv import load_dotenv #find_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# _dotenv = find_dotenv(filename=".env", usecwd=True)
# if not _dotenv:
#     _dotenv = str(Path(__file__).resolve().parent / ".env")
load_dotenv()

# if not os.getenv("OPENAI_API_KEY"):
#     raise EnvironmentError(
#         "OPENAI_API_KEY is not set. Put it in a .env at the project root or set the environment variable."
#     )

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load this file in python program
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()

# print(f"Loaded {len(docs)} pages from {pdf_path}")
# if docs:
#     preview = docs[0].page_content[:200].replace("\n", " ")
#     print(f"First page preview: {preview}...")

# Splits the docs into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 400
)

chunks = text_splitter.split_documents(documents=docs)

#vector Embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large"  # correct model name
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    url="http://localhost:6333",
    collection_name="learning_rag",
)

print("Indexing of documents done...")


#PyPDFLoader turns your PDF from a file on disk into structured Document objects that your RAG pipeline can process further.