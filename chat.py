# import os
from dotenv import load_dotenv #find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# Robust .env loading (CWD first, then this script's folder)
# _dotenv = find_dotenv(filename=".env", usecwd=True)
# if not _dotenv:
#     import pathlib
#     _dotenv = str(pathlib.Path(__file__).resolve().parent / ".env")
load_dotenv()

openai_client = OpenAI()

#vector Embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large"  # correct model name
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embeddings_model,
)

# Take user input 
user_query = input("Ask something: ")

# Relevant chunks from the vector db
search_result = vector_db.similarity_search(query=user_query)

def _meta_page(md: dict):
    # Try common keys from loaders
    return md.get('page') or md.get('page_number') or md.get('page_label') or md.get('Page_label') or 'unknown'

context = "\n\n\n".join([
    f"Page Content: {res.page_content}\nPage Number: {_meta_page(res.metadata)}\nFile Location: {res.metadata.get('source','unknown')}"
    for res in search_result
])

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who answeres user query based on the available 
    context retrieved from a PDF file along with page_contents and Page number. 

    You should only ans the user based on the following context and navigate the user 
    to open the right page number to know more. 

    context: 
    {context}

"""

response = openai_client.chat.completions.create(
    model = "gpt-4o",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT}, 
        {"role": "user", "content":user_query},
    ]
)

print(f"🤖: {response.choices[0].message.content}")