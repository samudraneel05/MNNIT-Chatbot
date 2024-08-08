from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Calling the functions defined in helper.py for setup
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initializing the Pinecone VectorDB
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mnnit-chatbot"

#Creating Embeddings for each text chunk & storing it
docsearch = PineconeVectorStore.from_documents(
    text_chunks,
    embeddings,
    index_name = index_name
)