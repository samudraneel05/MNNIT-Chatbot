from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extracting pdf data
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob = "*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Creating text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Downloading embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings