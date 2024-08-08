from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Get credentials in order
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Load the embedding model
embeddings = download_hugging_face_embeddings()

# Initialize the Pinecone VectorDB
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mnnit-chatbot"

#Creating Embeddings for each text chunk & storing it
docsearch = PineconeVectorStore.from_existing_index(embedding = embeddings, index_name = index_name)

# Get the prompt template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# Initialize the LLM
llm = CTransformers(model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type = "llama",
                    verbose = True,
                    config = {'max_new_tokens':512,
                              'temperature':0.8})

# Getting the retrievalQA started
qa=RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever = docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Starting a chat
@app.route("/")
def index():
    return render_template('chat.html')

# Continuing a chat
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

# Custom port deployment
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)