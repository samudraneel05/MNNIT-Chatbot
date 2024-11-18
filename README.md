# MNNIT-Chatbot

## Overview

The **MNNIT Chatbot** is a conversational AI system designed to provide accurate and context-aware responses using a fallback Retrieval-Augmented Generation (RAG) mechanism. It integrates cutting-edge AI tools and frameworks to leverage both cloud-based and local AI models for an optimal user experience. This project is built specifically for use cases involving MNNIT-related information, sourced from two provided PDFs.

---

## Key Features

- **Hybrid Query Mechanism**: Queries first interact with a cloud-based AI model (OpenAI GPT-4/Omni API). If the response is unsatisfactory, the system falls back to a local Llama 2-7B model using a RAG mechanism.
- **PDF Integration**: Extracts, chunks, and embeds information from two provided MNNIT PDFs.
- **Pinecone Vector Database**: Stores embeddings for efficient retrieval during query processing.
- **LangChain Integration**: Builds robust prompt templates and retrieval-based question-answering (QA) workflows.
- **Local LLM Deployment**: Uses the Llama 2-7B model (quantized) via CTransformers for resource-efficient local execution.
- **Flask Web Application**: Provides a user-friendly interface for interaction and deployment.

---

## Technical Workflow

1. **PDF Processing**:
   - Two PDFs containing MNNIT-specific information are ingested.
   - Text is extracted and divided into manageable chunks.
   - Embeddings are created for these chunks using a pre-trained model.

2. **Vector Database**:
   - Generated embeddings are stored in the Pinecone vector database.
   - Pinecone enables efficient similarity searches to fetch the most relevant chunks during queries.

3. **Query Processing**:
   - User queries are first sent to a cloud-based AI model (e.g., GPT-4/Omni API).
   - If the response quality is deemed unsatisfactory, the fallback RAG mechanism is activated.
   - The fallback mechanism performs:
     - **Contextual Retrieval**: Relevant information is retrieved from Pinecone.
     - **Local LLM Execution**: The Llama 2-7B model processes the retrieved context to generate a refined response.

4. **Prompt Engineering**:
   - LangChain is used to structure prompts for both the cloud-based and local models.
   - A custom prompt template ensures alignment with the chatbot’s context and objectives.

5. **Deployment**:
   - The Flask framework serves as the frontend interface for interacting with the chatbot.
   - End-users can query the chatbot via a web-based application.

---

## Prerequisites

- Python 3.8
- Conda (for environment management)
- Access to APIs (OpenAI GPT-4/Omni)
- Pre-downloaded **Llama 2-7B Model**: 
  - File: `llama-2-7b-chat.ggmlv3.q4_0.bin`
  - [Download Link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/samudraneel05/MNNIT-Chatbot.git
   cd MNNIT-Chatbot
2. Run the following commands one by one:
```bash
conda create -n mnnitchatbot python=3.8 -y
```

```bash
conda activate mnnitchatbot
```

```bash
pip install -r requirements.txt
```
