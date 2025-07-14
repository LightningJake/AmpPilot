import logging
import os
import faiss
import time
import ollama
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define storage paths
DOCUMENTS_DIR = r"D:\Infosys hackathon\FULL_FINAL\knowledge"
FAISS_DB_PATH = r"D:\Infosys hackathon\FULL_FINAL\vector_store"

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set embedding model (only set once)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# Function to build FAISS index from files in DOCUMENTS_DIR
def build_and_store_index(documents):
    global query_engine
    faiss_index = faiss.IndexFlatL2(384)
    faiss_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=faiss_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=FAISS_DB_PATH)
    query_engine = index.as_query_engine()
    logger.info("FAISS index built successfully.")

# Always (re)build index from DOCUMENTS_DIR at startup
logger.info("Building FAISS index from files in DOCUMENTS_DIR...")
docs = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
build_and_store_index(docs)

@app.post("/query/")
async def query_model(question: str = Form(...)):
    if query_engine is None:
        raise HTTPException(status_code=400, detail="No FAISS index found. Check your documents.")

    action_start_time = time.time()
    logger.info(f"User question: {question}")

    try:
        response = query_engine.query(question)
        relevant_context = response.response if response else "No relevant information found."
    except Exception as e:
        logger.error(f"Error querying FAISS: {e}")
        relevant_context = "Error retrieving relevant information."

    try:
        prompt = f"""
        You are an intelligent assistant on AmpPilot — a platform that provides smart EV charging routes and optimized travel planning for electric vehicles.

        Using only the context provided below, answer the user's question clearly and accurately. Do not make assumptions. Avoid hallucinating information. Be concise and helpful.

        Context:
        {relevant_context}

        User's Question:
        {question}

        AmpPilot Assistant's Response:"""
        ollama_response = ollama.chat("llama3.2", messages=[{"role": "user", "content": prompt}])
        bot_response = ollama_response.message.content
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        bot_response = "Error processing your request."

    return {"response": bot_response}