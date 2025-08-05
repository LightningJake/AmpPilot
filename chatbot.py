import logging
import os
import faiss
import time
import ollama
import numpy as np
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings
import warnings
import requests
from typing import Optional
import uvicorn

warnings.filterwarnings('ignore')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DOCUMENTS_DIR = "./knowledge"
FAISS_DB_PATH = "./vector_store"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embeddings and LLM setup
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# Index setup
query_engine = None

def build_or_load_index():
    global query_engine
    if os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
        logger.info("Loading existing FAISS index...")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=FAISS_DB_PATH))
    else:
        logger.info("Building new FAISS index...")
        docs = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
        faiss_index = faiss.IndexFlatL2(384)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        index.storage_context.persist(persist_dir=FAISS_DB_PATH)
    query_engine = index.as_query_engine()

build_or_load_index()

# --- Intent Detection ---
def detect_intent(question: str):
    q = question.lower()
    if any(k in q for k in ["towing", "tow", "breakdown", "repair", "help near me"]):
        return "emergency_help"
    elif any(k in q for k in ["nearby", "charging station", "station near me"]):
        return "station_search"
    elif any(k in q for k in ["battery health", "soh", "state of health"]):
        return "battery_health"
    elif any(k in q for k in ["range", "how far can i go"]):
        return "range_query"
    elif any(k in q for k in ["mechanic", "garage", "repair shop", "fix my car"]):
        return "emergency_help"

    return "doc_qa"

# --- API Helpers ---
def call_main_api(endpoint: str, params: dict):
    try:
        response = requests.get(f"http://localhost:8000{endpoint}", params=params, timeout=10)
        return response.json()
    except Exception as e:
        logger.error(f"Main API call to {endpoint} failed: {e}")
        return {"error": str(e)}

def summarize_station_data(result: dict) -> str:
    if "error" in result:
        return f"**Error:** {result['error']}"
    
    summary = result.get("summary", {})
    stations = result.get("stations", [])[:3]
    
    total = summary.get('total', 'N/A')
    closest = summary.get('closest_distance', 'N/A')
    average = summary.get('average_distance', 'N/A')
    
    lines = [
        f"**Charging Station Summary:**\n",
        f"- **Total stations available:** {total}",
        f"- **Closest station:** {closest} km away",
        f"- **Average distance:** {average} km\n",
        f"**Top 3 Nearby Stations:**"
    ]
    
    for i, s in enumerate(stations, 1):
        name = s.get('name', 'Unknown')
        occ = s.get('predicted_occupancy', 0.0)
        dist = s.get('distance_km', 0.0)
        lines.append(f"{i}. **{name}** - {dist:.1f} km away, {occ:.0%} occupancy")
    
    return "\n".join(lines)

def summarize_emergency_data(result: dict) -> str:
    if "error" in result:
        return f"**Error:** {result['error']}"

    garages = result.get("garages", [])[:3]
    towing = result.get("tow_services", [])[:3]
    total = result.get("total_services", 0)

    if not garages and not towing:
        return "**No emergency services found** near your location.\n\nTry expanding your search radius or contact local authorities for assistance."

    lines = [f"**{total} Emergency Services Found Near You**\n"]

    if garages:
        lines.append("🔧 **Nearby Mechanics & Garages:**")
        for i, g in enumerate(garages, 1):
            name = g.get("name", "Unknown")
            dist = g.get("distance_km", 0.0)
            phone = g.get("phone", "N/A")
            lines.append(f"{i}. **{name}** - {dist:.1f} km away | 📞 {phone}")
        lines.append("")  # Add space

    if towing:
        lines.append("🚗 **Nearby Towing Services:**")
        for i, t in enumerate(towing, 1):
            name = t.get("name", "Unknown")
            dist = t.get("distance_km", 0.0)
            phone = t.get("phone", "N/A")
            lines.append(f"{i}. **{name}** - {dist:.1f} km away | 📞 {phone}")

    return "\n".join(lines)



# --- Voice Intent Processing Endpoint ---
@app.post("/voice-intent/")
async def process_voice_intent(
    request: Request,
    intent_type: str = Form(...),
    parameters: str = Form(...),  # JSON string of parameters
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None)
):
    """Process voice-extracted intents and trigger appropriate actions"""
    try:
        import json
        params = json.loads(parameters) if parameters else {}
        
        # Use fallback location if not provided
        lat = lat or 17.43497122561351
        lng = lng or 78.34982693413811
        
        if intent_type == "route":
            # Route planning intent
            source = params.get('source', '')
            destination = params.get('destination', '')
            range_km = params.get('range', 50)
            
            if not source or not destination:
                return {
                    "success": False,
                    "error": "Source and destination are required for route planning.",
                    "intent": intent_type
                }
            
            # Call the main API route endpoint
            route_response = call_main_api("http://localhost:8000/ev-route", {
                "origin": source,
                "destination": destination,
                "current_range_km": range_km
            })
            
            if "error" not in route_response:
                return {
                    "success": True,
                    "intent": intent_type,
                    "message": f"Route planned from {source} to {destination} with {range_km}km range.",
                    "data": route_response
                }
            else:
                return {
                    "success": False,
                    "error": f"Route planning failed: {route_response.get('error', 'Unknown error')}",
                    "intent": intent_type
                }
                
        elif intent_type == "charging_station":
            # Charging station search intent
            location = params.get('location', 'current location')
            
            try:
                result = call_main_api("http://localhost:8000/nearby-stations", {"lat": lat, "lng": lng})
                summary = summarize_station_data(result)
                
                return {
                    "success": True,
                    "intent": intent_type,
                    "message": f"Found charging stations near {location}.",
                    "data": result,
                    "summary": summary
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Station search failed: {str(e)}",
                    "intent": intent_type
                }
                
        elif intent_type == "battery_health":
            # Battery health analysis intent
            return {
                "success": True,
                "intent": intent_type,
                "message": "Battery health analysis form will be opened.",
                "action": "open_battery_form"
            }
            
        elif intent_type == "emergency":
            # Emergency services intent
            try:
                result = requests.post(
                    "http://localhost:8000/car-breakdown",
                    json={"latitude": lat, "longitude": lng},
                    timeout=10
                ).json()
                summary = summarize_emergency_data(result)
                
                return {
                    "success": True,
                    "intent": intent_type,
                    "message": "Found emergency services nearby.",
                    "data": result,
                    "summary": summary
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Emergency service lookup failed: {str(e)}",
                    "intent": intent_type
                }
        
        else:
            return {
                "success": False,
                "error": f"Unknown intent type: {intent_type}",
                "intent": intent_type
            }
            
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "Invalid parameters format.",
            "intent": intent_type
        }
    except Exception as e:
        logger.error(f"Voice intent processing error: {e}")
        return {
            "success": False,
            "error": f"Failed to process voice intent: {str(e)}",
            "intent": intent_type
        }

# --- Chat Endpoint ---
@app.post("/query/")
async def query_model(
    request: Request,
    question: str = Form(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None)
):
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized.")

    intent = detect_intent(question)
    logger.info(f"Detected intent: {intent}")

    # Use fallback if not provided
    lat = lat or 17.43497122561351
    lng = lng or 78.34982693413811
    logger.info(f"Using location: lat={lat}, lng={lng}")

    relevant_context = ""
    if intent == "station_search":
        try:
            result = call_main_api("/nearby-stations", {"lat": lat, "lng": lng})
            relevant_context = summarize_station_data(result)
        except Exception as e:
            logger.error(f"Station search API call failed: {e}")
            relevant_context = "**Station search is currently unavailable.** Please try again later or check your internet connection."
    elif intent == "emergency_help":
        try:
            result = requests.post(
                "http://localhost:8000/car-breakdown",
                json={"latitude": lat, "longitude": lng},
                timeout=10
            ).json()
            relevant_context = summarize_emergency_data(result)
        except Exception as e:
            logger.error(f"Emergency API call failed: {e}")
            relevant_context = "Emergency service lookup is currently unavailable. Please contact local emergency services directly."
    elif intent == "battery_health":
        relevant_context = """**Battery Health Prediction** is available through our EV analysis module.

**To check your battery health:**
1. Navigate to the **Battery Health Form** in the main application
2. Enter your vehicle details and usage patterns  
3. Get detailed **State of Health (SOH)** analysis
4. Receive personalized maintenance recommendations

This feature provides accurate battery degradation analysis based on your specific EV model and usage patterns."""
    elif intent == "range_query":
        relevant_context = """**EV Range** depends on several key factors:

**Primary Factors:**
- **Battery Health**: Current state of charge and overall battery condition
- **Temperature**: Extreme hot or cold weather reduces range significantly  
- **Driving Style**: Aggressive acceleration and high speeds consume more energy
- **Terrain**: Hills and inclines require more power
- **Climate Control**: AC/heating usage impacts range

**Tips to Maximize Range:**
1. Maintain optimal battery temperature
2. Use eco-driving modes when available
3. Pre-condition your vehicle while plugged in
4. Plan routes with charging stations using our route planner

For specific range estimates for your vehicle, please use our **Range Calculator** feature."""
    else:
        try:
            response = query_engine.query(question)
            relevant_context = response.response if response else "No relevant information found."
        except Exception as e:
            logger.error(f"FAISS error: {e}")
            relevant_context = "Document retrieval error."

    prompt = f"""
You are AmpPilot Assistant - India's smart EV companion ⚡

Instructions for formatting your response:
- Use **bold text** for important terms like vehicle names, key specifications, or important phrases
- Use bullet points (- item) for simple lists
- Use numbered lists (1. item) for step-by-step instructions or procedures  
- Use line breaks between different sections
- Keep responses clear, concise and well-structured
- Focus on Indian EV ecosystem and infrastructure

Context Information:
{relevant_context}

User's Question: {question}

Please provide a well-formatted response following the above guidelines:"""

    try:
        ollama_response = ollama.chat("llama3.2", messages=[{"role": "user", "content": prompt}])
        bot_response = ollama_response.message.content.strip()
        
        # Ensure response is not empty
        if not bot_response:
            bot_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        bot_response = "I'm currently experiencing technical difficulties. Please try again in a moment."

    return {
        "response": bot_response,
        "intent": intent,
        "status": "success"
    }

# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok", "query_engine_ready": query_engine is not None}

# --- Startup ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)