import os
import shutil
import re
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# --- CONFIG ---
BASE_DIR = Path(__file__).parent
DB_FOLDER = "geology_deep_index_db"
DB_PATH = BASE_DIR / DB_FOLDER
MEMORY_FILE = BASE_DIR / "learned_facts.txt"

# Load Env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Global Variables (Loaded on Startup)
vector_db = None
embedding_model = None
client = None

# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load the heavy AI models once
    global vector_db, embedding_model, client
    print("⚙️  Booting up Geology API...")
    
    if not GROQ_API_KEY:
        print("❌ CRITICAL: GROQ_API_KEY missing.")
    
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    
    print("🧠 Loading Embedding Model (This takes a moment)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    if os.path.exists(DB_PATH):
        print("📂 Loading Vector Database...")
        vector_db = FAISS.load_local(str(DB_PATH), embedding_model, allow_dangerous_deserialization=True)
        print(f"✅ Ready! Indexed Concepts: {vector_db.index.ntotal}")
    else:
        print("⚠️ WARNING: Database not found. Please upload 'geology_deep_index_db'.")
    
    yield
    # SHUTDOWN
    print("💤 Shutting down...")

# --- APP SETUP ---
app = FastAPI(title="Geology Professor API", lifespan=lifespan)

# Enable CORS (Allows websites to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    question: str
    history: list[str] = []

class ChatResponse(BaseModel):
    answer: str
    source_documents: list[str] = []

# --- HELPER FUNCTIONS ---
def get_learned_facts():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r") as f:
            return f.read()
    return ""

def learn_from_feedback(user_input: str, last_response: str):
    # Simple self-correction logic
    triggers = ["no", "wrong", "mistake", "error", "actually", "chapter is"]
    if not any(t in user_input.lower() for t in triggers):
        return

    sys_msg = "You are a Database Updater. Extract the factual correction. Return ONLY the fact."
    prompt = f"Bot said: {last_response[:100]}... User said: {user_input}. Extract the fact."
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0
        )
        fact = completion.choices[0].message.content.strip()
        if fact and "None" not in fact:
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- {fact}\n")
    except:
        pass

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "Online", "message": "Geology Professor is ready."}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    global vector_db
    
    if not vector_db:
        raise HTTPException(status_code=503, detail="Database not loaded.")

    user_input = payload.question
    history = payload.history
    
    # 1. Smart Topic Map
    chapter_map = {
        "15": "Mass Wasting Landslides",
        "16": "Atmosphere Composition Weather"
    }
    search_query = user_input
    match = re.search(r"chapter\s*(\d+)", user_input.lower())
    if match and match.group(1) in chapter_map:
        search_query = f"{chapter_map[match.group(1)]} {user_input}"

    # 2. Retrieve
    # We use a try/except block in case of index issues
    try:
        results = vector_db.similarity_search(search_query, k=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

    evidence = ""
    sources = []
    for doc in results:
        src = f"Pg {doc.metadata.get('page', '?')}"
        evidence += f"- [{src}]: {doc.page_content.replace('\n', ' ')}\n"
        sources.append(src)

    # 3. Prompt
    learned_context = get_learned_facts()
    system_msg = f"""
    You are an intelligent Geology Professor.
    
    ⚠️ USER CORRECTIONS:
    {learned_context}
    
    DATA FRAGMENTS:
    {evidence}
    
    INSTRUCTIONS:
    - If the user specifies a Chapter, FOCUS ONLY ON THAT CHAPTER'S TOPIC.
    - Ignore unrelated testbank questions.
    - If providing questions, include Options and Answers.
    """

    messages = [{"role": "system", "content": system_msg}]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h})
    messages.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500
        )
        answer = completion.choices[0].message.content
        
        # Trigger background learning
        # (In a real production app, this should be a background task)
        learn_from_feedback(user_input, "Previous context placeholder") 
        
        return {"answer": answer, "source_documents": list(set(sources))}
        
    except Exception as e:
        return {"answer": f"API Error: {str(e)}", "source_documents": []}