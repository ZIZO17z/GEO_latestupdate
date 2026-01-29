import os
import sys
import re
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists(): load_dotenv(dotenv_path)

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    sys.exit("❌ ERROR: GROQ_API_KEY missing.")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)

# 🧠 MEMORY FILES
DB_FOLDER = "geology_deep_index_db"
DB_PATH = BASE_DIR / DB_FOLDER
MEMORY_FILE = BASE_DIR / "learned_facts.txt" # <--- NEW PERMANENT BRAIN

FILES_CONFIG = {
    "textbook": "Essentials_of_Geology.pdf",
    "testbank": "testbank.pdf"
}

print("⚙️  Loading Neural Network...")
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# ==========================================
# 2. THE LEARNING SYSTEM (NEW)
# ==========================================
def learn_from_interaction(user_input, last_bot_response):
    """
    Checks if the user is correcting the AI. If yes, saves the fact.
    """
    # Only run if user seems to be correcting something
    triggers = ["no", "wrong", "actually", "mistake", "error", "chapter is", "not correct", "bad"]
    if not any(t in user_input.lower() for t in triggers):
        return

    # Ask Groq to extract the fact
    sys_msg = "You are a Database Updater. Extract the factual correction from the user's complaint. Return ONLY the fact."
    prompt = f"""
    Bot said: "{last_bot_response[:100]}..."
    User said: "{user_input}"
    
    If the user is correcting a Chapter Number, Topic, or Page, extract it as a statement.
    Example: "Chapter 16 is actually Atmosphere."
    If no fact, return "None".
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        fact = completion.choices[0].message.content.strip()
        
        if fact and "None" not in fact:
            with open(MEMORY_FILE, "a") as f:
                f.write(f"- {fact}\n")
            print(f"\n   🧠 [Self-Learning]: Memorized new fact -> {fact}")
            
    except:
        pass # Fail silently if learning fails

# ==========================================
# 3. THE BUILDER
# ==========================================
def build_deep_knowledge_base():
    # (Visual Bar Helper)
    def print_progress_bar(iteration, total, prefix='', length=40):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '░' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% Complete')
        sys.stdout.flush()

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        time.sleep(1)

    print("\n🚀 INITIALIZING DEEP SCAN PROTOCOL")
    all_docs = []
    
    for doc_type, filename in FILES_CONFIG.items():
        file_path = BASE_DIR / filename
        if file_path.exists():
            print(f"   📖 Reading {filename}...")
            loader = PyPDFLoader(str(file_path))
            raw_docs = loader.load()
            for doc in raw_docs:
                pg = doc.metadata.get("page", 0) + 1
                doc.metadata["source"] = doc_type
                doc.metadata["page"] = pg
                doc.page_content = f"SOURCE: {doc_type.upper()} | PDF_PAGE: {pg} | CONTENT: {doc.page_content}"
            all_docs.extend(raw_docs)

    if not all_docs: sys.exit("❌ No files found.")

    print("   ✂️  Fracturing text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_docs)
    total_chunks = len(split_docs)
    
    print_progress_bar(0, total_chunks, prefix='   🧠 Processing:', length=40)
    batch_size = 50 
    db = None
    
    for i in range(0, total_chunks, batch_size):
        batch = split_docs[i : i + batch_size]
        if db is None:
            db = FAISS.from_documents(batch, embedding_model)
        else:
            db.add_documents(batch)
        print_progress_bar(min(i + len(batch), total_chunks), total_chunks, prefix='   🧠 Processing:', length=40)

    print("\n   ✅ Knowledge Base Built.")
    db.save_local(str(DB_PATH))
    return db

# ==========================================
# 4. THE PROFESSOR (WITH MEMORY)
# ==========================================
def ask_groq_professor(user_input: str, db, history: list):
    
    # 1. LOAD LONG-TERM MEMORY
    learned_context = ""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r") as f:
            learned_context = f.read()

    # 2. TOPIC MAPPING (With Regex)
    chapter_map = {
        "15": "Mass Wasting Landslides",
        "16": "Atmosphere Composition Weather" # Added your correction manually too
    }
    
    search_query = user_input
    match = re.search(r"chapter\s*(\d+)", user_input.lower())
    if match:
        ch_num = match.group(1)
        if ch_num in chapter_map:
            search_query = f"{chapter_map[ch_num]} {user_input}"

    # 3. RETRIEVE
    results = db.similarity_search(search_query, k=25)
    evidence = ""
    for doc in results:
        src = doc.metadata.get("source", "UNKNOWN")
        pg = doc.metadata.get("page", 0)
        content = doc.page_content.replace("\n", " ")
        evidence += f"- [Source: {src} | Pg {pg}]: {content}\n"

    # 4. PROMPT
    system_msg = f"""
    You are an intelligent Geology Professor.
    
    ⚠️ CRITICAL: PREVIOUS USER CORRECTIONS (OBEY THESE ABOVE ALL):
    {learned_context}
    
    DATA FRAGMENTS FROM BOOK:
    {evidence}
    
    INSTRUCTIONS:
    1. Check "USER CORRECTIONS" first. If the user previously taught you that "Chapter 16 is Atmosphere", believe them.
    2. Synthesize the "DATA FRAGMENTS" to answer.
    3. If asked for questions, ensure they have options. Ignore broken text.
    4. If the data is missing, admit it, but suggest the closest topic found.
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
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if os.path.exists(DB_PATH):
        print("❓ Database found.")
        choice = input("   Type 'build' to re-train or 'chat': ").strip().lower()
        if choice == 'build':
            vector_db = build_deep_knowledge_base()
        else:
            print("   📂 Loading data...")
            vector_db = FAISS.load_local(str(DB_PATH), embedding_model, allow_dangerous_deserialization=True)
    else:
        vector_db = build_deep_knowledge_base()

    print("\n=============================================")
    print("🌋 GROQ GEOLOGY: SELF-LEARNING EDITION")
    print(f"📊 Concepts: {vector_db.index.ntotal}")
    print("=============================================")
    
    history = []
    last_response = ""
    
    while True:
        try:
            q = input("\nStudent: ").strip()
            if not q: continue
            if q.lower() in ['exit', 'quit']: break
            
            # 1. Check if user is teaching us something new
            if history:
                print("🧠 Checking for corrections...", end="\r")
                learn_from_interaction(q, last_response)
            
            # 2. Generate Answer
            print("⚡ Groq is thinking...           ", end="\r")
            ans = ask_groq_professor(q, vector_db, history)
            
            print(" " * 40, end="\r")
            print(f"Professor: {ans}")
            
            history.append(q)
            last_response = ans
            
        except KeyboardInterrupt:
            break