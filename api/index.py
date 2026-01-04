"""
ArionXiv Serverless API for Vercel
Lightweight proxy to MongoDB for user auth, library, and settings
"""

import os
import hashlib
import secrets
import hmac
import re
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId

# Initialize FastAPI
app = FastAPI(title="ArionXiv API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# MongoDB connection (lazy)
_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise HTTPException(500, "Database not configured")
        _client = MongoClient(uri)
        _db = _client[os.environ.get("DATABASE_NAME", "arionxiv")]
    return _db

# JWT helpers
def get_secret():
    key = os.environ.get("JWT_SECRET_KEY")
    if not key:
        raise HTTPException(500, "JWT not configured")
    return key

def create_token(user_data: dict) -> str:
    payload = {
        "user_id": str(user_data.get("_id")),
        "email": user_data.get("email"),
        "user_name": user_data.get("user_name"),
        "exp": datetime.utcnow() + timedelta(days=30),  # 30-day token validity
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, get_secret(), algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, get_secret(), algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# Password helpers
def hash_password(password: str, salt: bytes = None) -> tuple:
    if salt is None:
        salt = secrets.token_bytes(32)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return pw_hash.hex(), salt.hex()

def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    salt = bytes.fromhex(stored_salt)
    pw_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(pw_hash, stored_hash)

# Request models
class RegisterRequest(BaseModel):
    email: str
    user_name: str
    password: str
    full_name: str = ""

class LoginRequest(BaseModel):
    identifier: str
    password: str

class LibraryAddRequest(BaseModel):
    arxiv_id: str
    title: str = ""
    authors: List[str] = []
    categories: List[str] = []
    abstract: str = ""
    tags: List[str] = []
    notes: str = ""

class ChatSessionRequest(BaseModel):
    paper_id: str
    title: Optional[str] = None
    paper_title: Optional[str] = None

# Endpoints

@app.get("/")
async def root():
    return {"status": "ok", "service": "ArionXiv API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to check if env vars are configured (returns only presence, not values)"""
    return {
        "OPENROUTER_API_KEY": "set" if os.environ.get("OPENROUTER_API_KEY") else "NOT SET",
        "OPENROUTER_API_KEY_length": len(os.environ.get("OPENROUTER_API_KEY", "")),
        "MONGODB_URI": "set" if os.environ.get("MONGODB_URI") else "NOT SET",
        "JWT_SECRET_KEY": "set" if os.environ.get("JWT_SECRET_KEY") else "NOT SET",
    }

@app.get("/debug/db-test")
async def debug_db_test():
    """Test MongoDB connection - remove after debugging"""
    try:
        db = get_db()
        # Try to insert a test document
        test_id = ObjectId()
        test_doc = {
            "_id": test_id,
            "test": True,
            "created_at": datetime.utcnow()
        }
        result = db.debug_test.insert_one(test_doc)
        # Delete it immediately
        db.debug_test.delete_one({"_id": test_id})
        return {
            "success": True,
            "message": "MongoDB connection works",
            "inserted_id": str(test_id),
            "database": db.name
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Auth endpoints
@app.post("/auth/register")
async def register(request: RegisterRequest):
    db = get_db()
    
    # Validate
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', request.email):
        return {"success": False, "error": "Invalid email format"}
    
    user_name = request.user_name.strip().lower()
    if not re.match(r'^[a-z0-9._-]+$', user_name) or len(user_name) < 3:
        return {"success": False, "error": "Invalid username"}
    
    if len(request.password) < 8:
        return {"success": False, "error": "Password too short"}
    
    # Check exists
    if db.users.find_one({"user_name": user_name}):
        return {"success": False, "error": "Username taken"}
    if db.users.find_one({"email": request.email}):
        return {"success": False, "error": "Email exists"}
    
    # Create user
    pw_hash, salt = hash_password(request.password)
    user = {
        "email": request.email,
        "user_name": user_name,
        "username": user_name,
        "full_name": request.full_name,
        "password_hash": pw_hash,
        "password_salt": salt,
        "created_at": datetime.utcnow(),
        "is_active": True,
        "auth_provider": "local"
    }
    result = db.users.insert_one(user)
    
    return {
        "success": True,
        "user": {
            "id": str(result.inserted_id),
            "email": request.email,
            "user_name": user_name,
            "full_name": request.full_name
        }
    }

@app.post("/auth/login")
async def login(request: LoginRequest):
    db = get_db()
    
    # Find user
    query = {"email": request.identifier} if "@" in request.identifier else {"user_name": request.identifier.lower()}
    user = db.users.find_one(query)
    
    if not user:
        return {"success": False, "error": "Invalid credentials"}
    
    if not verify_password(request.password, user["password_hash"], user["password_salt"]):
        return {"success": False, "error": "Invalid credentials"}
    
    # Update last login
    db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    
    token = create_token(user)
    return {
        "success": True,
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "user_name": user.get("user_name") or user.get("username"),
            "full_name": user.get("full_name", "")
        }
    }

@app.get("/auth/profile")
async def get_profile(current_user: dict = Depends(verify_token)):
    db = get_db()
    user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "success": True,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "user_name": user.get("user_name"),
            "full_name": user.get("full_name", "")
        }
    }

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(verify_token)):
    return {"success": True, "message": "Logged out"}

# Library endpoints
@app.get("/library")
async def get_library(
    current_user: dict = Depends(verify_token),
    limit: int = Query(default=20, ge=1, le=100)
):
    db = get_db()
    papers = list(db.user_libraries.find({"user_id": current_user["user_id"]}).limit(limit))
    for p in papers:
        p["_id"] = str(p["_id"])
    return {"success": True, "papers": papers, "count": len(papers)}

@app.post("/library")
async def add_to_library(request: LibraryAddRequest, current_user: dict = Depends(verify_token)):
    db = get_db()
    
    # Check exists
    existing = db.user_libraries.find_one({
        "user_id": current_user["user_id"],
        "arxiv_id": request.arxiv_id
    })
    if existing:
        return {"success": False, "message": "Paper already in library"}
    
    entry = {
        "user_id": current_user["user_id"],
        "arxiv_id": request.arxiv_id,
        "title": request.title,
        "authors": request.authors,
        "categories": request.categories,
        "abstract": request.abstract,
        "tags": request.tags,
        "notes": request.notes,
        "read_status": "unread",
        "added_at": datetime.utcnow()
    }
    db.user_libraries.insert_one(entry)
    return {"success": True, "message": "Paper added to library"}

@app.delete("/library/{arxiv_id}")
async def remove_from_library(arxiv_id: str, current_user: dict = Depends(verify_token)):
    db = get_db()
    result = db.user_libraries.delete_one({
        "user_id": current_user["user_id"],
        "arxiv_id": arxiv_id
    })
    if result.deleted_count > 0:
        return {"success": True, "message": "Paper removed"}
    return {"success": False, "message": "Paper not found"}

# Settings endpoints
@app.get("/settings")
async def get_settings(current_user: dict = Depends(verify_token)):
    db = get_db()
    user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    return {"success": True, "settings": user.get("settings", {})}

@app.put("/settings")
async def update_settings(settings: dict, current_user: dict = Depends(verify_token)):
    db = get_db()
    db.users.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$set": {"settings": settings}}
    )
    return {"success": True, "message": "Settings updated"}

# Chat sessions
@app.get("/chat/sessions")
async def get_chat_sessions(current_user: dict = Depends(verify_token)):
    db = get_db()
    cutoff = datetime.utcnow() - timedelta(hours=24)
    sessions = list(db.chat_sessions.find({
        "user_id": current_user["user_id"],
        "updated_at": {"$gte": cutoff}
    }).limit(10))
    for s in sessions:
        # Add session_id as alias for _id for client compatibility
        s["session_id"] = str(s["_id"])
        s["_id"] = str(s["_id"])
    return {"success": True, "sessions": sessions}

@app.post("/chat/session")
async def create_chat_session(request: ChatSessionRequest, current_user: dict = Depends(verify_token)):
    try:
        db = get_db()
        title = request.title or request.paper_title or ""
        paper_title = request.paper_title or request.title or ""
        
        # Generate ObjectId explicitly
        session_id = ObjectId()
        
        session_data = {
            "_id": session_id,
            "session_id": str(session_id),  # Add session_id field (MongoDB has unique index on this)
            "user_id": current_user["user_id"],
            "paper_id": request.paper_id,
            "title": title,
            "paper_title": paper_title,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        db.chat_sessions.insert_one(session_data)
        return {"success": True, "session_id": str(session_id)}
    except Exception as e:
        import traceback
        error_detail = f"Failed to create session: {str(e)}. Traceback: {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.put("/chat/session/{session_id}")
async def update_chat_session(session_id: str, messages: List[dict], current_user: dict = Depends(verify_token)):
    db = get_db()
    db.chat_sessions.update_one(
        {"_id": ObjectId(session_id), "user_id": current_user["user_id"]},
        {"$set": {"messages": messages, "updated_at": datetime.utcnow()}}
    )
    return {"success": True, "message": "Session updated"}

# Daily analysis
@app.get("/daily")
async def get_daily_analysis(current_user: dict = Depends(verify_token)):
    db = get_db()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Note: collection is 'daily_dose' (not 'daily_doses') to match service
    dose = db.daily_dose.find_one({
        "user_id": current_user["user_id"],
        "date": today
    })
    if dose:
        dose["_id"] = str(dose["_id"])
    return {"success": True, "dose": dose}


# Chat message endpoint - LLM inference via OpenRouter with fallback
class ChatMessageRequest(BaseModel):
    message: str
    paper_id: str
    session_id: Optional[str] = None

# Fallback models (same as CLI openrouter_client.py)
FALLBACK_MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "moonshotai/kimi-k2:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

# RAG Chat prompt template (same as prompts/prompts.py)
RAG_CHAT_PROMPT = """You are an AI research assistant helping users understand research papers. Your role is to provide accurate, helpful answers based on the paper content.

CONVERSATION HISTORY:
{history}

USER QUESTION: {message}

Instructions:
- Provide comprehensive, detailed answers
- If you don't know, say so clearly
- Be conversational but maintain technical accuracy
- Structure longer answers with clear sections"""

@app.post("/chat/message")
async def send_chat_message(request: ChatMessageRequest, current_user: dict = Depends(verify_token)):
    """Generate AI response using OpenRouter with session history support"""
    import httpx
    import traceback
    
    try:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise HTTPException(500, "OPENROUTER_API_KEY not set in environment")
        
        # Fetch session history if session_id is provided
        history_text = "No previous conversation."
        session = None
        db = None
        if request.session_id:
            try:
                db = get_db()
                session = db.chat_sessions.find_one({
                    "$or": [
                        {"_id": ObjectId(request.session_id)},
                        {"session_id": request.session_id}
                    ],
                    "user_id": current_user["user_id"]
                })
                if session and session.get("messages"):
                    history_lines = []
                    for msg in session["messages"][-10:]:
                        role = "User" if msg.get("type") == "user" else "Assistant"
                        history_lines.append(f"{role}: {msg.get('content', '')}")
                    history_text = "\n".join(history_lines)
            except Exception as db_err:
                logger.warning(f"Session fetch error (continuing): {db_err}")
        
        # Build prompt
        prompt = RAG_CHAT_PROMPT.format(history=history_text, message=request.message)
        
        primary_model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
        
        last_error = None
        response_text = None
        used_model = None
        
        async with httpx.AsyncClient() as client:
            for model in models_to_try:
                try:
                    logger.info(f"Trying model: {model}")
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                        timeout=60.0
                    )
                    logger.info(f"OpenRouter response: {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        response_text = data["choices"][0]["message"]["content"]
                        used_model = model
                        break
                    else:
                        last_error = f"{model}: HTTP {response.status_code} - {response.text[:200]}"
                        logger.warning(last_error)
                except Exception as model_err:
                    last_error = f"{model}: {str(model_err)}"
                    logger.warning(f"Model {model} failed: {model_err}")
        
        if not response_text:
            raise HTTPException(500, f"All models failed. Last: {last_error}")
        
        # Update session if exists
        if session and db:
            try:
                new_messages = session.get("messages", []) + [
                    {"type": "user", "content": request.message, "timestamp": datetime.utcnow().isoformat()},
                    {"type": "assistant", "content": response_text, "timestamp": datetime.utcnow().isoformat()}
                ]
                db.chat_sessions.update_one(
                    {"_id": session["_id"]},
                    {"$set": {"messages": new_messages, "updated_at": datetime.utcnow()}}
                )
            except Exception:
                pass
        
        return {"success": True, "response": response_text, "model": used_model}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Server error: {str(e)}")
