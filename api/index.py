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
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps

# Setup logging
logger = logging.getLogger(__name__)

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

@app.get("/debug/openrouter-test")
async def debug_openrouter_test():
    """Test OpenRouter API directly - ultra defensive error handling"""
    try:
        import requests
    except Exception as e:
        return {"error": f"Failed to import requests: {e}", "type": "ImportError"}
    
    try:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            return {"error": "OPENROUTER_API_KEY not set"}
        
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}"},
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [{"role": "user", "content": "Say hello"}]
            },
            timeout=30
        )
        return {
            "status_code": resp.status_code,
            "response_preview": resp.text[:500] if resp.text else "empty",
            "success": resp.status_code == 200
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

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

@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str, current_user: dict = Depends(verify_token)):
    """Get a specific chat session with full details"""
    db = get_db()
    try:
        session = db.chat_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": current_user["user_id"]
        })
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session["session_id"] = str(session["_id"])
        session["_id"] = str(session["_id"])
        return {"success": True, "session": session}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")


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


@app.get("/daily/settings")
async def get_daily_settings(current_user: dict = Depends(verify_token)):
    """Get user's daily dose settings"""
    db = get_db()
    user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    preferences = user.get("preferences", {})
    daily_dose = preferences.get("daily_dose", {})
    
    return {
        "success": True,
        "settings": {
            "enabled": daily_dose.get("enabled", False),
            "scheduled_time": daily_dose.get("scheduled_time"),
            "keywords": daily_dose.get("keywords", preferences.get("keywords", [])),
            "max_papers": daily_dose.get("max_papers", 5),
            "categories": preferences.get("categories", ["cs.AI", "cs.LG"])
        }
    }


@app.put("/daily/settings")
async def update_daily_settings(request: dict, current_user: dict = Depends(verify_token)):
    """Update user's daily dose settings"""
    db = get_db()
    
    # Build update dict
    updates = {}
    if "enabled" in request:
        updates["preferences.daily_dose.enabled"] = request["enabled"]
    if "scheduled_time" in request:
        updates["preferences.daily_dose.scheduled_time"] = request["scheduled_time"]
    if "keywords" in request:
        updates["preferences.daily_dose.keywords"] = request["keywords"]
        updates["preferences.keywords"] = request["keywords"]  # Also update main keywords
    if "max_papers" in request:
        updates["preferences.daily_dose.max_papers"] = min(request["max_papers"], 10)
    
    if updates:
        updates["preferences.daily_dose.updated_at"] = datetime.utcnow()
        db.users.update_one(
            {"_id": ObjectId(current_user["user_id"])},
            {"$set": updates}
        )
    
    return {"success": True, "message": "Daily dose settings updated"}


@app.post("/daily/run")
async def run_daily_dose(current_user: dict = Depends(verify_token)):
    """Manually trigger daily dose generation for the user"""
    from arionxiv.services.unified_daily_dose_service import daily_dose_service
    
    user_id = current_user["user_id"]
    
    try:
        # Execute daily dose generation
        result = await daily_dose_service.execute_daily_dose(user_id)
        
        if result.get("success"):
            # Build a dose object compatible with CLI expectations
            # Prefer a full dose returned by the service if available
            dose = result.get("dose")
            if dose is None:
                dose = {
                    "papers": result.get("papers", []),
                    "summary": result.get("summary", {}),
                    "generated_at": result.get("generated_at"),
                }
            
            return {
                "success": True,
                "message": "Daily dose generated successfully",
                "papers_count": result.get("papers_count", 0),
                "dose_id": result.get("dose_id") or result.get("analysis_id"),
                "dose": dose
            }
        else:
            return {
                "success": False,
                "message": result.get("error", "Failed to generate daily dose")
            }
    except Exception as e:
        logger.error(f"Daily dose generation error: {e}")
        return {
            "success": False,
            "message": f"Error generating daily dose: {str(e)}"
        }


# Embeddings cache endpoints - for avoiding re-processing PDFs
@app.get("/embeddings/{paper_id}")
async def get_embeddings(paper_id: str, current_user: dict = Depends(verify_token)):
    """Get cached embeddings for a paper"""
    db = get_db()
    cached = db.paper_embeddings.find_one({
        "paper_id": paper_id,
        "user_id": current_user["user_id"]
    })
    if cached:
        cached["_id"] = str(cached["_id"])
        return {"success": True, "embeddings": cached.get("embeddings", []), "chunks": cached.get("chunks", [])}
    return {"success": False, "message": "No cached embeddings found"}


@app.post("/embeddings/{paper_id}")
async def save_embeddings(paper_id: str, request: dict, current_user: dict = Depends(verify_token)):
    """Save embeddings for a paper with 24-hour TTL"""
    db = get_db()
    embeddings_data = {
        "paper_id": paper_id,
        "user_id": current_user["user_id"],
        "embeddings": request.get("embeddings", []),
        "chunks": request.get("chunks", []),
        "updated_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24)  # 24-hour TTL
    }
    db.paper_embeddings.update_one(
        {"paper_id": paper_id, "user_id": current_user["user_id"]},
        {"$set": embeddings_data},
        upsert=True
    )
    return {"success": True, "message": "Embeddings saved"}


# Chat message endpoint - LLM inference via OpenRouter with fallback
class ChatMessageRequest(BaseModel):
    message: str
    paper_id: str
    session_id: Optional[str] = None
    context: Optional[str] = None  # RAG context from client (paper chunks)
    paper_title: Optional[str] = None  # Paper title for better context

# Get model from environment variable, defaulting to free models
# User can override with OPENROUTER_MODEL in .env (e.g., openai/gpt-4o-mini for paid tier)
DEFAULT_CHAT_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

# Fast free models for Vercel serverless (10s timeout) - carefully chosen from OpenRouter free tier
FAST_MODELS = [
    "qwen/qwen-2-7b-instruct:free",             # Alternative free model
    "meta-llama/llama-3.1-8b-instruct:free",    # Fallback - larger, still free
    "meta-llama/llama-3.2-3b-instruct:free",    # Primary - fast, free
]

@app.post("/chat/message")
async def send_chat_message(request: ChatMessageRequest, current_user: dict = Depends(verify_token)):
    """Generate AI response using OpenRouter - optimized for Vercel serverless (10s limit)"""
    import requests
    
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(500, detail="Chat service not configured. Please set your own OPENROUTER_API_KEY in 'arionxiv settings api' for chat functionality.")
    
    # Build prompt with paper context if available
    if request.context:
        # Use RAG context from client - this contains relevant paper chunks
        paper_info = f"Paper: {request.paper_title}\n\n" if request.paper_title else ""
        prompt = f"""You are ArionXiv, an AI research assistant. Answer questions about research papers based on the provided context.

{paper_info}RELEVANT SECTIONS FROM THE PAPER:
{request.context}

USER QUESTION: {request.message}

Instructions:
- Answer based ONLY on the provided paper content
- Be specific and cite relevant details from the paper
- If the answer is not in the context, say so clearly
- Be concise but thorough"""
    else:
        # Fallback for requests without context
        prompt = f"Answer this question about a research paper concisely: {request.message}"
    
    # Use configured model first, then fallback to free models
    # Try at most 2 models to leave time for response within Vercel's 10s limit
    models = [DEFAULT_CHAT_MODEL] + [m for m in FAST_MODELS if m != DEFAULT_CHAT_MODEL][:1]
    
    last_error = ""
    for model in models:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "HTTP-Referer": "https://github.com/ArionDas/ArionXiv",
                    "X-Title": "ArionXiv"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,  # Limit response size for speed
                },
                timeout=7  # 7s timeout to fit within Vercel's 10s limit
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return {
                        "success": True,
                        "response": content,
                        "model": model.split("/")[-1].replace(":free", "")
                    }
                last_error = f"{model}: Empty response"
            else:
                last_error = f"{model}: {resp.status_code}"
        except requests.exceptions.Timeout:
            last_error = f"{model}: Timeout"
        except Exception as e:
            last_error = f"{model}: {str(e)[:50]}"
    
    # User-friendly error with guidance
    raise HTTPException(
        503,
        detail=f"Chat service temporarily unavailable (serverless timeout). For reliable chat, set your own OPENROUTER_API_KEY via 'arionxiv settings api'. Error: {last_error}"
    )


