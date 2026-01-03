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
