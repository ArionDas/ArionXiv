"""
FastAPI server for ArionXiv package
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
import logging

from .services.unified_database_service import unified_database_service
from .services.unified_auth_service import unified_auth_service, verify_token, security
from .services.unified_paper_service import unified_paper_service
from .services.unified_user_service import unified_user_service
from .services.unified_analysis_service import unified_analysis_service, rag_chat_system
from .arxiv_operations.client import ArxivClient
from .arxiv_operations.searcher import ArxivSearcher
from .arxiv_operations.fetcher import ArxivFetcher
from .utils.api_helpers import (
    RegisterRequest,
    LoginRequest,
    RefreshTokenRequest,
    ChatMessageRequest,
    ChatSessionRequest,
    LibraryAddRequest,
    LibraryUpdateRequest,
    PaperSearchRequest,
    create_error_response,
    handle_service_error,
    sanitize_arxiv_id,
    format_user_response
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ArionXiv API",
    description="AI-powered research paper ingestion pipeline with user authentication and daily analysis",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
arxiv_client = ArxivClient()
arxiv_searcher = ArxivSearcher()
arxiv_fetcher = ArxivFetcher()

# Startup event to initialize database connections
@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    try:
        logger.info("Starting ArionXiv API server")
        await asyncio.wait_for(unified_database_service.connect_mongodb(), timeout=10.0)
        logger.info("ArionXiv API server started successfully")
    except asyncio.TimeoutError:
        logger.error("Database connection timeout during startup")
        raise Exception("Database connection timeout")
    except Exception as e:
        logger.error(f"Failed to start ArionXiv API server: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    try:
        await unified_database_service.disconnect()
        logger.info("ArionXiv API server shut down gracefully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to ArionXiv API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post("/auth/register")
async def register_user(request: RegisterRequest):
    """Register a new user account"""
    try:
        logger.info(f"Registration attempt for: {request.email}")
        result = await unified_auth_service.register_user(
            email=request.email,
            user_name=request.user_name,
            password=request.password,
            full_name=request.full_name or ""
        )
        
        if not result.get("success"):
            handle_service_error(result, "Registration")
        
        logger.info(f"User registered successfully: {request.user_name}")
        return {
            "success": True,
            "message": "User registered successfully",
            "user": result.get("user")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        raise create_error_response(500, "Registration failed", "InternalError")


@app.post("/auth/login")
async def login_user(request: LoginRequest):
    """Authenticate user and return JWT token"""
    try:
        logger.info(f"Login attempt for: {request.identifier}")
        result = await unified_auth_service.authenticate_user(
            identifier=request.identifier,
            password=request.password
        )
        
        if not result.get("success"):
            handle_service_error(result, "Authentication")
        
        logger.info(f"User logged in successfully: {request.identifier}")
        return {
            "success": True,
            "message": "Login successful",
            "user": result.get("user"),
            "token": result.get("token")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}", exc_info=True)
        raise create_error_response(500, "Login failed", "InternalError")


@app.post("/auth/logout")
async def logout_user(current_user: Dict = Depends(verify_token)):
    """Logout user (client should discard token)"""
    logger.info(f"User logged out: {current_user.get('email')}")
    return {
        "success": True,
        "message": "Logged out successfully"
    }


@app.post("/auth/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """Refresh JWT token"""
    try:
        result = unified_auth_service.verify_token(request.token)
        
        if not result.get("valid"):
            raise create_error_response(401, result.get("error", "Invalid token"), "AuthenticationError")
        
        payload = result.get("payload", {})
        user_data = {
            "_id": payload.get("user_id"),
            "email": payload.get("email"),
            "user_name": payload.get("user_name")
        }
        
        new_token = unified_auth_service.create_access_token(user_data)
        
        return {
            "success": True,
            "message": "Token refreshed",
            "token": new_token
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}", exc_info=True)
        raise create_error_response(500, "Token refresh failed", "InternalError")


# =============================================================================
# USER ENDPOINTS
# =============================================================================

@app.get("/user/profile")
async def get_user_profile(current_user: Dict = Depends(verify_token)):
    """Get current user profile"""
    try:
        logger.debug(f"Fetching profile for user: {current_user['email']}")
        user = await unified_user_service.get_user_by_email(current_user["email"])
        if not user:
            logger.warning(f"User not found: {current_user['email']}")
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.debug(f"Profile fetched for user: {current_user['email']}")
        return {"success": True, "user": format_user_response(user)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.put("/user/settings")
async def update_user_settings(
    settings: Dict[str, Any],
    current_user: Dict = Depends(verify_token)
):
    """Update user settings"""
    try:
        result = await unified_auth_service.update_user_settings(
            user_id=current_user.get("id") or current_user.get("user_id"),
            settings=settings
        )
        
        if not result.get("success"):
            handle_service_error(result, "Settings update")
        
        return {"success": True, "message": "Settings updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update settings")


@app.get("/user/settings")
async def get_user_settings(current_user: Dict = Depends(verify_token)):
    """Get user settings"""
    try:
        result = await unified_auth_service.get_user_settings(
            user_id=current_user.get("id") or current_user.get("user_id")
        )
        
        if not result.get("success"):
            handle_service_error(result, "Get settings")
        
        return {"success": True, "settings": result.get("settings", {})}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get settings")


# =============================================================================
# PAPER MANAGEMENT ENDPOINTS
# =============================================================================

# Paper Management Endpoints
@app.get("/papers/search")
async def search_papers(
    query: str,
    max_results: int = 10,
    category: Optional[str] = None,
    current_user: Dict = Depends(verify_token)
):
    """Search for papers on arXiv"""
    try:
        logger.info(f"Searching papers: query='{query}', max_results={max_results}, category={category}")
        papers = await arxiv_searcher.search_papers(
            query=query,
            max_results=max_results,
            category=category
        )
        logger.info(f"Paper search completed: {len(papers)} results found")
        return {
            "papers": papers,
            "count": len(papers),
            "query": query
        }
    except Exception as e:
        logger.error(f"Paper search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/papers/{arxiv_id}/fetch")
async def fetch_paper(
    arxiv_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Fetch and store a specific paper"""
    try:
        logger.info(f"Fetching paper: {arxiv_id}")
        paper = await arxiv_fetcher.fetch_paper(arxiv_id)
        
        if not paper:
            logger.warning(f"Paper not found: {arxiv_id}")
            raise HTTPException(status_code=404, detail="Paper not found")
        
        stored_paper = await unified_paper_service.store_paper(paper)
        logger.info(f"Paper stored successfully: {arxiv_id}")
        
        return {
            "message": "Paper fetched successfully",
            "paper": stored_paper
        }
    except Exception as e:
        logger.error(f"Paper fetch failed for {arxiv_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch paper")

@app.post("/papers/{paper_id}/analyze")
async def analyze_paper(
    paper_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Analyze a stored paper"""
    try:
        logger.info(f"Starting paper analysis: {paper_id}")
        paper = await unified_paper_service.get_paper_by_id(paper_id)
        
        if not paper:
            logger.warning(f"Paper not found for analysis: {paper_id}")
            raise HTTPException(status_code=404, detail="Paper not found")
        
        analysis = await unified_analysis_service.analyze_paper(paper)
        logger.info(f"Paper analysis completed: {paper_id}")
        
        return {
            "success": True,
            "message": "Paper analyzed successfully",
            "analysis": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper analysis failed for {paper_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed")


# =============================================================================
# LIBRARY MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/library")
async def get_library(
    current_user: Dict = Depends(verify_token),
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0)
):
    """Get user's paper library"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        logger.debug(f"Fetching library for user: {user_name}")
        
        papers = await unified_database_service.get_user_papers(user_name)
        
        total = len(papers)
        paginated = papers[skip:skip + limit]
        
        return {
            "success": True,
            "papers": paginated,
            "count": len(paginated),
            "total": total,
            "has_more": skip + limit < total
        }
    except Exception as e:
        logger.error(f"Failed to get library: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve library")


@app.post("/library/add")
async def add_to_library(
    request: LibraryAddRequest,
    current_user: Dict = Depends(verify_token)
):
    """Add a paper to user's library"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        arxiv_id = sanitize_arxiv_id(request.arxiv_id)
        
        logger.info(f"Adding paper {arxiv_id} to library for user: {user_name}")
        
        # Fetch paper metadata from arXiv
        paper_data = await arxiv_fetcher.fetch_paper(arxiv_id)
        if not paper_data:
            raise HTTPException(status_code=404, detail="Paper not found on arXiv")
        
        # Add to user's library
        library_entry = {
            "arxiv_id": arxiv_id,
            "user_name": user_name,
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", []),
            "abstract": paper_data.get("abstract", ""),
            "categories": paper_data.get("categories", []),
            "tags": request.tags or [],
            "notes": request.notes or "",
            "added_at": datetime.utcnow(),
            "paper_data": paper_data
        }
        
        result = await unified_database_service.insert_one("user_library", library_entry)
        
        if result:
            logger.info(f"Paper added to library: {arxiv_id}")
            return {
                "success": True,
                "message": "Paper added to library",
                "paper": {
                    "arxiv_id": arxiv_id,
                    "title": paper_data.get("title", "")
                }
            }
        
        raise HTTPException(status_code=500, detail="Failed to add paper to library")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add to library: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add paper to library")


@app.delete("/library/{arxiv_id}")
async def remove_from_library(
    arxiv_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Remove a paper from user's library"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        arxiv_id = sanitize_arxiv_id(arxiv_id)
        
        logger.info(f"Removing paper {arxiv_id} from library for user: {user_name}")
        
        result = await unified_database_service.delete_one(
            "user_library",
            {"arxiv_id": arxiv_id, "user_name": user_name}
        )
        
        if result and getattr(result, 'deleted_count', 0) > 0:
            return {"success": True, "message": "Paper removed from library"}
        
        raise HTTPException(status_code=404, detail="Paper not found in library")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove from library: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to remove paper")


@app.put("/library/{arxiv_id}")
async def update_library_paper(
    arxiv_id: str,
    request: LibraryUpdateRequest,
    current_user: Dict = Depends(verify_token)
):
    """Update paper metadata in library (tags, notes)"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        arxiv_id = sanitize_arxiv_id(arxiv_id)
        
        update_data = {}
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.notes is not None:
            update_data["notes"] = request.notes
        
        if not update_data:
            return {"success": True, "message": "No updates provided"}
        
        update_data["updated_at"] = datetime.utcnow()
        
        result = await unified_database_service.update_one(
            "user_library",
            {"arxiv_id": arxiv_id, "user_name": user_name},
            {"$set": update_data}
        )
        
        matched = getattr(result, 'matched_count', 0)
        if matched > 0:
            return {"success": True, "message": "Paper updated"}
        
        raise HTTPException(status_code=404, detail="Paper not found in library")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update library paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update paper")


@app.get("/library/search")
async def search_library(
    query: str = Query(..., min_length=1),
    current_user: Dict = Depends(verify_token)
):
    """Search user's library"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        
        # Search in title, abstract, tags, notes
        search_filter = {
            "user_name": user_name,
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"abstract": {"$regex": query, "$options": "i"}},
                {"tags": {"$regex": query, "$options": "i"}},
                {"notes": {"$regex": query, "$options": "i"}}
            ]
        }
        
        papers = await unified_database_service.find_many("user_library", search_filter)
        
        return {
            "success": True,
            "papers": papers or [],
            "count": len(papers) if papers else 0,
            "query": query
        }
    except Exception as e:
        logger.error(f"Library search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")


# =============================================================================
# CHAT / RAG ENDPOINTS
# =============================================================================

@app.post("/chat/session")
async def create_chat_session(
    request: ChatSessionRequest,
    current_user: Dict = Depends(verify_token)
):
    """Create a new chat session for a paper"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        arxiv_id = sanitize_arxiv_id(request.paper_id)
        
        logger.info(f"Creating chat session for paper {arxiv_id}, user: {user_name}")
        
        session_data = {
            "user_name": user_name,
            "arxiv_id": arxiv_id,
            "title": request.title or f"Chat about {arxiv_id}",
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "messages": [],
            "is_active": True
        }
        
        result = await unified_database_service.insert_one("chat_sessions", session_data)
        
        if result and getattr(result, 'inserted_id', None):
            return {
                "success": True,
                "message": "Chat session created",
                "session_id": str(result.inserted_id),
                "arxiv_id": arxiv_id
            }
        
        raise HTTPException(status_code=500, detail="Failed to create session")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create chat session")


@app.post("/chat/message")
async def send_chat_message(
    request: ChatMessageRequest,
    current_user: Dict = Depends(verify_token)
):
    """Send a message and get AI response using RAG"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        arxiv_id = sanitize_arxiv_id(request.paper_id)
        
        logger.info(f"Chat message for paper {arxiv_id} from user: {user_name}")
        
        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_data = {
                "user_name": user_name,
                "arxiv_id": arxiv_id,
                "title": f"Chat about {arxiv_id}",
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "messages": [],
                "is_active": True
            }
            result = await unified_database_service.insert_one("chat_sessions", session_data)
            session_id = str(result.inserted_id) if result and getattr(result, 'inserted_id', None) else None
            if not session_id:
                raise HTTPException(status_code=500, detail="Failed to create session")
        
        # Get RAG response
        response_data = await rag_chat_system.chat(
            user_name=user_name,
            paper_id=arxiv_id,
            message=request.message,
            session_id=session_id
        )
        
        # Store message in session
        message_entry = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow()
        }
        assistant_entry = {
            "role": "assistant",
            "content": response_data.get("response", ""),
            "timestamp": datetime.utcnow(),
            "sources": response_data.get("sources", [])
        }
        
        from bson import ObjectId
        await unified_database_service.update_one(
            "chat_sessions",
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": {"$each": [message_entry, assistant_entry]}},
                "$set": {"last_activity": datetime.utcnow()}
            }
        )
        
        return {
            "success": True,
            "response": response_data.get("response", ""),
            "session_id": session_id,
            "sources": response_data.get("sources", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat failed")


@app.get("/chat/sessions")
async def get_chat_sessions(
    current_user: Dict = Depends(verify_token),
    active_only: bool = Query(default=True)
):
    """Get user's chat sessions"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        
        filter_query = {"user_name": user_name}
        if active_only:
            filter_query["is_active"] = True
        
        sessions = await unified_database_service.find_many("chat_sessions", filter_query)
        
        # Format sessions for response
        formatted = []
        for s in (sessions or []):
            formatted.append({
                "session_id": str(s.get("_id", "")),
                "arxiv_id": s.get("arxiv_id", ""),
                "title": s.get("title", ""),
                "created_at": s.get("created_at", ""),
                "last_activity": s.get("last_activity", ""),
                "message_count": len(s.get("messages", []))
            })
        
        return {
            "success": True,
            "sessions": formatted,
            "count": len(formatted)
        }
    except Exception as e:
        logger.error(f"Failed to get chat sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get sessions")


@app.get("/chat/session/{session_id}")
async def get_chat_session(
    session_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Get a specific chat session with messages"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        
        from bson import ObjectId
        session = await unified_database_service.find_one(
            "chat_sessions",
            {"_id": ObjectId(session_id), "user_name": user_name}
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session": {
                "session_id": str(session.get("_id", "")),
                "arxiv_id": session.get("arxiv_id", ""),
                "title": session.get("title", ""),
                "messages": session.get("messages", []),
                "created_at": session.get("created_at", ""),
                "last_activity": session.get("last_activity", "")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get session")


@app.put("/chat/session/{session_id}")
async def update_chat_session(
    session_id: str,
    messages: List[Dict[str, Any]],
    current_user: Dict = Depends(verify_token)
):
    """Update chat session with new messages"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        
        from bson import ObjectId
        # Verify session belongs to user
        session = await unified_database_service.find_one(
            "chat_sessions",
            {"_id": ObjectId(session_id), "user_name": user_name}
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update messages and last_activity
        result = await unified_database_service.update_one(
            "chat_sessions",
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    "messages": messages,
                    "last_activity": datetime.utcnow()
                }
            }
        )
        
        if result and getattr(result, 'modified_count', 0) > 0:
            return {
                "success": True,
                "message": "Session updated",
                "message_count": len(messages)
            }
        
        # Even if no modification (same data), return success
        return {
            "success": True,
            "message": "Session update processed",
            "message_count": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update session")


@app.delete("/chat/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: Dict = Depends(verify_token)
):
    """Delete a chat session"""
    try:
        user_name = current_user.get("user_name") or current_user.get("email", "").split("@")[0]
        
        from bson import ObjectId
        result = await unified_database_service.delete_one(
            "chat_sessions",
            {"_id": ObjectId(session_id), "user_name": user_name}
        )
        
        if result and getattr(result, 'deleted_count', 0) > 0:
            return {"success": True, "message": "Session deleted"}
        
        raise HTTPException(status_code=404, detail="Session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session")


# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================

@app.get("/papers/user")
async def get_user_papers(
    current_user: Dict = Depends(verify_token),
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0)
):
    """Get papers associated with current user"""
    try:
        logger.debug(f"Fetching papers for user: {current_user['email']} (limit={limit}, skip={skip})")
        papers = await unified_paper_service.get_user_papers(
            user_email=current_user["email"],
            limit=limit,
            skip=skip
        )
        logger.debug(f"Retrieved {len(papers)} papers for user: {current_user['email']}")
        
        return {
            "success": True,
            "papers": papers,
            "count": len(papers)
        }
    except Exception as e:
        logger.error(f"Failed to get user papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve papers")

# Daily Analysis Endpoints
@app.get("/analysis/daily")
async def get_daily_analysis(
    date: Optional[str] = Query(default=None),
    current_user: Dict = Depends(verify_token)
):
    """Get daily analysis for a specific date"""
    try:
        if date:
            analysis_date = datetime.fromisoformat(date)
        else:
            analysis_date = datetime.utcnow()
        
        logger.info(f"Fetching daily analysis for date: {analysis_date.isoformat()}")
        analysis = await unified_analysis_service.get_daily_analysis(analysis_date)
        logger.debug(f"Daily analysis retrieved for: {analysis_date.isoformat()}")
        
        return {
            "success": True,
            "analysis": analysis,
            "date": analysis_date.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get daily analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")


@app.get("/daily/dose")
async def get_daily_dose(current_user: Dict = Depends(verify_token)):
    """Get user's latest daily dose"""
    try:
        user_id = current_user.get("id") or current_user.get("user_id") or str(current_user.get("_id", ""))
        
        # Get the latest daily dose from the database
        daily_dose = await unified_database_service.find_one(
            "daily_dose",
            {"user_id": user_id},
            sort=[("generated_at", -1)]
        )
        
        if not daily_dose:
            return {
                "success": False,
                "message": "No daily dose found. Generate one first.",
                "dose": None
            }
        
        # Convert ObjectId to string
        if "_id" in daily_dose:
            daily_dose["_id"] = str(daily_dose["_id"])
        
        return {
            "success": True,
            "dose": daily_dose
        }
    except Exception as e:
        logger.error(f"Failed to get daily dose: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve daily dose")


@app.post("/daily/run")
async def run_daily_dose(current_user: Dict = Depends(verify_token)):
    """Run daily dose analysis for the user"""
    try:
        from .services.unified_daily_dose_service import unified_daily_dose_service
        
        user_id = current_user.get("id") or current_user.get("user_id") or str(current_user.get("_id", ""))
        
        logger.info(f"Running daily dose for user {user_id}")
        
        result = await unified_daily_dose_service.execute_daily_dose(user_id=user_id)
        
        return result
    except Exception as e:
        logger.error(f"Failed to run daily dose: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to run daily dose")


@app.get("/daily/settings")
async def get_daily_dose_settings(current_user: Dict = Depends(verify_token)):
    """Get user's daily dose settings"""
    try:
        from .services.unified_daily_dose_service import unified_daily_dose_service
        
        user_id = current_user.get("id") or current_user.get("user_id") or str(current_user.get("_id", ""))
        
        result = await unified_daily_dose_service.get_user_daily_dose_settings(user_id)
        
        return result
    except Exception as e:
        logger.error(f"Failed to get daily dose settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@app.put("/daily/settings")
async def update_daily_dose_settings(
    settings: Dict[str, Any],
    current_user: Dict = Depends(verify_token)
):
    """Update user's daily dose settings"""
    try:
        from .services.unified_daily_dose_service import unified_daily_dose_service
        
        user_id = current_user.get("id") or current_user.get("user_id") or str(current_user.get("_id", ""))
        
        result = await unified_daily_dose_service.update_user_daily_dose_settings(
            user_id=user_id,
            keywords=settings.get("keywords"),
            max_papers=settings.get("max_papers"),
            scheduled_time=settings.get("scheduled_time"),
            enabled=settings.get("enabled")
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to update daily dose settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update settings")


@app.post("/analysis/daily/trigger")
async def trigger_daily_analysis(current_user: Dict = Depends(verify_token)):
    """Manually trigger daily analysis"""
    try:
        logger.info("Manually triggering daily analysis")
        analysis_task = asyncio.create_task(
            unified_analysis_service.run_daily_analysis()
        )
        logger.info("Daily analysis task started")
        
        return {
            "success": True,
            "message": "Daily analysis triggered",
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Failed to trigger daily analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to trigger analysis")


async def main():
    """Main function to run the server"""
    import uvicorn
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    logger.info("Starting ArionXiv API server on http://0.0.0.0:8000")
    logger.info("API Documentation: http://0.0.0.0:8000/docs")
    logger.info("Health Check: http://0.0.0.0:8000/health")
    
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())