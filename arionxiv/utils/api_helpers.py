"""
API Helper utilities for ArionXiv server
Shared response models, error handlers, and common API utilities
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class RegisterRequest(BaseModel):
    """User registration request model"""
    email: str = Field(..., description="User email address")
    user_name: str = Field(..., min_length=3, max_length=32, description="Username")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(default="", description="Full name")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v.lower().strip()
    
    @field_validator('user_name')
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r'^[a-z0-9._-]+$', v):
            raise ValueError('Username can only contain lowercase letters, numbers, dot, underscore, or hyphen')
        return v


class LoginRequest(BaseModel):
    """User login request model"""
    identifier: str = Field(..., description="Email or username")
    password: str = Field(..., description="Password")


class RefreshTokenRequest(BaseModel):
    """Token refresh request model"""
    token: str = Field(..., description="Current JWT token to refresh")


class ChatMessageRequest(BaseModel):
    """Chat message request model"""
    message: str = Field(..., min_length=1, description="User message")
    paper_id: str = Field(..., description="ArXiv paper ID for context")
    session_id: Optional[str] = Field(default=None, description="Existing chat session ID")
    

class ChatSessionRequest(BaseModel):
    """Create chat session request model"""
    paper_id: str = Field(..., description="ArXiv paper ID")
    title: Optional[str] = Field(default=None, description="Session title")


class LibraryAddRequest(BaseModel):
    """Add paper to library request model"""
    arxiv_id: str = Field(..., description="ArXiv paper ID")
    tags: Optional[List[str]] = Field(default=None, description="Tags for the paper")
    notes: Optional[str] = Field(default=None, description="Personal notes")


class LibraryUpdateRequest(BaseModel):
    """Update library paper request model"""
    tags: Optional[List[str]] = Field(default=None, description="Updated tags")
    notes: Optional[str] = Field(default=None, description="Updated notes")


class PaperSearchRequest(BaseModel):
    """Paper search request model"""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=10, ge=1, le=100, description="Max results")
    category: Optional[str] = Field(default=None, description="ArXiv category filter")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AuthResponse(BaseModel):
    """Authentication response model"""
    success: bool
    message: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    token: Optional[str] = None
    error: Optional[str] = None


class PaperListResponse(BaseModel):
    """Paper list response model"""
    papers: List[Dict[str, Any]]
    count: int
    total: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    sources: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# ERROR HANDLING
# =============================================================================

def create_error_response(
    status_code: int,
    detail: str,
    error_type: str = "APIError"
) -> HTTPException:
    """Create standardized HTTP exception"""
    logger.error(f"{error_type}: {detail}")
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error_type,
            "message": detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def handle_service_error(result: Dict[str, Any], operation: str) -> None:
    """Handle service layer errors and raise appropriate HTTP exceptions"""
    if not result.get("success", False):
        error_msg = result.get("error") or result.get("message") or f"{operation} failed"
        logger.error(f"{operation} failed: {error_msg}")
        
        # Map common errors to status codes
        error_lower = error_msg.lower()
        if "not found" in error_lower:
            raise create_error_response(404, error_msg, "NotFoundError")
        elif "already exists" in error_lower or "already taken" in error_lower:
            raise create_error_response(409, error_msg, "ConflictError")
        elif "invalid" in error_lower or "required" in error_lower:
            raise create_error_response(400, error_msg, "ValidationError")
        elif "unauthorized" in error_lower or "authentication" in error_lower:
            raise create_error_response(401, error_msg, "AuthenticationError")
        else:
            raise create_error_response(500, error_msg, "InternalError")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Import sanitize_arxiv_id from the consolidated llm_utils module
from ..services.llm_inference.llm_utils import sanitize_arxiv_id


def format_user_response(user: Dict[str, Any]) -> Dict[str, Any]:
    """Format user data for API response (remove sensitive fields)"""
    return {
        "id": str(user.get("_id", user.get("id", ""))),
        "email": user.get("email", ""),
        "user_name": user.get("user_name") or user.get("username", ""),
        "full_name": user.get("full_name", ""),
        "created_at": user.get("created_at", ""),
        "last_login": user.get("last_login", "")
    }


def paginate_results(
    items: List[Any],
    skip: int = 0,
    limit: int = 20
) -> Dict[str, Any]:
    """Paginate a list of items"""
    total = len(items)
    paginated = items[skip:skip + limit]
    return {
        "items": paginated,
        "count": len(paginated),
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total
    }
