"""
API Client for ArionXiv CLI
Handles communication with the ArionXiv backend server
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Default API URL - the hosted ArionXiv backend on Vercel
DEFAULT_API_URL = "https://arion-xiv.vercel.app"


class APIClientError(Exception):
    """Custom exception for API client errors"""
    def __init__(self, message: str, status_code: int = None, details: Dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ArionXivAPIClient:
    """
    API client for ArionXiv backend.
    Provides methods for all API endpoints.
    Uses httpx for async HTTP requests.
    """
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("ARIONXIV_API_URL", DEFAULT_API_URL)
        self._token: Optional[str] = None
        self._token_file = Path.home() / ".arionxiv" / "token.json"
        self._httpx_client = None
        self._load_token()
    
    @property
    def httpx_client(self):
        """Lazy load httpx client"""
        if self._httpx_client is None:
            try:
                import httpx
                self._httpx_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=30.0
                )
            except ImportError:
                logger.warning("httpx not installed, API client will not work")
                raise ImportError("httpx is required for API client. Install with: pip install httpx")
        return self._httpx_client
    
    def _load_token(self) -> None:
        """Load stored token from file"""
        try:
            if self._token_file.exists():
                data = json.loads(self._token_file.read_text())
                self._token = data.get("token")
        except Exception as e:
            logger.debug(f"Could not load token: {e}")
    
    def _save_token(self, token: str) -> None:
        """Save token to file"""
        try:
            self._token_file.parent.mkdir(parents=True, exist_ok=True)
            self._token_file.write_text(json.dumps({"token": token}))
            self._token = token
        except Exception as e:
            logger.warning(f"Could not save token: {e}")
    
    def _clear_token(self) -> None:
        """Clear stored token"""
        try:
            if self._token_file.exists():
                self._token_file.unlink()
            self._token = None
        except Exception as e:
            logger.warning(f"Could not clear token: {e}")
    
    def _get_headers(self, auth_required: bool = True) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if auth_required and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    async def _handle_response(self, response) -> Dict[str, Any]:
        """Handle API response"""
        try:
            data = response.json()
        except Exception:
            data = {"message": response.text}
        
        if response.status_code >= 400:
            # Extract error message from various possible formats
            error_msg = data.get("detail", data.get("error", data.get("message", "")))
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", "") or str(error_msg) if error_msg else ""
            if not error_msg or error_msg == "{}":
                error_msg = f"API error {response.status_code}"
            
            raise APIClientError(
                message=str(error_msg),
                status_code=response.status_code,
                details=data
            )
        
        return data
    
    # =========================================================================
    # AUTHENTICATION ENDPOINTS
    # =========================================================================
    
    async def register(
        self,
        email: str,
        user_name: str,
        password: str,
        full_name: str = ""
    ) -> Dict[str, Any]:
        """Register a new user"""
        response = await self.httpx_client.post(
            "/auth/register",
            json={
                "email": email,
                "user_name": user_name,
                "password": password,
                "full_name": full_name
            },
            headers=self._get_headers(auth_required=False)
        )
        return await self._handle_response(response)
    
    async def login(self, identifier: str, password: str) -> Dict[str, Any]:
        """Login user and store token"""
        response = await self.httpx_client.post(
            "/auth/login",
            json={"identifier": identifier, "password": password},
            headers=self._get_headers(auth_required=False)
        )
        data = await self._handle_response(response)
        
        if data.get("success") and data.get("token"):
            self._save_token(data["token"])
        
        return data
    
    async def logout(self) -> Dict[str, Any]:
        """Logout user and clear token"""
        try:
            response = await self.httpx_client.post(
                "/auth/logout",
                headers=self._get_headers()
            )
            await self._handle_response(response)
        except Exception:
            pass  # Logout should always succeed client-side
        
        self._clear_token()
        return {"success": True, "message": "Logged out"}
    
    async def refresh_token(self) -> Dict[str, Any]:
        """Refresh authentication token"""
        if not self._token:
            raise APIClientError("No token to refresh", status_code=401)
        
        response = await self.httpx_client.post(
            "/auth/refresh",
            json={"token": self._token},
            headers=self._get_headers(auth_required=False)
        )
        data = await self._handle_response(response)
        
        if data.get("success") and data.get("token"):
            self._save_token(data["token"])
        
        return data
    
    def is_authenticated(self) -> bool:
        """Check if user has a stored token"""
        return self._token is not None
    
    # =========================================================================
    # USER ENDPOINTS
    # =========================================================================
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get current user profile"""
        response = await self.httpx_client.get(
            "/auth/profile",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def get_settings(self) -> Dict[str, Any]:
        """Get user settings"""
        response = await self.httpx_client.get(
            "/settings",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update user settings"""
        response = await self.httpx_client.put(
            "/settings",
            json=settings,
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    # =========================================================================
    # PAPER ENDPOINTS
    # =========================================================================
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 10,
        category: str = None
    ) -> Dict[str, Any]:
        """Search arXiv papers"""
        params = {"query": query, "max_results": max_results}
        if category:
            params["category"] = category
        
        response = await self.httpx_client.get(
            "/papers/search",
            params=params,
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def fetch_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch a paper from arXiv"""
        response = await self.httpx_client.post(
            f"/papers/{arxiv_id}/fetch",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def analyze_paper(self, paper_id: str) -> Dict[str, Any]:
        """Analyze a paper"""
        response = await self.httpx_client.post(
            f"/papers/{paper_id}/analyze",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def get_user_papers(
        self,
        limit: int = 20,
        skip: int = 0
    ) -> Dict[str, Any]:
        """Get user's papers"""
        response = await self.httpx_client.get(
            "/papers/user",
            params={"limit": limit, "skip": skip},
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    # =========================================================================
    # LIBRARY ENDPOINTS
    # =========================================================================
    
    async def get_library(
        self,
        limit: int = 20,
        skip: int = 0
    ) -> Dict[str, Any]:
        """Get user's library"""
        response = await self.httpx_client.get(
            "/library",
            params={"limit": limit, "skip": skip},
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def add_to_library(
        self,
        arxiv_id: str,
        title: str = "",
        authors: List[str] = None,
        categories: List[str] = None,
        abstract: str = "",
        tags: List[str] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Add paper to library"""
        response = await self.httpx_client.post(
            "/library",
            json={
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors or [],
                "categories": categories or [],
                "abstract": abstract,
                "tags": tags or [],
                "notes": notes
            },
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def remove_from_library(self, arxiv_id: str) -> Dict[str, Any]:
        """Remove paper from library"""
        response = await self.httpx_client.delete(
            f"/library/{arxiv_id}",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def update_library_paper(
        self,
        arxiv_id: str,
        tags: List[str] = None,
        notes: str = None
    ) -> Dict[str, Any]:
        """Update paper in library"""
        response = await self.httpx_client.put(
            f"/library/{arxiv_id}",
            json={"tags": tags, "notes": notes},
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def search_library(self, query: str) -> Dict[str, Any]:
        """Search user's library"""
        response = await self.httpx_client.get(
            "/library/search",
            params={"query": query},
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    # =========================================================================
    # CHAT ENDPOINTS
    # =========================================================================
    
    async def create_chat_session(
        self,
        paper_id: str,
        title: str = None
    ) -> Dict[str, Any]:
        """Create a new chat session"""
        response = await self.httpx_client.post(
            "/chat/session",
            json={"paper_id": paper_id, "title": title},
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def update_chat_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update chat session with new messages"""
        response = await self.httpx_client.put(
            f"/chat/session/{session_id}",
            json=messages,
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def send_chat_message(
        self,
        message: str,
        paper_id: str,
        session_id: str = None,
        context: str = None,
        paper_title: str = None
    ) -> Dict[str, Any]:
        """Send a chat message with optional RAG context"""
        payload = {
            "message": message,
            "paper_id": paper_id,
            "session_id": session_id
        }
        # Include context if provided (for RAG-enhanced responses)
        if context:
            payload["context"] = context
        if paper_title:
            payload["paper_title"] = paper_title
            
        response = await self.httpx_client.post(
            "/chat/message",
            json=payload,
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def get_chat_sessions(self, active_only: bool = True) -> Dict[str, Any]:
        """Get user's chat sessions"""
        response = await self.httpx_client.get(
            "/chat/sessions",
            params={"active_only": active_only},
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def get_chat_session(self, session_id: str) -> Dict[str, Any]:
        """Get a specific chat session"""
        response = await self.httpx_client.get(
            f"/chat/session/{session_id}",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def delete_chat_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a chat session"""
        response = await self.httpx_client.delete(
            f"/chat/session/{session_id}",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    # =========================================================================
    # ANALYSIS ENDPOINTS
    # =========================================================================
    
    async def get_daily_analysis(self, date: str = None) -> Dict[str, Any]:
        """Get daily analysis"""
        params = {}
        if date:
            params["date"] = date
        
        response = await self.httpx_client.get(
            "/daily",
            params=params,
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    async def trigger_daily_analysis(self) -> Dict[str, Any]:
        """Trigger daily analysis"""
        response = await self.httpx_client.post(
            "/analysis/daily/trigger",
            headers=self._get_headers()
        )
        return await self._handle_response(response)
    
    # =========================================================================
    # HEALTH CHECK
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = await self.httpx_client.get("/health")
        return await self._handle_response(response)
    
    async def close(self):
        """Close the HTTP client"""
        if self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None


# Global instance
api_client = ArionXivAPIClient()


# Convenience functions for direct import
async def login(identifier: str, password: str) -> Dict[str, Any]:
    """Login convenience function"""
    return await api_client.login(identifier, password)


async def register(
    email: str,
    user_name: str,
    password: str,
    full_name: str = ""
) -> Dict[str, Any]:
    """Register convenience function"""
    return await api_client.register(email, user_name, password, full_name)


async def logout() -> Dict[str, Any]:
    """Logout convenience function"""
    return await api_client.logout()


def is_authenticated() -> bool:
    """Check authentication status"""
    return api_client.is_authenticated()


__all__ = [
    'ArionXivAPIClient',
    'APIClientError',
    'api_client',
    'login',
    'register',
    'logout',
    'is_authenticated'
]
