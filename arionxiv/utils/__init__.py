"""
Utility functions for ArionXiv
"""

from .ip_helper import get_public_ip, display_ip_whitelist_help, check_mongodb_connection_error
from .file_cleanup import file_cleanup_manager, FileCleanupManager
from .api_helpers import (
    RegisterRequest,
    LoginRequest,
    RefreshTokenRequest,
    ChatMessageRequest,
    ChatSessionRequest,
    LibraryAddRequest,
    LibraryUpdateRequest,
    PaperSearchRequest,
    APIResponse,
    AuthResponse,
    PaperListResponse,
    ChatResponse,
    create_error_response,
    handle_service_error,
    sanitize_arxiv_id,
    format_user_response,
    paginate_results
)

__all__ = [
    'get_public_ip',
    'display_ip_whitelist_help',
    'check_mongodb_connection_error',
    'file_cleanup_manager',
    'FileCleanupManager',
    # API helpers
    'RegisterRequest',
    'LoginRequest',
    'RefreshTokenRequest',
    'ChatMessageRequest',
    'ChatSessionRequest',
    'LibraryAddRequest',
    'LibraryUpdateRequest',
    'PaperSearchRequest',
    'APIResponse',
    'AuthResponse',
    'PaperListResponse',
    'ChatResponse',
    'create_error_response',
    'handle_service_error',
    'sanitize_arxiv_id',
    'format_user_response',
    'paginate_results'
]
