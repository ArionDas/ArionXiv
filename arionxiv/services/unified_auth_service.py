"""
Unified Authentication Service for ArionXiv
Consolidates auth_service.py and auth_utils.py into a single local-auth module
"""

import hashlib
import secrets
import re
import jwt
import os
import hmac
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId

from .unified_database_service import unified_database_service

logger = logging.getLogger(__name__)
security = HTTPBearer()


class UnifiedAuthenticationService:
    """Authentication service for local (password) accounts"""
    
    def __init__(self):
        # Password security settings
        self.password_salt_length = int(os.getenv("PASSWORD_SALT_LENGTH", "32"))
        self.pbkdf2_iterations = int(os.getenv("PBKDF2_ITERATIONS", "100000"))
        self.password_algorithm = os.getenv("PASSWORD_HASH_ALGO", "pbkdf2_sha256")
        self.min_password_length = 8
        self.max_password_length = 128
        self.min_user_name_length = 3
        self.max_user_name_length = 32
        self._user_name_pattern = re.compile(r'^[a-z0-9._-]+$')
        self.session_duration_days = 30
        
        # JWT settings - lazy loaded to allow module import without env vars
        # This is needed for GitHub Actions runner which imports the module before setting env vars
        self._secret_key = None
        self.algorithm = "HS256"
        self.token_expiry_hours = 24
        
        logger.info("UnifiedAuthenticationService initialized")
    
    @property
    def secret_key(self) -> str:
        """Lazy-load JWT secret key - only required when auth methods are actually called."""
        if self._secret_key is None:
            self._secret_key = os.getenv("JWT_SECRET_KEY")
            if not self._secret_key:
                raise ValueError("JWT_SECRET_KEY environment variable is required for security.")
        return self._secret_key

    # ============================================================
    # PASSWORD AUTHENTICATION (from auth_service.py)
    # ============================================================
    
    def _hash_password(self, password: str, salt: bytes = None) -> Tuple[str, str]:
        """
        Hash password with salt

        Purpose: Securely hash a plaintext password using PBKDF2 with SHA-256 and a salt.

        Args:
            password (str): The plaintext password to hash
            salt (bytes, optional): Optional salt. If None, a new salt is generated. Defaults to None.
        """
        if salt is None:
            salt = secrets.token_bytes(self.password_salt_length)
        
        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            self.pbkdf2_iterations
        )
        
        return password_hash.hex(), salt.hex()
    
    def _verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """
        Verify password against stored hash using the stored salt.

        Args:
            password (str): The plaintext password to verify
            stored_hash (str): The stored password hash
            stored_salt (str): The stored salt used for hashing
        """
        try:
            salt = bytes.fromhex(stored_salt)
            password_hash, _ = self._hash_password(password, salt)
            return hmac.compare_digest(password_hash, stored_hash)
        except Exception as e:
            logger.error("Password verification failed: %s", str(e), exc_info=True)
            return False
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """
        Validate password strength

        Args:
            password (str): The plaintext password to validate
        """

        if len(password) < self.min_password_length:
            return False, f"Password must be at least {self.min_password_length} characters long"
        
        if len(password) > self.max_password_length:
            return False, f"Password cannot exceed {self.max_password_length} characters"
        
        # Check for at least one letter and one digit
        if not re.search(r'[a-zA-Z]', password):
            return False, "Password must contain at least one letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        return True, "Password is valid"
    
    def _validate_email(self, email: str) -> bool:
        """
        Validate email format

        Args:
            email (str): The email address to validate
        """
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_user_name(self, user_name: str) -> Tuple[bool, str]:
        """Validate user_name and return normalized value"""
        if not user_name:
            return False, "Username is required"
        normalized = user_name.strip().lower()
        if len(normalized) < self.min_user_name_length or len(normalized) > self.max_user_name_length:
            return False, f"Username must be between {self.min_user_name_length} and {self.max_user_name_length} characters"
        if not self._user_name_pattern.match(normalized):
            return False, "Username can only contain lowercase letters, numbers, dot, underscore, or hyphen"
        return True, normalized
    
    async def register_user(self, email: str, user_name: str, password: str, full_name: str = "") -> Dict[str, Any]:
        """Register a new user with user_name as the primary identifier"""
        try:
            if not self._validate_email(email):
                return {'success': False, 'error': 'Invalid email format'}

            name_valid, normalized_user_name = self._validate_user_name(user_name)
            if not name_valid:
                return {'success': False, 'error': normalized_user_name}

            password_valid, password_message = self._validate_password(password)
            if not password_valid:
                return {'success': False, 'error': password_message}

            # Enforce uniqueness on user_name and best-effort on email
            existing_by_name = await unified_database_service.find_one('users', {'user_name': normalized_user_name})
            if existing_by_name:
                return {'success': False, 'error': 'Username is already taken'}

            existing_by_email = await unified_database_service.find_one('users', {'email': email})
            if existing_by_email:
                return {'success': False, 'error': 'User with this email already exists'}

            password_hash, salt = self._hash_password(password)

            user_data = {
                'email': email,
                'user_name': normalized_user_name,
                'username': normalized_user_name,
                'full_name': full_name,
                'password_hash': password_hash,
                'password_salt': salt,
                'password_algo': self.password_algorithm,
                'created_at': datetime.utcnow(),
                'last_login': None,
                'is_active': True,
                'auth_provider': 'local'
            }

            result = await unified_database_service.insert_one('users', user_data)

            if result and getattr(result, 'inserted_id', None):
                user_payload = {
                    'id': str(result.inserted_id),
                    'email': email,
                    'user_name': normalized_user_name,
                    'full_name': full_name
                }
                return {
                    'success': True,
                    'message': 'User registered successfully',
                    'user': user_payload
                }

            return {'success': False, 'error': 'Failed to create user'}
                
        except Exception as e:
            logger.error("User registration failed: %s", str(e), exc_info=True)
            return {
                'success': False,
                'error': 'Registration failed due to server error'
            }
    
    async def authenticate_user(self, identifier: str, password: str) -> Dict[str, Any]:
        """Authenticate user by user_name or email"""

        try:
            lookup_filter = None
            if self._validate_email(identifier):
                lookup_filter = {'email': identifier}
            else:
                name_valid, normalized_user_name = self._validate_user_name(identifier)
                if not name_valid:
                    return {'success': False, 'error': 'Invalid username or password'}
                lookup_filter = {'user_name': normalized_user_name}
            
            user = await unified_database_service.find_one('users', lookup_filter)
            if not user:
                return {'success': False, 'error': 'Invalid username or password'}
            
            if not user.get('is_active', True):
                return {'success': False, 'error': 'Account is deactivated'}
            
            if user.get('auth_provider') != 'local':
                return {
                    'success': False,
                    'error': 'Please sign in with your linked provider'
                }
            
            if not self._verify_password(password, user['password_hash'], user['password_salt']):
                return {'success': False, 'error': 'Invalid username or password'}
            
            await unified_database_service.update_one(
                'users',
                {'_id': user['_id']},
                {'$set': {'last_login': datetime.utcnow()}}
            )
            
            token = self.create_access_token(user)
            primary_user_name = user.get('user_name') or user.get('username') or user['email'].split('@')[0]
            user_payload = {
                'id': str(user['_id']),
                'email': user['email'],
                'user_name': primary_user_name,
                'full_name': user.get('full_name', '')
            }
            
            return {
                'success': True,
                'message': 'Authentication successful',
                'user': user_payload,
                'token': token
            }
            
        except Exception as e:
            logger.error("User authentication failed: %s", str(e), exc_info=True)
            return {
                'success': False,
                'error': 'Authentication failed due to server error'
            }

    async def login_user(self, identifier: str, password: str) -> Dict[str, Any]:
        """Compatibility wrapper for CLI login flow"""
        return await self.authenticate_user(identifier, password)

    # ============================================================
    # JWT UTILITIES (from auth_utils.py)
    # ============================================================
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """
        Creates JWT access token

        Purpose: Generate a JWT token for authenticated users to access protected resources during the active session.

        Args:
            user_data (Dict[str, Any]): User data to include in the token payload
        
        Returns:
            str: Encoded JWT token
        """
        try:
            payload = {
                "user_id": str(user_data.get("_id", user_data.get("id"))),
                "email": user_data.get("email"),
                "user_name": user_data.get("user_name") or user_data.get("username"),
                "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                "iat": datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            logger.error("Failed to create access token: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail="Token creation failed")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verifies JWT token and return payload

        Args:
            token (str): JWT token to verify
        
        Returns:
            Dict[str, Any]: Verification result with validity and payload or error message
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {"valid": True, "payload": payload}
            
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token has expired"}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": f"Invalid token: {str(e)}"}
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """
        Get current user from token

        Args:
            credentials (HTTPAuthorizationCredentials): Authorization credentials containing the JWT token
        
        Returns:
            Dict[str, Any]: Current user information extracted from the token
        """

        result = self.verify_token(credentials.credentials)
        
        if not result["valid"]:
            raise HTTPException(status_code=401, detail=result["error"])
        
        user_id = result["payload"]["user_id"]
        
        # Get user from database
        try:
            user = await unified_database_service.find_one('users', {'_id': ObjectId(user_id)})
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            return {
                "id": str(user["_id"]),
                "email": user["email"],
                "user_name": user.get("user_name") or user.get("username") or user["email"].split("@")[0]
            }
            
        except Exception as e:
            logger.error("Failed to get current user: %s", str(e), exc_info=True)
            raise HTTPException(status_code=401, detail="Authentication failed")

    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """Get user settings from database"""
        try:
            user_object_id = ObjectId(user_id)
            user = await unified_database_service.find_one('users', {'_id': user_object_id})
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            return {
                'success': True,
                'settings': user.get('settings', {})
            }
        except Exception as e:
            logger.error("Failed to get user settings: %s", str(e), exc_info=True)
            return {
                'success': False,
                'error': 'Failed to retrieve user settings'
            }

    async def update_user_settings(self, user_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update user settings in database"""
        try:
            user_object_id = ObjectId(user_id)
            result = await unified_database_service.update_one(
                'users',
                {'_id': user_object_id},
                {'$set': {'settings': settings}}
            )
            
            # Check if document was matched (user exists)
            matched_count = getattr(result, 'matched_count', None)
            if matched_count is None and isinstance(result, dict):
                matched_count = result.get('matched_count', 0)
            
            modified_count = getattr(result, 'modified_count', None)
            if modified_count is None and isinstance(result, dict):
                modified_count = result.get('modified_count', 0)
            
            # Success if document was found (matched), even if no changes (same values)
            if matched_count and matched_count > 0:
                return {
                    'success': True,
                    'message': 'Settings updated successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'User not found'
                }
        except Exception as e:
            logger.error("Failed to update user settings: %s", str(e), exc_info=True)
            return {
                'success': False,
                'error': 'Failed to update user settings'
            }


# Global instance
unified_auth_service = UnifiedAuthenticationService()

# Backwards compatibility
auth_service = unified_auth_service
auth_utils = unified_auth_service

# Export commonly used functions for convenience
create_access_token = unified_auth_service.create_access_token
verify_token = unified_auth_service.verify_token
get_current_user = unified_auth_service.get_current_user
register_user = unified_auth_service.register_user
authenticate_user = unified_auth_service.authenticate_user
login_user = unified_auth_service.login_user
get_user_settings = unified_auth_service.get_user_settings
update_user_settings = unified_auth_service.update_user_settings

# Make available for imports
__all__ = [
    'UnifiedAuthenticationService',
    'unified_auth_service',
    'auth_service',
    'auth_utils', 
    'create_access_token',
    'verify_token',
    'get_current_user',
    'register_user',
    'authenticate_user',
    'login_user',
    'get_user_settings',
    'update_user_settings',
    'security'
]
