"""Enhanced Configuration Manager with Database Integration"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

# Import from parent directory
import sys
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from ..ui.global_theme_manager import global_theme_manager
from ...services.unified_user_service import unified_user_service
from ...services.unified_database_service import unified_database_service
from ...services.unified_auth_service import auth_service

console = Console()

class DatabaseConfigManager:
    """Manages CLI configuration with database integration for authenticated users"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".arionxiv"
        self.local_config_file = self.config_dir / "local_config.json"
        self.config = {}
        self._initialized = False
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
    
    def _ensure_initialized(self):
        """Ensure basic initialization without async calls"""
        if not self._initialized:
            try:
                self.config = self._load_local_config()
                self._initialized = True
            except Exception:
                self.config = self.get_default_config()
                self._initialized = True
    
    async def load_config(self, quiet: bool = True) -> Dict[str, Any]:
        """Load configuration (from database if authenticated, local otherwise)"""
        try:
            # Check if user is authenticated
            if unified_user_service.is_authenticated():
                return await self._load_from_database(quiet=quiet)
            else:
                return self._load_local_config(quiet=quiet)
        except Exception:
            # Silently fall back to defaults - no need to spam console
            return self.get_default_config()
    
    async def _load_from_database(self, quiet: bool = True) -> Dict[str, Any]:
        """Load configuration from database for authenticated user"""
        try:
            # Ensure database connection
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            user = unified_user_service.get_current_user()
            if not user:
                return self._load_local_config()
            
            # Get settings from database
            result = await auth_service.get_user_settings(user["id"])
            
            if result.get("success"):
                settings = result.get("settings", {})
                
                # Convert database format to local config format
                self.config = {
                    "user": {
                        "id": user["id"],
                        "user_name": user["user_name"],
                        "email": user["email"],
                        "full_name": user["full_name"],
                        "preferences": settings.get("preferences", {})
                    },
                    "display": {
                        "theme_color": settings.get("theme_color", "blue"),
                        **settings.get("display", {})
                    },
                    "paths": {
                        "downloads": str(self.config_dir / "downloads"),
                        "data": str(self.config_dir / "data"),
                        "cache": str(self.config_dir / "data" / "cache")
                    },
                    "first_time_user": False,
                    "database_mode": True
                }
                
                return self.config
            else:
                # Fallback to local config if database fails
                return self._load_local_config()
                
        except Exception:
            # Silently fall back to local config
            return self._load_local_config(quiet=quiet)
    
    def _load_local_config(self, quiet: bool = True) -> Dict[str, Any]:
        """Load configuration from local file"""
        try:
            if self.local_config_file.exists():
                with open(self.local_config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self.get_default_config()
                self._save_local_config()
            
            self.config["database_mode"] = False
            return self.config
            
        except Exception:
            # Silently fall back to defaults
            self.config = self.get_default_config()
            self.config["database_mode"] = False
            return self.config
    
    def _save_local_config(self) -> bool:
        """Save configuration to local file"""
        try:
            with open(self.local_config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            console.print(f"[red]Error saving local config: {e}[/red]")
            return False
    
    async def save_config(self) -> bool:
        """Save configuration (to database if authenticated, local otherwise)"""
        try:
            if unified_user_service.is_authenticated():
                return await self._save_to_database()
            else:
                return self._save_local_config()
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
            return False
    
    async def _save_to_database(self) -> bool:
        """Save configuration to database for authenticated user"""
        try:
            # Always save to local config as well for offline/new-process access
            self._save_local_config()
            
            # Ensure database connection
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            user = unified_user_service.get_current_user()
            if not user:
                return True  # Local already saved
            
            # Convert local config format to database format
            settings_update = {
                "theme_color": self.config.get("display", {}).get("theme_color", "blue"),
                "preferences": self.config.get("user", {}).get("preferences", {}),
                "display": {k: v for k, v in self.config.get("display", {}).items() if k != "theme_color"}
            }
            
            # Save to database
            result = await auth_service.update_user_settings(user["id"], settings_update)
            
            if result.get("success"):
                return True
            else:
                error_msg = result.get('error') or result.get('message') or 'Unknown error'
                console.print(f"[yellow]Database sync failed: {error_msg}[/yellow]")
                return True  # Local already saved, so return True
                
        except Exception as e:
            # Local config was already saved at the beginning of the method
            console.print(f"[yellow]Database sync failed: {e}[/yellow]")
            return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "user": {
                "id": "",
                "user_name": "",
                "email": "",
                "full_name": "",
                "preferences": {
                    "categories": ["cs.AI", "cs.LG", "cs.CL"],
                    "keywords": [],
                    "max_daily_papers": 10,
                    "analysis_depth": "standard",
                    "auto_download": False,
                    "email_notifications": False
                }
            },
            "display": {
                "theme": "auto",
                "theme_color": "blue",
                "table_style": "grid",
                "show_abstracts": True,
                "max_abstract_length": 200,
                "papers_per_page": 10
            },
            "paths": {
                "downloads": str(self.config_dir / "downloads"),
                "data": str(self.config_dir / "data"),
                "cache": str(self.config_dir / "data" / "cache")
            },
            "first_time_user": True,
            "database_mode": False
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        self._ensure_initialized()
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    async def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        return await self.save_config()
    
    async def update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        if "user" not in self.config:
            self.config["user"] = {}
        if "preferences" not in self.config["user"]:
            self.config["user"]["preferences"] = {}
        
        self.config["user"]["preferences"].update(preferences)
        return await self.save_config()
    
    def is_theme_configured(self) -> bool:
        """Check if theme color has been configured by user"""
        # If authenticated, theme is always considered configured
        if unified_user_service.is_authenticated():
            return True
        return self.get("display.theme_color_configured", False)
    
    async def set_theme_color(self, color: str) -> bool:
        """Set the theme color and mark as configured"""
        success = await self.set("display.theme_color", color)
        if success:
            # Update global theme manager
            try:
                global_theme_manager.set_theme(color)
            except Exception:
                pass  # Silently ignore if theme manager not available
            
            if not unified_user_service.is_authenticated():
                # Only set configured flag for local config
                success = await self.set("display.theme_color_configured", True)
        return success
    
    def get_theme_color(self) -> str:
        """Get the current theme color"""
        self._ensure_initialized()
        color = self.get("display.theme_color", "blue")
        
        # Sync with global theme manager
        try:
            current = global_theme_manager.get_current_theme()
            if current != color:
                global_theme_manager.set_theme(color)
        except Exception:
            pass  # Silently ignore if theme manager not available
        
        return color
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        try:
            return unified_user_service.is_authenticated()
        except Exception:
            return False
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user"""
        try:
            return unified_user_service.get_current_user()
        except Exception:
            return None
    
    def is_database_mode(self) -> bool:
        """Check if running in database mode"""
        self._ensure_initialized()
        return self.get("database_mode", False)
    
    async def reset_config(self) -> bool:
        """Reset configuration to defaults"""
        if unified_user_service.is_authenticated():
            # Reset database settings
            try:
                # Ensure database connection
                if unified_database_service.db is None:
                    await unified_database_service.connect_mongodb()
                    
                user = unified_user_service.get_current_user()
                if user:
                    default_settings = {
                        "theme_color": "blue",
                        "preferences": {
                            "categories": ["cs.AI", "cs.LG", "cs.CL"],
                            "keywords": [],
                            "max_daily_papers": 10,
                            "analysis_depth": "standard",
                            "auto_download": False,
                            "email_notifications": False
                        },
                        "display": {
                            "theme": "auto",
                            "table_style": "grid",
                            "show_abstracts": True,
                            "max_abstract_length": 200,
                            "papers_per_page": 10
                        }
                    }
                    
                    result = await auth_service.update_user_settings(user["id"], default_settings)
                    if result.get("success"):
                        self.config = self.get_default_config()
                        self.config["database_mode"] = True
                        self.config["user"].update(user)
                        return True
            except Exception as e:
                console.print(f"[yellow]Database reset failed ({e}), resetting locally[/yellow]")
        
        # Reset local config
        self.config = self.get_default_config()
        return self._save_local_config()

# Global configuration manager instance
db_config_manager = DatabaseConfigManager()

# For backward compatibility
def DbConfigManager():
    """Factory function to get config manager"""
    return db_config_manager

# Async wrapper for backward compatibility
def load_config_sync():
    """Synchronous wrapper for loading config"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(db_config_manager.load_config())
    except:
        return db_config_manager.get_default_config()

# Helper function for synchronous access to default config
def get_default_config():
    """Get default config synchronously"""
    return db_config_manager.get_default_config()
