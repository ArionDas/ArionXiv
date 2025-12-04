"""Global Theme Manager for ArionXiv CLI

This module manages theme state globally across the entire application.
It fetches user theme preferences from the database and ensures consistent
theming throughout all components.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GlobalThemeManager:
    """Global theme manager that maintains theme state across the application"""
    
    _instance = None
    _current_theme = None
    _user_theme_preference = None
    _initialized = False
    _local_config_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._current_theme = None  # Will be loaded on first access
            self._user_theme_preference = None
            self._initialized = True
            self._local_config_loaded = False
    
    def _load_from_local_config_sync(self) -> str:
        """Synchronously load theme from local config file"""
        if self._local_config_loaded and self._current_theme:
            return self._current_theme
        
        try:
            config_file = Path.home() / ".arionxiv" / "local_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    theme_color = config.get("display", {}).get("theme_color", "blue")
                    self._current_theme = theme_color
                    self._local_config_loaded = True
                    logger.debug(f"Theme loaded from local config (sync): {theme_color}")
                    return theme_color
        except Exception as e:
            logger.debug(f"Could not load local config: {e}")
        
        # Default fallback
        self._current_theme = 'blue'
        self._local_config_loaded = True
        return 'blue'
    
    async def initialize_user_theme(self, user_id: str) -> str:
        """Initialize theme from user preferences in database"""
        try:
            # Import here to avoid circular imports
            from arionxiv.services.unified_user_service import unified_user_service
            from arionxiv.services.unified_database_service import unified_database_service
            
            if unified_database_service.db is None:
                logger.debug("Database not available, using local config")
                return await self._load_from_local_config()
            
            # Fetch user preferences from database
            user_result = await unified_database_service.get_user_by_id(user_id)
            if user_result["success"]:
                user_data = user_result["user"]
                preferences = user_data.get("preferences", {})
                theme_color = preferences.get("theme_color", "blue")
                
                self._user_theme_preference = theme_color
                self._current_theme = theme_color
                
                logger.info(f"User theme loaded from database: {theme_color}")
                return theme_color
            else:
                logger.warning("Failed to fetch user from database, using local config")
                return await self._load_from_local_config()
                
        except Exception as e:
            logger.debug(f"Theme initialization fallback to local: {e}")
            return await self._load_from_local_config()
    
    async def _load_from_local_config(self) -> str:
        """Fallback to local config if database is unavailable"""
        try:
            from arionxiv.cli.utils.db_config_manager import db_config_manager
            await db_config_manager.load_config()
            theme_color = db_config_manager.get_theme_color()
            
            self._current_theme = theme_color
            self._local_config_loaded = True
            logger.debug(f"Theme loaded from local config: {theme_color}")
            return theme_color
        except Exception as e:
            logger.debug(f"Using default theme: {e}")
            self._current_theme = 'blue'
            return 'blue'
    
    def get_current_theme(self) -> str:
        """Get the current theme color - loads from local config if not set"""
        if self._current_theme is None:
            # Synchronously load from local config on first access
            return self._load_from_local_config_sync()
        return self._current_theme
    
    def set_theme(self, theme_color: str) -> None:
        """Set the current theme color"""
        self._current_theme = theme_color
        self._user_theme_preference = theme_color
        logger.debug(f"Theme updated to: {theme_color}")
    
    async def update_user_theme_in_database(self, user_id: str, theme_color: str) -> bool:
        """Update user theme preference in database"""
        try:
            from arionxiv.services.unified_database_service import unified_database_service
            
            if unified_database_service.db is None:
                logger.debug("Database not available, saving to local config only")
                return await self._save_to_local_config(theme_color)
            
            # Update user preferences in database
            update_data = {
                "preferences.theme_color": theme_color,
                "updated_at": asyncio.get_event_loop().time()
            }
            
            result = await unified_database_service.update_user(user_id, update_data)
            if result["success"]:
                self.set_theme(theme_color)
                logger.debug(f"User theme updated in database: {theme_color}")
                return True
            else:
                logger.debug("Failed to update user theme in database")
                return False
                
        except Exception as e:
            logger.debug(f"Theme update fallback to local: {e}")
            return await self._save_to_local_config(theme_color)
    
    async def _save_to_local_config(self, theme_color: str) -> bool:
        """Fallback to save in local config"""
        try:
            from arionxiv.cli.utils.db_config_manager import db_config_manager
            await db_config_manager.load_config()
            success = await db_config_manager.set_theme_color(theme_color)
            if success:
                self.set_theme(theme_color)
                logger.debug(f"Theme saved to local config: {theme_color}")
            return success
        except Exception as e:
            logger.debug(f"Error saving to local config: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if theme has been initialized from user preferences"""
        return self._user_theme_preference is not None
    
    def reload_from_config(self) -> str:
        """Force reload theme from local config"""
        self._local_config_loaded = False
        self._current_theme = None
        return self._load_from_local_config_sync()

# Global singleton instance
global_theme_manager = GlobalThemeManager()
