"""
Unified Prompt Service for ArionXiv
Manages LLM prompts from MongoDB with on-demand loading and TTL caching
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from .unified_database_service import unified_database_service

logger = logging.getLogger(__name__)


class UnifiedPromptService:
    """
    Service for managing LLM prompts stored in MongoDB.
    Prompts are loaded on-demand and cached with TTL.
    """
    
    def __init__(self):
        # Admin username for prompt management
        self.admin_user_name = os.getenv("ADMIN_USER_NAME", "ariondas")
        
        # Cache for prompts with TTL - stores {prompt_name: (template, expires_at)}
        self._prompt_cache: Dict[str, Tuple[str, datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes TTL
    
    def _is_admin(self, user_name: str) -> bool:
        """Check if user is admin"""
        if not user_name:
            return False
        return user_name.lower() == self.admin_user_name.lower()
    
    def _is_cached(self, prompt_name: str) -> bool:
        """Check if prompt is cached and not expired"""
        if prompt_name not in self._prompt_cache:
            return False
        
        _, expires_at = self._prompt_cache[prompt_name]
        return datetime.utcnow() < expires_at
    
    def _get_from_cache(self, prompt_name: str) -> Optional[str]:
        """Get prompt from cache if valid"""
        if self._is_cached(prompt_name):
            template, _ = self._prompt_cache[prompt_name]
            return template
        return None
    
    def _add_to_cache(self, prompt_name: str, template: str):
        """Add prompt to cache with TTL"""
        expires_at = datetime.utcnow() + timedelta(seconds=self._cache_ttl_seconds)
        self._prompt_cache[prompt_name] = (template, expires_at)
    
    async def save_prompt(
        self,
        prompt_name: str,
        template: str,
        user_name: str = None
    ) -> Dict[str, Any]:
        """
        Save or update a prompt in database (admin only)
        
        Args:
            prompt_name: Unique key for the prompt
            template: The prompt template string
            user_name: Username of user (must be admin)
        """
        if not user_name or not self._is_admin(user_name):
            return {
                "success": False,
                "error": "Only admin can save prompts"
            }
        
        try:
            prompt_doc = {
                "prompt_name": prompt_name,
                "template": template,
                "updated_at": datetime.utcnow(),
                "updated_by": user_name
            }
            
            # Upsert - create or update
            result = await unified_database_service.db.prompts.update_one(
                {"prompt_name": prompt_name},
                {"$set": prompt_doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
                upsert=True
            )
            
            # Invalidate cache for this prompt
            if prompt_name in self._prompt_cache:
                del self._prompt_cache[prompt_name]
            
            return {
                "success": True,
                "modified": result.modified_count > 0,
                "created": result.upserted_id is not None
            }
                
        except Exception as e:
            logger.error(f"Failed to save prompt {prompt_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_prompts_batch(self, prompt_names: List[str]) -> Dict[str, str]:
        """
        Get multiple prompts in a single query with caching
        Returns dict of {prompt_name: template}
        """
        result = {}
        to_fetch = []
        
        # Check cache first
        for name in prompt_names:
            cached = self._get_from_cache(name)
            if cached:
                result[name] = cached
            else:
                to_fetch.append(name)
        
        # Fetch uncached prompts from database
        if to_fetch:
            try:
                cursor = unified_database_service.db.prompts.find(
                    {"prompt_name": {"$in": to_fetch}}
                )
                prompts = await cursor.to_list(length=None)
                
                for prompt in prompts:
                    name = prompt["prompt_name"]
                    template = prompt["template"]
                    result[name] = template
                    # Add to cache
                    self._add_to_cache(name, template)
                    
            except Exception as e:
                logger.error(f"Failed to fetch prompts batch: {str(e)}")
        
        return result
    
    async def get_prompt(self, name: str) -> Dict[str, Any]:
        """
        Get a single prompt by name with TTL caching
        """
        try:
            # Check cache first
            cached_template = self._get_from_cache(name)
            if cached_template:
                return {
                    "success": True,
                    "template": cached_template,
                    "cached": True
                }
            
            # Query database
            prompt = await unified_database_service.find_one(
                "prompts",
                {"prompt_name": name}
            )
            
            if prompt:
                template = prompt["template"]
                # Add to cache
                self._add_to_cache(name, template)
                
                return {
                    "success": True,
                    "template": template,
                    "cached": False
                }
            else:
                return {
                    "success": False,
                    "error": f"Prompt '{name}' not found"
                }
                
        except Exception as e:
            logger.error(f"Failed to get prompt {name}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get prompt: {str(e)}"
            }
    
    async def list_all_prompts(self, user_name: str = None) -> Dict[str, Any]:
        """
        List all available prompts (admin only for viewing all)
        """
        if not user_name or not self._is_admin(user_name):
            return {
                "success": False,
                "error": "Admin only"
            }
        
        try:
            cursor = unified_database_service.db.prompts.find({}).sort("prompt_name", 1)
            prompts = await cursor.to_list(length=None)
            
            return {
                "success": True,
                "prompts": [
                    {
                        "prompt_name": p["prompt_name"],
                        "updated_at": p.get("updated_at"),
                        "template_length": len(p.get("template", ""))
                    }
                    for p in prompts
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to list prompts: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_prompt(self, prompt_name: str, user_name: str = None) -> Dict[str, Any]:
        """
        Delete a prompt (admin only)
        """
        if not user_name or not self._is_admin(user_name):
            return {
                "success": False,
                "error": "Admin only"
            }
        
        try:
            result = await unified_database_service.db.prompts.delete_one(
                {"prompt_name": prompt_name}
            )
            
            # Remove from cache
            if prompt_name in self._prompt_cache:
                del self._prompt_cache[prompt_name]
            
            return {
                "success": result.deleted_count > 0,
                "deleted": result.deleted_count
            }
                
        except Exception as e:
            logger.error(f"Failed to delete prompt {prompt_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get and format a prompt with variables
        """
        result = await self.get_prompt(prompt_name)
        
        if not result["success"]:
            raise ValueError(f"Prompt not found: {prompt_name}")
        
        template = result["template"]
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable for prompt '{prompt_name}': {e}")
    
    def clear_cache(self, prompt_name: str = None):
        """Clear cache for specific prompt or all prompts"""
        if prompt_name:
            if prompt_name in self._prompt_cache:
                del self._prompt_cache[prompt_name]
        else:
            self._prompt_cache = {}


# Global instance
unified_prompt_service = UnifiedPromptService()

# Export for convenience
prompt_service = unified_prompt_service

__all__ = [
    'UnifiedPromptService',
    'unified_prompt_service',
    'prompt_service'
]
