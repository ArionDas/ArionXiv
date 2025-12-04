"""
Unified User Service for ArionXiv
Consolidates user_service.py, session_manager.py, and preferences_service.py
Provides comprehensive user management, session handling, and preferences
"""

import json
import os
import secrets
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from bson import ObjectId, errors as bson_errors

from .unified_database_service import unified_database_service
from arionxiv.cli.ui.global_theme_manager import global_theme_manager

logger = logging.getLogger(__name__)


class UnifiedUserService:
    """
    Comprehensive user service that handles:
    1. User account management (user_service.py functionality)
    2. Session management for CLI authentication (session_manager.py functionality)
    3. Paper preferences and relevance scoring (preferences_service.py functionality)
    """
    
    def __init__(self):
        # Session management
        self.session_dir = Path.home() / ".arionxiv"
        self.session_file = self.session_dir / "session.json"
        self.session_duration_days = 30
        
        # Ensure session directory exists
        self.session_dir.mkdir(exist_ok=True)
        
        # ArXiv categories for preferences
        self.arxiv_categories = {
            # Computer Science
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.RO": "Robotics",
            "cs.CR": "Cryptography and Security",
            "cs.DC": "Distributed, Parallel, and Cluster Computing",
            "cs.DS": "Data Structures and Algorithms",
            "cs.IR": "Information Retrieval",
            "cs.IT": "Information Theory",
            "cs.SE": "Software Engineering",
            "cs.SY": "Systems and Control",
            
            # Mathematics
            "math.ST": "Statistics Theory",
            "math.PR": "Probability",
            "math.OC": "Optimization and Control",
            "math.NA": "Numerical Analysis",
            
            # Physics
            "physics.data-an": "Data Analysis, Statistics and Probability",
            "physics.comp-ph": "Computational Physics",
            
            # Statistics
            "stat.ML": "Machine Learning",
            "stat.AP": "Applications",
            "stat.CO": "Computation",
            "stat.ME": "Methodology",
            
            # Quantitative Biology
            "q-bio.QM": "Quantitative Methods",
            "q-bio.NC": "Neurons and Cognition",
            
            # Economics
            "econ.EM": "Econometrics",
            "econ.TH": "Theoretical Economics"
        }
        
        logger.info("UnifiedUserService initialized")

    # ====================
    # INTERNAL HELPERS
    # ====================

    def _to_object_id(self, user_id: str) -> Optional[ObjectId]:
        """Best-effort conversion of a user identifier to ObjectId"""
        if not user_id:
            return None
        try:
            return ObjectId(str(user_id))
        except (bson_errors.InvalidId, TypeError):
            return None

    def _build_user_lookup_filter(self, user_id: str) -> Dict[str, Any]:
        """Create a Mongo filter for user lookups with ObjectId guardrails"""
        object_id = self._to_object_id(user_id)
        if object_id:
            return {'_id': object_id}
        # Fallback to alternate identifiers when ObjectId conversion fails
        return {'$or': [{'id': user_id}, {'email': user_id}]}
    
    # ====================
    # USER MANAGEMENT (from user_service.py)
    # ====================
    
    async def create_or_get_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user or get existing user"""
        try:
            # Check if database is available
            if unified_database_service.db is None:
                logger.warning("Database not available, returning demo user")
                return {
                    "success": True,
                    "user": {
                        "id": "demo-user",
                        "name": user_data.get("name", "Demo User"),
                        "email": user_data.get("email", "demo@arionxiv.com"),
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
                
            email = user_data.get("email")
            if not email:
                return {"success": False, "message": "Email is required"}
            
            # Check if user exists
            existing_user = await unified_database_service.get_user(email)
            if existing_user:
                # Update last login
                await self._update_last_login(str(existing_user["_id"]))
                existing_user["_id"] = str(existing_user["_id"])
                return {"success": True, "user": existing_user}
            
            # Create new user
            new_user_data = {
                "email": email,
                "name": user_data.get("name", ""),
                "picture": user_data.get("picture", ""),
                "preferences": {
                    "categories": ["cs.AI", "cs.LG", "cs.CV"],  # Default categories
                    "keywords": [],
                    "authors": [],
                    "exclude_keywords": [],
                    "min_relevance_score": 0.2,
                    "max_papers_per_day": 10,
                    "daily_digest": True,
                    "email_notifications": True,
                    "theme_color": "blue"
                },
                "stats": {
                    "papers_read": 0,
                    "papers_bookmarked": 0,
                    "analysis_count": 0
                },
                "last_login": datetime.utcnow()
            }
            
            success = await unified_database_service.create_user(new_user_data)
            if success:
                logger.info("New user created", email=email)
                # Get the created user
                created_user = await unified_database_service.get_user(email)
                if created_user:
                    created_user["_id"] = str(created_user["_id"])
                    return {"success": True, "user": created_user}
            
            return {"success": False, "message": "Failed to create user"}
            
        except Exception as e:
            logger.error("Failed to create or get user", error=str(e))
            # Return demo user as fallback
            return {
                "success": True,
                "user": {
                    "id": "demo-user-fallback",
                    "name": user_data.get("name", "Demo User"),
                    "email": user_data.get("email", "demo@arionxiv.com"),
                    "created_at": datetime.utcnow().isoformat()
                }
            }
    
    async def get_user_by_id(self, user_id: str) -> Dict[str, Any]:
        """Get user by ID"""
        try:
            # Check if database is available
            if unified_database_service.db is None:
                logger.warning("Database not available, returning demo user")
                return {
                    "success": True,
                    "user": {
                        "_id": user_id,
                        "id": user_id,
                        "name": "Demo User",
                        "email": "demo@arionxiv.com",
                        "preferences": {
                            "categories": ["cs.AI", "cs.LG"],
                            "keywords": [],
                            "maxDailyPapers": 10,
                            "analysisDepth": "standard",
                            "emailNotifications": True,
                            "language": "en"
                        }
                    }
                }
            
            query = self._build_user_lookup_filter(user_id)
            user = await unified_database_service.db.users.find_one(query)
            if user:
                user["_id"] = str(user["_id"])
                return {"success": True, "user": user}
            
            return {"success": False, "message": "User not found"}
            
        except Exception as e:
            logger.error("Failed to get user by ID", error=str(e))
            # Return fallback demo user
            return {
                "success": True,
                "user": {
                    "_id": user_id,
                    "id": user_id,
                    "name": "Demo User",
                    "email": "demo@arionxiv.com",
                    "preferences": {
                        "categories": ["cs.AI", "cs.LG"],
                        "keywords": [],
                        "maxDailyPapers": 10,
                        "analysisDepth": "standard",
                        "emailNotifications": True,
                        "language": "en"
                    }
                }
            }
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences"""
        try:
            # Check if database is available
            if unified_database_service.db is None:
                logger.warning("Database not available for updating preferences")
                return {"success": True, "message": "Preferences updated (offline mode)"}
            
            update_data = {
                "preferences": preferences,
                "updated_at": datetime.utcnow()
            }
            
            query = self._build_user_lookup_filter(user_id)
            result = await unified_database_service.db.users.update_one(
                query,
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info("User preferences updated successfully", user_id=user_id)
                return {"success": True, "message": "Preferences updated"}
            
            # If no document was modified, try to find if user exists
            user_count = await unified_database_service.db.users.count_documents(query)
            if user_count == 0:
                logger.warning("User not found for preference update", user_id=user_id)
                return {"success": False, "message": "User not found"}
            else:
                logger.info("No changes made to user preferences", user_id=user_id)
                return {"success": True, "message": "No changes made (preferences already up to date)"}
                
        except Exception as e:
            logger.error("Failed to update preferences", error=str(e), user_id=user_id)
            return {"success": False, "message": str(e)}
    
    async def mark_paper_viewed(self, user_id: str, paper_id: str) -> Dict[str, Any]:
        """Mark a paper as viewed by user"""
        try:
            # Check if database is available
            if unified_database_service.db is None:
                logger.warning("Database not available for marking paper viewed")
                return {"success": True, "message": "Paper marked as viewed (offline mode)"}
                
            interaction_data = {
                "user_id": user_id,
                "paper_id": paper_id,
                "action": "view",
                "timestamp": datetime.utcnow()
            }
            
            # Use upsert to avoid duplicates
            await unified_database_service.db.user_papers.update_one(
                {"user_id": user_id, "paper_id": paper_id},
                {"$set": interaction_data, "$inc": {"view_count": 1}},
                upsert=True
            )
            
            # Update user stats
            await self._increment_user_stat(user_id, "papers_read")
            
            return {"success": True, "message": "Paper marked as viewed"}
        except Exception as e:
            logger.error("Failed to mark paper as viewed", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def bookmark_paper(self, user_id: str, paper_id: str) -> Dict[str, Any]:
        """Bookmark a paper for user"""
        try:
            # Check if database is available
            if unified_database_service.db is None:
                logger.warning("Database not available for bookmarking")
                return {"success": True, "message": "Paper bookmarked (offline mode)"}
                
            # Check if already bookmarked
            existing = await unified_database_service.db.user_papers.find_one({
                "user_id": user_id,
                "paper_id": paper_id,
                "bookmarked": True
            })
            
            if existing:
                return {"success": False, "message": "Paper already bookmarked"}
            
            bookmark_data = {
                "user_id": user_id,
                "paper_id": paper_id,
                "action": "bookmark",
                "bookmarked": True,
                "timestamp": datetime.utcnow()
            }
            
            await unified_database_service.db.user_papers.update_one(
                {"user_id": user_id, "paper_id": paper_id},
                {"$set": bookmark_data},
                upsert=True
            )
            
            # Update user stats
            await self._increment_user_stat(user_id, "papers_bookmarked")
            
            return {"success": True, "message": "Paper bookmarked successfully"}
        except Exception as e:
            logger.error("Failed to bookmark paper", error=str(e))
            return {"success": True, "message": "Paper bookmarked (fallback mode)"}
    
    async def get_user_bookmarks(self, user_id: str) -> Dict[str, Any]:
        """Get user's bookmarked papers"""
        try:
            pipeline = [
                {"$match": {"user_id": user_id, "bookmarked": True}},
                {
                    "$lookup": {
                        "from": "papers",
                        "localField": "paper_id",
                        "foreignField": "arxiv_id",
                        "as": "paper_details"
                    }
                },
                {"$unwind": "$paper_details"},
                {"$sort": {"timestamp": -1}}
            ]
            
            bookmarks = []
            async for bookmark in unified_database_service.db.user_papers.aggregate(pipeline):
                bookmark["paper_details"]["_id"] = str(bookmark["paper_details"]["_id"])
                bookmarks.append(bookmark["paper_details"])
            
            return {"success": True, "bookmarks": bookmarks}
        except Exception as e:
            logger.error("Failed to get user bookmarks", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            user_result = await self.get_user_by_id(user_id)
            if not user_result["success"]:
                return {"success": False, "message": "User not found"}
            
            stats = user_result["user"].get("stats", {})
            
            # Add real-time stats
            bookmark_count = await unified_database_service.db.user_papers.count_documents({
                "user_id": user_id,
                "bookmarked": True
            })
            
            view_count = await unified_database_service.db.user_papers.count_documents({
                "user_id": user_id,
                "action": "view"
            })
            
            stats.update({
                "papers_bookmarked": bookmark_count,
                "papers_read": view_count
            })
            
            return {"success": True, "stats": stats}
        except Exception as e:
            logger.error("Failed to get user stats", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def _update_last_login(self, user_id: str):
        """Update user's last login time"""
        try:
            object_id = self._to_object_id(user_id)
            if not object_id:
                logger.warning("Cannot update last login for non-ObjectId user_id", user_id=user_id)
                return
            await unified_database_service.db.users.update_one(
                {"_id": object_id},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        except Exception as e:
            logger.warning("Failed to update last login", error=str(e))
    
    async def _increment_user_stat(self, user_id: str, stat_name: str):
        """Increment user statistic"""
        try:
            object_id = self._to_object_id(user_id)
            if not object_id:
                logger.warning("Cannot increment stat for non-ObjectId user_id", user_id=user_id, stat=stat_name)
                return
            await unified_database_service.db.users.update_one(
                {"_id": object_id},
                {"$inc": {f"stats.{stat_name}": 1}}
            )
        except Exception as e:
            logger.warning("Failed to increment user stat", error=str(e))
    
    # ====================
    # SESSION MANAGEMENT (from session_manager.py)
    # ====================
    
    def create_session(self, user_data: Dict[str, Any]) -> str:
        """Create a new session for user and initialize theme"""
        try:
            session_token = secrets.token_urlsafe(32)
            
            session_data = {
                "user_id": user_data["id"],
                "email": user_data["email"],
                "user_name": user_data["user_name"],
                "full_name": user_data.get("full_name", ""),
                "session_token": session_token,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(days=self.session_duration_days)).isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            # Save session to file
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Set file permissions (readable only by user)
            os.chmod(self.session_file, 0o600)
            
            # Initialize user theme
            self._initialize_user_theme(user_data["id"])
            
            logger.info("Session created", user_id=user_data["id"])
            return session_token
            
        except Exception as e:
            logger.error("Failed to create session", error=str(e))
            return ""
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get current active session"""
        try:
            if not self.session_file.exists():
                return None
            
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.utcnow() > expires_at:
                self.clear_session()
                return None
            
            # Update last activity
            session_data["last_activity"] = datetime.utcnow().isoformat()
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Initialize user theme if not already done
            try:
                from arionxiv.cli.ui.global_theme_manager import global_theme_manager
                if not global_theme_manager.is_initialized():
                    self._initialize_user_theme(session_data["user_id"])
            except Exception as e:
                logger.debug(f"Theme initialization skipped: {e}")
            
            return session_data
            
        except Exception as e:
            logger.error("Failed to get current session", error=str(e))
            return None
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        session = self.get_current_session()
        return session is not None
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user info"""
        session = self.get_current_session()
        if session:
            return {
                "id": session["user_id"],
                "email": session["email"],
                "user_name": session["user_name"],
                "full_name": session["full_name"]
            }
        return None
    
    def clear_session(self):
        """Clear current session (logout)"""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            logger.info("Session cleared")
        except Exception as e:
            logger.error("Failed to clear session", error=str(e))
    
    def extend_session(self, days: int = None):
        """Extend current session expiry"""
        try:
            if not self.session_file.exists():
                return False
            
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            if days is None:
                days = self.session_duration_days
            
            session_data["expires_at"] = (datetime.utcnow() + timedelta(days=days)).isoformat()
            session_data["last_activity"] = datetime.utcnow().isoformat()
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error("Failed to extend session", error=str(e))
            return False
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get session information for display"""
        session = self.get_current_session()
        if session:
            created_at = datetime.fromisoformat(session["created_at"])
            expires_at = datetime.fromisoformat(session["expires_at"])
            last_activity = datetime.fromisoformat(session["last_activity"])
            
            return {
                "user": {
                    "user_name": session["user_name"],
                    "email": session["email"],
                    "full_name": session["full_name"]
                },
                "session": {
                    "created": created_at.strftime("%Y-%m-%d %H:%M"),
                    "expires": expires_at.strftime("%Y-%m-%d %H:%M"),
                    "last_activity": last_activity.strftime("%Y-%m-%d %H:%M"),
                    "days_remaining": (expires_at - datetime.utcnow()).days
                }
            }
        return None
    
    def _initialize_user_theme(self, user_id: str):
        """Initialize user theme from database preferences"""
        try:
            def init_theme():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Import here to avoid circular imports
                    from arionxiv.cli.ui.global_theme_manager import global_theme_manager
                    theme = loop.run_until_complete(global_theme_manager.initialize_user_theme(user_id))
                    logger.info(f"User theme initialized: {theme}", user_id=user_id)
                    loop.close()
                except Exception as e:
                    logger.error(f"Failed to initialize user theme: {e}")
            
            # Initialize in background thread to avoid blocking
            theme_thread = threading.Thread(target=init_theme, daemon=True)
            theme_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting theme initialization: {e}")
    
    # ====================
    # PREFERENCES MANAGEMENT (from preferences_service.py)
    # ====================
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's paper preferences"""
        try:
            # Ensure database is connected
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            # Get user and their preferences
            user_result = await self.get_user_by_id(user_id)
            if user_result["success"]:
                user = user_result["user"]
                preferences = user.get("preferences", {})
                
                # Return structured preferences with defaults
                return {
                    "success": True,
                    "preferences": {
                        "categories": preferences.get("categories", ["cs.AI", "cs.LG", "cs.CV"]),
                        "keywords": preferences.get("keywords", []),
                        "authors": preferences.get("authors", []),
                        "exclude_keywords": preferences.get("exclude_keywords", []),
                        "min_relevance_score": preferences.get("min_relevance_score", 0.2),
                        "max_papers_per_day": preferences.get("max_papers_per_day", 10)
                    }
                }
            else:
                return {"success": True, "preferences": self._get_default_preferences()}
                
        except Exception as e:
            logger.error("Failed to get user preferences", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Save user's paper preferences"""
        try:
            # Validate preferences
            validated_prefs = self._validate_preferences(preferences)
            
            # Update user preferences
            return await self.update_user_preferences(user_id, {"preferences": validated_prefs})
            
        except Exception as e:
            logger.error("Failed to save user preferences", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def update_user_preferences_partial(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific user preferences without overwriting all preferences"""
        try:
            # Get current preferences
            current_prefs_result = await self.get_user_preferences(user_id)
            if not current_prefs_result["success"]:
                return current_prefs_result
                
            current_prefs = current_prefs_result["preferences"]
            
            # Apply updates
            for key, value in updates.items():
                if key in ["categories", "keywords", "authors", "exclude_keywords"]:
                    # For list fields, handle append/remove operations
                    if isinstance(value, dict):
                        if "add" in value:
                            if key not in current_prefs:
                                current_prefs[key] = []
                            for item in value["add"]:
                                if item not in current_prefs[key]:
                                    current_prefs[key].append(item)
                        if "remove" in value:
                            if key in current_prefs:
                                for item in value["remove"]:
                                    if item in current_prefs[key]:
                                        current_prefs[key].remove(item)
                    else:
                        # Direct assignment
                        current_prefs[key] = value
                else:
                    # Direct assignment for non-list fields
                    current_prefs[key] = value
            
            # Save updated preferences
            return await self.save_user_preferences(user_id, current_prefs)
            
        except Exception as e:
            logger.error("Failed to update user preferences", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def get_relevant_papers(self, user_id: str, days_back: int = 1) -> Dict[str, Any]:
        """Get papers relevant to user's preferences"""
        try:
            # Ensure database is connected
            if unified_database_service.db is None:
                await unified_database_service.connect_mongodb()
            
            # Get user preferences
            prefs_result = await self.get_user_preferences(user_id)
            if not prefs_result["success"]:
                return prefs_result
            
            preferences = prefs_result["preferences"]
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Build query based on preferences
            query = {
                "published_date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Add category filter - this should be the PRIMARY filter
            if preferences["categories"]:
                query["categories"] = {"$in": preferences["categories"]}
            
            # Exclude papers with unwanted keywords
            if preferences["exclude_keywords"]:
                exclude_conditions = []
                for exclude_kw in preferences["exclude_keywords"]:
                    exclude_conditions.extend([
                        {"title": {"$not": {"$regex": exclude_kw, "$options": "i"}}},
                        {"abstract": {"$not": {"$regex": exclude_kw, "$options": "i"}}}
                    ])
                if exclude_conditions:
                    query["$and"] = exclude_conditions
            
            # Fetch papers from database
            papers = await unified_database_service.db.papers.find(query).limit(
                preferences["max_papers_per_day"]
            ).to_list(length=None)
            
            # Calculate relevance scores
            scored_papers = []
            for paper in papers:
                score = self._calculate_relevance_score(paper, preferences)
                if score >= preferences["min_relevance_score"]:
                    paper["relevance_score"] = score
                    scored_papers.append(paper)
            
            # Sort by relevance score
            scored_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "success": True,
                "papers": scored_papers[:preferences["max_papers_per_day"]],
                "total_found": len(scored_papers),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Failed to get relevant papers", error=str(e))
            return {"success": False, "message": str(e)}
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default paper preferences"""
        return {
            "categories": ["cs.AI", "cs.LG", "cs.CV"],
            "keywords": [],
            "authors": [],
            "exclude_keywords": [],
            "min_relevance_score": 0.2,
            "max_papers_per_day": 10,
            "daily_dose_enabled": False,
            "daily_dose_time": "08:00"
        }
    
    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean preferences"""
        validated = {}
        
        # Validate categories
        categories = preferences.get("categories", [])
        if isinstance(categories, list):
            validated["categories"] = [cat for cat in categories if cat in self.arxiv_categories]
        else:
            validated["categories"] = []
        
        # Validate keywords
        keywords = preferences.get("keywords", [])
        if isinstance(keywords, list):
            validated["keywords"] = [kw.strip() for kw in keywords if kw.strip()]
        else:
            validated["keywords"] = []
        
        # Validate authors
        authors = preferences.get("authors", [])
        if isinstance(authors, list):
            validated["authors"] = [auth.strip() for auth in authors if auth.strip()]
        else:
            validated["authors"] = []
        
        # Validate exclude keywords
        exclude_keywords = preferences.get("exclude_keywords", [])
        if isinstance(exclude_keywords, list):
            validated["exclude_keywords"] = [kw.strip() for kw in exclude_keywords if kw.strip()]
        else:
            validated["exclude_keywords"] = []
        
        # Validate min relevance score
        min_score = preferences.get("min_relevance_score", 0.2)
        if isinstance(min_score, (int, float)) and 0 <= min_score <= 1:
            validated["min_relevance_score"] = min_score
        else:
            validated["min_relevance_score"] = 0.2
        
        # Validate max papers per day
        max_papers = preferences.get("max_papers_per_day", 10)
        if isinstance(max_papers, int) and 1 <= max_papers <= 50:
            validated["max_papers_per_day"] = max_papers
        else:
            validated["max_papers_per_day"] = 10
        
        # Validate daily dose enabled
        validated["daily_dose_enabled"] = bool(preferences.get("daily_dose_enabled", False))
        
        # Validate daily dose time
        daily_time = preferences.get("daily_dose_time", "08:00")
        if isinstance(daily_time, str) and len(daily_time.split(":")) == 2:
            try:
                hour, minute = map(int, daily_time.split(":"))
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    validated["daily_dose_time"] = daily_time
                else:
                    validated["daily_dose_time"] = "08:00"
            except ValueError:
                validated["daily_dose_time"] = "08:00"
        else:
            validated["daily_dose_time"] = "08:00"
        
        return validated
    
    def _calculate_relevance_score(self, paper: Dict[str, Any], preferences: Dict[str, Any]) -> float:
        """Calculate relevance score for a paper based on user preferences"""
        score = 0.0
        
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        categories = paper.get("categories", [])
        authors = paper.get("authors", [])
        
        # Category match is PRIMARY (70% of score)
        if preferences["categories"]:
            category_matches = len(set(categories) & set(preferences["categories"]))
            if category_matches > 0:
                category_score = 0.7 + (category_matches - 1) * 0.1
                score += min(category_score, 0.8)
        else:
            score += 0.3
        
        # Keyword match in title (15% of score)
        if preferences["keywords"]:
            title_matches = sum(1 for kw in preferences["keywords"] if kw.lower() in title)
            if title_matches > 0:
                title_score = min(title_matches / len(preferences["keywords"]) * 0.15, 0.15)
                score += title_score
        
        # Keyword match in abstract (10% of score)
        if preferences["keywords"]:
            abstract_matches = sum(1 for kw in preferences["keywords"] if kw.lower() in abstract)
            if abstract_matches > 0:
                abstract_score = min(abstract_matches / len(preferences["keywords"]) * 0.10, 0.10)
                score += abstract_score
        
        # Author match (5% of score)
        if preferences["authors"]:
            author_matches = sum(1 for auth in preferences["authors"] 
                               if any(auth.lower() in author.lower() for author in authors))
            if author_matches > 0:
                author_score = min(author_matches / len(preferences["authors"]) * 0.05, 0.05)
                score += author_score
        
        # Penalty for exclude keywords
        if preferences["exclude_keywords"]:
            for exclude_kw in preferences["exclude_keywords"]:
                if exclude_kw.lower() in title or exclude_kw.lower() in abstract:
                    score *= 0.3  # 70% penalty
        
        return min(score, 1.0)
    
    def get_available_categories(self) -> Dict[str, str]:
        """Get available arXiv categories"""
        return self.arxiv_categories


# Global instances
unified_user_service = UnifiedUserService()

# Backwards compatibility
user_service = unified_user_service
session_manager = unified_user_service
preferences_service = unified_user_service

# Export commonly used functions
create_or_get_user = unified_user_service.create_or_get_user
get_user_by_id = unified_user_service.get_user_by_id
create_session = unified_user_service.create_session
get_current_session = unified_user_service.get_current_session
is_authenticated = unified_user_service.is_authenticated
get_user_preferences = unified_user_service.get_user_preferences
save_user_preferences = unified_user_service.save_user_preferences

__all__ = [
    'UnifiedUserService',
    'unified_user_service',
    'user_service',
    'session_manager',
    'preferences_service',
    'create_or_get_user',
    'get_user_by_id',
    'create_session',
    'get_current_session',
    'is_authenticated',
    'get_user_preferences',
    'save_user_preferences'
]