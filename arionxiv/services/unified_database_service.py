"""
Unified Database Service for ArionXiv
Consolidates database_client.py and sync_db_wrapper.py
Provides comprehensive database management with both async and sync interfaces
"""

import os
import asyncio
import threading
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import IP helper for better error messages
try:
    from ..utils.ip_helper import check_mongodb_connection_error, get_public_ip
    IP_HELPER_AVAILABLE = True
except ImportError:
    IP_HELPER_AVAILABLE = False


class UnifiedDatabaseService:
    """
    Comprehensive database service that handles:
    1. MongoDB connections with TTL-based caching
    2. Synchronous wrapper for CLI operations (sync_db_wrapper.py functionality)
    3. All CRUD operations for papers, users, sessions, and analysis
    """
    
    # Default production MongoDB URI for the ArionXiv hosted service
    # This allows the PyPI package to work out-of-the-box without user configuration
    # Environment variables can override this for development/testing
    DEFAULT_MONGODB_URI = "mongodb+srv://ariondasad:Und3O0QEypfYqawd@arionxiv.cycfdc0.mongodb.net/?retryWrites=true&w=majority"
    
    def __init__(self):
        # Async database clients
        self.mongodb_client: Optional[AsyncIOMotorClient] = None
        self.db = None
        
        # Sync operations support
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # MongoDB connection string - uses default production URI, can be overridden by env vars
        self._db_url = None
        self.database_name = os.getenv("DATABASE_NAME", "arionxiv")
        
        logger.info("UnifiedDatabaseService initialized")
    
    @property
    def db_url(self) -> str:
        """Get MongoDB URL - uses environment variable if set, otherwise uses default production URI."""
        if self._db_url is None:
            # Check environment variables first (for development/testing override)
            self._db_url = os.getenv('MONGODB_URI') or os.getenv('MONGODB_URL')
            # Fall back to default production URI
            if not self._db_url:
                self._db_url = self.DEFAULT_MONGODB_URI
        return self._db_url
    
    # ============================================================
    # CONNECTION MANAGEMENT (from database_client.py)
    # ============================================================
    
    async def connect_mongodb(self):
        """
        Connect to MongoDB Atlas with proper SSL configuration
        """

        try:
            logger.info("Attempting to connect to MongoDB Atlas...")
            
            # Import config service for connection parameters
            from .unified_config_service import unified_config_service
            connection_params = unified_config_service.get_mongodb_connection_config()
            
            logger.info(f"Connection timeout: {connection_params['connectTimeoutMS']}ms")
            
            # For MongoDB Atlas (mongodb+srv://) - use property to trigger lazy load
            if 'mongodb+srv://' in self.db_url:
                logger.info("Connecting to MongoDB Atlas cluster...")
                self.mongodb_client = AsyncIOMotorClient(self.db_url, **connection_params)
            else:
                logger.info("Connecting to MongoDB instance...")
                self.mongodb_client = AsyncIOMotorClient(self.db_url, **connection_params)
            
            # Set database
            self.db = self.mongodb_client[self.database_name]
            
            # Test the connection with a ping
            logger.info("Testing MongoDB connection...")
            await asyncio.wait_for(
                self.mongodb_client.admin.command('ping'), 
                timeout=30.0
            )
            
            logger.info(f"Successfully connected to MongoDB: {self.database_name}")
            
            # Create indexes
            await self.create_indexes()
            logger.info("Database indexes created/verified")
            
        except asyncio.TimeoutError:
            logger.error("MongoDB connection timeout - check your internet connection")
            if IP_HELPER_AVAILABLE:
                check_mongodb_connection_error("connection timeout")
            raise Exception("MongoDB connection timeout")
        except Exception as e:
            error_msg = str(e)
            
            # Check for common IP whitelisting issues
            if IP_HELPER_AVAILABLE:
                check_mongodb_connection_error(error_msg)
            
            if "SSL handshake failed" in error_msg and "TLSV1_ALERT_INTERNAL_ERROR" in error_msg:
                logger.error("MongoDB Atlas SSL handshake failed")
                logger.error("This typically indicates:")
                logger.error("   1. Invalid credentials (username/password)")
                logger.error("   2. MongoDB Atlas cluster is paused or deleted")
                logger.error("   3. IP address not whitelisted in Atlas Network Access")
                logger.error("   4. Cluster URL has changed")
                logger.error("Please check your MongoDB Atlas console")
                raise Exception("MongoDB Atlas authentication/access failed - check credentials and cluster status")
            else:
                logger.error(f"MongoDB connection failed: {str(e)}")
                raise Exception(f"Failed to connect to MongoDB: {str(e)}")
    
    def _enable_offline_mode(self):
        """Enable offline mode with in-memory storage"""
        logger.info("ArionXiv running in offline mode - using in-memory storage")
        # Create a simple in-memory storage
        self._offline_storage = {
            'papers': {},
            'users': {},
            'sessions': {},
            'analyses': {}
        }
        # Mark as offline mode
        self._offline_mode = True
    
    def is_offline(self) -> bool:
        """
        Check if the service is in offline mode, i.e. not connected to MongoDB
        """
        return getattr(self, '_offline_mode', False)
    
    async def create_indexes(self):
        """
        Create necessary indexes for collections

        Indices:
        - Papers: arxiv_id (unique), title, authors, categories, published, text search on title+abstract
        - Users: user_name (unique), email, user_id, created_at, updated_at, etc.
        - User Papers: user_name, arxiv_id, category, added_at, TTL index for cleanup
        - Cache: key (unique), expires_at (TTL)
        - Chat Sessions: user_name, created_at, TTL index for cleanup
        - Daily Analysis: user_id, analysis_type, analyzed_at, paper_id, created_at, generated_at
        - Cron Jobs: user_id, job_type (unique), status, updated_at
        - Paper Texts and Embeddings: paper_id (unique), chunk_id (unique), expires_at (TTL)
        """
        if self.db is None:
            return
            
        try:
            # Papers collection indexes
            await self.db.papers.create_index("arxiv_id", unique=True)
            await self.db.papers.create_index("title")
            await self.db.papers.create_index("authors")
            await self.db.papers.create_index("categories")
            await self.db.papers.create_index("published")
            await self.db.papers.create_index([("title", "text"), ("abstract", "text")])
            
            # Users collection indexes
            await self.db.users.create_index("user_name", unique=True)
            await self.db.users.create_index("email")
            
            # User papers collection indexes
            # Drop old stale indexes if they exist (migration from old field names)
            try:
                await self.db.user_papers.drop_index("user_id_1_paper_id_1")
            except Exception:
                pass  # Index doesn't exist, that's fine
            
            # Clean up any documents with null/missing key fields from old schema
            try:
                await self.db.user_papers.delete_many({
                    "$or": [
                        {"user_name": {"$exists": False}},
                        {"user_name": None},
                        {"arxiv_id": {"$exists": False}},
                        {"arxiv_id": None}
                    ]
                })
            except Exception:
                pass
            
            await self.db.user_papers.create_index([("user_name", 1), ("arxiv_id", 1)], unique=True, sparse=True)
            await self.db.user_papers.create_index("user_name")
            await self.db.user_papers.create_index("category")
            await self.db.user_papers.create_index("added_at")
            
            # TTL index for cleanup (papers older than 30 days will be removed)
            try:
                await self.db.user_papers.create_index("added_at", name="user_papers_ttl", expireAfterSeconds=30*24*60*60)
            except Exception:
                pass  # Index might already exist
            
            # MongoDB TTL-based cache collection indexes
            await self.db.cache.create_index("key", unique=True)
            try:
                await self.db.cache.create_index("expires_at", name="cache_ttl", expireAfterSeconds=0)
            except Exception:
                pass  # Index might already exist
            
            # Chat sessions collection indexes
            await self.db.chat_sessions.create_index("user_name")
            await self.db.chat_sessions.create_index("session_id", unique=True)
            await self.db.chat_sessions.create_index("created_at")
            try:
                # 24-hour TTL based on expires_at field
                await self.db.chat_sessions.create_index("expires_at", name="chat_sessions_ttl", expireAfterSeconds=0)
            except Exception:
                pass  # Index might already exist
            
            # Daily analysis collection indexes
            await self.db.daily_analysis.create_index([("user_id", 1), ("analysis_type", 1)])
            await self.db.daily_analysis.create_index([("user_id", 1), ("analyzed_at", 1)])
            await self.db.daily_analysis.create_index([("user_id", 1), ("paper_id", 1)])
            await self.db.daily_analysis.create_index("created_at")
            await self.db.daily_analysis.create_index("generated_at")
            
            # Cron jobs collection indexes
            await self.db.cron_jobs.create_index([("user_id", 1), ("job_type", 1)], unique=True)
            await self.db.cron_jobs.create_index("status")
            await self.db.cron_jobs.create_index("updated_at")
            
            # Paper texts and embeddings indexes
            await self.db.paper_texts.create_index("paper_id", unique=True)
            try:
                await self.db.paper_texts.create_index("expires_at", name="paper_texts_ttl", expireAfterSeconds=0)
            except Exception:
                pass  # Index might already exist
            await self.db.paper_embeddings.create_index("paper_id")
            await self.db.paper_embeddings.create_index("chunk_id", unique=True)
            try:
                await self.db.paper_embeddings.create_index("expires_at", name="paper_embeddings_ttl", expireAfterSeconds=0)
            except Exception:
                pass  # Index might already exist
            
            # Prompts collection indexes
            await self.db.prompts.create_index("prompt_name", unique=True)
            await self.db.prompts.create_index("updated_at")
            
            # Daily dose collection indexes
            await self.db.daily_dose.create_index("user_id")
            await self.db.daily_dose.create_index([("user_id", 1), ("generated_at", -1)])
            await self.db.daily_dose.create_index("date")
            await self.db.daily_dose.create_index("created_at")
            
            logger.debug("Database indexes created successfully")
            
        except Exception as e:
            # Suppress index creation warnings - often caused by existing data inconsistencies
            logger.debug(f"Index creation skipped: {str(e)}")
    
    async def connect(self):
        """
        Connect to MongoDB with fallback to offline mode

        Purpose: Establish database connections and initialize in offline mode if connection fails
        """

        logger.info("Initializing database connections...")
        
        # Try to connect to MongoDB with fallback to offline mode
        try:
            await self.connect_mongodb()
            logger.info(" MongoDB connection successful - running in ONLINE mode")
            self._offline_mode = False
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {str(e)}")
            logger.warning(" Falling back to offline mode with in-memory storage")
            self._offline_mode = True
            self.offline_papers = {}
            self.offline_users = {}
            self.offline_auth_sessions = {}
        
        logger.info("Database service ready!")
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        try:
            if self.mongodb_client:
                self.mongodb_client.close()
                
            logger.debug("Disconnected from databases")
            
        except Exception as e:
            logger.error(f"Error disconnecting from databases: {str(e)}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of database connections"""
        health = {"mongodb": False}
        
        # MongoDB health check
        try:
            if self.db:
                await self.mongodb_client.admin.command('ping')
                health["mongodb"] = True
        except Exception as e:
            logger.debug(f"MongoDB health check failed: {str(e)}")
        
        return health
    
    # ============================================================
    # BASIC CRUD OPERATIONS
    # ============================================================
    
    async def find_one(self, collection: str, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection

        Args:
            collection (str): The name of the collection to search
            filter_dict (Dict[str, Any]): The filter criteria for the document

        Returns:
            Optional[Dict[str, Any]]: The found document or None if not found
        """

        if self.is_offline():
            # Offline mode - search in memory
            logger.info("Running in offline mode - searching in in-memory storage")
            storage = getattr(self, '_offline_storage', {})
            for doc_id, doc in storage.get(collection, {}).items():
                if all(doc.get(k) == v for k, v in filter_dict.items()):
                    return doc
            return None
            
        if self.db is None:
            logger.warning("No database connection available")
            return None
            
        try:
            logger.debug(f"Querying collection '{collection}' with filter: {filter_dict}")
            result = await self.db[collection].find_one(filter_dict)
            return result
        except RuntimeError as e:
            # Handle closed event loop gracefully
            if "Event loop is closed" in str(e):
                logger.debug("Query skipped - event loop closed during shutdown")
            else:
                logger.debug(f"Query error: {str(e)}")
            return None
        except Exception as e:
            logger.debug(f"Query error: {str(e)}")
            return None
    
    async def update_one(self, collection: str, filter_dict: Dict[str, Any], update_dict: Dict[str, Any], upsert: bool = False):
        """
        Update a single document in a collection

        Args:
            collection (str): The name of the collection to update
            filter_dict (Dict[str, Any]): The filter criteria for the document to update
            update_dict (Dict[str, Any]): The update operations to apply
            upsert (bool): If True, insert document if not found
        
        Returns:
            The result of the update operation
        """

        if self.is_offline():
            logger.error("Cannot update document in offline mode - Database connection required")
            raise ConnectionError("Database connection is not available. Please check your MongoDB connection string in the .env file.")
            
        if self.db is None:
            logger.error("No database connection available")
            raise ConnectionError("Database connection is not available. Please check your MongoDB connection string in the .env file.")
            
        try:
            result = await self.db[collection].update_one(filter_dict, update_dict, upsert=upsert)
            return result
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return None
    
    async def insert_one(self, collection: str, document: Dict[str, Any]):
        """
        Insert a single document into a collection

        Args:
            collection (str): The name of the collection to insert into
            document (Dict[str, Any]): The document to insert
        
        Returns:
            The result of the insert operation
        """

        if self.db is None:
            logger.warning("No database connection available")
            return None
            
        try:
            result = await self.db[collection].insert_one(document)
            return result
        except Exception as e:
            logger.error(f"Failed to insert document: {str(e)}")
            return None
    
    async def delete_one(self, collection: str, filter_dict: Dict[str, Any]):
        """Delete a single document from a collection"""
        if self.db is None:
            logger.warning("No database connection available")
            return None
            
        try:
            result = await self.db[collection].delete_one(filter_dict)
            return result
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return None
    
    async def aggregate(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run an aggregation pipeline on a collection

        Args:
            collection (str): The name of the collection
            pipeline (List[Dict[str, Any]]): The aggregation pipeline stages

        Returns:
            List[Dict[str, Any]]: Results from the aggregation
        """
        if self.db is None:
            logger.warning("No database connection available")
            return []
            
        try:
            cursor = self.db[collection].aggregate(pipeline)
            results = await cursor.to_list(length=100)
            return results
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            return []
    
    async def find_many(self, collection: str, filter_dict: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find multiple documents in a collection

        Args:
            collection (str): The name of the collection to search
            filter_dict (Dict[str, Any]): The filter criteria (optional)
            limit (int): Maximum number of documents to return

        Returns:
            List[Dict[str, Any]]: List of found documents
        """
        if self.db is None:
            logger.warning("No database connection available")
            return []
            
        try:
            filter_dict = filter_dict or {}
            cursor = self.db[collection].find(filter_dict).limit(limit)
            results = await cursor.to_list(length=limit)
            return results
        except Exception as e:
            logger.error(f"Find many failed: {str(e)}")
            return []
    
    async def delete_many(self, collection: str, filter_dict: Dict[str, Any]) -> int:
        """
        Delete multiple documents from a collection

        Args:
            collection (str): The name of the collection
            filter_dict (Dict[str, Any]): The filter criteria for deletion

        Returns:
            int: Number of documents deleted
        """
        if self.db is None:
            logger.warning("No database connection available")
            return 0
            
        try:
            result = await self.db[collection].delete_many(filter_dict)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Delete many failed: {str(e)}")
            return 0
    
    async def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[Any]:
        """
        Insert multiple documents into a collection

        Args:
            collection (str): The name of the collection
            documents (List[Dict[str, Any]]): List of documents to insert

        Returns:
            List[Any]: List of inserted document IDs
        """
        if self.db is None:
            logger.warning("No database connection available")
            return []
            
        if not documents:
            return []
            
        try:
            result = await self.db[collection].insert_many(documents)
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Insert many failed: {str(e)}")
            return []
    
    # ============================================================
    # PAPER MANAGEMENT
    # ============================================================
    
    async def save_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a paper to the database

        Args:
            paper_data (Dict[str, Any]): The paper data to save
        
        Returns:
            Dict[str, Any]: Result of the save operation
        """

        if self.db is None:
            logger.warning("No database connection available")
            return {"success": False, "message": "Database not connected"}
            
        try:
            # Add timestamp
            paper_data["saved_at"] = datetime.utcnow()
            
            # Upsert the paper
            result = await self.db.papers.update_one(
                {"arxiv_id": paper_data["arxiv_id"]},
                {"$set": paper_data},
                upsert=True
            )
            
            logger.debug(f"Paper saved: {paper_data['arxiv_id']}")
            return {"success": True, "message": "Paper saved successfully"}
            
        except Exception as e:
            logger.error(f"Failed to save paper: {str(e)}")
            return {"success": False, "message": str(e)}
    
    async def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a paper by its arXiv ID

        Args:
            arxiv_id (str): The arXiv ID of the paper to retrieve
        """

        if self.db is None:
            return None
            
        try:
            paper = await self.db.papers.find_one({"arxiv_id": arxiv_id})
            return paper
            
        except Exception as e:
            logger.error(f"Failed to get paper: {str(e)}")
            return None
    
    async def search_papers(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search papers using text search

        Args:
            query (str): The search query
            limit (int): Maximum number of papers to return

        Returns:
            List[Dict[str, Any]]: List of papers matching the query
        """

        if self.db is None:
            return []
            
        try:
            cursor = self.db.papers.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            papers = await cursor.to_list(length=limit)
            return papers
            
        except Exception as e:
            logger.error(f"Failed to search papers: {str(e)}")
            return []
    
    async def get_papers_by_category(self, category: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get papers by category

        Args:
            category (str): The category to filter papers by
            limit (int): Maximum number of papers to return

        Returns:
            List[Dict[str, Any]]: List of papers in the specified category
        """
        if self.db is None:
            return []
            
        try:
            cursor = self.db.papers.find(
                {"categories": {"$in": [category]}}
            ).sort("published", -1).limit(limit)
            
            papers = await cursor.to_list(length=limit)
            return papers
            
        except Exception as e:
            logger.error(f"Failed to get papers by category: {str(e)}")
            return []
    
    async def get_recent_papers(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get papers published in the last 'days' days

        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of papers to return
        
        Returns:
            List[Dict[str, Any]]: List of recent papers
        """

        if self.db is None:
            return []
            
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            cursor = self.db.papers.find(
                {"published": {"$gte": cutoff_date.isoformat()}}
            ).sort("published", -1).limit(limit)
            
            papers = await cursor.to_list(length=limit)
            return papers
            
        except Exception as e:
            logger.error(f"Failed to get recent papers: {str(e)}")
            return []
    
    # ============================================================
    # USER MANAGEMENT
    # ============================================================
    
    async def create_user(self, user_data: Dict[str, Any]) -> bool:
        """
        Create a new user

        Args:
            user_data (Dict[str, Any]): The user data to create an account for
        """

        if self.db is None:
            return False
            
        try:
            user_data["created_at"] = datetime.utcnow()
            user_data["updated_at"] = datetime.utcnow()
            
            await self.db.users.insert_one(user_data)
            logger.debug(f"User created: {user_data.get('email')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            return False
    
    async def get_user(self, user_name: str) -> Optional[Dict[str, Any]]:
        """Get user by primary key (user_name) with legacy fallback"""
        if self.db is None:
            return None
            
        if not user_name:
            return None
            
        try:
            query = {
                "$or": [
                    {"user_name": user_name},
                    {"username": user_name}
                ]
            }
            user = await self.db.users.find_one(query)
            return user
            
        except Exception as e:
            logger.error(f"Failed to get user by name: {str(e)}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email (non-unique helper)"""
        if self.db is None or not email:
            return None
            
        try:
            return await self.db.users.find_one({"email": email})
        except Exception as e:
            logger.error(f"Failed to get user by email: {str(e)}")
            return None
    
    async def update_user(self, user_name: str, update_data: Dict[str, Any]) -> bool:
        """Update user data keyed by user_name"""
        if self.db is None or not user_name:
            return False
            
        try:
            update_data["updated_at"] = datetime.utcnow()
            
            result = await self.db.users.update_one(
                {"user_name": user_name},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update user: {str(e)}")
            return False
    
    # ====================
    # USER PAPERS MANAGEMENT
    # ====================
    
    async def add_user_paper(self, user_name: str, arxiv_id: str, category: str = None, _retry_count: int = 0) -> bool:
        """Add a paper to user's library"""
        import asyncio
        
        if self.db is None:
            return False
        
        MAX_RETRIES = 3
            
        try:
            paper_data = {
                "user_name": user_name,
                "arxiv_id": arxiv_id,
                "category": category,
                "added_at": datetime.utcnow()
            }
            
            await self.db.user_papers.update_one(
                {"user_name": user_name, "arxiv_id": arxiv_id},
                {"$set": paper_data},
                upsert=True
            )
            
            logger.debug(f"Paper added to user library: {arxiv_id}")
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle rate limit errors - retry with backoff
            if any(term in error_str for term in ['rate limit', 'too many requests', '429', 'throttl']):
                if _retry_count < MAX_RETRIES:
                    wait_time = (2 ** _retry_count) * 2  # 2, 4, 8 seconds
                    logger.info(f"Rate limit hit, waiting {wait_time}s before retry {_retry_count + 1}/{MAX_RETRIES}")
                    await asyncio.sleep(wait_time)
                    return await self.add_user_paper(user_name, arxiv_id, category, _retry_count + 1)
                else:
                    logger.error("Max retries reached for rate limit")
                    return False
            
            # Handle stale index error - drop the old index and retry
            if "user_id_1_paper_id_1" in str(e):
                try:
                    logger.info("Dropping stale index user_id_1_paper_id_1 and retrying...")
                    await self.db.user_papers.drop_index("user_id_1_paper_id_1")
                    # Also clean up any orphaned documents with old schema
                    await self.db.user_papers.delete_many({
                        "$or": [
                            {"user_id": {"$exists": True}},
                            {"paper_id": {"$exists": True}},
                            {"user_name": None},
                            {"arxiv_id": None}
                        ]
                    })
                    # Retry the insert
                    paper_data = {
                        "user_name": user_name,
                        "arxiv_id": arxiv_id,
                        "category": category,
                        "added_at": datetime.utcnow()
                    }
                    await self.db.user_papers.update_one(
                        {"user_name": user_name, "arxiv_id": arxiv_id},
                        {"$set": paper_data},
                        upsert=True
                    )
                    logger.info("Successfully added paper after dropping stale index")
                    return True
                except Exception as retry_error:
                    logger.error(f"Failed to add user paper after index cleanup: {str(retry_error)}")
                    return False
            
            logger.error(f"Failed to add user paper: {str(e)}")
            return False
    
    async def remove_user_paper(self, user_name: str, arxiv_id: str) -> bool:
        """Remove a paper from user's library"""
        if self.db is None:
            return False
            
        try:
            result = await self.db.user_papers.delete_one(
                {"user_name": user_name, "arxiv_id": arxiv_id}
            )
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to remove user paper: {str(e)}")
            return False
    
    async def get_user_papers(self, user_name: str, category: str = None) -> List[Dict[str, Any]]:
        """Get user's papers, optionally filtered by category"""
        if self.db is None:
            return []
            
        try:
            query = {"user_name": user_name}
            if category:
                query["category"] = category
            
            # Get user paper records
            cursor = self.db.user_papers.find(query).sort("added_at", -1)
            user_papers = await cursor.to_list(length=None)
            
            # Get full paper details
            if user_papers:
                arxiv_ids = [up["arxiv_id"] for up in user_papers]
                papers_cursor = self.db.papers.find({"arxiv_id": {"$in": arxiv_ids}})
                papers = await papers_cursor.to_list(length=None)
                
                # Create a mapping for quick lookup
                papers_dict = {p["arxiv_id"]: p for p in papers}
                
                # Combine user paper info with full paper details
                result = []
                for user_paper in user_papers:
                    arxiv_id = user_paper["arxiv_id"]
                    if arxiv_id in papers_dict:
                        paper = papers_dict[arxiv_id].copy()
                        paper["user_category"] = user_paper.get("category")
                        paper["added_at"] = user_paper["added_at"]
                        result.append(paper)
                
                return result
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get user papers: {str(e)}")
            return []
    
    async def get_user_paper_categories(self, user_name: str) -> List[str]:
        """Get unique categories from user's papers"""
        if self.db is None:
            return []
            
        try:
            pipeline = [
                {"$match": {"user_name": user_name, "category": {"$ne": None}}},
                {"$group": {"_id": "$category"}},
                {"$sort": {"_id": 1}}
            ]
            
            cursor = self.db.user_papers.aggregate(pipeline)
            categories = await cursor.to_list(length=None)
            
            return [cat["_id"] for cat in categories if cat["_id"]]
            
        except Exception as e:
            logger.error(f"Failed to get user paper categories: {str(e)}")
            return []
    
    # ====================
    # CHAT SESSIONS MANAGEMENT
    # ====================
    
    async def create_chat_session(self, user_name: str, session_data: Dict[str, Any]) -> str:
        """Create a new chat session"""
        if self.db is None:
            return ""
            
        try:
            session_data.update({
                "user_name": user_name,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db.chat_sessions.insert_one(session_data)
            session_id = str(result.inserted_id)
            
            logger.debug(f"Chat session created: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create chat session: {str(e)}")
            return ""
    
    async def get_chat_session(self, session_id: str, user_name: str = None) -> Optional[Dict[str, Any]]:
        """Get a chat session"""
        if self.db is None:
            return None
            
        try:
            from bson import ObjectId
            
            query = {"_id": ObjectId(session_id)}
            if user_name:
                query["user_name"] = user_name
            
            session = await self.db.chat_sessions.find_one(query)
            if session:
                session["_id"] = str(session["_id"])
                
            return session
            
        except Exception as e:
            logger.error(f"Failed to get chat session: {str(e)}")
            return None
    
    async def update_chat_session(self, session_id: str, update_data: Dict[str, Any], user_name: str = None) -> bool:
        """Update a chat session"""
        if self.db is None:
            return False
            
        try:
            from bson import ObjectId
            
            query = {"_id": ObjectId(session_id)}
            if user_name:
                query["user_name"] = user_name
            
            update_data["updated_at"] = datetime.utcnow()
            
            result = await self.db.chat_sessions.update_one(
                query,
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update chat session: {str(e)}")
            return False
    
    async def get_user_chat_sessions(self, user_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's recent chat sessions"""
        if self.db is None:
            return []
            
        try:
            cursor = self.db.chat_sessions.find(
                {"user_name": user_name}
            ).sort("updated_at", -1).limit(limit)
            
            sessions = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for session in sessions:
                session["_id"] = str(session["_id"])
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get user chat sessions: {str(e)}")
            return []
    
    async def get_active_chat_sessions(self, user_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's active chat sessions within the last 24 hours (not expired)"""
        if self.db is None:
            return []
            
        try:
            now = datetime.utcnow()
            
            # Find sessions that haven't expired yet
            cursor = self.db.chat_sessions.find({
                "user_id": user_name,
                "expires_at": {"$gt": now}
            }).sort("last_activity", -1).limit(limit)
            
            sessions = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and add message count
            for session in sessions:
                session["_id"] = str(session["_id"])
                session["message_count"] = len(session.get("messages", []))
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get active chat sessions: {str(e)}")
            return []
    
    async def extend_chat_session_ttl(self, session_id: str, hours: int = 24) -> bool:
        """Extend the TTL of a chat session by the specified hours"""
        if self.db is None:
            return False
            
        try:
            new_expiry = datetime.utcnow() + timedelta(hours=hours)
            
            result = await self.db.chat_sessions.update_one(
                {"session_id": session_id},
                {"$set": {
                    "expires_at": new_expiry,
                    "last_activity": datetime.utcnow()
                }}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to extend chat session TTL: {str(e)}")
            return False
    
    # ====================
    # DAILY ANALYSIS MANAGEMENT
    # ====================
    
    async def save_daily_analysis(self, user_id: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save daily analysis for a user, replacing any existing analysis for today"""
        if self.db is None:
            return {"success": False, "message": "Database not connected"}
            
        try:
            # Get today's date boundaries
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            # Delete any existing analysis for today (enforce one per day rule)
            await self.db.daily_analysis.delete_many({
                "user_id": user_id,
                "generated_at": {
                    "$gte": start_of_day.isoformat(),
                    "$lte": end_of_day.isoformat()
                }
            })
            
            # Delete yesterday's analysis to keep only current analysis
            yesterday = today - timedelta(days=1)
            start_of_yesterday = datetime.combine(yesterday, datetime.min.time())
            end_of_yesterday = datetime.combine(yesterday, datetime.max.time())
            
            deleted_result = await self.db.daily_analysis.delete_many({
                "user_id": user_id,
                "generated_at": {
                    "$gte": start_of_yesterday.isoformat(),
                    "$lte": end_of_yesterday.isoformat()
                }
            })
            
            if deleted_result.deleted_count > 0:
                logger.info(f"Deleted {deleted_result.deleted_count} previous daily analysis records for user {user_id}")
            
            # Add metadata
            analysis_data.update({
                "user_id": user_id,
                "analysis_type": "daily_analysis",
                "created_at": datetime.utcnow(),
                "generated_at": datetime.utcnow().isoformat()
            })
            
            # Insert new analysis
            result = await self.db.daily_analysis.insert_one(analysis_data)
            analysis_id = str(result.inserted_id)
            
            logger.info(f"Saved daily analysis for user {user_id}, analysis_id: {analysis_id}")
            return {
                "success": True,
                "analysis_id": analysis_id,
                "message": "Daily analysis saved successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to save daily analysis for user {user_id}", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def get_latest_daily_analysis(self, user_id: str) -> Dict[str, Any]:
        """Get the latest daily analysis for a user"""
        if self.db is None:
            return {"success": False, "message": "Database not connected"}
            
        try:
            # Get the most recent daily analysis
            analysis = await self.db.daily_analysis.find_one(
                {"user_id": user_id, "analysis_type": "daily_analysis"},
                sort=[("created_at", -1)]
            )
            
            if analysis:
                # Convert ObjectId to string
                analysis["_id"] = str(analysis["_id"])
                
                return {
                    "success": True,
                    "analysis": {
                        "id": analysis["_id"],
                        "data": analysis,
                        "created_at": analysis["created_at"]
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "No daily analysis found for user"
                }
                
        except Exception as e:
            logger.error(f"Failed to get latest daily analysis for user {user_id}", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def schedule_daily_job(self, user_id: str, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a daily job for a user"""
        if self.db is None:
            return {"success": False, "message": "Database not connected"}
            
        try:
            job_data = {
                "user_id": user_id,
                "job_type": "daily_dose",
                "config": job_config,
                "status": "active",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Upsert the job configuration
            await self.db.cron_jobs.update_one(
                {"user_id": user_id, "job_type": "daily_dose"},
                {"$set": job_data},
                upsert=True
            )
            
            logger.info(f"Scheduled daily job for user {user_id}")
            return {"success": True, "message": "Daily job scheduled successfully"}
            
        except Exception as e:
            logger.error(f"Failed to schedule daily job for user {user_id}", error=str(e))
            return {"success": False, "message": str(e)}
    
    # ====================
    # CACHING METHODS (MongoDB TTL)
    # ====================
    
    async def cache_set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set a cache value using MongoDB TTL index"""
        if self.db is None:
            return False
            
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            await self.db.cache.update_one(
                {"key": key},
                {
                    "$set": {
                        "value": value,
                        "expires_at": expires_at,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
            
        except Exception as e:
            logger.debug(f"Failed to set cache: {str(e)}")
            return False
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get a cache value from MongoDB"""
        if self.db is None:
            return None
            
        try:
            result = await self.db.cache.find_one({"key": key})
            
            # Check if expired (extra safety, TTL index should handle this)
            if result and result.get("expires_at"):
                if result["expires_at"] < datetime.utcnow():
                    # Already expired, delete it
                    await self.db.cache.delete_one({"key": key})
                    return None
            
            return result.get("value") if result else None
            
        except Exception as e:
            logger.debug(f"Failed to get cache: {str(e)}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete a cache value from MongoDB"""
        if self.db is None:
            return False
            
        try:
            result = await self.db.cache.delete_one({"key": key})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.debug(f"Failed to delete cache: {str(e)}")
            return False
    
    # ====================
    # SYNCHRONOUS WRAPPERS (from sync_db_wrapper.py)
    # ====================
    
    def _run_async_in_thread(self, coro):
        """Run async coroutine in a separate thread with its own event loop"""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                return result
            finally:
                loop.close()
        
        future = self.executor.submit(run_in_thread)
        return future.result(timeout=30)
    
    def save_paper_text_sync(self, paper_id: str, title: str, content: str) -> bool:
        """Synchronous wrapper for save_paper_text"""
        async def save_operation():
            if self.db is None:
                # Create separate client for sync operations
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                expires_at = datetime.utcnow() + timedelta(hours=2)
                
                paper_doc = {
                    "paper_id": paper_id,
                    "title": title,
                    "content": content,
                    "created_at": datetime.utcnow(),
                    "expires_at": expires_at
                }
                
                await db.paper_texts.replace_one(
                    {"paper_id": paper_id},
                    paper_doc,
                    upsert=True
                )
                
                if self.db is None:
                    client.close()
                return True
                
            except Exception as e:
                logger.error("Failed to save paper text (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return False
        
        try:
            return self._run_async_in_thread(save_operation())
        except Exception as e:
            logger.error("Failed to save paper text (sync)", error=str(e))
            return False
    
    def save_paper_embeddings_sync(self, paper_id: str, embeddings_data: list) -> bool:
        """Synchronous wrapper for save_paper_embeddings"""
        async def save_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                expires_at = datetime.utcnow() + timedelta(hours=2)
                
                # Delete existing embeddings for this paper
                await db.paper_embeddings.delete_many({"paper_id": paper_id})
                
                # Prepare embedding documents
                embedding_docs = []
                for i, embedding_data in enumerate(embeddings_data):
                    doc = {
                        "paper_id": paper_id,
                        "chunk_id": f"{paper_id}_chunk_{i}",
                        "text": embedding_data["text"],
                        "embedding": embedding_data["embedding"],
                        "metadata": embedding_data.get("metadata", {}),
                        "created_at": datetime.utcnow(),
                        "expires_at": expires_at
                    }
                    embedding_docs.append(doc)
                
                if embedding_docs:
                    await db.paper_embeddings.insert_many(embedding_docs)
                
                if self.db is None:
                    client.close()
                return True
                
            except Exception as e:
                logger.error("Failed to save paper embeddings (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return False
        
        try:
            return self._run_async_in_thread(save_operation())
        except Exception as e:
            logger.error("Failed to save paper embeddings (sync)", error=str(e))
            return False
    
    def save_chat_session_sync(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Synchronous wrapper for save_chat_session"""
        async def save_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                expires_at = datetime.utcnow() + timedelta(hours=2)
                
                session_doc = {
                    "session_id": session_id,
                    "user_id": session_data.get("user_id", "default"),
                    "conversation_history": session_data.get("conversation_history", []),
                    "paper_id": session_data.get("paper_id"),
                    "created_at": datetime.utcnow(),
                    "expires_at": expires_at
                }
                
                await db.chat_sessions.replace_one(
                    {"session_id": session_id},
                    session_doc,
                    upsert=True
                )
                
                if self.db is None:
                    client.close()
                return True
                
            except Exception as e:
                logger.error("Failed to save chat session (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return False
        
        try:
            return self._run_async_in_thread(save_operation())
        except Exception as e:
            logger.error("Failed to save chat session (sync)", error=str(e))
            return False
    
    def load_chat_session_sync(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for load_chat_session"""
        async def load_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                session = await db.chat_sessions.find_one({"session_id": session_id})
                
                if session and session.get("expires_at", datetime.utcnow()) > datetime.utcnow():
                    if self.db is None:
                        client.close()
                    return session
                else:
                    if session:
                        await db.chat_sessions.delete_one({"session_id": session_id})
                    if self.db is None:
                        client.close()
                    return None
                    
            except Exception as e:
                logger.error("Failed to load chat session (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return None
        
        try:
            return self._run_async_in_thread(load_operation())
        except Exception as e:
            logger.error("Failed to load chat session (sync)", error=str(e))
            return None
    
    def clear_chat_session_sync(self, session_id: str) -> bool:
        """Synchronous wrapper for clear_chat_session"""
        async def clear_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                await db.chat_sessions.delete_one({"session_id": session_id})
                if self.db is None:
                    client.close()
                return True
                
            except Exception as e:
                logger.error("Failed to clear chat session (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return False
        
        try:
            return self._run_async_in_thread(clear_operation())
        except Exception as e:
            logger.error("Failed to clear chat session (sync)", error=str(e))
            return False
    
    def get_paper_embeddings_sync(self, paper_id: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_paper_embeddings"""
        async def get_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                current_time = datetime.utcnow()
                embeddings = []
                
                async for doc in db.paper_embeddings.find({
                    "paper_id": paper_id,
                    "expires_at": {"$gt": current_time}
                }):
                    embeddings.append(doc)
                
                if self.db is None:
                    client.close()
                return embeddings
                
            except Exception as e:
                logger.error("Failed to get paper embeddings (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return []
        
        try:
            return self._run_async_in_thread(get_operation())
        except Exception as e:
            logger.error("Failed to get paper embeddings (sync)", error=str(e))
            return []
    
    def save_user_paper_sync(self, user_id: str, paper_data: Dict[str, Any]) -> bool:
        """Synchronous wrapper for save_user_paper"""
        async def save_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                expires_at = datetime.utcnow() + timedelta(hours=24)
                
                paper_doc = {
                    "user_id": user_id,
                    "paper_id": paper_data.get("paper_id"),
                    "title": paper_data.get("title"),
                    "authors": paper_data.get("authors", []),
                    "categories": paper_data.get("categories", []),
                    "abstract": paper_data.get("abstract"),
                    "published": paper_data.get("published"),
                    "url": paper_data.get("url"),
                    "added_at": datetime.utcnow(),
                    "expires_at": expires_at
                }
                
                await db.user_papers.replace_one(
                    {"user_id": user_id, "paper_id": paper_data.get("paper_id")},
                    paper_doc,
                    upsert=True
                )
                
                if self.db is None:
                    client.close()
                return True
                
            except Exception as e:
                logger.error("Failed to save user paper (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return False
        
        try:
            return self._run_async_in_thread(save_operation())
        except Exception as e:
            logger.error("Failed to save user paper (sync)", error=str(e))
            return False
    
    def get_user_papers_sync(self, user_id: str, category: str = None) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_user_papers"""
        async def get_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                query = {"user_id": user_id, "expires_at": {"$gt": datetime.utcnow()}}
                
                if category:
                    query["categories"] = {"$in": [category]}
                
                papers = []
                async for paper in db.user_papers.find(query).sort("added_at", -1):
                    papers.append(paper)
                
                if self.db is None:
                    client.close()
                return papers
                
            except Exception as e:
                logger.error("Failed to get user papers (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return []
        
        try:
            return self._run_async_in_thread(get_operation())
        except Exception as e:
            logger.error("Failed to get user papers (sync)", error=str(e))
            return []
    
    def get_user_paper_categories_sync(self, user_id: str) -> List[str]:
        """Synchronous wrapper for get_user_paper_categories"""
        async def get_operation():
            if self.db is None:
                client = AsyncIOMotorClient(self.db_url)
                db = client[self.database_name]
            else:
                db = self.db
            
            try:
                pipeline = [
                    {"$match": {"user_id": user_id, "expires_at": {"$gt": datetime.utcnow()}}},
                    {"$unwind": "$categories"},
                    {"$group": {"_id": "$categories"}},
                    {"$sort": {"_id": 1}}
                ]
                
                categories = []
                async for doc in db.user_papers.aggregate(pipeline):
                    categories.append(doc["_id"])
                
                if self.db is None:
                    client.close()
                return categories
                
            except Exception as e:
                logger.error("Failed to get user paper categories (sync)", error=str(e))
                if self.db is None:
                    client.close()
                return []
        
        try:
            return self._run_async_in_thread(get_operation())
        except Exception as e:
            logger.error("Failed to get user paper categories (sync)", error=str(e))
            return []


# Global instances
unified_database_service = UnifiedDatabaseService()

# Backwards compatibility
database_client = unified_database_service
sync_db = unified_database_service

# Export commonly used methods and aliases
database_service = unified_database_service
save_paper = unified_database_service.save_paper
get_paper = unified_database_service.get_paper
search_papers = unified_database_service.search_papers
create_user = unified_database_service.create_user
get_user = unified_database_service.get_user
get_user_by_email = unified_database_service.get_user_by_email
create_chat_session = unified_database_service.create_chat_session
get_chat_session = unified_database_service.get_chat_session

__all__ = [
    'UnifiedDatabaseService',
    'unified_database_service',
    'database_service',
    'database_client',
    'sync_db',
    'save_paper',
    'get_paper',
    'search_papers',
    'create_user',
    'get_user',
    'get_user_by_email',
    'create_chat_session',
    'get_chat_session'
]