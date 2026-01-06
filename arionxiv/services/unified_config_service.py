"""
Unified Config Service for ArionXiv
Consolidates config.py and logging_config.py
Provides comprehensive configuration management and logging setup
"""

import os
import structlog
import logging
from typing import List, Dict, Any, Optional
from datetime import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class UnifiedConfigService:
    """
    Comprehensive configuration service that handles:
    1. System configuration management (config.py functionality)
    2. Logging configuration and verbosity control (logging_config.py functionality)
    """
    
    def __init__(self, debug_mode: bool = False, quiet: bool = False):
        # Configuration constants - MongoDB URI for local development only
        # End users use the hosted Vercel API, so this is not required
        self.MONGODB_URI = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
        # Silent - end users don't need local MongoDB
        self.DATABASE_NAME = os.getenv("DATABASE_NAME", "arionxiv")
        
        # MongoDB Connection Settings
        self.MONGODB_CONNECT_TIMEOUT = int(os.getenv("MONGODB_CONNECT_TIMEOUT", "30000"))
        self.MONGODB_SERVER_SELECTION_TIMEOUT = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT", "30000"))
        self.MONGODB_SOCKET_TIMEOUT = int(os.getenv("MONGODB_SOCKET_TIMEOUT", "30000"))
        self.MONGODB_MAX_POOL_SIZE = int(os.getenv("MONGODB_MAX_POOL_SIZE", "10"))
        self.MONGODB_RETRY_WRITES = os.getenv("MONGODB_RETRY_WRITES", "true").lower() == "true"
        self.MONGODB_RETRY_READS = os.getenv("MONGODB_RETRY_READS", "true").lower() == "true"
        
        # All Collections
        self.USERS_COLLECTION = "users"
        self.PAPERS_COLLECTION = "papers"
        self.DAILY_ANALYSIS_COLLECTION = "daily_analysis"
        self.ANALYSIS_RESULTS_COLLECTION = "analysis_results"
        self.USER_PAPERS_COLLECTION = "user_papers"
        self.CRON_JOBS_COLLECTION = "cron_jobs"
        self.RAG_VECTOR_COLLECTION = "paper_embeddings"
        self.RAG_CHAT_COLLECTION = "chat_sessions"
        
        # ArXiv Configuration
        self.ARXIV_MAX_RESULTS_PER_QUERY = 50
        self.ARXIV_SEARCH_DAYS_BACK = 7
        self.ARXIV_DEFAULT_CATEGORIES = [
            "cs.CL", "cs.LG", "cs.AI", "stat.ML", "cs.CV", "cs.NE"
        ]
        
        # Daily Cron Job Configuration
        self.DAILY_CRON_HOUR = 6
        self.DAILY_CRON_MINUTE = 0
        self.TIMEZONE = "UTC"
        
        # Analysis Configuration
        self.ANALYSIS_BATCH_SIZE = 5
        self.ANALYSIS_TIMEOUT_SECONDS = 60
        
        # LLM Configuration
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.LLM_MAX_TOKENS = 4000
        self.LLM_TEMPERATURE = 0.3
        
        # Embedding Model Configuration
        # Primary: Google Gemini (FREE API, requires GEMINI_API_KEY)
        # Fallbacks: Small HuggingFace models that run locally
        self.EMBEDDING_PRIMARY_MODEL = "gemini"  # Use Gemini embedding-001 (FREE)
        self.EMBEDDING_FALLBACK_1 = "ibm-granite/granite-embedding-30m-english"  # ~120MB
        self.EMBEDDING_FALLBACK_2 = "all-MiniLM-L6-v2"  # ~80MB, fast
        self.EMBEDDING_DIMENSION_DEFAULT = 768  # Gemini dimension
        self.EMBEDDING_ENABLE_GEMINI = True
        self.EMBEDDING_ENABLE_HUGGINGFACE = True
        self.EMBEDDING_BATCH_SIZE = 10
        self.EMBEDDING_CACHE_ENABLED = True
        
        # RAG System Configuration
        self.RAG_CHUNK_SIZE = 1000
        self.RAG_CHUNK_OVERLAP = 400
        self.RAG_TOP_K_RESULTS = 10  # Increased from 5 for richer context
        self.RAG_TTL_HOURS = 24
        self.RAG_VECTOR_COLLECTION = self.RAG_VECTOR_COLLECTION
        self.RAG_CHAT_COLLECTION = self.RAG_CHAT_COLLECTION
        
        # User Preferences Default Categories
        self.DEFAULT_USER_CATEGORIES = ["cs.LG", "cs.AI"]
        
        # Paper Storage Configuration
        self.PAPER_PDF_STORAGE_DAYS = 30
        
        self.debug_mode = debug_mode or os.getenv("ARIONXIV_DEBUG", "false").lower() == "true"
        self.quiet_mode = quiet or os.getenv("ARIONXIV_QUIET", "false").lower() == "true"
        
        self.setup_logging()
        
        logging.getLogger(__name__).info("UnifiedConfigService initialized")
    
    # ============================================================
    # CONFIGURATION MANAGEMENT (from config.py)
    # ============================================================
    
    def get_mongodb_uri(self) -> str:
        """
        Get MongoDB URI with environment variable override

        Purpose: Retrieve the MongoDB connection URI, allowing for
        overrides via environment variables for flexibility across
        different deployment environments.
        """
        return os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL") or self.MONGODB_URI
    
    def get_database_name(self) -> str:
        """
        Get Database Name with environment variable override
        """
        return os.getenv("DATABASE_NAME", self.DATABASE_NAME)
    
    def get_mongodb_connection_config(self) -> Dict[str, Any]:
        """
        Get MongoDB connection configuration options

        Purpose: Provide a dictionary of MongoDB connection options
        to ensure consistent and optimized database connectivity
        """
        return {
            "connectTimeoutMS": self.MONGODB_CONNECT_TIMEOUT,
            "serverSelectionTimeoutMS": self.MONGODB_SERVER_SELECTION_TIMEOUT,
            "socketTimeoutMS": self.MONGODB_SOCKET_TIMEOUT,
            "maxPoolSize": self.MONGODB_MAX_POOL_SIZE,
            "retryWrites": self.MONGODB_RETRY_WRITES,
            "retryReads": self.MONGODB_RETRY_READS,
            "w": "majority",
            "authSource": "admin"
        }
    
    def get_groq_api_key(self) -> str:
        """
        Get Groq API key from environment variables

        Purpose: Retrieve the Groq API key for accessing Groq's
        language models, ensuring secure and flexible configuration
        """
        return os.getenv("GROQ_API_KEY", "")
    
    def get_gemini_api_key(self) -> str:
        """
        Get Gemini API key from environment variables

        Purpose: Retrieve the Gemini API key for accessing Gemini
        language models, ensuring secure and flexible configuration
        """
        return os.getenv("GEMINI_API_KEY", "")
    
    def get_openai_api_key(self) -> str:
        """
        Get OpenAI API key from environment variables

        Purpose: Retrieve the OpenAI API key for accessing OpenAI
        language models, ensuring secure and flexible configuration
        """
        return os.getenv("OPENAI_API_KEY", "")
    
    def get_cron_schedule(self) -> Dict[str, Any]:
        """
        Get daily cron job schedule configuration

        Purpose: Provide the scheduling configuration for daily
        cron jobs, allowing customization of execution time and timezone
        """
        return {
            "hour": int(os.getenv("DAILY_CRON_HOUR", self.DAILY_CRON_HOUR)),
            "minute": int(os.getenv("DAILY_CRON_MINUTE", self.DAILY_CRON_MINUTE)),
            "timezone": os.getenv("TIMEZONE", self.TIMEZONE)
        }
    
    def get_arxiv_config(self) -> Dict[str, Any]:
        """
        Get ArXiv configuration

        Purpose: Provide configuration settings for ArXiv API queries
        such as maximum results, search duration, and default categories
        """
        return {
            "max_results_per_query": int(os.getenv("ARXIV_MAX_RESULTS", self.ARXIV_MAX_RESULTS_PER_QUERY)),
            "search_days_back": int(os.getenv("ARXIV_SEARCH_DAYS", self.ARXIV_SEARCH_DAYS_BACK)),
            "default_categories": self.ARXIV_DEFAULT_CATEGORIES
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis configuration

        Purpose: Provide configuration settings for document analysis
        such as batch size, timeout, LLM model, max tokens, and temperature
        """
        return {
            "batch_size": int(os.getenv("ANALYSIS_BATCH_SIZE", self.ANALYSIS_BATCH_SIZE)),
            "timeout_seconds": int(os.getenv("ANALYSIS_TIMEOUT", self.ANALYSIS_TIMEOUT_SECONDS)),
            "llm_model": os.getenv("LLM_MODEL", self.LLM_MODEL),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", self.LLM_MAX_TOKENS)),
            "temperature": float(os.getenv("LLM_TEMPERATURE", self.LLM_TEMPERATURE))
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding model configuration

        Purpose: Provide configuration settings for embedding models
        including primary and fallback models, dimensions, and caching
        """
        return {
            "primary_model": os.getenv("EMBEDDING_PRIMARY_MODEL", self.EMBEDDING_PRIMARY_MODEL),
            "fallback_1": os.getenv("EMBEDDING_FALLBACK_1", self.EMBEDDING_FALLBACK_1),
            "fallback_2": os.getenv("EMBEDDING_FALLBACK_2", self.EMBEDDING_FALLBACK_2),
            "dimension_default": int(os.getenv("EMBEDDING_DIMENSION_DEFAULT", self.EMBEDDING_DIMENSION_DEFAULT)),
            "enable_gemini": os.getenv("EMBEDDING_ENABLE_GEMINI", str(self.EMBEDDING_ENABLE_GEMINI)).lower() == "true",
            "enable_huggingface": os.getenv("EMBEDDING_ENABLE_HUGGINGFACE", str(self.EMBEDDING_ENABLE_HUGGINGFACE)).lower() == "true",
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", self.EMBEDDING_BATCH_SIZE)),
            "cache_enabled": os.getenv("EMBEDDING_CACHE_ENABLED", str(self.EMBEDDING_CACHE_ENABLED)).lower() == "true"
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG system configuration

        Purpose: Provide configuration settings for the Retrieval-Augmented Generation (RAG) system including chunk size, overlap, top-k results, TTL, and collection names
        """
        return {
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", self.RAG_CHUNK_SIZE)),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", self.RAG_CHUNK_OVERLAP)),
            "top_k_results": int(os.getenv("RAG_TOP_K_RESULTS", self.RAG_TOP_K_RESULTS)),
            "ttl_hours": int(os.getenv("RAG_TTL_HOURS", self.RAG_TTL_HOURS)),
            "vector_collection": os.getenv("RAG_VECTOR_COLLECTION", self.RAG_VECTOR_COLLECTION),
            "chat_collection": os.getenv("RAG_CHAT_COLLECTION", self.RAG_CHAT_COLLECTION)
        }
    
    # ====================
    # LOGGING CONFIGURATION (from logging_config.py)
    # ====================
    
    def setup_logging(self):
        """
        Configure structured logging with appropriate verbosity
        
        Purpose: Set up structured logging using structlog, allowing
        for different verbosity levels based on debug and quiet modes
        """
        
        # Set different log levels based on debug mode
        if self.quiet_mode:
            log_level = logging.ERROR  # Only show errors in quiet mode
            init_message_level = "debug"
        elif self.debug_mode:
            log_level = logging.DEBUG
            init_message_level = "info"
        else:
            log_level = logging.WARNING  # Only show warnings and errors in normal mode
            init_message_level = "debug"
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if self.debug_mode else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Set root logger level
        logging.basicConfig(level=log_level)
        
        # Suppress specific noisy loggers in normal mode
        if not self.debug_mode:
            # Hide HTTP client messages
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("aiohttp").setLevel(logging.WARNING)
            
            # Hide model loading messages
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            
            # Hide embedding provider messages
            logging.getLogger("embedding_service").setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a configured logger"""
        return logging.getLogger(name)
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.debug_mode
    
    def enable_debug_mode(self):
        """Enable debug mode for verbose logging"""
        self.debug_mode = True
        self.quiet_mode = False
        self.setup_logging()
    
    def enable_quiet_mode(self):
        """Enable quiet mode - minimal logging"""
        self.debug_mode = False
        self.quiet_mode = True
        self.setup_logging()
    
    # ============================================================
    # ENHANCED CONFIGURATION METHODS
    # ============================================================
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Complete configuration dictionary

        Purpose: Provide a comprehensive dictionary of all configuration settings
        """

        return {
            "database": {
                "uri": self.get_mongodb_uri(),
                "name": self.get_database_name(),
                "collections": {
                    "users": self.USERS_COLLECTION,
                    "papers": self.PAPERS_COLLECTION,
                    "daily_analysis": self.DAILY_ANALYSIS_COLLECTION,
                    "analysis_results": self.ANALYSIS_RESULTS_COLLECTION,
                    "user_papers": self.USER_PAPERS_COLLECTION,
                    "cron_jobs": self.CRON_JOBS_COLLECTION
                }
            },
            "arxiv": self.get_arxiv_config(),
            "analysis": self.get_analysis_config(),
            "cron": self.get_cron_schedule(),
            "logging": {
                "debug_mode": self.debug_mode,
                "quiet_mode": self.quiet_mode
            }
        }
    
    def update_config(self, config_section: str, updates: Dict[str, Any]):
        """
        Update configuration settings dynamically
        """
        
        if config_section == "arxiv":
            if "max_results_per_query" in updates:
                self.ARXIV_MAX_RESULTS_PER_QUERY = updates["max_results_per_query"]
            if "search_days_back" in updates:
                self.ARXIV_SEARCH_DAYS_BACK = updates["search_days_back"]
            if "default_categories" in updates:
                self.ARXIV_DEFAULT_CATEGORIES = updates["default_categories"]
        
        elif config_section == "analysis":
            if "batch_size" in updates:
                self.ANALYSIS_BATCH_SIZE = updates["batch_size"]
            if "timeout_seconds" in updates:
                self.ANALYSIS_TIMEOUT_SECONDS = updates["timeout_seconds"]
            if "llm_model" in updates:
                self.LLM_MODEL = updates["llm_model"]
            if "max_tokens" in updates:
                self.LLM_MAX_TOKENS = updates["max_tokens"]
            if "temperature" in updates:
                self.LLM_TEMPERATURE = updates["temperature"]
        
        elif config_section == "logging":
            if "debug_mode" in updates:
                self.debug_mode = updates["debug_mode"]
                if self.debug_mode:
                    self.enable_debug_mode()
            if "quiet_mode" in updates:
                self.quiet_mode = updates["quiet_mode"]
                if self.quiet_mode:
                    self.enable_quiet_mode()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validate ArXiv configuration
        if self.ARXIV_MAX_RESULTS_PER_QUERY < 1 or self.ARXIV_MAX_RESULTS_PER_QUERY > 1000:
            validation_results["warnings"].append("ARXIV_MAX_RESULTS_PER_QUERY should be between 1 and 1000")
        
        if self.ARXIV_SEARCH_DAYS_BACK < 1 or self.ARXIV_SEARCH_DAYS_BACK > 365:
            validation_results["warnings"].append("ARXIV_SEARCH_DAYS_BACK should be between 1 and 365")
        
        # Validate analysis configuration
        if self.ANALYSIS_BATCH_SIZE < 1 or self.ANALYSIS_BATCH_SIZE > 50:
            validation_results["warnings"].append("ANALYSIS_BATCH_SIZE should be between 1 and 50")
        
        if self.LLM_MAX_TOKENS < 100 or self.LLM_MAX_TOKENS > 8000:
            validation_results["warnings"].append("LLM_MAX_TOKENS should be between 100 and 8000")
        
        if self.LLM_TEMPERATURE < 0 or self.LLM_TEMPERATURE > 2:
            validation_results["warnings"].append("LLM_TEMPERATURE should be between 0 and 2")
        
        # Check for critical errors
        if not self.get_mongodb_uri():
            validation_results["errors"].append("MongoDB URI is required")
            validation_results["valid"] = False
        
        if not self.get_database_name():
            validation_results["errors"].append("Database name is required")
            validation_results["valid"] = False
        
        return validation_results


# Global instances
unified_config_service = UnifiedConfigService()

# Backwards compatibility
config = unified_config_service
logging_config = unified_config_service

# Export commonly used functions
get_mongodb_uri = unified_config_service.get_mongodb_uri
get_database_name = unified_config_service.get_database_name
get_arxiv_config = unified_config_service.get_arxiv_config
get_analysis_config = unified_config_service.get_analysis_config
get_embedding_config = unified_config_service.get_embedding_config
get_rag_config = unified_config_service.get_rag_config
get_logger = unified_config_service.get_logger
is_debug_mode = unified_config_service.is_debug
__all__ = [
    'UnifiedConfigService',
    'unified_config_service',
    'config',
    'logging_config',
    'get_mongodb_uri',
    'get_database_name',
    'get_arxiv_config',
    'get_analysis_config',
    'get_embedding_config',
    'get_rag_config',
    'get_logger',
    'is_debug_mode'
]