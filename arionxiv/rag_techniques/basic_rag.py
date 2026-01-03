"""
Basic RAG Implementation for ArionXiv
Provides standard Retrieval-Augmented Generation with text chunking, embedding generation, and vector search
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from pymongo import IndexModel
import os

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
    np = None
    SentenceTransformer = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Global cache for loaded embedding models to avoid reloading across sessions
# This persists the model in memory once loaded
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}

# Import theme system for consistent styling
try:
    from ..cli.ui.theme import (
        create_themed_console, get_theme_colors, style_text,
        print_success, print_error, print_warning, print_info
    )
    from ..cli.utils.animations import left_to_right_reveal, stream_markdown_response
    from ..cli.utils.command_suggestions import show_command_suggestions
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    def get_theme_colors(db_service=None):
        return {'primary': 'blue', 'secondary': 'cyan', 'success': 'green', 
                'warning': 'yellow', 'error': 'red', 'muted': 'dim'}
    def style_text(text, style='primary', db_service=None):
        colors = get_theme_colors()
        return f"[{colors.get(style, 'white')}]{text}[/{colors.get(style, 'white')}]"
    def create_themed_console(db_service=None):
        return Console()
    def left_to_right_reveal(console, text, style="", duration=1.0):
        console.print(text)
    def stream_markdown_response(console, text, panel_title="", border_style=None, duration=3.0):
        colors = get_theme_colors()
        actual_style = border_style or colors.get('primary', 'blue')
        console.print(Panel(Markdown(text), title=panel_title, border_style=actual_style))
    def show_command_suggestions(console, context="general", **kwargs):
        pass  # No-op fallback

# Import API config manager to check if Gemini key is available
try:
    from ..cli.utils.api_config import api_config_manager
    API_CONFIG_AVAILABLE = True
except ImportError:
    API_CONFIG_AVAILABLE = False
    api_config_manager = None

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider using embedding-001 model (FREE!)"""
    
    def __init__(self, api_key: str = None, console: Console = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = "models/embedding-001"
        self.dimension = 768
        self._console = console or Console()
        
        logger.info("Gemini embedding provider initialized with free API")
        
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Gemini API (FREE!) with rate limit handling"""
        try:
            batch_size = 10
            all_embeddings = []
            max_retries = 3
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                for text in batch:
                    retries = 0
                    while retries < max_retries:
                        try:
                            result = genai.embed_content(
                                model=self.model,
                                content=text,
                                task_type="retrieval_document"
                            )
                            batch_embeddings.append(result['embedding'])
                            await asyncio.sleep(0.1)
                            break  # Success, exit retry loop
                        except Exception as e:
                            error_str = str(e).lower()
                            # Check for rate limit errors - silently retry with backoff
                            if any(term in error_str for term in ['rate limit', 'quota', '429', 'resource exhausted', 'too many']):
                                retries += 1
                                if retries < max_retries:
                                    wait_time = (2 ** retries) * 2  # Exponential backoff: 4, 8, 16 seconds
                                    await asyncio.sleep(wait_time)
                                else:
                                    # Max retries reached, use fallback
                                    batch_embeddings.append([0.0] * self.dimension)
                            else:
                                logger.debug(f"Failed to embed text: {str(e)}")
                                batch_embeddings.append([0.0] * self.dimension)
                                break
                
                all_embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Gemini embedding failed: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def get_name(self) -> str:
        return "Google-Gemini-Embedding-001-FREE"


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers (fallback)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ML_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "ML dependencies not installed. Install with: pip install sentence-transformers numpy"
            )
        self.model_name = model_name
        self.model = None
        self._dimension = None
        self._console = Console()
        
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            colors = get_theme_colors()
            self._console.print(f"[{colors['muted']}]Loading fallback model: {self.model_name}[/{colors['muted']}]")
            
            # Suppress HuggingFace's internal progress bars
            import os
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self._console,
                transient=True
            ) as progress:
                task = progress.add_task(
                    f"[{colors['primary']}]Loading model...[/{colors['primary']}]", 
                    total=None
                )
                self.model = SentenceTransformer(self.model_name)
                self._dimension = self.model.get_sentence_embedding_dimension()
            
            # Re-enable progress bars for other operations
            os.environ.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)
            
            self._console.print(f"[{colors['primary']}][OK][/{colors['primary']}] Fallback model ready")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using HuggingFace model"""
        try:
            self._load_model()
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self.model.encode, texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def get_name(self) -> str:
        return f"HuggingFace-{self.model_name}"


class GraniteDoclingEmbeddingProvider(EmbeddingProvider):
    """
    IBM Granite embedding provider - small, fast, and runs locally
    
    Downloads the model on first use. Model is kept in memory during 
    the session and uses HuggingFace's default cache (~/.cache/huggingface/).
    """
    
    # Default model - IBM Granite 30M English (small, ~120MB download)
    DEFAULT_MODEL = "ibm-granite/granite-embedding-30m-english"
    
    def __init__(self, model_name: str = None):
        if not ML_DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "ML dependencies not installed. Install with: pip install sentence-transformers numpy"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self._dimension = None
        self._console = Console()
        
    @property
    def model(self):
        """Get model from global cache or None if not loaded"""
        return _GLOBAL_MODEL_CACHE.get(self.model_name)
    
    @model.setter
    def model(self, value):
        """Store model in global cache"""
        if value is not None:
            _GLOBAL_MODEL_CACHE[self.model_name] = value
        
    def _load_model(self):
        """Lazy load the model with progress indicator - uses global cache"""
        # Check global cache first - model persists across sessions
        if self.model_name in _GLOBAL_MODEL_CACHE:
            self._dimension = _GLOBAL_MODEL_CACHE[self.model_name].get_sentence_embedding_dimension()
            return  # Model already in memory, no loading needed
            
        colors = get_theme_colors()
        logger.info(f"Loading embedding model: {self.model_name}")
        
        # Check if model is already cached by HuggingFace on disk
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_name = f"models--{self.model_name.replace('/', '--')}"
        is_cached = (cache_dir / model_cache_name).exists()
        
        if is_cached:
            # Model is cached on disk - quick load message
            self._console.print(
                f"[{colors['muted']}]Loading embedding model from cache...[/{colors['muted']}]"
            )
        else:
            # First time - show download message
            self._console.print(
                f"[{colors['muted']}]Downloading embedding model: {self.model_name}[/{colors['muted']}]"
            )
            self._console.print(
                f"[{colors['muted']}](First run downloads ~120MB, uses HuggingFace cache)[/{colors['muted']}]"
            )
        
        try:
            # Suppress HuggingFace's internal progress bars to avoid flickering
            import os
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self._console,
                transient=True
            ) as progress:
                loading_msg = "Loading model from cache..." if is_cached else "Downloading and initializing embedding model..."
                task = progress.add_task(
                    f"[{colors['primary']}]{loading_msg}[/{colors['primary']}]", 
                    total=None
                )
                loaded_model = SentenceTransformer(self.model_name, trust_remote_code=True)
                self._dimension = loaded_model.get_sentence_embedding_dimension()
                # Store in global cache
                _GLOBAL_MODEL_CACHE[self.model_name] = loaded_model
            
            # Re-enable progress bars for other operations
            os.environ.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)
            
            self._console.print(
                f"[{colors['primary']}][OK][/{colors['primary']}] Embedding model ready "
                f"(dimension: {self._dimension})"
            )
            logger.info(f"Embedding model loaded successfully (dimension: {self._dimension})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {str(e)}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the configured embedding model"""
        try:
            self._load_model()
            model = _GLOBAL_MODEL_CACHE.get(self.model_name)
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, model.encode, texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed for {self.model_name}: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def get_name(self) -> str:
        return f"Granite-{self.model_name.split('/')[-1]}"


class BasicRAG:
    """
    Basic RAG (Retrieval-Augmented Generation) implementation
    Handles text chunking, embedding generation, vector search, and context retrieval
    """
    
    def __init__(self, database_service, config_service, llm_client, openrouter_client=None):
        """
        Initialize BasicRAG with required services
        
        Args:
            database_service: Database service for storing/retrieving embeddings
            config_service: Configuration service for RAG settings
            llm_client: LLM client for generating responses (Groq - fallback)
            openrouter_client: OpenRouter client for primary LLM (Kimi K2)
        """
        self.db_service = database_service
        self.config_service = config_service
        self.llm_client = llm_client
        self.openrouter_client = openrouter_client
        
        # Lazy initialization flags for embedding providers
        self._embedding_providers_initialized = False
        self._embedding_providers = []
        self._current_embedding_provider = None
        
        # Use OpenRouter as primary if available, otherwise fall back to Groq
        # Can be overridden with RAG_LLM_PROVIDER env var
        env_provider = os.getenv("RAG_LLM_PROVIDER", "").lower()
        if env_provider:
            self.llm_provider = env_provider
        elif openrouter_client and openrouter_client.is_available:
            self.llm_provider = "openrouter"
        else:
            self.llm_provider = "groq"
        
        rag_config = config_service.get_rag_config()
        embedding_config = config_service.get_embedding_config()
        
        self.vector_collection = rag_config["vector_collection"]
        self.chat_collection = rag_config["chat_collection"]
        self.chunk_size = rag_config["chunk_size"]
        self.chunk_overlap = rag_config["chunk_overlap"]
        self.top_k_results = rag_config["top_k_results"]
        self.ttl_hours = rag_config["ttl_hours"]
        
        self.embedding_batch_size = embedding_config["batch_size"]
        self.embedding_dimension = embedding_config["dimension_default"]
        self._embedding_config = embedding_config
        
        # In-memory embedding storage for current chat session
        # Format: {chunk_id: {text, embedding, metadata}}
        self._session_embeddings: Dict[str, Dict[str, Any]] = {}
        self._current_session_id: Optional[str] = None
        
        # In-memory session storage (fallback when database unavailable)
        self._in_memory_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.console = Console()
        
        logger.info("BasicRAG initialized (embedding providers lazy-loaded)")
    
    @property
    def embedding_providers(self):
        """Lazy initialize embedding providers"""
        if not self._embedding_providers_initialized:
            self._embedding_providers_initialized = True
            self._setup_embedding_providers(self._embedding_config)
        return self._embedding_providers
    
    @property
    def current_embedding_provider(self):
        """Get current embedding provider (lazy init if needed)"""
        if not self._embedding_providers_initialized:
            self._embedding_providers_initialized = True
            self._setup_embedding_providers(self._embedding_config)
        return self._current_embedding_provider
    
    @current_embedding_provider.setter
    def current_embedding_provider(self, value):
        """Set current embedding provider"""
        self._current_embedding_provider = value
    
    def _setup_embedding_providers(self, embedding_config):
        """
        Setup embedding providers in order of preference
        
        Order:
        1. Gemini (FREE API, if API key is configured)
        2. Granite/HuggingFace fallback models (run locally with 24h cache)
        
        If Gemini API key is not available, automatically falls back to
        local Granite model which is cached for 24 hours to avoid
        repeated downloads.
        """
        primary_model = embedding_config["primary_model"]
        fallback_1 = embedding_config["fallback_1"]
        fallback_2 = embedding_config["fallback_2"]
        enable_gemini = embedding_config["enable_gemini"]
        enable_huggingface = embedding_config["enable_huggingface"]
        
        # Check if Gemini API key is actually available
        gemini_key_available = False
        if API_CONFIG_AVAILABLE and api_config_manager:
            gemini_key_available = api_config_manager.is_configured("gemini")
        else:
            # Fallback: check environment variable directly
            gemini_key_available = bool(
                os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        
        # Primary: Gemini (if enabled AND API key is available)
        if enable_gemini and gemini_key_available and (primary_model.lower() == "gemini" or primary_model == ""):
            try:
                gemini_provider = GeminiEmbeddingProvider()
                self._embedding_providers.append(gemini_provider)
                logger.info("Gemini embedding provider initialized as PRIMARY (FREE API)")
            except Exception as e:
                logger.warning(f"Gemini embedding provider failed to initialize: {str(e)}")
        elif enable_gemini and not gemini_key_available:
            logger.info("Gemini API key not configured - will use local Granite model as fallback")
        
        # If Gemini is not available OR primary is a HuggingFace model, use Granite
        if enable_huggingface:
            # If Gemini failed/unavailable, Granite becomes primary
            if not self._embedding_providers:
                try:
                    # Use Granite as primary when Gemini is unavailable
                    granite_model = fallback_1 or GraniteDoclingEmbeddingProvider.DEFAULT_MODEL
                    granite_provider = GraniteDoclingEmbeddingProvider(model_name=granite_model)
                    self._embedding_providers.append(granite_provider)
                    logger.info(f"Granite embedding provider initialized as PRIMARY (local): {granite_model}")
                except Exception as e:
                    logger.warning(f"Granite embedding provider failed to initialize: {str(e)}")
            
            # If primary is explicitly a HuggingFace model (not "gemini"), add it
            elif primary_model.lower() != "gemini" and primary_model != "":
                try:
                    primary_provider = GraniteDoclingEmbeddingProvider(model_name=primary_model)
                    self._embedding_providers.append(primary_provider)
                    logger.info(f"Primary HuggingFace embedding provider initialized: {primary_model}")
                except Exception as e:
                    logger.warning(f"Primary embedding provider failed to initialize: {str(e)}")
            
            # Add fallback (Granite) if not already primary
            if fallback_1 and not any(
                isinstance(p, GraniteDoclingEmbeddingProvider) and p.model_name == fallback_1 
                for p in self._embedding_providers
            ):
                try:
                    fallback_1_provider = GraniteDoclingEmbeddingProvider(model_name=fallback_1)
                    self._embedding_providers.append(fallback_1_provider)
                    logger.info(f"Fallback embedding provider initialized: {fallback_1}")
                except Exception as e:
                    logger.warning(f"Fallback embedding provider failed: {str(e)}")
        
        if self._embedding_providers:
            self._current_embedding_provider = self._embedding_providers[0]
            logger.info(f"Using embedding provider: {self._current_embedding_provider.get_name()}")
        else:
            logger.error("No embedding providers available!")
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings with automatic fallback"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        for i, provider in enumerate(self.embedding_providers):
            try:
                embeddings = await provider.get_embeddings(texts)
                
                if provider != self.current_embedding_provider:
                    self.current_embedding_provider = provider
                    logger.info(f"Switched to embedding provider: {provider.get_name()}")
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Provider {provider.get_name()} failed: {str(e)}")
                if i == len(self.embedding_providers) - 1:
                    raise RuntimeError(f"All embedding providers failed. Last error: {str(e)}")
                continue
    
    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.current_embedding_provider:
            return self.current_embedding_provider.get_dimension()
        return self.embedding_dimension
    
    def get_embedding_provider_name(self) -> str:
        """Get current provider name"""
        if self.current_embedding_provider:
            return self.current_embedding_provider.get_name()
        return "None"
    
    def ensure_embedding_model_loaded(self):
        """Ensure the embedding model is loaded before starting batch operations.
        
        This prevents the model download progress from interfering with
        the embedding computation progress bar.
        """
        if self.current_embedding_provider:
            # Trigger model loading by calling get_dimension which internally calls _load_model
            try:
                self.current_embedding_provider.get_dimension()
            except Exception as e:
                logger.warning(f"Failed to pre-load embedding model: {e}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > self.chunk_size * 0.7:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    async def add_document_to_index(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to in-memory vector index for current session
        
        First checks if embeddings are cached in the database (24-hour TTL).
        If cached, loads them directly. Otherwise, computes and caches them.
        """
        try:
            # Check if embeddings are already cached in the database
            cached_embeddings = await self._get_cached_embeddings(doc_id)
            
            if cached_embeddings:
                # Load from cache
                await self._load_embeddings_from_cache(cached_embeddings)
                logger.info(f"Loaded {len(cached_embeddings)} cached embeddings for document {doc_id}")
                return True
            
            # No cache - compute embeddings
            chunks = self._chunk_text(text)
            embeddings = await self.get_embeddings(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                # Store in memory
                self._session_embeddings[chunk_id] = {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': metadata or {}
                }
            
            # Save to database cache for future use (24-hour TTL)
            await self._save_embeddings_to_cache(doc_id, chunks, embeddings, metadata)
            
            logger.info(f"Added {len(chunks)} chunks for document {doc_id} to in-memory index and cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to index: {str(e)}")
            return False
    
    async def add_document_to_index_with_progress(self, doc_id: str, text: str, metadata: Dict[str, Any] = None, console: Console = None) -> bool:
        """Add document to in-memory vector index with progress bar
        
        First checks if embeddings are cached in the database (24-hour TTL).
        If cached, loads them directly. Otherwise, computes and caches them.
        """
        try:
            colors = get_theme_colors()
            console = console or self.console
            
            # Check if embeddings are already cached in the database
            cached_embeddings = await self._get_cached_embeddings(doc_id)
            
            if cached_embeddings:
                # Load from cache - much faster!
                left_to_right_reveal(console, f"Loading cached embeddings ({len(cached_embeddings)} chunks)...", style=f"bold {colors['primary']}", duration=0.8)
                await self._load_embeddings_from_cache(cached_embeddings)
                
                # Note: We intentionally do NOT pre-load the embedding model here.
                # Query embeddings will use Gemini API if available (fast, no download needed).
                # The local Granite model will only be loaded lazily if Gemini fails.
                
                logger.info(f"Loaded {len(cached_embeddings)} cached embeddings for document {doc_id}")
                return True
            
            # No cache - need to compute embeddings
            # First, chunk the text
            chunks = self._chunk_text(text)
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                return False
            
            # Show subtle hint for large papers
            if total_chunks > 20:
                console.print(f"[white]Processing {total_chunks} chunks (this may take a moment for large papers)...[/white]")
            
            # Ensure embedding model is loaded BEFORE showing the computation progress bar
            # This prevents model download progress from interfering with embedding progress
            self.ensure_embedding_model_loaded()
            
            # Create progress bar for embedding computation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("-"),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task(
                    f"[bold {colors['primary']}]Computing embeddings...",
                    total=total_chunks
                )
                
                # Process chunks in batches for the API
                batch_size = 5
                all_embeddings = []
                
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_embeddings = await self.get_embeddings(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Update progress for each chunk in the batch
                    progress.update(task, advance=len(batch))
                
                # Store embeddings in memory
                for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    self._session_embeddings[chunk_id] = {
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'text': chunk,
                        'embedding': embedding,
                        'metadata': metadata or {}
                    }
            
            # Save to database cache for future use (24-hour TTL)
            await self._save_embeddings_to_cache(doc_id, chunks, all_embeddings, metadata)
            
            logger.info(f"Added {total_chunks} chunks for document {doc_id} to in-memory index and cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to index: {str(e)}")
            return False
    
    def clear_session_embeddings(self):
        """Clear in-memory embeddings when chat session ends"""
        count = len(self._session_embeddings)
        self._session_embeddings.clear()
        self._current_session_id = None
        logger.info(f"Cleared {count} embeddings from memory")
    
    async def _get_cached_embeddings(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """Check if embeddings for a document are cached in the database (24-hour TTL)"""
        try:
            # Find all embeddings for this document that haven't expired
            # Use a high limit to ensure we get all chunks (default find_many limit is 100)
            cached = await self.db_service.find_many(
                self.vector_collection,
                {
                    'doc_id': doc_id,
                    'expires_at': {'$gt': datetime.utcnow()}
                },
                limit=10000  # High limit to get all chunks for large papers
            )
            
            if cached and len(cached) > 0:
                logger.info(f"Found {len(cached)} cached embeddings for document {doc_id}")
                return cached
            return None
            
        except Exception as e:
            logger.warning(f"Failed to check cached embeddings: {str(e)}")
            return None
    
    async def _save_embeddings_to_cache(self, doc_id: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None):
        """Save embeddings to database with 24-hour TTL"""
        try:
            expires_at = datetime.utcnow() + timedelta(hours=24)
            
            # Delete any existing embeddings for this document first
            await self.db_service.delete_many(
                self.vector_collection,
                {'doc_id': doc_id}
            )
            
            # Save new embeddings
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                documents.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': metadata or {},
                    'created_at': datetime.utcnow(),
                    'expires_at': expires_at
                })
            
            if documents:
                await self.db_service.insert_many(self.vector_collection, documents)
                logger.info(f"Saved {len(documents)} embeddings to cache for document {doc_id} (expires in 24h)")
            
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {str(e)}")
    
    async def _load_embeddings_from_cache(self, cached_embeddings: List[Dict[str, Any]]):
        """Load cached embeddings into session memory"""
        for doc in cached_embeddings:
            chunk_id = doc.get('chunk_id')
            self._session_embeddings[chunk_id] = {
                'doc_id': doc.get('doc_id'),
                'chunk_id': chunk_id,
                'text': doc.get('text'),
                'embedding': doc.get('embedding'),
                'metadata': doc.get('metadata', {})
            }
        logger.info(f"Loaded {len(cached_embeddings)} embeddings from cache to session memory")

    async def search_similar_documents(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity (in-memory)"""
        try:
            query_embedding = await self.get_single_embedding(query)
            
            # Search in-memory embeddings
            scored_docs = []
            for chunk_id, doc in self._session_embeddings.items():
                # Apply metadata filters if provided
                if filters:
                    match = True
                    for key, value in filters.items():
                        # Handle nested keys like 'metadata.type'
                        keys = key.split('.')
                        doc_value = doc
                        for k in keys:
                            doc_value = doc_value.get(k, {}) if isinstance(doc_value, dict) else None
                        if doc_value != value:
                            match = False
                            break
                    if not match:
                        continue
                
                doc_embedding = doc.get('embedding', [])
                if doc_embedding:
                    score = await self.compute_similarity(query_embedding, doc_embedding)
                    scored_docs.append({
                        'doc_id': doc.get('doc_id'),
                        'chunk_id': doc.get('chunk_id'),
                        'text': doc.get('text'),
                        'metadata': doc.get('metadata', {}),
                        'score': score
                    })
            
            # Sort by score descending and take top k
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            return scored_docs[:self.top_k_results]
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            if not ML_DEPENDENCIES_AVAILABLE:
                return 0.0
            
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            return 0.0
    
    async def start_chat_session(self, papers: List[Dict[str, Any]], user_id: str = "default"):
        """Start interactive chat session with a single paper (v1)
        
        Embeddings are stored in memory during the session and cleared when done.
        Chat history is persisted to DB with 24-hour TTL for resumption.
        """
        try:
            if not papers:
                colors = get_theme_colors()
                self.console.print(f"[{colors['error']}]No papers provided for chat session[/{colors['error']}]")
                return
            
            # V1: Limit to single paper
            paper = papers[0]
            paper_id = paper.get('arxiv_id') or paper.get('id')
            
            if not paper_id:
                colors = get_theme_colors()
                self.console.print(f"[{colors['error']}]Paper has no ID[/{colors['error']}]")
                return
            
            colors = get_theme_colors()
            
            # Clear any previous session embeddings
            self.clear_session_embeddings()
            
            # Check if cached embeddings were passed (already fetched from DB)
            cached_embeddings = paper.get('_cached_embeddings')
            if cached_embeddings:
                # Load cached embeddings directly into session memory
                await self._load_embeddings_from_cache(cached_embeddings)
                logger.info(f"Loaded {len(cached_embeddings)} pre-cached embeddings for paper {paper_id}")
            else:
                # Generate embeddings and store in memory - with progress bar
                paper_text = self._extract_paper_text(paper)
                if paper_text:
                    await self.add_document_to_index_with_progress(
                        paper_id,
                        paper_text,
                        {'type': 'paper', 'title': paper.get('title', '')},
                        console=self.console
                    )
            
            # Create unique session ID
            import uuid
            session_id = f"{user_id}_{paper_id}_{uuid.uuid4().hex[:8]}"
            self._current_session_id = session_id
            
            # Create session document with 24-hour TTL
            session_doc = {
                'session_id': session_id,
                'paper_id': paper_id,  # Single paper in v1
                'paper_title': paper.get('title', ''),
                'user_id': user_id,
                'created_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(hours=24),  # 24-hour TTL
                'messages': []
            }
            
            # Store in-memory as fallback (always works)
            self._in_memory_sessions[session_id] = session_doc
            
            # Try to persist to Vercel API first (cloud storage)
            session_saved = False
            api_error = None
            try:
                from ..cli.utils.api_client import api_client
                api_result = await api_client.create_chat_session(
                    paper_id=paper_id,
                    title=paper.get('title', paper_id)
                )
                if api_result.get("success"):
                    # Update in-memory session with API session_id for consistency
                    api_session_id = api_result.get('session_id')
                    # Store the API session ID for later updates
                    self._in_memory_sessions[session_id]['api_session_id'] = api_session_id
                    logger.info(f"Chat session saved to cloud: {api_session_id}")
                    session_saved = True
                else:
                    api_error = f"API failure: {api_result}"
                    logger.warning(api_error)
            except Exception as api_err:
                api_error = f"API error: {api_err}"
                logger.warning(f"Session not saved to API: {api_err}")
            
            # Also try local database as backup (regardless of API success)
            try:
                await self.db_service.insert_one(self.chat_collection, session_doc)
                logger.info(f"Chat session saved to local DB: {session_id}")
                session_saved = True
            except Exception as db_err:
                logger.debug(f"Session not saved to local database: {db_err}")
            
            if not session_saved:
                logger.warning(f"Chat session only stored in-memory: {session_id}")
                if api_error:
                    logger.warning(f"API save failed: {api_error}")
            
            self.console.print(Panel(
                f"[bold {colors['primary']}]Chat Session Started[/bold {colors['primary']}]\n"
                f"Paper: [bold {colors['primary']}] {paper.get('title', paper_id)}[/bold {colors['primary']}]\n"
                f"Chunks indexed: [bold {colors['primary']}] {len(self._session_embeddings)}[/bold {colors['primary']}]\n"
                f"Type [bold {colors['primary']}]'quit'[/bold {colors['primary']}] or [bold {colors['primary']}]'exit'[/bold {colors['primary']}] to end the chat.",
                title=f"[bold]ArionXiv Paper Chat[/bold]",
                border_style=f"bold {colors['primary']}"
            ))
            
            try:
                await self._run_chat_loop(session_id)
            finally:
                # Always clean up embeddings when session ends
                self.clear_session_embeddings()
            
        except Exception as e:
            logger.error(f"Chat session failed: {str(e)}")
            colors = get_theme_colors()
            self.console.print(f"[{colors['error']}]Chat session failed: {str(e)}[/{colors['error']}]")
            # Clean up on error too
            self.clear_session_embeddings()
    
    async def continue_chat_session(self, session: Dict[str, Any], paper_info: Dict[str, Any]):
        """Continue an existing chat session
        
        Reloads the paper embeddings and continues the conversation.
        Extends the session TTL by 24 hours.
        """
        try:
            colors = get_theme_colors()
            session_id = session.get('session_id')
            paper_title = session.get('paper_title', 'Unknown Paper')
            messages = session.get('messages', [])
            
            if not session_id:
                self.console.print(f"[{colors['error']}]Invalid session: no session_id[/{colors['error']}]")
                return
            
            # Clear any previous session embeddings
            self.clear_session_embeddings()
            
            # Check if cached embeddings were passed
            cached_embeddings = paper_info.get('_cached_embeddings')
            if cached_embeddings:
                # Use pre-loaded cached embeddings directly
                self._session_embeddings = cached_embeddings
                logger.info(f"Loaded {len(cached_embeddings)} cached embeddings for session")
            else:
                # Re-index the paper content
                paper_text = self._extract_paper_text(paper_info)
                if paper_text:
                    paper_id = paper_info.get('arxiv_id') or paper_info.get('id')
                    await self.add_document_to_index_with_progress(
                        paper_id,
                        paper_text,
                        {'type': 'paper', 'title': paper_title},
                        console=self.console
                    )
            
            self._current_session_id = session_id
            
            # Extend the session TTL by 24 hours
            await self.db_service.extend_chat_session_ttl(session_id, hours=24)
            
            # Show session info with previous message count
            self.console.print(Panel(
                f"[bold {colors['primary']}]Continuing Chat Session[/bold {colors['primary']}]\n"
                f"Paper: [bold {colors['primary']}]{paper_title}[/bold {colors['primary']}]\n"
                f"Previous messages: [bold {colors['primary']}]{len(messages)}[/bold {colors['primary']}]\n"
                f"Chunks indexed: [bold {colors['primary']}]{len(self._session_embeddings)}[/bold {colors['primary']}]\n"
                f"Session extended by 24 hours.\n"
                f"Type [bold {colors['primary']}]'quit'[/bold {colors['primary']}] or [bold {colors['primary']}]'exit'[/bold {colors['primary']}] to end the chat.",
                title=f"[bold]ArionXiv Paper Chat - Resumed[/bold]",
                border_style=f"bold {colors['primary']}"
            ))
            
            # Show a summary of recent conversation if there are messages
            if messages:
                # Show last 8 Q&A pairs (16 messages total)
                num_pairs = min(8, len(messages) // 2)
                if num_pairs > 0:
                    recent = messages[-(num_pairs * 2):]
                else:
                    recent = messages  # Show whatever we have
                
                left_to_right_reveal(self.console, f"\nRecent conversation ({num_pairs} Q&A):", style=f"bold {colors['primary']}", duration=0.8)
                for msg in recent:
                    role = "You" if msg.get('type') == 'user' else "Assistant"
                    content = msg.get('content', '')
                    # Truncate long messages for display
                    display_content = content[:150] + "..." if len(content) > 150 else content
                    self.console.print(f"[dim {colors['primary']}]{role}: {display_content}[/dim {colors['primary']}]")
            
            try:
                await self._run_chat_loop(session_id)
            finally:
                self.clear_session_embeddings()
            
        except Exception as e:
            logger.error(f"Continue chat session failed: {str(e)}")
            colors = get_theme_colors()
            self.console.print(f"[{colors['error']}]Failed to continue session: {str(e)}[/{colors['error']}]")
            self.clear_session_embeddings()
    
    async def _run_chat_loop(self, session_id: str):
        """Run the chat interaction loop"""
        colors = get_theme_colors()
        while True:
            message = Prompt.ask(f"\n[bold {colors['primary']}]You[/bold {colors['primary']}]")
            
            if message.lower() in ['quit', 'exit', 'q']:
                left_to_right_reveal(self.console, "\nEnding chat session. Goodbye!", style=f"bold {colors['primary']}", duration=1.0)
                break
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
                task = progress.add_task(f"[bold {colors['primary']}]Thinking...", total=None)
                result = await self._chat_with_session(session_id, message)
            
            if result['success']:
                # Stream the response over 2 seconds
                stream_markdown_response(
                    self.console,
                    result['response'],
                    panel_title=f"[bold {colors['primary']}]ArionXiv Assistant[/bold {colors['primary']}]",
                    border_style=colors['primary'],
                    duration=1.0
                )
                
                # Build info line with chunks and model name
                info_parts = []
                if result['relevant_chunks'] > 0:
                    info_parts.append(f"Used {result['relevant_chunks']} relevant content chunks")
                if result.get('model_display'):
                    info_parts.append(f"• Model: {result['model_display']}")
                
                if info_parts:
                    info_text = "  ".join(info_parts)
                    left_to_right_reveal(self.console, info_text, style=f"dim {colors['muted']}", duration=1.0)
            else:
                left_to_right_reveal(self.console, f"Error: {result['error']}", style=f"bold {colors['error']}", duration=1.0)
    
    def _show_post_chat_commands(self):
        """Show helpful commands after chat session ends"""
        colors = get_theme_colors()
        
        commands = [
            ("arionxiv chat", "Start a new chat session"),
            ("arionxiv search <query>", "Search for more papers"),
            ("arionxiv settings papers", "Manage your saved papers"),
            ("arionxiv trending", "See trending papers"),
            ("arionxiv daily", "Get your daily paper digest"),
        ]
        
        self.console.print()
        self.console.print(Panel(
            "\n".join([
                f"[bold {colors['primary']}]{cmd}[/bold {colors['primary']}]  [white]→  {desc}[/white]"
                for cmd, desc in commands
            ]),
            title=f"[bold {colors['primary']}]What's Next?[/bold {colors['primary']}]",
            border_style=f"bold {colors['primary']}",
            padding=(1, 2)
        ))
    
    async def _chat_with_session(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a chat message and generate response"""
        try:
            # Try database first, fall back to in-memory
            session = None
            try:
                session = await self.db_service.find_one(self.chat_collection, {'session_id': session_id})
            except Exception:
                pass
            
            # Fall back to in-memory session
            if not session:
                session = self._in_memory_sessions.get(session_id)
            
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            relevant_chunks = await self.search_similar_documents(message, {'metadata.type': 'paper'})
            context = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:3]])
            
            # Get conversation history for context
            chat_history = session.get('messages', [])
            
            # Determine which LLM to use and generate response
            model_display = ""
            success = False
            response_text = ""
            error_msg = ""
            
            # Use OpenRouter for chat
            if self.openrouter_client and self.openrouter_client.is_available:
                try:
                    result = await self.openrouter_client.chat(
                        message=message,
                        context=context,
                        history=chat_history
                    )
                    
                    if result.get('success'):
                        response_text = result['response']
                        model_display = result.get('model_display', 'OpenRouter')
                        success = True
                    else:
                        error_msg = result.get('error', 'OpenRouter request failed')
                except Exception as openrouter_err:
                    error_str = str(openrouter_err).lower()
                    # Check for rate limit errors and provide user-friendly message
                    if 'rate' in error_str or '429' in error_str or 'limit' in error_str:
                        error_msg = "API rate limit reached. Try again later or add your own API key with: arionxiv settings api"
                    elif 'all models' in error_str:
                        error_msg = "API rate limit reached. Try again later or add your own API key with: arionxiv settings api"
                    else:
                        error_msg = f"Chat service unavailable. Try again later or add your own API key with: arionxiv settings api"
                    logger.debug(f"OpenRouter error: {openrouter_err}")
            else:
                error_msg = "API key not configured. Add your API key with: arionxiv settings api"
            
            if not success:
                return {'success': False, 'error': error_msg or 'Failed to generate response'}
            
            # Update in-memory session
            if session_id in self._in_memory_sessions:
                self._in_memory_sessions[session_id]['messages'].extend([
                    {'type': 'user', 'content': message, 'timestamp': datetime.utcnow()},
                    {'type': 'assistant', 'content': response_text, 'timestamp': datetime.utcnow()}
                ])
                self._in_memory_sessions[session_id]['last_activity'] = datetime.utcnow()
            
            # Try to persist to Vercel API (cloud storage)
            try:
                from ..cli.utils.api_client import api_client
                # Get full message history from in-memory session
                if session_id in self._in_memory_sessions:
                    # Use the API session ID (from MongoDB) for updates
                    api_session_id = self._in_memory_sessions[session_id].get('api_session_id')
                    if api_session_id:
                        all_messages = self._in_memory_sessions[session_id].get('messages', [])
                        # Convert datetime objects to ISO strings for JSON serialization
                        serializable_messages = []
                        for msg in all_messages:
                            serializable_messages.append({
                                'type': msg.get('type'),
                                'content': msg.get('content'),
                                'timestamp': msg.get('timestamp').isoformat() if hasattr(msg.get('timestamp'), 'isoformat') else str(msg.get('timestamp'))
                            })
                        await api_client.update_chat_session(api_session_id, serializable_messages)
                        logger.debug(f"Messages saved to API for session {api_session_id}")
            except Exception as api_err:
                logger.debug(f"Failed to save messages to API: {api_err}")
            
            # Try to persist to local database (may fail)
            try:
                await self.db_service.update_one(
                    self.chat_collection,
                    {'session_id': session_id},
                    {
                        '$push': {
                            'messages': {
                                '$each': [
                                    {'type': 'user', 'content': message, 'timestamp': datetime.utcnow()},
                                    {'type': 'assistant', 'content': response_text, 'timestamp': datetime.utcnow()}
                                ]
                            }
                        },
                        '$set': {'last_activity': datetime.utcnow()}
                    }
                )
            except Exception:
                pass  # In-memory session is already updated
            
            return {
                'success': True,
                'response': response_text,
                'relevant_chunks': len(relevant_chunks),
                'session_id': session_id,
                'model_display': model_display
            }
            
        except Exception as e:
            logger.error(f"Chat failed for session {session_id}: {str(e)}")
            return {'success': False, 'error': f'Chat failed: {str(e)}'}
    
    def _extract_paper_text(self, paper: Dict[str, Any]) -> str:
        """Extract text content from paper for indexing"""
        text_parts = []
        
        if paper.get('title'):
            text_parts.append(paper['title'])
        
        if paper.get('abstract'):
            text_parts.append(paper['abstract'])
        
        if paper.get('full_text'):
            text_parts.append(paper['full_text'])
        
        return '\n\n'.join(text_parts)
    
    def _build_chat_prompt(self, session: Dict[str, Any], message: str, context: str) -> str:
        """Build chat prompt with context"""
        from ..prompts import format_prompt
        
        chat_history = session.get('messages', [])
        
        history_text = ""
        recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        for msg in recent_messages:
            role = "User" if msg['type'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        return format_prompt("rag_chat",
                           context=context,
                           history=history_text,
                           message=message)

    def _parse_llm_response(self, response: Any) -> Tuple[bool, str, str]:
        """Normalize LLM responses that may return strings or dictionaries"""
        if isinstance(response, dict):
            if response.get('success', True) and isinstance(response.get('content'), str):
                content = response['content'].strip()
                if content:
                    return True, content, ""
            return False, "", response.get('error', 'LLM response missing content')
        if isinstance(response, str):
            text = response.strip()
            if text and not text.startswith('Error'):
                return True, text, ""
            return False, "", text or 'LLM returned empty response'
        if response is None:
            return False, "", 'LLM returned no response'
        return False, "", 'Unexpected LLM response type'
    
    async def cleanup_expired_data(self):
        """Clean up expired embeddings and chat sessions"""
        try:
            cutoff_time = datetime.utcnow()
            
            deleted_embeddings = await self.db_service.delete_many(
                self.vector_collection,
                {'expires_at': {'$lt': cutoff_time}}
            )
            
            chat_cutoff = datetime.utcnow() - timedelta(days=7)
            deleted_sessions = await self.db_service.delete_many(
                self.chat_collection,
                {'last_activity': {'$lt': chat_cutoff}}
            )
            
            logger.info(f"RAG cleanup completed: deleted {deleted_embeddings} embeddings, {deleted_sessions} sessions")
            
        except Exception as e:
            logger.error(f"RAG cleanup failed: {str(e)}")
