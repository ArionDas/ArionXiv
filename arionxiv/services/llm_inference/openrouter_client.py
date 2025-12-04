# OpenRouter LLM client for AI-powered paper analysis
# Uses free models like moonshotai/kimi-k2:free

from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
import logging
import json
import asyncio
import os
import hashlib
from datetime import datetime, timedelta
import time
import httpx
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# OPENROUTER CLIENT DEFINITION
# ============================================================================

class OpenRouterClient:
    """
    Production-ready client for LLM-based paper analysis using OpenRouter
    
    Features:
    - Access to multiple free AI models (Kimi K2, DeepSeek, etc.)
    - Rate limiting and concurrency control with async safety
    - Connection pooling and timeout handling
    - Thread-safe LRU caching with TTL support
    - Retry logic with exponential backoff
    - Token usage tracking and monitoring
    - Structured JSON response handling
    - Async context manager support for proper resource cleanup
    """
    
    # Base URL for OpenRouter API (OpenAI-compatible)
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Free model configurations with context limits
    # Free models on OpenRouter have ":free" suffix
    MODEL_CONFIGS = {
        "meta-llama/llama-3.3-70b-instruct:free": {
            "max_tokens": 8192,
            "optimal_completion": 4000,
            "rpm": 20,
            "description": "Meta Llama 3.3 70B - Free, reliable and fast"
        },
        "google/gemma-3-27b-it:free": {
            "max_tokens": 8192,
            "optimal_completion": 4000,
            "rpm": 20,
            "description": "Google Gemma 3 27B - Free"
        },
        "qwen/qwen3-32b:free": {
            "max_tokens": 40000,
            "optimal_completion": 8000,
            "rpm": 20,
            "description": "Qwen 3 32B - Free"
        },
        "meta-llama/llama-3.2-3b-instruct:free": {
            "max_tokens": 8192,
            "optimal_completion": 2000,
            "rpm": 30,
            "description": "Meta Llama 3.2 3B - Free, fast fallback"
        },
    }
    
    # Default model - Llama 3.3 70B is reliable and capable
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
    
    # Fallback models if primary fails (in order of preference)
    # Order based on reliability and capability
    FALLBACK_MODELS = [
        "google/gemma-3-27b-it:free",
        "qwen/qwen3-32b:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]
    
    def __init__(
        self, 
        max_concurrent_requests: int = 3, 
        enable_cache: bool = True, 
        cache_ttl_hours: int = 24, 
        console: Console = None,
        model: str = None
    ):
        """
        Initialize OpenRouter client with production-ready configuration
        
        Args:
            max_concurrent_requests: Maximum concurrent API requests
            enable_cache: Enable LRU caching for repeated analyses
            cache_ttl_hours: Time-to-live for cache entries in hours
            console: Rich console for output (optional)
            model: Model to use (default: moonshotai/kimi-k2:free)
        """
        # API configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
        self.timeout = 120  # Longer timeout for free models
        self._console = console or Console()
        
        # App identification for OpenRouter rankings
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com/ArionDas/ArionXiv")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "ArionXiv")
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_retries = 3
        
        # Thread-safe caching with TTL
        self.enable_cache = enable_cache
        self.cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self.cache_max_size = 100
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_lock = asyncio.Lock()
        
        # Monitoring and metrics
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_cache_hits = 0
        self.total_errors = 0
        
        # HTTP client for API calls
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Validate API key
        if self.api_key:
            logger.info(
                "OpenRouter client initialized",
                extra={
                    "model": self.model,
                    "max_concurrent": max_concurrent_requests,
                    "cache_enabled": enable_cache
                }
            )
        else:
            logger.warning("No OPENROUTER_API_KEY found - OpenRouter features will be unavailable")
    
    @property
    def is_available(self) -> bool:
        """Check if the client is properly configured"""
        return self.api_key is not None
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model
    
    def get_model_display_name(self) -> str:
        """Get a user-friendly model display name"""
        model_name = self.model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        if ":free" in model_name:
            model_name = model_name.replace(":free", "")
        return model_name.replace("-", " ").title()
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                    "Content-Type": "application/json"
                }
            )
        return self._http_client
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper resource cleanup"""
        try:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
            async with self.cache_lock:
                self.cache.clear()
            logger.info("OpenRouter client closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error during client cleanup: {str(e)}")
    
    async def close(self):
        """Explicitly close the client"""
        await self.__aexit__(None, None, None)
    
    def _generate_cache_key(self, content: str, prompt_type: str) -> str:
        """Generate cache key from content and prompt type"""
        cache_input = f"{prompt_type}:{self.model}:{content[:500]}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve result from cache if available and not expired"""
        if not self.enable_cache:
            return None
        
        async with self.cache_lock:
            if cache_key not in self.cache:
                return None
            
            result, timestamp = self.cache[cache_key]
            
            if datetime.now() - timestamp > self.cache_ttl:
                del self.cache[cache_key]
                return None
            
            self.cache.move_to_end(cache_key)
            self.total_cache_hits += 1
            return result
    
    async def _add_to_cache(self, cache_key: str, result: Any) -> None:
        """Add result to cache with TTL and LRU eviction"""
        if not self.enable_cache:
            return
        
        async with self.cache_lock:
            if cache_key in self.cache:
                self.cache[cache_key] = (result, datetime.now())
                self.cache.move_to_end(cache_key)
            else:
                if len(self.cache) >= self.cache_max_size:
                    oldest_key = next(iter(self.cache))
                    self.cache.pop(oldest_key)
                
                self.cache[cache_key] = (result, datetime.now())
    
    def _parse_json_response(self, response_content: str, max_retries: int = 3) -> Dict[str, Any]:
        """Parse JSON response with retry logic and fallback handling"""
        original_content = response_content
        
        for attempt in range(max_retries):
            try:
                clean_content = response_content.strip()
                
                # Remove markdown code blocks
                if clean_content.startswith("```"):
                    lines = clean_content.split("\n")
                    start_idx = 0
                    end_idx = len(lines)
                    for i, line in enumerate(lines):
                        if line.strip().startswith("```") and i == 0:
                            start_idx = 1
                        elif line.strip() == "```":
                            end_idx = i
                            break
                    clean_content = "\n".join(lines[start_idx:end_idx]).strip()
                    if clean_content.startswith("json"):
                        clean_content = clean_content[4:].strip()
                
                try:
                    return json.loads(clean_content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', clean_content)
                    if json_match:
                        clean_content = json_match.group(0)
                    return json.loads(clean_content)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    import re
                    nested_match = re.search(r'\{["\'](?:summary|analysis)["\'][\s]*:', original_content)
                    if nested_match:
                        start = nested_match.start()
                        brace_count = 0
                        for i, char in enumerate(original_content[start:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    response_content = original_content[start:start + i + 1]
                                    break
                    else:
                        response_content = original_content.strip().strip('`').strip()
                    continue
        
        # Fallback response
        logger.error("JSON parsing failed after all retries")
        raw_text = original_content.strip().replace("```json", "").replace("```", "").strip()
        
        return {
            "summary": raw_text[:1000] if len(raw_text) > 100 else "Analysis completed but could not be formatted properly.",
            "raw_response": original_content[:2000],
            "error": "JSON decode failed - displaying raw analysis",
            "key_findings": ["See summary for analysis details"],
            "methodology": "",
            "strengths": [],
            "limitations": [],
            "confidence_score": 0.5
        }
    
    async def _api_call_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 8000,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make API call with retry logic, exponential backoff, and model fallback"""
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")
        
        # Build list of models to try: current model + fallbacks
        models_to_try = [self.model]
        for fallback in self.FALLBACK_MODELS:
            if fallback != self.model and fallback not in models_to_try:
                models_to_try.append(fallback)
        
        client = await self._get_http_client()
        last_error = None
        
        for model in models_to_try:
            model_config = self.MODEL_CONFIGS.get(model, {
                "max_tokens": 8192,
                "optimal_completion": 4000
            })
            model_max_tokens = min(max_tokens, model_config.get("optimal_completion", 4000))
            
            for attempt in range(self.max_retries):
                try:
                    async with self.semaphore:
                        payload = {
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": model_max_tokens
                        }
                        
                        if response_format:
                            payload["response_format"] = response_format
                        
                        response = await client.post("/chat/completions", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            self.total_requests += 1
                            
                            if "usage" in result:
                                self.total_tokens_used += result["usage"].get("total_tokens", 0)
                            
                            # If we switched models, update for future calls
                            if model != self.model:
                                logger.info(f"Switched from {self.model} to {model} due to failures")
                                self.model = model
                            
                            return result
                        
                        elif response.status_code == 429:
                            wait_time = (2 ** attempt) * 2
                            await asyncio.sleep(wait_time)
                            continue
                        
                        elif response.status_code >= 500:
                            wait_time = (2 ** attempt) * 1
                            await asyncio.sleep(wait_time)
                            continue
                        
                        else:
                            error_detail = response.text
                            last_error = f"API error: {response.status_code} - {error_detail}"
                            logger.warning(f"Model {model} failed: {last_error}")
                            break  # Try next model
                            
                except httpx.TimeoutException:
                    self.total_errors += 1
                    last_error = f"Timeout for model {model}"
                    wait_time = (2 ** attempt) * 2
                    await asyncio.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    last_error = str(e)
                    if attempt == self.max_retries - 1:
                        logger.warning(f"Model {model} exhausted retries: {last_error}")
                        break  # Try next model
                    
                    wait_time = (2 ** attempt) * 1
                    await asyncio.sleep(wait_time)
            
            # If we got here, this model failed - try the next one
            logger.info(f"Trying fallback model after {model} failed")
        
        # All models failed
        self.total_errors += 1
        raise Exception(f"API call failed after trying all models. Last error: {last_error}")
    
    async def get_completion(
        self, 
        prompt: str, 
        system_message: str = None,
        temperature: float = 0.3,
        max_tokens: int = 8000
    ) -> str:
        """Get a simple text completion from the model"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self._api_call_with_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response["choices"][0]["message"]["content"]
    
    async def get_json_completion(
        self, 
        prompt: str, 
        system_message: str = None,
        temperature: float = 0.2,
        max_tokens: int = 8000
    ) -> Dict[str, Any]:
        """Get a JSON-formatted completion from the model"""
        json_system = (system_message or "") + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanations, just the JSON object."
        
        messages = [
            {"role": "system", "content": json_system.strip()},
            {"role": "user", "content": prompt}
        ]
        
        response = await self._api_call_with_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        content = response["choices"][0]["message"]["content"]
        return self._parse_json_response(content)
    
    async def analyze_paper(self, content: str, cache_key: str = None) -> Dict[str, Any]:
        """Analyze a research paper using the configured model"""
        if cache_key:
            cached = await self._get_from_cache(cache_key)
            if cached:
                return cached
        
        from ...prompts import format_prompt
        prompt = format_prompt("comprehensive_paper_analysis", content=content)
        
        system_message = """You are an expert research analyst specializing in academic papers. 
Provide thorough, accurate analysis with specific details from the paper.
Always respond with valid JSON in the exact format requested."""
        
        result = await self.get_json_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2,
            max_tokens=8000
        )
        
        result["_model"] = self.model
        result["_model_display"] = self.get_model_display_name()
        
        if cache_key:
            await self._add_to_cache(cache_key, result)
        
        return result
    
    async def chat(
        self, 
        message: str, 
        context: str = "", 
        history: List[Dict[str, str]] = None,
        system_message: str = None
    ) -> Dict[str, Any]:
        """Have a conversation with context (for RAG chat)"""
        from ...prompts import format_prompt
        
        history_text = ""
        if history:
            for msg in history[-10:]:
                role = msg.get("type", msg.get("role", "user"))
                content = msg.get("content", "")
                history_text += f"{role.upper()}: {content}\n"
        
        prompt = format_prompt(
            "rag_chat",
            context=context,
            history=history_text,
            message=message
        )
        
        default_system = """You are ArionXiv, an AI research assistant specializing in academic papers.
Provide accurate, helpful answers based on the paper content provided.
Be conversational but maintain technical accuracy."""
        
        response_text = await self.get_completion(
            prompt=prompt,
            system_message=system_message or default_system,
            temperature=0.4,
            max_tokens=4000
        )
        
        return {
            "success": True,
            "response": response_text,
            "model": self.model,
            "model_display": self.get_model_display_name()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client usage metrics"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "cache_hits": self.total_cache_hits,
            "errors": self.total_errors,
            "model": self.model,
            "cache_size": len(self.cache)
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_default_client: Optional[OpenRouterClient] = None

def get_openrouter_client(console: Console = None) -> OpenRouterClient:
    """Get or create the default OpenRouter client instance"""
    global _default_client
    if _default_client is None:
        _default_client = OpenRouterClient(console=console)
    return _default_client

# Create default singleton
openrouter_client = get_openrouter_client()

async def close_openrouter_client():
    """Close the default OpenRouter client"""
    global _default_client
    if _default_client:
        await _default_client.close()
        _default_client = None
