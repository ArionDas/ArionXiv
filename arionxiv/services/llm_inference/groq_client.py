# Groq LLM client for AI-powered paper analysis
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
import logging
import json
import asyncio
import os
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
import time
from groq import AsyncGroq
from dotenv import load_dotenv
import httpx
from rich.console import Console

## File imports
from ...prompts import format_prompt

load_dotenv()

# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# GROQ CLIENT DEFINITION
# ============================================================================

class GroqClient:
    """
    Production-ready client for LLM-based paper analysis using Groq
    
    Features:
    - Rate limiting and concurrency control with async safety
    - Connection pooling and timeout handling
    - Thread-safe LRU caching with TTL support
    - Retry logic with exponential backoff and rate limit handling
    - Token usage tracking and monitoring
    - Structured logging
    - Async context manager support for proper resource cleanup
    """
    
    # Model configuration with context limits
    MODEL_LIMITS = {
        "llama-3.3-70b-versatile": {"max_tokens": 32768, "optimal_completion": 8000, "rpm": 30, "tpm": 14400},
        "llama-3.1-70b-versatile": {"max_tokens": 32768, "optimal_completion": 8000, "rpm": 30, "tpm": 14400},
        "llama-3.1-8b-instant": {"max_tokens": 8192, "optimal_completion": 4000, "rpm": 30, "tpm": 14400},
    }
    
    def __init__(self, max_concurrent_requests: int = 5, enable_cache: bool = True, cache_ttl_hours: int = 24, console: Console = None):
        """
        Initialize Groq client with production-ready configuration
        
        Args:
            max_concurrent_requests: Maximum number of concurrent API requests (consider model RPM limits)
            enable_cache: Enable LRU caching for repeated analyses
            cache_ttl_hours: Time-to-live for cache entries in hours (default: 24)
            console: Rich console for output (optional)
        """
        # API configuration
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("DEFAULT_ANALYSIS_MODEL", "llama-3.3-70b-versatile")
        self.timeout = 60
        self._console = console or Console()
        
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
        
        # Client initialization
        self.client = None
        
        # Initialize Groq client if API key is available
        if self.api_key:
            try:
                # Configure with connection pooling and proper timeouts
                self.client = AsyncGroq(
                    api_key=self.api_key,
                    max_retries=2,
                    timeout=httpx.Timeout(60.0, connect=5.0)
                )
                logger.info(
                    "Groq client initialized",
                    extra={
                        "model": self.model,
                        "max_concurrent": max_concurrent_requests,
                        "cache_enabled": enable_cache
                    }
                )
            except Exception as e:
                logger.error("Failed to initialize Groq client", extra={"error": str(e)})
                self.client = None
        else:
            logger.warning("No GROQ_API_KEY found in environment - Groq features will be unavailable")
            self.client = None
    
    @property
    def is_available(self) -> bool:
        """Check if the client is properly configured"""
        return self.client is not None
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model
    
    def get_model_display_name(self) -> str:
        """Get a user-friendly model display name"""
        # Extract model name
        model_name = self.model
        if "-" in model_name:
            parts = model_name.split("-")
            # e.g., "llama-3.3-70b-versatile" -> "Llama 3.3 70B"
            if len(parts) >= 3:
                return f"{parts[0].title()} {parts[1]} {parts[2].upper()}"
        return model_name.title()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper resource cleanup"""
        try:
            if self.client and hasattr(self.client, 'aclose'):
                await self.client.aclose()
            async with self.cache_lock:
                self.cache.clear()
            logger.info("Groq client closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error during client cleanup: {str(e)}")
    
    async def close(self):
        """Explicitly close the client"""
        await self.__aexit__(None, None, None)
    
    def _generate_cache_key(self, content: str, prompt_type: str) -> str:
        """Generate cache key from content and prompt type"""
        cache_input = f"{prompt_type}:{content[:500]}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _generate_paper_cache_key(self, paper: Dict[str, Any]) -> str:
        """Generate unique cache key for a paper using stable identifiers"""
        paper_id = paper.get('arxiv_id') or paper.get('doi') or paper.get('id')
        
        if not paper_id:
            title = paper.get('title', 'Unknown')
            authors = paper.get('authors', [])
            if authors and len(authors) > 0:
                first_author = authors[0] if isinstance(authors, list) else str(authors)
                paper_id = f"{title}:{first_author}"
            else:
                paper_id = title
        
        return str(paper_id)
    
    def _format_paper_metadata(self, paper: Dict[str, Any], index: Optional[int] = None) -> str:
        """Format paper metadata into a standardized string"""
        title = paper.get('title', 'Unknown')
        abstract = paper.get('abstract', 'No abstract available')
        categories = paper.get('categories', [])
        authors = paper.get('authors', [])
        
        cat_str = ', '.join(categories[:3]) if categories else 'N/A'
        author_count = len(authors) if isinstance(authors, list) else 0
        
        prefix = f"Paper {index}: " if index is not None else ""
        
        return (
            f"{prefix}{title}\n"
            f"Categories: {cat_str}\n"
            f"Authors: {author_count} author(s)\n"
            f"Abstract: {abstract}"
        )
    
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
                    parsed_result = json.loads(clean_content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', clean_content)
                    if json_match:
                        clean_content = json_match.group(0)
                    parsed_result = json.loads(clean_content)
                
                return parsed_result
                
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
        temperature: float = 0.2, 
        max_tokens: int = 8000
    ) -> Any:
        """Make API call with retry logic and exponential backoff"""
        if not self.client:
            raise ValueError("Service temporarily unavailable. Please try again later.")
        
        model_limits = self.MODEL_LIMITS.get(self.model, {"optimal_completion": 4000, "max_tokens": 32768})
        max_tokens = min(max_tokens, model_limits["optimal_completion"])
        
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=self.timeout
                    )
                    
                    self.total_requests += 1
                    if hasattr(response, 'usage'):
                        self.total_tokens_used += response.usage.total_tokens
                    
                    return response
                    
            except asyncio.TimeoutError:
                logger.error(f"Request timed out (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    self.total_errors += 1
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                # Check for rate limit error (429)
                if hasattr(e, 'status_code') and e.status_code == 429:
                    retry_after = getattr(e, 'headers', {}).get('retry-after', 2 ** attempt)
                    try:
                        retry_after = float(retry_after)
                    except (ValueError, TypeError):
                        retry_after = 2 ** attempt
                    
                    await asyncio.sleep(retry_after)
                    continue
                
                if attempt == self.max_retries - 1:
                    self.total_errors += 1
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    async def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 8000) -> str:
        """
        Get a completion from the LLM for a given prompt
        
        Args:
            prompt: Input prompt string
            temperature: Sampling temperature
            max_tokens: Maximum tokens for completion
        
        Returns:
            Completion text string
        """
        try:
            if not self.client:
                return "Service temporarily unavailable"
            
            response = await self._api_call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq completion failed: {str(e)}")
            return "Sorry, I encountered an error while processing your request. Please try again."
    
    async def get_json_completion(
        self, 
        prompt: str, 
        system_message: str = None,
        temperature: float = 0.2,
        max_tokens: int = 8000
    ) -> Dict[str, Any]:
        """Get a JSON-formatted completion from the model"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self._api_call_with_retry(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content = response.choices[0].message.content
        return self._parse_json_response(content)
    
    async def analyze_paper(self, content: str) -> Dict[str, Any]:
        """Analyze a single paper using Groq LLM"""
        start_time = time.time()
        
        try:
            if not content.strip():
                return {"analysis": "No content provided for analysis."}
            
            if not self.client:
                return {"error": "Service temporarily unavailable", "success": False}
            
            cache_key = self._generate_cache_key(content, "paper_analysis")
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            prompt = format_prompt("comprehensive_paper_analysis", content=content)
            
            response = await self._api_call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=8000
            )
            
            response_content = response.choices[0].message.content
            analysis = self._parse_json_response(response_content)
            
            await self._add_to_cache(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Paper analysis failed: {str(e)}")
            raise
    
    async def generate_insights(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cross-paper insights"""
        try:
            if not papers:
                return {"message": "No papers provided for insight generation"}
            
            if not self.client:
                return {"error": "Service temporarily unavailable", "success": False}
            
            papers_to_analyze = papers[:10]
            papers_summary = []
            
            for i, paper in enumerate(papers_to_analyze):
                metadata = self._format_paper_metadata(paper, index=i+1)
                papers_summary.append(metadata)
            
            papers_data = f"Papers analyzed ({len(papers)} total):\n\n{chr(10).join(papers_summary)}"
            prompt = format_prompt("trend_analysis", papers_data=papers_data)
            
            response = await self._api_call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=8000
            )
            
            return self._parse_json_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            raise
    
    async def summarize_collection(self, papers: List[Dict[str, Any]]) -> str:
        """Generate a concise summary for a collection of papers"""
        try:
            if not papers:
                return "No papers provided for summarization"
            
            if not self.client:
                return "Service temporarily unavailable"
            
            papers_to_summarize = papers[:15]
            papers_info = []
            
            for paper in papers_to_summarize:
                title = paper.get('title', 'Unknown')
                abstract = paper.get('abstract', 'No abstract')
                papers_info.append(f"- {title}: {abstract[:150]}...")
            
            papers_data = chr(10).join(papers_info)
            prompt = format_prompt("paper_summary", papers_data=papers_data)
            
            response = await self._api_call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Collection summarization failed: {str(e)}")
            return f"Collection of {len(papers)} papers covering diverse topics."
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status and metrics"""
        return {
            "configured": self.client is not None,
            "model": self.model,
            "api_service": "Groq",
            "timeout": self.timeout,
            "metrics": {
                "total_requests": self.total_requests,
                "total_tokens_used": self.total_tokens_used,
                "cache_hits": self.total_cache_hits,
                "total_errors": self.total_errors,
                "cache_size": len(self.cache)
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        total_operations = self.total_requests + self.total_cache_hits
        cache_hit_rate = (self.total_cache_hits / total_operations * 100) if total_operations > 0 else 0.0
        
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "cache_hits": self.total_cache_hits,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "total_errors": self.total_errors,
            "model": self.model
        }


# ============================================================================
# FACTORY AND SINGLETON
# ============================================================================

def create_groq_client(
    max_concurrent_requests: int = 5, 
    enable_cache: bool = True, 
    cache_ttl_hours: int = 24
) -> GroqClient:
    """Factory function to create Groq client instances"""
    return GroqClient(
        max_concurrent_requests=max_concurrent_requests,
        enable_cache=enable_cache,
        cache_ttl_hours=cache_ttl_hours
    )

# Global singleton instance
groq_client = create_groq_client()

# Backward compatibility alias
LLMClient = GroqClient
llm_client = groq_client
create_llm_client = create_groq_client
