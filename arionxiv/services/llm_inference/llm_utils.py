"""
Shared utilities for LLM inference clients
Consolidates common functionality to avoid code duplication
"""

import json
import re
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)


def parse_json_response(response_content: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Parse JSON response with retry logic and fallback handling.
    
    Shared utility for all LLM clients to parse JSON from potentially
    markdown-wrapped or malformed responses.
    
    Args:
        response_content: Raw response from LLM
        max_retries: Number of parsing attempts
        
    Returns:
        Parsed JSON as dictionary, or fallback response on failure
    """
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
                # Try to extract JSON object
                json_match = re.search(r'\{[\s\S]*\}', clean_content)
                if json_match:
                    clean_content = json_match.group(0)
                return json.loads(clean_content)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                # Try to find nested JSON structure
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


def generate_cache_key(content: str, prompt_type: str, model: str = "") -> str:
    """Generate cache key from content, prompt type, and optionally model"""
    cache_input = f"{prompt_type}:{model}:{content[:500]}" if model else f"{prompt_type}:{content[:500]}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def generate_paper_cache_key(paper: Dict[str, Any]) -> str:
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


def format_paper_metadata(paper: Dict[str, Any], index: Optional[int] = None) -> str:
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


class AsyncLRUCache:
    """
    Async-safe LRU cache with TTL support.
    Shared utility for all LLM clients.
    """
    
    def __init__(self, max_size: int = 100, ttl_hours: float = 1.0):
        self.cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        async with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            value, timestamp = self.cache[key]
            
            if datetime.now() - timestamp > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return value
    
    async def set(self, key: str, value: Any) -> None:
        """Add value to cache with eviction if needed"""
        async with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = (value, datetime.now())
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def sanitize_arxiv_id(arxiv_id: str, remove_version: bool = False) -> str:
    """
    Sanitize and normalize arXiv ID.
    
    Consolidates duplicate implementations from:
    - arxiv_operations/utils.py
    - utils/api_helpers.py
    
    Args:
        arxiv_id: Raw arXiv ID or URL
        remove_version: If True, strips version suffix (v1, v2, etc.)
        
    Returns:
        Cleaned arXiv ID
    """
    if not arxiv_id:
        return arxiv_id
    
    arxiv_id = arxiv_id.strip()
    
    # Remove common URL prefixes
    prefixes = [
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
        "arXiv:",
    ]
    for prefix in prefixes:
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
    
    # Remove .pdf extension if present
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    
    # Extract just the ID part if there's a path
    if "/" in arxiv_id:
        arxiv_id = arxiv_id.split("/")[-1]
    
    # Optionally remove version suffix
    if remove_version:
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    
    return arxiv_id.strip()


__all__ = [
    'parse_json_response',
    'generate_cache_key',
    'generate_paper_cache_key',
    'format_paper_metadata',
    'AsyncLRUCache',
    'sanitize_arxiv_id',
]
