"""
LLM Inference Module for ArionXiv

Provides unified access to multiple LLM providers:
- Groq (fast inference, Llama models)
- OpenRouter (access to free models like Kimi K2, DeepSeek, etc.)

Usage:
    from arionxiv.services.llm_inference import groq_client, openrouter_client
    
    # Use Groq for fast inference
    result = await groq_client.get_completion(prompt)
    
    # Use OpenRouter for free models
    result = await openrouter_client.chat(message, context)
"""

# Shared utilities
from .llm_utils import (
    parse_json_response,
    generate_cache_key,
    generate_paper_cache_key,
    format_paper_metadata,
    AsyncLRUCache,
    sanitize_arxiv_id,
)

# Groq client (primary LLM provider)
from .groq_client import GroqClient, create_groq_client, groq_client

# OpenRouter client (free models)
try:
    from .openrouter_client import OpenRouterClient, get_openrouter_client, openrouter_client
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    OpenRouterClient = None
    get_openrouter_client = None
    openrouter_client = None

__all__ = [
    # Shared utilities
    'parse_json_response',
    'generate_cache_key',
    'generate_paper_cache_key',
    'format_paper_metadata',
    'AsyncLRUCache',
    'sanitize_arxiv_id',
    # Groq
    'GroqClient',
    'create_groq_client', 
    'groq_client',
    # OpenRouter
    'OpenRouterClient',
    'get_openrouter_client',
    'openrouter_client',
    'OPENROUTER_AVAILABLE',
]
