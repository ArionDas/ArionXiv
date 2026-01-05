"""
Services module for ArionXiv

This module contains all the core service classes for paper analysis,
database operations, configuration management, and more.

NOTE: Services are lazily imported to avoid requiring fastapi for CLI usage.
The auth_service requires fastapi and is only needed for server/API functionality.
"""

# Core services that don't require fastapi - import directly
from .unified_config_service import config
from .unified_database_service import database_service
from .unified_paper_service import paper_service
from .unified_pdf_service import pdf_service
from .unified_prompt_service import prompt_service

# LLM Inference clients (new organized location)
from .llm_inference import groq_client, GroqClient, create_groq_client
from .llm_inference import OPENROUTER_AVAILABLE

if OPENROUTER_AVAILABLE:
    from .llm_inference import openrouter_client, OpenRouterClient, get_openrouter_client
else:
    openrouter_client = None
    OpenRouterClient = None
    get_openrouter_client = None

# Backward compatibility
from .llm_client import llm_client, LLMClient, create_llm_client

# Lazy imports for services that have heavy dependencies (fastapi, etc.)
# These are only loaded when actually accessed
_lazy_imports = {
    "auth_service": ".unified_auth_service",
    "llm_service": ".unified_llm_service",
    "analysis_service": ".unified_analysis_service",
    "trigger_user_daily_dose": ".unified_scheduler_service",
    "unified_scheduler": ".unified_scheduler_service",
}

def __getattr__(name):
    """Lazy import of services with heavy dependencies."""
    if name in _lazy_imports:
        module_path = _lazy_imports[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "config",
    "database_service",
    "paper_service",
    "analysis_service", 
    "pdf_service",
    "auth_service",
    "llm_service",
    "trigger_user_daily_dose",
    "unified_scheduler",
    "prompt_service",
    # LLM clients
    "groq_client",
    "GroqClient",
    "create_groq_client",
    "openrouter_client",
    "OpenRouterClient",
    "get_openrouter_client",
    # Backward compatibility
    "llm_client",
    "LLMClient",
    "create_llm_client",
]