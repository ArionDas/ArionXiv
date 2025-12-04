"""
Services module for ArionXiv

This module contains all the core service classes for paper analysis,
database operations, configuration management, and more.
"""

from .unified_config_service import config
from .unified_database_service import database_service
from .unified_paper_service import paper_service
from .unified_analysis_service import analysis_service
from .unified_pdf_service import pdf_service
from .unified_auth_service import auth_service
from .unified_llm_service import llm_service
from .unified_scheduler_service import trigger_user_daily_dose, unified_scheduler
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