# Backward compatibility shim - imports from new location
# The LLM clients have been moved to arionxiv/services/llm_inference/

"""
DEPRECATED: This module is kept for backward compatibility only.
Please import from arionxiv.services.llm_inference instead:

    from arionxiv.services.llm_inference import groq_client, GroqClient
    from arionxiv.services.llm_inference import openrouter_client, OpenRouterClient
"""

from .llm_inference.groq_client import (
    GroqClient,
    create_groq_client,
    groq_client,
    # Backward compatibility aliases
    LLMClient,
    llm_client,
    create_llm_client,
)

# Re-export for backward compatibility
__all__ = [
    'GroqClient',
    'create_groq_client',
    'groq_client',
    'LLMClient',
    'llm_client',
    'create_llm_client',
]
