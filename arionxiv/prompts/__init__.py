"""Public interface for the prompts module."""

from .prompts import format_prompt, format_prompt_async, get_all_prompts

__all__ = [
    'format_prompt',
    'format_prompt_async', 
    'get_all_prompts'
]
