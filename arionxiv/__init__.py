"""
ArionXiv - AI-Powered Research Paper Analysis and Management

A comprehensive tool for discovering, analyzing, and managing research papers
from arXiv with AI-powered insights and organizational features.
"""

__version__ = "1.0.18"
__author__ = "Arion Das"
__email__ = "ariondasad@gmail.com"
__description__ = "AI-Powered Research Paper Analysis and Management"

# Core imports for easy access
from .services.unified_config_service import config
from .services.unified_database_service import database_service
from .services.unified_paper_service import paper_service
from .services.unified_analysis_service import analysis_service

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "config",
    "database_service",
    "paper_service", 
    "analysis_service"
]