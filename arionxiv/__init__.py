"""
ArionXiv - AI-Powered Research Paper Analysis and Management

A comprehensive tool for discovering, analyzing, and managing research papers
from arXiv with AI-powered insights and organizational features.
"""

__version__ = "1.0.20"
__author__ = "Arion Das"
__email__ = "ariondasad@gmail.com"
__description__ = "AI-Powered Research Paper Analysis and Management"

# Lazy imports to avoid requiring fastapi for CLI/GitHub Actions usage
# Services are imported on-demand when accessed
def __getattr__(name):
    """Lazy import of services to avoid loading fastapi for CLI usage."""
    if name == "config":
        from .services.unified_config_service import config
        return config
    elif name == "database_service":
        from .services.unified_database_service import database_service
        return database_service
    elif name == "paper_service":
        from .services.unified_paper_service import paper_service
        return paper_service
    elif name == "analysis_service":
        from .services.unified_analysis_service import analysis_service
        return analysis_service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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