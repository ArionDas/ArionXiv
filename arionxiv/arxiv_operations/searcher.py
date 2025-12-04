# Simple text-based search for arXiv papers
from typing import List, Dict, Any, Optional
import logging
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from .client import arxiv_client

logger = logging.getLogger(__name__)


class ArxivSearcher:
    """Simple text-based search for arXiv papers"""
    
    def __init__(self):
        self.client = arxiv_client
        
        # Common categories for reference
        self.categories = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning", 
            "cs.CV": "Computer Vision",
            "cs.CL": "Computation and Language",
            "cs.RO": "Robotics",
            "stat.ML": "Machine Learning (Statistics)",
            "cs.DC": "Distributed Computing",
            "cs.DB": "Databases",
            "cs.IR": "Information Retrieval",
            "math.OC": "Optimization and Control",
        }
    
    async def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Simple text search that returns the closest matching papers.
        
        Args:
            query: Search text
            max_results: Number of results to return (default 10)
        
        Returns:
            Dict with success status and list of papers
        """
        try:
            # Direct search via arXiv API (uses relevance sorting by default)
            papers = self.client.search_papers(query=query, max_results=max_results)
            
            return {
                "success": True,
                "papers": papers,
                "count": len(papers),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"success": False, "error": str(e), "papers": []}
    
    async def search_by_category(self, query: str, category: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search within a specific category.
        
        Args:
            query: Search text
            category: arXiv category (e.g., cs.LG, cs.AI)
            max_results: Number of results to return
        
        Returns:
            Dict with success status and list of papers
        """
        try:
            # Combine query with category filter
            full_query = f"{query} AND cat:{category}" if query else f"cat:{category}"
            papers = self.client.search_papers(query=full_query, max_results=max_results)
            
            return {
                "success": True,
                "papers": papers,
                "count": len(papers),
                "query": query,
                "category": category
            }
            
        except Exception as e:
            logger.error(f"Category search failed: {str(e)}")
            return {"success": False, "error": str(e), "papers": []}
    
    async def search_by_author(self, author: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search papers by author name.
        
        Args:
            author: Author name
            max_results: Number of results to return
        
        Returns:
            Dict with success status and list of papers
        """
        try:
            papers = self.client.get_papers_by_author(author=author, max_results=max_results)
            
            return {
                "success": True,
                "papers": papers,
                "count": len(papers),
                "author": author
            }
            
        except Exception as e:
            logger.error(f"Author search failed: {str(e)}")
            return {"success": False, "error": str(e), "papers": []}
    
    def get_available_categories(self) -> Dict[str, str]:
        """Get available paper categories"""
        return self.categories.copy()


# Global instance
arxiv_searcher = ArxivSearcher()
