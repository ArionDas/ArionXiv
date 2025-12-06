"""
Unified Paper Service for ArionXiv
Consolidates paper_service.py, paper_library_manager.py, and arxiv_service.py
Provides comprehensive paper management, ArXiv fetching, and library navigation
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text

from .unified_database_service import unified_database_service
from .unified_config_service import unified_config_service
from ..arxiv_operations.client import arxiv_client
from ..arxiv_operations.fetcher import arxiv_fetcher

try:
    from ..cli.ui.theme_system import create_themed_console, style_text, get_theme_colors
except ImportError:
    try:
        from ..cli.ui.theme import create_themed_console, style_text, get_theme_colors
    except ImportError:
        # Fallback for when running without CLI context (e.g., server mode)
        def create_themed_console():
            from rich.console import Console
            return Console()
        def style_text(text, style=None):
            return text
        def get_theme_colors():
            return {}

logger = logging.getLogger(__name__)


class UnifiedPaperService:
    """
    Comprehensive paper service that handles:
    1. Paper storage and retrieval (paper_service.py functionality)
    2. ArXiv paper fetching based on preferences (arxiv_service.py functionality) 
    3. Interactive paper library management (paper_library_manager.py functionality)
    """
    
    def __init__(self, database_client_instance=None):
        # ArXiv configuration
        self.config = unified_config_service.get_arxiv_config()
        self.max_results_per_query = self.config["max_results_per_query"]
        self.search_days_back = self.config["search_days_back"]
        self.default_categories = self.config["default_categories"]
        self._database_client = database_client_instance
        
        # Console for interactive UI
        self.console = create_themed_console()
        
        logger.info("UnifiedPaperService initialized")
    
    def _get_database_client(self):
        """Get database client instance"""
        if self._database_client is not None:
            return self._database_client
        return unified_database_service
    
    # ====================
    # PAPER STORAGE & RETRIEVAL (from paper_service.py)
    # ====================
    
    async def save_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a research paper"""
        try:
            db_client = self._get_database_client()
            return await db_client.save_paper(paper_data)
        except Exception as e:
            logger.error("Failed to save paper", error=str(e))
            return {"success": False, "message": str(e)}
    
    async def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """Get paper by ID"""
        try:
            db_client = self._get_database_client()
            paper = await db_client.db.papers.find_one({"arxiv_id": paper_id})
            if paper:
                paper["_id"] = str(paper["_id"])
                return {"success": True, "paper": paper}
            return {"success": False, "message": "Paper not found"}
        except Exception as e:
            logger.error("Failed to get paper", paper_id=paper_id, error=str(e))
            return {"success": False, "message": str(e)}
    
    async def get_papers_by_user(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get papers associated with a user"""
        try:
            db_client = self._get_database_client()
            papers = await db_client.db.papers.find(
                {"user_id": user_id}
            ).limit(limit).to_list(limit)
            
            # Convert ObjectId to string
            for paper in papers:
                paper["_id"] = str(paper["_id"])
            
            return {"success": True, "papers": papers, "count": len(papers)}
        except Exception as e:
            logger.error("Failed to get user papers", user_id=user_id, error=str(e))
            return {"success": False, "message": str(e)}
    
    async def search_papers(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search papers with text query and filters"""
        try:
            db_client = self._get_database_client()
            search_filter = {"$text": {"$search": query}}
            
            if filters:
                search_filter.update(filters)
            
            papers = await db_client.db.papers.find(search_filter).to_list(100)
            
            # Convert ObjectId to string
            for paper in papers:
                paper["_id"] = str(paper["_id"])
            
            return {"success": True, "papers": papers, "count": len(papers)}
        except Exception as e:
            logger.error("Failed to search papers", query=query, error=str(e))
            return {"success": False, "message": str(e)}
    
    async def update_paper(self, paper_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update paper data"""
        try:
            db_client = self._get_database_client()
            result = await db_client.db.papers.update_one(
                {"arxiv_id": paper_id},
                {"$set": {**updates, "updated_at": datetime.utcnow()}}
            )
            
            if result.modified_count > 0:
                return {"success": True, "message": "Paper updated successfully"}
            else:
                return {"success": False, "message": "Paper not found or no changes made"}
        except Exception as e:
            logger.error("Failed to update paper", paper_id=paper_id, error=str(e))
            return {"success": False, "message": str(e)}
    
    async def delete_paper(self, paper_id: str) -> Dict[str, Any]:
        """Delete a paper"""
        try:
            db_client = self._get_database_client()
            result = await db_client.db.papers.delete_one({"arxiv_id": paper_id})
            
            if result.deleted_count > 0:
                return {"success": True, "message": "Paper deleted successfully"}
            else:
                return {"success": False, "message": "Paper not found"}
        except Exception as e:
            logger.error("Failed to delete paper", paper_id=paper_id, error=str(e))
            return {"success": False, "message": str(e)}
    
    # ====================
    # ARXIV FETCHING (from arxiv_service.py)
    # ====================
    
    async def fetch_papers_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch papers based on user's preferences"""
        try:
            db_client = self._get_database_client()
            
            # Get user preferences
            user_prefs = await db_client.db.user_preferences.find_one({"user_id": user_id})
            if not user_prefs:
                logger.info(f"No preferences found for user {user_id}, using defaults")
                user_prefs = {"categories": self.default_categories}
            
            # Fetch papers for each category
            all_papers = []
            categories = user_prefs.get("categories", self.default_categories)
            keywords = user_prefs.get("keywords", [])
            authors = user_prefs.get("authors", [])
            
            for category in categories:
                try:
                    # Build search query
                    query_parts = [f"cat:{category}"]
                    
                    if keywords:
                        keyword_query = " OR ".join([f'"{keyword}"' for keyword in keywords])
                        query_parts.append(f"({keyword_query})")
                    
                    if authors:
                        author_query = " OR ".join([f"au:{author}" for author in authors])
                        query_parts.append(f"({author_query})")
                    
                    search_query = " AND ".join(query_parts)
                    
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.search_days_back)
                    
                    # Fetch papers
                    papers = await arxiv_client.search_papers(
                        query=search_query,
                        max_results=self.max_results_per_query,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Add user context
                    for paper in papers:
                        paper["fetched_for_user"] = user_id
                        paper["fetched_at"] = datetime.utcnow()
                        paper["category_matched"] = category
                    
                    all_papers.extend(papers)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch papers for category {category}", error=str(e))
                    continue
            
            # Remove duplicates based on arxiv_id
            seen_ids = set()
            unique_papers = []
            for paper in all_papers:
                if paper.get("id") not in seen_ids:
                    seen_ids.add(paper.get("id"))
                    unique_papers.append(paper)
            
            logger.info(f"Fetched {len(unique_papers)} unique papers for user {user_id}")
            return unique_papers
            
        except Exception as e:
            logger.error(f"Failed to fetch papers for user {user_id}", error=str(e))
            return []
    
    async def fetch_trending_papers(self, category: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch trending papers from ArXiv"""
        try:
            # Build query for trending papers
            if category:
                query = f"cat:{category}"
            else:
                query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch papers
            papers = await arxiv_client.search_papers(
                query=query,
                max_results=self.max_results_per_query * 2,  # Get more for trending
                start_date=start_date,
                end_date=end_date,
                sort_by="submittedDate",
                sort_order="descending"
            )
            
            # Add trending metadata
            for paper in papers:
                paper["is_trending"] = True
                paper["fetched_at"] = datetime.utcnow()
            
            logger.info(f"Fetched {len(papers)} trending papers")
            return papers
            
        except Exception as e:
            logger.error("Failed to fetch trending papers", error=str(e))
            return []
    
    async def search_arxiv_papers(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search ArXiv directly with custom query"""
        try:
            papers = await arxiv_client.search_papers(
                query=query,
                max_results=max_results
            )
            
            for paper in papers:
                paper["search_query"] = query
                paper["fetched_at"] = datetime.utcnow()
            
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search failed for query: {query}", error=str(e))
            return []
    
    # ====================
    # LIBRARY MANAGEMENT (from paper_library_manager.py)
    # ====================
    
    async def show_paper_library(self, user_id: str = "default") -> Optional[str]:
        """Show interactive paper library and return selected paper ID"""
        try:
            # Get user's papers from database
            papers_result = await self.get_papers_by_user(user_id)
            
            if not papers_result["success"] or not papers_result["papers"]:
                self.console.print("[yellow]No papers found in your library.[/yellow]")
                
                # Offer to fetch new papers
                if Confirm.ask("Would you like to fetch some papers from ArXiv?"):
                    colors = get_theme_colors()
                    with self.console.status(f"[bold {colors['primary']}]Fetching papers..."):
                        new_papers = await self.fetch_papers_for_user(user_id)
                        
                        if new_papers:
                            # Save fetched papers
                            for paper in new_papers:
                                await self.save_paper(paper)
                            
                            self.console.print(f"[green]Fetched and saved {len(new_papers)} papers![/green]")
                            return await self.show_paper_library(user_id)  # Recursive call
                        else:
                            self.console.print("[red]Failed to fetch papers.[/red]")
                            return None
                else:
                    return None
            
            papers = papers_result["papers"]
            colors = get_theme_colors()
            
            # Display papers in a table
            table = Table(title="Your Paper Library", header_style=f"bold {colors['primary']}")
            table.add_column("ID", style="bold white", no_wrap=True)
            table.add_column("Title", style="white", max_width=50)
            table.add_column("Authors", style="white", max_width=30)
            table.add_column("Date", style="white")
            table.add_column("Categories", style="white", max_width=20)
            
            for i, paper in enumerate(papers):
                # Truncate long titles and authors
                title = paper.get("title", "Unknown")[:47] + "..." if len(paper.get("title", "")) > 50 else paper.get("title", "Unknown")
                
                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    authors_str = ", ".join(authors[:2])  # Show first 2 authors
                    if len(authors) > 2:
                        authors_str += f" (+{len(authors)-2} more)"
                else:
                    authors_str = str(authors)[:27] + "..." if len(str(authors)) > 30 else str(authors)
                
                date_str = paper.get("published", paper.get("submitted", "Unknown"))[:10]  # Just date part
                
                categories = paper.get("categories", [])
                if isinstance(categories, list):
                    cat_str = ", ".join(categories[:2])
                    if len(categories) > 2:
                        cat_str += "..."
                else:
                    cat_str = str(categories)[:17] + "..." if len(str(categories)) > 20 else str(categories)
                
                table.add_row(
                    str(i + 1),
                    title,
                    authors_str,
                    date_str,
                    cat_str
                )
            
            self.console.print(table)
            
            # Show options
            colors = get_theme_colors()
            self.console.print(f"\n[bold {colors['primary']}]Options:[/bold {colors['primary']}]")
            self.console.print("• Enter a number (1-{}) to select a paper".format(len(papers)))
            self.console.print("• Type 'search' to search your library")
            self.console.print("• Type 'fetch' to get new papers from ArXiv")
            self.console.print("• Type 'refresh' to refresh the library")
            self.console.print("• Type 'quit' to exit")
            
            while True:
                choice = Prompt.ask("\n[bold green]Your choice[/bold green]").strip().lower()
                
                if choice == 'quit':
                    return None
                elif choice == 'search':
                    return await self._handle_library_search(user_id)
                elif choice == 'fetch':
                    return await self._handle_fetch_new_papers(user_id)
                elif choice == 'refresh':
                    return await self.show_paper_library(user_id)  # Recursive refresh
                else:
                    try:
                        paper_index = int(choice) - 1
                        if 0 <= paper_index < len(papers):
                            selected_paper = papers[paper_index]
                            paper_id = selected_paper.get("arxiv_id", selected_paper.get("id"))
                            
                            # Show paper details
                            self._show_paper_details(selected_paper)
                            
                            if Confirm.ask("Use this paper?"):
                                return paper_id
                        else:
                            self.console.print(f"[red]Please enter a number between 1 and {len(papers)}[/red]")
                    except ValueError:
                        self.console.print("[red]Please enter a valid number or command[/red]")
            
        except Exception as e:
            logger.error("Paper library display failed", error=str(e))
            self.console.print(f"[red]Error showing library: {str(e)}[/red]")
            return None
    
    async def _handle_library_search(self, user_id: str) -> Optional[str]:
        """Handle library search functionality"""
        query = Prompt.ask("Enter search terms")
        colors = get_theme_colors()
        
        with self.console.status(f"[bold {colors['primary']}]Searching for '{query}'..."):
            search_result = await self.search_papers(query, {"user_id": user_id})
        
        if not search_result["success"] or not search_result["papers"]:
            self.console.print("[yellow]No papers found matching your search.[/yellow]")
            return await self.show_paper_library(user_id)
        
        # Show search results
        papers = search_result["papers"]
        colors = get_theme_colors()
        self.console.print(f"\n[{colors['primary']}]Found {len(papers)} papers matching '{query}':[/{colors['primary']}]")
        
        # Create simplified table for search results
        table = Table(header_style=f"bold {colors['primary']}")
        table.add_column("ID", style="bold white")
        table.add_column("Title", style="white", max_width=60)
        table.add_column("Relevance", style="white")
        
        for i, paper in enumerate(papers):
            title = paper.get("title", "Unknown")[:57] + "..." if len(paper.get("title", "")) > 60 else paper.get("title", "Unknown")
            table.add_row(str(i + 1), title, "***")  # Placeholder relevance
        
        self.console.print(table)
        
        # Let user select from search results
        while True:
            choice = Prompt.ask("Select a paper (number) or 'back' to return to library")
            
            if choice.lower() == 'back':
                return await self.show_paper_library(user_id)
            
            try:
                paper_index = int(choice) - 1
                if 0 <= paper_index < len(papers):
                    selected_paper = papers[paper_index]
                    paper_id = selected_paper.get("arxiv_id", selected_paper.get("id"))
                    
                    self._show_paper_details(selected_paper)
                    
                    if Confirm.ask("Use this paper?"):
                        return paper_id
                else:
                    self.console.print(f"[red]Please enter a number between 1 and {len(papers)}[/red]")
            except ValueError:
                self.console.print("[red]Please enter a valid number or 'back'[/red]")
    
    async def _handle_fetch_new_papers(self, user_id: str) -> Optional[str]:
        """Handle fetching new papers"""
        colors = get_theme_colors()
        self.console.print(f"[{colors['primary']}]Fetching new papers based on your preferences...[/{colors['primary']}]")
        
        with self.console.status(f"[bold {colors['primary']}]Fetching from ArXiv..."):
            new_papers = await self.fetch_papers_for_user(user_id)
        
        if not new_papers:
            self.console.print("[red]Failed to fetch new papers.[/red]")
            return await self.show_paper_library(user_id)
        
        # Save new papers
        saved_count = 0
        for paper in new_papers:
            result = await self.save_paper(paper)
            if result.get("success"):
                saved_count += 1
        
        self.console.print(f"[green]Fetched and saved {saved_count} new papers![/green]")
        
        if Confirm.ask("View the updated library?"):
            return await self.show_paper_library(user_id)
        else:
            return None
    
    def _show_paper_details(self, paper: Dict[str, Any]):
        """Show detailed information about a paper"""
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        abstract = paper.get("abstract", "No abstract available")
        categories = paper.get("categories", [])
        published = paper.get("published", paper.get("submitted", "Unknown"))
        
        # Format authors
        if isinstance(authors, list):
            authors_str = ", ".join(authors)
        else:
            authors_str = str(authors)
        
        # Format categories
        if isinstance(categories, list):
            categories_str = ", ".join(categories)
        else:
            categories_str = str(categories)
        
        # Create detail panel
        detail_text = f"""[bold]Title:[/bold] {title}

[bold]Authors:[/bold] {authors_str}

[bold]Published:[/bold] {published}

[bold]Categories:[/bold] {categories_str}

[bold]Abstract:[/bold]
{abstract[:500]}{'...' if len(abstract) > 500 else ''}"""
        
        colors = get_theme_colors()
        panel = Panel(
            detail_text,
            title=f"[bold {colors['primary']}]Paper Details[/bold {colors['primary']}]",
            border_style=colors['primary'],
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    # ====================
    # ENHANCED OPERATIONS
    # ====================
    
    async def get_paper_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """Get statistics about papers in the library"""
        try:
            db_client = self._get_database_client()
            
            # Build filter
            filter_query = {}
            if user_id:
                filter_query["user_id"] = user_id
            
            # Get counts
            total_papers = await db_client.db.papers.count_documents(filter_query)
            
            # Get papers by category
            pipeline = [
                {"$match": filter_query},
                {"$unwind": "$categories"},
                {"$group": {"_id": "$categories", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            category_stats = await db_client.db.papers.aggregate(pipeline).to_list(50)
            
            # Get recent papers (last 30 days)
            recent_date = datetime.utcnow() - timedelta(days=30)
            recent_papers = await db_client.db.papers.count_documents({
                **filter_query,
                "fetched_at": {"$gte": recent_date}
            })
            
            return {
                "success": True,
                "total_papers": total_papers,
                "recent_papers": recent_papers,
                "categories": category_stats
            }
            
        except Exception as e:
            logger.error("Failed to get paper statistics", error=str(e))
            return {"success": False, "error": str(e)}


# Global instance
unified_paper_service = UnifiedPaperService()

# Backwards compatibility
paper_service = unified_paper_service
paper_library_manager = unified_paper_service
arxiv_service = unified_paper_service

# Export commonly used functions
save_paper = unified_paper_service.save_paper
get_paper_by_id = unified_paper_service.get_paper_by_id
fetch_papers_for_user = unified_paper_service.fetch_papers_for_user
show_paper_library = unified_paper_service.show_paper_library
search_papers = unified_paper_service.search_papers
fetch_trending_papers = unified_paper_service.fetch_trending_papers

__all__ = [
    'UnifiedPaperService',
    'unified_paper_service',
    'paper_service',
    'paper_library_manager',
    'arxiv_service',
    'save_paper',
    'get_paper_by_id',
    'fetch_papers_for_user',
    'show_paper_library',
    'search_papers',
    'fetch_trending_papers'
]