"""Library command for ArionXiv CLI"""

import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from typing import List, Dict, Any

from ...arxiv_operations.client import arxiv_client
from ...arxiv_operations.utils import ArxivUtils
from ..utils.db_config_manager import db_config_manager as config_manager
from ..ui.theme import create_themed_console, get_theme_colors
from ...services.unified_user_service import unified_user_service
from ...services.unified_database_service import unified_database_service

logger = logging.getLogger(__name__)

console = create_themed_console()

@click.group()
def library_command():
    """
    Manage your research library
    
    Examples:
    \b
        arionxiv library add 2301.07041
        arionxiv library list
        arionxiv library remove 2301.07041
        arionxiv library search "transformer"
    """
    pass

@library_command.command()
@click.argument('paper_id')
@click.option('--tags', help='Comma-separated tags for the paper')
@click.option('--notes', help='Personal notes about the paper')
def add(paper_id: str, tags: str, notes: str):
    """Add a paper to your library"""
    
    async def _add_paper():
        # Get theme colors for consistent styling
        colors = get_theme_colors()
        
        # Check authentication
        if not unified_user_service.is_authenticated():
            console.print("ERROR: You must be logged in to use the library", style=colors['error'])
            return
        
        user = unified_user_service.get_current_user()
        user_name = user["user_name"]
        
        # Clean paper ID
        clean_paper_id = ArxivUtils.normalize_arxiv_id(paper_id)
        
        # Check if paper already exists in user's library
        existing = await unified_database_service.find_one('user_papers', {
            'user_name': user_name,
            'arxiv_id': clean_paper_id
        })
        
        if existing:
            console.print(f"Paper {clean_paper_id} is already in your library", style=colors['warning'])
            return
        
        # Get paper metadata
        console.print("Fetching paper metadata...", style=colors['info'])
        paper_metadata = arxiv_client.get_paper_by_id(clean_paper_id)
        
        if not paper_metadata:
            console.print(f"Paper not found: {paper_id}", style=colors['error'])
            return
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
        
        # Create library entry with user_name for proper bifurcation
        library_entry = {
            "user_name": user_name,
            "arxiv_id": clean_paper_id,
            "title": paper_metadata.get("title", "Unknown"),
            "authors": paper_metadata.get("authors", []),
            "categories": paper_metadata.get("categories", []),
            "published": paper_metadata.get("published", ""),
            "added_at": datetime.utcnow(),
            "tags": tag_list,
            "notes": notes or "",
            "read_status": "unread",
            "rating": 0
        }
        
        # Add to database
        result = await unified_database_service.insert_one('user_papers', library_entry)
        
        if result:
            console.print(f"Added paper to library: {paper_metadata.get('title', 'Unknown')}", style=colors['primary'])
            
            if tag_list:
                console.print(f"Tags: {', '.join(tag_list)}", style=colors['info'])
            if notes:
                console.print(f"Notes: {notes}", style=colors['info'])
        else:
            console.print("Failed to add paper to library", style=colors['error'])
    
    asyncio.run(_add_paper())

@library_command.command()
@click.option('--tags', help='Filter by tags')
@click.option('--category', help='Filter by category')
@click.option('--status', type=click.Choice(['read', 'unread', 'reading']), help='Filter by read status')
def list(tags: str, category: str, status: str):
    """List papers in your library"""
    
    async def _list_papers():
        # Get theme colors for consistent styling
        colors = get_theme_colors()
        
        # Check authentication
        if not unified_user_service.is_authenticated():
            console.print("ERROR: You must be logged in to view your library", style=colors['error'])
            return
        
        user = unified_user_service.get_current_user()
        user_name = user["user_name"]
        
        # Build query filter for this specific user
        query = {"user_name": user_name}
        
        if category:
            query["categories"] = {"$in": [category]}
        
        if status:
            query["read_status"] = status
        
        if tags:
            tag_list = [t.strip() for t in tags.split(',')]
            query["tags"] = {"$in": tag_list}
        
        # Fetch user's papers from database
        cursor = unified_database_service.db.user_papers.find(query).sort("added_at", -1)
        library = await cursor.to_list(length=100)
        
        if not library:
            console.print("Your library is empty. Use 'arionxiv library add <paper_id>' to add papers.", style=colors['warning'])
            return
        
        # Create table with papers
        table = Table(title=f"{user['user_name']}'s Library", header_style=f"bold {colors['primary']}")
        table.add_column("#", style="bold white", width=4)
        table.add_column("Paper ID", style="white", width=12)
        table.add_column("Title", style="white", width=50)
        table.add_column("Status", style="white", width=10)
        table.add_column("Added", style="white", width=12)
        
        for i, item in enumerate(library[:20], 1):
            title = item.get('title', 'Unknown')
            if len(title) > 47:
                title = title[:47] + "..."
            
            added = item.get('added_at', datetime.utcnow())
            if isinstance(added, datetime):
                added_str = added.strftime('%Y-%m-%d')
            else:
                added_str = str(added)[:10]
            
            table.add_row(
                str(i),
                item.get('arxiv_id', 'Unknown')[:12],
                title,
                item.get('read_status', 'unread'),
                added_str
            )
        
        console.print(table)
        console.print(f"\nTotal papers: {len(library)}", style=colors['primary'])
    
    asyncio.run(_list_papers())

@library_command.command()
def stats():
    """Show library statistics"""
    
    async def _show_stats():
        # Get theme colors for consistent styling
        colors = get_theme_colors()
        
        # Check authentication
        if not unified_user_service.is_authenticated():
            console.print("ERROR: You must be logged in to view library stats", style=colors['error'])
            return
        
        user = unified_user_service.get_current_user()
        user_name = user["user_name"]
        
        # Get total count
        total = await unified_database_service.db.user_papers.count_documents({"user_name": user_name})
        
        if total == 0:
            console.print("Your library is empty.", style=colors['warning'])
            return
        
        # Get stats by category
        category_pipeline = [
            {"$match": {"user_name": user_name}},
            {"$unwind": "$categories"},
            {"$group": {"_id": "$categories", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        
        category_stats = await unified_database_service.db.user_papers.aggregate(category_pipeline).to_list(5)
        
        # Get stats by status
        status_pipeline = [
            {"$match": {"user_name": user_name}},
            {"$group": {"_id": "$read_status", "count": {"$sum": 1}}}
        ]
        
        status_stats = await unified_database_service.db.user_papers.aggregate(status_pipeline).to_list(10)
        
        # Display stats
        console.print(Panel(
            f"[bold]Total Papers:[/bold] {total}\n\n" +
            "[bold]Top Categories:[/bold]\n" +
            "\n".join([f"  • {stat['_id']}: {stat['count']}" for stat in category_stats]) +
            "\n\n[bold]Reading Status:[/bold]\n" +
            "\n".join([f"  • {stat['_id']}: {stat['count']}" for stat in status_stats]),
            title=f"{user['user_name']}'s Library Statistics",
            border_style=colors['primary']
        ))
    
    asyncio.run(_show_stats())

@library_command.command()
@click.argument('paper_id')
def remove(paper_id: str):
    """Remove a paper from your library"""
    
    async def _remove_paper():
        colors = get_theme_colors()
        
        # Check authentication
        if not unified_user_service.is_authenticated():
            console.print("ERROR: You must be logged in to remove papers", style=colors['error'])
            return
        
        user = unified_user_service.get_current_user()
        user_name = user["user_name"]
        
        clean_paper_id = ArxivUtils.normalize_arxiv_id(paper_id)
        
        result = await unified_database_service.delete_one('user_papers', {
            'user_name': user_name,
            'arxiv_id': clean_paper_id
        })
        
        if result and result.deleted_count > 0:
            console.print(f"Removed paper {clean_paper_id} from your library", style=colors['primary'])
        else:
            console.print(f"Paper {clean_paper_id} not found in your library", style=colors['warning'])
    
    asyncio.run(_remove_paper())
