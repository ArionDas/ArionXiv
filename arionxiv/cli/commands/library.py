"""Library command for ArionXiv CLI - Uses hosted API"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...arxiv_operations.client import arxiv_client
from ...arxiv_operations.utils import ArxivUtils
from ..utils.api_client import api_client, APIClientError
from ..ui.theme import create_themed_console, get_theme_colors
from ...services.unified_user_service import unified_user_service

logger = logging.getLogger(__name__)
console = create_themed_console()


class LibraryGroup(click.Group):
    """Custom Click group for library with proper error handling"""
    
    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except click.UsageError as e:
            self._show_error(e, ctx)
            raise SystemExit(1)
    
    def _show_error(self, error, ctx):
        colors = get_theme_colors()
        error_console = Console()
        
        error_console.print()
        error_console.print(f"[bold {colors['error']}]Invalid Library Command[/bold {colors['error']}]")
        error_console.print(f"[{colors['error']}]{error}[/{colors['error']}]")
        error_console.print()
        
        error_console.print(f"[bold white]Available 'library' subcommands:[/bold white]")
        for cmd_name in sorted(self.list_commands(ctx)):
            cmd = self.get_command(ctx, cmd_name)
            if cmd and not cmd.hidden:
                help_text = cmd.get_short_help_str(limit=50)
                error_console.print(f"  [{colors['primary']}]{cmd_name}[/{colors['primary']}]  {help_text}")
        
        error_console.print()
        error_console.print(f"Run [{colors['primary']}]arionxiv library --help[/{colors['primary']}] for more information.")


@click.group(cls=LibraryGroup)
def library_command():
    """
    Manage your research library
    
    Examples:
    \b
        arionxiv library add 2301.07041
        arionxiv library list
        arionxiv library remove 2301.07041
    """
    pass


def _check_auth() -> bool:
    """Check if user is authenticated, show error if not"""
    colors = get_theme_colors()
    if not unified_user_service.is_authenticated() and not api_client.is_authenticated():
        console.print("You must be logged in to use the library. Run: arionxiv login", style=colors['error'])
        return False
    return True


@library_command.command()
@click.argument('paper_id')
@click.option('--tags', help='Comma-separated tags for the paper')
@click.option('--notes', help='Personal notes about the paper')
def add(paper_id: str, tags: str, notes: str):
    """Add a paper to your library"""
    
    async def _add_paper():
        colors = get_theme_colors()
        
        if not _check_auth():
            return
        
        clean_paper_id = ArxivUtils.normalize_arxiv_id(paper_id)
        
        console.print("Fetching paper metadata...", style=colors['info'])
        paper_metadata = arxiv_client.get_paper_by_id(clean_paper_id)
        
        if not paper_metadata:
            console.print(f"Paper not found: {paper_id}", style=colors['error'])
            return
        
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
        
        try:
            result = await api_client.add_to_library(
                arxiv_id=clean_paper_id,
                title=paper_metadata.get('title', ''),
                authors=paper_metadata.get('authors', []),
                categories=paper_metadata.get('categories', []),
                abstract=paper_metadata.get('summary', ''),
                tags=tag_list,
                notes=notes or ''
            )
            
            if result.get("success"):
                console.print(f"Added to library: {paper_metadata.get('title', clean_paper_id)}", style=colors['primary'])
                if tag_list:
                    console.print(f"Tags: {', '.join(tag_list)}", style=colors['info'])
                if notes:
                    console.print(f"Notes: {notes}", style=colors['info'])
            else:
                msg = result.get("message", "Failed to add paper")
                console.print(msg, style=colors['warning'])
                
        except APIClientError as e:
            if "already" in str(e.message).lower():
                console.print(f"Paper {clean_paper_id} is already in your library", style=colors['warning'])
            else:
                console.print(f"Error: {e.message}", style=colors['error'])
    
    asyncio.run(_add_paper())


@library_command.command()
@click.option('--tags', help='Filter by tags')
@click.option('--category', help='Filter by category')
@click.option('--status', type=click.Choice(['read', 'unread', 'reading']), help='Filter by read status')
def list(tags: str, category: str, status: str):
    """List papers in your library"""
    
    async def _list_papers():
        colors = get_theme_colors()
        
        if not _check_auth():
            return
        
        try:
            result = await api_client.get_library(limit=100)
            
            if not result.get("success"):
                console.print(result.get("message", "Failed to fetch library"), style=colors['error'])
                return
            
            library = result.get("papers", [])
            
            if not library:
                console.print("Your library is empty. Use 'arionxiv library add <paper_id>' to add papers.", style=colors['warning'])
                return
            
            # Apply local filters if specified
            if category:
                library = [p for p in library if category in p.get("categories", [])]
            if status:
                library = [p for p in library if p.get("read_status") == status]
            if tags:
                tag_list = [t.strip() for t in tags.split(',')]
                library = [p for p in library if any(t in p.get("tags", []) for t in tag_list)]
            
            if not library:
                console.print("No papers match your filters.", style=colors['warning'])
                return
            
            user = unified_user_service.get_current_user()
            user_name = user.get("user_name", "User") if user else "User"
            
            table = Table(title=f"{user_name}'s Library", header_style=f"bold {colors['primary']}")
            table.add_column("#", style="bold white", width=4)
            table.add_column("Paper ID", style="white", width=12)
            table.add_column("Title", style="white", width=50)
            table.add_column("Status", style="white", width=10)
            table.add_column("Added", style="white", width=12)
            
            for i, item in enumerate(library[:20], 1):
                title = item.get('title', 'Unknown')
                
                added = item.get('added_at', '')
                if isinstance(added, datetime):
                    added_str = added.strftime('%Y-%m-%d')
                else:
                    added_str = str(added)[:10] if added else 'Unknown'
                
                table.add_row(
                    str(i),
                    item.get('arxiv_id', 'Unknown')[:12],
                    title,
                    item.get('read_status', 'unread'),
                    added_str
                )
            
            console.print(table)
            console.print(f"\nTotal papers: {len(library)}", style=colors['primary'])
            
        except APIClientError as e:
            console.print(f"Error: {e.message}", style=colors['error'])
    
    asyncio.run(_list_papers())


@library_command.command()
def stats():
    """Show library statistics"""
    
    async def _show_stats():
        colors = get_theme_colors()
        
        if not _check_auth():
            return
        
        try:
            result = await api_client.get_library(limit=100)
            
            if not result.get("success"):
                console.print(result.get("message", "Failed to fetch library"), style=colors['error'])
                return
            
            library = result.get("papers", [])
            
            if not library:
                console.print("Your library is empty.", style=colors['warning'])
                return
            
            total = len(library)
            
            category_counts: Dict[str, int] = {}
            for paper in library:
                for cat in paper.get("categories", []):
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            status_counts: Dict[str, int] = {}
            for paper in library:
                s = paper.get("read_status", "unread")
                status_counts[s] = status_counts.get(s, 0) + 1
            
            user = unified_user_service.get_current_user()
            user_name = user.get("user_name", "User") if user else "User"
            
            stats_text = f"[bold]Total Papers:[/bold] {total}\n\n"
            stats_text += "[bold]Top Categories:[/bold]\n"
            stats_text += "\n".join([f"  - {cat}: {count}" for cat, count in top_categories])
            stats_text += "\n\n[bold]Reading Status:[/bold]\n"
            stats_text += "\n".join([f"  - {s}: {count}" for s, count in status_counts.items()])
            
            console.print(Panel(
                stats_text,
                title=f"{user_name}'s Library Statistics",
                border_style=colors['primary']
            ))
            
        except APIClientError as e:
            console.print(f"Error: {e.message}", style=colors['error'])
    
    asyncio.run(_show_stats())


@library_command.command()
@click.argument('paper_id')
def remove(paper_id: str):
    """Remove a paper from your library"""
    
    async def _remove_paper():
        colors = get_theme_colors()
        
        if not _check_auth():
            return
        
        clean_paper_id = ArxivUtils.normalize_arxiv_id(paper_id)
        
        try:
            result = await api_client.remove_from_library(clean_paper_id)
            
            if result.get("success"):
                console.print(f"Removed paper {clean_paper_id} from your library", style=colors['primary'])
            else:
                console.print(result.get("message", f"Paper {clean_paper_id} not found in your library"), style=colors['warning'])
                
        except APIClientError as e:
            console.print(f"Error: {e.message}", style=colors['error'])
    
    asyncio.run(_remove_paper())
