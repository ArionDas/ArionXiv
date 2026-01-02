"""Library command for ArionXiv CLI - Uses hosted API"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        error_console.print(f"[bold {colors['error']}]{error}[/bold {colors['error']}]")
        error_console.print()
        
        error_console.print(f"[bold white]Available 'library' subcommands:[/bold white]")
        for cmd_name in sorted(self.list_commands(ctx)):
            cmd = self.get_command(ctx, cmd_name)
            if cmd and not cmd.hidden:
                help_text = cmd.get_short_help_str(limit=50)
                error_console.print(f"  [bold {colors['primary']}]{cmd_name}[/bold {colors['primary']}]  {help_text}")
        
        error_console.print()
        error_console.print(f"Run [bold {colors['primary']}]arionxiv library --help[/bold {colors['primary']}] for more information.")


@click.group(cls=LibraryGroup, invoke_without_command=True)
@click.pass_context
def library_command(ctx):
    """
    Manage your research library
    
    Examples:
    \b
        arionxiv library           # View library dashboard
        arionxiv library list      # List all papers
        arionxiv library stats     # View statistics
    """
    if ctx.invoked_subcommand is None:
        asyncio.run(_show_library_dashboard())


def _check_auth() -> bool:
    """Check if user is authenticated, show error if not"""
    colors = get_theme_colors()
    if not unified_user_service.is_authenticated() and not api_client.is_authenticated():
        console.print("You must be logged in to use the library. Run: arionxiv login", style=f"bold {colors['error']}")
        return False
    return True


async def _show_library_dashboard():
    """Show library analytics dashboard"""
    colors = get_theme_colors()
    
    if not _check_auth():
        return
    
    console.print(f"\n[bold {colors['primary']}]Library Dashboard[/bold {colors['primary']}]")
    console.rule(style=f"bold {colors['primary']}")
    
    try:
        # Fetch library and chat sessions data with progress spinner
        with Progress(
            SpinnerColumn(style=f"bold {colors['primary']}"),
            TextColumn(f"[bold {colors['primary']}]{{task.description}}[/bold {colors['primary']}]"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Fetching your library data...", total=None)
            library_result = await api_client.get_library(limit=100)
            progress.update(task, description="Fetching chat sessions...")
            sessions_result = await api_client.get_chat_sessions()
            progress.update(task, description="Done!")
        
        papers = library_result.get("papers", []) if library_result.get("success") else []
        sessions = sessions_result.get("sessions", []) if sessions_result.get("success") else []
        
        total_papers = len(papers)
        active_sessions = len(sessions)
        
        # Count papers by read status
        unread = sum(1 for p in papers if p.get("read_status") == "unread")
        reading = sum(1 for p in papers if p.get("read_status") == "reading")
        completed = sum(1 for p in papers if p.get("read_status") == "completed")
        
        # Get recent papers (added in last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_papers = 0
        for p in papers:
            added_at = p.get("added_at")
            if added_at:
                if isinstance(added_at, str):
                    try:
                        added_at = datetime.fromisoformat(added_at.replace('Z', '+00:00').replace('+00:00', ''))
                    except:
                        continue
                if isinstance(added_at, datetime) and added_at > week_ago:
                    recent_papers += 1
        
        # Stats panel
        stats_content = (
            f"[bold]Total Papers Saved:[/bold] [bold {colors['primary']}]{total_papers}[/bold {colors['primary']}]\n"
            f"[bold]Added This Week:[/bold] [bold {colors['primary']}]{recent_papers}[/bold {colors['primary']}]\n"
            f"[bold]Active Chat Sessions:[/bold] [bold {colors['primary']}]{active_sessions}[/bold {colors['primary']}]\n\n"
            f"[bold]Reading Progress:[/bold]\n"
            f"  [bold {colors['primary']}]Unread:[/bold {colors['primary']}] {unread}  "
            f"[bold {colors['primary']}]Reading:[/bold {colors['primary']}] {reading}  "
            f"[bold {colors['primary']}]Completed:[/bold {colors['primary']}] {completed}"
        )
        
        console.print(Panel(
            stats_content,
            title=f"[bold {colors['primary']}]Library Statistics[/bold {colors['primary']}]",
            border_style=f"bold {colors['primary']}"
        ))
        
        # Quick actions
        console.print(f"\n[bold {colors['primary']}]Quick Actions:[/bold {colors['primary']}]")
        
        actions_table = Table(show_header=False, box=None, padding=(0, 2))
        actions_table.add_column("Command", style=f"bold {colors['primary']}")
        actions_table.add_column("Description", style="white")
        
        actions_table.add_row("arionxiv library list", "View all saved papers")
        actions_table.add_row("arionxiv library stats", "View detailed statistics")
        actions_table.add_row("arionxiv chat", "Start chatting with a paper")
        
        console.print(actions_table)
        
    except APIClientError as e:
        console.print(f"Error loading library: {e.message}", style=f"bold {colors['error']}")
    except Exception as e:
        logger.error(f"Library dashboard error: {e}", exc_info=True)
        console.print(f"Error: {str(e)}", style=f"bold {colors['error']}")

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
                console.print(result.get("message", "Failed to fetch library"), style=f"bold {colors['error']}")
                return
            
            library = result.get("papers", [])
            
            if not library:
                console.print("Your library is empty. Use 'arionxiv library add <paper_id>' to add papers.", style=f"bold {colors['warning']}")
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
                console.print("No papers match your filters.", style=f"bold {colors['warning']}")
                return
            
            user = unified_user_service.get_current_user()
            user_name = user.get("user_name", "User") if user else "User"
            
            table = Table(title=f"{user_name}'s Library", header_style=f"bold {colors['primary']}")
            table.add_column("#", style=f"bold {colors['primary']}", width=4)
            table.add_column("Paper ID", style=f"bold {colors['primary']}", width=12)
            table.add_column("Title", style=f"bold {colors['primary']}", width=50)
            table.add_column("Status", style=f"bold {colors['primary']}", width=10)
            table.add_column("Added", style=f"bold {colors['primary']}", width=12)
            
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
            console.print(f"\nTotal papers: {len(library)}", style=f"bold {colors['primary']}")
            
        except APIClientError as e:
            console.print(f"Error: {e.message}", style=f"bold {colors['error']}")
    
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
                console.print(result.get("message", "Failed to fetch library"), style=f"bold {colors['error']}")
                return
            
            library = result.get("papers", [])
            
            if not library:
                console.print("Your library is empty.", style=f"bold {colors['warning']}")
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
                border_style=f"bold {colors['primary']}"
            ))
            
        except APIClientError as e:
            console.print(f"Error: {e.message}", style=f"bold {colors['error']}")
    
    asyncio.run(_show_stats())
