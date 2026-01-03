"""
Enhanced Chat Interface for ArionXiv
Chat with research papers using RAG - Uses hosted API for user data
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.api_client import api_client, APIClientError
from ...services.unified_user_service import unified_user_service
from ...services.unified_database_service import unified_database_service
from ...services.unified_analysis_service import rag_chat_system
from ...arxiv_operations.client import arxiv_client
from ...arxiv_operations.fetcher import arxiv_fetcher
from ...arxiv_operations.searcher import arxiv_searcher
from ...arxiv_operations.utils import ArxivUtils
from ..ui.theme import create_themed_console, style_text, get_theme_colors, create_themed_table
from ..utils.animations import left_to_right_reveal, row_by_row_table_reveal
from ..utils.command_suggestions import show_command_suggestions

logger = logging.getLogger(__name__)
MAX_USER_PAPERS = 10


@click.command()
@click.option('--paper-id', '-p', help='ArXiv ID to chat with directly')
def chat_command(paper_id: Optional[str] = None):
    """Start chat session with papers"""
    asyncio.run(run_chat_command(paper_id))


async def run_chat_command(paper_id: Optional[str] = None):
    """Main chat command interface"""
    console = create_themed_console()
    colors = get_theme_colors()
    
    # Initialize database connection for RAG session management
    try:
        await unified_database_service.connect_mongodb()
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Chat sessions will be stored in memory only.")
    
    console.print(Panel(
        f"[bold {colors['primary']}]ArionXiv Chat System[/bold {colors['primary']}]\n"
        f"[bold {colors['primary']}]Intelligent chat with your research papers[/bold {colors['primary']}]",
        title=f"[bold {colors['primary']}]Chat Interface[/bold {colors['primary']}]",
        border_style=f"bold {colors['primary']}"
    ))
    
    try:
        # Check authentication
        if not unified_user_service.is_authenticated() and not api_client.is_authenticated():
            left_to_right_reveal(console, "No user logged in. Please login first with: arionxiv login", 
                               style=f"bold {colors['warning']}", duration=1.0)
            return
        
        user_data = unified_user_service.get_current_user()
        user_name = user_data.get('user_name', 'User') if user_data else 'User'
        left_to_right_reveal(console, f"\nLogged in as: {user_name}\n", 
                           style=f"bold {colors['primary']}", duration=1.0)
        
        selected_paper = None
        
        if paper_id:
            selected_paper = await _fetch_paper_by_id(console, colors, paper_id)
        else:
            selected_paper = await _show_chat_menu(console, colors, user_name)
        
        if not selected_paper:
            show_command_suggestions(console, context='chat')
            return
        
        if selected_paper == "SESSION_COMPLETED":
            return
        
        await _start_chat_with_paper(console, colors, user_name, selected_paper)
        
    except KeyboardInterrupt:
        console.print(f"\n[bold {colors['warning']}]Interrupted by user.[/bold {colors['warning']}]")
    except Exception as e:
        console.print(Panel(
            f"[bold {colors['error']}]Error: {str(e)}[/bold {colors['error']}]",
            title=f"[bold {colors['error']}]Chat Error[/bold {colors['error']}]",
            border_style=f"bold {colors['error']}"
        ))
        logger.error(f"Chat command error: {str(e)}", exc_info=True)


async def _get_user_papers_from_api() -> List[Dict]:
    """Get user's saved papers from API"""
    try:
        result = await api_client.get_library(limit=MAX_USER_PAPERS)
        if result.get("success"):
            return result.get("papers", [])
    except APIClientError as e:
        logger.error(f"Failed to fetch user papers from API: {e.message}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to fetch user papers: {e}", exc_info=True)
    return []


async def _get_chat_sessions_from_api() -> List[Dict]:
    """Get active chat sessions from API"""
    try:
        result = await api_client.get_chat_sessions(active_only=True)
        if result.get("success"):
            return result.get("sessions", [])
    except APIClientError as e:
        logger.error(f"Failed to fetch chat sessions from API: {e.message}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to fetch chat sessions: {e}", exc_info=True)
    return []


async def _show_chat_menu(console: Console, colors: Dict, user_name: str) -> Optional[Dict[str, Any]]:
    """Show main chat menu with options"""
    
    while True:
        user_papers = await _get_user_papers_from_api()
        active_sessions = await _get_chat_sessions_from_api()
        
        left_to_right_reveal(console, "What would you like to do?", style=f"bold {colors['primary']}", duration=0.5)
        console.print()
        left_to_right_reveal(console, "1. Search for a new paper", style=f"bold {colors['primary']}", duration=0.5)
        
        if user_papers:
            left_to_right_reveal(console, f"2. Chat with saved papers ({len(user_papers)} saved)", 
                               style=f"bold {colors['primary']}", duration=0.5)
        else:
            left_to_right_reveal(console, "2. Chat with saved papers (none saved)", 
                               style=f"bold {colors['primary']}", duration=0.5)
        
        if active_sessions:
            left_to_right_reveal(console, f"3. Continue a previous chat ({len(active_sessions)} active)", 
                               style=f"bold {colors['primary']}", duration=0.5)
        else:
            left_to_right_reveal(console, "3. Continue a previous chat (no active sessions)", 
                               style=f"bold {colors['primary']}", duration=0.5)
        
        left_to_right_reveal(console, "0. Exit", style=f"bold {colors['primary']}", duration=0.5)
        
        choice = Prompt.ask(f"\n[bold {colors['primary']}]Select option[/bold {colors['primary']}]", 
                          choices=["0", "1", "2", "3"], default="1")
        
        if choice == "0":
            return None
        elif choice == "1":
            result = await _search_and_select_paper(console, colors)
            if result == "GO_BACK":
                continue
            return result
        elif choice == "2":
            if not user_papers:
                left_to_right_reveal(console, "\nNo saved papers. Please search for a paper first.", 
                                   style=f"bold {colors['warning']}", duration=0.5)
                result = await _search_and_select_paper(console, colors)
                if result == "GO_BACK":
                    continue
                return result
            result = await _select_from_saved_papers(console, colors, user_papers)
            if result == "GO_BACK":
                console.print()
                continue
            return result
        elif choice == "3":
            if not active_sessions:
                left_to_right_reveal(console, "\nNo active chat sessions within the last 24 hours.", 
                                   style=f"bold {colors['warning']}", duration=0.5)
                continue
            result = await _select_and_continue_session(console, colors, user_name, active_sessions)
            if result == "GO_BACK":
                console.print()
                continue
            if result == "SESSION_CONTINUED":
                return "SESSION_COMPLETED"
            return result


async def _search_and_select_paper(console: Console, colors: Dict) -> Optional[Dict[str, Any]]:
    """Search arXiv and let user select a paper. Returns 'GO_BACK' to go back to menu."""
    
    query = Prompt.ask(f"\n[bold {colors['primary']}]Enter search query (or 0 to go back)[/bold {colors['primary']}]")
    
    if not query.strip() or query.strip() == "0":
        return "GO_BACK"
    
    left_to_right_reveal(console, "\nSearching arXiv...", style=f"bold {colors['primary']}", duration=0.5)
    
    try:
        results = await arxiv_searcher.search(query=query, max_results=10)
        
        if not results.get("success") or not results.get("papers"):
            left_to_right_reveal(console, f"No papers found for: {query}", 
                               style=f"bold {colors['warning']}", duration=0.5)
            return "GO_BACK"
        
        papers = results["papers"]
        
        left_to_right_reveal(console, f"\nFound {len(papers)} papers:", 
                           style=f"bold {colors['primary']}", duration=0.5)
        console.print()
        
        await _display_papers_table_animated(console, colors, papers, "Search Results")
        
        choice = Prompt.ask(f"\n[bold {colors['primary']}]Select paper (1-{len(papers)}) or 0 to go back[/bold {colors['primary']}]")
        
        try:
            idx = int(choice) - 1
            if idx == -1:
                return "GO_BACK"
            if idx < 0 or idx >= len(papers):
                left_to_right_reveal(console, "Invalid selection.", style=f"bold {colors['error']}", duration=0.5)
                return None
        except ValueError:
            left_to_right_reveal(console, "Invalid input.", style=f"bold {colors['error']}", duration=0.5)
            return None
        
        return papers[idx]
        
    except Exception as e:
        left_to_right_reveal(console, f"Search failed: {str(e)}", style=f"bold {colors['error']}", duration=0.5)
        return "GO_BACK"


async def _display_papers_table_animated(console: Console, colors: Dict, papers: List[Dict], title_str: str):
    """Display papers table with row-by-row animation"""
    
    def create_table_with_rows(num_rows: int) -> Table:
        table = create_themed_table(title_str)
        table.expand = True
        table.add_column("#", style="bold white", width=4)
        table.add_column("Title", style="white")
        table.add_column("Authors", style="white", width=30)
        table.add_column("Date", style="white", width=12)
        
        for i in range(num_rows):
            paper = papers[i]
            title_text = paper.get("title", "Unknown")
            authors = paper.get("authors", [])
            author_str = authors[0] + (f" +{len(authors)-1}" if len(authors) > 1 else "") if authors else "Unknown"
            pub_date = paper.get("published", "")[:10] if paper.get("published") else "Unknown"
            table.add_row(str(i + 1), title_text, author_str, pub_date)
        return table
    
    await row_by_row_table_reveal(console, create_table_with_rows, len(papers))


async def _select_from_saved_papers(console: Console, colors: Dict, papers: List[Dict]) -> Optional[Dict[str, Any]]:
    """Let user select from their saved papers. Returns 'GO_BACK' to go back to menu."""
    
    left_to_right_reveal(console, "\nYour saved papers:", style=f"bold {colors['primary']}", duration=0.5)
    console.print()
    
    await _display_saved_papers_animated(console, colors, papers)
    
    choice = Prompt.ask(f"\n[bold {colors['primary']}]Select paper (1-{len(papers)}) or 0 to go back[/bold {colors['primary']}]")
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return "GO_BACK"
        if idx < 0 or idx >= len(papers):
            left_to_right_reveal(console, "Invalid selection.", style=f"bold {colors['error']}", duration=0.5)
            return None
    except ValueError:
        left_to_right_reveal(console, "Invalid input.", style=f"bold {colors['error']}", duration=0.5)
        return None
    
    return papers[idx]


async def _display_saved_papers_animated(console: Console, colors: Dict, papers: List[Dict]):
    """Display saved papers table with row-by-row animation"""
    
    def create_table_with_rows(num_rows: int) -> Table:
        table = create_themed_table("Saved Papers")
        table.expand = True
        table.add_column("#", style="bold white", width=4)
        table.add_column("Title", style="white")
        table.add_column("ArXiv ID", style="white", width=18)
        table.add_column("Added", style="white", width=12)
        
        for i in range(num_rows):
            paper = papers[i]
            title = paper.get("title", "Unknown")
            arxiv_id = paper.get("arxiv_id", "Unknown")
            added_at = paper.get("added_at")
            added_str = added_at.strftime("%Y-%m-%d") if hasattr(added_at, 'strftime') else str(added_at)[:10] if added_at else "Unknown"
            table.add_row(str(i + 1), title, arxiv_id, added_str)
        return table
    
    await row_by_row_table_reveal(console, create_table_with_rows, len(papers))


async def _select_and_continue_session(console: Console, colors: Dict, user_name: str, sessions: List[Dict]) -> Optional[str]:
    """Let user select from active chat sessions and continue."""
    from datetime import datetime
    
    left_to_right_reveal(console, "\nActive chat sessions (last 24 hours):", 
                        style=f"bold {colors['primary']}", duration=0.5)
    console.print()
    
    await _display_sessions_table_animated(console, colors, sessions)
    
    choice = Prompt.ask(f"\n[bold {colors['primary']}]Select session (1-{len(sessions)}) or 0 to go back[/bold {colors['primary']}]")
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return "GO_BACK"
        if idx < 0 or idx >= len(sessions):
            left_to_right_reveal(console, "Invalid selection.", style=f"bold {colors['error']}", duration=0.5)
            return None
    except ValueError:
        left_to_right_reveal(console, "Invalid input.", style=f"bold {colors['error']}", duration=0.5)
        return None
    
    selected_session = sessions[idx]
    await _continue_chat_session(console, colors, user_name, selected_session)
    
    return "SESSION_CONTINUED"


async def _display_sessions_table_animated(console: Console, colors: Dict, sessions: List[Dict]):
    """Display active chat sessions table with row-by-row animation"""
    from datetime import datetime
    
    def create_table_with_rows(num_rows: int) -> Table:
        table = create_themed_table("Active Chat Sessions")
        table.expand = True
        table.add_column("#", style="bold white", width=4)
        table.add_column("Paper Title", style="white")
        table.add_column("Last Activity", style="white", width=18)
        table.add_column("Messages", style="white", width=10)
        
        for i in range(num_rows):
            session = sessions[i]
            title = session.get("paper_title", "Unknown Paper")
            if len(title) > 45:
                title = title[:42] + "..."
            
            last_activity = session.get("last_activity") or session.get("updated_at")
            if last_activity:
                if isinstance(last_activity, datetime):
                    time_diff = datetime.utcnow() - last_activity
                    if time_diff.total_seconds() < 3600:
                        time_str = f"{int(time_diff.total_seconds() / 60)} min ago"
                    else:
                        time_str = f"{int(time_diff.total_seconds() / 3600)} hrs ago"
                elif isinstance(last_activity, str):
                    # Parse ISO datetime string
                    try:
                        dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                        time_diff = datetime.utcnow() - dt.replace(tzinfo=None)
                        if time_diff.total_seconds() < 3600:
                            time_str = f"{int(time_diff.total_seconds() / 60)} min ago"
                        else:
                            time_str = f"{int(time_diff.total_seconds() / 3600)} hrs ago"
                    except:
                        time_str = str(last_activity)[:16]
                else:
                    time_str = str(last_activity)[:16]
            else:
                time_str = "Recent"
            
            msg_count = session.get("message_count", len(session.get("messages", [])))
            exchanges = msg_count // 2
            
            table.add_row(str(i + 1), title, time_str, str(exchanges))
        return table
    
    await row_by_row_table_reveal(console, create_table_with_rows, len(sessions))


async def _continue_chat_session(console: Console, colors: Dict, user_name: str, session: Dict[str, Any]):
    """Continue an existing chat session"""
    
    paper_id = session.get('paper_id', '')
    paper_title = session.get('paper_title', 'Unknown Paper')
    
    left_to_right_reveal(console, f"\nResuming chat with: {paper_title}", style=f"bold {colors['primary']}", duration=0.5)
    
    # Fetch paper data - try local cache first
    paper_metadata = await asyncio.to_thread(arxiv_client.get_paper_by_id, paper_id)
    
    if not paper_metadata:
        left_to_right_reveal(console, f"Could not retrieve paper {paper_id}.", style=f"bold {colors['error']}", duration=0.5)
        return
    
    # Download and extract text
    pdf_url = paper_metadata.get('pdf_url')
    if not pdf_url:
        left_to_right_reveal(console, "No PDF URL available for this paper.", style=f"bold {colors['error']}", duration=0.5)
        return
    
    left_to_right_reveal(console, "Downloading PDF...", style=f"bold {colors['primary']}", duration=0.5)
    pdf_path = await asyncio.to_thread(arxiv_fetcher.fetch_paper_sync, paper_id, pdf_url)
    
    if not pdf_path:
        left_to_right_reveal(console, "Failed to download PDF.", style=f"bold {colors['error']}", duration=0.5)
        return
    
    left_to_right_reveal(console, "Extracting text...", style=f"bold {colors['primary']}", duration=0.5)
    from ...services.unified_pdf_service import pdf_processor
    text_content = await pdf_processor.extract_text(pdf_path)
    
    if not text_content:
        left_to_right_reveal(console, "Failed to extract text from PDF.", style=f"bold {colors['error']}", duration=0.5)
        return
    
    paper_info = {
        'arxiv_id': paper_id,
        'title': paper_metadata.get('title', paper_title),
        'authors': paper_metadata.get('authors', []),
        'abstract': paper_metadata.get('summary', paper_metadata.get('abstract', '')),
        'full_text': text_content
    }
    
    await rag_chat_system.continue_chat_session(session, paper_info)


async def _fetch_paper_by_id(console: Console, colors: Dict, arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper metadata by arXiv ID"""
    
    left_to_right_reveal(console, f"\nFetching paper {arxiv_id}...", style=f"bold {colors['primary']}", duration=0.5)
    
    paper_metadata = await asyncio.to_thread(arxiv_client.get_paper_by_id, arxiv_id)
    
    if not paper_metadata:
        left_to_right_reveal(console, f"Failed to fetch paper {arxiv_id} from ArXiv", style=f"bold {colors['error']}", duration=0.5)
        return None
    
    return paper_metadata


async def _start_chat_with_paper(console: Console, colors: Dict, user_name: str, paper: Dict[str, Any]):
    """Start chat session with selected paper
    
    Flow:
    1. Check if embeddings are cached in DB (24hr TTL)
    2. If cached: load from DB, skip PDF download
    3. If not cached: download PDF, extract text, compute embeddings
    """
    
    raw_arxiv_id = paper.get('arxiv_id') or paper.get('id', '')
    arxiv_id = ArxivUtils.normalize_arxiv_id(raw_arxiv_id)
    title = paper.get('title', arxiv_id)
    
    left_to_right_reveal(console, f"\nSelected: {title}", style=f"bold {colors['primary']}", duration=0.5)
    
    # Check for cached embeddings in DB first
    from ...services.unified_database_service import unified_database_service
    cached_embeddings = None
    try:
        vector_collection = "paper_embeddings"
        cached = await unified_database_service.find_many(
            vector_collection,
            {'doc_id': arxiv_id, 'expires_at': {'$gt': __import__('datetime').datetime.utcnow()}},
            limit=10000
        )
        if cached and len(cached) > 0:
            cached_embeddings = cached
            left_to_right_reveal(console, f"Loading cached embeddings ({len(cached)} chunks)...", style=f"bold {colors['primary']}", duration=0.5)
    except Exception as e:
        logger.debug(f"Could not check cached embeddings: {e}")
    
    text_content = None
    if not cached_embeddings:
        # No cache - need to download and process PDF
        pdf_url = paper.get('pdf_url')
        if not pdf_url:
            # Saved papers may not have pdf_url - fetch from arXiv
            left_to_right_reveal(console, "Fetching paper info from arXiv...", style=f"bold {colors['primary']}", duration=0.5)
            arxiv_paper = arxiv_client.get_paper_by_id(arxiv_id)
            if arxiv_paper:
                pdf_url = arxiv_paper.get('pdf_url')
                paper.update({
                    'pdf_url': pdf_url,
                    'summary': arxiv_paper.get('summary', paper.get('abstract', '')),
                    'authors': arxiv_paper.get('authors', paper.get('authors', []))
                })
        
        if not pdf_url:
            left_to_right_reveal(console, "Could not find PDF URL for this paper", style=f"bold {colors['error']}", duration=0.5)
            return
        
        left_to_right_reveal(console, "Downloading PDF...", style=f"bold {colors['primary']}", duration=0.5)
        pdf_path = await asyncio.to_thread(arxiv_fetcher.fetch_paper_sync, arxiv_id, pdf_url)
        
        if not pdf_path:
            left_to_right_reveal(console, "Failed to download PDF", style=f"bold {colors['error']}", duration=0.5)
            return
        
        left_to_right_reveal(console, "Extracting text...", style=f"bold {colors['primary']}", duration=0.5)
        from ...services.unified_pdf_service import pdf_processor
        text_content = await pdf_processor.extract_text(pdf_path)
        
        if not text_content:
            left_to_right_reveal(console, "Failed to extract text from PDF", style=f"bold {colors['error']}", duration=0.5)
            return
    
    paper_info = {
        'arxiv_id': arxiv_id,
        'title': paper.get('title', arxiv_id),
        'authors': paper.get('authors', []),
        'abstract': paper.get('summary', paper.get('abstract', '')),
        'published': paper.get('published', ''),
        'full_text': text_content or '',  # Empty if using cached embeddings
        '_cached_embeddings': cached_embeddings  # Pass cached embeddings to RAG
    }
    
    left_to_right_reveal(console, "\nStarting chat session...\n", style=f"bold {colors['primary']}", duration=0.5)
    await rag_chat_system.start_chat_session([paper_info], user_id=user_name)
    
    await _offer_save_paper(console, colors, arxiv_id, title)
    show_command_suggestions(console, context='chat')


async def _offer_save_paper(console: Console, colors: Dict, arxiv_id: str, title: str):
    """Offer to save paper to user's library after chat"""
    
    try:
        # Check if already in library
        library_result = await api_client.get_library(limit=100)
        if library_result.get("success"):
            papers = library_result.get("papers", [])
            if any(p.get('arxiv_id') == arxiv_id for p in papers):
                return  # Already saved
            
            if len(papers) >= MAX_USER_PAPERS:
                left_to_right_reveal(console, f"\nYou have reached the maximum of {MAX_USER_PAPERS} saved papers.", style=f"bold {colors['warning']}", duration=0.5)
                left_to_right_reveal(console, "Use 'arionxiv settings' to manage your saved papers.", style=f"bold {colors['primary']}", duration=0.5)
                return
        
        save_choice = Prompt.ask(
            f"\n[bold {colors['primary']}]Save this paper to your library for quick access? (y/n)[/bold {colors['primary']}]",
            choices=["y", "n"],
            default="y"
        )
        
        if save_choice == "y":
            from ...arxiv_operations.client import arxiv_client as arxiv_client_local
            paper_metadata = arxiv_client_local.get_paper_by_id(arxiv_id) or {}
            
            result = await api_client.add_to_library(
                arxiv_id=arxiv_id,
                title=title or paper_metadata.get('title', ''),
                authors=paper_metadata.get('authors', []),
                categories=paper_metadata.get('categories', []),
                abstract=paper_metadata.get('summary', '')
            )
            
            if result.get("success"):
                left_to_right_reveal(console, "Paper saved to your library!", style=f"bold {colors['primary']}", duration=0.5)
            else:
                left_to_right_reveal(console, "Could not save paper at this time.", style=f"bold {colors['warning']}", duration=0.5)
                
    except APIClientError as e:
        logger.debug(f"Error offering to save paper: {e.message}")
    except Exception as e:
        logger.debug(f"Error offering to save paper: {e}")


async def delete_user_papers_menu(console: Console, colors: Dict, user_name: str):
    """Show menu to delete saved papers - called from settings"""
    
    try:
        result = await api_client.get_library(limit=100)
        if not result.get("success"):
            console.print(f"\n[bold {colors['error']}]Failed to fetch library.[/bold {colors['error']}]")
            return
        
        user_papers = result.get("papers", [])
    except APIClientError as e:
        console.print(f"\n[bold {colors['error']}]Failed to fetch library: {e.message}[/bold {colors['error']}]")
        return
    except Exception as e:
        console.print(f"\n[bold {colors['error']}]Failed to fetch library: {e}[/bold {colors['error']}]")
        return
    
    if not user_papers:
        console.print(f"\n[bold {colors['warning']}]No saved papers to delete.[/bold {colors['warning']}]")
        return
    
    console.print(f"\n[bold {colors['primary']}]Your saved papers:[/bold {colors['primary']}]\n")
    
    table = create_themed_table("Saved Papers")
    table.add_column("#", style="bold white", width=3)
    table.add_column("Title", style="white", max_width=50)
    table.add_column("ArXiv ID", style="white", width=15)
    
    for i, paper in enumerate(user_papers):
        title = paper.get("title", "Unknown")
        arxiv_id = paper.get("arxiv_id", "Unknown")
        table.add_row(str(i + 1), title, arxiv_id)
    
    console.print(table)
    
    console.print(f"\n[bold {colors['primary']}]Enter paper numbers to delete (comma-separated, e.g., 1,3,5) or 0 to cancel:[/bold {colors['primary']}]")
    
    choice = Prompt.ask(f"[bold {colors['primary']}]Papers to delete[/bold {colors['primary']}]")
    
    if choice.strip() == "0" or not choice.strip():
        console.print(f"[bold {colors['primary']}]Cancelled.[/bold {colors['primary']}]")
        return
    
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        valid_indices = [i for i in indices if 0 <= i < len(user_papers)]
        
        if not valid_indices:
            console.print(f"[bold {colors['error']}]No valid selections.[/bold {colors['error']}]")
            return
        
        deleted_count = 0
        for idx in valid_indices:
            paper = user_papers[idx]
            arxiv_id = paper.get("arxiv_id")
            if arxiv_id:
                try:
                    result = await api_client.remove_from_library(arxiv_id)
                    if result.get("success"):
                        deleted_count += 1
                except APIClientError as e:
                    logger.error(f"Failed to remove paper from library: {e.message}", exc_info=True)
                    console.print(f"[bold {colors['error']}]Failed to remove paper from library.[/bold {colors['error']}]")
                except Exception as e:
                    logger.error(f"Failed to remove paper from library: {e}", exc_info=True)
                    console.print(f"[bold {colors['error']}]Failed to remove paper from library.[/bold {colors['error']}]")
        
        console.print(f"\n[bold {colors['primary']}]Deleted {deleted_count} paper(s) from your library.[/bold {colors['primary']}]")
        
    except ValueError:
        console.print(f"[bold {colors['error']}]Invalid input. Use comma-separated numbers.[/bold {colors['error']}]")


if __name__ == "__main__":
    chat_command()
