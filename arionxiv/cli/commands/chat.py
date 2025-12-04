"""
Enhanced Chat Interface for ArionXiv
Chat with research papers using RAG
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...services.unified_paper_service import unified_paper_service
from ...services.unified_analysis_service import rag_chat_system
from ...services.unified_user_service import unified_user_service
from ...services.unified_database_service import unified_database_service
from ...arxiv_operations.client import arxiv_client
from ...arxiv_operations.fetcher import arxiv_fetcher
from ...arxiv_operations.searcher import arxiv_searcher
from ...arxiv_operations.utils import ArxivUtils
from ..ui.theme import create_themed_console, style_text, get_theme_colors, create_themed_table
from ..utils.animations import *
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
    
    await unified_database_service.connect()
    
    console.print(Panel(
        f"[bold {colors['primary']}]ArionXiv Chat System[/bold {colors['primary']}]\n"
        f"[bold {colors['primary']}]Intelligent chat with your research papers[/bold {colors['primary']}]",
        title=f"[bold {colors['primary']}]Chat Interface[/bold {colors['primary']}]",
        border_style=colors['primary']
    ))
    
    try:
        user_data = unified_user_service.get_current_user()
        if not user_data:
            left_to_right_reveal(console, "No user logged in. Please login first with: arionxiv login", style=f"bold {colors['warning']}", duration=1.0)
            return
        
        user_name = user_data.get('user_name', 'default')
        left_to_right_reveal(console, f"\nLogged in as: {user_name}\n", style=f"bold {colors['primary']}", duration=1.0)
        
        selected_paper = None
        
        if paper_id:
            selected_paper = await _fetch_paper_by_id(console, colors, paper_id)
        else:
            selected_paper = await _show_chat_menu(console, colors, user_name)
        
        if not selected_paper:
            # User chose to exit - show command suggestions
            show_command_suggestions(console, context='chat')
            return
        
        if selected_paper == "SESSION_COMPLETED":
            # Session was continued and completed, suggestions already shown
            return
        
        await _start_chat_with_paper(console, colors, user_name, selected_paper)
        
    except KeyboardInterrupt:
        console.print(f"\n[bold {colors['warning']}]Interrupted by user.[/bold {colors['warning']}]")
    except Exception as e:
        console.print(Panel(
            f"[bold {colors['error']}]Error: {str(e)}[/bold {colors['error']}]",
            title=f"[bold {colors['error']}]Chat Error[/bold {colors['error']}]",
            border_style=colors['error']
        ))
        logger.error(f"Chat command error: {str(e)}", exc_info=True)


async def _show_chat_menu(console: Console, colors: Dict, user_name: str) -> Optional[Dict[str, Any]]:
    """Show main chat menu with options"""
    
    while True:  # Loop to allow going back
        user_papers = await unified_database_service.get_user_papers(user_name)
        active_sessions = await unified_database_service.get_active_chat_sessions(user_name)
        
        left_to_right_reveal(console, "What would you like to do?", style=f"bold {colors['primary']}", duration=1.0)
        console.print()
        left_to_right_reveal(console, "1. Search for a new paper", style=f"bold {colors['primary']}", duration=1.0)
        
        if user_papers:
            left_to_right_reveal(console, f"2. Chat with saved papers ({len(user_papers)} saved)", style=f"bold {colors['primary']}", duration=1.0)
        else:
            left_to_right_reveal(console, "2. Chat with saved papers (none saved)", style=f"bold {colors['primary']}", duration=1.0)
        
        if active_sessions:
            left_to_right_reveal(console, f"3. Continue a previous chat ({len(active_sessions)} active)", style=f"bold {colors['primary']}", duration=1.0)
        else:
            left_to_right_reveal(console, "3. Continue a previous chat (no active sessions)", style=f"bold {colors['primary']}", duration=1.0)
        
        left_to_right_reveal(console, "0. Exit", style=f"bold {colors['primary']}", duration=1.0)
        
        choice = Prompt.ask(f"\n[bold {colors['primary']}]Select option[/bold {colors['primary']}]", choices=["0", "1", "2", "3"], default="1")
        
        if choice == "0":
            return None
        elif choice == "1":
            result = await _search_and_select_paper(console, colors)
            if result == "GO_BACK":
                continue  # Go back to menu
            return result
        elif choice == "2":
            if not user_papers:
                left_to_right_reveal(console, "\nNo saved papers. Please search for a paper first.", style=f"bold {colors['warning']}", duration=1.0)
                result = await _search_and_select_paper(console, colors)
                if result == "GO_BACK":
                    continue
                return result
            result = await _select_from_saved_papers(console, colors, user_papers)
            if result == "GO_BACK":
                console.print()  # Add spacing before showing menu again
                continue  # Go back to menu
            return result
        elif choice == "3":
            if not active_sessions:
                left_to_right_reveal(console, "\nNo active chat sessions within the last 24 hours.", style=f"bold {colors['warning']}", duration=1.0)
                continue
            result = await _select_and_continue_session(console, colors, user_name, active_sessions)
            if result == "GO_BACK":
                console.print()
                continue
            if result == "SESSION_CONTINUED":
                # Session was continued and completed, exit (suggestions already shown)
                return "SESSION_COMPLETED"
            return result


async def _search_and_select_paper(console: Console, colors: Dict) -> Optional[Dict[str, Any]]:
    """Search arXiv and let user select a paper. Returns 'GO_BACK' to go back to menu."""
    
    query = Prompt.ask(f"\n[bold {colors['primary']}]Enter search query (or 0 to go back)[/bold {colors['primary']}]")
    
    if not query.strip():
        left_to_right_reveal(console, "No query provided.", style=f"bold {colors['warning']}", duration=1.0)
        return "GO_BACK"
    
    if query.strip() == "0":
        return "GO_BACK"
    
    left_to_right_reveal(console, "\nSearching arXiv...", style=f"bold {colors['primary']}", duration=1.0)
    
    try:
        results = await arxiv_searcher.search(query=query, max_results=10)
        
        if not results.get("success") or not results.get("papers"):
            left_to_right_reveal(console, f"No papers found for: {query}", style=f"bold {colors['warning']}", duration=1.0)
            return "GO_BACK"
        
        papers = results["papers"]
        
        left_to_right_reveal(console, f"\nFound {len(papers)} papers:", style=f"bold {colors['primary']}", duration=1.0)
        console.print()
        
        # Display table row by row with animation
        await _display_papers_table_animated(console, colors, papers, "Search Results")
        
        choice = Prompt.ask(f"\n[bold {colors['primary']}]Select paper (1-{len(papers)}) or 0 to go back[/bold {colors['primary']}]")
        
        try:
            idx = int(choice) - 1
            if idx == -1:
                return "GO_BACK"
            if idx < 0 or idx >= len(papers):
                left_to_right_reveal(console, "Invalid selection.", style=f"bold {colors['error']}", duration=1.0)
                return None
        except ValueError:
            left_to_right_reveal(console, "Invalid input.", style=f"bold {colors['error']}", duration=1.0)
            return None
        
        return papers[idx]
        
    except Exception as e:
        left_to_right_reveal(console, f"Search failed: {str(e)}", style=f"bold {colors['error']}", duration=1.0)
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
    
    left_to_right_reveal(console, "\nYour saved papers:", style=f"bold {colors['primary']}", duration=1.0)
    console.print()
    
    # Display table row by row with animation
    await _display_saved_papers_animated(console, colors, papers)
    
    choice = Prompt.ask(f"\n[bold {colors['primary']}]Select paper (1-{len(papers)}) or 0 to go back[/bold {colors['primary']}]")
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return "GO_BACK"  # Return special value to go back to menu
        if idx < 0 or idx >= len(papers):
            left_to_right_reveal(console, "Invalid selection.", style=f"bold {colors['error']}", duration=1.0)
            return None
    except ValueError:
        left_to_right_reveal(console, "Invalid input.", style=f"bold {colors['error']}", duration=1.0)
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
    """Let user select from active chat sessions and continue. Returns 'GO_BACK', 'SESSION_CONTINUED', or None."""
    from datetime import datetime
    
    left_to_right_reveal(console, "\nActive chat sessions (last 24 hours):", style=f"bold {colors['primary']}", duration=1.0)
    console.print()
    
    # Display sessions table
    await _display_sessions_table_animated(console, colors, sessions)
    
    choice = Prompt.ask(f"\n[bold {colors['primary']}]Select session (1-{len(sessions)}) or 0 to go back[/bold {colors['primary']}]")
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return "GO_BACK"
        if idx < 0 or idx >= len(sessions):
            left_to_right_reveal(console, "Invalid selection.", style=f"bold {colors['error']}", duration=1.0)
            return None
    except ValueError:
        left_to_right_reveal(console, "Invalid input.", style=f"bold {colors['error']}", duration=1.0)
        return None
    
    selected_session = sessions[idx]
    
    # Continue the selected session
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
            
            last_activity = session.get("last_activity")
            if last_activity:
                if isinstance(last_activity, datetime):
                    time_diff = datetime.utcnow() - last_activity
                    if time_diff.total_seconds() < 3600:
                        time_str = f"{int(time_diff.total_seconds() / 60)} min ago"
                    else:
                        time_str = f"{int(time_diff.total_seconds() / 3600)} hrs ago"
                else:
                    time_str = str(last_activity)[:16]
            else:
                time_str = "Unknown"
            
            msg_count = session.get("message_count", len(session.get("messages", [])))
            # Each exchange is 2 messages (user + assistant)
            exchanges = msg_count // 2
            
            table.add_row(str(i + 1), title, time_str, str(exchanges))
        return table
    
    await row_by_row_table_reveal(console, create_table_with_rows, len(sessions))


async def _continue_chat_session(console: Console, colors: Dict, user_name: str, session: Dict[str, Any]):
    """Continue an existing chat session"""
    
    paper_id = session.get('paper_id', '')
    paper_title = session.get('paper_title', 'Unknown Paper')
    
    left_to_right_reveal(console, f"\nResuming chat with: {paper_title}", style=f"bold {colors['primary']}", duration=1.0)
    
    # Fetch paper data to re-index
    existing_paper = await unified_database_service.get_paper(paper_id)
    
    if not existing_paper or not existing_paper.get('full_text'):
        # Try to fetch from ArXiv if not in DB
        left_to_right_reveal(console, "Fetching paper content...", style=f"bold {colors['primary']}", duration=1.0)
        paper_metadata = await asyncio.to_thread(arxiv_client.get_paper_by_id, paper_id)
        
        if not paper_metadata:
            left_to_right_reveal(console, f"Could not retrieve paper {paper_id}. Session may have expired.", style=f"bold {colors['error']}", duration=1.0)
            return
        
        # Download and extract text
        pdf_url = paper_metadata.get('pdf_url')
        if not pdf_url:
            left_to_right_reveal(console, "No PDF URL available for this paper.", style=f"bold {colors['error']}", duration=1.0)
            return
        
        left_to_right_reveal(console, "Downloading PDF...", style=f"bold {colors['primary']}", duration=1.0)
        pdf_path = await asyncio.to_thread(arxiv_fetcher.fetch_paper_sync, paper_id, pdf_url)
        
        if not pdf_path:
            left_to_right_reveal(console, "Failed to download PDF.", style=f"bold {colors['error']}", duration=1.0)
            return
        
        left_to_right_reveal(console, "Extracting text...", style=f"bold {colors['primary']}", duration=1.0)
        from ...services.unified_pdf_service import pdf_processor
        text_content = await pdf_processor.extract_text(pdf_path)
        
        if not text_content:
            left_to_right_reveal(console, "Failed to extract text from PDF.", style=f"bold {colors['error']}", duration=1.0)
            return
        
        paper_info = {
            'arxiv_id': paper_id,
            'title': paper_metadata.get('title', paper_title),
            'authors': paper_metadata.get('authors', []),
            'abstract': paper_metadata.get('summary', paper_metadata.get('abstract', '')),
            'full_text': text_content
        }
    else:
        paper_info = existing_paper
    
    # Continue the chat session
    await rag_chat_system.continue_chat_session(session, paper_info)


async def _fetch_paper_by_id(console: Console, colors: Dict, arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper metadata by arXiv ID"""
    
    left_to_right_reveal(console, f"\nFetching paper {arxiv_id}...", style=f"bold {colors['primary']}", duration=1.0)
    
    paper_metadata = await asyncio.to_thread(arxiv_client.get_paper_by_id, arxiv_id)
    
    if not paper_metadata:
        left_to_right_reveal(console, f"Failed to fetch paper {arxiv_id} from ArXiv", style=f"bold {colors['error']}", duration=1.0)
        return None
    
    return paper_metadata


async def _start_chat_with_paper(console: Console, colors: Dict, user_name: str, paper: Dict[str, Any]):
    """Start chat session with selected paper"""
    
    # Normalize arxiv_id to ensure consistent lookup (strip version numbers)
    raw_arxiv_id = paper.get('arxiv_id') or paper.get('id', '')
    arxiv_id = ArxivUtils.normalize_arxiv_id(raw_arxiv_id)
    title = paper.get('title', arxiv_id)
    
    left_to_right_reveal(console, f"\nSelected: {title}", style=f"bold {colors['primary']}", duration=1.0)
    
    existing_paper = await unified_database_service.get_paper(arxiv_id)
    text_content = None
    pdf_path = None
    text_path = None
    
    # Verify the cached paper matches what we expect
    # Check BOTH the title metadata AND that the full_text actually contains the title
    # This prevents using wrong cached text if there was data corruption
    cached_text_valid = False
    if existing_paper and existing_paper.get('full_text'):
        cached_title = existing_paper.get('title', '').lower().strip()
        expected_title = title.lower().strip()
        full_text_lower = existing_paper['full_text'].lower()
        
        # Normalize titles for comparison (remove newlines, extra spaces)
        import re
        cached_title_normalized = re.sub(r'\s+', ' ', cached_title)
        expected_title_normalized = re.sub(r'\s+', ' ', expected_title)
        
        # Check 1: Title metadata matches (first 40 chars to handle minor differences)
        title_metadata_match = cached_title_normalized[:40] == expected_title_normalized[:40]
        
        # Check 2: The full_text actually contains a significant portion of the title
        # This catches cases where metadata is correct but text content is from wrong paper
        title_words = expected_title_normalized.split()[:5]  # First 5 words of title
        title_phrase = ' '.join(title_words)
        text_contains_title = title_phrase in full_text_lower
        
        if title_metadata_match and text_contains_title:
            cached_text_valid = True
        else:
            logger.warning(f"Cached paper validation failed for {arxiv_id}: "
                          f"metadata_match={title_metadata_match}, text_contains_title={text_contains_title}. "
                          f"Expected title: '{expected_title[:50]}', cached: '{cached_title[:50]}'. Re-downloading.")
    
    if cached_text_valid:
        left_to_right_reveal(console, "Using cached paper text", style=f"bold {colors['primary']}", duration=1.0)
        text_content = existing_paper['full_text']
        pdf_path = existing_paper.get('pdf_path', '')
        text_path = existing_paper.get('text_path', '')
    else:
        pdf_url = paper.get('pdf_url')
        if not pdf_url:
            left_to_right_reveal(console, "No PDF URL found for paper", style=f"bold {colors['error']}", duration=1.0)
            return
        
        left_to_right_reveal(console, "Downloading PDF...", style=f"bold {colors['primary']}", duration=1.0)
        pdf_path = await asyncio.to_thread(arxiv_fetcher.fetch_paper_sync, arxiv_id, pdf_url)
        
        if not pdf_path:
            left_to_right_reveal(console, "Failed to download PDF", style=f"bold {colors['error']}", duration=1.0)
            return
        
        left_to_right_reveal(console, "Extracting text...", style=f"bold {colors['primary']}", duration=1.0)
        from ...services.unified_pdf_service import pdf_processor
        text_content = await pdf_processor.extract_text(pdf_path)
        
        if not text_content:
            left_to_right_reveal(console, "Failed to extract text from PDF", style=f"bold {colors['error']}", duration=1.0)
            return
        
        text_path = pdf_path.replace('.pdf', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        paper_to_save = {
            'arxiv_id': arxiv_id,
            'title': paper.get('title', ''),
            'authors': paper.get('authors', []),
            'abstract': paper.get('summary', paper.get('abstract', '')),
            'categories': paper.get('categories', []),
            'published': paper.get('published', ''),
            'updated': paper.get('updated', ''),
            'pdf_url': paper.get('pdf_url', ''),
            'pdf_path': pdf_path,
            'text_path': text_path,
            'full_text': text_content,
        }
        await unified_database_service.save_paper(paper_to_save)
    
    paper_info = {
        'arxiv_id': arxiv_id,
        'title': paper.get('title', arxiv_id),
        'authors': paper.get('authors', []),
        'abstract': paper.get('summary', paper.get('abstract', '')),
        'published': paper.get('published', ''),
        'pdf_path': pdf_path or '',
        'text_path': text_path or '',
        'full_text': text_content
    }
    
    left_to_right_reveal(console, "\nStarting chat session...\n", style=f"bold {colors['primary']}", duration=1.0)
    await rag_chat_system.start_chat_session([paper_info], user_id=user_name)
    
    await _offer_save_paper(console, colors, user_name, arxiv_id, title)
    
    # Show "What's Next?" after the chat and save prompt
    show_command_suggestions(console, context='chat')


async def _offer_save_paper(console: Console, colors: Dict, user_name: str, arxiv_id: str, title: str):
    """Offer to save paper to user's library after chat"""
    
    user_papers = await unified_database_service.get_user_papers(user_name)
    
    already_saved = any(p.get('arxiv_id') == arxiv_id for p in user_papers)
    if already_saved:
        return
    
    if len(user_papers) >= MAX_USER_PAPERS:
        left_to_right_reveal(console, f"\nYou have reached the maximum of {MAX_USER_PAPERS} saved papers.", style=f"bold {colors['warning']}", duration=1.0)
        left_to_right_reveal(console, "Use 'arionxiv settings' to manage your saved papers.", style=f"bold {colors['primary']}", duration=1.0)
        return
    
    save_choice = Prompt.ask(
        f"\n[bold {colors['primary']}]Save this paper to your library for quick access? (y/n)[/bold {colors['primary']}]",
        choices=["y", "n"],
        default="y"
    )
    
    if save_choice == "y":
        # Show progress indicator while saving (handles potential rate limits gracefully)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"[{colors['primary']}]Saving paper to library...[/{colors['primary']}]", total=None)
            success = await unified_database_service.add_user_paper(user_name, arxiv_id, category="chat")
        
        if success:
            left_to_right_reveal(console, "Paper saved to your library!", style=f"bold {colors['primary']}", duration=1.0)
        else:
            # Show a friendlier error message
            left_to_right_reveal(console, "Could not save paper at this time. Please try again later.", style=f"bold {colors['warning']}", duration=1.0)


async def delete_user_papers_menu(console: Console, colors: Dict, user_name: str):
    """Show menu to delete saved papers - called from settings"""
    
    user_papers = await unified_database_service.get_user_papers(user_name)
    
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
        if len(title) > 47:
            title = title[:44] + "..."
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
                success = await unified_database_service.remove_user_paper(user_name, arxiv_id)
                if success:
                    deleted_count += 1
        
        console.print(f"\n[bold {colors['primary']}]Deleted {deleted_count} paper(s) from your library.[/bold {colors['primary']}]")
        
    except ValueError:
        console.print(f"[bold {colors['error']}]Invalid input. Use comma-separated numbers.[/bold {colors['error']}]")


if __name__ == "__main__":
    chat_command()
