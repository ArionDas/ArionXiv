"""Search command for ArionXiv CLI"""

import sys
import asyncio
import subprocess
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt
from typing import Optional

from ...arxiv_operations.searcher import arxiv_searcher
from ..ui.theme import create_themed_console, create_themed_table, print_header, style_text, print_success, print_error, print_warning, get_theme_colors
from ..utils.animations import *
from ..utils.command_suggestions import show_command_suggestions

console = create_themed_console()
colors = get_theme_colors()


@click.command()
@click.argument('keywords', required=False, default=None)
@click.option('--category', '-c', help='Filter by category (e.g., cs.AI, cs.LG)')
@click.option('--author', '-a', help='Filter by author name')
def search_command(keywords: Optional[str], category: Optional[str], author: Optional[str]):
    """
    Search for research papers on arXiv.
    
    Returns the top 10 matching papers for you to select from.
    
    Examples:
    \b
        arionxiv search "transformer attention"
        arionxiv search "neural networks" --category cs.LG
        arionxiv search "deep learning" --author "Hinton"
    """
    # Check if keywords are provided
    if not keywords and not author:
        left_to_right_reveal(
            console, 
            "Please provide some keywords to search for papers.", 
            style=f"bold {colors['primary']}", 
            duration=1.0
        )
        console.print()
        left_to_right_reveal(
            console,
            'Example: arionxiv search "transformer attention"',
            style=f"{colors['primary']}",
            duration=0.5
        )
        return
    
    asyncio.run(_search_papers(keywords or "", category, author))


async def _search_papers(keywords: str, category: Optional[str], author: Optional[str]):
    """Execute the paper search"""
    
    left_to_right_reveal(console, f"Search: [{colors['primary']}]{keywords}[/{colors['primary']}]", style="bold", duration=1.0)
    
    with Progress(
        TextColumn(f"[{colors['primary']}]Searching arXiv...[/{colors['primary']}]"),
        "[progress.bar]",
        console=console
    ) as progress:
        task = progress.add_task("", total=100)
        for i in range(100):
            await asyncio.sleep(0.01)
            progress.update(task, advance=1)
        
        try:
            # Perform search based on filters
            if author:
                results = await arxiv_searcher.search_by_author(author=author, max_results=10)
            elif category:
                results = await arxiv_searcher.search_by_category(query=keywords, category=category, max_results=10)
            else:
                results = await arxiv_searcher.search(query=keywords, max_results=10)
            
            progress.remove_task(task)
            
            if not results["success"]:
                print_error(console, f"Search failed: {results.get('error', 'Unknown error')}")
                return
            
            papers = results["papers"]
            
            if not papers:
                print_warning(console, f"No papers found for: {keywords}")
                return
            
        except Exception as e:
            progress.remove_task(task)
            print_error(console, f"Search error: {str(e)}")
            return
    
    # Display results with animation
    left_to_right_reveal(console, f"Found {len(papers)} papers", style=f"bold {colors['primary']}", duration=1.0)
    console.print()
    
    # Show results table with row-by-row animation
    await _display_papers_table_animated(papers)
    
    # Interactive selection
    _show_selection_menu(papers)


async def _display_papers_table_animated(papers):
    """Display papers table with row-by-row animation"""
    
    def create_table_with_rows(num_rows: int) -> Table:
        table = create_themed_table("Results")
        table.expand = True
        table.add_column("#", style="bold white", width=4)
        table.add_column("Title", style="white")
        table.add_column("Authors", style="white", width=30)
        table.add_column("Date", style="white", width=12)
        
        for i in range(num_rows):
            paper = papers[i]
            title = paper.get("title", "Unknown")
            authors = paper.get("authors", [])
            author_str = authors[0] + (f" +{len(authors)-1}" if len(authors) > 1 else "") if authors else "Unknown"
            pub_date = paper.get("published", "")
            if pub_date:
                try:
                    from datetime import datetime
                    date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    date_str = date_obj.strftime("%Y-%m-%d")
                except:
                    date_str = pub_date[:10] if len(pub_date) >= 10 else "Unknown"
            else:
                date_str = "Unknown"
            table.add_row(str(i + 1), title, author_str, date_str)
        return table
    
    await row_by_row_table_reveal(console, create_table_with_rows, len(papers))


def _show_selection_menu(papers):
    """Show selection menu for papers"""
    
    left_to_right_reveal(console, f"Select a paper (1-{len(papers)}) or press Enter to exit:", style=f"bold {colors['primary']}", duration=1.0)
    console.print()
    
    choice = Prompt.ask("Paper number", default="")
    
    if not choice:
        return
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(papers):
            print_error(console, "Invalid selection")
            return
    except ValueError:
        print_error(console, "Enter a valid number")
        return
    
    selected = papers[idx]
    paper_id = selected.get("arxiv_id", "")
    
    if not paper_id:
        print_error(console, "Paper ID not available")
        return
    
    # Show selected paper details
    console.print()
    _display_paper_details(selected)
    
    # Show actions
    left_to_right_reveal(console, f"\nActions:", style=f"bold {colors['primary']}", duration=0.5)
    left_to_right_reveal(console, f"1. Fetch (download PDF)", style=f"bold {colors['primary']}", duration=0.5)
    left_to_right_reveal(console, f"2. Analyze (AI analysis)", style=f"bold {colors['primary']}", duration=0.5)
    left_to_right_reveal(console, f"3. Chat (discuss the paper)", style=f"bold {colors['primary']}", duration=0.5)
    left_to_right_reveal(console, f"4. Exit", style=f"bold {colors['primary']}", duration=0.5)
    
    console.print()
    action = Prompt.ask(f"[bold {colors['primary']}]Action[/bold {colors['primary']}]", choices=["1", "2", "3", "4"], default="4")
    
    if action == "1":
        left_to_right_reveal(console, f"\nFetching paper...", style=f"bold {colors['primary']}")
        subprocess.run([sys.executable, "-m", "arionxiv", "fetch", paper_id])
    elif action == "2":
        left_to_right_reveal(console, f"\nAnalyzing paper...", style=f"bold {colors['primary']}")
        subprocess.run([sys.executable, "-m", "arionxiv", "analyze", paper_id])
    elif action == "3":
        left_to_right_reveal(console, f"\nStarting chat...", style=f"bold {colors['primary']}")
        subprocess.run([sys.executable, "-m", "arionxiv", "chat", "--paper-id", paper_id])
    elif action == "4":
        show_command_suggestions(console, context='search')


def _display_paper_details(paper):
    """Display paper details"""
    
    title = paper.get("title", "Unknown Title")
    
    # Authors
    authors = paper.get("authors", [])
    author_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
    
    # Categories
    categories = paper.get("categories", [])
    categories_str = ', '.join(categories) if categories else ""
    
    # Date
    pub_date = paper.get("published", "Unknown")
    if pub_date and len(pub_date) >= 10:
        pub_date = pub_date[:10]
    
    # arXiv ID
    arxiv_id = paper.get("arxiv_id", "")
    
    # Abstract
    abstract = paper.get("abstract", "No abstract available")
    
    # Build content for panel
    content_lines = [
        f"[bold white]{title}[/bold white]\n",
        f"[{colors['primary']}]Authors:[/{colors['primary']}] {author_str}",
    ]
    
    if categories_str:
        content_lines.append(f"[{colors['primary']}]Categories:[/{colors['primary']}] {categories_str}")
    
    content_lines.append(f"[{colors['primary']}]Published:[/{colors['primary']}] {pub_date}")
    
    if arxiv_id:
        content_lines.append(f"[{colors['primary']}]arXiv ID:[/{colors['primary']}] {arxiv_id}")
        content_lines.append(f"[{colors['primary']}]URL:[/{colors['primary']}] https://arxiv.org/abs/{arxiv_id}")
    
    content_lines.append(f"\n[{colors['primary']}]Abstract:[/{colors['primary']}]")
    content_lines.append(abstract)
    
    content = "\n".join(content_lines)
    
    console.print(Panel(content, title="Selected Paper", border_style=colors['primary']))
