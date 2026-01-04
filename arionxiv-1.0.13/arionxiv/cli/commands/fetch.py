"""Fetch command for ArionXiv CLI"""

import sys
import asyncio
import logging
from pathlib import Path

backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text
from typing import Optional

from ...arxiv_operations.fetcher import arxiv_fetcher
from ...arxiv_operations.client import arxiv_client
from ...arxiv_operations.utils import ArxivUtils
from ...services.unified_pdf_service import pdf_processor
from ..utils.db_config_manager import db_config_manager as config_manager
from ..ui.theme import create_themed_console, print_header, style_text, print_success, print_error, print_warning, get_theme_colors

logger = logging.getLogger(__name__)

console = create_themed_console()

@click.command()
@click.argument('paper_id')
@click.option('--save-path', help='Custom save location for the PDF')
@click.option('--extract-text', is_flag=True, help='Extract text content from PDF')
@click.option('--no-download', is_flag=True, help='Skip PDF download, only fetch metadata')
def fetch_command(paper_id: str, save_path: str, extract_text: bool, no_download: bool):
    """
    Fetch and download a research paper
    
    Examples:
    \b
        arionxiv fetch 2301.07041
        arionxiv fetch 2301.07041 --save-path ./papers/
        arionxiv fetch 2301.07041 --extract-text
        arionxiv fetch 2301.07041 --no-download
    """
    asyncio.run(_fetch_paper(paper_id, save_path, extract_text, no_download))

async def _fetch_paper(paper_id: str, save_path: Optional[str], extract_text: bool, no_download: bool):
    """Execute the paper fetch operation"""
    
    logger.info(f"Fetching paper: {paper_id}, extract_text={extract_text}, no_download={no_download}")
    
    # Get theme colors for consistent styling
    colors = get_theme_colors()
    
    # Clean up paper ID (remove version suffix if present)
    clean_paper_id = ArxivUtils.normalize_arxiv_id(paper_id)
    logger.debug(f"Normalized paper ID: {clean_paper_id}")
    
    # Set up progress columns based on download flag
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}")
    ]
    
    if not no_download:
        progress_columns.append(BarColumn())
    
    with Progress(*progress_columns, console=console) as progress:
        
        # Step 1: Fetch metadata
        metadata_task = progress.add_task("Fetching paper metadata...", total=None)
        
        try:
            logger.debug("Fetching paper metadata from arXiv")
            paper_metadata = arxiv_client.get_paper_by_id(clean_paper_id)
            
            if not paper_metadata:
                logger.warning(f"Paper not found: {paper_id}")
                progress.remove_task(metadata_task)
                console.print(f"Paper not found: {paper_id}", style=colors['error'])
                return
            
            logger.info(f"Metadata fetched for: {paper_metadata.get('title', 'Unknown')[:50]}")
            progress.update(metadata_task, description="Metadata fetched")
            progress.remove_task(metadata_task)
            
            # Display paper info
            _display_paper_info(paper_metadata)
            
            if no_download:
                logger.debug("Download skipped as requested")
                console.print("\nMetadata fetch complete (download skipped)", style=colors['primary'])
                return
            
            # Step 2: Download PDF
            download_task = progress.add_task("Downloading PDF...", total=100)
            
            # Determine save path
            if not save_path:
                downloads_dir = Path(backend_path.parent) / "downloads"
                downloads_dir.mkdir(exist_ok=True)
                save_path = downloads_dir
            else:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Save path: {save_path}")
            
            # Generate filename
            title_clean = "".join(c for c in paper_metadata.get("title", "paper") if c.isalnum() or c in (' ', '-', '_')).rstrip()
            title_clean = title_clean.replace(' ', '_')[:50]  # Limit length
            filename = f"{clean_paper_id}_{title_clean}.pdf"
            file_path = save_path / filename
            
            # Download the PDF
            pdf_url = paper_metadata.get("pdf_url", "")
            if not pdf_url:
                logger.error("No PDF URL found for paper")
                progress.remove_task(download_task)
                console.print("No PDF URL found for this paper", style=colors['error'])
                return
            
            # Use fetcher to download
            logger.info(f"Downloading PDF from: {pdf_url}")
            download_result = await arxiv_fetcher.download_paper(clean_paper_id, str(save_path))
            
            progress.update(download_task, completed=100)
            progress.remove_task(download_task)
            
            if not download_result["success"]:
                logger.error(f"Download failed: {download_result.get('error', 'Unknown error')}")
                console.print(f"Download failed: {download_result.get('error', 'Unknown error')}", style=colors['error'])
                return
            
            downloaded_path = download_result["file_path"]
            console.print(f"\nPDF downloaded to: {downloaded_path}", style=colors['primary'])
            
            # Step 3: Extract text if requested
            if extract_text:
                extract_task = progress.add_task("Extracting text content...", total=None)
                
                try:
                    text_content = pdf_processor.extract_text(downloaded_path)
                    
                    if text_content:
                        # Save text to file
                        text_file = Path(downloaded_path).with_suffix('.txt')
                        with open(text_file, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        
                        progress.update(extract_task, description="Text extracted")
                        progress.remove_task(extract_task)
                        
                        console.print(f"Text extracted to: {text_file}", style=colors['primary'])
                        
                        # Show preview
                        preview = text_content[:500] + "..." if len(text_content) > 500 else text_content
                        console.print(f"\nText Preview:", style="bold")
                        console.print(Panel(preview, border_style="dim"))
                    else:
                        progress.remove_task(extract_task)
                        console.print("Could not extract text from PDF", style=colors['warning'])
                        
                except Exception as e:
                    progress.remove_task(extract_task)
                    console.print(f"Text extraction failed: {str(e)}", style=colors['error'])
            
            # Show next steps
            _show_next_steps(clean_paper_id, downloaded_path)
            
        except Exception as e:
            console.print(f"Fetch error: {str(e)}", style=colors['error'])
            return

def _display_paper_info(paper):
    """Display paper information"""
    # Get theme colors for consistent styling
    colors = get_theme_colors()
    
    console.print(f"\nPaper Information", style=f"bold {colors['primary']}")
    
    title = paper.get("title", "Unknown Title")
    console.print(f"Title: {title}", style="bold")
    
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        author_str = ", ".join(authors)
    else:
        author_str = str(authors)
    console.print(f"Authors: {author_str}", style=colors['primary'])
    
    categories = paper.get("categories", [])
    category_str = ", ".join(categories) if categories else "Unknown"
    console.print(f"Categories: {category_str}", style=colors['info'])
    
    pub_date = paper.get("published", "Unknown")
    console.print(f"Published: {pub_date}", style=colors['warning'])
    
    arxiv_id = paper.get("arxiv_id", "")
    if arxiv_id:
        console.print(f"ArXiv ID: {arxiv_id}", style=colors['primary'])
    
    abstract = paper.get("abstract", "No abstract available")
    if len(abstract) > 300:
        abstract = abstract[:297] + "..."
    
    console.print(f"\nAbstract:", style="bold")
    console.print(Panel(abstract, border_style="dim"))

def _show_next_steps(paper_id: str, file_path: str):
    """Show suggested next steps"""
    console.print(f"\nNext Steps:", style="bold")
    console.print(f"  arionxiv analyze {paper_id} - Get AI analysis of this paper")
    console.print(f"  arionxiv chat - Start an interactive chat")
    console.print(f"  arionxiv library - View your research library")
    
    console.print(f"\nFile location: {file_path}", style="dim")
