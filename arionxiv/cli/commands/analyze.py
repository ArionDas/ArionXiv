"""Analyze command for ArionXiv CLI"""

import sys
import asyncio
import warnings
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.markdown import Markdown
from typing import Optional

from ...arxiv_operations.client import arxiv_client
from ...arxiv_operations.fetcher import arxiv_fetcher
from ...arxiv_operations.utils import ArxivUtils
from ...services.unified_pdf_service import pdf_processor
from ...services.unified_analysis_service import unified_analysis_service
from ..utils.api_client import api_client, APIClientError
from ..ui.theme import create_themed_console, print_header, style_text, print_success, print_error, print_warning, get_theme_colors
from ..utils.command_suggestions import show_command_suggestions
from ..utils.animations import left_to_right_reveal

console = create_themed_console()
logger = logging.getLogger(__name__)

# Maximum papers a user can save
MAX_USER_PAPERS = 10

@click.command()
@click.argument('query')  # Changed from paper_id to query
@click.option('--analysis-type', '-t', type=click.Choice(['summary', 'detailed', 'technical', 'insights']), 
              default='summary', help='Type of analysis to perform')
@click.option('--save-results', '-s', is_flag=True, help='Save analysis results to file')
@click.option('--use-local', '-l', is_flag=True, help='Use local PDF if available')
def analyze_command(query: str, analysis_type: str, save_results: bool, use_local: bool):
    """
    Analyze a research paper with AI
    
    You can provide either:
    - arXiv ID (e.g., 2301.07041) 
    - Paper title or keywords (e.g., "attention is all you need")
    
    Examples:
    \b
        arionxiv analyze "attention is all you need"
        arionxiv analyze 2301.07041 --analysis-type detailed
        arionxiv analyze "transformer architecture" --save-results
        arionxiv analyze "neural machine translation" --analysis-type insights
    """
    # Run with proper session cleanup
    async def run_analysis():
        try:
            await _analyze_paper(query, analysis_type, save_results, use_local)
        finally:
            # Clean up any remaining async sessions
            try:
                import gc
                import asyncio
                import aiohttp
                
                # Close all unclosed sessions
                for obj in gc.get_objects():
                    if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                        try:
                            await obj.close()
                        except:
                            pass
                
                # Give aiohttp time to clean up connections
                await asyncio.sleep(0.1)
                gc.collect()
            except:
                pass  # Ignore cleanup errors
    
    try:
        asyncio.run(run_analysis())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Handle already running event loop
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(run_analysis())
        else:
            console = create_themed_console()
            colors = get_theme_colors()
            left_to_right_reveal(console, f"Analysis error: {str(e)}", style=colors['error'])
    except Exception as e:
        console = create_themed_console()
        colors = get_theme_colors()
        left_to_right_reveal(console, f"Analysis error: {str(e)}", style=colors['error'])

async def _analyze_paper(query: str, analysis_type: str, save_results: bool, use_local: bool):
    """Execute the paper analysis - handles both arXiv IDs and natural language queries"""
    
    logger.info(f"Starting analysis: query='{query}', type={analysis_type}, save={save_results}")
    
    # Get theme colors for consistent styling
    from ..ui.theme import get_theme_colors
    colors = get_theme_colors()
    
    # Determine if query is an arXiv ID or search term
    import re
    arxiv_id_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
    
    if re.match(arxiv_id_pattern, query):
        logger.debug(f"Query recognized as arXiv ID: {query}")
        # Direct arXiv ID provided
        clean_paper_id = ArxivUtils.normalize_arxiv_id(query)
        paper_metadata = None
    else:
        # Search query provided - find the most relevant paper
        left_to_right_reveal(console, f"Searching for papers matching: '{query}'...", style="white")
        search_results = arxiv_client.search_papers(query, max_results=5)
        
        if not search_results:
            left_to_right_reveal(console, f"No papers found matching '{query}'. Please try a different search term.", style=colors['error'])
            return
        
        # Show search results and let user choose (for now, auto-select first result)
        papers = search_results  # search_results is already a list of papers
        selected_paper = papers[0]  # Auto-select most relevant
        
        left_to_right_reveal(console, f"Found paper: {selected_paper.get('title', 'Unknown')}", style=colors['primary'])
        
        clean_paper_id = ArxivUtils.normalize_arxiv_id(selected_paper.get('arxiv_id', ''))
        paper_metadata = selected_paper
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        try:
            # Step 1: Get paper metadata (if not already retrieved from search)
            metadata_task = progress.add_task("Fetching paper metadata...", total=None)
            
            if paper_metadata is None:
                paper_metadata = arxiv_client.get_paper_by_id(clean_paper_id)
                if not paper_metadata:
                    progress.remove_task(metadata_task)
                    left_to_right_reveal(console, f"Paper not found: {clean_paper_id}", style=colors['error'])
                    return
            
            # Ensure arxiv_id is set correctly in paper_metadata (for later save)
            paper_metadata['arxiv_id'] = clean_paper_id
            
            progress.update(metadata_task, description="Metadata fetched")
            progress.remove_task(metadata_task)
            
            # Step 2: Get paper content
            content_task = progress.add_task("Preparing paper content...", total=None)
            
            paper_text = None
            pdf_path = None
            text_file = None
            
            # Try to use local file first if requested
            if use_local:
                downloads_dir = Path(backend_path.parent) / "downloads"
                local_files = list(downloads_dir.glob(f"{clean_paper_id}*.txt"))
                if local_files:
                    text_file = local_files[0]
                    with open(text_file, 'r', encoding='utf-8') as f:
                        paper_text = f.read()
                    progress.update(content_task, description="Using local text file")
                else:
                    left_to_right_reveal(console, "No local text file found, downloading PDF...", style=colors['warning'])
            
            # If no local text, download and extract
            if not paper_text:
                # Download PDF
                pdf_url = f"https://arxiv.org/pdf/{clean_paper_id}.pdf"
                download_result = await arxiv_fetcher.fetch_paper_pdf(clean_paper_id, pdf_url)
                
                if not download_result:
                    progress.remove_task(content_task)
                    left_to_right_reveal(console, "Failed to download paper: PDF download failed", style=colors['error'])
                    return
                
                # Extract text
                pdf_path = download_result
                paper_text = await pdf_processor.extract_text(pdf_path)
                
                if not paper_text:
                    progress.remove_task(content_task)
                    left_to_right_reveal(console, "Failed to extract text from PDF", style=colors['error'])
                    return
                
                # Save extracted text
                text_file = Path(pdf_path).with_suffix('.txt')
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(paper_text)
            
            progress.update(content_task, description="Content prepared")
            progress.remove_task(content_task)
            
            # Step 3: Perform AI analysis
            analysis_task = progress.add_task(f"Performing {analysis_type} analysis...", total=None)
            
            # Build analysis prompt based on type
            analysis_prompt = _build_analysis_prompt(analysis_type, paper_metadata, paper_text)
            
            # Get AI analysis using LLM client directly for better results
            from ...services.llm_client import llm_client
            
            # Create comprehensive content for analysis
            paper_content = f"""
Title: {paper_metadata.get('title', 'Unknown Title')}

Authors: {', '.join(paper_metadata.get('authors', []))}

Categories: {', '.join(paper_metadata.get('categories', []))}

Published: {paper_metadata.get('published', 'Unknown')}

Full Paper Content:
{paper_text}
"""
            
            analysis_result = await llm_client.analyze_paper(paper_content)
            
            progress.update(analysis_task, description="Analysis complete")
            progress.remove_task(analysis_task)
            
            if not analysis_result:
                left_to_right_reveal(console, "Analysis failed: No result returned", style=colors['error'])
                return
            
        except Exception as e:
            progress.stop()
            from rich.panel import Panel
            error_message = str(e)
            error_panel = Panel(
                f"[bold {colors['error']}]Error:[/bold {colors['error']}] {error_message}\n\n"
                f"The analysis could not be completed due to the above issue.\n"
                f"Please address the error and try again.",
                title="Analysis Failed",
                border_style=colors['error']
            )
            console.print(error_panel)
            return
    
    # Display results
    _display_analysis_results(paper_metadata, analysis_result, analysis_type)
    
    # Save results if requested
    if save_results:
        _save_analysis_results(clean_paper_id, analysis_result, analysis_type)
        left_to_right_reveal(console, f"\nAnalysis results saved!", style=colors['primary'])
    
    # Offer to save paper to library (similar to chat flow)
    await _offer_save_paper_to_library(console, colors, paper_metadata)
    
    # Show command suggestions after save prompt
    show_command_suggestions(console, context='analyze')
    
    # Cleanup downloaded files after analysis
    try:
        from ...utils.file_cleanup import file_cleanup_manager
        left_to_right_reveal(console, "\nCleaning up downloaded files...", style=colors['primary'])
        if file_cleanup_manager.cleanup_paper_files(clean_paper_id):
            left_to_right_reveal(console, "Files cleaned up successfully", style=colors['primary'])
    except Exception as cleanup_error:
        left_to_right_reveal(console, f"File cleanup warning: {cleanup_error}", style=colors['warning'])

def _build_analysis_prompt(analysis_type: str, metadata: dict, paper_text: str) -> str:
    """Build analysis prompt based on type"""
    from ...prompts import format_prompt
    
    title = metadata.get('title', 'Unknown')
    authors = ', '.join(metadata.get('authors', []))
    categories = ', '.join(metadata.get('categories', []))
    published = metadata.get('published', 'Unknown')
    
    if analysis_type == "summary":
        return format_prompt("summary_analysis",
                           title=title,
                           abstract=metadata.get('abstract', ''),
                           content=paper_text)
    
    elif analysis_type == "detailed":
        return format_prompt("detailed_analysis",
                           title=title,
                           authors=authors,
                           categories=categories,
                           published=published)
    
    elif analysis_type == "technical":
        return format_prompt("technical_analysis",
                           title=title,
                           authors=authors,
                           categories=categories,
                           published=published)
    
    elif analysis_type == "insights":
        return format_prompt("insights_analysis",
                           title=title,
                           authors=authors,
                           categories=categories,
                           published=published)
    
    return f"Paper Title: {title}\nAuthors: {authors}\nCategories: {categories}\nPublished: {published}"

def _display_animated_panel(console: Console, content: str, title: str, colors: dict):
    """Display a panel with left-to-right animated text"""
    # Use left_to_right animation for the content
    left_to_right_reveal(console, content, style=colors['primary'])
    # Then display in a panel
    console.print(Panel(content, border_style=colors['primary'], title=title))

def _display_analysis_results(metadata: dict, analysis_result: dict, analysis_type: str):
    """Display the analysis results with proper theming"""
    from ..ui.theme import create_themed_console, get_theme_colors, create_themed_panel
    from ..ui.logo import display_header
    
    console = create_themed_console()
    colors = get_theme_colors()
    
    # Display header with logo
    display_header(console)
    
    # Analysis type header with animation
    header_text = f"\nAI Analysis Results ({analysis_type.title()})"
    left_to_right_reveal(console, header_text, style=f"bold {colors['primary']}")
    
    # Paper details in a box
    title = metadata.get("title", "Unknown Title")
    arxiv_id = metadata.get('arxiv_id', 'Unknown')
    authors = metadata.get('authors', [])
    authors_str = ', '.join(authors[:3]) + (f" +{len(authors)-3} more" if len(authors) > 3 else "") if authors else "Unknown"
    categories = metadata.get('categories', [])
    categories_str = ', '.join(categories) if categories else "Unknown"
    published = metadata.get('published', 'Unknown')
    if published and len(published) > 10:
        published = published[:10]
    
    paper_info = f"""[bold]Title:[/bold] {title}

[bold]arXiv ID:[/bold] {arxiv_id}

[bold]Authors:[/bold] {authors_str}

[bold]Categories:[/bold] {categories_str}

[bold]Published:[/bold] {published}"""
    
    console.print(Panel(paper_info, border_style=colors['primary'], title=f"[bold {colors['primary']}]Paper Details[/bold {colors['primary']}]"))
    console.print()
    
    # Analysis content - handle both analysis service and direct LLM results
    if isinstance(analysis_result.get("analysis"), dict):
        # Handle structured analysis result from analysis service
        analysis_data = analysis_result["analysis"]
        
        # Display summary
        if "summary" in analysis_result:
            content = analysis_result["summary"]
            left_to_right_reveal(console, "Summary", style=f"bold {colors['primary']}")
            left_to_right_reveal(console, content, style=colors['primary'])
            console.print()
        
        # Display key findings
        if "key_ideas" in analysis_data:
            key_ideas = analysis_data["key_ideas"]
            if isinstance(key_ideas, list):
                ideas_text = "\n\n".join([f"• {idea}" for idea in key_ideas])
            else:
                ideas_text = str(key_ideas)
            left_to_right_reveal(console, "Key Ideas", style=f"bold {colors['primary']}")
            left_to_right_reveal(console, ideas_text, style=colors['primary'])
            console.print()
        
        # Display technical approach if available
        if "technical_approach" in analysis_data:
            tech = analysis_data["technical_approach"]
            if isinstance(tech, dict) and "methodology" in tech:
                content = tech["methodology"]
                left_to_right_reveal(console, "Methodology", style=f"bold {colors['primary']}")
                left_to_right_reveal(console, content, style=colors['primary'])
                console.print()
        
        # Display significance
        if "significance_impact" in analysis_data:
            sig = analysis_data["significance_impact"]
            if isinstance(sig, dict) and "field_impact" in sig:
                content = sig["field_impact"]
                left_to_right_reveal(console, "Impact", style=f"bold {colors['primary']}")
                left_to_right_reveal(console, content, style=colors['primary'])
                console.print()
    
    elif "summary" in analysis_result:
        # Handle direct LLM analysis results with animated reveal
        summary_content = analysis_result["summary"]
        left_to_right_reveal(console, "── Summary ──", style=f"bold {colors['primary']}")
        left_to_right_reveal(console, summary_content, style="white")
        console.print()
        
        if "key_findings" in analysis_result and analysis_result["key_findings"]:
            findings = analysis_result["key_findings"]
            if isinstance(findings, list) and findings:
                findings_text = "\n".join([f"• {finding}" for finding in findings if finding and finding.strip()])
                if findings_text.strip():
                    left_to_right_reveal(console, "── Key Findings ──", style=f"bold {colors['primary']}")
                    left_to_right_reveal(console, findings_text, style="white")
                    console.print()
            elif isinstance(findings, str) and findings.strip():
                left_to_right_reveal(console, "── Key Findings ──", style=f"bold {colors['primary']}")
                left_to_right_reveal(console, findings, style="white")
                console.print()
        
        if "methodology" in analysis_result and analysis_result["methodology"] and analysis_result["methodology"].strip():
            methodology_content = analysis_result["methodology"]
            left_to_right_reveal(console, "── Methodology ──", style=f"bold {colors['primary']}")
            left_to_right_reveal(console, methodology_content, style="white")
            console.print()
        
        # Add new comprehensive fields with animated reveal
        if "technical_details" in analysis_result and analysis_result["technical_details"] and analysis_result["technical_details"].strip():
            technical_content = analysis_result["technical_details"]
            left_to_right_reveal(console, "── Technical Details ──", style=f"bold {colors['primary']}")
            left_to_right_reveal(console, technical_content, style="white")
            console.print()
        
        if "broader_impact" in analysis_result and analysis_result["broader_impact"] and analysis_result["broader_impact"].strip():
            impact_content = analysis_result["broader_impact"]
            left_to_right_reveal(console, "── Broader Impact & Future Directions ──", style=f"bold {colors['primary']}")
            left_to_right_reveal(console, impact_content, style="white")
            console.print()
        
        if "strengths" in analysis_result and analysis_result["strengths"]:
            strengths = analysis_result["strengths"]
            if isinstance(strengths, list) and strengths:
                strengths_text = "\n".join([f"• {strength}" for strength in strengths if strength and strength.strip()])
                if strengths_text.strip():
                    left_to_right_reveal(console, "── Strengths ──", style=f"bold {colors['primary']}")
                    left_to_right_reveal(console, strengths_text, style="white")
                    console.print()
            elif isinstance(strengths, str) and strengths.strip():
                left_to_right_reveal(console, "── Strengths ──", style=f"bold {colors['primary']}")
                left_to_right_reveal(console, strengths, style="white")
                console.print()
        
        if "limitations" in analysis_result and analysis_result["limitations"]:
            limitations = analysis_result["limitations"]
            if isinstance(limitations, list) and limitations:
                limitations_text = "\n".join([f"• {limitation}" for limitation in limitations if limitation and limitation.strip()])
                if limitations_text.strip():
                    left_to_right_reveal(console, "── Limitations ──", style=f"bold {colors['primary']}")
                    left_to_right_reveal(console, limitations_text, style="white")
                    console.print()
            elif isinstance(limitations, str) and limitations.strip():
                left_to_right_reveal(console, "── Limitations ──", style=f"bold {colors['primary']}")
                left_to_right_reveal(console, limitations, style="white")
                console.print()
    
    else:
        # Handle simple string analysis
        analysis_content = analysis_result.get("analysis", "No analysis available")
        if isinstance(analysis_content, str):
            # Split analysis into sections if it contains numbered points
            sections = analysis_content.split('\n\n')
            
            for i, section in enumerate(sections):
                if section.strip():
                    left_to_right_reveal(console, section.strip(), style="white")
                    console.print()
        else:
            content = str(analysis_content)
            left_to_right_reveal(console, "── Analysis ──", style=f"bold {colors['primary']}")
            left_to_right_reveal(console, content, style="white")
            console.print()


async def _offer_save_paper_to_library(console: Console, colors: dict, paper_metadata: dict):
    """Offer to save paper to user's library after analysis via API"""
    from rich.prompt import Prompt
    from ...services.unified_user_service import unified_user_service
    
    # Check auth
    if not unified_user_service.is_authenticated() and not api_client.is_authenticated():
        return
    
    arxiv_id = paper_metadata.get('arxiv_id', '')
    if not arxiv_id:
        return
    
    try:
        # Check if already in library
        library_result = await api_client.get_library(limit=100)
        if library_result.get("success"):
            papers = library_result.get("papers", [])
            if any(p.get('arxiv_id') == arxiv_id for p in papers):
                left_to_right_reveal(console, "\nThis paper is already in your library.", 
                                   style=f"bold {colors['primary']}", duration=1.0)
                return
            
            if len(papers) >= MAX_USER_PAPERS:
                left_to_right_reveal(console, f"\nYou have reached the maximum of {MAX_USER_PAPERS} saved papers.", 
                                   style=f"bold {colors['warning']}", duration=1.0)
                left_to_right_reveal(console, "Use 'arionxiv settings' to manage your saved papers.", 
                                   style=f"bold {colors['primary']}", duration=1.0)
                return
        
        # Ask user if they want to save
        save_choice = Prompt.ask(
            f"\n[bold {colors['primary']}]Save this paper to your library for quick access? (y/n)[/bold {colors['primary']}]",
            choices=["y", "n"],
            default="y"
        )
        
        if save_choice == "y":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task(f"[{colors['primary']}]Saving paper to library...[/{colors['primary']}]", total=None)
                result = await api_client.add_to_library(arxiv_id=arxiv_id)
            
            if result.get("success"):
                left_to_right_reveal(console, "Paper saved to your library!", 
                                   style=f"bold {colors['primary']}", duration=1.0)
            else:
                left_to_right_reveal(console, "Could not save paper at this time.", 
                                   style=f"bold {colors['warning']}", duration=1.0)
                
    except APIClientError as e:
        logger.debug(f"API error saving paper: {e.message}")
    except Exception as e:
        logger.debug(f"Error saving paper: {e}")


def _save_analysis_results(paper_id: str, analysis_result: dict, analysis_type: str):
    """Save analysis results to file"""
    try:
        # Get theme colors for consistent styling
        from ..ui.theme import get_theme_colors
        colors = get_theme_colors()
        
        # Create analysis directory
        analysis_dir = Path(backend_path.parent) / "analysis_results"
        analysis_dir.mkdir(exist_ok=True)
        
        # Generate filename
        filename = f"{paper_id}_{analysis_type}_analysis.md"
        file_path = analysis_dir / filename
        
        # Prepare content
        content = f"""# Analysis Results

**Paper ID:** {paper_id}
**Analysis Type:** {analysis_type.title()}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis

{analysis_result.get("analysis", "No analysis available")}

## Metadata

{analysis_result.get("metadata", {})}
"""
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        left_to_right_reveal(console, f"Results saved to: {file_path}", style=colors['primary'])
        
    except Exception as e:
        left_to_right_reveal(console, f"Error saving results: {str(e)}", style=colors['error'])

# End of file - old next steps function removed
