"""Daily dose command for ArionXiv CLI"""

import sys
import asyncio
import logging
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from datetime import datetime

from ..ui.theme import create_themed_console, print_header, style_text, print_success, print_warning, print_error, get_theme_colors
from ..utils.animations import left_to_right_reveal, stream_text_response
from ...services.unified_user_service import unified_user_service
from ..utils.command_suggestions import show_command_suggestions

console = create_themed_console()
logger = logging.getLogger(__name__)


@click.command()
@click.option('--config', '-c', is_flag=True, help='Configure daily dose preferences')
@click.option('--run', '-r', is_flag=True, help='Run daily analysis now')
@click.option('--view', '-v', is_flag=True, help='View latest daily dose')
@click.option('--dose', '-d', is_flag=True, help='Get your daily dose (same as --view)')
def daily_command(config: bool, run: bool, view: bool, dose: bool):
    """
    Daily dose of research papers - Your personalized paper recommendations
    
    Examples:
    \b
        arionxiv daily --dose       # Get your daily dose
        arionxiv daily --run        # Generate new daily dose
        arionxiv daily --config     # Configure daily dose settings
        arionxiv daily --view       # View latest daily dose
    """
    
    async def _handle_daily():
        # Lazy imports to avoid circular dependencies
        from ...services.unified_scheduler_service import trigger_user_daily_dose
        from ...services.unified_database_service import unified_database_service
        from ...services.unified_daily_dose_service import unified_daily_dose_service
        
        print_header(console, "ArionXiv Daily Dose")
        
        # Check authentication
        if not unified_user_service.is_authenticated():
            print_error(console, "You must be logged in to use daily dose")
            console.print("\nUse [bold]arionxiv auth --login[/bold] to log in")
            return
        
        user = unified_user_service.get_current_user()
        user_id = user["id"]
        
        if config:
            colors = get_theme_colors()
            console.print(f"[{colors['primary']}]Daily dose configuration is managed in settings[/{colors['primary']}]")
            console.print(f"Use [{colors['primary']}]arionxiv settings daily[/{colors['primary']}] to configure")
        elif run:
            await _run_daily_dose(user_id, unified_daily_dose_service)
        elif view or dose:
            await _view_daily_dose(user_id, unified_daily_dose_service)
        else:
            await _show_daily_dashboard(user_id, unified_daily_dose_service)
    
    # Run async function
    asyncio.run(_handle_daily())


async def _run_daily_dose(user_id: str, daily_dose_service):
    """Generate a new daily dose for the user"""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Generating Your Daily Dose[/bold {colors['primary']}]")
    console.print(f"[{colors['primary']}]{'─' * 50}[/{colors['primary']}]")
    
    # Check if keywords are configured
    settings_result = await daily_dose_service.get_user_daily_dose_settings(user_id)
    settings = settings_result.get("settings", {})
    keywords = settings.get("keywords", [])
    categories = settings.get("categories", [])
    
    if not keywords and not categories:
        print_warning(console, "No keywords or categories configured.")
        console.print(f"\nPlease configure your preferences first:")
        console.print(f"  [{colors['primary']}]arionxiv settings daily --keywords 'machine learning, transformers'[/{colors['primary']}]")
        console.print(f"  [{colors['primary']}]arionxiv settings categories --add cs.AI --add cs.LG[/{colors['primary']}]")
        return
    
    try:
        # Track progress state
        progress_state = {"current_paper": 0, "total_papers": 0, "phase": "init"}
        
        with Progress(
            SpinnerColumn(style=colors['primary']),
            TextColumn(f"[{colors['primary']}]{{task.description}}[/{colors['primary']}]"),
            BarColumn(complete_style=colors['primary'], finished_style=colors['success']),
            TaskProgressColumn(),
            console=console,
            transient=False
        ) as progress:
            
            # Main task
            main_task = progress.add_task("Initializing...", total=100)
            
            # Progress callback to update the progress bar
            def show_progress(step: str, detail: str = ""):
                nonlocal progress_state
                
                if "Starting" in step:
                    progress.update(main_task, description="Starting daily dose...", completed=5)
                elif "Loading settings" in step:
                    progress.update(main_task, description="Loading settings...", completed=10)
                elif "Settings loaded" in step:
                    progress.update(main_task, description="Settings loaded", completed=15)
                elif "Searching arXiv" in step:
                    progress.update(main_task, description="Searching arXiv...", completed=20)
                elif "Papers found" in step:
                    # Extract number of papers from detail
                    import re
                    match = re.search(r'(\d+)', detail)
                    if match:
                        progress_state["total_papers"] = int(match.group(1))
                    progress.update(main_task, description=f"Found {progress_state['total_papers']} papers", completed=25)
                elif "Analyzing paper" in step:
                    # Extract paper number
                    import re
                    match = re.search(r'(\d+)/(\d+)', step)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        progress_state["current_paper"] = current
                        progress_state["total_papers"] = total
                        # Scale from 25% to 85% based on paper progress
                        pct = 25 + int((current / total) * 60)
                        short_title = detail[:40] + "..." if len(detail) > 40 else detail
                        progress.update(main_task, description=f"Analyzing [{current}/{total}]: {short_title}", completed=pct)
                elif "analyzed" in step:
                    pass  # Keep current progress
                elif "Saving" in step:
                    progress.update(main_task, description="Saving to database...", completed=90)
                elif "Complete" in step:
                    progress.update(main_task, description="Complete!", completed=100)
            
            # Run daily dose generation with progress callback
            result = await daily_dose_service.execute_daily_dose(user_id, progress_callback=show_progress)
        
        console.print(f"[{colors['primary']}]{'─' * 50}[/{colors['primary']}]")
        
        if result["success"]:
            papers_count = result.get("papers_count", 0)
            execution_time = result.get("execution_time", 0)
            
            print_success(console, f"Daily dose generated successfully")
            console.print(f"[{colors['primary']}]Papers analyzed:[/{colors['primary']}] {papers_count}")
            console.print(f"[{colors['primary']}]Execution time:[/{colors['primary']}] {execution_time:.1f}s")
            
            if papers_count > 0:
                console.print(f"\nUse [{colors['primary']}]arionxiv daily --dose[/{colors['primary']}] to view your daily dose")
            else:
                print_warning(console, "No papers found matching your keywords.")
                console.print(f"\nTry adjusting your keywords in settings:")
                console.print(f"  [{colors['primary']}]arionxiv settings daily[/{colors['primary']}]")
        else:
            print_error(console, f"Failed to generate daily dose: {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        error_panel = Panel(
            f"[{colors['error']}]Error:[/{colors['error']}] {str(e)}\n\n"
            f"Failed to generate your daily dose.\n"
            f"Please check your network connection and try again.",
            title="[bold]Daily Dose Generation Failed[/bold]",
            border_style=colors['error']
        )
        console.print(error_panel)


async def _view_daily_dose(user_id: str, daily_dose_service):
    """View the latest daily dose with paper selection"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Your Latest Daily Dose[/bold {colors['primary']}]")
    console.print("-" * 50)
    
    try:
        # Get latest daily dose
        result = await daily_dose_service.get_user_daily_dose(user_id)
        
        if not result["success"]:
            print_warning(console, "No daily dose available yet")
            console.print(f"\nGenerate your first daily dose with:")
            console.print(f"  [{colors['primary']}]arionxiv daily --run[/{colors['primary']}]")
            return
        
        daily_dose = result["data"]
        papers = daily_dose.get("papers", [])
        summary = daily_dose.get("summary", {})
        generated_at = daily_dose.get("generated_at")
        
        # Format generation time
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
        elif isinstance(generated_at, datetime):
            pass
        else:
            generated_at = datetime.utcnow()
        
        time_str = generated_at.strftime("%B %d, %Y at %H:%M")
        
        # Display header with animation
        header_text = f"Daily Dose - {time_str}"
        left_to_right_reveal(console, header_text, style=f"bold {colors['primary']}", duration=1.0)
        
        console.print(f"\n[{colors['primary']}]Papers found:[/{colors['primary']}] {summary.get('total_papers', 0)}")
        console.print(f"[{colors['primary']}]Average relevance:[/{colors['primary']}] {summary.get('avg_relevance_score', 0):.1f}/10")
        
        if not papers:
            print_warning(console, "No papers in this daily dose.")
            return
        
        # Display papers table
        await _display_papers_list(papers, colors)
        
        # Interactive paper selection
        await _interactive_paper_view(papers, colors)
        
    except Exception as e:
        error_panel = Panel(
            f"[{colors['error']}]Error:[/{colors['error']}] {str(e)}\n\n"
            f"Failed to view your daily dose.\n"
            f"Please try again.",
            title="[bold]Daily Dose View Failed[/bold]",
            border_style=colors['error']
        )
        console.print(error_panel)


async def _display_papers_list(papers: list, colors: dict):
    """Display list of papers in a table"""
    console.print(f"\n[bold {colors['primary']}]Papers in Your Dose:[/bold {colors['primary']}]\n")
    
    table = Table(show_header=True, header_style=f"bold {colors['primary']}", border_style=colors['primary'])
    table.add_column("#", style="bold white", width=3)
    table.add_column("Title", style="white", max_width=55)
    table.add_column("Score", style="white", width=6, justify="center")
    table.add_column("Category", style="white", width=12)
    
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown Title")
        if len(title) > 52:
            title = title[:49] + "..."
        
        score = paper.get("relevance_score", 0)
        if isinstance(score, dict):
            score = score.get("relevance_score", 5)
        
        categories = paper.get("categories", [])
        primary_cat = categories[0] if categories else "N/A"
        
        # Color score based on value
        if score >= 8:
            score_style = colors['success']
        elif score >= 5:
            score_style = colors['primary']
        else:
            score_style = colors['warning']
        
        table.add_row(
            str(i),
            title,
            f"[{score_style}]{score}/10[/{score_style}]",
            primary_cat
        )
    
    console.print(table)


async def _interactive_paper_view(papers: list, colors: dict):
    """Interactive paper selection and analysis view"""
    console.print(f"\n[bold {colors['primary']}]Select a paper to view its analysis (or 0 to return to menu):[/bold {colors['primary']}]")
    
    while True:
        try:
            choice = Prompt.ask(
                f"[{colors['primary']}]Paper number[/{colors['primary']}]",
                default="0"
            )
            
            if choice == "0" or choice.lower() == "exit":
                # Show command suggestions on exit
                show_command_suggestions(console, context='daily')
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(papers):
                paper = papers[idx]
                await _display_paper_analysis(paper, colors)
                
                # Ask if user wants to view another
                console.print(f"\n[{colors['primary']}]Enter another paper number or 0 to return to menu:[/{colors['primary']}]")
            else:
                print_warning(console, f"Please enter a number between 1 and {len(papers)}")
                
        except ValueError:
            print_warning(console, "Please enter a valid number")
        except KeyboardInterrupt:
            show_command_suggestions(console, context='daily')
            break


async def _display_paper_analysis(paper: dict, colors: dict):
    """Display detailed analysis for a paper with streaming"""
    console.print("\n" + "=" * 60)
    
    title = paper.get("title", "Unknown Title")
    authors = paper.get("authors", [])
    categories = paper.get("categories", [])
    arxiv_id = paper.get("arxiv_id", "")
    analysis = paper.get("analysis", {})
    
    # Title with animation
    left_to_right_reveal(console, title, style=f"bold {colors['primary']}", duration=1.0)
    
    console.print(f"\n[{colors['primary']}]Authors:[/{colors['primary']}] {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
    console.print(f"[{colors['primary']}]Categories:[/{colors['primary']}] {', '.join(categories[:3])}")
    console.print(f"[{colors['primary']}]ArXiv ID:[/{colors['primary']}] {arxiv_id}")
    
    if not analysis:
        print_warning(console, "No analysis available for this paper.")
        return
    
    console.print(f"\n[bold {colors['primary']}]--- Analysis ---[/bold {colors['primary']}]\n")
    
    # Summary with streaming
    summary = analysis.get("summary", "")
    if summary:
        console.print(f"[bold {colors['primary']}]Summary:[/bold {colors['primary']}]")
        stream_text_response(console, summary, style="", duration=3.0)
    
    # Key findings
    key_findings = analysis.get("key_findings", [])
    if key_findings:
        console.print(f"\n[bold {colors['primary']}]Key Findings:[/bold {colors['primary']}]")
        for i, finding in enumerate(key_findings[:4], 1):
            if finding:
                console.print(f"  [{colors['primary']}]{i}.[/{colors['primary']}] {finding}")
    
    # Methodology
    methodology = analysis.get("methodology", "")
    if methodology:
        console.print(f"\n[bold {colors['primary']}]Methodology:[/bold {colors['primary']}]")
        console.print(f"  {methodology[:300]}{'...' if len(methodology) > 300 else ''}")
    
    # Significance
    significance = analysis.get("significance", "")
    if significance:
        console.print(f"\n[bold {colors['primary']}]Significance:[/bold {colors['primary']}]")
        console.print(f"  {significance[:300]}{'...' if len(significance) > 300 else ''}")
    
    # Limitations
    limitations = analysis.get("limitations", "")
    if limitations:
        console.print(f"\n[bold {colors['primary']}]Limitations:[/bold {colors['primary']}]")
        console.print(f"  {limitations[:200]}{'...' if len(limitations) > 200 else ''}")
    
    # Relevance score
    score = analysis.get("relevance_score", 5)
    if score >= 8:
        score_style = colors['success']
    elif score >= 5:
        score_style = colors['primary']
    else:
        score_style = colors['warning']
    
    console.print(f"\n[bold {colors['primary']}]Relevance Score:[/bold {colors['primary']}] [{score_style}]{score}/10[/{score_style}]")
    
    # PDF link
    pdf_url = paper.get("pdf_url", "")
    if pdf_url:
        console.print(f"\n[{colors['primary']}]PDF:[/{colors['primary']}] {pdf_url}")
    
    console.print("\n" + "=" * 60)


async def _show_daily_dashboard(user_id: str, daily_dose_service):
    """Show daily dose dashboard"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Daily Dose Dashboard[/bold {colors['primary']}]")
    console.print("-" * 50)
    
    try:
        # Get settings
        settings_result = await daily_dose_service.get_user_daily_dose_settings(user_id)
        settings = settings_result.get("settings", {})
        
        # Get latest daily dose
        dose_result = await daily_dose_service.get_user_daily_dose(user_id)
        
        # Settings panel
        enabled = settings.get("enabled", False)
        scheduled_time = settings.get("scheduled_time", "Not set")
        max_papers = settings.get("max_papers", 5)
        keywords = settings.get("keywords", [])
        
        status_color = colors['primary'] if enabled else colors['warning']
        
        settings_content = (
            f"[bold]Status:[/bold] [{status_color}]{'Enabled' if enabled else 'Disabled'}[/{status_color}]\n"
            f"[bold]Scheduled Time (UTC):[/bold] {scheduled_time if scheduled_time else 'Not configured'}\n"
            f"[bold]Max Papers:[/bold] {max_papers}\n"
            f"[bold]Keywords:[/bold] {', '.join(keywords[:5]) if keywords else 'None configured'}"
        )
        
        settings_panel = Panel(
            settings_content,
            title=f"[bold {colors['primary']}]Settings[/bold {colors['primary']}]",
            border_style=colors['primary']
        )
        console.print(settings_panel)
        
        # Latest dose status
        if dose_result["success"]:
            daily_dose = dose_result["data"]
            generated_at = daily_dose.get("generated_at")
            summary = daily_dose.get("summary", {})
            
            if isinstance(generated_at, str):
                generated_at = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            elif not isinstance(generated_at, datetime):
                generated_at = datetime.utcnow()
            
            time_str = generated_at.strftime("%B %d, %Y at %H:%M")
            
            dose_content = (
                f"[bold]Last Generated:[/bold] {time_str}\n"
                f"[bold]Papers Analyzed:[/bold] {summary.get('total_papers', 0)}\n"
                f"[bold]Avg Relevance:[/bold] {summary.get('avg_relevance_score', 0):.1f}/10\n"
                f"[bold]Status:[/bold] [{colors['primary']}]Ready[/{colors['primary']}]"
            )
            
            dose_panel = Panel(
                dose_content,
                title=f"[bold {colors['primary']}]Latest Dose[/bold {colors['primary']}]",
                border_style=colors['primary']
            )
        else:
            dose_panel = Panel(
                "No daily dose available yet.\n"
                "Generate your first dose with the options below.",
                title=f"[bold {colors['warning']}]Latest Dose[/bold {colors['warning']}]",
                border_style=colors['warning']
            )
        
        console.print(dose_panel)
        
        # Quick actions
        console.print(f"\n[bold {colors['primary']}]Quick Actions:[/bold {colors['primary']}]")
        
        actions_table = Table(show_header=False, box=None, padding=(0, 2))
        actions_table.add_column("Command", style="bold white")
        actions_table.add_column("Description", style="white")
        
        actions_table.add_row("arionxiv daily --dose", "View your latest daily dose")
        actions_table.add_row("arionxiv daily --run", "Generate new daily dose")
        actions_table.add_row("arionxiv settings daily", "Configure daily dose settings")
        actions_table.add_row("arionxiv settings keywords", "Set search keywords")
        
        console.print(actions_table)
        
        # Show command suggestions
        show_command_suggestions(console, context='daily')
        
    except Exception as e:
        error_panel = Panel(
            f"[{colors['error']}]Error:[/{colors['error']}] {str(e)}\n\n"
            f"Failed to load the daily dose dashboard.\n"
            f"Please try again.",
            title="[bold]Dashboard Load Failed[/bold]",
            border_style=colors['error']
        )
        console.print(error_panel)


if __name__ == "__main__":
    daily_command()
