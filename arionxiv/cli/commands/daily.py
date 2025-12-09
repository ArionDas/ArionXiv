"""Daily dose command for ArionXiv CLI - Uses hosted API"""

import asyncio
import logging
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..ui.theme import (
    create_themed_console, print_header, style_text, 
    print_success, print_warning, print_error, get_theme_colors
)
from ..utils.animations import left_to_right_reveal, stream_text_response
from ..utils.api_client import api_client, APIClientError
from ..utils.command_suggestions import show_command_suggestions
from ...services.unified_user_service import unified_user_service

console = create_themed_console()
logger = logging.getLogger(__name__)


def _check_auth() -> bool:
    """Check if user is authenticated"""
    if not unified_user_service.is_authenticated() and not api_client.is_authenticated():
        print_error(console, "You must be logged in to use daily dose")
        console.print("\nUse [bold]arionxiv login[/bold] to log in")
        return False
    return True


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
        print_header(console, "ArionXiv Daily Dose")
        
        if not _check_auth():
            return
        
        colors = get_theme_colors()
        
        if config:
            console.print(f"[{colors['primary']}]Daily dose configuration is managed in settings[/{colors['primary']}]")
            console.print(f"Use [{colors['primary']}]arionxiv settings daily[/{colors['primary']}] to configure")
        elif run:
            await _run_daily_dose()
        elif view or dose:
            await _view_daily_dose()
        else:
            await _show_daily_dashboard()
    
    asyncio.run(_handle_daily())


async def _run_daily_dose():
    """Generate a new daily dose via API"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Generating Your Daily Dose[/bold {colors['primary']}]")
    console.print(f"[{colors['primary']}]{'─' * 50}[/{colors['primary']}]")
    
    try:
        with Progress(
            SpinnerColumn(style=colors['primary']),
            TextColumn(f"[{colors['primary']}]{{task.description}}[/{colors['primary']}]"),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("Triggering daily dose generation...", total=None)
            
            result = await api_client.trigger_daily_analysis()
            
            progress.update(task, description="Complete!")
        
        console.print(f"[{colors['primary']}]{'─' * 50}[/{colors['primary']}]")
        
        if result.get("success"):
            papers_count = result.get("papers_count", result.get("total_papers", 0))
            
            print_success(console, "Daily dose triggered successfully")
            console.print(f"[{colors['primary']}]Papers analyzed:[/{colors['primary']}] {papers_count}")
            
            if papers_count > 0:
                console.print(f"\nUse [{colors['primary']}]arionxiv daily --dose[/{colors['primary']}] to view your daily dose")
            else:
                print_warning(console, "No papers found matching your keywords.")
                console.print(f"\nTry adjusting your keywords in settings:")
                console.print(f"  [{colors['primary']}]arionxiv settings daily[/{colors['primary']}]")
        else:
            msg = result.get("message", "Unknown error")
            print_error(console, f"Failed to generate daily dose: {msg}")
            
    except APIClientError as e:
        print_error(console, f"API Error: {e.message}")
    except Exception as e:
        logger.error(f"Daily dose error: {e}", exc_info=True)
        error_panel = Panel(
            f"[{colors['error']}]Error:[/{colors['error']}] {str(e)}\n\n"
            f"Failed to generate your daily dose.\n"
            f"Please check your network connection and try again.",
            title="[bold]Daily Dose Generation Failed[/bold]",
            border_style=colors['error']
        )
        console.print(error_panel)


async def _view_daily_dose():
    """View the latest daily dose via API"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Your Latest Daily Dose[/bold {colors['primary']}]")
    console.print("-" * 50)
    
    try:
        result = await api_client.get_daily_analysis()
        
        if not result.get("success"):
            print_warning(console, "No daily dose available yet")
            console.print(f"\nGenerate your first daily dose with:")
            console.print(f"  [{colors['primary']}]arionxiv daily --run[/{colors['primary']}]")
            return
        
        daily_dose = result.get("data", result)
        papers = daily_dose.get("papers", [])
        summary = daily_dose.get("summary", {})
        generated_at = daily_dose.get("generated_at")
        
        # Format generation time
        if isinstance(generated_at, str):
            try:
                generated_at = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            except ValueError:
                generated_at = datetime.utcnow()
        elif not isinstance(generated_at, datetime):
            generated_at = datetime.utcnow()
        
        time_str = generated_at.strftime("%B %d, %Y at %H:%M")
        
        header_text = f"Daily Dose - {time_str}"
        left_to_right_reveal(console, header_text, style=f"bold {colors['primary']}", duration=1.0)
        
        console.print(f"\n[{colors['primary']}]Papers found:[/{colors['primary']}] {summary.get('total_papers', len(papers))}")
        console.print(f"[{colors['primary']}]Average relevance:[/{colors['primary']}] {summary.get('avg_relevance_score', 0):.1f}/10")
        
        if not papers:
            print_warning(console, "No papers in this daily dose.")
            return
        
        await _display_papers_list(papers, colors)
        await _interactive_paper_view(papers, colors)
        
    except APIClientError as e:
        print_error(console, f"API Error: {e.message}")
    except Exception as e:
        logger.error(f"View daily dose error: {e}", exc_info=True)
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
    console.print(f"\n[bold {colors['primary']}]Select a paper to view its analysis (or 0 to exit):[/bold {colors['primary']}]")
    
    while True:
        try:
            choice = Prompt.ask(f"[{colors['primary']}]Paper number[/{colors['primary']}]", default="0")
            
            if choice == "0" or choice.lower() == "exit":
                show_command_suggestions(console, context='daily')
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(papers):
                paper = papers[idx]
                await _display_paper_analysis(paper, colors)
                console.print(f"\n[{colors['primary']}]Enter another paper number or 0 to exit:[/{colors['primary']}]")
            else:
                print_warning(console, f"Please enter a number between 1 and {len(papers)}")
                
        except ValueError:
            print_warning(console, "Please enter a valid number")
        except KeyboardInterrupt:
            show_command_suggestions(console, context='daily')
            break


async def _display_paper_analysis(paper: dict, colors: dict):
    """Display detailed analysis for a paper"""
    console.print("\n" + "=" * 60)
    
    title = paper.get("title", "Unknown Title")
    authors = paper.get("authors", [])
    categories = paper.get("categories", [])
    arxiv_id = paper.get("arxiv_id", "")
    analysis = paper.get("analysis", {})
    
    left_to_right_reveal(console, title, style=f"bold {colors['primary']}", duration=1.0)
    
    console.print(f"\n[{colors['primary']}]Authors:[/{colors['primary']}] {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
    console.print(f"[{colors['primary']}]Categories:[/{colors['primary']}] {', '.join(categories[:3])}")
    console.print(f"[{colors['primary']}]ArXiv ID:[/{colors['primary']}] {arxiv_id}")
    
    if not analysis:
        print_warning(console, "No analysis available for this paper.")
        return
    
    console.print(f"\n[bold {colors['primary']}]--- Analysis ---[/bold {colors['primary']}]\n")
    
    # Summary
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
    
    # Score
    score = analysis.get("relevance_score", 5)
    if score >= 8:
        score_style = colors['success']
    elif score >= 5:
        score_style = colors['primary']
    else:
        score_style = colors['warning']
    
    console.print(f"\n[bold {colors['primary']}]Relevance Score:[/bold {colors['primary']}] [{score_style}]{score}/10[/{score_style}]")
    
    pdf_url = paper.get("pdf_url", "")
    if pdf_url:
        console.print(f"\n[{colors['primary']}]PDF:[/{colors['primary']}] {pdf_url}")
    
    console.print("\n" + "=" * 60)


async def _show_daily_dashboard():
    """Show daily dose dashboard via API"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Daily Dose Dashboard[/bold {colors['primary']}]")
    console.print("-" * 50)
    
    try:
        # Get settings from API
        settings_result = await api_client.get_settings()
        settings = settings_result.get("settings", {}).get("daily_dose", {}) if settings_result.get("success") else {}
        
        # Get latest daily dose
        dose_result = await api_client.get_daily_analysis()
        
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
        if dose_result.get("success"):
            daily_dose = dose_result.get("data", dose_result)
            generated_at = daily_dose.get("generated_at")
            summary = daily_dose.get("summary", {})
            
            if isinstance(generated_at, str):
                try:
                    generated_at = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                except ValueError:
                    generated_at = datetime.utcnow()
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
        
        console.print(actions_table)
        show_command_suggestions(console, context='daily')
        
    except APIClientError as e:
        print_error(console, f"API Error: {e.message}")
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        error_panel = Panel(
            f"[{colors['error']}]Error:[/{colors['error']}] {str(e)}\n\n"
            f"Failed to load the daily dose dashboard.",
            title="[bold]Dashboard Load Failed[/bold]",
            border_style=colors['error']
        )
        console.print(error_panel)


if __name__ == "__main__":
    daily_command()
