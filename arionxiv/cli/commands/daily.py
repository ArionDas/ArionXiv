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
            console.print(f"[bold {colors['primary']}]Daily dose configuration is managed in settings[/]")
            console.print(f"Use [bold {colors['primary']}]arionxiv settings daily[/] to configure")
        elif run:
            await _run_daily_dose()
        elif view or dose:
            await _view_daily_dose()
        else:
            await _show_daily_dashboard()
    
    asyncio.run(_handle_daily())


async def _run_daily_dose():
    """Run daily dose generation locally (API has timeout limits)"""
    from ...services.unified_daily_dose_service import daily_dose_service
    
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Running Daily Dose Generation[/]")
    console.rule(style=f"bold {colors['primary']}")
    
    try:
        # Get user_id from local session
        current_user = unified_user_service.get_current_user()
        if not current_user:
            print_error(console, "You must be logged in to run daily dose. Use 'arionxiv login' first.")
            return
        
        user_id = current_user.get("id") or current_user.get("user_id")
        if not user_id:
            print_error(console, "Could not determine user ID. Please login again.")
            return
        
        console.print(f"\n[dim {colors['primary']}]Fetching papers and generating personalized summary...[/dim]")
        
        # Progress callback for real-time updates
        def progress_callback(step: str, detail: str = ""):
            if detail:
                console.print(f"  [{colors['secondary']}]• {step}:[/] [white]{detail}[/white]")
            else:
                console.print(f"  [{colors['secondary']}]• {step}[/]")
        
        # Execute locally with progress
        result = await daily_dose_service.execute_daily_dose(user_id, progress_callback=progress_callback)
        
        if result.get("success"):
            dose = result.get("dose", {})
            papers = dose.get("papers", [])
            summary = dose.get("summary", {})
            
            console.print(f"\n[bold {colors['primary']}]✓ Daily dose generated successfully![/]\n")
            
            # Show summary
            if summary:
                total_papers = summary.get("total_papers", 0)
                avg_relevance = summary.get("avg_relevance_score", 0)
                
                console.print(f"[bold {colors['primary']}]Summary:[/]")
                console.print(f"  [bold {colors['primary']}]Papers analyzed: {total_papers}[/]")
                console.print(f"  [bold {colors['primary']}]Average relevance: {avg_relevance:.1f}/10[/]\n")
            
            # Show paper count
            if papers:
                console.print(f"[bold {colors['primary']}]Papers found:[/] {len(papers)}")
            
            console.print(f"\n[dim]View full details with:[/dim]")
            console.print(f"  [bold {colors['primary']}]arionxiv daily --dose[/]")
        else:
            msg = result.get("message", result.get("error", "Unknown error"))
            print_error(console, f"Failed to generate daily dose: {msg}")
            
    except Exception as e:
        logger.error(f"Daily dose run error: {e}", exc_info=True)
        print_error(console, str(e))


async def _view_daily_dose():
    """View the latest daily dose via API"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Your Latest Daily Dose[/]")
    console.rule(style=f"bold {colors['primary']}")
    
    try:
        result = await api_client.get_daily_analysis()
        
        if not result.get("success") or not result.get("dose"):
            print_warning(console, "No daily dose available yet")
            console.print(f"\nGenerate your first daily dose with:")
            console.print(f"  [bold {colors['primary']}]arionxiv daily --run[/]")
            return
        
        # Vercel API returns {"success": True, "dose": {...}}
        daily_dose = result.get("dose")
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
        
        console.print(f"\n[bold {colors['primary']}]Papers found:[/] {summary.get('total_papers', len(papers))}")
        console.print(f"[bold {colors['primary']}]Average relevance:[/] {summary.get('avg_relevance_score', 0):.1f}/10")
        
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
            f"[{colors['error']}]Error:[/] {str(e)}\n\n"
            f"Failed to view your daily dose.\n"
            f"Please try again.",
            title="[bold]Daily Dose View Failed[/bold]",
            border_style=f"bold {colors['error']}"
        )
        console.print(error_panel)


async def _display_papers_list(papers: list, colors: dict):
    """Display list of papers in a table"""
    console.print(f"\n[bold {colors['primary']}]Papers in Your Dose:[/]\n")
    
    table = Table(show_header=True, header_style=f"bold {colors['primary']}", border_style=f"bold {colors['primary']}")
    table.add_column("#", style="bold white", width=3)
    table.add_column("Title", style="white", max_width=55)
    table.add_column("Date", style="white", width=10)
    table.add_column("Score", style="white", width=6, justify="center")
    table.add_column("Category", style="white", width=12)
    
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown Title")
        if len(title) > 52:
            title = title[:49] + "..."
        
        # Parse published date
        published = paper.get("published", "")
        if published:
            try:
                from datetime import datetime
                if isinstance(published, str):
                    pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    date_str = pub_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(published)[:10]
            except:
                date_str = str(published)[:10] if published else "N/A"
        else:
            date_str = "N/A"
        
        score = paper.get("relevance_score", 0)
        if isinstance(score, dict):
            score = score.get("relevance_score", 5)
        
        categories = paper.get("categories", [])
        primary_cat = categories[0] if categories else "N/A"
        
        if score >= 8:
            score_style = f"bold {colors['success']}"
        elif score >= 5:
            score_style = f"bold {colors['primary']}"
        else:
            score_style = f"bold {colors['warning']}"
        
        table.add_row(
            str(i),
            title,
            date_str,
            f"[{score_style}]{score}/10[/{score_style}]",
            primary_cat
        )
    
    console.print(table)


async def _interactive_paper_view(papers: list, colors: dict):
    """Interactive paper selection and analysis view"""
    console.print(f"\n[bold {colors['primary']}]Select a paper to view its analysis (or 0 to exit):[/]")
    
    while True:
        try:
            choice = Prompt.ask(f"[bold {colors['primary']}]Paper number[/]", default="0")
            
            if choice == "0" or choice.lower() == "exit":
                show_command_suggestions(console, context='daily')
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(papers):
                paper = papers[idx]
                await _display_paper_analysis(paper, colors)
                console.print(f"\n[bold {colors['primary']}]Enter another paper number or 0 to exit:[/]")
            else:
                print_warning(console, f"Please enter a number between 1 and {len(papers)}")
                
        except ValueError:
            print_warning(console, "Please enter a valid number")
        except KeyboardInterrupt:
            show_command_suggestions(console, context='daily')
            break


async def _display_paper_analysis(paper: dict, colors: dict):
    """Display detailed analysis for a paper with properly formatted sections"""
    console.rule(style=f"bold {colors['primary']}")
    
    title = paper.get("title", "Unknown Title")
    authors = paper.get("authors", [])
    categories = paper.get("categories", [])
    arxiv_id = paper.get("arxiv_id", "")
    analysis = paper.get("analysis", {})
    
    left_to_right_reveal(console, title, style=f"bold {colors['primary']}", duration=1.0)
    
    console.print(f"\n[bold {colors['primary']}]Authors:[/] {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
    console.print(f"[bold {colors['primary']}]Categories:[/] {', '.join(categories[:3])}")
    console.print(f"[bold {colors['primary']}]ArXiv ID:[/] {arxiv_id}")
    
    if not analysis:
        print_warning(console, "No analysis available for this paper.")
        return
    
    console.print(f"\n[bold {colors['primary']}]─── Analysis ───[/]\n")
    
    # Helper to clean markdown formatting and section headers from LLM responses
    def clean_text(text):
        if not text:
            return text
        import re
        # Remove markdown bold/italic markers
        text = text.replace("**", "").replace("*", "").replace("__", "")
        # Remove any remaining section headers that might have leaked through
        text = re.sub(r'^(?:\d+\.\s*)?(?:SUMMARY|KEY\s*FINDINGS?|METHODOLOGY|SIGNIFICANCE|LIMITATIONS?|RELEVANCE\s*SCORE)[:\s]*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove leading/trailing whitespace
        return text.strip()
    
    # Summary section
    summary = clean_text(analysis.get("summary", ""))
    if summary:
        console.print(f"[bold {colors['primary']}]Summary[/]")
        console.print(f"   {summary}\n")
    
    # Key findings section - display as numbered list
    key_findings = analysis.get("key_findings", [])
    if key_findings:
        console.print(f"[bold {colors['primary']}]Key Findings[/]")
        if isinstance(key_findings, list):
            import re
            for i, finding in enumerate(key_findings, 1):
                if finding:
                    cleaned = clean_text(finding)
                    # Remove leading numbers that might have leaked through (e.g., "1. " at start)
                    cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', cleaned)
                    if cleaned:
                        console.print(f"  {cleaned}")
        else:
            console.print(f"   {clean_text(key_findings)}")
        console.print()
    
    # Methodology section
    methodology = clean_text(analysis.get("methodology", ""))
    if methodology:
        console.print(f"[bold {colors['primary']}]Methodology[/]")
        console.print(f"   {methodology}\n")
    
    # Significance section
    significance = clean_text(analysis.get("significance", ""))
    if significance:
        console.print(f"[bold {colors['primary']}]Significance[/]")
        console.print(f"   {significance}\n")
    
    # Limitations section - handle as list or text
    limitations = analysis.get("limitations", "")
    if limitations:
        console.print(f"[bold {colors['primary']}]Limitations[/]")
        if isinstance(limitations, list):
            for lim in limitations:
                cleaned = clean_text(lim)
                if cleaned:
                    console.print(f"   • {cleaned}")
        else:
            console.print(f"   {clean_text(limitations)}")
        console.print()
    
    # Relevance score with color coding
    score = analysis.get("relevance_score", 5)
    if score >= 8:
        score_style = colors['success']
    elif score >= 5:
        score_style = colors['primary']
    else:
        score_style = colors['warning']
    
    console.print(f"[bold {colors['primary']}]Relevance Score:[/] [{score_style}]{score}/10[/{score_style}]")
    
    pdf_url = paper.get("pdf_url", "")
    if pdf_url:
        console.print(f"\n[bold {colors['primary']}]PDF:[/] {pdf_url}")
    
    console.rule(style=f"bold {colors['primary']}")


async def _show_daily_dashboard():
    """Show daily dose dashboard via Vercel API"""
    colors = get_theme_colors()
    
    console.print(f"\n[bold {colors['primary']}]Daily Dose Dashboard[/]")
    console.rule(style=f"bold {colors['primary']}")
    
    try:
        # Get settings from Vercel API (new dedicated endpoint)
        settings_result = await api_client.get_daily_dose_settings()
        settings = settings_result.get("settings", {}) if settings_result.get("success") else {}
        
        # Get latest daily dose
        dose_result = await api_client.get_daily_analysis()
        
        # Settings panel
        enabled = settings.get("enabled", False)
        scheduled_time = settings.get("scheduled_time", "Not set")
        max_papers = settings.get("max_papers", 5)
        keywords = settings.get("keywords", [])
        
        status_color = colors['primary'] if enabled else colors['warning']
        
        settings_content = (
            f"[bold]Status:[/bold] [bold {status_color}]{'Enabled' if enabled else 'Disabled'}[/bold {status_color}]\n"
            f"[bold]Scheduled Time (UTC):[/bold] [bold {colors['primary']}] {scheduled_time if scheduled_time else 'Not configured'}[/]\n"
            f"[bold]Max Papers:[/bold] [bold {colors['primary']}] {max_papers}[/]\n"
            f"[bold]Keywords:[/bold] [bold {colors['primary']}] {', '.join(keywords[:5]) if keywords else 'None configured'}[/]"
        )
        
        settings_panel = Panel(
            settings_content,
            title=f"[bold {colors['primary']}]Settings[/]",
            border_style=f"bold {colors['primary']}"
        )
        console.print(settings_panel)
        
        # Latest dose status
        if dose_result.get("success") and dose_result.get("dose"):
            # Vercel API returns {"success": True, "dose": {...}}
            daily_dose = dose_result.get("dose")
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
                f"[bold]Last Generated:[/bold] [bold {colors['primary']}]{time_str}[/]\n"
                f"[bold]Papers Analyzed:[/bold] [bold {colors['primary']}]{summary.get('total_papers', 0)}[/]\n"
                f"[bold]Avg Relevance:[/bold] [bold {colors['primary']}]{summary.get('avg_relevance_score', 0):.1f}/10[/]\n"
                f"[bold]Status:[/bold] [bold {colors['primary']}]Ready[/]"
            )
            
            dose_panel = Panel(
                dose_content,
                title=f"[bold {colors['primary']}]Latest Dose[/]",
                border_style=f"bold {colors['primary']}"
            )
        else:
            dose_panel = Panel(
                "No daily dose available yet.\n"
                "Generate your first dose with the options below.",
                title=f"[bold {colors['warning']}]Latest Dose[/]",
                border_style=f"bold {colors['warning']}"
            )
        
        console.print(dose_panel)
        
        # Quick actions
        console.print(f"\n[bold {colors['primary']}]Quick Actions:[/]")
        
        actions_table = Table(show_header=False, box=None, padding=(0, 2))
        actions_table.add_column("Command", style=f"bold {colors['primary']}")
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
            border_style=f"bold {colors['error']}"
        )
        console.print(error_panel)


if __name__ == "__main__":
    daily_command()


