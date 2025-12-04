"""
ArionXiv Welcome Dashboard - Unified logo and feature showcase
Clean interface without emojis for professional presentation
"""

import click
import time
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from ..ui.theme import create_themed_console, get_theme_colors
from ..ui.logo import get_ascii_logo
from ..utils.animations import slam_content, slam_columns, left_to_right_reveal

console = create_themed_console()


def show_logo_and_features(console_instance=None, animate: bool = True):
    """Unified function to show logo and features - used by CLI main and welcome command
    
    Args:
        console_instance: Rich console instance
        animate: Whether to use animated reveal effects (default True for welcome command)
    
    Animation sequence (when animate=True):
    1. Shake the logo
    2. Reveal tagline left to right 
    3. Shake feature boxes one by one 
    4. Final text top to bottom
    """
    if console_instance is None:
        console_instance = console
    
    colors = get_theme_colors()
    primary_color = colors['primary']
    
    logo = get_ascii_logo()
    
    if animate:
        # Step 1: SLAM THE LOGO onto screen with impact!
        slam_content(console_instance, logo, style=f"bold {primary_color}", duration=2.0)
        
        # Step 2: REVEAL TAGLINE LEFT TO RIGHT
        console_instance.print()
        tagline = "It would not take all night long on the internet anymore"
        left_to_right_reveal(console_instance, tagline, style=f"bold {primary_color}", duration=2.0)
        tagline = "A humble effort to bring down time taken to read academic papers"
        left_to_right_reveal(console_instance, tagline, style=f"bold {primary_color}", duration=2.0)
        print("\n")
    else:
        logo_text = Text(logo, style=f"bold {primary_color}")
        console_instance.print(logo_text)
        console_instance.print(f"\n[bold {primary_color}]It would not take all night long on the internet anymore[/bold {primary_color}]")
        console_instance.print(f"[bold {primary_color}]A humble effort to bring down time taken to read academic papers[/bold {primary_color}]\n\n")
    
    # Create feature panels
    panel_width = 50
    panel_height = 15
    
    daily_commands = f"""[bold {primary_color}]Personalized Research Feed[/bold {primary_color}]

Get AI-curated papers daily based on your:
- Research categories (cs.AI, cs.LG, etc.)
- Keywords and interests  
- Preferred authors
- Relevance scoring

[bold {primary_color}]Commands:[/bold {primary_color}]
[bold {primary_color}]arionxiv daily[/bold {primary_color}]          View dashboard
[bold {primary_color}]arionxiv daily --run[/bold {primary_color}]    Generate new dose
[bold {primary_color}]arionxiv settings daily[/bold {primary_color}] Configure preferences"""
    
    chat_commands = f"""[bold {primary_color}]Intelligent Paper Chat[/bold {primary_color}]

Chat with research papers using AI:
- Enter arXiv ID when prompted
- Ask questions about content
- Get summaries and insights
- Context-aware conversations

[bold {primary_color}]Commands:[/bold {primary_color}]
[bold {primary_color}]arionxiv chat[/bold {primary_color}]     Start interactive chat"""
    
    research_commands = f"""[bold {primary_color}]Research Tools[/bold {primary_color}]

Powerful research capabilities:
- Search arXiv database
- Fetch papers with metadata
- Analyze content with AI
- Manage personal library

[bold {primary_color}]Commands:[/bold {primary_color}]
[bold {primary_color}]arionxiv search "query"[/bold {primary_color}]  Search papers
[bold {primary_color}]arionxiv fetch ID[/bold {primary_color}]        Download paper
[bold {primary_color}]arionxiv analyze file.pdf[/bold {primary_color}] AI analysis
[bold {primary_color}]arionxiv library[/bold {primary_color}]         Manage collection"""
    
    settings_commands = f"""[bold {primary_color}]Customization[/bold {primary_color}]

Personalize your experience:
- Set research preferences
- Configure daily dose
- Choose UI themes
- Manage user settings

[bold {primary_color}]Commands:[/bold {primary_color}]
[bold {primary_color}]arionxiv settings[/bold {primary_color}]        All settings
[bold {primary_color}]arionxiv settings theme[/bold {primary_color}]  Change theme
[bold {primary_color}]arionxiv preferences[/bold {primary_color}]     Set interests"""
    
    daily_panel = Panel(daily_commands, title=f"[bold {primary_color}]Daily Dose[/bold {primary_color}]", border_style=primary_color, width=panel_width, height=panel_height)
    chat_panel = Panel(chat_commands, title=f"[bold {primary_color}]PDF & Paper Chat[/bold {primary_color}]", border_style=primary_color, width=panel_width, height=panel_height)
    research_panel = Panel(research_commands, title=f"[bold {primary_color}]Research & Analysis[/bold {primary_color}]", border_style=primary_color, width=panel_width, height=panel_height)
    settings_panel = Panel(settings_commands, title=f"[bold {primary_color}]Settings & Preferences[/bold {primary_color}]", border_style=primary_color, width=panel_width, height=panel_height)
    
    # Step 3: SLAM FEATURE BOXES onto screen one by one!
    if animate:
        row1 = Columns([daily_panel, chat_panel], equal=True)
        slam_columns(console_instance, row1, duration=1.0)
        
        row2 = Columns([research_panel, settings_panel], equal=True)
        slam_columns(console_instance, row2, duration=1.0)
    else:
        console_instance.print(Columns([daily_panel, chat_panel], equal=True))
        console_instance.print(Columns([research_panel, settings_panel], equal=True))
    
    print("\n")
    # Quick start guide
    quick_start_content = f"""
[bold {primary_color}]Quick Start Guide:[/bold {primary_color}]
[bold {primary_color}]1. arionxiv settings categories[/bold {primary_color}] - Set your research areas
[bold {primary_color}]2. arionxiv daily --run[/bold {primary_color}] - Generate your first daily dose
[bold {primary_color}]3. arionxiv chat[/bold {primary_color}] - Chat with a research paper
[bold {primary_color}]4. arionxiv search "your topic"[/bold {primary_color}] - Find relevant papers"""
    
    tips_content = f"""
[bold {primary_color}]Pro Tips:[/bold {primary_color}]
- Use tab completion for commands
- Run [bold {primary_color}]command --help[/bold {primary_color}] for detailed options
- Configure preferences once for better results
- Access all settings with [bold {primary_color}]arionxiv settings[/bold {primary_color}]

[bold {primary_color}]Ready to explore research? Choose a feature above![/bold {primary_color}]"""

    # Step 4: FINAL TEXT TOP TO BOTTOM
    if animate:
        quick_start_lines = quick_start_content.strip().split('\n')
        for line in quick_start_lines:
            console_instance.print(line)
            time.sleep(0.4)
        print("\n")
        tips_lines = tips_content.strip().split('\n')
        for line in tips_lines:
            console_instance.print(line)
            time.sleep(0.4)
    else:
        console_instance.print(quick_start_content)
        console_instance.print(tips_content)


@click.command()
@click.option('--quick', '-q', is_flag=True, help='Skip animations for quick display')
def welcome(quick: bool):
    """ArionXiv Welcome Dashboard - Explore all features"""
    show_logo_and_features(console, animate=not quick)


if __name__ == "__main__":
    welcome()
