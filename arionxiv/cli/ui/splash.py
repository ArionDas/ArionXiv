"""Splash screen for ArionXiv CLI"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.table import Table
from .theme import get_theme_colors

console = Console()

def show_splash():
    """Display the ArionXiv CLI splash screen"""
    colors = get_theme_colors()
    primary = colors['primary']
    
    # ASCII Art Logo
    logo = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║      █████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗██╗  ██╗██╗██╗   ██╗       ║
    ║     ██╔══██╗██╔══██╗██║██╔═══██╗████╗  ██║╚██╗██╔╝██║██║   ██║       ║
    ║     ███████║██████╔╝██║██║   ██║██╔██╗ ██║ ╚███╔╝ ██║██║   ██║       ║
    ║     ██╔══██║██╔══██╗██║██║   ██║██║╚██╗██║ ██╔██╗ ██║╚██╗ ██╔╝       ║
    ║     ██║  ██║██║  ██║██║╚██████╔╝██║ ╚████║██╔╝ ██╗██║ ╚████╔╝        ║
    ║     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝         ║
    ║                                                                       ║
    ║               AI-Powered Research Paper Analysis                ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    
    console.print(logo, style=f"bold {primary}")
    
    # Feature highlights
    features = Table(title="Key Features", show_header=False, box=None)
    features.add_column("Icon", style="bold white")
    features.add_column("Feature", style="white")
    
    features.add_row("*", "Smart Paper Search & Discovery")
    features.add_row("*", "AI-Powered Paper Analysis")
    features.add_row("*", "Interactive Chat with Papers")
    features.add_row("*", "Daily Intelligence Reports")
    features.add_row("*", "Personal Research Library")
    features.add_row("*", "Trending Papers & Insights")
    
    console.print()
    console.print(Align.center(features))
    
    # Quick start
    quick_start = Panel(
        f"""[bold {colors['primary']}]Quick Start:[/bold {colors['primary']}]
        
[bold]arionxiv search[/bold] "machine learning"     - Search for papers
[bold]arionxiv daily[/bold]                         - Get daily recommendations  
[bold]arionxiv trending[/bold]                      - See trending papers
[bold]arionxiv config[/bold]                        - Configure preferences
        
[white]Type [bold]arionxiv <command> --help[/bold] for detailed usage[/white]""",
        title="Get Started",
        border_style=f"bold {colors['primary']}"
    )
    
    console.print()
    console.print(quick_start)
    
    # Status info
    status_text = Text()
    status_text.append("Version: ", style="white")
    status_text.append("1.0.0", style=f"bold {colors['primary']}")
    status_text.append(" | Mode: ", style="white")
    status_text.append("Terminal", style=f"bold {primary}")
    status_text.append(" | Ready: ", style="white")
    status_text.append("YES", style=f"bold {colors['primary']}")
    
    console.print()
    console.print(Align.center(status_text))

def show_welcome_message():
    """Show a brief welcome message"""
    colors = get_theme_colors()
    welcome = Panel(
        "[bold green]Welcome to ArionXiv CLI![/bold green]\n\n"
        "Your AI-powered research companion in the terminal.\n"
        "Type [bold]arionxiv --help[/bold] to get started.",
        title="Welcome",
        border_style=f"bold {colors['primary']}"
    )
    console.print(welcome)
