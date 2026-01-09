"""ArionXiv ASCII Logo and Branding"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import shutil

def get_ascii_logo():
    """Returns the ArionXiv ASCII logo"""
    return """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗██╗  ██╗██╗██╗   ██╗ ║
    ║    ██╔══██╗██╔══██╗██║██╔═══██╗████╗  ██║╚██╗██╔╝██║██║   ██║ ║
    ║    ███████║██████╔╝██║██║   ██║██╔██╗ ██║ ╚███╔╝ ██║██║   ██║ ║
    ║    ██╔══██║██╔══██╗██║██║   ██║██║╚██╗██║ ██╔██╗ ██║╚██╗ ██╔╝ ║
    ║    ██║  ██║██║  ██║██║╚██████╔╝██║ ╚████║██╔╝ ██╗██║ ╚████╔╝  ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝   ║
    ║                                                               ║
    ║          Don't read academic papers manually anymore!         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

def get_mini_logo():
    """Returns a mini version of ArionXiv for headers"""
    return "ARIONXIV"

def get_header_bar(console: Console):
    """Create a header bar with mini logo on the right"""
    from .theme import get_theme_colors
    colors = get_theme_colors()
    
    terminal_width = shutil.get_terminal_size().columns
    logo = get_mini_logo()
    padding = " " * (terminal_width - len(logo) - 2)
    
    header_text = Text()
    header_text.append(padding, style="")
    header_text.append(logo, style=f"bold {colors['primary']}")
    
    return header_text

def display_header(console: Console):
    """Display the persistent header with mini logo"""
    from .theme import get_theme_colors
    colors = get_theme_colors()
    
    header = get_header_bar(console)
    console.print(header)
    console.print("─" * shutil.get_terminal_size().columns, style=f"bold {colors['primary']}")

def get_simple_logo():
    """Returns a simpler version of the logo"""
    return """
    ╔══════════════════════════════════════════╗
    ║                                          ║
    ║           A R I O N   X I V              ║
    ║                                          ║
    ║      Avoid reading papers manually!      ║
    ║                                          ║
    ╚══════════════════════════════════════════╝
    """

def display_logo(console: Console, simple: bool = False):
    """Display the ArionXiv logo"""
    from .theme import get_theme_colors
    colors = get_theme_colors()
    
    logo = get_simple_logo() if simple else get_ascii_logo()
    
    # Create rich text with current theme color
    text = Text(logo)
    text.stylize(f"bold {colors['primary']}", 0, len(logo))
    
    console.print()
    console.print(text)
    console.print()

def display_welcome_message(console: Console):
    """Display welcome message for first-time users"""
    from .theme import get_theme_colors
    colors = get_theme_colors()
    
    welcome_text = """
    Welcome to ArionXiv CLI!
    
    ArionXiv is a powerful terminal-based research paper analysis platform
    that helps researchers discover, analyze, and interact with academic papers.
    
    Key Features:
    • Search and discover research papers from arXiv
    • Download and extract text from PDFs
    • AI-powered paper analysis and insights
    • Interactive chat with papers
    • Personal research library management
    
    Get started with: arionxiv search "your research topic"
    For help: arionxiv --help
    """
    
    panel = Panel(
        welcome_text.strip(),
        title=f"[bold {colors['primary']}]Welcome to ArionXiv[/bold {colors['primary']}]",
        border_style=f"bold {colors['primary']}",
        padding=(1, 2)
    )
    
    console.print(panel)
    console.print()

def display_startup_info(console: Console):
    """Display startup information"""
    from .theme import get_theme_colors
    colors = get_theme_colors()
    
    startup_text = Text()
    startup_text.append("ArionXiv CLI", style=f"bold {colors['primary']}")
    startup_text.append(" - Research Paper Analysis Platform\n\n")
    startup_text.append("Type ", style="")
    startup_text.append("'arionxiv --help'", style=f"bold {colors['primary']}")
    startup_text.append(" to see all available commands\n")
    startup_text.append("Visit our documentation for advanced usage tips")
    
    console.print(startup_text)
    console.print()
