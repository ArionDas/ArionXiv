"""
Command Suggestions Utility
Shows helpful next commands after each feature completes
"""

from rich.console import Console
from rich.panel import Panel
from typing import List, Tuple, Optional


def _get_theme_colors():
    """Get theme colors with lazy import to avoid circular imports"""
    try:
        from ..ui.theme_system import get_theme_colors
        return get_theme_colors()
    except ImportError:
        return {'primary': 'cyan', 'secondary': 'blue', 'success': 'green', 
                'warning': 'yellow', 'error': 'red', 'muted': 'dim'}


# Command categories for different contexts
CHAT_COMMANDS = [
    ("arionxiv chat", "Start a new chat session"),
    ("arionxiv search <query>", "Search for more papers"),
    ("arionxiv settings papers", "Manage your saved papers"),
]

SEARCH_COMMANDS = [
    ("arionxiv chat", "Chat with a paper"),
    ("arionxiv search <query>", "Search for different papers"),
    ("arionxiv trending", "See trending papers"),
]

SETTINGS_COMMANDS = [
    ("arionxiv settings", "Back to settings menu"),
    ("arionxiv chat", "Start a chat session"),
    ("arionxiv search <query>", "Search for papers"),
]

TRENDING_COMMANDS = [
    ("arionxiv chat", "Chat with a paper"),
    ("arionxiv search <query>", "Search for specific papers"),
    ("arionxiv daily", "Get your daily digest"),
]

DAILY_COMMANDS = [
    ("arionxiv chat", "Chat with a paper"),
    ("arionxiv search <query>", "Search for papers"),
    ("arionxiv trending", "See trending papers"),
]

ANALYZE_COMMANDS = [
    ("arionxiv chat", "Chat with this paper"),
    ("arionxiv search <query>", "Search for related papers"),
    ("arionxiv settings papers", "Save to your library"),
]

LIBRARY_COMMANDS = [
    ("arionxiv chat", "Chat with a saved paper"),
    ("arionxiv search <query>", "Find new papers"),
    ("arionxiv settings papers", "Manage saved papers"),
]

GENERAL_COMMANDS = [
    ("arionxiv chat", "Chat with papers using AI"),
    ("arionxiv search <query>", "Search arXiv papers"),
    ("arionxiv trending", "See trending papers"),
    ("arionxiv daily", "Daily paper digest"),
    ("arionxiv settings", "Configure preferences"),
]

# Navigation commands
NAVIGATION_COMMANDS = [
    ("arionxiv", "Go to homepage"),
    ("arionxiv --help", "Show all commands"),
]


def show_command_suggestions(
    console: Console,
    context: str = "general",
    custom_commands: Optional[List[Tuple[str, str]]] = None,
    show_navigation: bool = True,
    title: str = "What's Next?"
):
    """
    Show helpful command suggestions after a feature completes.
    
    Args:
        console: Rich console instance
        context: One of 'chat', 'search', 'settings', 'trending', 'daily', 
                'analyze', 'library', 'general'
        custom_commands: Optional list of (command, description) tuples to show instead
        show_navigation: Whether to show navigation commands (homepage, help)
        title: Panel title
    """
    colors = _get_theme_colors()
    
    # Get commands based on context
    context_commands = {
        'chat': CHAT_COMMANDS,
        'search': SEARCH_COMMANDS,
        'settings': SETTINGS_COMMANDS,
        'trending': TRENDING_COMMANDS,
        'daily': DAILY_COMMANDS,
        'analyze': ANALYZE_COMMANDS,
        'library': LIBRARY_COMMANDS,
        'general': GENERAL_COMMANDS,
    }
    
    commands = custom_commands or context_commands.get(context, GENERAL_COMMANDS)
    
    # Build command lines
    lines = []
    for cmd, desc in commands:
        lines.append(
            f"  [bold {colors['primary']}]{cmd}[/bold {colors['primary']}]  "
            f"[white]→  {desc}[/white]"
        )
    
    # Add separator and navigation if requested
    if show_navigation:
        lines.append("")  # Empty line as separator
        lines.append(f"  [white]─────────────────────────────────────[/white]")
        for cmd, desc in NAVIGATION_COMMANDS:
            lines.append(
                f"  [bold {colors['primary']}]{cmd}[/bold {colors['primary']}]  "
                f"[white]→  {desc}[/white]"
            )
    
    console.print()
    console.print(Panel(
        "\n".join(lines),
        title=f"[bold {colors['primary']}]{title}[/bold {colors['primary']}]",
        border_style=f"bold {colors['primary']}",
        padding=(1, 2)
    ))


def show_back_to_home(console: Console):
    """Show a simple message about going back to homepage"""
    colors = _get_theme_colors()
    console.print()
    console.print(
        f"[white]Run [bold {colors['primary']}]arionxiv[/bold {colors['primary']}] "
        f"to go back to homepage[/white]"
    )
