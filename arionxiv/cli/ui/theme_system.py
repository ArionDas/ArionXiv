"""
ArionXiv CLI Theme System - Consolidated theme management and selection
Combines theme.py and theme_selector.py functionality
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.prompt import Prompt, IntPrompt
from typing import Dict, Any
from .logo import display_header
from .global_theme_manager import global_theme_manager

# Import animation utility
try:
    from ..utils.animations import left_to_right_reveal
    ANIMATIONS_AVAILABLE = True
except ImportError:
    ANIMATIONS_AVAILABLE = False
    def left_to_right_reveal(console, text, style="", duration=1.0):
        console.print(text, style=style)

# Available color themes
AVAILABLE_THEMES = {
    'red': {
        'name': 'Classic Red',
        'primary': 'bright_red',
        'secondary': 'bright_red', 
        'accent': 'bright_red',
        'success': 'bright_green',
        'warning': 'bright_yellow',
        'error': 'bright_red',
        'info': 'bright_red',
        'muted': 'dim white'
    },
    'blue': {
        'name': 'Ocean Blue',
        'primary': 'bright_blue',
        'secondary': 'bright_blue',
        'accent': 'bright_blue',
        'success': 'bright_green',
        'warning': 'bright_yellow',
        'error': 'bright_red',
        'info': 'bright_blue',
        'muted': 'dim white'
    },
    'green': {
        'name': 'Forest Green',
        'primary': 'bright_green',
        'secondary': 'bright_green',
        'accent': 'bright_green',
        'success': 'bright_green',
        'warning': 'yellow',
        'error': 'bright_red',
        'info': 'bright_green',
        'muted': 'dim white'
    },
    'purple': {
        'name': 'Royal Purple',
        'primary': 'magenta',
        'secondary': 'magenta',
        'accent': 'magenta',
        'success': 'bright_green',
        'warning': 'bright_yellow',
        'error': 'bright_red',
        'info': 'magenta',
        'muted': 'dim white'
    },
    'amber': {
        'name': 'Warm Amber',
        'primary': 'bright_yellow',
        'secondary': 'bright_yellow',
        'accent': 'bright_yellow',
        'success': 'bright_green',
        'warning': 'bright_yellow',
        'error': 'bright_red',
        'info': 'bright_yellow',
        'muted': 'dim white'
    },
    'cyan': {
        'name': 'Electric Cyan',
        'primary': 'bright_cyan',
        'secondary': 'bright_cyan',
        'accent': 'bright_cyan',
        'success': 'bright_green',
        'warning': 'bright_yellow',
        'error': 'bright_red',
        'info': 'bright_cyan',
        'muted': 'dim white'
    }
}

# Current theme colors (will be set dynamically)
THEME_COLORS = AVAILABLE_THEMES['blue'].copy()  # Default to blue

# ================================
# CORE THEME FUNCTIONS
# ================================

def get_current_theme_color():
    """Get the current theme color from global theme manager"""
    return global_theme_manager.get_current_theme()

def set_theme_colors(theme_name=None):
    """Set the global theme colors"""
    global THEME_COLORS
    
    # Use current theme from global manager if no theme specified
    if theme_name is None:
        theme_name = global_theme_manager.get_current_theme()
    
    if theme_name in AVAILABLE_THEMES:
        THEME_COLORS = AVAILABLE_THEMES[theme_name].copy()
    else:
        # Fallback to blue if theme not found
        THEME_COLORS = AVAILABLE_THEMES['blue'].copy()
        
    # Update global theme manager
    global_theme_manager.set_theme(theme_name)

def get_theme_colors():
    """Get current theme colors, loading from config if needed"""
    current_theme = get_current_theme_color()
    set_theme_colors(current_theme)
    return THEME_COLORS

# ================================
# THEME CREATION FUNCTIONS
# ================================

def create_themed_console():
    """Create a console with current theme"""
    get_theme_colors()  # Ensure theme is loaded
    return Console()

def create_themed_table(title: str = None, show_header: bool = True):
    """Create a table with current theme"""
    colors = get_theme_colors()
    table = Table(
        title=title,
        show_header=show_header,
        header_style=f"bold {colors['primary']}",
        border_style=f"bold {colors['primary']}",
        title_style=f"bold {colors['primary']}"
    )
    return table

def create_themed_panel(content: str, title: str = None, border_style: str = None):
    """Create a panel with current theme"""
    colors = get_theme_colors()
    if border_style is None:
        border_style = f"bold {colors['primary']}"
    
    return Panel(
        content,
        title=f"[bold {colors['primary']}]{title}[/bold {colors['primary']}]" if title else None,
        border_style=border_style,
        padding=(1, 2)
    )

# ================================
# PRINTING FUNCTIONS
# ================================

def print_header(console: Console, title: str = None):
    """Print header with mini logo and optional title"""
    colors = get_theme_colors()
    display_header(console)
    if title:
        console.print(f"\n[bold {colors['primary']}]{title}[/bold {colors['primary']}]\n")

def print_success(console: Console, message: str):
    """Print success message with theme and animation"""
    colors = get_theme_colors()
    left_to_right_reveal(console, f"[SUCCESS] {message}", style=f"bold {colors['primary']}", duration=1.0)

def print_error(console: Console, message: str):
    """Print error message with theme and animation"""
    colors = get_theme_colors()
    left_to_right_reveal(console, f"[ERROR] {message}", style=f"bold {colors['error']}", duration=1.0)

def print_warning(console: Console, message: str):
    """Print warning message with theme and animation"""
    colors = get_theme_colors()
    left_to_right_reveal(console, f"[WARNING] {message}", style=f"bold {colors['warning']}", duration=1.0)

def print_info(console: Console, message: str):
    """Print info message with theme and animation"""
    colors = get_theme_colors()
    left_to_right_reveal(console, f"[INFO] {message}", style=f"bold {colors['primary']}", duration=1.0)

def style_text(text: str, style: str = 'primary'):
    """Apply theme styling to text"""
    colors = get_theme_colors()
    color = colors.get(style, style)
    return f"[{color}]{text}[/{color}]"

# ================================
# THEME SELECTION FUNCTIONS
# ================================

def display_theme_preview(console: Console, theme_name: str, theme_data: Dict[str, str]):
    """Display a clean preview of a theme - just name and color"""
    primary = theme_data['primary']
    
    # Clean, minimal preview
    preview = Text()
    preview.append("ARIONXIV", style=f"bold {primary}")
    
    return Panel(
        preview,
        title=f"[bold {primary}]{theme_data['name']}[/bold {primary}]",
        border_style=f"bold {primary}",
        padding=(0, 2)
    )

def show_all_themes(console: Console):
    """Display all available themes with clean previews"""
    from ..utils.animations import left_to_right_reveal
    colors = get_theme_colors()
    
    left_to_right_reveal(console, "Available Color Themes", style=f"bold {colors['primary']}", duration=0.5)
    console.print()
    
    themes_list = list(AVAILABLE_THEMES.items())
    
    # Show themes in groups of 3
    for i in range(0, len(themes_list), 3):
        panels = []
        for j in range(3):
            if i + j < len(themes_list):
                theme = themes_list[i + j]
                panels.append(display_theme_preview(console, theme[0], theme[1]))
        
        console.print(Columns(panels, equal=True))

def show_themes_table(console: Console):
    """Show themes in a compact list format"""
    from ..utils.animations import left_to_right_reveal
    colors = get_theme_colors()
    
    left_to_right_reveal(console, "Available Themes", style=f"bold {colors['primary']}", duration=0.5)
    console.print()
    
    for i, (theme_key, theme_data) in enumerate(AVAILABLE_THEMES.items(), 1):
        primary = theme_data['primary']
        left_to_right_reveal(
            console, 
            f"  {i}. {theme_data['name']}", 
            style=f"bold {primary}", 
            duration=0.3
        )

def get_theme_choice(console: Console, show_previews: bool = True) -> str:
    """Get user's theme choice with clean interface"""
    from ..utils.animations import left_to_right_reveal
    
    # Show themes in cute boxes
    show_all_themes(console)
    
    console.print()
    themes_list = list(AVAILABLE_THEMES.keys())
    colors = get_theme_colors()
    
    while True:
        try:
            choice = IntPrompt.ask(
                f"[bold {colors['primary']}]Select theme (1-6)[/bold {colors['primary']}]",
                choices=[str(i) for i in range(1, len(themes_list) + 1)],
                show_choices=False
            )
            
            selected_theme = themes_list[choice - 1]
            theme_data = AVAILABLE_THEMES[selected_theme]
            selected_color = theme_data['primary']
            
            console.print()
            left_to_right_reveal(
                console,
                f"Selected: {theme_data['name']}",
                style=f"bold {selected_color}",
                duration=0.5
            )
            
            # Use the selected theme color for the Confirm prompt, show y/n options
            confirm = Prompt.ask(
                f"[bold {selected_color}]Confirm? (y/n)[/bold {selected_color}]",
                choices=["y", "n"],
                default="y",
                show_choices=False
            ).lower()
            
            if confirm == "y":
                return selected_theme
            else:
                console.print()
                continue
                
        except (ValueError, EOFError, KeyboardInterrupt):
            left_to_right_reveal(console, "Cancelled. Using current theme.", style="white", duration=0.5)
            return get_current_theme_color()

def run_theme_selection(console: Console, compact: bool = False) -> str:
    """Run the complete theme selection process - clean and minimal"""
    from ..utils.animations import left_to_right_reveal
    
    console.print()
    colors = get_theme_colors()
    left_to_right_reveal(console, "Theme Selection", style=f"bold {colors['primary']}", duration=0.5)
    console.print()
    
    # Get user choice
    selected_theme = get_theme_choice(console, show_previews=not compact)
    
    # Show clean confirmation
    theme_data = AVAILABLE_THEMES[selected_theme]
    console.print()
    left_to_right_reveal(
        console, 
        f"Theme set to {theme_data['name']}", 
        style=f"bold {theme_data['primary']}", 
        duration=1.0
    )
    console.print()
    
    return selected_theme

def quick_theme_select(console: Console) -> str:
    """Quick theme selection without previews"""
    return get_theme_choice(console, show_previews=False)

# ================================
# VALIDATION AND UTILITIES
# ================================

def is_valid_theme(theme_name: str) -> bool:
    """Check if a theme name is valid"""
    return theme_name in AVAILABLE_THEMES

def get_theme_info(theme_name: str) -> Dict[str, Any]:
    """Get information about a specific theme"""
    if theme_name in AVAILABLE_THEMES:
        return AVAILABLE_THEMES[theme_name].copy()
    return {}

def list_available_themes() -> list:
    """Get list of available theme names"""
    return list(AVAILABLE_THEMES.keys())

def get_theme_names_and_descriptions() -> Dict[str, str]:
    """Get mapping of theme keys to display names"""
    return {key: data['name'] for key, data in AVAILABLE_THEMES.items()}

# ================================
# BACKWARDS COMPATIBILITY
# ================================

# Ensure backwards compatibility with existing imports
def create_themed_console_legacy():
    """Legacy function name for backwards compatibility"""
    return create_themed_console()

# Export all theme-related functionality
__all__ = [
    'AVAILABLE_THEMES',
    'THEME_COLORS',
    'get_current_theme_color',
    'set_theme_colors', 
    'get_theme_colors',
    'create_themed_console',
    'create_themed_table',
    'create_themed_panel',
    'print_header',
    'print_success',
    'print_error', 
    'print_warning',
    'print_info',
    'style_text',
    'display_theme_preview',
    'show_all_themes',
    'show_themes_table',
    'get_theme_choice',
    'run_theme_selection',
    'quick_theme_select',
    'is_valid_theme',
    'get_theme_info',
    'list_available_themes',
    'get_theme_names_and_descriptions'
]