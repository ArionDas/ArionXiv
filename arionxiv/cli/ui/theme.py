"""ArionXiv CLI Theme Configuration - Compatibility layer for theme_system.py"""

# Import everything from the new consolidated theme system
from .theme_system import *

# Maintain backwards compatibility by re-exporting all functions
from .theme_system import (
    AVAILABLE_THEMES,
    THEME_COLORS,
    get_current_theme_color,
    set_theme_colors,
    get_theme_colors,
    create_themed_console,
    create_themed_table,
    create_themed_panel,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    style_text,
    display_theme_preview,
    show_all_themes,
    show_themes_table,
    get_theme_choice,
    run_theme_selection,
    quick_theme_select,
    is_valid_theme,
    get_theme_info,
    list_available_themes,
    get_theme_names_and_descriptions
)
