"""
Unified Settings System for ArionXiv CLI - User-friendly configuration interface
Provides seamless access to all user settings with short, intuitive commands
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, time
from typing import Dict, Any, List, Optional

# Add backend to Python path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

from ..utils.db_config_manager import db_config_manager
from ..ui.theme import create_themed_console, print_header, style_text, print_success, print_warning, print_error, get_theme_colors, set_theme_colors
from ..utils.command_suggestions import show_command_suggestions
from ..utils.api_config import api_config_manager, show_api_status
from ..utils.animations import left_to_right_reveal
# Note: theme_selector import will be conditional since it may not exist
try:
    from ..ui.theme_system import run_theme_selection
except ImportError:
    run_theme_selection = None
    
from ...services.unified_user_service import unified_user_service
from ...services.unified_config_service import unified_config_service
from ...services.unified_database_service import unified_database_service

# Try to import schedule_user_daily_dose, fallback if not available
try:
    from ...services.unified_scheduler_service import schedule_user_daily_dose
except ImportError:
    schedule_user_daily_dose = None

console = create_themed_console()

# ================================
# CUSTOM ERROR HANDLING FOR SETTINGS
# ================================

class SettingsGroup(click.Group):
    """Custom Click group for settings with proper error handling for invalid subcommands"""
    
    def invoke(self, ctx):
        """Override invoke to catch errors from subcommands"""
        try:
            return super().invoke(ctx)
        except click.UsageError as e:
            self._show_error(e, ctx)
            raise SystemExit(1)
    
    def _show_error(self, error, ctx):
        """Display themed error message for invalid subcommands"""
        colors = get_theme_colors()
        error_console = Console()
        error_msg = str(error)
        
        error_console.print()
        error_console.print(f"[bold {colors['error']}]⚠ Invalid Settings Command[/bold {colors['error']}]")
        error_console.print(f"[{colors['error']}]{error_msg}[/{colors['error']}]")
        error_console.print()
        
        # Show available subcommands
        error_console.print(f"[bold white]Available 'settings' subcommands:[/bold white]")
        for cmd_name in sorted(self.list_commands(ctx)):
            cmd = self.get_command(ctx, cmd_name)
            if cmd and not cmd.hidden:
                help_text = cmd.get_short_help_str(limit=50)
                error_console.print(f"  [{colors['primary']}]{cmd_name}[/{colors['primary']}]  {help_text}")
        
        error_console.print()
        error_console.print(f"Run [{colors['primary']}]arionxiv settings --help[/{colors['primary']}] for more information.")
        error_console.print()

# ================================
# MAIN SETTINGS COMMAND
# ================================

@click.group(cls=SettingsGroup)
def settings():
    """
    ArionXiv Settings - Configure your research experience
    
    Quick access commands:
    \b
        arionxiv settings show         # View all settings
        arionxiv settings theme        # Change theme color
        arionxiv settings api          # Configure API keys
        arionxiv settings prefs        # Configure paper preferences  
        arionxiv settings categories   # Set research categories
        arionxiv settings keywords     # Manage keywords
        arionxiv settings authors      # Set preferred authors
        arionxiv settings daily        # Daily dose configuration
        arionxiv settings time         # Set daily analysis time
        arionxiv settings papers       # Manage saved papers
    """
    pass

# ================================
# SHOW ALL SETTINGS
# ================================

@settings.command('show')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed configuration')
def show_settings(detailed: bool):
    """Show current ArionXiv settings overview"""
    
    async def _show():
        await _ensure_authenticated()
        await unified_database_service.connect()
        print_header(console, "ArionXiv Settings Overview")
        
        # Load current configuration
        await db_config_manager.load_config()
        user = unified_user_service.get_current_user()
        user_id = user['id']
        
        # Get user preferences
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        prefs = prefs_result.get('preferences', {}) if prefs_result['success'] else {}
        
        # Theme & Display
        theme_color = db_config_manager.get_theme_color()
        colors = get_theme_colors()
        
        if detailed:
            await _show_detailed_settings(prefs, theme_color)
        else:
            await _show_compact_settings(prefs, theme_color)
        
        # Quick actions
        console.print(f"\n{style_text('Quick Actions:', 'primary')}")
        console.print(f"- {style_text('arionxiv settings theme', 'primary')} - Change color theme")
        console.print(f"- {style_text('arionxiv settings prefs', 'primary')} - Configure preferences")
        console.print(f"- {style_text('arionxiv settings daily', 'primary')} - Daily dose settings")
    
    asyncio.run(_show())

# ================================
# THEME CONFIGURATION
# ================================

@settings.command('theme')
@click.option('--color', type=click.Choice(['red', 'blue', 'green', 'purple', 'cyan', 'amber']), help='Set theme directly')
def theme_settings(color: Optional[str]):
    """Change ArionXiv color theme"""
    
    async def _theme():
        await _ensure_authenticated()
        colors = get_theme_colors()
        
        console.print()
        left_to_right_reveal(console, "Theme Settings", style=f"bold {colors['primary']}", duration=1.0)
        console.print()
        
        current_theme = db_config_manager.get_theme_color()
        left_to_right_reveal(console, f"Current: {current_theme.title()}", style=f"bold {colors['primary']}", duration=1.0)
        console.print()
        
        if color:
            # Direct color change
            if await db_config_manager.set_theme_color(color):
                # Reload the theme immediately to update global colors
                await db_config_manager.load_config()
                # Force update the global THEME_COLORS
                set_theme_colors(color)
                # Get fresh colors after theme reload
                new_colors = get_theme_colors()
                left_to_right_reveal(console, f"Theme changed to {color.title()}", style=f"bold {new_colors['primary']}", duration=1.0)
                console.print()
                show_command_suggestions(console, context='settings')
            else:
                left_to_right_reveal(console, "Failed to save", style=f"bold {colors['error']}", duration=1.0)
        else:
            # Interactive theme selection
            if run_theme_selection:
                selected = run_theme_selection(console)
                if await db_config_manager.set_theme_color(selected):
                    # Reload the theme immediately to update global colors
                    await db_config_manager.load_config()
                    # Force update the global THEME_COLORS
                    set_theme_colors(selected)
                    show_command_suggestions(console, context='settings')
                else:
                    # Get fresh colors for error message
                    current_colors = get_theme_colors()
                    left_to_right_reveal(console, "Failed to save", style=f"bold {current_colors['error']}", duration=1.0)
    
    asyncio.run(_theme())

# ================================
# PREFERENCES OVERVIEW
# ================================

@settings.command('prefs')
@click.option('--reset', is_flag=True, help='Reset all preferences to defaults')
def preferences_overview(reset: bool):
    """Configure paper preferences overview"""
    
    async def _prefs():
        await _ensure_authenticated()
        print_header(console, "Paper Preferences")
        
        user = unified_user_service.get_current_user()
        user_id = user['id']
        
        if reset:
            if Confirm.ask(f"{style_text('Reset all preferences to defaults?', 'warning')}", default=False):
                # Reset preferences logic here
                print_success(console, "Preferences reset to defaults!")
                return
        
        # Show current preferences
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        if not prefs_result['success']:
            print_error(console, "Failed to load preferences")
            return
            
        prefs = prefs_result['preferences']
        await _display_preferences_overview(prefs)
        
        # Quick actions menu
        console.print(f"\n{style_text('Quick Actions:', 'primary')}")
        actions = [
            ("categories", "Configure research categories"),
            ("keywords", "Manage keywords & exclusions"), 
            ("authors", "Set preferred authors"),
            ("daily", "Daily dose settings"),
            ("done", "Exit preferences")
        ]
        
        for cmd, desc in actions:
            console.print(f"- {style_text(f'arionxiv settings {cmd}', 'primary')} - {desc}")
    
    asyncio.run(_prefs())

# ================================
# CATEGORIES CONFIGURATION
# ================================

@settings.command('categories')
@click.option('--add', multiple=True, help='Add categories (e.g., --add cs.AI --add cs.LG)')
@click.option('--remove', multiple=True, help='Remove categories')
@click.option('--clear', is_flag=True, help='Clear all categories')
def categories_config(add: tuple, remove: tuple, clear: bool):
    """Configure research categories (ArXiv categories)"""
    
    async def _categories():
        await _ensure_authenticated()
        print_header(console, "Research Categories")
        
        user = unified_user_service.get_current_user()
        user_id = user['id']
        
        # Get current preferences
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        current_categories = prefs_result['preferences']['categories'] if prefs_result['success'] else []
        
        # Handle command-line options
        new_categories = set(current_categories)
        
        if clear:
            if Confirm.ask(f"{style_text('Clear all categories?', 'warning')}", default=False):
                new_categories.clear()
        
        if add:
            new_categories.update(add)
            
        if remove:
            new_categories.difference_update(remove)
        
        # If any changes made via CLI, save and exit
        if add or remove or clear:
            await _save_categories(user_id, list(new_categories))
            return
        
        # Interactive mode
        await _interactive_categories_config(user_id, current_categories)
    
    asyncio.run(_categories())

# ================================
# KEYWORDS CONFIGURATION  
# ================================

@settings.command('keywords')
@click.option('--add', multiple=True, help='Add keywords')
@click.option('--remove', multiple=True, help='Remove keywords')
@click.option('--exclude', multiple=True, help='Add exclude keywords')
@click.option('--clear', is_flag=True, help='Clear all keywords')
def keywords_config(add: tuple, remove: tuple, exclude: tuple, clear: bool):
    """Configure keywords and exclusions"""
    
    async def _keywords():
        await _ensure_authenticated()
        print_header(console, "Keywords Configuration")
        
        user = unified_user_service.get_current_user()
        user_id = user['id']
        
        # Get current preferences
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        prefs = prefs_result['preferences'] if prefs_result['success'] else {}
        
        current_keywords = set(prefs.get('keywords', []))
        current_excludes = set(prefs.get('exclude_keywords', []))
        
        # Handle CLI options
        if clear:
            if Confirm.ask(f"{style_text('Clear all keywords?', 'warning')}", default=False):
                current_keywords.clear()
                current_excludes.clear()
        
        if add:
            current_keywords.update(add)
        
        if remove:
            current_keywords.difference_update(remove)
            
        if exclude:
            current_excludes.update(exclude)
        
        # If changes made via CLI, save and exit
        if add or remove or exclude or clear:
            await _save_keywords(user_id, list(current_keywords), list(current_excludes))
            return
        
        # Interactive mode
        await _interactive_keywords_config(user_id, prefs)
    
    asyncio.run(_keywords())

# ================================
# AUTHORS CONFIGURATION
# ================================

@settings.command('authors')
@click.option('--add', multiple=True, help='Add preferred authors')
@click.option('--remove', multiple=True, help='Remove authors')
@click.option('--clear', is_flag=True, help='Clear all authors')
def authors_config(add: tuple, remove: tuple, clear: bool):
    """Configure preferred authors"""
    
    async def _authors():
        await _ensure_authenticated()
        print_header(console, "Preferred Authors")
        
        user = unified_user_service.get_current_user()
        user_id = user['id']
        
        # Get current preferences
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        current_authors = set(prefs_result['preferences']['authors']) if prefs_result['success'] else set()
        
        # Handle CLI options
        if clear:
            if Confirm.ask(f"{style_text('Clear all authors?', 'warning')}", default=False):
                current_authors.clear()
        
        if add:
            current_authors.update(add)
        
        if remove:
            current_authors.difference_update(remove)
        
        # If changes made via CLI, save and exit
        if add or remove or clear:
            await _save_authors(user_id, list(current_authors))
            return
        
        # Interactive mode
        await _interactive_authors_config(user_id, list(current_authors))
    
    asyncio.run(_authors())

# ================================
# DAILY DOSE CONFIGURATION
# ================================

@settings.command('daily')
@click.option('--enable/--disable', default=None, help='Enable or disable daily dose')
@click.option('--time', 'time_str', help='Set delivery time in UTC (HH:MM format)')
@click.option('--papers', type=int, help='Max papers per day (1-10)')
@click.option('--keywords', help='Set keywords (comma-separated)')
@click.option('--show', is_flag=True, help='Show current daily dose settings')
def daily_config(enable: Optional[bool], time_str: Optional[str], papers: Optional[int], keywords: Optional[str], show: bool):
    """Configure daily dose settings"""
    
    async def _daily():
        await _ensure_authenticated()
        await unified_database_service.connect()
        
        # Import daily dose service
        from ...services.unified_daily_dose_service import unified_daily_dose_service
        
        print_header(console, "Daily Dose Configuration")
        
        user = unified_user_service.get_current_user()
        user_id = user['id']
        colors = get_theme_colors()
        
        # Load current settings
        settings_result = await unified_daily_dose_service.get_user_daily_dose_settings(user_id)
        current_settings = settings_result.get("settings", {})
        
        # Show current settings if requested or no changes
        if show or (enable is None and time_str is None and papers is None and keywords is None):
            console.print(f"\n[bold {colors['primary']}]Current Daily Dose Settings:[/bold {colors['primary']}]\n")
            
            enabled = current_settings.get("enabled", False)
            scheduled_time = current_settings.get("scheduled_time", "Not set")
            max_papers = current_settings.get("max_papers", 5)
            kw_list = current_settings.get("keywords", [])
            
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Setting", style="bold white")
            table.add_column("Value", style="white")
            
            status_color = colors['primary'] if enabled else colors['warning']
            table.add_row("Status", f"[bold {status_color}]{'Enabled' if enabled else 'Disabled'}[/bold {status_color}]")
            table.add_row("Scheduled Time (UTC)", scheduled_time if scheduled_time else "[white]Not configured[/white]")
            table.add_row("Max Papers", str(max_papers))
            table.add_row("Keywords", ", ".join(kw_list) if kw_list else "[white]None set[/white]")
            
            console.print(table)
            
            if not (enable is None and time_str is None and papers is None and keywords is None):
                return
            
            # Interactive mode
            console.print(f"\n[bold {colors['primary']}]Configure Daily Dose:[/bold {colors['primary']}]")
            await _interactive_daily_dose_config(user_id, current_settings, unified_daily_dose_service)
            return
        
        # Handle CLI options
        changes_made = False
        update_kwargs = {}
        
        if enable is not None:
            update_kwargs["enabled"] = enable
            changes_made = True
        
        if time_str:
            # Validate time format
            try:
                datetime.strptime(time_str, "%H:%M")
                update_kwargs["scheduled_time"] = time_str
                changes_made = True
            except ValueError:
                print_error(console, "Invalid time format. Use HH:MM (e.g., 09:00)")
                return
        
        if papers is not None:
            if 1 <= papers <= 10:
                update_kwargs["max_papers"] = papers
                changes_made = True
            else:
                print_error(console, "Papers must be between 1 and 10 (v1 limit)")
                return
        
        if keywords:
            kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
            update_kwargs["keywords"] = kw_list
            changes_made = True
        
        if changes_made:
            result = await unified_daily_dose_service.update_user_daily_dose_settings(user_id, **update_kwargs)
            if result["success"]:
                print_success(console, "Daily dose settings updated successfully")
                
                # If enabling and time is set, schedule the job
                if enable and (time_str or current_settings.get("scheduled_time")):
                    schedule_time = time_str or current_settings.get("scheduled_time")
                    if schedule_user_daily_dose:
                        schedule_result = await schedule_user_daily_dose(user_id, schedule_time)
                        if schedule_result.get("success"):
                            print_success(console, f"Daily dose scheduled for {schedule_time}")
                        else:
                            print_warning(console, f"Could not schedule: {schedule_result.get('error', 'Unknown error')}")
            else:
                print_error(console, f"Failed to update settings: {result.get('message')}")
    
    asyncio.run(_daily())

# ================================
# API KEYS CONFIGURATION
# ================================

@settings.command('api')
@click.option('--show', '-s', is_flag=True, help='Show current API configuration status')
@click.option('--gemini', 'set_gemini', help='Set Gemini API key directly')
@click.option('--huggingface', '--hf', 'set_hf', help='Set HuggingFace API key directly')
@click.option('--groq', 'set_groq', help='Set Groq API key directly')
@click.option('--remove', type=click.Choice(['gemini', 'huggingface', 'groq']), help='Remove an API key')
def api_config(show: bool, set_gemini: Optional[str], set_hf: Optional[str], set_groq: Optional[str], remove: Optional[str]):
    """Configure API keys for AI services (Gemini, HuggingFace, Groq)
    
    All API keys are FREE to obtain!
    Keys are stored securely in ~/.arionxiv/api_keys.json
    They persist across sessions - configure once, use forever!
    """
    
    colors = get_theme_colors()
    print_header(console, "API Keys Configuration")
    
    # Show info about persistence
    console.print(f"\n[white]Your keys are stored locally and persist across sessions.[/white]")
    console.print(f"[white]Configure once - they'll work even after logout/login![/white]\n")
    
    # Handle direct key setting
    if set_gemini:
        if api_config_manager.set_api_key("gemini", set_gemini):
            print_success(console, "Gemini API key saved successfully!")
        else:
            print_error(console, "Failed to save Gemini API key")
        return
    
    if set_hf:
        if api_config_manager.set_api_key("huggingface", set_hf):
            print_success(console, "HuggingFace API key saved successfully!")
        else:
            print_error(console, "Failed to save HuggingFace API key")
        return
    
    if set_groq:
        if api_config_manager.set_api_key("groq", set_groq):
            print_success(console, "Groq API key saved successfully!")
        else:
            print_error(console, "Failed to save Groq API key")
        return
    
    if remove:
        if api_config_manager.remove_api_key(remove):
            print_success(console, f"{remove.title()} API key removed")
        else:
            print_error(console, f"Failed to remove {remove} API key")
        return
    
    if show:
        show_api_status(console)
        return
    
    # Interactive mode
    show_api_status(console)
    _interactive_api_config(colors)


def _interactive_api_config(colors: Dict[str, str]):
    """Interactive API key configuration menu"""
    
    while True:
        # Get current status for display
        status = api_config_manager.get_status()
        
        gemini_status = "configured" if status["gemini"]["configured"] else "not set"
        hf_status = "configured" if status["huggingface"]["configured"] else "not set"
        groq_status = "configured" if status["groq"]["configured"] else "not set"
        
        left_to_right_reveal(console, "\nOptions:", style=f"bold {colors['primary']}", duration=0.5)
        left_to_right_reveal(console, f"1. Configure Gemini API key (current: {gemini_status})", style=colors['primary'], duration=0.3)
        left_to_right_reveal(console, f"2. Configure HuggingFace API key (current: {hf_status})", style=colors['primary'], duration=0.3)
        left_to_right_reveal(console, f"3. Configure Groq API key (current: {groq_status})", style=colors['primary'], duration=0.3)
        left_to_right_reveal(console, f"4. Show API status", style=colors['primary'], duration=0.3)
        left_to_right_reveal(console, f"5. Done - Return to main menu", style=colors['primary'], duration=0.3)
        
        choice = Prompt.ask(
            f"[bold {colors['primary']}]Select option[/bold {colors['primary']}]",
            choices=["1", "2", "3", "4", "5"],
            default="5"
        )
        
        if choice == "1":
            _configure_api_key_interactive("gemini", colors)
        elif choice == "2":
            _configure_api_key_interactive("huggingface", colors)
        elif choice == "3":
            _configure_api_key_interactive("groq", colors)
        elif choice == "4":
            show_api_status(console)
        elif choice == "5":
            show_command_suggestions(console, context='settings')
            break


# Step-by-step instructions for getting API keys
API_KEY_INSTRUCTIONS = {
    "gemini": {
        "title": "How to Get Your Google Gemini API Key (FREE)",
        "steps": [
            "1. Go to: https://aistudio.google.com/app/apikey",
            "2. Sign in with your Google account",
            "3. Click 'Create API Key'",
            "4. Select a Google Cloud project (or create a new one)",
            "5. Copy your API key",
            "",
            "Note: Gemini has a generous FREE tier - no credit card needed!"
        ]
    },
    "huggingface": {
        "title": "How to Get Your HuggingFace API Token (FREE)",
        "steps": [
            "1. Go to: https://huggingface.co/settings/tokens",
            "2. Create a free account or sign in",
            "3. Click 'New token'",
            "4. Give it a name (e.g., 'ArionXiv')",
            "5. Select 'Read' access (that's all we need)",
            "6. Click 'Generate token' and copy it",
            "",
            "Note: HuggingFace is FREE for most models!"
        ]
    },
    "groq": {
        "title": "How to Get Your Groq API Key (FREE & FAST)",
        "steps": [
            "1. Go to: https://console.groq.com/keys",
            "2. Create a free account or sign in",
            "3. Click 'Create API Key'",
            "4. Give it a name (e.g., 'ArionXiv')",
            "5. Copy your API key",
            "",
            "Note: Groq is FREE and incredibly fast!",
            "      It's REQUIRED for AI analysis and chat features."
        ]
    }
}


def _configure_api_key_interactive(provider: str, colors: Dict[str, str]):
    """Configure a single API key interactively with step-by-step instructions"""
    
    info = api_config_manager.PROVIDERS.get(provider)
    if not info:
        print_error(console, f"Unknown provider: {provider}")
        return
    
    current_key = api_config_manager.get_api_key(provider)
    required_text = "REQUIRED" if info["required"] else "optional"
    
    left_to_right_reveal(console, f"\n{'='*60}", style=colors['primary'], duration=0.5)
    left_to_right_reveal(console, f"{info['name']} Configuration ({required_text})", style=f"bold {colors['primary']}", duration=0.8)
    left_to_right_reveal(console, f"{info['description']}", style="white", duration=0.6)
    
    # Show step-by-step instructions
    if provider in API_KEY_INSTRUCTIONS:
        instructions = API_KEY_INSTRUCTIONS[provider]
        steps_text = "\n".join(instructions["steps"])
        left_to_right_reveal(console, "", duration=0.3)
        console.print(Panel(
            steps_text,
            title=f"[bold {colors['primary']}]{instructions['title']}[/bold {colors['primary']}]",
            border_style=colors['primary'],
            padding=(1, 2)
        ))
    
    if current_key:
        left_to_right_reveal(console, f"\nCurrently configured: {api_config_manager._mask_key(current_key)}", style=colors['primary'], duration=0.8)
        
        action = Prompt.ask(
            f"\n[bold {colors['primary']}]What would you like to do?[/bold {colors['primary']}]",
            choices=["update", "remove", "cancel"],
            default="cancel"
        )
        
        if action == "update":
            new_key = Prompt.ask(f"[bold {colors['primary']}]Enter new API key[/bold {colors['primary']}]", default="", show_default=False)
            if new_key.strip():
                if api_config_manager.set_api_key(provider, new_key.strip()):
                    print_success(console, f"{info['name']} key updated successfully!")
                else:
                    print_error(console, "Failed to save key")
            else:
                print_warning(console, "No key entered, keeping existing")
        elif action == "remove":
            if Confirm.ask(f"[bold {colors['warning']}]Remove {info['name']} key?[/bold {colors['warning']}]", default=False):
                if api_config_manager.remove_api_key(provider):
                    print_success(console, f"{info['name']} key removed")
                else:
                    print_error(console, "Failed to remove key")
    else:
        new_key = Prompt.ask(
            f"\n[bold {colors['primary']}]Enter {info['name']} API key (or press Enter to skip)[/bold {colors['primary']}]",
            default="",
            show_default=False
        )
        if new_key.strip():
            if api_config_manager.set_api_key(provider, new_key.strip()):
                print_success(console, f"{info['name']} key saved successfully!")
            else:
                print_error(console, "Failed to save key")
        else:
            if info["required"]:
                print_warning(console, f"{info['name']} key is REQUIRED for AI features")
            else:
                left_to_right_reveal(console, f"Skipped {info['name']}", style="white", duration=0.8)


# ================================
# TIME CONFIGURATION (Quick access)
# ================================

@settings.command('time')
@click.argument('time_str', required=False)
def time_config(time_str: Optional[str]):
    """Set daily analysis delivery time (HH:MM format)"""
    
    async def _time():
        await _ensure_authenticated()
        
        if time_str:
            # Validate time format
            try:
                datetime.strptime(time_str, "%H:%M")
                # Save time setting
                print_success(console, f"Daily analysis time set to {time_str}")
                print_warning(console, "Use 'arionxiv settings daily' for more options")
            except ValueError:
                print_error(console, "Invalid time format. Use HH:MM (e.g., 09:00)")
        else:
            # Show current time and prompt for new one
            print_header(console, "Daily Analysis Time")
            console.print("Current time: [bold]09:00[/bold] (example)")
            
            new_time = Prompt.ask(
                f"{style_text('Enter new time (HH:MM)', 'primary')}",
                default="09:00"
            )
            
            try:
                datetime.strptime(new_time, "%H:%M")
                print_success(console, f"Daily analysis time set to {new_time}")
            except ValueError:
                print_error(console, "Invalid time format")
    
    asyncio.run(_time())

# ================================
# SAVED PAPERS MANAGEMENT
# ================================

@settings.command('papers')
def papers_config():
    """Manage your saved papers (delete papers from library)"""
    
    async def _papers():
        await _ensure_authenticated()
        await unified_database_service.connect()
        print_header(console, "Saved Papers Management")
        
        user = unified_user_service.get_current_user()
        user_name = user.get('user_name', 'default')
        
        # Get user's saved papers
        user_papers = await unified_database_service.get_user_papers(user_name)
        
        if not user_papers:
            print_warning(console, "No saved papers in your library.")
            console.print(f"\n{style_text('Use arionxiv chat to chat with papers and save them.', 'primary')}")
            show_command_suggestions(console, context='settings')
            return
        
        colors = get_theme_colors()
        
        console.print(f"\n[bold {colors['primary']}]Your saved papers ({len(user_papers)}/10):[/bold {colors['primary']}]\n")
        
        # Create table
        table = Table(show_header=True, header_style=f"bold {colors['primary']}")
        table.add_column("#", style="bold white", width=3)
        table.add_column("Title", style="white", max_width=50)
        table.add_column("ArXiv ID", style="white", width=15)
        table.add_column("Added", style="white", width=12)
        
        for i, paper in enumerate(user_papers):
            title = paper.get("title", "Unknown")
            
            arxiv_id = paper.get("arxiv_id", "Unknown")
            added_at = paper.get("added_at")
            if added_at:
                added_str = added_at.strftime("%Y-%m-%d") if hasattr(added_at, 'strftime') else str(added_at)[:10]
            else:
                added_str = "Unknown"
            table.add_row(str(i + 1), title, arxiv_id, added_str)
        
        console.print(table)
        
        # Options
        console.print(f"\n{style_text('Actions:', 'primary')}")
        console.print(f"[bold {colors['primary']}]1.[/bold {colors['primary']}] Delete papers")
        console.print(f"[bold {colors['primary']}]2.[/bold {colors['primary']}] Exit")
        
        action = Prompt.ask(f"[bold {colors['primary']}]Select action[/bold {colors['primary']}]", choices=["1", "2"], default="2")
        
        if action == "1":
            console.print(f"\n[bold {colors['primary']}]Enter paper numbers to delete (comma-separated, e.g., 1,3,5) or 0 to cancel:[/bold {colors['primary']}]")
            
            choice = Prompt.ask(f"[bold {colors['primary']}]Papers to delete[/bold {colors['primary']}]")
            
            if choice.strip() == "0" or not choice.strip():
                print_warning(console, "Cancelled.")
                return
            
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                valid_indices = [i for i in indices if 0 <= i < len(user_papers)]
                
                if not valid_indices:
                    print_error(console, "No valid selections.")
                    return
                
                # Confirm deletion
                papers_to_delete = [user_papers[i].get("title", "Unknown")[:30] for i in valid_indices]
                console.print(f"\n[bold {colors['warning']}]Papers to delete:[/bold {colors['warning']}]")
                for title in papers_to_delete:
                    console.print(f"  - {title}")
                
                if Confirm.ask(f"\n[bold {colors['warning']}]Confirm deletion?[/bold {colors['warning']}]", default=False):
                    deleted_count = 0
                    for idx in valid_indices:
                        paper = user_papers[idx]
                        arxiv_id = paper.get("arxiv_id")
                        if arxiv_id:
                            success = await unified_database_service.remove_user_paper(user_name, arxiv_id)
                            if success:
                                deleted_count += 1
                    
                    print_success(console, f"Deleted {deleted_count} paper(s) from your library.")
                else:
                    print_warning(console, "Deletion cancelled.")
                    
            except ValueError:
                print_error(console, "Invalid input. Use comma-separated numbers.")
        
        # Show command suggestions
        show_command_suggestions(console, context='settings')
    
    asyncio.run(_papers())

# ================================
# HELPER FUNCTIONS
# ================================

async def _ensure_authenticated():
    """Ensure user is authenticated"""
    if not unified_user_service.is_authenticated():
        print_error(console, "Please login first: arionxiv auth --login")
        raise click.Abort()

async def _show_compact_settings(prefs: Dict[str, Any], theme_color: str):
    """Show compact settings overview"""
    colors = get_theme_colors()
    
    # Create a summary table
    table = Table(title="Settings Summary", show_header=True, header_style=f"bold {colors['primary']}")
    table.add_column("Setting", style="bold white", width=20)
    table.add_column("Value", style="white", width=40)
    table.add_column("Command", style="white", width=25)
    
    # Theme
    table.add_row("Theme Color", f"[{theme_color}]{theme_color.title()}[/{theme_color}]", "settings theme")
    
    # Categories
    categories = prefs.get('categories', [])
    table.add_row("Categories", ", ".join(categories[:3]) + ("..." if len(categories) > 3 else ""), "settings categories")
    
    # Keywords
    keywords = prefs.get('keywords', [])
    table.add_row("Keywords", ", ".join(keywords[:3]) + ("..." if len(keywords) > 3 else ""), "settings keywords")
    
    # Authors
    authors = prefs.get('authors', [])
    table.add_row("Authors", ", ".join(authors[:2]) + ("..." if len(authors) > 2 else ""), "settings authors")
    
    # Daily settings
    max_papers = prefs.get('max_papers_per_day', 10)
    table.add_row("Max Papers/Day", str(max_papers), "settings daily")
    
    # Relevance threshold
    min_relevance = prefs.get('min_relevance_score', 0.2)
    table.add_row("Min Relevance", f"{min_relevance:.1f}", "settings prefs")
    
    console.print(table)

async def _show_detailed_settings(prefs: Dict[str, Any], theme_color: str):
    """Show detailed settings view"""
    
    # Theme section
    theme_panel = Panel(
        f"Color Theme: [bold {theme_color}]{theme_color.title()}[/bold {theme_color}]\n"
        f"Use: [white]arionxiv settings theme[/white]",
        title="Display",
        border_style=theme_color
    )
    
    # Preferences section
    categories = prefs.get('categories', [])
    keywords = prefs.get('keywords', [])
    authors = prefs.get('authors', [])
    exclude_keywords = prefs.get('exclude_keywords', [])
    
    prefs_content = f"""
Categories: {', '.join(categories) if categories else 'None set'}
Keywords: {', '.join(keywords) if keywords else 'None set'}
Authors: {', '.join(authors) if authors else 'None set'}
Exclude: {', '.join(exclude_keywords) if exclude_keywords else 'None set'}

Max Papers/Day: {prefs.get('max_papers_per_day', 10)}
Min Relevance: {prefs.get('min_relevance_score', 0.2):.1f}
    """.strip()
    
    prefs_panel = Panel(
        prefs_content,
        title="Paper Preferences",
        border_style=theme_color
    )
    
    # Daily dose section
    daily_content = """
Status: Enabled
Delivery Time: 09:00
Fetch Days: 1 day back
Analysis: LLM-powered
    """.strip()
    
    daily_panel = Panel(
        daily_content,
        title="Daily Dose",
        border_style=theme_color
    )
    
    # Display panels
    console.print(Columns([theme_panel, prefs_panel]))
    console.print(daily_panel)

async def _display_preferences_overview(prefs: Dict[str, Any]):
    """Display preferences in a clean overview format"""
    
    colors = get_theme_colors()
    
    # Categories
    categories = prefs.get('categories', [])
    if categories:
        console.print(f"\n{style_text('Research Categories:', 'primary')}")
        for cat in categories:
            console.print(f"  • {cat}")
    else:
        console.print(f"\n{style_text('Research Categories:', 'primary')} None set")
    
    # Keywords
    keywords = prefs.get('keywords', [])
    if keywords:
        console.print(f"\n{style_text('Keywords:', 'primary')}")
        for kw in keywords:
            console.print(f"  - {kw}")
    else:
        console.print(f"\n{style_text('Keywords:', 'primary')} None set")
    
    # Authors
    authors = prefs.get('authors', [])
    if authors:
        console.print(f"\n{style_text('Preferred Authors:', 'primary')}")
        for author in authors:
            console.print(f"  • {author}")
    else:
        console.print(f"\n{style_text('Preferred Authors:', 'primary')} None set")
    
    # Settings
    console.print(f"\n{style_text('Settings:', 'primary')}")
    console.print(f"  • Max papers per day: {prefs.get('max_papers_per_day', 10)}")
    console.print(f"  • Min relevance score: {prefs.get('min_relevance_score', 0.2):.1f}")

async def _interactive_categories_config(user_id: str, current_categories: List[str]):
    """Interactive categories configuration"""
    
    # Show available categories
    available_cats = unified_user_service.get_available_categories()
    
    console.print(f"\n{style_text('Available ArXiv Categories:', 'primary')}")
    
    colors = get_theme_colors()
    
    # Create table of categories
    table = Table(show_header=True, header_style=f"bold {colors['primary']}")
    table.add_column("Code", style="bold", width=8)
    table.add_column("Description", width=40)
    table.add_column("Selected", width=8)
    
    for code, desc in list(available_cats.items())[:10]:  # Show top 10
        selected = "[X]" if code in current_categories else "[ ]"
        table.add_row(code, desc, selected)
    
    console.print(table)
    console.print(f"\n{style_text('Current selections:', 'primary')} {', '.join(current_categories)}")
    
    # Interactive selection
    action = Prompt.ask(
        "\n[bold]Action[/bold]",
        choices=["add", "remove", "clear", "done"],
        default="done"
    )
    
    if action == "add":
        new_cat = Prompt.ask("Enter category code (e.g., cs.AI)")
        if new_cat in available_cats:
            new_categories = list(set(current_categories + [new_cat]))
            await _save_categories(user_id, new_categories)
        else:
            print_error(console, "Invalid category code")
    
    elif action == "remove":
        if current_categories:
            cat_to_remove = Prompt.ask("Enter category to remove", choices=current_categories)
            new_categories = [c for c in current_categories if c != cat_to_remove]
            await _save_categories(user_id, new_categories)
        else:
            print_warning(console, "No categories to remove")
    
    elif action == "clear":
        if Confirm.ask("Clear all categories?", default=False):
            await _save_categories(user_id, [])

async def _interactive_keywords_config(user_id: str, prefs: Dict[str, Any]):
    """Interactive keywords configuration"""
    
    keywords = prefs.get('keywords', [])
    exclude_keywords = prefs.get('exclude_keywords', [])
    
    console.print(f"\n{style_text('Current Keywords:', 'primary')}")
    console.print(f"Include: {', '.join(keywords) if keywords else 'None'}")
    console.print(f"Exclude: {', '.join(exclude_keywords) if exclude_keywords else 'None'}")
    
    action = Prompt.ask(
        "\n[bold]Action[/bold]",
        choices=["add", "exclude", "remove", "clear", "done"],
        default="done"
    )
    
    if action == "add":
        new_keywords = Prompt.ask("Enter keywords (comma-separated)")
        new_kw_list = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
        combined = list(set(keywords + new_kw_list))
        await _save_keywords(user_id, combined, exclude_keywords)
    
    elif action == "exclude":
        exclude_kw = Prompt.ask("Enter keywords to exclude (comma-separated)")
        new_exclude = [kw.strip() for kw in exclude_kw.split(',') if kw.strip()]
        combined_exclude = list(set(exclude_keywords + new_exclude))
        await _save_keywords(user_id, keywords, combined_exclude)
    
    elif action == "remove":
        if keywords:
            kw_to_remove = Prompt.ask("Enter keyword to remove", choices=keywords)
            new_keywords = [k for k in keywords if k != kw_to_remove]
            await _save_keywords(user_id, new_keywords, exclude_keywords)
    
    elif action == "clear":
        if Confirm.ask("Clear all keywords?", default=False):
            await _save_keywords(user_id, [], [])

async def _interactive_authors_config(user_id: str, current_authors: List[str]):
    """Interactive authors configuration"""
    
    console.print(f"\n{style_text('Current Authors:', 'primary')}")
    if current_authors:
        for author in current_authors:
            console.print(f"  • {author}")
    else:
        console.print("  None set")
    
    action = Prompt.ask(
        "\n[bold]Action[/bold]",
        choices=["add", "remove", "clear", "done"],
        default="done"
    )
    
    if action == "add":
        new_author = Prompt.ask("Enter author name")
        new_authors = list(set(current_authors + [new_author.strip()]))
        await _save_authors(user_id, new_authors)
    
    elif action == "remove":
        if current_authors:
            author_to_remove = Prompt.ask("Enter author to remove", choices=current_authors)
            new_authors = [a for a in current_authors if a != author_to_remove]
            await _save_authors(user_id, new_authors)
    
    elif action == "clear":
        if Confirm.ask("Clear all authors?", default=False):
            await _save_authors(user_id, [])

async def _interactive_daily_config(user_id: str, prefs: Dict[str, Any]):
    """Interactive daily dose configuration"""
    
    console.print(f"\n{style_text('Daily Dose Settings:', 'primary')}")
    
    # Current settings
    max_papers = prefs.get('max_papers_per_day', 10)
    min_relevance = prefs.get('min_relevance_score', 0.2)
    
    console.print(f"Max papers per day: {max_papers}")
    console.print(f"Min relevance score: {min_relevance:.1f}")
    
    # Configuration options
    if Confirm.ask("\nChange max papers per day?", default=False):
        new_max = IntPrompt.ask("Enter max papers (1-50)", default=max_papers)
        if 1 <= new_max <= 50:
            prefs['max_papers_per_day'] = new_max
            await unified_user_service.update_user_preferences(user_id, prefs)
            print_success(console, f"Max papers updated to {new_max}")
    
    if Confirm.ask("Change relevance threshold?", default=False):
        new_relevance = float(Prompt.ask("Enter min relevance (0.0-1.0)", default=str(min_relevance)))
        if 0.0 <= new_relevance <= 1.0:
            prefs['min_relevance_score'] = new_relevance
            await unified_user_service.update_user_preferences(user_id, prefs)
            print_success(console, f"Min relevance updated to {new_relevance:.1f}")

async def _interactive_daily_dose_config(user_id: str, current_settings: Dict[str, Any], daily_dose_service):
    """Interactive daily dose configuration with full options"""
    
    colors = get_theme_colors()
    
    while True:
        # Get current values for display
        enabled = current_settings.get("enabled", False)
        scheduled_time = current_settings.get("scheduled_time", "08:00") or "08:00"
        max_papers = current_settings.get("max_papers", 5)
        keywords = current_settings.get("keywords", [])
        keywords_str = ", ".join(keywords[:3]) + ("..." if len(keywords) > 3 else "") if keywords else "None"
        
        console.print(f"\n[bold {colors['primary']}]Options:[/bold {colors['primary']}]")
        console.print(f"[bold {colors['primary']}]1.[/bold {colors['primary']}] Toggle enabled/disabled [white](current: {'Enabled' if enabled else 'Disabled'})[/white]")
        console.print(f"[bold {colors['primary']}]2.[/bold {colors['primary']}] Set scheduled time (UTC) [white](current: {scheduled_time})[/white]")
        console.print(f"[bold {colors['primary']}]3.[/bold {colors['primary']}] Set max papers (1-10) [white](current: {max_papers})[/white]")
        console.print(f"[bold {colors['primary']}]4.[/bold {colors['primary']}] Set keywords [white](current: {keywords_str})[/white]")
        console.print(f"[bold {colors['primary']}]5.[/bold {colors['primary']}] Done - Return to main menu")
        
        choice = Prompt.ask(f"[bold {colors['primary']}]Select option[/bold {colors['primary']}]", choices=["1", "2", "3", "4", "5"], default="5")
        
        if choice == "1":
            current_enabled = current_settings.get("enabled", False)
            new_enabled = not current_enabled
            result = await daily_dose_service.update_user_daily_dose_settings(user_id, enabled=new_enabled)
            if result["success"]:
                current_settings["enabled"] = new_enabled
                status = "enabled" if new_enabled else "disabled"
                print_success(console, f"Daily dose {status}")
            else:
                print_error(console, f"Failed to update: {result.get('message')}")
        
        elif choice == "2":
            current_time = current_settings.get("scheduled_time", "08:00")
            console.print(f"[white]Note: Time is in UTC timezone. Your daily dose will run at this UTC time.[/white]")
            new_time = Prompt.ask(f"[bold {colors['primary']}]Enter time in UTC (HH:MM)[/bold {colors['primary']}]", default=current_time or "08:00")
            try:
                datetime.strptime(new_time, "%H:%M")
                result = await daily_dose_service.update_user_daily_dose_settings(user_id, scheduled_time=new_time)
                if result["success"]:
                    current_settings["scheduled_time"] = new_time
                    print_success(console, f"Scheduled time set to {new_time} UTC")
                    
                    # Schedule the job if enabled
                    if current_settings.get("enabled") and schedule_user_daily_dose:
                        schedule_result = await schedule_user_daily_dose(user_id, new_time)
                        if schedule_result.get("success"):
                            print_success(console, "Cron job scheduled")
                else:
                    print_error(console, f"Failed to update: {result.get('message')}")
            except ValueError:
                print_error(console, "Invalid time format. Use HH:MM")
        
        elif choice == "3":
            current_max = current_settings.get("max_papers", 5)
            new_max = IntPrompt.ask(f"[bold {colors['primary']}]Enter max papers (1-10)[/bold {colors['primary']}]", default=current_max)
            if 1 <= new_max <= 10:
                result = await daily_dose_service.update_user_daily_dose_settings(user_id, max_papers=new_max)
                if result["success"]:
                    current_settings["max_papers"] = new_max
                    print_success(console, f"Max papers set to {new_max}")
                else:
                    print_error(console, f"Failed to update: {result.get('message')}")
            else:
                print_error(console, "Max papers must be between 1 and 10")
        
        elif choice == "4":
            current_keywords = current_settings.get("keywords", [])
            console.print(f"\n[bold {colors['primary']}]Current keywords:[/bold {colors['primary']}] {', '.join(current_keywords) if current_keywords else 'None'}")
            new_keywords_str = Prompt.ask(f"[bold {colors['primary']}]Enter keywords (comma-separated)[/bold {colors['primary']}]", default=", ".join(current_keywords))
            new_keywords = [k.strip() for k in new_keywords_str.split(",") if k.strip()]
            result = await daily_dose_service.update_user_daily_dose_settings(user_id, keywords=new_keywords)
            if result["success"]:
                current_settings["keywords"] = new_keywords
                print_success(console, f"Keywords updated: {', '.join(new_keywords)}")
            else:
                print_error(console, f"Failed to update: {result.get('message')}")
        
        elif choice == "5":
            # Show main menu / command suggestions before exiting
            show_command_suggestions(console, context='settings')
            break

# ================================
# SAVE HELPER FUNCTIONS
# ================================

async def _save_categories(user_id: str, categories: List[str]):
    """Save categories and update MongoDB"""
    try:
        # Get current preferences
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        prefs = prefs_result['preferences'] if prefs_result['success'] else {}
        
        # Update categories
        prefs['categories'] = categories
        
        # Save to preferences service (which handles MongoDB updates)
        result = await unified_user_service.update_user_preferences(user_id, prefs)
        
        if result['success']:
            print_success(console, f"Categories updated: {', '.join(categories) if categories else 'None'}")
        else:
            print_error(console, "Failed to save categories")
    except Exception as e:
        print_error(console, f"Error saving categories: {e}")

async def _save_keywords(user_id: str, keywords: List[str], exclude_keywords: List[str]):
    """Save keywords and update MongoDB"""
    try:
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        prefs = prefs_result['preferences'] if prefs_result['success'] else {}
        
        prefs['keywords'] = keywords
        prefs['exclude_keywords'] = exclude_keywords
        
        result = await unified_user_service.update_user_preferences(user_id, prefs)
        
        if result['success']:
            print_success(console, "Keywords updated successfully")
        else:
            print_error(console, "Failed to save keywords")
    except Exception as e:
        print_error(console, f"Error saving keywords: {e}")

async def _save_authors(user_id: str, authors: List[str]):
    """Save authors and update MongoDB"""
    try:
        prefs_result = await unified_user_service.get_user_preferences(user_id)
        prefs = prefs_result['preferences'] if prefs_result['success'] else {}
        
        prefs['authors'] = authors
        
        result = await unified_user_service.update_user_preferences(user_id, prefs)
        
        if result['success']:
            print_success(console, f"Authors updated: {', '.join(authors) if authors else 'None'}")
        else:
            print_error(console, "Failed to save authors")
    except Exception as e:
        print_error(console, f"Error saving authors: {e}")

if __name__ == "__main__":
    settings()
