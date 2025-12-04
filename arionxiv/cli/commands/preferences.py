"""Preferences command for ArionXiv CLI - Paper preferences management"""

import sys
import asyncio
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from ..utils.db_config_manager import db_config_manager
from ..ui.theme import create_themed_console, print_header, style_text, print_success, print_warning, print_error, get_theme_colors
from ...services.unified_user_service import unified_user_service

console = create_themed_console()

@click.command()
@click.option('--categories', '-c', is_flag=True, help='Configure preferred categories')
@click.option('--keywords', '-k', is_flag=True, help='Configure keywords')
@click.option('--authors', '-a', is_flag=True, help='Configure preferred authors')
@click.option('--schedule', '-t', is_flag=True, help='Configure daily dose schedule')
@click.option('--show', '-s', is_flag=True, help='Show current preferences')
@click.option('--reset', '-r', is_flag=True, help='Reset to default preferences')
def preferences_command(categories: bool, keywords: bool, authors: bool, schedule: bool, show: bool, reset: bool):
    """
    Configure your paper preferences for daily dose
    
    Examples:
    \b
        arionxiv preferences --show
        arionxiv preferences --categories
        arionxiv preferences --keywords
        arionxiv preferences --authors
        arionxiv preferences --schedule
        arionxiv preferences --reset
    """
    
    async def _handle_preferences():
        # Lazy import to avoid circular dependencies
        from ...services.unified_config_service import unified_config_service
        
        print_header(console, "Paper Preferences")
        
        # Check authentication
        if not unified_user_service.is_authenticated():
            print_error(console, "ERROR: You must be logged in to manage preferences")
            return
        
        user = unified_user_service.get_current_user()
        user_id = user["id"]
        
        if show:
            await _show_preferences(user_id)
        elif categories:
            await _configure_categories(user_id)
        elif keywords:
            await _configure_keywords(user_id)
        elif authors:
            await _configure_authors(user_id)
        elif schedule:
            await _configure_schedule(user_id)
        elif reset:
            await _reset_preferences(user_id)
        else:
            await _show_preferences_menu(user_id)
    
    # Run async function
    asyncio.run(_handle_preferences())

async def _show_preferences(user_id: str):
    """Show current user preferences"""
    # Import here to avoid circular dependencies
    from ...services.unified_config_service import unified_config_service
    
    console.print(f"\n{style_text('Current Paper Preferences', 'primary')}")
    console.print("─" * 50)
    
    result = await unified_config_service.get_user_preferences(user_id)
    
    if not result["success"]:
        print_error(console, f"Failed to load preferences: {result['message']}")
        return
    
    prefs = result["preferences"]
    
    # Categories
    console.print(f"\n{style_text('Preferred Categories:', 'accent')}")
    if prefs["categories"]:
        available_cats = unified_config_service.get_available_categories()
        for cat in prefs["categories"]:
            cat_name = available_cats.get(cat, cat)
            console.print(f"  • {cat} - {cat_name}")
    else:
        console.print("  No categories selected")
    
    # Keywords
    console.print(f"\n{style_text('Keywords:', 'accent')}")
    if prefs["keywords"]:
        for keyword in prefs["keywords"]:
            console.print(f"  • {keyword}")
    else:
        console.print("  No keywords set")
    
    # Authors
    console.print(f"\n{style_text('Preferred Authors:', 'accent')}")
    if prefs["authors"]:
        for author in prefs["authors"]:
            console.print(f"  • {author}")
    else:
        console.print("  No preferred authors")
    
    # Exclude keywords
    console.print(f"\n{style_text('Exclude Keywords:', 'accent')}")
    if prefs["exclude_keywords"]:
        for keyword in prefs["exclude_keywords"]:
            console.print(f"  • {keyword}")
    else:
        console.print("  No exclusions")
    
    # Settings
    console.print(f"\n{style_text('Settings:', 'accent')}")
    console.print(f"  • Minimum relevance score: {prefs['min_relevance_score']}")
    console.print(f"  • Maximum papers per day: {prefs['max_papers_per_day']}")
    
    # Daily dose schedule
    console.print(f"\n{style_text('Daily Dose Schedule:', 'accent')}")
    console.print(f"  • Enabled: {'Yes' if prefs.get('daily_dose_enabled', False) else 'No'}")
    console.print(f"  • Time: {prefs.get('daily_dose_time', '08:00')} UTC")

async def _configure_categories(user_id: str):
    """Configure preferred categories"""
    # Import here to avoid circular dependencies
    from ...services.unified_config_service import unified_config_service
    
    console.print(f"\n{style_text('Configure Preferred Categories', 'primary')}")
    console.print("─" * 50)
    
    # Get current preferences
    result = await unified_config_service.get_user_preferences(user_id)
    if not result["success"]:
        print_error(console, f"Failed to load preferences: {result['message']}")
        return
    
    current_categories = result["preferences"]["categories"]
    available_categories = unified_config_service.get_available_categories()
    colors = get_theme_colors()
    
    # Show available categories in a table
    console.print(f"\n{style_text('Available Categories:', 'info')}")
    
    table = Table(show_header=True, header_style=f"bold {colors['primary']}")
    table.add_column("ID", style="bold white", width=15)
    table.add_column("Category", style="white")
    table.add_column("Selected", style="white", width=10)
    
    for cat_id, cat_name in available_categories.items():
        selected = "[X]" if cat_id in current_categories else "[ ]"
        table.add_row(cat_id, cat_name, selected)
    
    console.print(table)
    
    # Get user selections
    console.print(f"\n{style_text('Current selections:', 'accent')} {', '.join(current_categories) if current_categories else 'None'}")
    
    console.print(f"\n{style_text('Instructions:', 'info')}")
    console.print("• Enter category IDs separated by commas (e.g., cs.AI, cs.LG, cs.CV)")
    console.print("• Leave empty to clear all selections")
    console.print("• Type 'cancel' to abort")
    
    while True:
        selection = Prompt.ask(
            "\n[bold]Enter category IDs[/bold]",
            default=", ".join(current_categories)
        )
        
        if selection.lower() == "cancel":
            print_warning(console, "Category configuration cancelled")
            return
        
        if not selection.strip():
            new_categories = []
            break
        
        # Parse and validate categories
        entered_cats = [cat.strip() for cat in selection.split(",")]
        invalid_cats = [cat for cat in entered_cats if cat not in available_categories]
        
        if invalid_cats:
            print_error(console, f"Invalid categories: {', '.join(invalid_cats)}")
            console.print("Please check the category IDs from the table above")
            continue
        
        new_categories = entered_cats
        break
    
    # Update preferences
    current_prefs = result["preferences"]
    current_prefs["categories"] = new_categories
    
    save_result = await unified_config_service.save_user_preferences(user_id, current_prefs)
    
    if save_result["success"]:
        print_success(console, "Categories updated successfully!")
        if new_categories:
            console.print(f"Selected: {', '.join(new_categories)}")
        else:
            console.print("All categories cleared")
    else:
        print_error(console, f"Failed to save: {save_result['message']}")

async def _configure_keywords(user_id: str):
    """Configure keywords"""
    # Import here to avoid circular dependencies
    from ...services.unified_config_service import unified_config_service
    
    console.print(f"\n{style_text('Configure Keywords', 'primary')}")
    console.print("─" * 50)
    
    # Get current preferences
    result = await unified_config_service.get_user_preferences(user_id)
    if not result["success"]:
        print_error(console, f"Failed to load preferences: {result['message']}")
        return
    
    current_prefs = result["preferences"]
    
    console.print(f"\n{style_text('Current keywords:', 'accent')} {', '.join(current_prefs['keywords']) if current_prefs['keywords'] else 'None'}")
    console.print(f"{style_text('Current exclusions:', 'accent')} {', '.join(current_prefs['exclude_keywords']) if current_prefs['exclude_keywords'] else 'None'}")
    
    console.print(f"\n{style_text('Instructions:', 'info')}")
    console.print("• Keywords help find relevant papers in titles and abstracts")
    console.print("• Enter keywords separated by commas")
    console.print("• Use specific terms like 'neural networks', 'deep learning', 'transformer'")
    
    # Configure include keywords
    include_keywords = Prompt.ask(
        "\n[bold]Include keywords[/bold]",
        default=", ".join(current_prefs["keywords"])
    )
    
    # Configure exclude keywords
    exclude_keywords = Prompt.ask(
        "\n[bold]Exclude keywords (papers with these will be filtered out)[/bold]",
        default=", ".join(current_prefs["exclude_keywords"])
    )
    
    # Process keywords
    new_include = [kw.strip() for kw in include_keywords.split(",") if kw.strip()] if include_keywords else []
    new_exclude = [kw.strip() for kw in exclude_keywords.split(",") if kw.strip()] if exclude_keywords else []
    
    # Update preferences
    current_prefs["keywords"] = new_include
    current_prefs["exclude_keywords"] = new_exclude
    
    save_result = await unified_config_service.save_user_preferences(user_id, current_prefs)
    
    if save_result["success"]:
        print_success(console, "Keywords updated successfully!")
        if new_include:
            console.print(f"Include: {', '.join(new_include)}")
        if new_exclude:
            console.print(f"Exclude: {', '.join(new_exclude)}")
    else:
        print_error(console, f"Failed to save: {save_result['message']}")

async def _configure_authors(user_id: str):
    """Configure preferred authors"""
    from ...services.unified_config_service import unified_config_service
    
    console.print(f"\n{style_text('Configure Preferred Authors', 'primary')}")
    console.print("─" * 50)
    
    result = await unified_config_service.get_user_preferences(user_id)
    if not result["success"]:
        print_error(console, f"Failed to load preferences: {result['message']}")
        return
    
    current_prefs = result["preferences"]
    
    console.print(f"\n{style_text('Current authors:', 'accent')} {', '.join(current_prefs['authors']) if current_prefs['authors'] else 'None'}")
    
    console.print(f"\n{style_text('Instructions:', 'info')}")
    console.print("• Add authors whose papers you want to prioritize")
    console.print("• Enter names separated by commas")
    console.print("• Use last names or full names (e.g., 'Hinton', 'Geoffrey Hinton')")
    
    authors_input = Prompt.ask(
        "\n[bold]Preferred authors[/bold]",
        default=", ".join(current_prefs["authors"])
    )
    
    new_authors = [auth.strip() for auth in authors_input.split(",") if auth.strip()] if authors_input else []
    
    current_prefs["authors"] = new_authors
    
    save_result = await unified_config_service.save_user_preferences(user_id, current_prefs)
    
    if save_result["success"]:
        print_success(console, "Authors updated successfully!")
        if new_authors:
            console.print(f"Preferred authors: {', '.join(new_authors)}")
    else:
        print_error(console, f"Failed to save: {save_result['message']}")

async def _configure_schedule(user_id: str):
    """Configure daily dose schedule"""
    from ...services.unified_config_service import unified_config_service
    
    console.print(f"\n{style_text('Configure Daily Dose Schedule', 'primary')}")
    console.print("─" * 50)
    
    result = await unified_config_service.get_user_preferences(user_id)
    if not result["success"]:
        print_error(console, f"Failed to load preferences: {result['message']}")
        return
    
    current_prefs = result["preferences"]
    current_enabled = current_prefs.get('daily_dose_enabled', False)
    current_time = current_prefs.get('daily_dose_time', '08:00')
    
    console.print(f"\n{style_text('Current settings:', 'accent')}")
    console.print(f"  • Status: {'Enabled' if current_enabled else 'Disabled'}")
    console.print(f"  • Time: {current_time} UTC")
    
    console.print(f"\n{style_text('Note:', 'info')}")
    console.print("• The scheduler daemon must be running in the cloud for this to work")
    console.print("• See DEPLOYMENT.txt for deployment instructions")
    console.print("• Time is in UTC timezone (24-hour format)")
    
    enable_schedule = Confirm.ask(
        "\n[bold]Enable daily dose schedule?[/bold]",
        default=current_enabled
    )
    
    if enable_schedule:
        console.print(f"\n{style_text('Time format:', 'info')} HH:MM (e.g., 08:00 for 8 AM, 14:30 for 2:30 PM)")
        
        while True:
            time_input = Prompt.ask(
                "\n[bold]Daily dose time (UTC)[/bold]",
                default=current_time
            )
            
            if len(time_input.split(":")) == 2:
                try:
                    hour, minute = map(int, time_input.split(":"))
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        current_prefs["daily_dose_time"] = time_input
                        break
                    else:
                        print_error(console, "Hour must be 0-23, minute must be 0-59")
                except ValueError:
                    print_error(console, "Invalid time format. Use HH:MM (e.g., 08:00)")
            else:
                print_error(console, "Invalid time format. Use HH:MM (e.g., 08:00)")
    
    current_prefs["daily_dose_enabled"] = enable_schedule
    
    save_result = await unified_config_service.save_user_preferences(user_id, current_prefs)
    
    if save_result["success"]:
        print_success(console, "Schedule updated successfully!")
        if enable_schedule:
            console.print(f"Daily dose will be delivered at {current_prefs['daily_dose_time']} UTC")
            console.print("\n[yellow]Make sure the scheduler daemon is deployed to the cloud[/yellow]")
        else:
            console.print("Daily dose schedule disabled")
    else:
        print_error(console, f"Failed to save: {save_result['message']}")

async def _reset_preferences(user_id: str):
    """Reset preferences to default"""
    # Import here to avoid circular dependencies
    from ...services.unified_config_service import unified_config_service
    
    if Confirm.ask("\n[bold red]Are you sure you want to reset all preferences to default?[/bold red]"):
        default_prefs = unified_config_service._get_default_preferences()
        result = await unified_config_service.save_user_preferences(user_id, default_prefs)
        
        if result["success"]:
            print_success(console, "Preferences reset to default!")
        else:
            print_error(console, f"Failed to reset: {result['message']}")
    else:
        print_warning(console, "Reset cancelled")

async def _show_preferences_menu(user_id: str):
    """Show interactive preferences menu"""
    await _show_preferences(user_id)
    
    console.print(f"\n{style_text('Preferences Menu', 'primary')}")
    console.print("─" * 50)
    
    choices = [
        ("1", "Configure Categories"),
        ("2", "Configure Keywords"),
        ("3", "Configure Authors"),
        ("4", "Configure Schedule"),
        ("5", "Reset to Defaults"),
        ("q", "Quit")
    ]
    
    for key, desc in choices:
        console.print(f"  [{key}] {desc}")
    
    choice = Prompt.ask("\n[bold]Select an option[/bold]", choices=[c[0] for c in choices], default="q")
    
    if choice == "1":
        await _configure_categories(user_id)
    elif choice == "2":
        await _configure_keywords(user_id)
    elif choice == "3":
        await _configure_authors(user_id)
    elif choice == "4":
        await _configure_schedule(user_id)
    elif choice == "5":
        await _reset_preferences(user_id)

if __name__ == "__main__":
    preferences_command()
