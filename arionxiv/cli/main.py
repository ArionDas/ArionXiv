#!/usr/bin/env python3
"""
ArionXiv CLI - Terminal-based research paper analysis tool
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Initialize quiet logging for better UX
from ..services.unified_config_service import unified_config_service
unified_config_service.setup_logging()

logger = logging.getLogger(__name__)

import click
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from .commands.search import search_command
from .commands.fetch import fetch_command
from .commands.analyze import analyze_command
from .commands.chat import chat_command
from .commands.daily import daily_command
from .commands.library import library_command
from .commands.trending import trending_command
from .commands.settings_unified import settings
from .commands.auth import auth_command, auth_interface, login_command, logout_command, register_command, session_command
from .commands.welcome import welcome, show_logo_and_features
from .ui.logo import display_logo, display_welcome_message, display_startup_info
from .ui.theme import create_themed_console, print_header, style_text, get_theme_colors
from .utils.db_config_manager import db_config_manager
from .utils.api_config import api_config_manager, run_first_time_api_setup
from .ui.theme import run_theme_selection
from .utils.animations import left_to_right_reveal, animated_help_line
from ..services.unified_user_service import unified_user_service

console = create_themed_console()


def _display_themed_help(ctx, cmd_or_group, is_group=True):
    """Display themed help with ARIONXIV header and animated commands"""
    from .ui.theme import get_theme_colors
    import shutil
    colors = get_theme_colors()
    primary = colors.get('primary', 'cyan')
    
    # Get terminal width for the line
    terminal_width = shutil.get_terminal_size().columns
    
    # Display ARIONXIV header at top right
    header_text = Text("ARIONXIV", style=f"bold {primary}")
    console.print(Align.right(header_text))
    # Draw horizontal line spanning the entire terminal
    console.print(f"[{primary}]{'─' * terminal_width}[/{primary}]")
    console.print()
    
    # Display usage - always use full command path (e.g., "arionxiv settings" not just "settings")
    # Normalize the command path to always start with "arionxiv"
    def normalize_prog_name(path):
        """Normalize command path to use 'arionxiv' as the base"""
        if not path:
            return 'arionxiv'
        # Replace common invocation patterns with just 'arionxiv'
        path = path.replace('python -m arionxiv', 'arionxiv')
        path = path.replace('__main__.py', 'arionxiv')
        if not path.startswith('arionxiv'):
            path = 'arionxiv ' + path
        return path
    
    if is_group:
        prog_name = normalize_prog_name(ctx.command_path or ctx.info_name)
        usage_text = f"Usage: [{primary}]{prog_name}[/{primary}] [OPTIONS] COMMAND [ARGS]..."
    else:
        prog_name = normalize_prog_name(ctx.command_path or ctx.info_name)
        # Build usage with argument placeholders
        args_str = ""
        for param in cmd_or_group.params:
            if isinstance(param, click.Argument):
                if param.required:
                    args_str += f" {param.name.upper()}"
                else:
                    args_str += f" [{param.name.upper()}]"
        usage_text = f"Usage: [{primary}]{prog_name}[/{primary}] [OPTIONS]{args_str}"
    
    # Animate the usage text (the function prints the final text)
    left_to_right_reveal(console, usage_text.replace(f"[{primary}]", "").replace(f"[/{primary}]", ""), style="white", duration=0.3)
    console.print()
    
    # Display description
    help_text = cmd_or_group.help or ""
    if help_text:
        for line in help_text.strip().split('\n'):
            # Color commands in quick access section (lines starting with 'arionxiv')
            stripped = line.strip()
            if stripped.startswith('arionxiv '):
                # Split into command and comment
                if '#' in stripped:
                    cmd_part, comment_part = stripped.split('#', 1)
                    console.print(f"  [{primary}]{cmd_part.strip()}[/{primary}]  # {comment_part.strip()}")
                else:
                    console.print(f"  [{primary}]{stripped}[/{primary}]")
            else:
                console.print(f"  {stripped}")
        console.print()
    
    # Display options
    params = cmd_or_group.params
    options = [p for p in params if isinstance(p, click.Option)]
    if options:
        console.print(Text("Options:", style="bold white"))
        
        # Calculate max option length for alignment
        max_opt_len = 0
        opt_data = []
        for param in options:
            opt_names = ', '.join(param.opts)
            if param.secondary_opts:
                opt_names += ', ' + ', '.join(param.secondary_opts)
            opt_help = param.help or ""
            opt_data.append((opt_names, opt_help))
            max_opt_len = max(max_opt_len, len(opt_names))
        
        # Add --help to calculation
        max_opt_len = max(max_opt_len, len("--help"))
        
        for opt_names, opt_help in opt_data:
            padding = ' ' * (max_opt_len - len(opt_names) + 2)
            animated_help_line(console, opt_names, opt_help, primary, padding, duration=0.5)
        
        # Always add --help
        help_opt = "--help"
        padding = ' ' * (max_opt_len - len(help_opt) + 2)
        animated_help_line(console, help_opt, "Show this message and exit.", primary, padding, duration=0.5)
        console.print()
    
    # Display commands (for groups only)
    if is_group and hasattr(cmd_or_group, 'list_commands'):
        commands = []
        for subcommand in cmd_or_group.list_commands(ctx):
            cmd = cmd_or_group.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            help_text = cmd.get_short_help_str(limit=80)
            commands.append((subcommand, help_text))
        
        if commands:
            console.print(Text("Commands:", style="bold white"))
            max_len = max(len(cmd) for cmd, _ in commands)
            
            for cmd_name, help_text in commands:
                padding = ' ' * (max_len - len(cmd_name) + 2)
                animated_help_line(console, cmd_name, help_text, primary, padding, duration=0.5)
            
            console.print()


class ThemedCommand(click.Command):
    """Custom Click command that uses themed help formatting with animations"""
    
    def format_help(self, ctx, formatter):
        """Override to use our custom themed help display"""
        _display_themed_help(ctx, self, is_group=False)


class ThemedSubGroup(click.Group):
    """Custom Click subgroup with themed help and error handling for invalid commands"""
    
    def format_help(self, ctx, formatter):
        """Override to use our custom themed help display"""
        _display_themed_help(ctx, self, is_group=True)
    
    def invoke(self, ctx):
        """Override invoke to catch errors from subcommands"""
        try:
            return super().invoke(ctx)
        except click.UsageError as e:
            logger.warning(f"Invalid subcommand: {str(e)}")
            self._show_error(e, ctx)
            raise SystemExit(1)
    
    def _show_error(self, error, ctx):
        """Display themed error message for invalid subcommands"""
        colors = get_theme_colors()
        error_console = Console()
        error_msg = str(error)
        
        # Get command path for better messaging
        cmd_path = ctx.info_name if ctx else "settings"
        logger.debug(f"Showing error for command path: {cmd_path}")
        
        error_console.print()
        error_console.print(f"[bold {colors['error']}]⚠ Invalid Command[/bold {colors['error']}]")
        error_console.print(f"[{colors['error']}]{error_msg}[/{colors['error']}]")
        error_console.print()
        
        # Show available subcommands
        error_console.print(f"[bold white]Available '{cmd_path}' subcommands:[/bold white]")
        for cmd_name in sorted(self.list_commands(ctx)):
            cmd = self.get_command(ctx, cmd_name)
            if cmd and not cmd.hidden:
                help_text = cmd.get_short_help_str(limit=50)
                error_console.print(f"  [{colors['primary']}]{cmd_name}[/{colors['primary']}]  {help_text}")
        
        error_console.print()
        error_console.print(f"Run [{colors['primary']}]arionxiv {cmd_path} --help[/{colors['primary']}] for more information.")
        error_console.print()


def _patch_command_help(cmd):
    """Patch a command to use themed help display, recursively for groups"""
    
    def themed_format_help(ctx, formatter):
        _display_themed_help(ctx, cmd, is_group=isinstance(cmd, click.Group))
    
    cmd.format_help = themed_format_help
    
    # Recursively patch subcommands if this is a group
    if isinstance(cmd, click.Group):
        for name in cmd.list_commands(None):
            subcmd = cmd.get_command(None, name)
            if subcmd:
                _patch_command_help(subcmd)
    
    return cmd


class ThemedGroup(click.Group):
    """Custom Click group that uses themed help formatting with animations"""
    
    def format_help(self, ctx, formatter):
        """Override to use our custom themed help display"""
        _display_themed_help(ctx, self, is_group=True)
    
    def add_command(self, cmd, name=None):
        """Override to patch commands with themed help"""
        # Patch the command's help formatting
        _patch_command_help(cmd)
        super().add_command(cmd, name)
    
    def invoke(self, ctx):
        """Override invoke to catch errors from subcommands"""
        try:
            return super().invoke(ctx)
        except click.UsageError as e:
            self._show_error(e, ctx)
            raise SystemExit(1)
    
    def _show_error(self, error, ctx):
        """Display themed error message for invalid commands"""
        import sys
        colors = get_theme_colors()
        error_console = Console()
        error_msg = str(error)
        
        # Try to determine which command/subcommand caused the error
        args_list = sys.argv[1:]
        
        # Check if this is a subcommand error (e.g., "settings invalidcmd")
        subgroup = None
        parent_cmd = None
        if args_list and len(args_list) >= 1:
            potential_subcmd = args_list[0]
            subcmd = self.get_command(ctx, potential_subcmd)
            if subcmd and isinstance(subcmd, click.Group):
                subgroup = subcmd
                parent_cmd = potential_subcmd
        
        logger.warning(f"Invalid command: {error_msg}")
        error_console.print()
        error_console.print(f"[bold {colors['error']}]⚠ Invalid Command[/bold {colors['error']}]")
        error_console.print(f"[{colors['error']}]{error_msg}[/{colors['error']}]")
        error_console.print()
        
        # Show available commands from the relevant group
        if subgroup and parent_cmd:
            # Show subcommands of the subgroup
            logger.debug(f"Showing subcommands for: {parent_cmd}")
            error_console.print(f"[bold white]Available '{parent_cmd}' subcommands:[/bold white]")
            for cmd_name in sorted(subgroup.list_commands(ctx)):
                cmd = subgroup.get_command(ctx, cmd_name)
                if cmd and not cmd.hidden:
                    help_text = cmd.get_short_help_str(limit=50)
                    error_console.print(f"  [{colors['primary']}]{cmd_name}[/{colors['primary']}]  {help_text}")
            error_console.print()
            error_console.print(f"Run [{colors['primary']}]arionxiv {parent_cmd} --help[/{colors['primary']}] for more information.")
        else:
            # Show main commands
            logger.debug("Showing main commands")
            error_console.print(f"[bold white]Available commands:[/bold white]")
            for cmd_name in sorted(self.list_commands(ctx)):
                cmd = self.get_command(ctx, cmd_name)
                if cmd and not cmd.hidden:
                    help_text = cmd.get_short_help_str(limit=50)
                    error_console.print(f"  [{colors['primary']}]{cmd_name}[/{colors['primary']}]  {help_text}")
            error_console.print()
            error_console.print(f"Run [{colors['primary']}]arionxiv --help[/{colors['primary']}] for more information.")
        
        error_console.print()
    
    def main(self, *args, standalone_mode=True, **kwargs):
        """Override main to intercept --help and handle invalid commands"""
        import sys
        logger.debug(f"CLI invoked with args: {sys.argv[1:]}")
        # Only intercept if --help is the first or second argument (for main group help)
        # Don't intercept if there's a subcommand before --help
        args_list = sys.argv[1:]
        
        # Check if --help is for the main group (not a subcommand)
        if args_list and args_list[0] in ('--help', '-h'):
            try:
                with self.make_context('arionxiv', []) as ctx:
                    _display_themed_help(ctx, self, is_group=True)
                    sys.exit(0)
            except click.exceptions.Exit:
                sys.exit(0)
            except Exception:
                pass
        
        # Use standalone_mode=False to handle exceptions ourselves
        try:
            return super().main(*args, standalone_mode=False, **kwargs)
        except click.UsageError as e:
            self._show_error(e, None)
            sys.exit(1)
        except click.Abort:
            sys.exit(1)
        except click.exceptions.Exit as e:
            sys.exit(e.exit_code)


@click.group(cls=ThemedGroup, invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.pass_context
def cli(ctx, version):
    """
    ArionXiv CLI - AI-Powered Research Paper Analysis
    
    A powerful terminal interface for discovering, analyzing, and chatting with research papers.
    """
    if version:
        console.print(style_text("ArionXiv CLI v1.0.0", "success"))
        return
    
    # Load stored API keys into environment
    api_config_manager.load_keys_to_environment()
    
    # Check for first-time API setup (only when no subcommand)
    if ctx.invoked_subcommand is None:
        if api_config_manager.is_first_time_setup_needed():
            # Show logo first
            show_logo_and_features(console, animate=False)
            # Run first-time API setup
            run_first_time_api_setup(console)
            console.print()
        else:
            # Show unified logo and features for any user
            show_logo_and_features(console, animate=False)
        
        # Show a simple getting started message
        colors = get_theme_colors()
        console.print(f"[bold {colors['primary']}]Ready to explore research papers? Try:[/bold {colors['primary']}]")
        console.print(f"   [bold {colors['primary']}]arionxiv search \"your topic\"[/bold {colors['primary']}]")
        console.print()

async def _handle_main_flow():
    """Handle the main CLI flow with authentication"""
    # Ensure user is authenticated
    user = await auth_interface.ensure_authenticated()
    if not user:
        return
    
    # Load configuration
    await db_config_manager.load_config()
    
    # Check if first time user or theme not configured
    is_first_time = db_config_manager.get('first_time_user', True)
    theme_configured = db_config_manager.is_theme_configured()
    
    # If first time or theme not configured, run theme selection
    if is_first_time or not theme_configured:
        selected_theme = run_theme_selection(console)
        await db_config_manager.set_theme_color(selected_theme)
    
    # Always show logo (now with selected theme)
    display_logo(console)
    
    if is_first_time:
        display_welcome_message(console)
        # Mark as not first time user
        await db_config_manager.set('first_time_user', False)
    else:
        display_startup_info(console)
    
    console.print(f"Use {style_text('arionxiv --help', 'primary')} to see available commands")

# Register all commands
cli.add_command(welcome, name="welcome")
cli.add_command(search_command, name="search")
cli.add_command(fetch_command, name="fetch")
cli.add_command(analyze_command, name="analyze")
cli.add_command(chat_command, name="chat")
cli.add_command(daily_command, name="daily")
cli.add_command(library_command, name="library")
cli.add_command(trending_command, name="trending")
cli.add_command(settings, name="settings")
cli.add_command(login_command, name="login")
cli.add_command(logout_command, name="logout")
cli.add_command(register_command, name="register")
cli.add_command(session_command, name="session")
cli.add_command(auth_command, name="auth")  # Hidden, for backward compatibility


# Create a preferences alias that redirects to settings prefs
@click.command('preferences')
@click.pass_context
def preferences_redirect(ctx):
    """Configure paper preferences (alias for 'settings prefs')"""
    console.print(f"[{get_theme_colors()['primary']}]Redirecting to settings prefs...[/{get_theme_colors()['primary']}]\n")
    ctx.invoke(settings.commands['prefs'])


cli.add_command(preferences_redirect, name="preferences")

if __name__ == "__main__":
    cli()
