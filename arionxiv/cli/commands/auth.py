"""CLI Authentication Interface for ArionXiv"""

import sys
import asyncio
import logging
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
import getpass
from typing import Optional, Dict, Any

from ...services.unified_auth_service import auth_service
from ...services.unified_database_service import unified_database_service
from ...services.unified_user_service import unified_user_service
from ..ui.theme import create_themed_console, style_text, print_success, print_error, print_warning, create_themed_panel, get_theme_colors
from ..utils.animations import shake_text, left_to_right_reveal
from .welcome import show_logo_and_features

console = create_themed_console()
logger = logging.getLogger(__name__)


class AuthInterface:
    """Handles CLI authentication interface"""
    
    def __init__(self):
        self.console = console
        logger.debug("AuthInterface initialized")
    
    async def ensure_authenticated(self) -> Optional[Dict[str, Any]]:
        """Ensure user is authenticated, prompt if not"""
        logger.debug("Checking authentication status")
        # Check if already authenticated
        if unified_user_service.is_authenticated():
            logger.info("User already authenticated")
            return unified_user_service.get_current_user()
        
        # Initialize database connection
        if unified_database_service.db is None:
            logger.info("Initializing database connection")
            await unified_database_service.connect_mongodb()
        
        # Show authentication prompt
        return await self._authentication_flow()
    
    async def _authentication_flow(self) -> Optional[Dict[str, Any]]:
        """Main authentication flow"""
        logger.debug("Starting authentication flow")
        self.console.print()
        self.console.print(create_themed_panel(
            "[bold]Welcome to ArionXiv![/bold]\n\n"
            "To access and interact with all the features, please provide your credentials below.\n"
            "Please login or create an account.",
            title="Authentication Required"
        ))
        
        while True:
            self.console.print(f"\n[bold]{style_text('Choose an option:', 'primary')}[/bold]")
            self.console.print(f"{style_text('1', 'primary')}. Login with existing account")
            self.console.print(f"{style_text('2', 'primary')}. Create new account")
            self.console.print(f"{style_text('3', 'primary')}. Exit")
            
            choice = Prompt.ask(
                f"\n[bold]{style_text('Select option (1-3)', 'primary')}[/bold]",
                choices=["1", "2", "3"],
                default=f"1"
            )
            
            # Add slide effect when user makes a selection
            left_to_right_reveal(self.console, f"Option {style_text(choice, 'primary')} selected!", duration=1.0)
            
            if choice == "1":
                user = await self._login_flow()
                if user:
                    return user
            elif choice == "2":
                user = await self._register_flow()
                if user:
                    return user
            elif choice == "3":
                left_to_right_reveal(self.console, f"\n{style_text('Goodbye!', 'warning')}", duration=1.0)
                return None
    
    async def _login_flow(self) -> Optional[Dict[str, Any]]:
        """Handle user login"""
        self.console.print(f"\n[bold]{style_text('Login to ArionXiv', 'primary')}[/bold]")
        self.console.print(f"[bold]{style_text('-' * 30, 'primary')}[/bold]")
        
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Get username/email
                identifier = Prompt.ask(
                    f"\n[bold]{style_text('Username or Email', 'primary')}[/bold]"
                ).strip()
                
                if not identifier:
                    print_error(self.console, f"{style_text('Username/Email is required', 'error')}")
                    continue
                
                # Get password
                self.console.print(f"\n[bold]{style_text('Password:', 'primary')}[/bold]")
                password = getpass.getpass(f"> ")
                
                if not password:
                    print_error(self.console, f"{style_text('Password is required', 'error')}")
                    continue
                
                # Attempt login
                self.console.print(f"\n[white]{style_text('Authenticating...', 'primary')}[/white]")
                logger.info(f"Attempting login for: {identifier}")
                result = await auth_service.login_user(identifier, password)
                
                if result["success"]:
                    user = result["user"]
                    logger.info(f"Login successful for user: {user['user_name']}")
                    
                    # Create session
                    session_token = unified_user_service.create_session(user)
                    if session_token:
                        logger.debug("Session created successfully")
                        # Slide effect for successful login!
                        left_to_right_reveal(self.console, f"Welcome back, [bold]{style_text(user['user_name'], 'primary')}![/bold]", duration=1.0)
                        self.console.print()
                        
                        # Show main menu page after successful login
                        show_logo_and_features(self.console, animate=False)
                        return user
                    else:
                        logger.error("Failed to create session after successful login")
                        print_error(self.console, f"{style_text('Failed to create session', 'error')}")
                        return None
                else:
                    attempts += 1
                    remaining = max_attempts - attempts
                    logger.warning(f"Login failed for {identifier}: {result.get('message') or result.get('error', 'Unknown')}")
                    print_error(self.console, f"{style_text(result.get('message') or result.get('error', 'Login failed'), 'error')}")
                    
                    if remaining > 0:
                        print_warning(self.console, f"You have {remaining} attempts remaining")
                    else:
                        print_error(self.console, f"{style_text('Maximum login attempts exceeded', 'error')}")
                        break
                
            except KeyboardInterrupt:
                self.console.print(f"\n{style_text('Login cancelled', 'warning')}")
                return None
            except Exception as e:
                print_error(self.console, f"Login error: {str(e)}")
                return None
        
        return None
    
    async def _register_flow(self) -> Optional[Dict[str, Any]]:
        """Handle user registration"""
        self.console.print(f"\n{style_text('Create ArionXiv Account', 'primary')}")
        self.console.print("-" * 40)
        
        try:
            # Get full name (optional)
            full_name = Prompt.ask(
                f"\n[bold]{style_text('Full Name (optional)', 'primary')}[/bold]",
                default=""
            ).strip()
            
            # Get email
            while True:
                email = Prompt.ask(
                    f"\n[bold]{style_text('Email Address', 'primary')}[/bold]",
                    default=""
                ).strip()
                
                if not email:
                    print_error(self.console, f"{style_text('Email is required', 'error')}")
                    continue
                break
            
            # Get username
            while True:
                user_name = Prompt.ask(
                    f"\n[bold]{style_text('Username', 'primary')}[/bold] (letters, numbers, underscore, hyphen only)",
                ).strip()
                
                if not user_name:
                    print_error(self.console, f"{style_text('Username is required', 'error')}")
                    continue
                break
            
            # Get password
            while True:
                self.console.print(f"\n[bold]{style_text('Password:', 'primary')}[/bold] (minimum 8 characters, must contain letter and number)")
                password = getpass.getpass("> ")
                
                if not password:
                    print_error(self.console, f"{style_text('Password is required', 'error')}")
                    continue
                
                # Confirm password
                password_confirm = getpass.getpass(f"{style_text('Confirm Password:', 'primary')} ")
                
                if password != password_confirm:
                    print_error(self.console, f"{style_text('Passwords do not match', 'error')}")
                    continue
                
                break
            
            # Show summary and confirm
            self.console.print(f"\n[bold]{style_text('Account Summary', 'primary')}[/bold]")
            self.console.print(f"Full Name: {style_text(full_name, 'primary') if full_name else style_text('Not provided', 'secondary')}")
            self.console.print(f"Email: {style_text(email, 'primary')}")
            self.console.print(f"Username: {style_text(user_name, 'primary')}")
            
            if not Confirm.ask(f"\n[bold]{style_text('Create account with these details?', 'primary')}[/bold]"):
                return None
            
            # Attempt registration
            self.console.print(f"\n[white]{style_text('Creating account...', 'primary')}[/white]")
            logger.info(f"Attempting registration for: {email} ({user_name})")
            result = await auth_service.register_user(email, user_name, password, full_name)
            
            if result["success"]:
                user = result["user"]
                logger.info(f"Registration successful for user: {user['user_name']}")
                
                # Create session
                session_token = unified_user_service.create_session(user)
                if session_token:
                    logger.debug("Session created for new user")
                    # Shake effect for successful registration!
                    shake_text(self.console, f"Account created! Welcome, {user['user_name']}!")
                    self.console.print()
                    
                    # Show main menu page after successful registration
                    show_logo_and_features(self.console, animate=False)
                    return user
                else:
                    logger.error("Failed to create session for new user")
                    print_error(self.console, style_text("Failed to create session", "error"))
                    return None
            else:
                logger.warning(f"Registration failed for {email}: {result.get('message') or result.get('error', 'Unknown')}")
                print_error(self.console, result.get("message") or result.get("error", style_text("Registration failed", "error")))
                return None
            
        except KeyboardInterrupt:
            logger.debug("Registration cancelled by user")
            self.console.print(f"\n{style_text('Registration cancelled', 'warning')}")
            return None
        except Exception as e:
            logger.error(f"Registration error: {str(e)}", exc_info=True)
            print_error(self.console, f"{style_text('Registration error:', 'error')} {str(e)}")
            return None
    
    def show_session_info(self):
        """Show current session information"""
        logger.debug("Showing session info")
        session_info = unified_user_service.get_session_info()
        
        if session_info:
            user = session_info["user"]
            session = session_info["session"]
            
            self.console.print(f"\n[bold]{style_text('Current Session', 'primary')}[/bold]")
            self.console.print(f"[bold]{style_text('-' * 30, 'primary')}[/bold]")
            self.console.print(f"User: [bold]{style_text(user['user_name'], 'primary')}[/bold] ({user['email']})")
            if user['full_name']:
                self.console.print(f"Name: [bold]{style_text(user['full_name'], 'primary')}[/bold]")
            self.console.print(f"Session created: {style_text(session['created'], 'primary')}")
            self.console.print(f"Expires: {style_text(session['expires'], 'primary')} ({style_text(session['days_remaining'], 'primary')} days remaining)")
            self.console.print(f"Last activity: {style_text(session['last_activity'], 'primary')}")
        else:
            print_warning(self.console, f"{style_text('No active session', 'warning')}")
    
    def logout(self):
        """Logout current user"""
        if unified_user_service.is_authenticated():
            user = unified_user_service.get_current_user()
            logger.info(f"Logging out user: {user['user_name']}")
            unified_user_service.clear_session()
            left_to_right_reveal(self.console, f"Goodbye, [bold]{style_text(user['user_name'], 'primary')}[/bold]!", duration=1.0)
        else:
            logger.debug("Logout called but no active session")
            print_warning(self.console, f"{style_text('No active session to logout', 'warning')}")

# Global auth interface instance
auth_interface = AuthInterface()


@click.command()
def login_command():
    """Login to your ArionXiv account"""
    async def _login():
        await auth_interface.ensure_authenticated()
    asyncio.run(_login())


@click.command()
def logout_command():
    """Logout from your ArionXiv account"""
    auth_interface.logout()


@click.command()
def register_command():
    """Create a new ArionXiv account"""
    async def _register():
        # Initialize database connection
        if unified_database_service.db is None:
            await unified_database_service.connect_mongodb()
        await auth_interface._register_flow()
    asyncio.run(_register())


@click.command()
def session_command():
    """Show current session information"""
    auth_interface.show_session_info()


# Keep legacy auth command for backward compatibility
@click.command(hidden=True)
@click.option('--login', '-l', is_flag=True, help='Force login prompt')
@click.option('--logout', '-o', is_flag=True, help='Logout current user')
@click.option('--info', '-i', is_flag=True, help='Show session information')
def auth_command(login: bool, logout: bool, info: bool):
    """
    Manage user authentication (legacy - use login/logout/session commands instead)
    """
    async def _handle_auth():
        if logout:
            auth_interface.logout()
        elif info:
            auth_interface.show_session_info()
        elif login:
            await auth_interface.ensure_authenticated()
        else:
            auth_interface.show_session_info()
    
    asyncio.run(_handle_auth())


if __name__ == "__main__":
    login_command()
