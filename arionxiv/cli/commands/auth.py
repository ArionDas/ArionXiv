"""CLI Authentication Interface for ArionXiv

Uses the hosted ArionXiv API for authentication - no local database required.
"""

import sys
import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
import getpass
from typing import Optional, Dict, Any

from ..utils.api_client import api_client, APIClientError
from ...services.unified_user_service import unified_user_service
from ..ui.theme import create_themed_console, style_text, print_success, print_error, print_warning, create_themed_panel, get_theme_colors
from ..utils.animations import shake_text, left_to_right_reveal
from .welcome import show_logo_and_features

console = create_themed_console()
logger = logging.getLogger(__name__)


class AuthInterface:
    """Handles CLI authentication interface using the hosted API"""
    
    def __init__(self):
        self.console = console
        logger.debug("AuthInterface initialized")
    
    async def ensure_authenticated(self) -> Optional[Dict[str, Any]]:
        """Ensure user is authenticated, prompt if not"""
        logger.debug("Checking authentication status")
        
        # Check if already authenticated locally
        if unified_user_service.is_authenticated():
            logger.info("User already authenticated")
            return unified_user_service.get_current_user()
        
        # Check if API client has a stored token
        if api_client.is_authenticated():
            try:
                profile = await api_client.get_profile()
                if profile.get("success") and profile.get("user"):
                    user = profile["user"]
                    unified_user_service.create_session(user)
                    logger.info(f"Restored session for: {user.get('user_name')}")
                    return user
            except APIClientError:
                logger.debug("Stored token invalid, need to re-authenticate")
        
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
            
            left_to_right_reveal(self.console, f"Option {style_text(choice, 'primary')} selected!", duration=0.5)
            
            if choice == "1":
                user = await self._login_flow()
                if user:
                    return user
            elif choice == "2":
                user = await self._register_flow()
                if user:
                    return user
            elif choice == "3":
                left_to_right_reveal(self.console, f"\n{style_text('Goodbye!', 'warning')}", duration=0.5)
                return None
    
    async def _login_flow(self) -> Optional[Dict[str, Any]]:
        """Handle user login via hosted API"""
        self.console.print(f"\n[bold]{style_text('Login to ArionXiv', 'primary')}[/bold]")
        self.console.print(f"[bold]{style_text('-' * 30, 'primary')}[/bold]")
        
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            try:
                identifier = Prompt.ask(
                    f"\n[bold]{style_text('Username or Email', 'primary')}[/bold]"
                ).strip()
                
                if not identifier:
                    print_error(self.console, f"{style_text('Username/Email is required', 'error')}")
                    continue
                
                self.console.print(f"\n[bold]{style_text('Password:', 'primary')}[/bold]")
                password = getpass.getpass(f"> ")
                
                if not password:
                    print_error(self.console, f"{style_text('Password is required', 'error')}")
                    continue
                
                self.console.print(f"\n[white]{style_text('Authenticating...', 'primary')}[/white]")
                logger.info(f"Attempting login for: {identifier}")
                
                result = await api_client.login(identifier, password)
                
                if result.get("success"):
                    user = result.get("user", {})
                    logger.info(f"Login successful for user: {user.get('user_name')}")
                    
                    session_token = unified_user_service.create_session(user)
                    if session_token:
                        logger.debug("Session created successfully")
                        left_to_right_reveal(self.console, f"Welcome back, [bold]{style_text(user.get('user_name', 'User'), 'primary')}![/bold]", duration=0.5)
                        self.console.print()
                        show_logo_and_features(self.console, animate=False)
                        return user
                    else:
                        logger.error("Failed to create session after successful login")
                        print_error(self.console, f"{style_text('Failed to create session', 'error')}")
                        return None
                else:
                    attempts += 1
                    remaining = max_attempts - attempts
                    error_msg = result.get('message') or result.get('error', 'Login failed')
                    logger.warning(f"Login failed for {identifier}: {error_msg}")
                    print_error(self.console, f"{style_text(error_msg, 'error')}")
                    
                    if remaining > 0:
                        print_warning(self.console, f"You have {remaining} attempts remaining")
                    else:
                        print_error(self.console, f"{style_text('Maximum login attempts exceeded', 'error')}")
                        break
                
            except APIClientError as e:
                attempts += 1
                remaining = max_attempts - attempts
                logger.warning(f"API login error: {e.message}")
                print_error(self.console, f"{style_text(e.message, 'error')}")
                if remaining > 0:
                    print_warning(self.console, f"You have {remaining} attempts remaining")
            except KeyboardInterrupt:
                self.console.print(f"\n{style_text('Login cancelled', 'warning')}")
                return None
            except Exception as e:
                logger.error(f"Login error: {str(e)}", exc_info=True)
                print_error(self.console, f"Login error: {str(e)}")
                return None
        
        return None
    
    async def _register_flow(self) -> Optional[Dict[str, Any]]:
        """Handle user registration via hosted API"""
        self.console.print(f"\n{style_text('Create ArionXiv Account', 'primary')}")
        self.console.print("-" * 40)
        
        try:
            full_name = Prompt.ask(
                f"\n[bold]{style_text('Full Name (optional)', 'primary')}[/bold]",
                default=""
            ).strip()
            
            while True:
                email = Prompt.ask(
                    f"\n[bold]{style_text('Email Address', 'primary')}[/bold]",
                    default=""
                ).strip()
                
                if not email:
                    print_error(self.console, f"{style_text('Email is required', 'error')}")
                    continue
                break
            
            while True:
                user_name = Prompt.ask(
                    f"\n[bold]{style_text('Username', 'primary')}[/bold] (letters, numbers, underscore, hyphen only)",
                ).strip()
                
                if not user_name:
                    print_error(self.console, f"{style_text('Username is required', 'error')}")
                    continue
                break
            
            while True:
                self.console.print(f"\n[bold]{style_text('Password:', 'primary')}[/bold] (minimum 8 characters, must contain letter and number)")
                password = getpass.getpass("> ")
                
                if not password:
                    print_error(self.console, f"{style_text('Password is required', 'error')}")
                    continue
                
                password_confirm = getpass.getpass(f"{style_text('Confirm Password:', 'primary')} ")
                
                if password != password_confirm:
                    print_error(self.console, f"{style_text('Passwords do not match', 'error')}")
                    continue
                
                break
            
            self.console.print(f"\n[bold]{style_text('Account Summary', 'primary')}[/bold]")
            self.console.print(f"Full Name: {style_text(full_name, 'primary') if full_name else style_text('Not provided', 'secondary')}")
            self.console.print(f"Email: {style_text(email, 'primary')}")
            self.console.print(f"Username: {style_text(user_name, 'primary')}")
            
            if not Confirm.ask(f"\n[bold]{style_text('Create account with these details?', 'primary')}[/bold]"):
                return None
            
            self.console.print(f"\n[white]{style_text('Creating account...', 'primary')}[/white]")
            logger.info(f"Attempting registration for: {email} ({user_name})")
            
            result = await api_client.register(email, user_name, password, full_name)
            
            if result.get("success"):
                user = result.get("user", {})
                logger.info(f"Registration successful for user: {user.get('user_name')}")
                
                try:
                    login_result = await api_client.login(user_name, password)
                    if login_result.get("success"):
                        user = login_result.get("user", user)
                except Exception:
                    pass
                
                session_token = unified_user_service.create_session(user)
                if session_token:
                    logger.debug("Session created for new user")
                    shake_text(self.console, f"Account created! Welcome, {user.get('user_name', 'User')}!")
                    self.console.print()
                    show_logo_and_features(self.console, animate=False)
                    return user
                else:
                    logger.error("Failed to create session for new user")
                    print_error(self.console, style_text("Failed to create session", "error"))
                    return None
            else:
                error_msg = result.get("message") or result.get("error", "Registration failed")
                logger.warning(f"Registration failed for {email}: {error_msg}")
                print_error(self.console, error_msg)
                return None
            
        except APIClientError as e:
            logger.warning(f"API registration error: {e.message}")
            print_error(self.console, f"{style_text(e.message, 'error')}")
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
            if user.get('full_name'):
                self.console.print(f"Name: [bold]{style_text(user['full_name'], 'primary')}[/bold]")
            self.console.print(f"Session created: {style_text(session['created'], 'primary')}")
            self.console.print(f"Expires: {style_text(session['expires'], 'primary')} ({style_text(session['days_remaining'], 'primary')} days remaining)")
            self.console.print(f"Last activity: {style_text(session['last_activity'], 'primary')}")
        else:
            print_warning(self.console, f"{style_text('No active session', 'warning')}")
    
    async def logout(self):
        """Logout current user"""
        if unified_user_service.is_authenticated() or api_client.is_authenticated():
            user = unified_user_service.get_current_user()
            user_name = user.get('user_name', 'User') if user else 'User'
            logger.info(f"Logging out user: {user_name}")
            
            try:
                await api_client.logout()
            except Exception:
                pass
            
            unified_user_service.clear_session()
            left_to_right_reveal(self.console, f"Goodbye, [bold]{style_text(user_name, 'primary')}[/bold]!", duration=0.5)
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
    async def _logout():
        await auth_interface.logout()
    asyncio.run(_logout())


@click.command()
def register_command():
    """Create a new ArionXiv account"""
    async def _register():
        await auth_interface._register_flow()
    asyncio.run(_register())


@click.command()
def session_command():
    """Show current session information"""
    auth_interface.show_session_info()


@click.command(hidden=True)
@click.option('--login', '-l', is_flag=True, help='Force login prompt')
@click.option('--logout', '-o', is_flag=True, help='Logout current user')
@click.option('--info', '-i', is_flag=True, help='Show session information')
def auth_command(login: bool, logout: bool, info: bool):
    """Manage user authentication (legacy)"""
    async def _handle_auth():
        if logout:
            await auth_interface.logout()
        elif info:
            auth_interface.show_session_info()
        elif login:
            await auth_interface.ensure_authenticated()
        else:
            auth_interface.show_session_info()
    
    asyncio.run(_handle_auth())


if __name__ == "__main__":
    login_command()
