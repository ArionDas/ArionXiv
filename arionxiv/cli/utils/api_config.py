"""
API Key Configuration Manager for ArionXiv
Handles secure storage and retrieval of API keys (Gemini, HuggingFace, etc.)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

# Import animation utility
try:
    from ..utils.animations import left_to_right_reveal
    ANIMATIONS_AVAILABLE = True
except ImportError:
    ANIMATIONS_AVAILABLE = False
    def left_to_right_reveal(console, text, style="", duration=1.0):
        console.print(text, style=style)


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


class APIConfigManager:
    """
    Manages API key configuration with secure local storage.
    
    Keys are stored in ~/.arionxiv/api_keys.json
    Environment variables take precedence over stored keys.
    """
    
    # Supported API providers
    PROVIDERS = {
        "gemini": {
            "name": "Google Gemini",
            "env_var": "GEMINI_API_KEY",
            "description": "Used for embeddings and AI features (FREE tier available)",
            "url": "https://aistudio.google.com/app/apikey",
            "required": False
        },
        "huggingface": {
            "name": "HuggingFace",
            "env_var": "HF_API_KEY",
            "description": "Used for model downloads and inference API",
            "url": "https://huggingface.co/settings/tokens",
            "required": False
        },
        "groq": {
            "name": "Groq",
            "env_var": "GROQ_API_KEY",
            "description": "Used for LLM-powered analysis and chat (FREE tier available)",
            "url": "https://console.groq.com/keys",
            "required": True
        }
    }
    
    def __init__(self):
        self.config_dir = Path.home() / ".arionxiv"
        self.api_keys_file = self.config_dir / "api_keys.json"
        self._keys: Dict[str, str] = {}
        self._loaded = False
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
    
    def _load_keys(self) -> Dict[str, str]:
        """Load API keys from file"""
        if self._loaded:
            return self._keys
        
        try:
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r') as f:
                    self._keys = json.load(f)
            else:
                self._keys = {}
        except Exception:
            self._keys = {}
        
        self._loaded = True
        return self._keys
    
    def _save_keys(self) -> bool:
        """Save API keys to file"""
        try:
            with open(self.api_keys_file, 'w') as f:
                json.dump(self._keys, f, indent=2)
            
            # Set restrictive permissions (owner read/write only)
            try:
                os.chmod(self.api_keys_file, 0o600)
            except Exception:
                pass  # Windows may not support chmod
            
            return True
        except Exception as e:
            console.print(f"[red]Error saving API keys: {e}[/red]")
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.
        Environment variables take precedence over stored keys.
        """
        if provider not in self.PROVIDERS:
            return None
        
        env_var = self.PROVIDERS[provider]["env_var"]
        
        # Check environment variable first
        env_key = os.getenv(env_var)
        if env_key:
            return env_key
        
        # Fall back to stored key
        self._load_keys()
        return self._keys.get(provider)
    
    def set_api_key(self, provider: str, key: str) -> bool:
        """Set API key for a provider"""
        if provider not in self.PROVIDERS:
            return False
        
        self._load_keys()
        self._keys[provider] = key
        
        # Also set as environment variable for current session
        env_var = self.PROVIDERS[provider]["env_var"]
        os.environ[env_var] = key
        
        return self._save_keys()
    
    def remove_api_key(self, provider: str) -> bool:
        """Remove API key for a provider"""
        if provider not in self.PROVIDERS:
            return False
        
        self._load_keys()
        if provider in self._keys:
            del self._keys[provider]
            return self._save_keys()
        return True
    
    def is_configured(self, provider: str) -> bool:
        """Check if a provider's API key is configured"""
        return self.get_api_key(provider) is not None
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration status for all providers"""
        status = {}
        for provider, info in self.PROVIDERS.items():
            key = self.get_api_key(provider)
            status[provider] = {
                "name": info["name"],
                "configured": key is not None,
                "source": "environment" if os.getenv(info["env_var"]) else ("stored" if key else "not set"),
                "required": info["required"],
                "masked_key": self._mask_key(key) if key else None
            }
        return status
    
    def _mask_key(self, key: str) -> str:
        """Mask API key for display (show first 4 and last 4 chars)"""
        if not key or len(key) < 12:
            return "****"
        return f"{key[:4]}...{key[-4:]}"
    
    def is_first_time_setup_needed(self) -> bool:
        """Check if first-time API setup is needed"""
        # Check if setup has been completed or skipped
        self._load_keys()
        if self._keys.get("_setup_completed"):
            return False
        
        # Check if at least the required key (Groq) is set
        if self.is_configured("groq"):
            return False
        
        return True
    
    def mark_setup_completed(self) -> bool:
        """Mark first-time setup as completed (even if skipped)"""
        self._load_keys()
        self._keys["_setup_completed"] = True
        return self._save_keys()
    
    def load_keys_to_environment(self):
        """Load stored keys into environment variables"""
        self._load_keys()
        for provider, key in self._keys.items():
            if provider.startswith("_"):
                continue  # Skip internal flags
            if provider in self.PROVIDERS:
                env_var = self.PROVIDERS[provider]["env_var"]
                if key and not os.getenv(env_var):
                    os.environ[env_var] = key


# Global instance
api_config_manager = APIConfigManager()


def _show_api_instructions(console_instance: Console, provider: str, colors: Dict[str, str]):
    """Display step-by-step instructions for getting an API key"""
    if provider not in API_KEY_INSTRUCTIONS:
        return
    
    instructions = API_KEY_INSTRUCTIONS[provider]
    steps_text = "\n".join(instructions["steps"])
    
    left_to_right_reveal(console_instance, "", duration=0.3)
    console_instance.print(Panel(
        steps_text,
        title=f"[bold {colors['primary']}]{instructions['title']}[/bold {colors['primary']}]",
        border_style=colors['primary'],
        padding=(1, 2)
    ))


def run_first_time_api_setup(console_instance: Console = None) -> bool:
    """
    Run first-time API key setup wizard.
    Returns True if setup was completed, False if skipped.
    """
    if console_instance is None:
        console_instance = console
    
    from ..ui.theme import get_theme_colors, style_text
    colors = get_theme_colors()
    
    console_instance.print()
    console_instance.print(Panel(
        "[bold]Welcome to ArionXiv![/bold]\n\n"
        "ArionXiv uses AI services for paper analysis and chat.\n"
        "All API keys are [bold]FREE[/bold] to obtain!\n\n"
        "[bold]Quick Overview:[/bold]\n"
        "  - Groq API Key - [bold]REQUIRED[/bold] for AI features (free & fast!)\n"
        "  - Gemini API Key - Optional, for enhanced embeddings\n"
        "  - HuggingFace Token - Optional, for model downloads\n\n"
        "[white]Your keys are stored securely in ~/.arionxiv/api_keys.json[/white]\n"
        "[white]They persist across sessions - configure once, use forever![/white]",
        title="[bold]First-Time API Setup[/bold]",
        border_style=colors['primary']
    ))
    
    # Ask if user wants to configure now
    if not Confirm.ask(
        f"\n[{colors['primary']}]Would you like to configure API keys now?[/{colors['primary']}]",
        default=True
    ):
        left_to_right_reveal(console_instance, f"\nSkipping API setup.", style=colors['warning'], duration=1.0)
        left_to_right_reveal(console_instance, f"You can configure keys later with: arionxiv settings api", style=colors['primary'], duration=1.0)
        api_config_manager.mark_setup_completed()
        return False
    
    console_instance.print()
    
    # Configure each provider with instructions
    for provider, info in api_config_manager.PROVIDERS.items():
        _configure_single_provider_with_instructions(console_instance, provider, info, colors, first_time=True)
    
    api_config_manager.mark_setup_completed()
    
    console_instance.print()
    console_instance.print(Panel(
        "API configuration complete!\n\n"
        f"Manage keys anytime with: [{colors['primary']}]arionxiv settings api[/{colors['primary']}]",
        title="[bold]Setup Complete[/bold]",
        border_style=colors['primary']
    ))
    
    return True


def _configure_single_provider_with_instructions(
    console_instance: Console,
    provider: str,
    info: Dict[str, Any],
    colors: Dict[str, str],
    first_time: bool = False,
    show_instructions: bool = True
) -> bool:
    """Configure a single API provider with step-by-step instructions"""
    
    current_key = api_config_manager.get_api_key(provider)
    required_text = "REQUIRED" if info["required"] else "optional"
    req_style = colors['error'] if info["required"] else colors['primary']
    
    left_to_right_reveal(console_instance, f"\n{'='*60}", style=colors['primary'], duration=0.5)
    left_to_right_reveal(console_instance, f"{info['name']} ({required_text})", style=f"bold {colors['primary']}", duration=0.8)
    left_to_right_reveal(console_instance, f"{info['description']}", style="white", duration=0.6)
    
    if current_key:
        left_to_right_reveal(console_instance, f"Already configured: {api_config_manager._mask_key(current_key)}", style=colors['primary'], duration=0.8)
        if first_time:
            # Already configured, skip
            return True
    
    # Show step-by-step instructions
    if show_instructions and provider in API_KEY_INSTRUCTIONS:
        _show_api_instructions(console_instance, provider, colors)
    
    # Ask for key
    prompt_text = f"Enter {info['name']} API key"
    if not info["required"]:
        prompt_text += " (or press Enter to skip)"
    
    key_input = Prompt.ask(
        f"\n[{colors['primary']}]{prompt_text}[/{colors['primary']}]",
        default="",
        show_default=False
    )
    
    if key_input.strip():
        if api_config_manager.set_api_key(provider, key_input.strip()):
            left_to_right_reveal(console_instance, f"{info['name']} key saved successfully!", style=colors['primary'], duration=1.0)
            return True
        else:
            left_to_right_reveal(console_instance, f"Failed to save {info['name']} key", style=colors['error'], duration=1.0)
            return False
    else:
        if info["required"]:
            left_to_right_reveal(console_instance, f"Warning: {info['name']} key is REQUIRED for AI features", style=colors['warning'], duration=1.0)
            left_to_right_reveal(console_instance, f"  You can add it later with: arionxiv settings api", style=colors['warning'], duration=0.8)
        else:
            left_to_right_reveal(console_instance, f"Skipped {info['name']} (you can add it later)", style="white", duration=0.8)
        return True


def _configure_single_provider(
    console_instance: Console,
    provider: str,
    info: Dict[str, Any],
    colors: Dict[str, str],
    first_time: bool = False
) -> bool:
    """Configure a single API provider (legacy - without detailed instructions)"""
    
    current_key = api_config_manager.get_api_key(provider)
    required_text = "[required]" if info["required"] else "[optional]"
    
    left_to_right_reveal(console_instance, f"\n{info['name']} {required_text}", style=f"bold {colors['primary']}", duration=0.8)
    left_to_right_reveal(console_instance, f"{info['description']}", style="white", duration=0.6)
    left_to_right_reveal(console_instance, f"Get your key at: {info['url']}", style="white", duration=0.6)
    
    if current_key:
        left_to_right_reveal(console_instance, f"Current: {api_config_manager._mask_key(current_key)}", style=colors['primary'], duration=0.8)
        if first_time:
            # Already configured, skip
            return True
    
    # Ask for key
    prompt_text = f"Enter {info['name']} API key"
    if not info["required"]:
        prompt_text += " (or press Enter to skip)"
    
    key_input = Prompt.ask(
        f"[{colors['primary']}]{prompt_text}[/{colors['primary']}]",
        default="",
        show_default=False
    )
    
    if key_input.strip():
        if api_config_manager.set_api_key(provider, key_input.strip()):
            left_to_right_reveal(console_instance, f"{info['name']} key saved successfully", style=colors['primary'], duration=1.0)
            return True
        else:
            left_to_right_reveal(console_instance, f"Failed to save {info['name']} key", style=colors['error'], duration=1.0)
            return False
    else:
        if info["required"]:
            left_to_right_reveal(console_instance, f"Warning: {info['name']} key is required for AI features", style=colors['warning'], duration=1.0)
        else:
            left_to_right_reveal(console_instance, f"Skipped {info['name']}", style="white", duration=0.8)
        return True


def show_api_status(console_instance: Console = None):
    """Display current API configuration status"""
    if console_instance is None:
        console_instance = console
    
    from ..ui.theme import get_theme_colors
    colors = get_theme_colors()
    
    status = api_config_manager.get_status()
    
    table = Table(
        title="API Configuration Status",
        show_header=True,
        header_style=f"bold {colors['primary']}",
        border_style=colors['primary']
    )
    table.add_column("Provider", style="bold white")
    table.add_column("Status", style="white", width=12)
    table.add_column("Source", style="white", width=12)
    table.add_column("Key", style="white", width=20)
    table.add_column("Required", style="white", width=10)
    
    for provider, info in status.items():
        status_text = f"[{colors['primary']}]Configured[/{colors['primary']}]" if info["configured"] else f"[{colors['warning']}]Not Set[/{colors['warning']}]"
        source_text = info["source"].title() if info["configured"] else "-"
        key_text = info["masked_key"] if info["masked_key"] else "-"
        required_text = "Yes" if info["required"] else "No"
        
        table.add_row(
            info["name"],
            status_text,
            source_text,
            key_text,
            required_text
        )
    
    console_instance.print()
    console_instance.print(table)
    console_instance.print()


__all__ = [
    'APIConfigManager',
    'api_config_manager',
    'run_first_time_api_setup',
    'show_api_status',
    'API_KEY_INSTRUCTIONS'
]
