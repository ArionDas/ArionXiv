"""
Animation utilities for ArionXiv CLI
Provides shake, reveal, slam, and other visual effects for terminal UI
"""

import time
from io import StringIO
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live


def slam_content(console_instance: Console, content: str, style: str = "", duration: float = 0.6):
    """
    Slam content onto screen with dramatic zoom-in and impact effect
    Content appears to fly in from distance and slam with a shake on impact
    
    Args:
        console_instance: Rich console instance
        content: Content to slam (can be multi-line)
        style: Rich style to apply
        duration: Duration of effect in seconds
    """
    lines = content.split('\n')
    num_lines = len(lines)
    
    # Phase 1: Zoom in (content grows from nothing) - 60% of duration
    # Phase 2: Impact shake - 40% of duration
    zoom_duration = duration * 0.5
    shake_duration = duration * 0.5
    
    zoom_frames = int(zoom_duration * 40)
    shake_frames = int(shake_duration * 30)
    
    with Live(console=console_instance, refresh_per_second=40, transient=True) as live:
        # Phase 1: Zoom in - show progressively more lines from center
        for i in range(zoom_frames):
            progress = (i + 1) / zoom_frames
            # Ease-out curve for dramatic deceleration
            eased = 1 - (1 - progress) ** 3
            
            # Calculate how many lines to show (from center outward)
            lines_to_show = max(1, int(num_lines * eased))
            
            # Center the visible lines
            start_idx = (num_lines - lines_to_show) // 2
            end_idx = start_idx + lines_to_show
            
            visible_lines = lines[start_idx:end_idx]
            
            # Add padding from top to simulate coming from distance
            top_padding = int((1 - eased) * 5)
            padded_content = '\n' * top_padding + '\n'.join(visible_lines)
            
            if style:
                live.update(Text(padded_content, style=style))
            else:
                live.update(Text(padded_content))
            
            time.sleep(zoom_duration / zoom_frames)
        
        # Phase 2: Impact shake - violent shake that settles
        for i in range(shake_frames):
            progress = i / shake_frames
            # Start intense, decay quickly
            intensity = int(10 * (1 - progress) ** 2)
            
            # Rapid oscillation
            if i % 2 == 0:
                offset = intensity
            else:
                offset = -intensity // 2
            
            shifted_lines = []
            for line in lines:
                if offset > 0:
                    shifted_lines.append(" " * offset + line)
                else:
                    shifted_lines.append(line)
            
            shifted_content = '\n'.join(shifted_lines)
            
            if style:
                live.update(Text(shifted_content, style=style))
            else:
                live.update(Text(shifted_content))
            
            time.sleep(shake_duration / shake_frames)
    
    # Print final stable content
    if style:
        console_instance.print(Text(content, style=style))
    else:
        console_instance.print(content)

def slam_columns(console_instance: Console, columns: Columns, duration: float = 0.6):
    """
    Slam Rich Columns onto screen with dramatic zoom-in and impact effect
    
    Args:
        console_instance: Rich console instance
        columns: Rich Columns object to slam
        duration: Duration of effect in seconds
    """
    # Capture columns as string
    temp_console = Console(file=StringIO(), force_terminal=True, width=console_instance.width)
    temp_console.print(columns)
    columns_str = temp_console.file.getvalue()
    
    lines = columns_str.split('\n')
    num_lines = len(lines)
    
    zoom_duration = duration * 0.5
    shake_duration = duration * 0.5
    
    zoom_frames = int(zoom_duration * 40)
    shake_frames = int(shake_duration * 30)
    
    with Live(console=console_instance, refresh_per_second=40, transient=True) as live:
        # Phase 1: Zoom in from center
        for i in range(zoom_frames):
            progress = (i + 1) / zoom_frames
            eased = 1 - (1 - progress) ** 3
            
            lines_to_show = max(1, int(num_lines * eased))
            start_idx = (num_lines - lines_to_show) // 2
            end_idx = start_idx + lines_to_show
            
            visible_lines = lines[start_idx:end_idx]
            
            # Horizontal squeeze effect - indent from sides
            h_squeeze = int((1 - eased) * 15)
            squeezed_lines = [" " * h_squeeze + line for line in visible_lines]
            
            live.update(Text.from_ansi('\n'.join(squeezed_lines)))
            time.sleep(zoom_duration / zoom_frames)
        
        # Phase 2: Impact shake
        for i in range(shake_frames):
            progress = i / shake_frames
            intensity = int(5 * (1 - progress) ** 2)
            
            if i % 2 == 0:
                offset = intensity
            else:
                offset = -intensity // 2
            
            shifted_lines = []
            for line in lines:
                if offset > 0:
                    shifted_lines.append(" " * offset + line)
                else:
                    shifted_lines.append(line)
            
            live.update(Text.from_ansi('\n'.join(shifted_lines)))
            time.sleep(shake_duration / shake_frames)
    
    console_instance.print(columns)

def shake_content(console_instance: Console, content: str, style: str = "", duration: float = 1.0, intensity: int = 4):
    """
    Shake content with earthquake effect for specified duration
    
    Args:
        console_instance: Rich console instance
        content: Content to shake (can be multi-line)
        style: Rich style to apply
        duration: Duration of shake in seconds
        intensity: Max spaces to shift left/right
    """
    frames = int(duration * 30)  # 30 fps
    
    with Live(console=console_instance, refresh_per_second=30, transient=True) as live:
        for i in range(frames):
            progress = i / frames
            current_intensity = int(intensity * (1 - progress * 0.5))
            offset = int(current_intensity * (1 if (i % 4) < 2 else -1) * (1 if (i % 2) == 0 else 0.5))
            
            lines = content.split('\n')
            shifted_lines = []
            for line in lines:
                if offset > 0:
                    shifted_lines.append(" " * offset + line)
                else:
                    shifted_lines.append(line)
            
            shifted_content = '\n'.join(shifted_lines)
            
            if style:
                live.update(Text(shifted_content, style=style))
            else:
                live.update(Text(shifted_content))
            
            time.sleep(duration / frames)
    
    if style:
        console_instance.print(Text(content, style=style))
    else:
        console_instance.print(content)

def shake_text(console_instance: Console, message: str, style: str = "", duration: float = 0.5, intensity: int = 3):
    """
    Shake a single line of text with earthquake effect
    
    Args:
        console_instance: Rich console instance
        message: The message to display with shake effect
        style: Rich style to apply
        duration: Duration of shake in seconds
        intensity: Max spaces to shift left/right
    """
    shake_content(console_instance, message, style=style, duration=duration, intensity=intensity)

def shake_columns(console_instance: Console, columns: Columns, duration: float = 1.0, intensity: int = 3):
    """
    Shake Rich Columns with earthquake effect
    
    Args:
        console_instance: Rich console instance
        columns: Rich Columns object to shake
        duration: Duration of shake in seconds
        intensity: Max spaces to shift
    """
    frames = int(duration * 25)
    
    temp_console = Console(file=StringIO(), force_terminal=True, width=console_instance.width)
    temp_console.print(columns)
    columns_str = temp_console.file.getvalue()
    
    with Live(console=console_instance, refresh_per_second=25, transient=True) as live:
        for i in range(frames):
            progress = i / frames
            current_intensity = int(intensity * (1 - progress * 0.3))
            offset = int(current_intensity * (1 if (i % 4) < 2 else -1) * (1 if (i % 2) == 0 else 0.6))
            
            lines = columns_str.split('\n')
            shifted_lines = []
            for line in lines:
                if offset > 0:
                    shifted_lines.append(" " * offset + line)
                else:
                    shifted_lines.append(line)
            
            live.update(Text.from_ansi('\n'.join(shifted_lines)))
            time.sleep(duration / frames)
    
    console_instance.print(columns)

def shake_panel(console_instance: Console, panel: Panel, duration: float = 1.0, intensity: int = 3):
    """
    Shake a Rich Panel with earthquake effect
    
    Args:
        console_instance: Rich console instance  
        panel: Rich Panel to shake
        duration: Duration of shake in seconds
        intensity: Max spaces to shift
    """
    frames = int(duration * 25)
    
    temp_console = Console(file=StringIO(), force_terminal=True, width=console_instance.width)
    temp_console.print(panel)
    panel_str = temp_console.file.getvalue()
    
    with Live(console=console_instance, refresh_per_second=25, transient=True) as live:
        for i in range(frames):
            progress = i / frames
            current_intensity = int(intensity * (1 - progress * 0.3))
            offset = int(current_intensity * (1 if (i % 4) < 2 else -1) * (1 if (i % 2) == 0 else 0.6))
            
            lines = panel_str.split('\n')
            shifted_lines = []
            for line in lines:
                if offset > 0:
                    shifted_lines.append(" " * offset + line)
                else:
                    shifted_lines.append(line)
            
            live.update(Text.from_ansi('\n'.join(shifted_lines)))
            time.sleep(duration / frames)
    
    console_instance.print(panel)

def left_to_right_reveal(console_instance: Console, text: str, style: str = "", duration: float = 0.8):
    """
    Reveal text from left to right character by character
    
    Args:
        console_instance: Rich console instance
        text: Text to reveal
        style: Rich style to apply
        duration: Total duration of reveal
    """
    delay = duration / len(text) if text else 0.01
    revealed = ""
    
    with Live(console=console_instance, refresh_per_second=60, transient=True) as live:
        for char in text:
            revealed += char
            if style:
                live.update(Text(revealed, style=style))
            else:
                live.update(Text(revealed))
            time.sleep(delay)
    
    if style:
        console_instance.print(Text(text, style=style))
    else:
        console_instance.print(text)

def top_to_bottom_reveal(console_instance: Console, content: str, style: str = "", duration: float = 0.5):
    """
    Reveal content line by line from top to bottom
    
    Args:
        console_instance: Rich console instance
        content: Multi-line content to reveal
        style: Rich style string
        duration: Total duration of reveal
    """
    lines = content.split('\n')
    line_delay = duration / len(lines) if lines else 0.03
    revealed = []
    
    with Live(console=console_instance, refresh_per_second=60, transient=True) as live:
        for line in lines:
            revealed.append(line)
            current_display = '\n'.join(revealed)
            if style:
                live.update(Text(current_display, style=style))
            else:
                live.update(Text.from_markup(current_display))
            time.sleep(line_delay)
    
    if style:
        console_instance.print(Text(content, style=style))
    else:
        console_instance.print(content)

def typewriter_reveal(console_instance: Console, text: str, style: str = "", delay: float = 0.02):
    """
    Reveal text with typewriter effect (character by character)
    
    Args:
        console_instance: Rich console instance
        text: Text to reveal
        style: Rich style to apply
        delay: Delay between each character
    """
    revealed = ""
    
    with Live(console=console_instance, refresh_per_second=60, transient=True) as live:
        for char in text:
            revealed += char
            if style:
                live.update(Text(revealed, style=style))
            else:
                live.update(Text(revealed))
            time.sleep(delay)
    
    if style:
        console_instance.print(Text(text, style=style))
    else:
        console_instance.print(text)

async def row_by_row_table_reveal(console_instance: Console, table_creator, num_rows: int, duration: float = 1.0):
    """
    Reveal a table row by row with animation
    
    Args:
        console_instance: Rich console instance
        table_creator: Callable that takes num_rows and returns a Table
        num_rows: Total number of rows to animate
        duration: Total duration for all rows to appear (default 1 second)
    """
    import asyncio
    
    if num_rows == 0:
        return
    
    row_delay = duration / num_rows
    
    with Live(console=console_instance, refresh_per_second=30, transient=True) as live:
        for i in range(1, num_rows + 1):
            live.update(table_creator(i))
            await asyncio.sleep(row_delay)
    
    # Print final table
    console_instance.print(table_creator(num_rows))

def stream_text_response(console_instance: Console, text: str, style: str = "", duration: float = 3.0):
    """
    Stream a long text response with smooth left-to-right flow
    Ideal for AI responses - streams word by word for natural reading
    
    Args:
        console_instance: Rich console instance
        text: Text to stream
        style: Rich style to apply
        duration: Total duration of streaming (default 3 seconds)
    """
    words = text.split()
    if not words:
        return
    
    delay = duration / len(words)
    revealed = ""
    
    with Live(console=console_instance, refresh_per_second=60, transient=True) as live:
        for word in words:
            revealed += word + " "
            if style:
                live.update(Text(revealed.strip(), style=style))
            else:
                live.update(Text(revealed.strip()))
            time.sleep(delay)
    
    if style:
        console_instance.print(Text(text, style=style))
    else:
        console_instance.print(text)

def stream_markdown_response(console_instance: Console, text: str, panel_title: str = "", border_style: str = None, duration: float = 3.0):
    """
    Stream a markdown response inside a panel with smooth word-by-word flow
    Perfect for AI assistant responses
    
    Args:
        console_instance: Rich console instance
        text: Markdown text to stream
        panel_title: Title for the panel
        border_style: Border style for the panel (defaults to theme primary color)
        duration: Total duration of streaming (default 3 seconds)
    """
    from rich.markdown import Markdown
    from ..ui.theme import get_theme_colors
    
    if border_style is None:
        colors = get_theme_colors()
        border_style = colors['primary']
    
    words = text.split()
    if not words:
        return
    
    delay = duration / len(words)
    revealed = ""
    
    with Live(console=console_instance, refresh_per_second=30, transient=True) as live:
        for word in words:
            revealed += word + " "
            panel = Panel(
                Markdown(revealed.strip()),
                title=panel_title,
                border_style=border_style
            )
            live.update(panel)
            time.sleep(delay)
    
    # Print final panel
    final_panel = Panel(
        Markdown(text),
        title=panel_title,
        border_style=border_style
    )
    console_instance.print(final_panel)


def animated_help_line(console_instance: Console, cmd_text: str, desc_text: str, primary_color: str, padding: str, duration: float = 0.5):
    """
    Animate a command/option line for help pages: reveal command name character by character,
    then show the full line with description.
    
    Args:
        console_instance: Rich console instance
        cmd_text: Command or option name to animate
        desc_text: Description text to show after command
        primary_color: Primary theme color for styling
        padding: Padding string between command and description
        duration: Duration of the animation in seconds
    """
    if not cmd_text:
        return
    
    delay = duration / len(cmd_text) if cmd_text else 0.01
    revealed = ""
    
    # Animate the command name character by character
    with Live(console=console_instance, refresh_per_second=60, transient=True) as live:
        for char in cmd_text:
            revealed += char
            live.update(Text(f"  {revealed}", style=f"bold {primary_color}"))
            time.sleep(delay)
    
    # Print the final line with both command and description
    console_instance.print(f"  [{primary_color} bold]{cmd_text}[/{primary_color} bold]{padding}{desc_text}")

