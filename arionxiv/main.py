"""
Main CLI entry point for ArionXiv package
"""

import sys
import logging

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the arionxiv CLI command"""
    try:        
        from .cli.main import cli
        cli()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"CLI error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()