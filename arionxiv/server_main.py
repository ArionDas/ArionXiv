"""
Server entry point for ArionXiv package
"""

import sys
import asyncio
import logging

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the arionxiv-server command"""
    try:
        from .server import main as server_main
        asyncio.run(server_main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()