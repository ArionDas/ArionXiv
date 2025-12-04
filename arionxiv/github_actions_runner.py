"""
GitHub Actions Runner for ArionXiv Daily Dose

This module is executed by GitHub Actions to run daily dose for users
who have their scheduled time matching the current hour (UTC).

Usage:
    python -m arionxiv.github_actions_runner

Environment Variables:
    MONGODB_URI: MongoDB connection string (required)
    OPENROUTER_API_KEY: OpenRouter API key for LLM (required, FREE tier available)
    GEMINI_API_KEY: Gemini API key for embeddings (optional)
    GROQ_API_KEY: Groq API key as fallback LLM (optional)
    FORCE_HOUR: Force run for specific hour (optional, for testing)
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

# Configure logging for GitHub Actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def get_users_for_hour(hour: int) -> List[Dict[str, Any]]:
    """
    Get all users who have daily dose enabled and scheduled for the given hour.
    
    Args:
        hour: Hour in 24-hour format (0-23)
        
    Returns:
        List of user documents matching the scheduled hour
    """
    from .services.unified_database_service import unified_database_service
    
    try:
        # Connect to MongoDB
        await unified_database_service.connect_mongodb()
        
        # Query users with daily dose enabled and matching hour
        # Time format is "HH:MM", so we match the hour part
        hour_prefix = f"{hour:02d}:"
        
        users_collection = unified_database_service.get_users_collection()
        
        # Find users where:
        # 1. Daily dose is enabled
        # 2. Scheduled time starts with the current hour
        query = {
            "$or": [
                # New format: preferences.daily_dose.enabled and preferences.daily_dose.scheduled_time
                {
                    "preferences.daily_dose.enabled": True,
                    "preferences.daily_dose.scheduled_time": {"$regex": f"^{hour_prefix}"}
                },
                # Legacy format: preferences.daily_dose_enabled and preferences.daily_dose_time
                {
                    "preferences.daily_dose_enabled": True,
                    "preferences.daily_dose_time": {"$regex": f"^{hour_prefix}"}
                }
            ]
        }
        
        cursor = users_collection.find(query)
        users = await cursor.to_list(length=None)
        
        logger.info(f"Found {len(users)} users scheduled for hour {hour:02d}:00 UTC")
        return users
        
    except Exception as e:
        logger.error(f"Error fetching users for hour {hour}: {e}")
        return []


async def run_daily_dose_for_user(user_id: str, user_email: str) -> bool:
    """
    Run daily dose for a specific user.
    
    Args:
        user_id: MongoDB user ID
        user_email: User email for logging
        
    Returns:
        True if successful, False otherwise
    """
    from .services.unified_daily_dose_service import unified_daily_dose_service
    
    try:
        logger.info(f"Running daily dose for user: {user_email}")
        
        result = await unified_daily_dose_service.generate_daily_dose(user_id)
        
        if result.get("success"):
            papers_count = len(result.get("papers", []))
            logger.info(f"Daily dose completed for {user_email}: {papers_count} papers analyzed")
            return True
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"Daily dose failed for {user_email}: {error}")
            return False
            
    except Exception as e:
        logger.error(f"Exception running daily dose for {user_email}: {e}")
        return False


async def main():
    """Main entry point for GitHub Actions runner."""
    
    logger.info("=" * 60)
    logger.info("ArionXiv Daily Dose - GitHub Actions Runner")
    logger.info("=" * 60)
    
    # Check required environment variables
    # OpenRouter is the primary LLM provider (FREE tier available)
    required_vars = ["MONGODB_URI", "OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please add these as GitHub Actions secrets.")
        logger.error("Get a FREE OpenRouter API key at: https://openrouter.ai/")
        sys.exit(1)
    
    # Determine the hour to process
    force_hour = os.environ.get("FORCE_HOUR", "").strip()
    
    if force_hour:
        try:
            current_hour = int(force_hour)
            if not 0 <= current_hour <= 23:
                raise ValueError("Hour must be between 0 and 23")
            logger.info(f"Using forced hour: {current_hour:02d}:00 UTC")
        except ValueError as e:
            logger.error(f"Invalid FORCE_HOUR value '{force_hour}': {e}")
            sys.exit(1)
    else:
        current_hour = datetime.now(timezone.utc).hour
        logger.info(f"Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(f"Processing hour: {current_hour:02d}:00 UTC")
    
    # Get users scheduled for this hour
    users = await get_users_for_hour(current_hour)
    
    if not users:
        logger.info(f"No users scheduled for {current_hour:02d}:00 UTC. Exiting.")
        return
    
    # Process each user
    success_count = 0
    failure_count = 0
    
    for user in users:
        user_id = str(user["_id"])
        user_email = user.get("email", "unknown")
        
        if await run_daily_dose_for_user(user_id, user_email):
            success_count += 1
        else:
            failure_count += 1
        
        # Small delay between users to avoid rate limiting
        await asyncio.sleep(2)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Daily Dose Run Complete")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")
    logger.info(f"  Total: {len(users)}")
    logger.info("=" * 60)
    
    # Exit with error if any failures
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
