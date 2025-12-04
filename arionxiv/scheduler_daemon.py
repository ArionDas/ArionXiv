import asyncio
import logging
import signal
import sys
from pathlib import Path

from .services.unified_scheduler_service import unified_scheduler
from .services.unified_config_service import unified_config_service
from .services.unified_database_service import unified_database_service

logger = logging.getLogger(__name__)

class SchedulerDaemon:
    def __init__(self):
        self.running = False
        self.scheduler = unified_scheduler
        
    async def start(self):
        try:
            logger.info("Initializing scheduler daemon")
            
            unified_config_service.setup_logging()
            
            await unified_database_service.connect_mongodb()
            logger.info("Connected to MongoDB")
            
            await self.scheduler.start()
            logger.info("Scheduler started successfully")
            
            await self._schedule_user_daily_doses()
            
            self.running = True
            
            while self.running:
                await asyncio.sleep(60)
                await self._check_and_update_schedules()
                
        except Exception as e:
            logger.error(f"Scheduler daemon error: {e}", exc_info=True)
            raise
    
    async def _schedule_user_daily_doses(self):
        """Schedule daily doses for all users who have it enabled"""
        try:
            logger.info("Scheduling user daily doses")
            
            users = await unified_database_service.get_all_active_users()
            scheduled_count = 0
            
            for user in users:
                try:
                    user_id = str(user['_id'])
                    preferences = user.get('preferences', {})
                    
                    if preferences.get('daily_dose_enabled', False):
                        daily_time = preferences.get('daily_dose_time', '08:00')
                        
                        result = await self.scheduler.schedule_user_daily_dose(
                            user_id=user_id,
                            time_str=daily_time,
                            days_back=1
                        )
                        
                        if result['success']:
                            scheduled_count += 1
                            logger.info(f"Scheduled daily dose for user {user_id} at {daily_time}")
                        else:
                            logger.error(f"Failed to schedule daily dose for user {user_id}: {result.get('error')}")
                            
                except Exception as e:
                    logger.error(f"Error scheduling daily dose for user {user.get('_id')}: {e}")
            
            logger.info(f"Scheduled daily doses for {scheduled_count} users")
            
        except Exception as e:
            logger.error(f"Error in _schedule_user_daily_doses: {e}", exc_info=True)
    
    async def _check_and_update_schedules(self):
        """Periodically check and update user schedules"""
        try:
            users = await unified_database_service.get_all_active_users()
            
            for user in users:
                try:
                    user_id = str(user['_id'])
                    preferences = user.get('preferences', {})
                    
                    job_id = f'daily_dose_{user_id}'
                    job_exists = self.scheduler.scheduler.get_job(job_id) is not None
                    
                    if preferences.get('daily_dose_enabled', False):
                        if not job_exists:
                            daily_time = preferences.get('daily_dose_time', '08:00')
                            await self.scheduler.schedule_user_daily_dose(
                                user_id=user_id,
                                time_str=daily_time,
                                days_back=1
                            )
                            logger.info(f"Re-scheduled daily dose for user {user_id}")
                    else:
                        if job_exists:
                            await self.scheduler.cancel_user_daily_dose(user_id)
                            logger.info(f"Cancelled daily dose for user {user_id}")
                            
                except Exception as e:
                    logger.error(f"Error checking schedule for user {user.get('_id')}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in _check_and_update_schedules: {e}")
            
    async def stop(self):
        logger.info("Stopping scheduler daemon")
        self.running = False
        await self.scheduler.stop()
        await unified_database_service.disconnect()
        logger.info("Scheduler daemon stopped")
        
    def handle_signal(self, sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        self.running = False

def main():
    daemon = SchedulerDaemon()
    
    signal.signal(signal.SIGINT, daemon.handle_signal)
    signal.signal(signal.SIGTERM, daemon.handle_signal)
    
    async def run():
        try:
            await daemon.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await daemon.stop()
    
    asyncio.run(run())

if __name__ == "__main__":
    main()
