"""
Unified Scheduler Service for ArionXiv - Consolidates scheduler.py and daily_dose_scheduler.py
Handles both daily analysis automation and user-specific daily dose scheduling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from .unified_config_service import unified_config_service
from .unified_paper_service import unified_paper_service
from .unified_analysis_service import unified_analysis_service
from .unified_database_service import unified_database_service

# Import daily dose service lazily to avoid circular imports
def get_daily_dose_service():
    from .unified_daily_dose_service import unified_daily_dose_service
    return unified_daily_dose_service

logger = logging.getLogger(__name__)

class UnifiedSchedulerService:
    """
    Unified scheduler service that handles:
    1. Daily analysis pipeline (scheduler.py functionality)
    2. User-specific daily dose scheduling (daily_dose_scheduler.py functionality)
    """
    
    def __init__(self):
        # Configure job store
        jobstores = {
            'default': MemoryJobStore()
        }
        
        # Configure scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            timezone=unified_config_service.get_cron_schedule()["timezone"]
        )
        
        self.is_running = False
        
    # ================================
    # CORE SCHEDULER MANAGEMENT
    # ================================
    
    async def start(self):
        """Start the scheduler service"""
        if not self.is_running:
            try:
                # Schedule daily analysis pipeline
                await self._schedule_daily_analysis()
                
                # Start the scheduler
                self.scheduler.start()
                self.is_running = True
                
                logger.info("Unified scheduler service started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start scheduler service: {e}")
                raise
    
    async def stop(self):
        """Stop the scheduler service"""
        if self.is_running:
            try:
                self.scheduler.shutdown()
                self.is_running = False
                logger.info("Scheduler service stopped")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and job information"""
        try:
            jobs = self.scheduler.get_jobs()
            
            return {
                'status': 'running' if self.is_running else 'stopped',
                'timezone': str(self.scheduler.timezone),
                'job_count': len(jobs),
                'jobs': [
                    {
                        'id': job.id,
                        'name': job.name,
                        'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                        'trigger': str(job.trigger)
                    }
                    for job in jobs
                ]
            }
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # ================================
    # DAILY ANALYSIS PIPELINE (from scheduler.py)
    # ================================
    
    async def _schedule_daily_analysis(self):
        """Schedule the daily analysis pipeline"""
        try:
            cron_config = unified_config_service.get_cron_schedule()
            
            # Create cron trigger
            trigger = CronTrigger(
                hour=cron_config["hour"],
                minute=cron_config["minute"],
                timezone=cron_config["timezone"]
            )
            
            # Schedule the job
            self.scheduler.add_job(
                func=self.run_daily_analysis_pipeline,
                trigger=trigger,
                id='daily_analysis_pipeline',
                name='Daily ArXiv Analysis Pipeline',
                replace_existing=True,
                max_instances=1
            )
            
            logger.info(f"Daily analysis scheduled for {cron_config['hour']:02d}:{cron_config['minute']:02d} {cron_config['timezone']}")
            
        except Exception as e:
            logger.error(f"Failed to schedule daily analysis: {e}")
            raise
    
    async def run_daily_analysis_pipeline(self):
        """Execute the complete daily analysis pipeline"""
        start_time = datetime.now()
        logger.info("Starting daily analysis pipeline")
        
        try:
            # Step 1: Fetch latest papers from ArXiv
            logger.info("Fetching latest papers from ArXiv...")
            papers = await unified_paper_service.fetch_daily_papers()
            
            if not papers:
                logger.warning("No papers fetched from ArXiv")
                return
            
            logger.info(f"Fetched {len(papers)} papers from ArXiv")
            
            # Step 2: Get all users for analysis
            users = await unified_database_service.get_all_active_users()
            logger.info(f"Processing daily analysis for {len(users)} users")
            
            # Step 3: Process each user's daily analysis
            successful_analyses = 0
            failed_analyses = 0
            
            for user in users:
                try:
                    user_id = str(user['_id'])
                    
                    # Cleanup previous daily analysis for this user
                    await unified_analysis_service.cleanup_previous_daily_analysis(user_id)
                    
                    # Run analysis for this user
                    result = await unified_analysis_service.analyze_papers_for_user(
                        user_id=user_id,
                        papers=papers,
                        analysis_type='daily_automated'
                    )
                    
                    if result['success']:
                        successful_analyses += 1
                        logger.info(f"Daily analysis completed for user {user_id}")
                    else:
                        failed_analyses += 1
                        logger.error(f"Daily analysis failed for user {user_id}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed_analyses += 1
                    logger.error(f"Error processing daily analysis for user {user.get('_id', 'unknown')}: {e}")
            
            # Step 4: Store pipeline execution stats
            execution_time = (datetime.now() - start_time).total_seconds()
            
            stats = {
                'execution_date': start_time,
                'papers_fetched': len(papers),
                'users_processed': len(users),
                'successful_analyses': successful_analyses,
                'failed_analyses': failed_analyses,
                'execution_time_seconds': execution_time,
                'status': 'completed'
            }
            
            await unified_database_service.store_pipeline_stats(stats)
            
            logger.info(f"Daily analysis pipeline completed in {execution_time:.2f} seconds")
            logger.info(f"Success: {successful_analyses}, Failed: {failed_analyses}")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Daily analysis pipeline failed after {execution_time:.2f} seconds: {e}")
            
            # Store failure stats
            stats = {
                'execution_date': start_time,
                'execution_time_seconds': execution_time,
                'status': 'failed',
                'error': str(e)
            }
            
            await unified_database_service.store_pipeline_stats(stats)
            raise
    
    async def trigger_manual_analysis(self) -> Dict[str, Any]:
        """Manually trigger the daily analysis pipeline"""
        try:
            logger.info("Manual daily analysis pipeline triggered")
            await self.run_daily_analysis_pipeline()
            return {'success': True, 'message': 'Manual analysis completed successfully'}
        except Exception as e:
            logger.error(f"Manual analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ================================
    # USER DAILY DOSE SCHEDULING (from daily_dose_scheduler.py)
    # ================================
    
    async def schedule_user_daily_dose(self, user_id: str, time_str: str, days_back: int = 1) -> Dict[str, Any]:
        """Schedule daily dose for a specific user"""
        try:
            # Parse time
            hour, minute = map(int, time_str.split(':'))
            
            # Validate time
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return {
                    'success': False,
                    'error': 'Invalid time format. Hour must be 0-23, minute must be 0-59'
                }
            
            # Create job ID
            job_id = f'daily_dose_{user_id}'
            
            # Create cron trigger for this user
            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                timezone=unified_config_service.get_cron_schedule()["timezone"]
            )
            
            # Schedule the job
            self.scheduler.add_job(
                func=self._execute_user_daily_dose,
                trigger=trigger,
                args=[user_id, days_back],
                id=job_id,
                name=f'Daily Dose for User {user_id}',
                replace_existing=True,
                max_instances=1
            )
            
            logger.info(f"Daily dose scheduled for user {user_id} at {time_str}")
            
            return {
                'success': True,
                'message': f'Daily dose scheduled for {time_str}',
                'user_id': user_id,
                'time': time_str,
                'job_id': job_id
            }
            
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid time format: {e}. Use HH:MM format'
            }
        except Exception as e:
            logger.error(f"Error scheduling daily dose for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def trigger_user_daily_dose(self, user_id: str, days_back: int = 1) -> Dict[str, Any]:
        """Trigger daily dose execution immediately for a specific user"""
        try:
            result = await self._execute_user_daily_dose(user_id, days_back)
            if result.get("success"):
                return {
                    'success': True,
                    'message': f'Daily dose executed successfully for user {user_id}',
                    'user_id': user_id,
                    'papers_count': result.get('papers_count', 0),
                    'analysis_id': result.get('analysis_id')
                }
            else:
                return {
                    'success': False,
                    'message': result.get('message', 'Unknown error'),
                    'user_id': user_id
                }
        except Exception as e:
            logger.error(f"Failed to execute daily dose for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to execute daily dose: {str(e)}',
                'user_id': user_id
            }
    
    async def _execute_user_daily_dose(self, user_id: str, days_back: int):
        """Execute daily dose for a specific user using the daily dose service."""
        try:
            logger.info(f"Executing daily dose for user {user_id}")
            
            # Use the daily dose service for execution
            daily_dose_service = get_daily_dose_service()
            result = await daily_dose_service.execute_daily_dose(user_id)
            
            if result["success"]:
                logger.info(f"Daily dose completed for user {user_id}: {result.get('papers_count', 0)} papers analyzed")
            else:
                logger.error(f"Daily dose failed for user {user_id}: {result.get('message')}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error executing daily dose for user {user_id}: {e}")
            return {
                "success": False,
                "message": str(e),
                "papers_count": 0
            }
    
    async def _fetch_personalized_papers(self, preferences: Dict[str, Any], start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch papers based on user preferences"""
        try:
            # Get preferred categories
            categories = preferences.get('categories', [])
            keywords = preferences.get('keywords', [])
            authors = preferences.get('authors', [])
            exclude_keywords = preferences.get('exclude_keywords', [])
            max_papers = preferences.get('max_papers_per_day', 10)
            
            # Build search query
            query_parts = []
            
            # Add categories to query
            if categories:
                category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
                query_parts.append(f'({category_query})')
            
            # Add keywords to query
            if keywords:
                keyword_query = ' OR '.join(keywords)
                query_parts.append(f'({keyword_query})')
            
            # Add authors to query
            if authors:
                author_query = ' OR '.join([f'au:{author}' for author in authors])
                query_parts.append(f'({author_query})')
            
            # Combine query parts
            query = ' AND '.join(query_parts) if query_parts else 'cat:cs.*'
            
            # Fetch papers from ArXiv
            papers = await unified_paper_service.search_papers(
                query=query,
                max_results=max_papers * 2,  # Fetch more to allow for filtering
                sort_by='submittedDate',
                sort_order='descending'
            )
            
            # Filter by date range
            filtered_papers = []
            for paper in papers:
                paper_date = datetime.fromisoformat(paper.get('published', '').replace('Z', '+00:00'))
                if start_date <= paper_date <= end_date:
                    # Check exclude keywords
                    if exclude_keywords:
                        title_and_abstract = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
                        if any(exclude_kw.lower() in title_and_abstract for exclude_kw in exclude_keywords):
                            continue
                    
                    filtered_papers.append(paper)
                    
                    if len(filtered_papers) >= max_papers:
                        break
            
            return filtered_papers
            
        except Exception as e:
            logger.error(f"Error fetching personalized papers: {e}")
            return []
    
    async def cancel_user_daily_dose(self, user_id: str) -> Dict[str, Any]:
        """Cancel daily dose for a specific user"""
        try:
            job_id = f'daily_dose_{user_id}'
            
            # Remove the job
            self.scheduler.remove_job(job_id)
            
            logger.info(f"Daily dose cancelled for user {user_id}")
            
            return {
                'success': True,
                'message': f'Daily dose cancelled for user {user_id}',
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error cancelling daily dose for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_user_daily_dose_status(self, user_id: str) -> Dict[str, Any]:
        """Get daily dose status for a specific user"""
        try:
            job_id = f'daily_dose_{user_id}'
            
            try:
                job = self.scheduler.get_job(job_id)
                if job:
                    return {
                        'success': True,
                        'scheduled': True,
                        'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                        'trigger': str(job.trigger)
                    }
                else:
                    return {
                        'success': True,
                        'scheduled': False,
                        'message': 'No daily dose scheduled for this user'
                    }
            except Exception:
                return {
                    'success': True,
                    'scheduled': False,
                    'message': 'No daily dose scheduled for this user'
                }
                
        except Exception as e:
            logger.error(f"Error getting daily dose status for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # ================================
    # STATISTICS AND MONITORING
    # ================================
    
    async def get_pipeline_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get pipeline execution statistics"""
        try:
            stats = await unified_database_service.get_pipeline_stats(days)
            return {
                'success': True,
                'stats': stats
            }
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_user_dose_history(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user's daily dose execution history"""
        try:
            # Get user's daily analyses from the last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            analyses = await unified_database_service.get_user_analyses(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                analysis_type='daily_dose'
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'history': analyses,
                'count': len(analyses)
            }
            
        except Exception as e:
            logger.error(f"Error getting dose history for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # ================================
    # UTILITY FUNCTIONS
    # ================================
    
    async def reschedule_daily_analysis(self, hour: int, minute: int) -> Dict[str, Any]:
        """Reschedule the daily analysis pipeline"""
        try:
            # Remove existing job
            try:
                self.scheduler.remove_job('daily_analysis_pipeline')
            except Exception:
                pass  # Job might not exist
            
            # Create new trigger
            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                timezone=unified_config_service.get_cron_schedule()["timezone"]
            )
            
            # Schedule new job
            self.scheduler.add_job(
                func=self.run_daily_analysis_pipeline,
                trigger=trigger,
                id='daily_analysis_pipeline',
                name='Daily ArXiv Analysis Pipeline',
                replace_existing=True,
                max_instances=1
            )
            
            logger.info(f"Daily analysis rescheduled for {hour:02d}:{minute:02d}")
            
            return {
                'success': True,
                'message': f'Daily analysis rescheduled for {hour:02d}:{minute:02d}'
            }
            
        except Exception as e:
            logger.error(f"Error rescheduling daily analysis: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# ================================
# SINGLETON INSTANCE
# ================================

# Create singleton instance
unified_scheduler = UnifiedSchedulerService()

# Export commonly used functions for backwards compatibility
async def start_scheduler():
    """Start the unified scheduler service"""
    return await unified_scheduler.start()

async def stop_scheduler():
    """Stop the unified scheduler service"""
    return await unified_scheduler.stop()

async def run_daily_analysis_pipeline():
    """Run the daily analysis pipeline manually"""
    return await unified_scheduler.trigger_manual_analysis()

async def schedule_user_daily_dose(user_id: str, time_str: str, days_back: int = 1):
    """Schedule daily dose for a user"""
    return await unified_scheduler.schedule_user_daily_dose(user_id, time_str, days_back)

async def trigger_user_daily_dose(user_id: str, days_back: int = 1):
    """Trigger daily dose execution immediately for a user"""
    return await unified_scheduler.trigger_user_daily_dose(user_id, days_back)

# For backwards compatibility
daily_scheduler = unified_scheduler
daily_dose_scheduler = unified_scheduler

# Export all public functions
__all__ = [
    'UnifiedSchedulerService',
    'unified_scheduler',
    'daily_scheduler',
    'daily_dose_scheduler',
    'start_scheduler',
    'stop_scheduler',
    'run_daily_analysis_pipeline',
    'schedule_user_daily_dose'
]
