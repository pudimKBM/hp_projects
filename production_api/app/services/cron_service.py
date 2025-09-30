"""
Cron Job Scheduling Service

This service manages automated scheduling of scraping jobs using cron-like functionality.
It provides job status tracking, execution history logging, and failure handling.
"""

import logging
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from flask import current_app
from sqlalchemy.exc import SQLAlchemyError

from ..models import db, ScrapingJob, SystemHealth
from .scraper_service import ScraperService


logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Enumeration for job execution status."""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CronJobConfig:
    """Configuration for a cron job."""
    name: str
    schedule_expression: str  # Cron-like expression or schedule library format
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    enabled: bool = True
    max_retries: int = 3
    timeout_minutes: int = 60
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class CronJobManager:
    """Manager for scheduling and executing cron jobs."""
    
    def __init__(self):
        """Initialize the cron job manager."""
        self.jobs = {}
        self.running_jobs = {}
        self.job_history = []
        self.scheduler_thread = None
        self.is_running = False
        self._lock = threading.Lock()
        
    def add_job(self, job_config: CronJobConfig) -> bool:
        """
        Add a new cron job to the scheduler.
        
        Args:
            job_config: Configuration for the cron job
            
        Returns:
            bool: True if job was added successfully
        """
        try:
            with self._lock:
                if job_config.name in self.jobs:
                    logger.warning(f"Job '{job_config.name}' already exists, updating configuration")
                
                self.jobs[job_config.name] = job_config
                
                # Schedule the job if enabled
                if job_config.enabled:
                    self._schedule_job(job_config)
                
                logger.info(f"Added cron job: {job_config.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add job '{job_config.name}': {e}")
            return False
    
    def remove_job(self, job_name: str) -> bool:
        """
        Remove a cron job from the scheduler.
        
        Args:
            job_name: Name of the job to remove
            
        Returns:
            bool: True if job was removed successfully
        """
        try:
            with self._lock:
                if job_name not in self.jobs:
                    logger.warning(f"Job '{job_name}' not found")
                    return False
                
                # Cancel job if running
                if job_name in self.running_jobs:
                    self._cancel_running_job(job_name)
                
                # Remove from schedule
                schedule.clear(job_name)
                
                # Remove from jobs dict
                del self.jobs[job_name]
                
                logger.info(f"Removed cron job: {job_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove job '{job_name}': {e}")
            return False
    
    def enable_job(self, job_name: str) -> bool:
        """Enable a cron job."""
        try:
            with self._lock:
                if job_name not in self.jobs:
                    logger.error(f"Job '{job_name}' not found")
                    return False
                
                job_config = self.jobs[job_name]
                job_config.enabled = True
                self._schedule_job(job_config)
                
                logger.info(f"Enabled cron job: {job_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to enable job '{job_name}': {e}")
            return False
    
    def disable_job(self, job_name: str) -> bool:
        """Disable a cron job."""
        try:
            with self._lock:
                if job_name not in self.jobs:
                    logger.error(f"Job '{job_name}' not found")
                    return False
                
                job_config = self.jobs[job_name]
                job_config.enabled = False
                
                # Cancel if running
                if job_name in self.running_jobs:
                    self._cancel_running_job(job_name)
                
                # Remove from schedule
                schedule.clear(job_name)
                
                logger.info(f"Disabled cron job: {job_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to disable job '{job_name}': {e}")
            return False
    
    def _schedule_job(self, job_config: CronJobConfig):
        """Schedule a job using the schedule library."""
        try:
            # Parse schedule expression and create schedule
            schedule_expr = job_config.schedule_expression.strip()
            
            # Handle different schedule formats
            if schedule_expr.startswith('every'):
                # Handle "every X minutes/hours/days" format
                self._parse_every_schedule(job_config, schedule_expr)
            elif len(schedule_expr.split()) == 5:
                # Handle cron format "0 */6 * * *"
                self._parse_cron_schedule(job_config, schedule_expr)
            else:
                raise ValueError(f"Unsupported schedule format: {schedule_expr}")
            
        except Exception as e:
            logger.error(f"Failed to schedule job '{job_config.name}': {e}")
            raise
    
    def _parse_every_schedule(self, job_config: CronJobConfig, schedule_expr: str):
        """Parse 'every X unit' schedule format."""
        parts = schedule_expr.lower().split()
        
        if len(parts) < 3:
            raise ValueError(f"Invalid 'every' schedule format: {schedule_expr}")
        
        interval = int(parts[1])
        unit = parts[2].rstrip('s')  # Remove plural 's'
        
        job_func = lambda: self._execute_job(job_config.name)
        
        if unit == 'minute':
            schedule.every(interval).minutes.do(job_func).tag(job_config.name)
        elif unit == 'hour':
            schedule.every(interval).hours.do(job_func).tag(job_config.name)
        elif unit == 'day':
            schedule.every(interval).days.do(job_func).tag(job_config.name)
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
    
    def _parse_cron_schedule(self, job_config: CronJobConfig, cron_expr: str):
        """Parse cron format schedule (simplified version)."""
        parts = cron_expr.split()
        
        if len(parts) != 5:
            raise ValueError(f"Invalid cron format: {cron_expr}")
        
        minute, hour, day, month, weekday = parts
        
        job_func = lambda: self._execute_job(job_config.name)
        
        # Handle common patterns
        if hour.startswith('*/'):
            # Every N hours
            interval = int(hour[2:])
            schedule.every(interval).hours.do(job_func).tag(job_config.name)
        elif minute.startswith('*/'):
            # Every N minutes
            interval = int(minute[2:])
            schedule.every(interval).minutes.do(job_func).tag(job_config.name)
        elif hour.isdigit() and minute.isdigit():
            # Specific time daily
            time_str = f"{hour.zfill(2)}:{minute.zfill(2)}"
            schedule.every().day.at(time_str).do(job_func).tag(job_config.name)
        else:
            # Default to every 6 hours for unsupported patterns
            logger.warning(f"Unsupported cron pattern '{cron_expr}', defaulting to every 6 hours")
            schedule.every(6).hours.do(job_func).tag(job_config.name)
    
    def _execute_job(self, job_name: str):
        """Execute a scheduled job with error handling and logging."""
        if job_name not in self.jobs:
            logger.error(f"Job '{job_name}' not found during execution")
            return
        
        job_config = self.jobs[job_name]
        
        # Check if job is already running
        if job_name in self.running_jobs:
            logger.warning(f"Job '{job_name}' is already running, skipping execution")
            return
        
        # Create execution record
        execution_id = f"{job_name}_{int(time.time())}"
        execution_record = {
            'execution_id': execution_id,
            'job_name': job_name,
            'status': JobStatus.RUNNING,
            'started_at': datetime.utcnow(),
            'completed_at': None,
            'result': None,
            'error': None,
            'retry_count': 0
        }
        
        self.running_jobs[job_name] = execution_record
        self.job_history.append(execution_record)
        
        logger.info(f"Starting execution of job '{job_name}' (ID: {execution_id})")
        
        # Execute job in separate thread to avoid blocking scheduler
        thread = threading.Thread(
            target=self._run_job_with_timeout,
            args=(job_config, execution_record),
            daemon=True
        )
        thread.start()
    
    def _run_job_with_timeout(self, job_config: CronJobConfig, execution_record: Dict):
        """Run job with timeout and retry logic."""
        job_name = job_config.name
        max_retries = job_config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                execution_record['retry_count'] = attempt
                
                # Execute the job function
                result = job_config.function(*job_config.args, **job_config.kwargs)
                
                # Mark as completed
                execution_record['status'] = JobStatus.COMPLETED
                execution_record['completed_at'] = datetime.utcnow()
                execution_record['result'] = result
                
                logger.info(f"Job '{job_name}' completed successfully (attempt {attempt + 1})")
                break
                
            except Exception as e:
                error_msg = f"Job '{job_name}' failed on attempt {attempt + 1}: {e}"
                logger.error(error_msg)
                
                execution_record['error'] = str(e)
                
                if attempt < max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = min(300, 2 ** attempt * 60)  # Max 5 minutes
                    logger.info(f"Retrying job '{job_name}' in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    # Mark as failed after all retries
                    execution_record['status'] = JobStatus.FAILED
                    execution_record['completed_at'] = datetime.utcnow()
                    
                    # Log system health issue
                    self._log_job_failure(job_name, str(e))
        
        # Clean up running jobs
        with self._lock:
            if job_name in self.running_jobs:
                del self.running_jobs[job_name]
    
    def _cancel_running_job(self, job_name: str):
        """Cancel a running job."""
        if job_name in self.running_jobs:
            execution_record = self.running_jobs[job_name]
            execution_record['status'] = JobStatus.CANCELLED
            execution_record['completed_at'] = datetime.utcnow()
            
            del self.running_jobs[job_name]
            logger.info(f"Cancelled running job: {job_name}")
    
    def _log_job_failure(self, job_name: str, error_message: str):
        """Log job failure to system health monitoring."""
        try:
            health_record = SystemHealth(
                component='cron_scheduler',
                status='unhealthy',
                metrics={'failed_job': job_name},
                message=f"Cron job '{job_name}' failed: {error_message}"
            )
            
            db.session.add(health_record)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to log job failure to system health: {e}")
    
    def start_scheduler(self):
        """Start the cron job scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        def run_scheduler():
            logger.info("Cron job scheduler started")
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(5)
            
            logger.info("Cron job scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the cron job scheduler."""
        self.is_running = False
        
        # Cancel all running jobs
        with self._lock:
            for job_name in list(self.running_jobs.keys()):
                self._cancel_running_job(job_name)
        
        # Clear all scheduled jobs
        schedule.clear()
        
        logger.info("Cron job scheduler stopped")
    
    def get_job_status(self, job_name: str) -> Optional[Dict]:
        """Get status information for a specific job."""
        if job_name not in self.jobs:
            return None
        
        job_config = self.jobs[job_name]
        
        # Get recent executions
        recent_executions = [
            record for record in self.job_history[-10:]
            if record['job_name'] == job_name
        ]
        
        return {
            'name': job_name,
            'enabled': job_config.enabled,
            'schedule': job_config.schedule_expression,
            'is_running': job_name in self.running_jobs,
            'recent_executions': recent_executions,
            'next_run': self._get_next_run_time(job_name)
        }
    
    def get_all_jobs_status(self) -> List[Dict]:
        """Get status information for all jobs."""
        return [
            self.get_job_status(job_name)
            for job_name in self.jobs.keys()
        ]
    
    def _get_next_run_time(self, job_name: str) -> Optional[str]:
        """Get the next scheduled run time for a job."""
        try:
            jobs = schedule.get_jobs(job_name)
            if jobs:
                next_run = jobs[0].next_run
                return next_run.isoformat() if next_run else None
        except Exception:
            pass
        
        return None
    
    def cleanup_job_history(self, max_records: int = 1000):
        """Clean up old job execution history."""
        if len(self.job_history) > max_records:
            # Keep only the most recent records
            self.job_history = self.job_history[-max_records:]
            logger.info(f"Cleaned up job history, keeping {max_records} records")


# Global cron job manager instance
cron_manager = CronJobManager()


def setup_default_scraping_jobs():
    """Set up default scraping jobs based on configuration."""
    try:
        # Get schedule from config
        schedule_expr = current_app.config.get('DEFAULT_CRON_SCHEDULE', '0 */6 * * *')
        
        # Create scraping job configuration
        scraping_job = CronJobConfig(
            name='scheduled_scraping',
            schedule_expression=schedule_expr,
            function=run_scheduled_scraping_job,
            enabled=True,
            max_retries=3,
            timeout_minutes=120
        )
        
        # Add to manager
        cron_manager.add_job(scraping_job)
        
        logger.info("Default scraping jobs configured")
        
    except Exception as e:
        logger.error(f"Failed to setup default scraping jobs: {e}")


def run_scheduled_scraping_job():
    """Execute a scheduled scraping job."""
    try:
        logger.info("Starting scheduled scraping job")
        
        scraper_service = ScraperService()
        products_found, products_stored, errors = scraper_service.run_scraping_job("scheduled")
        
        result = {
            'products_found': products_found,
            'products_stored': products_stored,
            'errors': errors,
            'success': len(errors) == 0
        }
        
        logger.info(f"Scheduled scraping completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Scheduled scraping job failed: {e}")
        raise


def start_cron_scheduler():
    """Start the global cron job scheduler."""
    setup_default_scraping_jobs()
    cron_manager.start_scheduler()


def stop_cron_scheduler():
    """Stop the global cron job scheduler."""
    cron_manager.stop_scheduler()


def get_scheduler_status() -> Dict:
    """Get overall scheduler status."""
    return {
        'is_running': cron_manager.is_running,
        'total_jobs': len(cron_manager.jobs),
        'running_jobs': len(cron_manager.running_jobs),
        'jobs': cron_manager.get_all_jobs_status()
    }