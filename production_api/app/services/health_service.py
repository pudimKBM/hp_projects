"""
System Health Monitoring Service

This service provides comprehensive health checking for all system components:
- Scraper service health and performance
- ML model availability and validation
- Database connectivity and integrity
- System resource monitoring
- Performance metrics collection
"""

import os
import sys
import psutil
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from flask import current_app
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from ..models import db, Product, Classification, ScrapingJob, SystemHealth
from ..utils.database import validate_database_integrity
from .ml_service import MLService
from .scraper_service import ScraperService

logger = logging.getLogger(__name__)


class HealthService:
    """
    Comprehensive system health monitoring service
    """
    
    def __init__(self):
        """Initialize the health service"""
        self.ml_service = None
        self.scraper_service = None
        self._last_full_check = None
        self._cached_results = {}
        self._cache_duration = timedelta(minutes=5)  # Cache results for 5 minutes
        
    def get_system_health(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive system health status
        
        Args:
            force_refresh: Force refresh of cached results
            
        Returns:
            Dictionary with complete system health information
        """
        now = datetime.utcnow()
        
        # Check if we can use cached results
        if (not force_refresh and 
            self._last_full_check and 
            now - self._last_full_check < self._cache_duration and
            self._cached_results):
            logger.debug("Returning cached health results")
            return self._cached_results
        
        logger.info("Performing full system health check")
        
        # Perform comprehensive health check
        health_status = {
            'status': 'healthy',
            'timestamp': now.isoformat(),
            'components': {},
            'performance_metrics': {},
            'system_resources': {},
            'alerts': []
        }
        
        # Check individual components
        try:
            # Database health
            db_health = self.check_database_health()
            health_status['components']['database'] = db_health
            
            # ML Model health
            ml_health = self.check_ml_model_health()
            health_status['components']['ml_model'] = ml_health
            
            # Scraper health
            scraper_health = self.check_scraper_health()
            health_status['components']['scraper'] = scraper_health
            
            # System resources
            resource_health = self.check_system_resources()
            health_status['system_resources'] = resource_health
            
            # Performance metrics
            performance_metrics = self.collect_performance_metrics()
            health_status['performance_metrics'] = performance_metrics
            
            # Determine overall status
            component_statuses = [
                db_health.get('status', 'unhealthy'),
                ml_health.get('status', 'unhealthy'),
                scraper_health.get('status', 'unhealthy')
            ]
            
            if all(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'healthy'
            elif any(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'degraded'
            else:
                health_status['status'] = 'unhealthy'
            
            # Collect alerts from components
            alerts = []
            for component_name, component_health in health_status['components'].items():
                if component_health.get('alerts'):
                    for alert in component_health['alerts']:
                        alerts.append(f"{component_name}: {alert}")
            
            health_status['alerts'] = alerts
            
        except Exception as e:
            logger.error(f"Error during system health check: {e}")
            health_status['status'] = 'error'
            health_status['error'] = str(e)
        
        # Cache results and store in database
        self._cached_results = health_status
        self._last_full_check = now
        self._store_health_record(health_status)
        
        return health_status
    
    def check_database_health(self) -> Dict[str, Any]:
        """
        Check database connectivity, integrity, and performance
        
        Returns:
            Dictionary with database health information
        """
        health_info = {
            'status': 'healthy',
            'connection_status': 'unknown',
            'integrity_status': 'unknown',
            'performance_metrics': {},
            'statistics': {},
            'alerts': [],
            'last_check': datetime.utcnow().isoformat()
        }
        
        try:
            # Test basic connectivity
            start_time = datetime.utcnow()
            
            # Simple query to test connection
            result = db.session.execute(text('SELECT 1')).fetchone()
            if result and result[0] == 1:
                health_info['connection_status'] = 'connected'
                
                # Measure query performance
                query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                health_info['performance_metrics']['connection_test_ms'] = query_time
                
                if query_time > 1000:  # More than 1 second
                    health_info['alerts'].append('Database connection is slow')
            else:
                health_info['connection_status'] = 'failed'
                health_info['status'] = 'unhealthy'
                health_info['alerts'].append('Database connection test failed')
            
            # Check database integrity
            integrity_results = validate_database_integrity(db)
            health_info['integrity_status'] = 'valid' if integrity_results['valid'] else 'invalid'
            health_info['statistics'] = integrity_results['statistics']
            
            if not integrity_results['valid']:
                health_info['status'] = 'degraded'
                health_info['alerts'].extend(integrity_results['issues'])
            
            # Check database file size and growth
            db_path = current_app.config.get('DATABASE_PATH', 'data/production.db')
            if os.path.exists(db_path):
                db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
                health_info['performance_metrics']['database_size_mb'] = round(db_size_mb, 2)
                
                # Alert if database is getting large
                if db_size_mb > 1000:  # More than 1GB
                    health_info['alerts'].append(f'Database size is large: {db_size_mb:.1f}MB')
            
            # Check connection pool status
            engine = db.engine
            pool = engine.pool
            health_info['performance_metrics']['connection_pool'] = {
                'size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            }
            
            # Alert if connection pool is exhausted
            if pool.checkedout() >= pool.size():
                health_info['alerts'].append('Database connection pool is exhausted')
            
        except SQLAlchemyError as e:
            logger.error(f"Database health check failed: {e}")
            health_info['status'] = 'unhealthy'
            health_info['connection_status'] = 'error'
            health_info['alerts'].append(f'Database error: {str(e)}')
        except Exception as e:
            logger.error(f"Unexpected error in database health check: {e}")
            health_info['status'] = 'error'
            health_info['alerts'].append(f'Health check error: {str(e)}')
        
        return health_info
    
    def check_ml_model_health(self) -> Dict[str, Any]:
        """
        Check ML model availability, validation, and performance
        
        Returns:
            Dictionary with ML model health information
        """
        health_info = {
            'status': 'healthy',
            'model_status': 'unknown',
            'model_info': {},
            'performance_metrics': {},
            'alerts': [],
            'last_check': datetime.utcnow().isoformat()
        }
        
        try:
            # Initialize ML service if needed
            if not self.ml_service:
                models_path = current_app.config.get('MODELS_PATH', 'models')
                self.ml_service = MLService(models_path)
            
            # Get service status
            service_status = self.ml_service.get_service_status()
            health_info['model_status'] = 'loaded' if service_status['service_healthy'] else 'error'
            health_info['model_info'] = {
                'loaded_models': service_status['loaded_models'],
                'fallback_models_available': service_status['fallback_models_available'],
                'models_base_path': service_status['models_base_path']
            }
            
            # Check if any models are loaded
            if not service_status['loaded_models']:
                # Try to load default model
                try:
                    default_model = current_app.config.get('DEFAULT_MODEL_NAME', 'best_model')
                    load_result = self.ml_service.load_primary_model(default_model)
                    
                    if load_result['success']:
                        health_info['model_status'] = 'loaded'
                        health_info['model_info']['default_model_loaded'] = True
                    else:
                        health_info['status'] = 'unhealthy'
                        health_info['alerts'].append('No models loaded and default model failed to load')
                        
                except Exception as e:
                    health_info['status'] = 'unhealthy'
                    health_info['alerts'].append(f'Failed to load default model: {str(e)}')
            
            # Validate loaded models
            for model_name, model_status in service_status['loaded_models'].items():
                if not model_status['healthy']:
                    health_info['status'] = 'degraded'
                    health_info['alerts'].extend([
                        f"Model {model_name} is unhealthy: {', '.join(model_status['issues'])}"
                    ])
            
            # Check model performance metrics
            try:
                # Get recent classification performance
                recent_classifications = Classification.query.filter(
                    Classification.classified_at >= datetime.utcnow() - timedelta(hours=24)
                ).all()
                
                if recent_classifications:
                    processing_times = [c.processing_time_ms for c in recent_classifications if c.processing_time_ms]
                    confidence_scores = [c.confidence_score for c in recent_classifications if c.confidence_score]
                    
                    if processing_times:
                        health_info['performance_metrics']['avg_processing_time_ms'] = sum(processing_times) / len(processing_times)
                        health_info['performance_metrics']['max_processing_time_ms'] = max(processing_times)
                        
                        # Alert if processing is slow
                        avg_time = health_info['performance_metrics']['avg_processing_time_ms']
                        if avg_time > 5000:  # More than 5 seconds
                            health_info['alerts'].append(f'ML processing is slow: {avg_time:.0f}ms average')
                    
                    if confidence_scores:
                        health_info['performance_metrics']['avg_confidence_score'] = sum(confidence_scores) / len(confidence_scores)
                        health_info['performance_metrics']['low_confidence_rate'] = len([s for s in confidence_scores if s < 0.7]) / len(confidence_scores)
                        
                        # Alert if confidence is consistently low
                        low_confidence_rate = health_info['performance_metrics']['low_confidence_rate']
                        if low_confidence_rate > 0.5:  # More than 50% low confidence
                            health_info['alerts'].append(f'High rate of low-confidence predictions: {low_confidence_rate:.1%}')
                    
                    health_info['performance_metrics']['classifications_last_24h'] = len(recent_classifications)
                else:
                    health_info['performance_metrics']['classifications_last_24h'] = 0
                    health_info['alerts'].append('No recent classifications found')
                    
            except Exception as e:
                logger.warning(f"Error collecting ML performance metrics: {e}")
                health_info['alerts'].append('Could not collect performance metrics')
            
        except Exception as e:
            logger.error(f"ML model health check failed: {e}")
            health_info['status'] = 'error'
            health_info['model_status'] = 'error'
            health_info['alerts'].append(f'ML health check error: {str(e)}')
        
        return health_info
    
    def check_scraper_health(self) -> Dict[str, Any]:
        """
        Check scraper service health and recent performance
        
        Returns:
            Dictionary with scraper health information
        """
        health_info = {
            'status': 'healthy',
            'scraper_status': 'unknown',
            'recent_jobs': [],
            'performance_metrics': {},
            'alerts': [],
            'last_check': datetime.utcnow().isoformat()
        }
        
        try:
            # Initialize scraper service if needed
            if not self.scraper_service:
                self.scraper_service = ScraperService()
            
            # Get recent scraping jobs
            recent_jobs = self.scraper_service.get_recent_jobs(limit=5)
            health_info['recent_jobs'] = recent_jobs
            
            if recent_jobs:
                # Analyze recent job performance
                completed_jobs = [job for job in recent_jobs if job['status'] == 'completed']
                failed_jobs = [job for job in recent_jobs if job['status'] == 'failed']
                running_jobs = [job for job in recent_jobs if job['status'] == 'running']
                
                # Calculate success rate
                if len(recent_jobs) > 0:
                    success_rate = len(completed_jobs) / len(recent_jobs)
                    health_info['performance_metrics']['success_rate'] = success_rate
                    
                    if success_rate < 0.8:  # Less than 80% success
                        health_info['status'] = 'degraded'
                        health_info['alerts'].append(f'Low scraper success rate: {success_rate:.1%}')
                
                # Check for stuck jobs
                stuck_jobs = []
                for job in running_jobs:
                    started_at = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                    if datetime.utcnow().replace(tzinfo=started_at.tzinfo) - started_at > timedelta(hours=2):
                        stuck_jobs.append(job)
                
                if stuck_jobs:
                    health_info['status'] = 'degraded'
                    health_info['alerts'].append(f'{len(stuck_jobs)} scraping jobs appear to be stuck')
                
                # Get performance metrics from completed jobs
                if completed_jobs:
                    products_per_job = [job.get('products_found', 0) for job in completed_jobs]
                    processing_times = []
                    
                    for job in completed_jobs:
                        if job.get('started_at') and job.get('completed_at'):
                            start = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                            end = datetime.fromisoformat(job['completed_at'].replace('Z', '+00:00'))
                            processing_times.append((end - start).total_seconds())
                    
                    if products_per_job:
                        health_info['performance_metrics']['avg_products_per_job'] = sum(products_per_job) / len(products_per_job)
                        health_info['performance_metrics']['total_products_scraped'] = sum(products_per_job)
                    
                    if processing_times:
                        health_info['performance_metrics']['avg_job_duration_seconds'] = sum(processing_times) / len(processing_times)
                
                # Check last successful scraping
                last_successful = None
                for job in recent_jobs:
                    if job['status'] == 'completed':
                        last_successful = job
                        break
                
                if last_successful:
                    last_run = datetime.fromisoformat(last_successful['completed_at'].replace('Z', '+00:00'))
                    hours_since_last = (datetime.utcnow().replace(tzinfo=last_run.tzinfo) - last_run).total_seconds() / 3600
                    health_info['performance_metrics']['hours_since_last_success'] = hours_since_last
                    
                    # Alert if no successful scraping in a while
                    if hours_since_last > 24:  # More than 24 hours
                        health_info['status'] = 'degraded'
                        health_info['alerts'].append(f'No successful scraping in {hours_since_last:.1f} hours')
                else:
                    health_info['status'] = 'degraded'
                    health_info['alerts'].append('No recent successful scraping jobs found')
                
                health_info['scraper_status'] = 'active'
            else:
                health_info['scraper_status'] = 'inactive'
                health_info['alerts'].append('No recent scraping jobs found')
            
            # Check scraper configuration
            config_issues = self._validate_scraper_config()
            if config_issues:
                health_info['alerts'].extend(config_issues)
                if health_info['status'] == 'healthy':
                    health_info['status'] = 'degraded'
            
        except Exception as e:
            logger.error(f"Scraper health check failed: {e}")
            health_info['status'] = 'error'
            health_info['scraper_status'] = 'error'
            health_info['alerts'].append(f'Scraper health check error: {str(e)}')
        
        return health_info
    
    def check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resource usage (CPU, memory, disk)
        
        Returns:
            Dictionary with system resource information
        """
        resource_info = {
            'cpu_usage_percent': 0,
            'memory_usage_percent': 0,
            'disk_usage_percent': 0,
            'available_memory_mb': 0,
            'available_disk_gb': 0,
            'alerts': [],
            'last_check': datetime.utcnow().isoformat()
        }
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            resource_info['cpu_usage_percent'] = cpu_percent
            
            if cpu_percent > 90:
                resource_info['alerts'].append(f'High CPU usage: {cpu_percent:.1f}%')
            
            # Memory usage
            memory = psutil.virtual_memory()
            resource_info['memory_usage_percent'] = memory.percent
            resource_info['available_memory_mb'] = memory.available / (1024 * 1024)
            
            if memory.percent > 90:
                resource_info['alerts'].append(f'High memory usage: {memory.percent:.1f}%')
            
            # Disk usage
            disk = psutil.disk_usage('/')
            resource_info['disk_usage_percent'] = (disk.used / disk.total) * 100
            resource_info['available_disk_gb'] = disk.free / (1024 * 1024 * 1024)
            
            if resource_info['disk_usage_percent'] > 90:
                resource_info['alerts'].append(f'High disk usage: {resource_info["disk_usage_percent"]:.1f}%')
            
            # Process-specific information
            current_process = psutil.Process()
            resource_info['process_memory_mb'] = current_process.memory_info().rss / (1024 * 1024)
            resource_info['process_cpu_percent'] = current_process.cpu_percent()
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            resource_info['alerts'].append(f'Resource monitoring error: {str(e)}')
        
        return resource_info
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect overall system performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'uptime_hours': 0,
            'total_products': 0,
            'total_classifications': 0,
            'classifications_per_hour': 0,
            'accuracy_last_24h': 0,
            'avg_classification_time_ms': 0,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        try:
            # System uptime (approximate based on first health record)
            first_health = SystemHealth.query.order_by(SystemHealth.timestamp.asc()).first()
            if first_health:
                uptime = datetime.utcnow() - first_health.timestamp
                metrics['uptime_hours'] = uptime.total_seconds() / 3600
            
            # Database statistics
            metrics['total_products'] = Product.query.count()
            metrics['total_classifications'] = Classification.query.count()
            
            # Recent performance
            last_24h = datetime.utcnow() - timedelta(hours=24)
            recent_classifications = Classification.query.filter(
                Classification.classified_at >= last_24h
            ).all()
            
            if recent_classifications:
                metrics['classifications_per_hour'] = len(recent_classifications) / 24
                
                # Calculate average processing time
                processing_times = [c.processing_time_ms for c in recent_classifications if c.processing_time_ms]
                if processing_times:
                    metrics['avg_classification_time_ms'] = sum(processing_times) / len(processing_times)
                
                # Estimate accuracy (this would need ground truth data in practice)
                # For now, use high-confidence predictions as a proxy
                high_confidence = len([c for c in recent_classifications if c.confidence_score and c.confidence_score > 0.8])
                if recent_classifications:
                    metrics['accuracy_last_24h'] = high_confidence / len(recent_classifications)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def _validate_scraper_config(self) -> List[str]:
        """
        Validate scraper configuration
        
        Returns:
            List of configuration issues
        """
        issues = []
        
        try:
            # Check required configuration
            required_configs = ['DEFAULT_SEARCH_TERMS', 'MAX_PAGES_PER_TERM', 'SCRAPER_DELAY']
            for config_key in required_configs:
                if not current_app.config.get(config_key):
                    issues.append(f'Missing scraper configuration: {config_key}')
            
            # Validate search terms
            search_terms = current_app.config.get('DEFAULT_SEARCH_TERMS', [])
            if not search_terms or len(search_terms) == 0:
                issues.append('No default search terms configured')
            
            # Check if models directory exists
            models_path = current_app.config.get('MODELS_PATH', 'models')
            if not os.path.exists(models_path):
                issues.append(f'Models directory does not exist: {models_path}')
            
        except Exception as e:
            issues.append(f'Error validating configuration: {str(e)}')
        
        return issues
    
    def _store_health_record(self, health_status: Dict[str, Any]) -> None:
        """
        Store health check results in database
        
        Args:
            health_status: Health status dictionary to store
        """
        try:
            health_record = SystemHealth(
                component='system',
                status=health_status['status'],
                metrics=health_status,
                message=f"System health check - Status: {health_status['status']}"
            )
            
            db.session.add(health_record)
            db.session.commit()
            
            # Log health check event
            from production_api.app.utils.logging import log_health_check_event
            log_health_check_event(
                component='system',
                status=health_status['status'],
                issues=health_status.get('alerts', []),
                metrics=health_status.get('performance_metrics', {})
            )
            
        except Exception as e:
            logger.error(f"Failed to store health record: {e}")
            db.session.rollback()
    
    def get_health_history(self, hours: int = 24, component: str = None) -> List[Dict[str, Any]]:
        """
        Get health check history
        
        Args:
            hours: Number of hours of history to retrieve
            component: Specific component to filter by
            
        Returns:
            List of health records
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            query = SystemHealth.query.filter(SystemHealth.timestamp >= cutoff_time)
            
            if component:
                query = query.filter(SystemHealth.component == component)
            
            records = query.order_by(SystemHealth.timestamp.desc()).all()
            
            return [record.to_dict() for record in records]
            
        except Exception as e:
            logger.error(f"Error retrieving health history: {e}")
            return []
    
    def cleanup_old_health_records(self, days_to_keep: int = 7) -> int:
        """
        Clean up old health records
        
        Args:
            days_to_keep: Number of days of health records to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            deleted_count = SystemHealth.query.filter(
                SystemHealth.timestamp < cutoff_date
            ).delete()
            
            db.session.commit()
            logger.info(f"Cleaned up {deleted_count} old health records")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up health records: {e}")
            db.session.rollback()
            return 0


# Convenience functions
def get_system_health(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get system health status
    
    Args:
        force_refresh: Force refresh of cached results
        
    Returns:
        System health dictionary
    """
    service = HealthService()
    return service.get_system_health(force_refresh)


def check_component_health(component: str) -> Dict[str, Any]:
    """
    Check health of a specific component
    
    Args:
        component: Component name ('database', 'ml_model', 'scraper')
        
    Returns:
        Component health dictionary
    """
    service = HealthService()
    
    if component == 'database':
        return service.check_database_health()
    elif component == 'ml_model':
        return service.check_ml_model_health()
    elif component == 'scraper':
        return service.check_scraper_health()
    else:
        return {'status': 'error', 'error': f'Unknown component: {component}'}