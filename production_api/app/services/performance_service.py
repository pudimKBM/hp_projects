"""
Performance Monitoring Service

This service provides comprehensive performance tracking for:
- Classification performance (accuracy, processing time)
- System resource monitoring (memory, CPU usage)
- Structured logging for all system operations and errors
- Performance metrics collection and analysis
"""

import os
import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import deque, defaultdict
from contextlib import contextmanager

from flask import current_app, g
from sqlalchemy.exc import SQLAlchemyError

from ..models import db, Classification, Product, ScrapingJob, SystemHealth

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics with thread-safe operations"""
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize performance metrics container
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self._lock = threading.Lock()
        
        # Classification metrics
        self.classification_times = deque(maxlen=max_samples)
        self.confidence_scores = deque(maxlen=max_samples)
        self.prediction_counts = defaultdict(int)
        
        # System resource metrics
        self.cpu_usage_history = deque(maxlen=max_samples)
        self.memory_usage_history = deque(maxlen=max_samples)
        self.disk_usage_history = deque(maxlen=max_samples)
        
        # API performance metrics
        self.api_response_times = deque(maxlen=max_samples)
        self.api_error_counts = defaultdict(int)
        self.api_request_counts = defaultdict(int)
        
        # Scraper performance metrics
        self.scraper_job_times = deque(maxlen=max_samples)
        self.scraper_success_rates = deque(maxlen=max_samples)
        
        # Timestamps for metrics
        self.last_updated = datetime.utcnow()
        self.start_time = datetime.utcnow()
    
    def add_classification_metric(self, processing_time_ms: float, confidence_score: float, prediction: str):
        """Add classification performance metric"""
        with self._lock:
            self.classification_times.append(processing_time_ms)
            self.confidence_scores.append(confidence_score)
            self.prediction_counts[prediction] += 1
            self.last_updated = datetime.utcnow()
    
    def add_resource_metric(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Add system resource metric"""
        with self._lock:
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_percent)
            self.disk_usage_history.append(disk_percent)
            self.last_updated = datetime.utcnow()
    
    def add_api_metric(self, endpoint: str, response_time_ms: float, status_code: int):
        """Add API performance metric"""
        with self._lock:
            self.api_response_times.append(response_time_ms)
            self.api_request_counts[endpoint] += 1
            if status_code >= 400:
                self.api_error_counts[endpoint] += 1
            self.last_updated = datetime.utcnow()
    
    def add_scraper_metric(self, job_duration_seconds: float, success_rate: float):
        """Add scraper performance metric"""
        with self._lock:
            self.scraper_job_times.append(job_duration_seconds)
            self.scraper_success_rates.append(success_rate)
            self.last_updated = datetime.utcnow()
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification performance statistics"""
        with self._lock:
            if not self.classification_times:
                return {}
            
            return {
                'avg_processing_time_ms': sum(self.classification_times) / len(self.classification_times),
                'max_processing_time_ms': max(self.classification_times),
                'min_processing_time_ms': min(self.classification_times),
                'avg_confidence_score': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
                'low_confidence_rate': len([s for s in self.confidence_scores if s < 0.7]) / len(self.confidence_scores) if self.confidence_scores else 0,
                'prediction_distribution': dict(self.prediction_counts),
                'total_classifications': len(self.classification_times)
            }
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get system resource statistics"""
        with self._lock:
            stats = {}
            
            if self.cpu_usage_history:
                stats['cpu'] = {
                    'avg_usage_percent': sum(self.cpu_usage_history) / len(self.cpu_usage_history),
                    'max_usage_percent': max(self.cpu_usage_history),
                    'current_usage_percent': self.cpu_usage_history[-1] if self.cpu_usage_history else 0
                }
            
            if self.memory_usage_history:
                stats['memory'] = {
                    'avg_usage_percent': sum(self.memory_usage_history) / len(self.memory_usage_history),
                    'max_usage_percent': max(self.memory_usage_history),
                    'current_usage_percent': self.memory_usage_history[-1] if self.memory_usage_history else 0
                }
            
            if self.disk_usage_history:
                stats['disk'] = {
                    'avg_usage_percent': sum(self.disk_usage_history) / len(self.disk_usage_history),
                    'max_usage_percent': max(self.disk_usage_history),
                    'current_usage_percent': self.disk_usage_history[-1] if self.disk_usage_history else 0
                }
            
            return stats
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        with self._lock:
            if not self.api_response_times:
                return {}
            
            total_requests = sum(self.api_request_counts.values())
            total_errors = sum(self.api_error_counts.values())
            
            return {
                'avg_response_time_ms': sum(self.api_response_times) / len(self.api_response_times),
                'max_response_time_ms': max(self.api_response_times),
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': total_errors / total_requests if total_requests > 0 else 0,
                'requests_per_endpoint': dict(self.api_request_counts),
                'errors_per_endpoint': dict(self.api_error_counts)
            }
    
    def get_scraper_stats(self) -> Dict[str, Any]:
        """Get scraper performance statistics"""
        with self._lock:
            if not self.scraper_job_times:
                return {}
            
            return {
                'avg_job_duration_seconds': sum(self.scraper_job_times) / len(self.scraper_job_times),
                'max_job_duration_seconds': max(self.scraper_job_times),
                'avg_success_rate': sum(self.scraper_success_rates) / len(self.scraper_success_rates) if self.scraper_success_rates else 0,
                'total_jobs': len(self.scraper_job_times)
            }
    
    def get_uptime_seconds(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()


class PerformanceService:
    """
    Performance monitoring and logging service
    """
    
    def __init__(self):
        """Initialize performance service"""
        self.metrics = PerformanceMetrics()
        self._monitoring_active = False
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Performance thresholds from config
        self.thresholds = {}
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self._monitoring_active = True
        self._stop_monitoring.clear()
        
        # Load thresholds from config
        self.thresholds = current_app.config.get('PERFORMANCE_ALERT_THRESHOLDS', {})
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name='PerformanceMonitor'
        )
        self._monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        self._stop_monitoring.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        check_interval = current_app.config.get('RESOURCE_CHECK_INTERVAL', 60)
        
        while not self._stop_monitoring.wait(check_interval):
            try:
                self._collect_system_metrics()
                self._check_performance_alerts()
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect current system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Add to metrics
            self.metrics.add_resource_metric(cpu_percent, memory_percent, disk_percent)
            
            # Log if thresholds exceeded
            if cpu_percent > self.thresholds.get('cpu_usage_percent', 90):
                logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
            
            if memory_percent > self.thresholds.get('memory_usage_percent', 90):
                logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
            
            if disk_percent > self.thresholds.get('disk_usage_percent', 90):
                logger.warning(f"High disk usage detected: {disk_percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        try:
            # Check classification performance
            classification_stats = self.metrics.get_classification_stats()
            if classification_stats:
                avg_time = classification_stats.get('avg_processing_time_ms', 0)
                if avg_time > self.thresholds.get('avg_processing_time_ms', 5000):
                    logger.warning(f"Slow classification performance: {avg_time:.0f}ms average")
                
                low_confidence_rate = classification_stats.get('low_confidence_rate', 0)
                if low_confidence_rate > self.thresholds.get('low_confidence_rate', 0.5):
                    logger.warning(f"High low-confidence rate: {low_confidence_rate:.1%}")
            
            # Check scraper performance
            scraper_stats = self.metrics.get_scraper_stats()
            if scraper_stats:
                success_rate = scraper_stats.get('avg_success_rate', 1.0)
                if success_rate < self.thresholds.get('success_rate', 0.8):
                    logger.warning(f"Low scraper success rate: {success_rate:.1%}")
                    
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    @contextmanager
    def track_classification_performance(self):
        """Context manager to track classification performance"""
        start_time = time.time()
        try:
            yield
        finally:
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            # The actual metrics will be added when classification completes
            # This is just for timing the operation
            if hasattr(g, 'classification_start_time'):
                g.classification_processing_time = processing_time
    
    def record_classification_performance(self, processing_time_ms: float, confidence_score: float, prediction: str):
        """Record classification performance metrics"""
        try:
            self.metrics.add_classification_metric(processing_time_ms, confidence_score, prediction)
            
            # Log performance info
            logger.info(
                f"Classification completed",
                extra={
                    'processing_time_ms': processing_time_ms,
                    'confidence_score': confidence_score,
                    'prediction': prediction,
                    'event_type': 'classification_performance'
                }
            )
            
            # Check for performance issues
            if processing_time_ms > self.thresholds.get('avg_processing_time_ms', 5000):
                logger.warning(
                    f"Slow classification detected: {processing_time_ms:.0f}ms",
                    extra={'event_type': 'performance_alert', 'alert_type': 'slow_classification'}
                )
            
            if confidence_score < 0.7:
                logger.info(
                    f"Low confidence classification: {confidence_score:.3f}",
                    extra={'event_type': 'low_confidence', 'confidence_score': confidence_score}
                )
                
        except Exception as e:
            logger.error(f"Error recording classification performance: {e}")
    
    @contextmanager
    def track_api_performance(self, endpoint: str):
        """Context manager to track API performance"""
        start_time = time.time()
        status_code = 200
        
        try:
            yield
        except Exception as e:
            status_code = 500
            raise
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.metrics.add_api_metric(endpoint, response_time, status_code)
            
            # Log API performance
            logger.info(
                f"API request completed",
                extra={
                    'endpoint': endpoint,
                    'response_time_ms': response_time,
                    'status_code': status_code,
                    'event_type': 'api_performance'
                }
            )
    
    def record_scraper_performance(self, job_duration_seconds: float, products_found: int, products_processed: int, errors: List[str] = None):
        """Record scraper performance metrics"""
        try:
            success_rate = products_processed / products_found if products_found > 0 else 0
            self.metrics.add_scraper_metric(job_duration_seconds, success_rate)
            
            # Log scraper performance
            logger.info(
                f"Scraper job completed",
                extra={
                    'job_duration_seconds': job_duration_seconds,
                    'products_found': products_found,
                    'products_processed': products_processed,
                    'success_rate': success_rate,
                    'error_count': len(errors) if errors else 0,
                    'event_type': 'scraper_performance'
                }
            )
            
            # Check for performance issues
            if success_rate < self.thresholds.get('success_rate', 0.8):
                logger.warning(
                    f"Low scraper success rate: {success_rate:.1%}",
                    extra={'event_type': 'performance_alert', 'alert_type': 'low_scraper_success'}
                )
                
        except Exception as e:
            logger.error(f"Error recording scraper performance: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            return {
                'classification_performance': self.metrics.get_classification_stats(),
                'system_resources': self.metrics.get_resource_stats(),
                'api_performance': self.metrics.get_api_stats(),
                'scraper_performance': self.metrics.get_scraper_stats(),
                'uptime_seconds': self.metrics.get_uptime_seconds(),
                'monitoring_active': self._monitoring_active,
                'last_updated': self.metrics.last_updated.isoformat(),
                'thresholds': self.thresholds
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def get_performance_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance history from database"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get recent classifications for performance analysis
            recent_classifications = Classification.query.filter(
                Classification.classified_at >= cutoff_time
            ).all()
            
            # Get recent scraping jobs
            recent_jobs = ScrapingJob.query.filter(
                ScrapingJob.started_at >= cutoff_time
            ).all()
            
            # Analyze classification performance
            classification_history = []
            if recent_classifications:
                for classification in recent_classifications:
                    classification_history.append({
                        'timestamp': classification.classified_at.isoformat(),
                        'processing_time_ms': classification.processing_time_ms,
                        'confidence_score': classification.confidence_score,
                        'prediction': classification.prediction
                    })
            
            # Analyze scraper performance
            scraper_history = []
            for job in recent_jobs:
                if job.completed_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    success_rate = job.products_processed / job.products_found if job.products_found > 0 else 0
                    
                    scraper_history.append({
                        'timestamp': job.started_at.isoformat(),
                        'duration_seconds': duration,
                        'products_found': job.products_found,
                        'products_processed': job.products_processed,
                        'success_rate': success_rate,
                        'status': job.status
                    })
            
            return {
                'classification_history': classification_history,
                'scraper_history': scraper_history,
                'time_range_hours': hours,
                'total_classifications': len(classification_history),
                'total_scraper_jobs': len(scraper_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return {'error': str(e)}
    
    def export_performance_metrics(self, format: str = 'json') -> Dict[str, Any]:
        """Export performance metrics for external monitoring systems"""
        try:
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_info': {
                    'uptime_seconds': self.metrics.get_uptime_seconds(),
                    'monitoring_active': self._monitoring_active
                },
                'performance_metrics': self.get_performance_summary(),
                'current_resources': self._get_current_resource_usage(),
                'database_stats': self._get_database_stats()
            }
            
            if format == 'prometheus':
                # Convert to Prometheus format (simplified)
                prometheus_metrics = self._convert_to_prometheus_format(metrics_data)
                return prometheus_metrics
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error exporting performance metrics: {e}")
            return {'error': str(e)}
    
    def _get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                'process_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting current resource usage: {e}")
            return {}
    
    def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            return {
                'total_products': Product.query.count(),
                'total_classifications': Classification.query.count(),
                'total_scraping_jobs': ScrapingJob.query.count(),
                'recent_classifications_24h': Classification.query.filter(
                    Classification.classified_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def _convert_to_prometheus_format(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to Prometheus format"""
        # This is a simplified conversion - in practice you'd use a proper Prometheus client
        prometheus_metrics = {}
        
        try:
            # System metrics
            if 'current_resources' in metrics_data:
                resources = metrics_data['current_resources']
                prometheus_metrics.update({
                    'system_cpu_usage_percent': resources.get('cpu_percent', 0),
                    'system_memory_usage_percent': resources.get('memory_percent', 0),
                    'system_disk_usage_percent': resources.get('disk_percent', 0),
                    'process_memory_usage_mb': resources.get('process_memory_mb', 0)
                })
            
            # Performance metrics
            perf_metrics = metrics_data.get('performance_metrics', {})
            classification_perf = perf_metrics.get('classification_performance', {})
            
            if classification_perf:
                prometheus_metrics.update({
                    'classification_avg_processing_time_ms': classification_perf.get('avg_processing_time_ms', 0),
                    'classification_avg_confidence_score': classification_perf.get('avg_confidence_score', 0),
                    'classification_low_confidence_rate': classification_perf.get('low_confidence_rate', 0),
                    'classification_total_count': classification_perf.get('total_classifications', 0)
                })
            
            # Database metrics
            db_stats = metrics_data.get('database_stats', {})
            prometheus_metrics.update({
                'database_total_products': db_stats.get('total_products', 0),
                'database_total_classifications': db_stats.get('total_classifications', 0),
                'database_recent_classifications_24h': db_stats.get('recent_classifications_24h', 0)
            })
            
        except Exception as e:
            logger.error(f"Error converting to Prometheus format: {e}")
        
        return prometheus_metrics


# Global performance service instance
_performance_service = None


def get_performance_service() -> PerformanceService:
    """Get or create global performance service instance"""
    global _performance_service
    if _performance_service is None:
        _performance_service = PerformanceService()
    return _performance_service


def start_performance_monitoring():
    """Start global performance monitoring"""
    service = get_performance_service()
    service.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring"""
    service = get_performance_service()
    service.stop_monitoring()


# Convenience functions for tracking performance
def track_classification_performance(processing_time_ms: float, confidence_score: float, prediction: str):
    """Track classification performance"""
    service = get_performance_service()
    service.record_classification_performance(processing_time_ms, confidence_score, prediction)


def track_scraper_performance(job_duration_seconds: float, products_found: int, products_processed: int, errors: List[str] = None):
    """Track scraper performance"""
    service = get_performance_service()
    service.record_scraper_performance(job_duration_seconds, products_found, products_processed, errors)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary"""
    service = get_performance_service()
    return service.get_performance_summary()