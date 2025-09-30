"""
Structured Logging Configuration

This module provides structured logging setup for the production API with:
- JSON formatted logs for production
- Contextual logging with request IDs
- Performance metrics logging
- Error tracking and alerting
- Log rotation and management
"""

import os
import sys
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, g, has_request_context
from pythonjsonlogger import jsonlogger


class ContextualFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def filter(self, record):
        """Add context to log record"""
        # Add request context if available
        if has_request_context():
            record.request_id = getattr(g, 'request_id', 'unknown')
            record.endpoint = request.endpoint if request else 'unknown'
            record.method = request.method if request else 'unknown'
            record.remote_addr = request.remote_addr if request else 'unknown'
        else:
            record.request_id = 'no-request'
            record.endpoint = 'system'
            record.method = 'system'
            record.remote_addr = 'system'
        
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Add service information
        record.service = 'hp-classification-api'
        record.version = os.environ.get('API_VERSION', 'v1')
        
        return True


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log records"""
    
    def filter(self, record):
        """Only allow performance-related records"""
        return hasattr(record, 'event_type') and record.event_type in [
            'classification_performance',
            'api_performance', 
            'scraper_performance',
            'performance_alert'
        ]


class ErrorFilter(logging.Filter):
    """Filter for error-related log records"""
    
    def filter(self, record):
        """Only allow error and warning records"""
        return record.levelno >= logging.WARNING


class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)
        
        # Ensure timestamp is present
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add log level name
        log_record['level'] = record.levelname
        
        # Add logger name
        log_record['logger'] = record.name
        
        # Add process and thread info
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def setup_logging(app: Flask) -> None:
    """
    Setup comprehensive logging for the Flask application
    
    Args:
        app: Flask application instance
    """
    # Get configuration
    log_level = app.config.get('LOG_LEVEL', 'INFO')
    log_format = app.config.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(app.instance_path), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup different handlers based on environment
    if app.config.get('DEBUG'):
        setup_development_logging(app, root_logger)
    else:
        setup_production_logging(app, root_logger, logs_dir)
    
    # Setup application-specific loggers
    setup_application_loggers(app)
    
    # Setup request logging
    setup_request_logging(app)
    
    app.logger.info("Logging configuration completed")


def setup_development_logging(app: Flask, root_logger: logging.Logger) -> None:
    """Setup logging for development environment"""
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Simple format for development
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add contextual filter
    console_handler.addFilter(ContextualFilter())
    
    root_logger.addHandler(console_handler)


def setup_production_logging(app: Flask, root_logger: logging.Logger, logs_dir: str) -> None:
    """Setup logging for production environment"""
    
    # Main application log (JSON format)
    app_log_file = os.path.join(logs_dir, 'application.log')
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    app_handler.setLevel(logging.INFO)
    
    # JSON formatter for structured logging
    json_formatter = CustomJSONFormatter(
        '%(timestamp)s %(level)s %(logger)s %(message)s'
    )
    app_handler.setFormatter(json_formatter)
    app_handler.addFilter(ContextualFilter())
    
    root_logger.addHandler(app_handler)
    
    # Performance log (separate file for performance metrics)
    perf_log_file = os.path.join(logs_dir, 'performance.log')
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(json_formatter)
    perf_handler.addFilter(PerformanceFilter())
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False
    
    # Error log (separate file for errors and warnings)
    error_log_file = os.path.join(logs_dir, 'errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=10
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(json_formatter)
    error_handler.addFilter(ErrorFilter())
    
    root_logger.addHandler(error_handler)
    
    # Console handler for critical errors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - CRITICAL - %(name)s - %(message)s'
    ))
    
    root_logger.addHandler(console_handler)


def setup_application_loggers(app: Flask) -> None:
    """Setup application-specific loggers"""
    
    # Database logger
    db_logger = logging.getLogger('sqlalchemy.engine')
    if app.config.get('DEBUG'):
        db_logger.setLevel(logging.INFO)
    else:
        db_logger.setLevel(logging.WARNING)
    
    # Scraper logger
    scraper_logger = logging.getLogger('scraper')
    scraper_logger.setLevel(logging.INFO)
    
    # ML service logger
    ml_logger = logging.getLogger('ml_service')
    ml_logger.setLevel(logging.INFO)
    
    # Health service logger
    health_logger = logging.getLogger('health_service')
    health_logger.setLevel(logging.INFO)
    
    # Performance logger
    performance_logger = logging.getLogger('performance')
    performance_logger.setLevel(logging.INFO)


def setup_request_logging(app: Flask) -> None:
    """Setup request/response logging"""
    
    @app.before_request
    def before_request():
        """Log request start and setup request context"""
        import uuid
        
        # Generate unique request ID
        g.request_id = str(uuid.uuid4())
        g.request_start_time = datetime.utcnow()
        
        # Log request start
        app.logger.info(
            f"Request started: {request.method} {request.path}",
            extra={
                'event_type': 'request_start',
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', ''),
                'content_length': request.content_length
            }
        )
    
    @app.after_request
    def after_request(response):
        """Log request completion"""
        if hasattr(g, 'request_start_time'):
            duration = (datetime.utcnow() - g.request_start_time).total_seconds() * 1000
            
            app.logger.info(
                f"Request completed: {request.method} {request.path} - {response.status_code}",
                extra={
                    'event_type': 'request_complete',
                    'method': request.method,
                    'path': request.path,
                    'status_code': response.status_code,
                    'duration_ms': duration,
                    'content_length': response.content_length
                }
            )
            
            # Track API performance
            try:
                from production_api.app.services.performance_service import get_performance_service
                perf_service = get_performance_service()
                perf_service.metrics.add_api_metric(
                    request.endpoint or request.path,
                    duration,
                    response.status_code
                )
            except Exception as e:
                app.logger.warning(f"Failed to track API performance: {e}")
        
        return response
    
    @app.errorhandler(Exception)
    def log_exception(error):
        """Log unhandled exceptions"""
        app.logger.error(
            f"Unhandled exception: {str(error)}",
            extra={
                'event_type': 'unhandled_exception',
                'exception_type': type(error).__name__,
                'method': request.method if request else 'unknown',
                'path': request.path if request else 'unknown'
            },
            exc_info=True
        )
        
        # Re-raise the exception
        raise error


def get_structured_logger(name: str) -> logging.Logger:
    """
    Get a structured logger with contextual information
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add contextual filter if not already present
    if not any(isinstance(f, ContextualFilter) for f in logger.filters):
        logger.addFilter(ContextualFilter())
    
    return logger


def log_performance_metric(
    event_type: str,
    metric_name: str,
    metric_value: float,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a performance metric in structured format
    
    Args:
        event_type: Type of performance event
        metric_name: Name of the metric
        metric_value: Metric value
        additional_data: Additional contextual data
    """
    logger = logging.getLogger('performance')
    
    log_data = {
        'event_type': event_type,
        'metric_name': metric_name,
        'metric_value': metric_value
    }
    
    if additional_data:
        log_data.update(additional_data)
    
    logger.info(f"Performance metric: {metric_name} = {metric_value}", extra=log_data)


def log_business_event(
    event_type: str,
    event_data: Dict[str, Any],
    level: str = 'info'
) -> None:
    """
    Log a business event in structured format
    
    Args:
        event_type: Type of business event
        event_data: Event data
        level: Log level
    """
    logger = logging.getLogger('business_events')
    
    log_data = {
        'event_type': event_type,
        **event_data
    }
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"Business event: {event_type}", extra=log_data)


def log_security_event(
    event_type: str,
    severity: str,
    description: str,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a security event
    
    Args:
        event_type: Type of security event
        severity: Severity level (low, medium, high, critical)
        description: Event description
        additional_data: Additional contextual data
    """
    logger = logging.getLogger('security')
    
    log_data = {
        'event_type': 'security_event',
        'security_event_type': event_type,
        'severity': severity,
        'description': description
    }
    
    if additional_data:
        log_data.update(additional_data)
    
    # Log at appropriate level based on severity
    if severity in ['critical', 'high']:
        logger.error(f"Security event: {description}", extra=log_data)
    elif severity == 'medium':
        logger.warning(f"Security event: {description}", extra=log_data)
    else:
        logger.info(f"Security event: {description}", extra=log_data)


# Convenience functions for common logging patterns
def log_classification_event(
    product_id: int,
    prediction: str,
    confidence_score: float,
    processing_time_ms: float,
    model_version: str
) -> None:
    """Log a classification event"""
    log_business_event('product_classification', {
        'product_id': product_id,
        'prediction': prediction,
        'confidence_score': confidence_score,
        'processing_time_ms': processing_time_ms,
        'model_version': model_version
    })


def log_scraper_event(
    job_id: int,
    job_type: str,
    status: str,
    products_found: int,
    products_processed: int,
    duration_seconds: float
) -> None:
    """Log a scraper event"""
    log_business_event('scraper_job', {
        'job_id': job_id,
        'job_type': job_type,
        'status': status,
        'products_found': products_found,
        'products_processed': products_processed,
        'duration_seconds': duration_seconds
    })


def log_health_check_event(
    component: str,
    status: str,
    issues: List[str] = None,
    metrics: Dict[str, Any] = None
) -> None:
    """Log a health check event"""
    log_business_event('health_check', {
        'component': component,
        'status': status,
        'issues': issues or [],
        'metrics': metrics or {}
    })