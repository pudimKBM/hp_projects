"""
Configuration settings for Production Classification API.

This module contains configuration classes for different environments
and database settings with validation and default value handling.
"""

import os
import logging
from datetime import timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """Base configuration class with validation and default value handling."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    
    # API settings
    API_TITLE = 'HP Product Classification API'
    API_VERSION = 'v1'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # Pagination settings
    DEFAULT_PAGE_SIZE = int(os.environ.get('DEFAULT_PAGE_SIZE', 20))
    MAX_PAGE_SIZE = int(os.environ.get('MAX_PAGE_SIZE', 100))
    
    # ML Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.7))
    MODEL_CACHE_SIZE = int(os.environ.get('MODEL_CACHE_SIZE', 3))
    MODEL_TIMEOUT_SECONDS = int(os.environ.get('MODEL_TIMEOUT_SECONDS', 30))
    FEATURE_ENGINEERING_CONFIG_PATH = os.environ.get('FEATURE_CONFIG_PATH') or os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'config.py')
    
    # Scraper settings
    SCRAPER_DELAY = float(os.environ.get('SCRAPER_DELAY', 2.0))  # seconds between requests
    MAX_PAGES_PER_TERM = int(os.environ.get('MAX_PAGES_PER_TERM', 5))
    SCRAPER_TIMEOUT = int(os.environ.get('SCRAPER_TIMEOUT', 30))
    SCRAPER_RETRY_ATTEMPTS = int(os.environ.get('SCRAPER_RETRY_ATTEMPTS', 3))
    SCRAPER_USER_AGENT = os.environ.get('SCRAPER_USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    # Default search terms (can be overridden via environment)
    _default_search_terms = [
        'cartucho hp original',
        'toner hp',
        'cartucho hp 664',
        'cartucho hp 122',
        'toner hp laserjet'
    ]
    
    @property
    def DEFAULT_SEARCH_TERMS(self) -> List[str]:
        """Get search terms from environment or use defaults."""
        env_terms = os.environ.get('SEARCH_TERMS')
        if env_terms:
            return [term.strip() for term in env_terms.split(',')]
        return self._default_search_terms
    
    # Cron job settings
    DEFAULT_CRON_SCHEDULE = os.environ.get('CRON_SCHEDULE', '0 */6 * * *')  # Every 6 hours
    CRON_JOB_TIMEOUT = int(os.environ.get('CRON_JOB_TIMEOUT', 3600))  # 1 hour
    
    # Health check settings
    HEALTH_CHECK_TIMEOUT = int(os.environ.get('HEALTH_CHECK_TIMEOUT', 30))  # seconds
    HEALTH_CACHE_DURATION = int(os.environ.get('HEALTH_CACHE_DURATION', 5))  # minutes to cache health results
    HEALTH_HISTORY_DAYS = int(os.environ.get('HEALTH_HISTORY_DAYS', 7))  # days to keep health records
    
    # Performance monitoring settings
    PERFORMANCE_ALERT_THRESHOLDS = {
        'cpu_usage_percent': float(os.environ.get('CPU_ALERT_THRESHOLD', 90)),
        'memory_usage_percent': float(os.environ.get('MEMORY_ALERT_THRESHOLD', 90)),
        'disk_usage_percent': float(os.environ.get('DISK_ALERT_THRESHOLD', 90)),
        'avg_processing_time_ms': int(os.environ.get('PROCESSING_TIME_ALERT_THRESHOLD', 5000)),
        'low_confidence_rate': float(os.environ.get('LOW_CONFIDENCE_ALERT_THRESHOLD', 0.5)),
        'success_rate': float(os.environ.get('SUCCESS_RATE_ALERT_THRESHOLD', 0.8))
    }
    
    # System resource monitoring
    RESOURCE_CHECK_INTERVAL = int(os.environ.get('RESOURCE_CHECK_INTERVAL', 60))  # seconds between resource checks
    
    # Rate limiting settings
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 100))
    RATE_LIMIT_STORAGE_URL = os.environ.get('RATE_LIMIT_STORAGE_URL', 'memory://')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT = os.environ.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE_PATH = os.environ.get('LOG_FILE_PATH')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
    
    def __init__(self):
        """Initialize configuration and validate settings."""
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate configuration settings and raise errors for invalid values."""
        errors = []
        
        # Validate pagination settings
        if self.DEFAULT_PAGE_SIZE <= 0:
            errors.append("DEFAULT_PAGE_SIZE must be positive")
        if self.MAX_PAGE_SIZE <= 0:
            errors.append("MAX_PAGE_SIZE must be positive")
        if self.DEFAULT_PAGE_SIZE > self.MAX_PAGE_SIZE:
            errors.append("DEFAULT_PAGE_SIZE cannot be greater than MAX_PAGE_SIZE")
        
        # Validate ML settings
        if not (0 <= self.CONFIDENCE_THRESHOLD <= 1):
            errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.MODEL_CACHE_SIZE <= 0:
            errors.append("MODEL_CACHE_SIZE must be positive")
        if self.MODEL_TIMEOUT_SECONDS <= 0:
            errors.append("MODEL_TIMEOUT_SECONDS must be positive")
        
        # Validate scraper settings
        if self.SCRAPER_DELAY < 0:
            errors.append("SCRAPER_DELAY cannot be negative")
        if self.MAX_PAGES_PER_TERM <= 0:
            errors.append("MAX_PAGES_PER_TERM must be positive")
        if self.SCRAPER_TIMEOUT <= 0:
            errors.append("SCRAPER_TIMEOUT must be positive")
        if self.SCRAPER_RETRY_ATTEMPTS < 0:
            errors.append("SCRAPER_RETRY_ATTEMPTS cannot be negative")
        
        # Validate health check settings
        if self.HEALTH_CHECK_TIMEOUT <= 0:
            errors.append("HEALTH_CHECK_TIMEOUT must be positive")
        if self.HEALTH_CACHE_DURATION <= 0:
            errors.append("HEALTH_CACHE_DURATION must be positive")
        if self.HEALTH_HISTORY_DAYS <= 0:
            errors.append("HEALTH_HISTORY_DAYS must be positive")
        
        # Validate performance thresholds
        for key, value in self.PERFORMANCE_ALERT_THRESHOLDS.items():
            if key.endswith('_percent') and not (0 <= value <= 100):
                errors.append(f"Performance threshold {key} must be between 0 and 100")
            elif key == 'avg_processing_time_ms' and value <= 0:
                errors.append(f"Performance threshold {key} must be positive")
            elif key in ['low_confidence_rate', 'success_rate'] and not (0 <= value <= 1):
                errors.append(f"Performance threshold {key} must be between 0 and 1")
        
        # Validate rate limiting
        if self.RATE_LIMIT_PER_MINUTE <= 0:
            errors.append("RATE_LIMIT_PER_MINUTE must be positive")
        
        # Validate logging settings
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of {valid_log_levels}")
        
        if self.LOG_MAX_BYTES <= 0:
            errors.append("LOG_MAX_BYTES must be positive")
        if self.LOG_BACKUP_COUNT < 0:
            errors.append("LOG_BACKUP_COUNT cannot be negative")
        
        # Validate paths exist (only warn, don't fail)
        model_path = Path(self.MODEL_PATH)
        if not model_path.exists():
            logging.warning(f"MODEL_PATH does not exist: {self.MODEL_PATH}")
        
        feature_config_path = Path(self.FEATURE_ENGINEERING_CONFIG_PATH)
        if not feature_config_path.exists():
            logging.warning(f"FEATURE_ENGINEERING_CONFIG_PATH does not exist: {self.FEATURE_ENGINEERING_CONFIG_PATH}")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get ML model configuration dictionary."""
        return {
            'model_path': self.MODEL_PATH,
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'cache_size': self.MODEL_CACHE_SIZE,
            'timeout_seconds': self.MODEL_TIMEOUT_SECONDS,
            'feature_config_path': self.FEATURE_ENGINEERING_CONFIG_PATH
        }
    
    def get_scraper_config(self) -> Dict[str, Any]:
        """Get scraper configuration dictionary."""
        return {
            'delay': self.SCRAPER_DELAY,
            'max_pages_per_term': self.MAX_PAGES_PER_TERM,
            'timeout': self.SCRAPER_TIMEOUT,
            'retry_attempts': self.SCRAPER_RETRY_ATTEMPTS,
            'user_agent': self.SCRAPER_USER_AGENT,
            'search_terms': self.DEFAULT_SEARCH_TERMS,
            'cron_schedule': self.DEFAULT_CRON_SCHEDULE,
            'job_timeout': self.CRON_JOB_TIMEOUT
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration dictionary."""
        return {
            'title': self.API_TITLE,
            'version': self.API_VERSION,
            'max_content_length': self.MAX_CONTENT_LENGTH,
            'default_page_size': self.DEFAULT_PAGE_SIZE,
            'max_page_size': self.MAX_PAGE_SIZE,
            'rate_limit_per_minute': self.RATE_LIMIT_PER_MINUTE,
            'rate_limit_storage_url': self.RATE_LIMIT_STORAGE_URL
        }
    
    def get_health_config(self) -> Dict[str, Any]:
        """Get health monitoring configuration dictionary."""
        return {
            'check_timeout': self.HEALTH_CHECK_TIMEOUT,
            'cache_duration': self.HEALTH_CACHE_DURATION,
            'history_days': self.HEALTH_HISTORY_DAYS,
            'alert_thresholds': self.PERFORMANCE_ALERT_THRESHOLDS,
            'resource_check_interval': self.RESOURCE_CHECK_INTERVAL
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return {
            'level': self.LOG_LEVEL,
            'format': self.LOG_FORMAT,
            'file_path': self.LOG_FILE_PATH,
            'max_bytes': self.LOG_MAX_BYTES,
            'backup_count': self.LOG_BACKUP_COUNT
        }


class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    DEBUG = True
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(__file__), '..', 'data', 'development.db')
    
    # Logging (override environment for development)
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
    
    # Scraper settings (more conservative for development)
    SCRAPER_DELAY = float(os.environ.get('SCRAPER_DELAY', 3.0))
    MAX_PAGES_PER_TERM = int(os.environ.get('MAX_PAGES_PER_TERM', 2))
    
    # Lower thresholds for development testing
    PERFORMANCE_ALERT_THRESHOLDS = {
        'cpu_usage_percent': float(os.environ.get('CPU_ALERT_THRESHOLD', 80)),
        'memory_usage_percent': float(os.environ.get('MEMORY_ALERT_THRESHOLD', 80)),
        'disk_usage_percent': float(os.environ.get('DISK_ALERT_THRESHOLD', 80)),
        'avg_processing_time_ms': int(os.environ.get('PROCESSING_TIME_ALERT_THRESHOLD', 3000)),
        'low_confidence_rate': float(os.environ.get('LOW_CONFIDENCE_ALERT_THRESHOLD', 0.3)),
        'success_rate': float(os.environ.get('SUCCESS_RATE_ALERT_THRESHOLD', 0.7))
    }


class ProductionConfig(Config):
    """Production environment configuration."""
    
    DEBUG = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(__file__), '..', 'data', 'production.db')
    
    # Performance settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_timeout': int(os.environ.get('DB_POOL_TIMEOUT', 20)),
        'pool_recycle': int(os.environ.get('DB_POOL_RECYCLE', -1)),
        'pool_pre_ping': True
    }
    
    # Production-specific settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 60))  # More restrictive in production
    
    def __init__(self):
        super().__init__()
        # Additional production validations
        self._validate_production_settings()
    
    def _validate_production_settings(self) -> None:
        """Additional validation for production environment."""
        errors = []
        
        # Check SECRET_KEY
        if not os.environ.get('SECRET_KEY') or os.environ.get('SECRET_KEY') == 'dev-secret-key-change-in-production':
            errors.append("SECRET_KEY environment variable must be set to a secure value in production")
        
        # Validate database URL for production
        db_url = os.environ.get('DATABASE_URL')
        if db_url and db_url.startswith('sqlite:///'):
            logging.warning("Using SQLite in production - consider using PostgreSQL or MySQL for better performance")
        
        # Ensure logging is configured properly
        if not os.environ.get('LOG_FILE_PATH'):
            logging.warning("LOG_FILE_PATH not set - logs will only go to console")
        
        if errors:
            raise ConfigValidationError(f"Production configuration validation failed: {'; '.join(errors)}")


class TestingConfig(Config):
    """Testing environment configuration."""
    
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Faster scraping for tests
    SCRAPER_DELAY = 0.1
    MAX_PAGES_PER_TERM = 1
    SCRAPER_TIMEOUT = 5
    MODEL_TIMEOUT_SECONDS = 5
    
    # Testing-specific settings
    LOG_LEVEL = 'DEBUG'
    HEALTH_CHECK_TIMEOUT = 5
    HEALTH_CACHE_DURATION = 1
    RATE_LIMIT_PER_MINUTE = 1000  # No rate limiting in tests
    
    def validate_config(self) -> None:
        """Skip path validation in testing environment."""
        errors = []
        
        # Only validate numeric ranges, skip path validation
        if self.DEFAULT_PAGE_SIZE <= 0:
            errors.append("DEFAULT_PAGE_SIZE must be positive")
        if self.MAX_PAGE_SIZE <= 0:
            errors.append("MAX_PAGE_SIZE must be positive")
        if self.DEFAULT_PAGE_SIZE > self.MAX_PAGE_SIZE:
            errors.append("DEFAULT_PAGE_SIZE cannot be greater than MAX_PAGE_SIZE")
        
        if not (0 <= self.CONFIDENCE_THRESHOLD <= 1):
            errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """
    Get configuration class based on environment.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, uses FLASK_ENV environment variable
    
    Returns:
        Configuration class
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, DevelopmentConfig)