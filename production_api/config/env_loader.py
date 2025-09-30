"""
Environment variable loading utilities for Production Classification API.

This module provides utilities for loading and validating environment variables
with proper type conversion and default value handling.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path


class EnvLoader:
    """Utility class for loading environment variables with validation."""
    
    @staticmethod
    def load_str(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Load string environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            String value or None
            
        Raises:
            ValueError: If required variable is missing
        """
        value = os.environ.get(key, default)
        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    @staticmethod
    def load_int(key: str, default: Optional[int] = None, required: bool = False, 
                 min_val: Optional[int] = None, max_val: Optional[int] = None) -> Optional[int]:
        """
        Load integer environment variable with validation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Integer value or None
            
        Raises:
            ValueError: If validation fails
        """
        str_value = EnvLoader.load_str(key, str(default) if default is not None else None, required)
        if str_value is None:
            return None
        
        try:
            value = int(str_value)
        except ValueError:
            raise ValueError(f"Environment variable {key} must be an integer, got: {str_value}")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"Environment variable {key} must be >= {min_val}, got: {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"Environment variable {key} must be <= {max_val}, got: {value}")
        
        return value
    
    @staticmethod
    def load_float(key: str, default: Optional[float] = None, required: bool = False,
                   min_val: Optional[float] = None, max_val: Optional[float] = None) -> Optional[float]:
        """
        Load float environment variable with validation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Float value or None
            
        Raises:
            ValueError: If validation fails
        """
        str_value = EnvLoader.load_str(key, str(default) if default is not None else None, required)
        if str_value is None:
            return None
        
        try:
            value = float(str_value)
        except ValueError:
            raise ValueError(f"Environment variable {key} must be a float, got: {str_value}")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"Environment variable {key} must be >= {min_val}, got: {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"Environment variable {key} must be <= {max_val}, got: {value}")
        
        return value
    
    @staticmethod
    def load_bool(key: str, default: Optional[bool] = None, required: bool = False) -> Optional[bool]:
        """
        Load boolean environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Boolean value or None
            
        Raises:
            ValueError: If validation fails
        """
        str_value = EnvLoader.load_str(key, str(default).lower() if default is not None else None, required)
        if str_value is None:
            return None
        
        str_value = str_value.lower()
        if str_value in ('true', '1', 'yes', 'on'):
            return True
        elif str_value in ('false', '0', 'no', 'off'):
            return False
        else:
            raise ValueError(f"Environment variable {key} must be a boolean (true/false), got: {str_value}")
    
    @staticmethod
    def load_list(key: str, default: Optional[List[str]] = None, required: bool = False,
                  separator: str = ',') -> Optional[List[str]]:
        """
        Load list environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            separator: Character to split on
            
        Returns:
            List of strings or None
            
        Raises:
            ValueError: If validation fails
        """
        str_value = EnvLoader.load_str(key, separator.join(default) if default else None, required)
        if str_value is None:
            return None
        
        return [item.strip() for item in str_value.split(separator) if item.strip()]
    
    @staticmethod
    def load_path(key: str, default: Optional[str] = None, required: bool = False,
                  must_exist: bool = False) -> Optional[Path]:
        """
        Load path environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            must_exist: Whether the path must exist
            
        Returns:
            Path object or None
            
        Raises:
            ValueError: If validation fails
        """
        str_value = EnvLoader.load_str(key, default, required)
        if str_value is None:
            return None
        
        path = Path(str_value)
        if must_exist and not path.exists():
            raise ValueError(f"Path specified in {key} does not exist: {str_value}")
        
        return path


class ConfigLoader:
    """Configuration loader with environment variable support."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            env_file: Optional .env file to load
        """
        self.env_file = env_file
        if env_file:
            self.load_env_file(env_file)
    
    def load_env_file(self, env_file: str) -> None:
        """
        Load environment variables from file.
        
        Args:
            env_file: Path to .env file
        """
        env_path = Path(env_file)
        if not env_path.exists():
            logging.warning(f"Environment file not found: {env_file}")
            return
        
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' not in line:
                        logging.warning(f"Invalid line {line_num} in {env_file}: {line}")
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
                        
        except Exception as e:
            logging.error(f"Error loading environment file {env_file}: {e}")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration from environment."""
        return {
            'url': EnvLoader.load_str('DATABASE_URL'),
            'pool_timeout': EnvLoader.load_int('DB_POOL_TIMEOUT', 20, min_val=1),
            'pool_recycle': EnvLoader.load_int('DB_POOL_RECYCLE', -1),
            'pool_size': EnvLoader.load_int('DB_POOL_SIZE', 10, min_val=1),
            'max_overflow': EnvLoader.load_int('DB_MAX_OVERFLOW', 20, min_val=0),
            'echo': EnvLoader.load_bool('DB_ECHO', False)
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration from environment."""
        return {
            'url': EnvLoader.load_str('REDIS_URL', 'redis://localhost:6379/0'),
            'password': EnvLoader.load_str('REDIS_PASSWORD'),
            'db': EnvLoader.load_int('REDIS_DB', 0, min_val=0, max_val=15),
            'socket_timeout': EnvLoader.load_int('REDIS_SOCKET_TIMEOUT', 30, min_val=1),
            'connection_pool_max_connections': EnvLoader.load_int('REDIS_MAX_CONNECTIONS', 50, min_val=1)
        }
    
    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration from environment."""
        return {
            'broker_url': EnvLoader.load_str('CELERY_BROKER_URL', 'redis://localhost:6379/1'),
            'result_backend': EnvLoader.load_str('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2'),
            'task_serializer': EnvLoader.load_str('CELERY_TASK_SERIALIZER', 'json'),
            'result_serializer': EnvLoader.load_str('CELERY_RESULT_SERIALIZER', 'json'),
            'accept_content': EnvLoader.load_list('CELERY_ACCEPT_CONTENT', ['json']),
            'timezone': EnvLoader.load_str('CELERY_TIMEZONE', 'UTC'),
            'enable_utc': EnvLoader.load_bool('CELERY_ENABLE_UTC', True)
        }


def load_config_from_env() -> Dict[str, Any]:
    """
    Load complete configuration from environment variables.
    
    Returns:
        Dictionary with all configuration values
    """
    loader = ConfigLoader()
    
    config = {
        'database': loader.get_database_config(),
        'redis': loader.get_redis_config(),
        'celery': loader.get_celery_config(),
        
        # API settings
        'api': {
            'host': EnvLoader.load_str('API_HOST', '0.0.0.0'),
            'port': EnvLoader.load_int('API_PORT', 5000, min_val=1, max_val=65535),
            'debug': EnvLoader.load_bool('API_DEBUG', False),
            'workers': EnvLoader.load_int('API_WORKERS', 4, min_val=1),
            'timeout': EnvLoader.load_int('API_TIMEOUT', 30, min_val=1)
        },
        
        # Security settings
        'security': {
            'secret_key': EnvLoader.load_str('SECRET_KEY', required=True),
            'jwt_secret': EnvLoader.load_str('JWT_SECRET_KEY'),
            'jwt_expiration_hours': EnvLoader.load_int('JWT_EXPIRATION_HOURS', 24, min_val=1),
            'cors_origins': EnvLoader.load_list('CORS_ORIGINS', ['*']),
            'rate_limit_enabled': EnvLoader.load_bool('RATE_LIMIT_ENABLED', True)
        },
        
        # Monitoring settings
        'monitoring': {
            'sentry_dsn': EnvLoader.load_str('SENTRY_DSN'),
            'metrics_enabled': EnvLoader.load_bool('METRICS_ENABLED', True),
            'health_check_enabled': EnvLoader.load_bool('HEALTH_CHECK_ENABLED', True),
            'performance_monitoring': EnvLoader.load_bool('PERFORMANCE_MONITORING', True)
        }
    }
    
    return config