"""
Configuration factory for Production Classification API.

This module provides utilities for creating and initializing configuration
objects based on environment settings and validation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config import get_config, ConfigValidationError
from .env_loader import ConfigLoader


class ConfigFactory:
    """Factory class for creating and managing configuration objects."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure single configuration instance."""
        if cls._instance is None:
            cls._instance = super(ConfigFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration factory."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config = None
            self._env_loader = None
    
    def create_config(self, environment: Optional[str] = None, 
                     env_file: Optional[str] = None,
                     validate: bool = True) -> Any:
        """
        Create configuration object for specified environment.
        
        Args:
            environment: Environment name ('development', 'production', 'testing')
            env_file: Optional .env file to load
            validate: Whether to validate configuration
            
        Returns:
            Configuration object
            
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        # Load environment file if specified
        if env_file:
            self._load_env_file(env_file)
        else:
            # Try to load environment-specific .env file
            self._load_default_env_file(environment)
        
        # Get configuration class
        config_class = get_config(environment)
        
        try:
            # Create configuration instance
            config_instance = config_class()
            
            if validate:
                # Validate configuration
                config_instance.validate_config()
            
            self._config = config_instance
            return config_instance
            
        except Exception as e:
            logging.error(f"Failed to create configuration: {e}")
            raise ConfigValidationError(f"Configuration creation failed: {e}")
    
    def get_config(self) -> Any:
        """
        Get current configuration instance.
        
        Returns:
            Current configuration object or None if not initialized
        """
        return self._config
    
    def reload_config(self, environment: Optional[str] = None,
                     env_file: Optional[str] = None) -> Any:
        """
        Reload configuration with new settings.
        
        Args:
            environment: Environment name
            env_file: Optional .env file to load
            
        Returns:
            New configuration object
        """
        self._config = None
        return self.create_config(environment, env_file)
    
    def _load_env_file(self, env_file: str) -> None:
        """
        Load environment variables from file.
        
        Args:
            env_file: Path to .env file
        """
        if not self._env_loader:
            self._env_loader = ConfigLoader()
        
        self._env_loader.load_env_file(env_file)
    
    def _load_default_env_file(self, environment: Optional[str] = None) -> None:
        """
        Load default environment file based on environment.
        
        Args:
            environment: Environment name
        """
        if environment is None:
            environment = os.environ.get('FLASK_ENV', 'development')
        
        # Look for environment-specific .env file
        config_dir = Path(__file__).parent
        env_file = config_dir / f'.env.{environment}'
        
        if env_file.exists():
            self._load_env_file(str(env_file))
            logging.info(f"Loaded environment file: {env_file}")
        else:
            logging.warning(f"Environment file not found: {env_file}")
    
    def validate_environment(self, environment: str) -> bool:
        """
        Validate if environment configuration is complete and correct.
        
        Args:
            environment: Environment name to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            config_class = get_config(environment)
            config_instance = config_class()
            config_instance.validate_config()
            return True
        except Exception as e:
            logging.error(f"Environment validation failed for {environment}: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about current environment configuration.
        
        Returns:
            Dictionary with environment information
        """
        if not self._config:
            return {'error': 'No configuration loaded'}
        
        return {
            'environment': os.environ.get('FLASK_ENV', 'unknown'),
            'config_class': self._config.__class__.__name__,
            'debug': getattr(self._config, 'DEBUG', False),
            'testing': getattr(self._config, 'TESTING', False),
            'database_url': getattr(self._config, 'SQLALCHEMY_DATABASE_URI', 'not set'),
            'model_path': getattr(self._config, 'MODEL_PATH', 'not set'),
            'log_level': getattr(self._config, 'LOG_LEVEL', 'not set'),
            'api_port': os.environ.get('API_PORT', 'not set'),
            'secret_key_set': bool(os.environ.get('SECRET_KEY')),
            'validation_passed': True
        }


def create_app_config(environment: Optional[str] = None,
                     env_file: Optional[str] = None) -> Any:
    """
    Convenience function to create application configuration.
    
    Args:
        environment: Environment name
        env_file: Optional .env file to load
        
    Returns:
        Configuration object
    """
    factory = ConfigFactory()
    return factory.create_config(environment, env_file)


def validate_config_files() -> Dict[str, bool]:
    """
    Validate all environment configuration files.
    
    Returns:
        Dictionary with validation results for each environment
    """
    factory = ConfigFactory()
    environments = ['development', 'production', 'testing']
    results = {}
    
    for env in environments:
        results[env] = factory.validate_environment(env)
    
    return results


def get_config_summary() -> Dict[str, Any]:
    """
    Get summary of current configuration state.
    
    Returns:
        Dictionary with configuration summary
    """
    factory = ConfigFactory()
    
    summary = {
        'current_environment': os.environ.get('FLASK_ENV', 'not set'),
        'config_loaded': factory.get_config() is not None,
        'environment_validations': validate_config_files(),
        'environment_variables': {
            'SECRET_KEY': 'set' if os.environ.get('SECRET_KEY') else 'not set',
            'DATABASE_URL': 'set' if os.environ.get('DATABASE_URL') else 'not set',
            'MODEL_PATH': 'set' if os.environ.get('MODEL_PATH') else 'not set',
            'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'not set'),
            'API_PORT': os.environ.get('API_PORT', 'not set')
        }
    }
    
    if factory.get_config():
        summary['config_info'] = factory.get_environment_info()
    
    return summary