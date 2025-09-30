#!/usr/bin/env python3
"""
Configuration validation script for Production Classification API.

This script validates configuration settings for all environments
and provides detailed feedback on any issues found.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to import config modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.factory import ConfigFactory, validate_config_files, get_config_summary
from config.config import ConfigValidationError


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_environment(environment: str, verbose: bool = False) -> bool:
    """
    Validate specific environment configuration.
    
    Args:
        environment: Environment name to validate
        verbose: Whether to show detailed output
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Validating {environment.upper()} environment")
    print(f"{'='*60}")
    
    try:
        factory = ConfigFactory()
        
        # Load environment-specific .env file
        config_dir = Path(__file__).parent
        env_file = config_dir / f'.env.{environment}'
        
        if env_file.exists():
            print(f"✓ Found environment file: {env_file}")
        else:
            print(f"⚠ Environment file not found: {env_file}")
        
        # Create and validate configuration
        config = factory.create_config(environment, str(env_file) if env_file.exists() else None)
        
        print(f"✓ Configuration class: {config.__class__.__name__}")
        print(f"✓ Debug mode: {getattr(config, 'DEBUG', False)}")
        print(f"✓ Testing mode: {getattr(config, 'TESTING', False)}")
        
        # Validate specific settings
        validation_results = []
        
        # Database validation
        db_url = getattr(config, 'SQLALCHEMY_DATABASE_URI', None)
        if db_url:
            print(f"✓ Database URL configured: {db_url[:20]}...")
            validation_results.append(('Database URL', True, 'Configured'))
        else:
            print("✗ Database URL not configured")
            validation_results.append(('Database URL', False, 'Not configured'))
        
        # Model path validation
        model_path = getattr(config, 'MODEL_PATH', None)
        if model_path and Path(model_path).exists():
            print(f"✓ Model path exists: {model_path}")
            validation_results.append(('Model Path', True, 'Exists'))
        elif model_path:
            print(f"⚠ Model path configured but doesn't exist: {model_path}")
            validation_results.append(('Model Path', False, 'Path does not exist'))
        else:
            print("✗ Model path not configured")
            validation_results.append(('Model Path', False, 'Not configured'))
        
        # Feature engineering config validation
        feature_config = getattr(config, 'FEATURE_ENGINEERING_CONFIG_PATH', None)
        if feature_config and Path(feature_config).exists():
            print(f"✓ Feature config exists: {feature_config}")
            validation_results.append(('Feature Config', True, 'Exists'))
        elif feature_config:
            print(f"⚠ Feature config configured but doesn't exist: {feature_config}")
            validation_results.append(('Feature Config', False, 'Path does not exist'))
        else:
            print("✗ Feature config not configured")
            validation_results.append(('Feature Config', False, 'Not configured'))
        
        # Security validation for production
        if environment == 'production':
            secret_key = os.environ.get('SECRET_KEY')
            if secret_key and secret_key != 'dev-secret-key-change-in-production':
                print("✓ Production secret key configured")
                validation_results.append(('Secret Key', True, 'Secure key set'))
            else:
                print("✗ Production secret key not properly configured")
                validation_results.append(('Secret Key', False, 'Insecure or default key'))
        
        # Show detailed configuration if verbose
        if verbose:
            print(f"\nDetailed Configuration:")
            print(f"  API Config: {config.get_api_config()}")
            print(f"  ML Config: {config.get_model_config()}")
            print(f"  Scraper Config: {config.get_scraper_config()}")
            print(f"  Health Config: {config.get_health_config()}")
        
        # Summary
        passed = sum(1 for _, success, _ in validation_results if success)
        total = len(validation_results)
        
        print(f"\nValidation Summary: {passed}/{total} checks passed")
        
        if passed == total:
            print(f"✅ {environment.upper()} environment validation PASSED")
            return True
        else:
            print(f"❌ {environment.upper()} environment validation FAILED")
            print("\nFailed checks:")
            for name, success, message in validation_results:
                if not success:
                    print(f"  - {name}: {message}")
            return False
        
    except ConfigValidationError as e:
        print(f"❌ Configuration validation error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def validate_all_environments(verbose: bool = False) -> bool:
    """
    Validate all environment configurations.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        True if all validations pass, False otherwise
    """
    environments = ['development', 'production', 'testing']
    results = {}
    
    print("Production Classification API - Configuration Validation")
    print("=" * 60)
    
    for env in environments:
        results[env] = validate_environment(env, verbose)
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for env, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{env.upper():12} : {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    return all_passed


def check_environment_files() -> None:
    """Check for existence of environment files."""
    print("\nEnvironment Files Check:")
    print("-" * 30)
    
    config_dir = Path(__file__).parent
    env_files = [
        '.env.example',
        '.env.development', 
        '.env.production',
        '.env.testing'
    ]
    
    for env_file in env_files:
        file_path = config_dir / env_file
        if file_path.exists():
            print(f"✓ {env_file}")
        else:
            print(f"✗ {env_file} (missing)")


def show_config_summary() -> None:
    """Show configuration summary."""
    print("\nConfiguration Summary:")
    print("-" * 30)
    
    try:
        summary = get_config_summary()
        
        print(f"Current Environment: {summary.get('current_environment', 'unknown')}")
        print(f"Config Loaded: {summary.get('config_loaded', False)}")
        
        env_vars = summary.get('environment_variables', {})
        print("\nEnvironment Variables:")
        for var, status in env_vars.items():
            print(f"  {var}: {status}")
        
        validations = summary.get('environment_validations', {})
        print("\nEnvironment Validations:")
        for env, valid in validations.items():
            status = "✅" if valid else "❌"
            print(f"  {env}: {status}")
            
    except Exception as e:
        print(f"Error getting config summary: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Validate Production Classification API configuration'
    )
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'production', 'testing', 'all'],
        default='all',
        help='Environment to validate (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Show configuration summary'
    )
    parser.add_argument(
        '--check-files', '-f',
        action='store_true',
        help='Check environment files existence'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    success = True
    
    if args.check_files:
        check_environment_files()
    
    if args.summary:
        show_config_summary()
    
    if args.environment == 'all':
        success = validate_all_environments(args.verbose)
    else:
        success = validate_environment(args.environment, args.verbose)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()