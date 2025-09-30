#!/usr/bin/env python3
"""
Deployment script for HP Product Classification API.

This script handles the complete deployment process including:
- Configuration validation
- Database initialization
- Cron job setup
- Service health checks
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.validate_config import validate_all_environments, get_config_summary
from setup_cron import CronSetup


class DeploymentManager:
    """Manages the deployment process for the HP Product Classification API."""
    
    def __init__(self, project_root: str):
        """
        Initialize deployment manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.python_executable = sys.executable
    
    def validate_prerequisites(self) -> bool:
        """
        Validate deployment prerequisites.
        
        Returns:
            True if all prerequisites are met, False otherwise
        """
        print("Validating deployment prerequisites...")
        print("-" * 50)
        
        success = True
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("✗ Python 3.8+ required")
            success = False
        else:
            print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required directories
        required_dirs = ['app', 'config', 'data', 'logs', 'models']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✓ Directory exists: {dir_name}/")
            else:
                print(f"⚠ Creating directory: {dir_name}/")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Check required files
        required_files = [
            'run_api.py',
            'run_scraper_standalone.py',
            'setup_cron.py',
            'requirements.txt',
            'config/config.py'
        ]
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                print(f"✓ File exists: {file_name}")
            else:
                print(f"✗ Missing required file: {file_name}")
                success = False
        
        # Check if models directory has models
        models_dir = self.project_root.parent / 'models'
        if models_dir.exists() and list(models_dir.glob('*.joblib')):
            print("✓ ML models found")
        else:
            print("⚠ No ML models found - classification may not work")
        
        # Check if src directory exists (for feature engineering)
        src_dir = self.project_root.parent / 'src'
        if src_dir.exists():
            print("✓ Source code directory found")
        else:
            print("⚠ Source code directory not found - feature engineering may not work")
        
        return success
    
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """
        Install Python dependencies.
        
        Args:
            upgrade: Whether to upgrade existing packages
            
        Returns:
            True if successful, False otherwise
        """
        print("Installing Python dependencies...")
        print("-" * 40)
        
        try:
            requirements_file = self.project_root / 'requirements.txt'
            
            cmd = [self.python_executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            if upgrade:
                cmd.append('--upgrade')
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Dependencies installed successfully")
                return True
            else:
                print(f"✗ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Error installing dependencies: {e}")
            return False
    
    def validate_configuration(self, environment: str) -> bool:
        """
        Validate configuration for specified environment.
        
        Args:
            environment: Environment to validate
            
        Returns:
            True if valid, False otherwise
        """
        print(f"Validating {environment} configuration...")
        print("-" * 40)
        
        try:
            if environment == 'all':
                return validate_all_environments(verbose=False)
            else:
                # Import here to avoid circular imports
                from config.factory import ConfigFactory
                factory = ConfigFactory()
                return factory.validate_environment(environment)
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return False
    
    def initialize_database(self, environment: str) -> bool:
        """
        Initialize database for specified environment.
        
        Args:
            environment: Environment to initialize
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Initializing database for {environment}...")
        print("-" * 40)
        
        try:
            # Set environment
            os.environ['FLASK_ENV'] = environment
            
            # Run database initialization
            init_script = self.project_root / 'init_db.py'
            result = subprocess.run([
                self.python_executable, str(init_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Database initialized successfully")
                return True
            else:
                print(f"✗ Database initialization failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Error initializing database: {e}")
            return False
    
    def setup_cron_jobs(self, environment: str, schedule: str = None) -> bool:
        """
        Setup cron jobs for automated scraping.
        
        Args:
            environment: Environment to setup jobs for
            schedule: Cron schedule (optional)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Setting up cron jobs for {environment}...")
        print("-" * 40)
        
        try:
            cron_setup = CronSetup(str(self.project_root))
            
            if not schedule:
                schedule = cron_setup.get_default_schedule()
            
            print(f"Schedule: {schedule}")
            
            # Setup cron job based on OS
            import platform
            if platform.system().lower() == 'windows':
                success = cron_setup.setup_windows_task(schedule, environment)
            else:
                success = cron_setup.setup_linux_cron(schedule, environment)
            
            return success
            
        except Exception as e:
            print(f"✗ Error setting up cron jobs: {e}")
            return False
    
    def run_health_check(self, environment: str) -> bool:
        """
        Run health check on deployed system.
        
        Args:
            environment: Environment to check
            
        Returns:
            True if healthy, False otherwise
        """
        print(f"Running health check for {environment}...")
        print("-" * 40)
        
        try:
            # Set environment
            os.environ['FLASK_ENV'] = environment
            
            # Test configuration loading
            from config import create_app_config
            config = create_app_config(environment)
            print("✓ Configuration loaded successfully")
            
            # Test database connection (if not in-memory)
            db_url = config.SQLALCHEMY_DATABASE_URI
            if 'memory' not in db_url:
                # Try to connect to database
                from sqlalchemy import create_engine
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                print("✓ Database connection successful")
            
            # Test model loading (if models exist)
            model_path = Path(config.MODEL_PATH)
            if model_path.exists() and list(model_path.glob('*.joblib')):
                print("✓ ML models available")
            else:
                print("⚠ No ML models found")
            
            # Test scraper configuration
            scraper_config = config.get_scraper_config()
            if scraper_config.get('search_terms'):
                print("✓ Scraper configuration valid")
            else:
                print("⚠ Scraper configuration incomplete")
            
            return True
            
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False
    
    def deploy(self, environment: str, 
               install_deps: bool = True,
               setup_cron: bool = True,
               cron_schedule: str = None) -> bool:
        """
        Run complete deployment process.
        
        Args:
            environment: Environment to deploy
            install_deps: Whether to install dependencies
            setup_cron: Whether to setup cron jobs
            cron_schedule: Custom cron schedule
            
        Returns:
            True if successful, False otherwise
        """
        print("=" * 60)
        print(f"HP Product Classification API - Deployment")
        print(f"Environment: {environment}")
        print("=" * 60)
        
        success = True
        
        # Step 1: Validate prerequisites
        if not self.validate_prerequisites():
            print("❌ Prerequisites validation failed")
            return False
        
        # Step 2: Install dependencies
        if install_deps:
            if not self.install_dependencies():
                print("❌ Dependency installation failed")
                return False
        
        # Step 3: Validate configuration
        if not self.validate_configuration(environment):
            print("❌ Configuration validation failed")
            return False
        
        # Step 4: Initialize database
        if not self.initialize_database(environment):
            print("❌ Database initialization failed")
            return False
        
        # Step 5: Setup cron jobs
        if setup_cron and environment != 'testing':
            if not self.setup_cron_jobs(environment, cron_schedule):
                print("⚠ Cron job setup failed (continuing anyway)")
        
        # Step 6: Run health check
        if not self.run_health_check(environment):
            print("⚠ Health check failed (deployment may have issues)")
        
        print("\n" + "=" * 60)
        if success:
            print("✅ DEPLOYMENT COMPLETED SUCCESSFULLY!")
            print("\nNext steps:")
            print(f"1. Start the API server: python run_api.py")
            print(f"2. Test the API endpoints")
            print(f"3. Monitor log files in logs/ directory")
            if setup_cron:
                print(f"4. Verify cron jobs are running")
        else:
            print("❌ DEPLOYMENT FAILED!")
        print("=" * 60)
        
        return success


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(
        description='Deploy HP Product Classification API'
    )
    
    parser.add_argument(
        'environment',
        choices=['development', 'production', 'testing'],
        help='Environment to deploy'
    )
    parser.add_argument(
        '--no-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    parser.add_argument(
        '--no-cron',
        action='store_true',
        help='Skip cron job setup'
    )
    parser.add_argument(
        '--schedule', '-s',
        help='Custom cron schedule'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration, do not deploy'
    )
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(project_root)
    
    if args.validate_only:
        # Only run validation
        success = deployment_manager.validate_configuration(args.environment)
        sys.exit(0 if success else 1)
    else:
        # Run full deployment
        success = deployment_manager.deploy(
            environment=args.environment,
            install_deps=not args.no_deps,
            setup_cron=not args.no_cron,
            cron_schedule=args.schedule
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()