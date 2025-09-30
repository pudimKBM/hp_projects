#!/usr/bin/env python3
"""
Cron job setup script for HP Product Classification API.

This script configures automated scheduling for scraping and classification jobs
using the system's cron scheduler or Windows Task Scheduler.
"""

import os
import sys
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add parent directory to path for src imports
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from production_api.config import create_app_config


class CronSetup:
    """Handles cron job setup for different operating systems."""
    
    def __init__(self, project_root: str):
        """
        Initialize cron setup.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.system = platform.system().lower()
        self.python_executable = sys.executable
        
    def get_default_schedule(self) -> str:
        """Get default cron schedule from configuration."""
        try:
            config = create_app_config()
            return config.DEFAULT_CRON_SCHEDULE
        except Exception:
            return "0 */6 * * *"  # Every 6 hours as fallback
    
    def create_cron_command(self, environment: str = 'production') -> str:
        """
        Create the command to run scraping job.
        
        Args:
            environment: Environment to run in
            
        Returns:
            Command string
        """
        # Use standalone scraper for cron jobs to avoid Flask dependencies
        scraper_script = self.project_root / 'run_scraper_standalone.py'
        
        # Set environment variables in the command
        env_vars = f"FLASK_ENV={environment}"
        
        # Create log file path
        log_file = self.project_root / 'logs' / 'cron_scraper.log'
        
        # Create full command
        command = f'{env_vars} {self.python_executable} {scraper_script} --verbose --log-file {log_file}'
        
        return command
    
    def setup_linux_cron(self, schedule: str, environment: str = 'production') -> bool:
        """
        Setup cron job on Linux/Unix systems.
        
        Args:
            schedule: Cron schedule expression
            environment: Environment to run in
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], 
                                  capture_output=True, text=True)
            current_crontab = result.stdout if result.returncode == 0 else ""
            
            # Create job command
            command = self.create_cron_command(environment)
            
            # Create log file path
            log_file = self.project_root / 'logs' / 'cron_scraper.log'
            log_file.parent.mkdir(exist_ok=True)
            
            # Create cron job entry
            job_comment = "# HP Product Classification API - Automated Scraping"
            job_entry = f"{schedule} {command} >> {log_file} 2>&1"
            
            # Check if job already exists
            if job_comment in current_crontab or 'run_scraper.py' in current_crontab:
                print("Cron job already exists. Updating...")
                # Remove existing job
                lines = current_crontab.split('\n')
                filtered_lines = []
                skip_next = False
                
                for line in lines:
                    if job_comment in line:
                        skip_next = True
                        continue
                    if skip_next and 'run_scraper.py' in line:
                        skip_next = False
                        continue
                    if line.strip():
                        filtered_lines.append(line)
                
                current_crontab = '\n'.join(filtered_lines)
            
            # Add new job
            new_crontab = current_crontab.strip()
            if new_crontab:
                new_crontab += '\n'
            new_crontab += f"{job_comment}\n{job_entry}\n"
            
            # Install new crontab
            process = subprocess.Popen(['crontab', '-'], 
                                     stdin=subprocess.PIPE, text=True)
            process.communicate(input=new_crontab)
            
            if process.returncode == 0:
                print(f"✓ Cron job installed successfully")
                print(f"  Schedule: {schedule}")
                print(f"  Command: {command}")
                print(f"  Log file: {log_file}")
                return True
            else:
                print("✗ Failed to install cron job")
                return False
                
        except FileNotFoundError:
            print("✗ Cron not available on this system")
            return False
        except Exception as e:
            print(f"✗ Error setting up cron job: {e}")
            return False
    
    def setup_windows_task(self, schedule: str, environment: str = 'production') -> bool:
        """
        Setup scheduled task on Windows systems.
        
        Args:
            schedule: Cron schedule expression (will be converted)
            environment: Environment to run in
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert cron schedule to Windows task scheduler format
            # This is a simplified conversion - for production use a proper library
            interval_minutes = self._cron_to_minutes(schedule)
            
            if interval_minutes is None:
                print("✗ Could not convert cron schedule to Windows format")
                return False
            
            # Create task command
            command = self.create_cron_command(environment)
            
            # Task name
            task_name = "HP_Product_Classification_Scraper"
            
            # Create scheduled task using schtasks
            schtasks_cmd = [
                'schtasks', '/create',
                '/tn', task_name,
                '/tr', command,
                '/sc', 'minute',
                '/mo', str(interval_minutes),
                '/f'  # Force overwrite if exists
            ]
            
            result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Windows scheduled task created successfully")
                print(f"  Task name: {task_name}")
                print(f"  Interval: Every {interval_minutes} minutes")
                print(f"  Command: {command}")
                return True
            else:
                print(f"✗ Failed to create scheduled task: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("✗ schtasks not available on this system")
            return False
        except Exception as e:
            print(f"✗ Error setting up Windows task: {e}")
            return False
    
    def _cron_to_minutes(self, cron_schedule: str) -> Optional[int]:
        """
        Convert simple cron schedule to minutes interval.
        
        Args:
            cron_schedule: Cron schedule string
            
        Returns:
            Interval in minutes or None if conversion fails
        """
        # Simple conversion for common patterns
        # "0 */6 * * *" -> every 6 hours = 360 minutes
        # "*/30 * * * *" -> every 30 minutes
        
        parts = cron_schedule.strip().split()
        if len(parts) != 5:
            return None
        
        minute, hour, day, month, weekday = parts
        
        # Every X hours pattern: "0 */X * * *"
        if minute == "0" and hour.startswith("*/"):
            try:
                hours = int(hour[2:])
                return hours * 60
            except ValueError:
                pass
        
        # Every X minutes pattern: "*/X * * * *"
        if minute.startswith("*/"):
            try:
                return int(minute[2:])
            except ValueError:
                pass
        
        # Default to 6 hours if can't parse
        return 360
    
    def remove_cron_job(self) -> bool:
        """
        Remove existing cron job.
        
        Returns:
            True if successful, False otherwise
        """
        if self.system == 'windows':
            return self._remove_windows_task()
        else:
            return self._remove_linux_cron()
    
    def _remove_linux_cron(self) -> bool:
        """Remove cron job on Linux/Unix systems."""
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("No crontab found")
                return True
            
            current_crontab = result.stdout
            
            # Remove HP scraper jobs
            lines = current_crontab.split('\n')
            filtered_lines = []
            skip_next = False
            
            for line in lines:
                if "HP Product Classification API" in line:
                    skip_next = True
                    continue
                if skip_next and 'run_scraper.py' in line:
                    skip_next = False
                    continue
                if line.strip():
                    filtered_lines.append(line)
            
            # Install filtered crontab
            new_crontab = '\n'.join(filtered_lines)
            if new_crontab.strip():
                new_crontab += '\n'
            
            process = subprocess.Popen(['crontab', '-'], 
                                     stdin=subprocess.PIPE, text=True)
            process.communicate(input=new_crontab)
            
            if process.returncode == 0:
                print("✓ Cron job removed successfully")
                return True
            else:
                print("✗ Failed to remove cron job")
                return False
                
        except Exception as e:
            print(f"✗ Error removing cron job: {e}")
            return False
    
    def _remove_windows_task(self) -> bool:
        """Remove scheduled task on Windows systems."""
        try:
            task_name = "HP_Product_Classification_Scraper"
            
            result = subprocess.run([
                'schtasks', '/delete', '/tn', task_name, '/f'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Windows scheduled task removed successfully")
                return True
            else:
                print(f"Task may not exist or removal failed: {result.stderr}")
                return True  # Don't fail if task doesn't exist
                
        except Exception as e:
            print(f"✗ Error removing Windows task: {e}")
            return False
    
    def list_jobs(self) -> None:
        """List existing scheduled jobs."""
        if self.system == 'windows':
            self._list_windows_tasks()
        else:
            self._list_linux_cron()
    
    def _list_linux_cron(self) -> None:
        """List cron jobs on Linux/Unix systems."""
        try:
            result = subprocess.run(['crontab', '-l'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                crontab = result.stdout.strip()
                if crontab:
                    print("Current cron jobs:")
                    print("-" * 40)
                    print(crontab)
                else:
                    print("No cron jobs found")
            else:
                print("No crontab found for current user")
        except Exception as e:
            print(f"Error listing cron jobs: {e}")
    
    def _list_windows_tasks(self) -> None:
        """List scheduled tasks on Windows systems."""
        try:
            result = subprocess.run([
                'schtasks', '/query', '/fo', 'table'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                hp_tasks = [line for line in lines if 'HP_Product' in line]
                
                if hp_tasks:
                    print("HP Product Classification tasks:")
                    print("-" * 40)
                    for task in hp_tasks:
                        print(task)
                else:
                    print("No HP Product Classification tasks found")
            else:
                print("Error querying scheduled tasks")
        except Exception as e:
            print(f"Error listing Windows tasks: {e}")


def main():
    """Main entry point for cron setup script."""
    parser = argparse.ArgumentParser(
        description='Setup automated scheduling for HP Product Classification API'
    )
    
    parser.add_argument(
        'action',
        choices=['install', 'remove', 'list', 'status'],
        help='Action to perform'
    )
    parser.add_argument(
        '--schedule', '-s',
        help='Cron schedule expression (default: from config)'
    )
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'production', 'testing'],
        default='production',
        help='Environment to run jobs in (default: production)'
    )
    
    args = parser.parse_args()
    
    # Initialize cron setup
    cron_setup = CronSetup(project_root)
    
    print(f"HP Product Classification API - Cron Setup")
    print(f"Operating System: {platform.system()}")
    print(f"Python: {sys.executable}")
    print(f"Project Root: {project_root}")
    print("-" * 60)
    
    if args.action == 'install':
        schedule = args.schedule or cron_setup.get_default_schedule()
        print(f"Installing cron job with schedule: {schedule}")
        
        if platform.system().lower() == 'windows':
            success = cron_setup.setup_windows_task(schedule, args.environment)
        else:
            success = cron_setup.setup_linux_cron(schedule, args.environment)
        
        if success:
            print("\n✅ Cron job setup completed successfully!")
            print("\nNext steps:")
            print("1. Verify the job is scheduled correctly")
            print("2. Check log files for job execution")
            print("3. Monitor system performance")
        else:
            print("\n❌ Cron job setup failed!")
            sys.exit(1)
    
    elif args.action == 'remove':
        print("Removing existing cron jobs...")
        success = cron_setup.remove_cron_job()
        if success:
            print("✅ Cron job removal completed!")
        else:
            print("❌ Cron job removal failed!")
            sys.exit(1)
    
    elif args.action == 'list':
        print("Listing scheduled jobs...")
        cron_setup.list_jobs()
    
    elif args.action == 'status':
        print("Checking cron job status...")
        cron_setup.list_jobs()
        
        # Also show configuration
        try:
            config = create_app_config(args.environment)
            scraper_config = config.get_scraper_config()
            print(f"\nScraper Configuration ({args.environment}):")
            print("-" * 30)
            print(f"Default schedule: {scraper_config.get('cron_schedule')}")
            print(f"Search terms: {scraper_config.get('search_terms')}")
            print(f"Max pages per term: {scraper_config.get('max_pages_per_term')}")
        except Exception as e:
            print(f"Error getting configuration: {e}")


if __name__ == '__main__':
    main()