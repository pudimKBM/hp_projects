#!/usr/bin/env python3
"""
Entry point for running manual scraping jobs.

This script allows manual execution of HP product scraping with
classification and database storage.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, List

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add parent directory to path for src imports
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from production_api.app import create_app, db
from production_api.app.services.scraper_service import ScraperService
from production_api.app.services.batch_processor import BatchProcessor
from production_api.config import create_app_config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_scraping_job(search_terms: Optional[List[str]] = None,
                    max_pages: Optional[int] = None,
                    classify: bool = True,
                    verbose: bool = False) -> bool:
    """
    Run a manual scraping job.
    
    Args:
        search_terms: List of search terms to use
        max_pages: Maximum pages per search term
        classify: Whether to classify scraped products
        verbose: Whether to show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get environment configuration
        config_name = os.environ.get('FLASK_ENV', 'development')
        
        # Create Flask application
        app = create_app(config_name)
        
        with app.app_context():
            # Initialize services
            scraper_service = ScraperService()
            batch_processor = BatchProcessor() if classify else None
            
            # Use provided search terms or get from config
            if not search_terms:
                search_terms = app.config.get('DEFAULT_SEARCH_TERMS', [
                    'cartucho hp original',
                    'toner hp',
                    'cartucho hp 664'
                ])
            
            # Use provided max pages or get from config
            if not max_pages:
                max_pages = app.config.get('MAX_PAGES_PER_TERM', 5)
            
            print(f"Starting scraping job at {datetime.now()}")
            print(f"Search terms: {search_terms}")
            print(f"Max pages per term: {max_pages}")
            print(f"Classification enabled: {classify}")
            print("-" * 60)
            
            total_products = 0
            total_classified = 0
            
            # Run scraping for each search term
            for i, term in enumerate(search_terms, 1):
                print(f"\n[{i}/{len(search_terms)}] Scraping: '{term}'")
                
                try:
                    # Run scraping
                    job_result = scraper_service.run_scraping_job(
                        search_terms=[term],
                        max_pages_per_term=max_pages,
                        job_type='manual'
                    )
                    
                    if job_result and job_result.get('success'):
                        products_found = job_result.get('products_scraped', 0)
                        total_products += products_found
                        print(f"  ✓ Scraped {products_found} products")
                        
                        # Run classification if enabled
                        if classify and batch_processor and products_found > 0:
                            print(f"  → Classifying products...")
                            
                            # Get newly scraped products for classification
                            classified_count = batch_processor.process_unclassified_products()
                            total_classified += classified_count
                            print(f"  ✓ Classified {classified_count} products")
                        
                    else:
                        error_msg = job_result.get('error', 'Unknown error') if job_result else 'No result returned'
                        print(f"  ✗ Scraping failed: {error_msg}")
                        
                except Exception as e:
                    print(f"  ✗ Error scraping '{term}': {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
            
            # Final summary
            print("\n" + "=" * 60)
            print("SCRAPING JOB SUMMARY")
            print("=" * 60)
            print(f"Total products scraped: {total_products}")
            if classify:
                print(f"Total products classified: {total_classified}")
            print(f"Job completed at: {datetime.now()}")
            
            return total_products > 0
            
    except Exception as e:
        print(f"Error running scraping job: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def list_recent_jobs(limit: int = 10) -> None:
    """
    List recent scraping jobs.
    
    Args:
        limit: Maximum number of jobs to show
    """
    try:
        # Get environment configuration
        config_name = os.environ.get('FLASK_ENV', 'development')
        
        # Create Flask application
        app = create_app(config_name)
        
        with app.app_context():
            from production_api.app.models import ScrapingJob
            
            # Get recent jobs
            jobs = ScrapingJob.query.order_by(ScrapingJob.started_at.desc()).limit(limit).all()
            
            if not jobs:
                print("No scraping jobs found.")
                return
            
            print(f"\nRecent Scraping Jobs (last {len(jobs)}):")
            print("-" * 80)
            print(f"{'ID':<5} {'Type':<10} {'Status':<12} {'Started':<20} {'Products':<10} {'Duration'}")
            print("-" * 80)
            
            for job in jobs:
                duration = ""
                if job.completed_at and job.started_at:
                    delta = job.completed_at - job.started_at
                    duration = f"{delta.total_seconds():.1f}s"
                
                print(f"{job.id:<5} {job.job_type:<10} {job.status:<12} "
                      f"{job.started_at.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{job.products_found or 0:<10} {duration}")
                
    except Exception as e:
        print(f"Error listing jobs: {e}")


def show_scraper_status() -> None:
    """Show current scraper service status."""
    try:
        # Get environment configuration
        config_name = os.environ.get('FLASK_ENV', 'development')
        
        # Create Flask application
        app = create_app(config_name)
        
        with app.app_context():
            scraper_service = ScraperService()
            
            # Get scraper configuration
            config = app.config.get_scraper_config()
            
            print("\nScraper Configuration:")
            print("-" * 30)
            print(f"Search terms: {config.get('search_terms', [])}")
            print(f"Max pages per term: {config.get('max_pages_per_term', 'not set')}")
            print(f"Delay between requests: {config.get('delay', 'not set')}s")
            print(f"Timeout: {config.get('timeout', 'not set')}s")
            print(f"Retry attempts: {config.get('retry_attempts', 'not set')}")
            print(f"User agent: {config.get('user_agent', 'not set')[:50]}...")
            
            # Check scraper health
            print(f"\nScraper Health:")
            print("-" * 20)
            try:
                health = scraper_service.check_health()
                print(f"Status: {'✓ Healthy' if health.get('healthy') else '✗ Unhealthy'}")
                if health.get('last_run'):
                    print(f"Last run: {health['last_run']}")
                if health.get('error'):
                    print(f"Error: {health['error']}")
            except Exception as e:
                print(f"Health check failed: {e}")
                
    except Exception as e:
        print(f"Error getting scraper status: {e}")


def main():
    """Main entry point for the scraper script."""
    parser = argparse.ArgumentParser(
        description='HP Product Classification API - Manual Scraper'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run scraping job
    run_parser = subparsers.add_parser('run', help='Run scraping job')
    run_parser.add_argument(
        '--terms', '-t',
        nargs='+',
        help='Search terms to use (space-separated)'
    )
    run_parser.add_argument(
        '--max-pages', '-p',
        type=int,
        help='Maximum pages per search term'
    )
    run_parser.add_argument(
        '--no-classify',
        action='store_true',
        help='Skip classification of scraped products'
    )
    run_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    # List recent jobs
    list_parser = subparsers.add_parser('list', help='List recent scraping jobs')
    list_parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Maximum number of jobs to show (default: 10)'
    )
    
    # Show status
    status_parser = subparsers.add_parser('status', help='Show scraper status')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(args, 'verbose', False))
    
    # Execute command
    if args.command == 'run':
        success = run_scraping_job(
            search_terms=args.terms,
            max_pages=args.max_pages,
            classify=not args.no_classify,
            verbose=args.verbose
        )
        sys.exit(0 if success else 1)
        
    elif args.command == 'list':
        list_recent_jobs(args.limit)
        
    elif args.command == 'status':
        show_scraper_status()
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()