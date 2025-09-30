#!/usr/bin/env python3
"""
Standalone scraper entry point that doesn't require Flask imports.

This script can be used for cron jobs and standalone scraping without
initializing the full Flask application.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add parent directory to path for src imports
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_standalone_scraping(search_terms: List[str],
                          max_pages: int = 5,
                          delay: float = 2.0,
                          verbose: bool = False) -> bool:
    """
    Run standalone scraping without Flask dependencies.
    
    Args:
        search_terms: List of search terms to scrape
        max_pages: Maximum pages per search term
        delay: Delay between requests in seconds
        verbose: Whether to show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import scraper directly
        sys.path.append(str(Path(project_root).parent / 'scripts'))
        from hp_scraper import HPScraper
        
        print(f"Starting standalone scraping job at {datetime.now()}")
        print(f"Search terms: {search_terms}")
        print(f"Max pages per term: {max_pages}")
        print(f"Delay between requests: {delay}s")
        print("-" * 60)
        
        total_products = 0
        
        # Initialize scraper
        scraper = HPScraper()
        
        # Run scraping for each search term
        for i, term in enumerate(search_terms, 1):
            print(f"\n[{i}/{len(search_terms)}] Scraping: '{term}'")
            
            try:
                # Run scraping for this term
                products = scraper.scrape_products(
                    search_term=term,
                    max_pages=max_pages,
                    delay=delay
                )
                
                if products:
                    print(f"  ✓ Scraped {len(products)} products")
                    total_products += len(products)
                    
                    # Save products to CSV
                    output_file = Path(project_root) / 'data' / f'scraped_{term.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    import pandas as pd
                    df = pd.DataFrame(products)
                    df.to_csv(output_file, index=False)
                    print(f"  ✓ Saved to: {output_file}")
                    
                else:
                    print(f"  ⚠ No products found for '{term}'")
                    
            except Exception as e:
                print(f"  ✗ Error scraping '{term}': {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Final summary
        print("\n" + "=" * 60)
        print("STANDALONE SCRAPING SUMMARY")
        print("=" * 60)
        print(f"Total products scraped: {total_products}")
        print(f"Job completed at: {datetime.now()}")
        
        return total_products > 0
        
    except ImportError as e:
        print(f"Error importing scraper: {e}")
        print("Make sure the hp_scraper.py script exists in the scripts/ directory")
        return False
    except Exception as e:
        print(f"Error running standalone scraping: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point for standalone scraper."""
    parser = argparse.ArgumentParser(
        description='HP Product Classification API - Standalone Scraper'
    )
    
    parser.add_argument(
        '--terms', '-t',
        nargs='+',
        default=['cartucho hp original', 'toner hp', 'cartucho hp 664'],
        help='Search terms to use (space-separated)'
    )
    parser.add_argument(
        '--max-pages', '-p',
        type=int,
        default=5,
        help='Maximum pages per search term (default: 5)'
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=2.0,
        help='Delay between requests in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--log-file', '-l',
        help='Log file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Run scraping
    success = run_standalone_scraping(
        search_terms=args.terms,
        max_pages=args.max_pages,
        delay=args.delay,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()