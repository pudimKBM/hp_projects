"""
Automated Scraper Service Wrapper

This service wraps the existing HP scraper functionality and integrates it
with the database for automated product collection and storage.
"""

import logging
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

from flask import current_app
from sqlalchemy.exc import SQLAlchemyError

from ..models import db, Product, ScrapingJob
from ..utils.database import get_db_session
from scripts.hp_scraper import HPProductScraper


logger = logging.getLogger(__name__)


class ScraperService:
    """Service class for automated HP product scraping with database integration."""
    
    def __init__(self):
        """Initialize the scraper service."""
        self.scraper = None
        self.current_job = None
        
    def create_scraping_job(self, job_type: str = "manual", search_terms: List[str] = None) -> ScrapingJob:
        """
        Create a new scraping job record in the database.
        
        Args:
            job_type: Type of job ("manual" or "scheduled")
            search_terms: List of search terms to use
            
        Returns:
            ScrapingJob: Created job record
        """
        try:
            job = ScrapingJob(
                job_type=job_type,
                status='running',
                search_terms=search_terms or current_app.config.get('DEFAULT_SEARCH_TERMS', [])
            )
            
            db.session.add(job)
            db.session.commit()
            
            logger.info(f"Created scraping job {job.id} of type '{job_type}'")
            return job
            
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Failed to create scraping job: {e}")
            raise
    
    def run_scraping_job(self, job_type: str = "manual", search_terms: List[str] = None, 
                        max_pages: int = None, max_retries: int = 3) -> Tuple[int, int, List[str]]:
        """
        Execute a complete scraping job with database integration.
        
        Args:
            job_type: Type of job ("manual" or "scheduled")
            search_terms: List of search terms to scrape
            max_pages: Maximum pages per search term
            max_retries: Maximum retry attempts for failed operations
            
        Returns:
            Tuple[int, int, List[str]]: (products_found, products_stored, errors)
        """
        # Set defaults from config
        if search_terms is None:
            search_terms = current_app.config.get('DEFAULT_SEARCH_TERMS', [])
        if max_pages is None:
            max_pages = current_app.config.get('MAX_PAGES_PER_TERM', 3)
        
        # Create job record
        self.current_job = self.create_scraping_job(job_type, search_terms)
        
        products_found = 0
        products_stored = 0
        errors = []
        
        try:
            logger.info(f"Starting scraping job {self.current_job.id}")
            logger.info(f"Search terms: {search_terms}")
            logger.info(f"Max pages per term: {max_pages}")
            
            # Initialize scraper with retry logic
            for attempt in range(max_retries):
                try:
                    self.scraper = HPProductScraper(headless=True)
                    break
                except Exception as e:
                    error_msg = f"Failed to initialize scraper (attempt {attempt + 1}): {e}"
                    logger.warning(error_msg)
                    if attempt == max_retries - 1:
                        errors.append(error_msg)
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Scrape products
            scraped_products = []
            for term in search_terms:
                try:
                    logger.info(f"Scraping term: '{term}'")
                    term_products = self._scrape_search_term(term, max_pages, max_retries)
                    scraped_products.extend(term_products)
                    products_found += len(term_products)
                    
                    # Add delay between search terms
                    delay = current_app.config.get('SCRAPER_DELAY', 2)
                    time.sleep(delay)
                    
                except Exception as e:
                    error_msg = f"Failed to scrape term '{term}': {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Store products in database
            products_stored = self._store_products(scraped_products, errors)
            
            # Mark job as completed
            self.current_job.mark_completed(
                products_found=products_found,
                products_processed=products_stored,
                errors=errors if errors else None
            )
            
            # Track scraper performance
            try:
                from production_api.app.services.performance_service import get_performance_service
                from production_api.app.utils.logging import log_scraper_event
                
                job_duration = (datetime.utcnow() - self.current_job.started_at).total_seconds()
                
                # Track performance metrics
                perf_service = get_performance_service()
                perf_service.record_scraper_performance(
                    job_duration_seconds=job_duration,
                    products_found=products_found,
                    products_processed=products_stored,
                    errors=errors
                )
                
                # Log scraper event
                log_scraper_event(
                    job_id=self.current_job.id,
                    job_type=job_type,
                    status='completed',
                    products_found=products_found,
                    products_processed=products_stored,
                    duration_seconds=job_duration
                )
                
            except Exception as e:
                logger.warning(f"Failed to track scraper performance: {e}")
            
            logger.info(f"Scraping job {self.current_job.id} completed successfully")
            logger.info(f"Products found: {products_found}, stored: {products_stored}")
            
        except Exception as e:
            error_msg = f"Scraping job {self.current_job.id} failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Mark job as failed
            self.current_job.mark_failed(errors)
            
        finally:
            # Clean up scraper
            if self.scraper:
                try:
                    self.scraper.close()
                except Exception as e:
                    logger.warning(f"Error closing scraper: {e}")
            
            # Commit job status
            try:
                db.session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Failed to commit job status: {e}")
                db.session.rollback()
        
        return products_found, products_stored, errors
    
    def _scrape_search_term(self, search_term: str, max_pages: int, max_retries: int) -> List[Dict]:
        """
        Scrape products for a single search term with retry logic.
        
        Args:
            search_term: Search term to scrape
            max_pages: Maximum pages to scrape
            max_retries: Maximum retry attempts
            
        Returns:
            List[Dict]: List of scraped product data
        """
        products = []
        
        for attempt in range(max_retries):
            try:
                # Use the existing scraper's search functionality
                term_products = self.scraper.search_mercado_livre([search_term], max_pages)
                products.extend(term_products)
                break
                
            except Exception as e:
                logger.warning(f"Scraping attempt {attempt + 1} failed for term '{search_term}': {e}")
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + (time.time() % 1)
                time.sleep(wait_time)
                
                # Reinitialize scraper if needed
                try:
                    if self.scraper:
                        self.scraper.close()
                    self.scraper = HPProductScraper(headless=True)
                except Exception as init_error:
                    logger.warning(f"Failed to reinitialize scraper: {init_error}")
        
        return products
    
    def _store_products(self, products: List[Dict], errors: List[str]) -> int:
        """
        Store scraped products in the database with validation and deduplication.
        
        Args:
            products: List of product data dictionaries
            errors: List to append any storage errors
            
        Returns:
            int: Number of products successfully stored
        """
        stored_count = 0
        
        for product_data in products:
            try:
                # Clean and validate product data
                cleaned_data = self._clean_product_data(product_data)
                if not self._validate_product_data(cleaned_data):
                    logger.warning(f"Invalid product data, skipping: {cleaned_data.get('title', 'Unknown')}")
                    continue
                
                # Check for duplicates
                if self._is_duplicate_product(cleaned_data):
                    logger.debug(f"Duplicate product found, skipping: {cleaned_data.get('title', 'Unknown')}")
                    continue
                
                # Create product record
                product = Product(
                    title=cleaned_data.get('title', ''),
                    description=cleaned_data.get('description', ''),
                    price_numeric=cleaned_data.get('price_numeric'),
                    seller_name=cleaned_data.get('seller_name', ''),
                    rating_numeric=cleaned_data.get('rating_numeric'),
                    reviews_count=cleaned_data.get('reviews_count', 0),
                    platform=cleaned_data.get('platform', 'mercadolivre'),
                    product_type=self._extract_product_type(cleaned_data.get('title', '')),
                    url=cleaned_data.get('url', ''),
                    raw_data=product_data  # Store original scraped data
                )
                
                db.session.add(product)
                stored_count += 1
                
                # Commit in batches to avoid memory issues
                if stored_count % 10 == 0:
                    db.session.commit()
                    logger.debug(f"Committed batch of products, total stored: {stored_count}")
                
            except Exception as e:
                error_msg = f"Failed to store product '{product_data.get('title', 'Unknown')}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                db.session.rollback()
                continue
        
        # Final commit
        try:
            db.session.commit()
            logger.info(f"Successfully stored {stored_count} products")
        except SQLAlchemyError as e:
            logger.error(f"Failed to commit final batch: {e}")
            db.session.rollback()
        
        return stored_count
    
    def _clean_product_data(self, data: Dict) -> Dict:
        """
        Clean and normalize product data for database storage.
        
        Args:
            data: Raw product data dictionary
            
        Returns:
            Dict: Cleaned product data
        """
        cleaned = {}
        
        # Clean title
        title = data.get('title', '').strip()
        cleaned['title'] = re.sub(r'\s+', ' ', title)[:500]  # Limit length
        
        # Clean description
        description = data.get('description', '').strip()
        cleaned['description'] = re.sub(r'\s+', ' ', description)[:2000]  # Limit length
        
        # Extract and clean price
        price_str = data.get('price', '')
        cleaned['price_numeric'] = self._extract_numeric_price(price_str)
        
        # Clean seller name
        seller = data.get('seller_name', '').strip()
        cleaned['seller_name'] = re.sub(r'\s+', ' ', seller)[:200]  # Limit length
        
        # Extract and clean rating
        rating_str = data.get('rating', '')
        cleaned['rating_numeric'] = self._extract_numeric_rating(rating_str)
        
        # Extract reviews count
        reviews_str = data.get('reviews_count', '')
        cleaned['reviews_count'] = self._extract_reviews_count(reviews_str)
        
        # Copy other fields
        cleaned['platform'] = data.get('platform', 'mercadolivre')
        cleaned['url'] = data.get('url', '')
        
        return cleaned
    
    def _extract_numeric_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from price string."""
        if not price_str:
            return None
        
        try:
            # Remove currency symbols and extract numbers
            price_clean = re.sub(r'[^\d,.]', '', price_str)
            
            # Handle Brazilian number format (comma as decimal separator)
            if ',' in price_clean and '.' in price_clean:
                # Format like 1.234,56
                price_clean = price_clean.replace('.', '').replace(',', '.')
            elif ',' in price_clean:
                # Check if comma is decimal separator or thousands separator
                parts = price_clean.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    # Decimal separator
                    price_clean = price_clean.replace(',', '.')
                else:
                    # Thousands separator
                    price_clean = price_clean.replace(',', '')
            
            return float(price_clean)
            
        except (ValueError, AttributeError):
            return None
    
    def _extract_numeric_rating(self, rating_str: str) -> Optional[float]:
        """Extract numeric rating from rating string."""
        if not rating_str:
            return None
        
        try:
            # Extract first number found
            match = re.search(r'(\d+(?:[.,]\d+)?)', rating_str)
            if match:
                rating_clean = match.group(1).replace(',', '.')
                rating = float(rating_clean)
                # Ensure rating is in valid range (0-5)
                return min(max(rating, 0), 5)
            
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def _extract_reviews_count(self, reviews_str: str) -> int:
        """Extract reviews count from reviews string."""
        if not reviews_str:
            return 0
        
        try:
            # Extract numbers from string
            numbers = re.findall(r'\d+', reviews_str)
            if numbers:
                return int(numbers[0])
                
        except (ValueError, AttributeError):
            pass
        
        return 0
    
    def _extract_product_type(self, title: str) -> str:
        """Extract product type from title."""
        title_lower = title.lower()
        
        if 'cartucho' in title_lower:
            return 'cartucho'
        elif 'toner' in title_lower:
            return 'toner'
        elif 'tinta' in title_lower:
            return 'tinta'
        else:
            return 'outros'
    
    def _validate_product_data(self, data: Dict) -> bool:
        """
        Validate that product data meets minimum requirements.
        
        Args:
            data: Product data dictionary
            
        Returns:
            bool: True if data is valid
        """
        # Must have title
        if not data.get('title'):
            return False
        
        # Must have URL
        if not data.get('url'):
            return False
        
        # Title must contain HP-related keywords
        title_lower = data.get('title', '').lower()
        hp_keywords = ['hp', 'hewlett', 'packard']
        if not any(keyword in title_lower for keyword in hp_keywords):
            return False
        
        return True
    
    def _is_duplicate_product(self, data: Dict) -> bool:
        """
        Check if product already exists in database.
        
        Args:
            data: Product data dictionary
            
        Returns:
            bool: True if duplicate exists
        """
        try:
            # Check by URL (most reliable)
            if data.get('url'):
                existing = Product.query.filter_by(url=data['url']).first()
                if existing:
                    return True
            
            # Check by title and seller (fallback)
            if data.get('title') and data.get('seller_name'):
                existing = Product.query.filter_by(
                    title=data['title'],
                    seller_name=data['seller_name']
                ).first()
                if existing:
                    return True
            
        except SQLAlchemyError as e:
            logger.warning(f"Error checking for duplicates: {e}")
        
        return False
    
    def get_job_status(self, job_id: int) -> Optional[Dict]:
        """
        Get status information for a scraping job.
        
        Args:
            job_id: ID of the scraping job
            
        Returns:
            Optional[Dict]: Job status information or None if not found
        """
        try:
            job = ScrapingJob.query.get(job_id)
            if job:
                return job.to_dict()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving job status: {e}")
        
        return None
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict]:
        """
        Get recent scraping jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List[Dict]: List of recent job information
        """
        try:
            jobs = ScrapingJob.query.order_by(
                ScrapingJob.started_at.desc()
            ).limit(limit).all()
            
            return [job.to_dict() for job in jobs]
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving recent jobs: {e}")
            return []
    
    def cleanup_old_jobs(self, days_to_keep: int = 30) -> int:
        """
        Clean up old scraping job records.
        
        Args:
            days_to_keep: Number of days of job history to keep
            
        Returns:
            int: Number of jobs deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            deleted_count = ScrapingJob.query.filter(
                ScrapingJob.started_at < cutoff_date
            ).delete()
            
            db.session.commit()
            logger.info(f"Cleaned up {deleted_count} old scraping jobs")
            
            return deleted_count
            
        except SQLAlchemyError as e:
            logger.error(f"Error cleaning up old jobs: {e}")
            db.session.rollback()
            return 0


# Convenience functions for external use
def run_manual_scraping(search_terms: List[str] = None, max_pages: int = None) -> Tuple[int, int, List[str]]:
    """
    Run a manual scraping job.
    
    Args:
        search_terms: List of search terms to scrape
        max_pages: Maximum pages per search term
        
    Returns:
        Tuple[int, int, List[str]]: (products_found, products_stored, errors)
    """
    service = ScraperService()
    return service.run_scraping_job("manual", search_terms, max_pages)


def run_scheduled_scraping() -> Tuple[int, int, List[str]]:
    """
    Run a scheduled scraping job with default parameters.
    
    Returns:
        Tuple[int, int, List[str]]: (products_found, products_stored, errors)
    """
    service = ScraperService()
    return service.run_scraping_job("scheduled")