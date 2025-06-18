#!/usr/bin/env python3
"""
Test script for HP Product Scraper
Runs a limited test to verify functionality
"""

from hp_scraper import HPProductScraper
import logging

# Configure logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scraper():
    """Run a limited test of the scraper"""
    logger.info("Starting limited test of HP Product Scraper...")
    
    # Test configuration - limited scope
    TEST_SEARCH_TERMS = ["cartucho hp 664"]  # Single term for testing
    MAX_PAGES = 1  # Only one page
    
    scraper = HPProductScraper(headless=True)
    
    try:
        # Override the product limit for testing
        original_method = scraper.extract_product_links
        
        def limited_extract_product_links():
            links = original_method()
            return links[:3]  # Limit to 3 products for testing
        
        scraper.extract_product_links = limited_extract_product_links
        
        # Run limited scraping
        scraper.run_scraping(TEST_SEARCH_TERMS, MAX_PAGES)
        
        logger.info(f"Test completed. Collected {len(scraper.products)} products.")
        
        # Print sample data
        if scraper.products:
            logger.info("Sample product data:")
            sample = scraper.products[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"  {key}: {value[:100]}...")
                else:
                    logger.info(f"  {key}: {value}")
        
        return len(scraper.products) > 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    finally:
        scraper.close()

if __name__ == "__main__":
    success = test_scraper()
    if success:
        print("\n✅ Test PASSED - Scraper is working correctly!")
    else:
        print("\n❌ Test FAILED - Check logs for details")

