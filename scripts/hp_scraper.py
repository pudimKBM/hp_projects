#!/usr/bin/env python3
"""
HP Product Web Scraper for E-commerce Platforms
Challenge Sprint - HP Project

This script automates the collection of HP product data from e-commerce platforms
to identify potential counterfeit products. It focuses on HP cartridges and related products.
"""

import time
import logging
import csv
import re
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hp_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HPProductScraper:
    """Main scraper class for HP products on e-commerce platforms"""
    
    def __init__(self, headless: bool = True):
        """
        Initialize the scraper with Chrome WebDriver
        
        Args:
            headless (bool): Run browser in headless mode
        """
        self.driver = None
        self.headless = headless
        self.products = []
        self.setup_driver()
    
    def setup_driver(self):
        """Setup Chrome WebDriver with appropriate options"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def search_mercado_livre(self, search_terms: List[str], max_pages: int = 3) -> List[Dict]:
        """
        Search for HP products on Mercado Livre
        
        Args:
            search_terms (List[str]): List of search terms to use
            max_pages (int): Maximum number of pages to scrape per search term
            
        Returns:
            List[Dict]: List of product data dictionaries
        """
        products = []
        
        for term in search_terms:
            logger.info(f"Searching for: {term}")
            
            try:
                # Navigate to Mercado Livre search
                search_url = f"https://lista.mercadolivre.com.br/{term.replace(' ', '-')}"
                self.driver.get(search_url)
                time.sleep(3)
                
                for page in range(1, max_pages + 1):
                    logger.info(f"Scraping page {page} for term: {term}")
                    
                    # Wait for products to load
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "ui-search-results"))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for search results on page {page}")
                        continue
                    
                    # Extract product links
                    product_links = self.extract_product_links()
                    
                    # Visit each product page and extract details
                    for link in product_links[:10]:  # Limit to 10 products per page
                        product_data = self.extract_product_details(link)
                        if product_data:
                            products.append(product_data)
                            time.sleep(2)  # Be respectful to the server
                    
                    # Navigate to next page
                    if page < max_pages:
                        if not self.go_to_next_page():
                            logger.info(f"No more pages available for term: {term}")
                            break
                    
                    time.sleep(3)  # Pause between pages
                    
            except Exception as e:
                logger.error(f"Error searching for term '{term}': {e}")
                continue
        
        return products
    
    def extract_product_links(self) -> List[str]:
        """Extract product links from search results page"""
        links = []
        try:
            # Find all product links
            product_elements = self.driver.find_elements(By.CSS_SELECTOR, ".ui-search-item__group__element a")
            
            for element in product_elements:
                href = element.get_attribute('href')
                if href and '/MLB-' in href:  # Mercado Livre product identifier
                    links.append(href)
            
            logger.info(f"Found {len(links)} product links")
            
        except Exception as e:
            logger.error(f"Error extracting product links: {e}")
        
        return links
    
    def extract_product_details(self, product_url: str) -> Optional[Dict]:
        """
        Extract detailed information from a product page
        
        Args:
            product_url (str): URL of the product page
            
        Returns:
            Optional[Dict]: Product data dictionary or None if extraction fails
        """
        try:
            self.driver.get(product_url)
            time.sleep(3)
            
            # Wait for product details to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-pdp-title"))
                )
            except TimeoutException:
                logger.warning(f"Timeout loading product page: {product_url}")
                return None
            
            # Extract product information
            product_data = {
                'url': product_url,
                'platform': 'Mercado Livre',
                'scraped_at': datetime.now().isoformat(),
                'title': self.safe_extract_text('.ui-pdp-title'),
                'price': self.extract_price(),
                'seller_name': self.safe_extract_text('.ui-pdp-seller__header__title'),
                'seller_reputation': self.extract_seller_reputation(),
                'reviews_count': self.extract_reviews_count(),
                'rating': self.extract_rating(),
                'description': self.extract_description(),
                'specifications': self.extract_specifications(),
                'images': self.extract_images(),
                'availability': self.safe_extract_text('.ui-pdp-buybox__quantity__available'),
                'shipping_info': self.safe_extract_text('.ui-pdp-shipping')
            }
            
            # Clean and validate data
            product_data = self.clean_product_data(product_data)
            
            logger.info(f"Successfully extracted data for: {product_data.get('title', 'Unknown')}")
            return product_data
            
        except Exception as e:
            logger.error(f"Error extracting product details from {product_url}: {e}")
            return None
    
    def safe_extract_text(self, selector: str, default: str = "") -> str:
        """Safely extract text from an element"""
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except NoSuchElementException:
            return default
        except Exception as e:
            logger.debug(f"Error extracting text with selector '{selector}': {e}")
            return default
    
    def extract_price(self) -> str:
        """Extract and clean price information"""
        price_selectors = [
            '.andes-money-amount__fraction',
            '.price-tag-fraction',
            '.ui-pdp-price__second-line .price-tag-fraction'
        ]
        
        for selector in price_selectors:
            try:
                price_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                price_text = price_element.text.strip()
                
                # Extract currency symbol
                currency_element = self.driver.find_element(By.CSS_SELECTOR, '.andes-money-amount__currency-symbol')
                currency = currency_element.text.strip() if currency_element else 'R$'
                
                return f"{currency} {price_text}"
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_seller_reputation(self) -> str:
        """Extract seller reputation information"""
        try:
            reputation_element = self.driver.find_element(By.CSS_SELECTOR, '.ui-pdp-seller__reputation')
            return reputation_element.text.strip()
        except NoSuchElementException:
            return ""
    
    def extract_reviews_count(self) -> str:
        """Extract number of reviews"""
        try:
            reviews_element = self.driver.find_element(By.CSS_SELECTOR, '.ui-pdp-review__amount')
            return reviews_element.text.strip()
        except NoSuchElementException:
            return ""
    
    def extract_rating(self) -> str:
        """Extract product rating"""
        try:
            rating_element = self.driver.find_element(By.CSS_SELECTOR, '.ui-pdp-review__rating')
            return rating_element.get_attribute('data-rating') or rating_element.text.strip()
        except NoSuchElementException:
            return ""
    
    def extract_description(self) -> str:
        """Extract product description"""
        description_parts = []
        
        # Try different description selectors
        description_selectors = [
            '.ui-pdp-description__content',
            '.item-description',
            '.ui-pdp-description'
        ]
        
        for selector in description_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and len(text) > 20:  # Only add substantial text
                        description_parts.append(text)
            except Exception:
                continue
        
        return " | ".join(description_parts[:3])  # Limit to first 3 substantial parts
    
    def extract_specifications(self) -> str:
        """Extract product specifications"""
        specs = []
        
        try:
            # Look for specification tables
            spec_elements = self.driver.find_elements(By.CSS_SELECTOR, '.ui-pdp-specs__table tr')
            
            for element in spec_elements:
                try:
                    key = element.find_element(By.CSS_SELECTOR, '.ui-pdp-specs__table__column-title').text.strip()
                    value = element.find_element(By.CSS_SELECTOR, '.ui-pdp-specs__table__column-value').text.strip()
                    if key and value:
                        specs.append(f"{key}: {value}")
                except NoSuchElementException:
                    continue
        
        except Exception as e:
            logger.debug(f"Error extracting specifications: {e}")
        
        return " | ".join(specs[:10])  # Limit to first 10 specs
    
    def extract_images(self) -> str:
        """Extract product image URLs"""
        images = []
        
        try:
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, '.ui-pdp-gallery img')
            
            for img in img_elements[:5]:  # Limit to first 5 images
                src = img.get_attribute('src')
                if src and 'http' in src:
                    images.append(src)
        
        except Exception as e:
            logger.debug(f"Error extracting images: {e}")
        
        return " | ".join(images)
    
    def go_to_next_page(self) -> bool:
        """Navigate to the next page of search results"""
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, '.andes-pagination__button--next')
            if next_button.is_enabled():
                next_button.click()
                time.sleep(3)
                return True
        except NoSuchElementException:
            pass
        
        return False
    
    def clean_product_data(self, data: Dict) -> Dict:
        """Clean and normalize product data"""
        # Clean price
        if data.get('price'):
            # Remove extra whitespace and normalize
            data['price'] = re.sub(r'\s+', ' ', data['price']).strip()
        
        # Clean title
        if data.get('title'):
            data['title'] = re.sub(r'\s+', ' ', data['title']).strip()
        
        # Clean description
        if data.get('description'):
            data['description'] = re.sub(r'\s+', ' ', data['description']).strip()[:500]  # Limit length
        
        # Clean specifications
        if data.get('specifications'):
            data['specifications'] = re.sub(r'\s+', ' ', data['specifications']).strip()[:1000]  # Limit length
        
        return data
    
    def save_to_csv(self, filename: str = None):
        """Save collected data to CSV file"""
        if not self.products:
            logger.warning("No products to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hp_products_{timestamp}.csv"
        
        try:
            # Convert to DataFrame for better handling
            df = pd.DataFrame(self.products)
            
            # Reorder columns for better readability
            column_order = [
                'title', 'price', 'seller_name', 'seller_reputation', 
                'reviews_count', 'rating', 'url', 'platform', 
                'description', 'specifications', 'images', 
                'availability', 'shipping_info', 'scraped_at'
            ]
            
            # Reorder columns, keeping any additional ones at the end
            existing_columns = [col for col in column_order if col in df.columns]
            remaining_columns = [col for col in df.columns if col not in column_order]
            final_columns = existing_columns + remaining_columns
            
            df = df[final_columns]
            
            # Save to CSV
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Saved {len(self.products)} products to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def run_scraping(self, search_terms: List[str] = None, max_pages: int = 3):
        """
        Main method to run the scraping process
        
        Args:
            search_terms (List[str]): Search terms to use
            max_pages (int): Maximum pages per search term
        """
        if search_terms is None:
            search_terms = [
                "cartucho hp original",
                "cartucho hp 664",
                "cartucho hp 122",
                "toner hp original",
                "cartucho hp compativel"
            ]
        
        logger.info("Starting HP product scraping...")
        logger.info(f"Search terms: {search_terms}")
        logger.info(f"Max pages per term: {max_pages}")
        
        try:
            # Scrape Mercado Livre
            ml_products = self.search_mercado_livre(search_terms, max_pages)
            self.products.extend(ml_products)
            
            logger.info(f"Scraping completed. Total products collected: {len(self.products)}")
            
            # Save results
            self.save_to_csv()
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
        
        finally:
            self.close()
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")


def main():
    """Main function to run the scraper"""
    # Configuration
    SEARCH_TERMS = [
        "cartucho hp original",
        "cartucho hp 664 original",
        "cartucho hp 122 original", 
        "toner hp original",
        "cartucho hp compativel"
    ]
    
    MAX_PAGES = 2  # Limit pages to avoid being blocked
    
    # Initialize and run scraper
    scraper = HPProductScraper(headless=True)
    
    try:
        scraper.run_scraping(SEARCH_TERMS, MAX_PAGES)
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()

