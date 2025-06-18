#!/usr/bin/env python3
"""
HP Product Web Scraper for E-commerce Platforms - Updated Version
Challenge Sprint - HP Project

This script automates the collection of HP product data from e-commerce platforms
to identify potential counterfeit products. Updated with correct selectors for Mercado Livre.

Author: RPA Team
Date: 2024
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


class HPProductScraperV2:
    """Updated scraper class for HP products on e-commerce platforms"""
    
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
    
    def search_mercado_livre(self, search_terms: List[str], max_pages: int = 2) -> List[Dict]:
        """
        Search for HP products on Mercado Livre with updated selectors
        
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
                time.sleep(5)  # Wait for page to load
                
                # Accept cookies if present
                try:
                    cookie_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Aceitar cookies')]"))
                    )
                    cookie_button.click()
                    time.sleep(2)
                except TimeoutException:
                    pass  # No cookie banner or already accepted
                
                for page in range(1, max_pages + 1):
                    logger.info(f"Scraping page {page} for term: {term}")
                    
                    # Wait for products to load
                    try:
                        WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/p/MLB'], a[href*='/MLB-']"))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for search results on page {page}")
                        continue
                    
                    # Extract product links
                    product_links = self.extract_product_links_v2()
                    logger.info(f"Found {len(product_links)} product links on page {page}")
                    
                    # Visit each product page and extract details
                    for i, link in enumerate(product_links[:5]):  # Limit to 5 products per page
                        logger.info(f"Processing product {i+1}/{len(product_links[:5])}: {link}")
                        product_data = self.extract_product_details_v2(link)
                        if product_data:
                            products.append(product_data)
                        time.sleep(3)  # Be respectful to the server
                    
                    # Navigate to next page
                    if page < max_pages:
                        if not self.go_to_next_page_v2():
                            logger.info(f"No more pages available for term: {term}")
                            break
                    
                    time.sleep(5)  # Pause between pages
                    
            except Exception as e:
                logger.error(f"Error searching for term '{term}': {e}")
                continue
        
        return products
    
    def extract_product_links_v2(self) -> List[str]:
        """Extract product links from search results page with updated selectors"""
        links = []
        try:
            # Find all product links using multiple selectors
            selectors = [
                "a[href*='/p/MLB']",
                "a[href*='/MLB-']",
                "a[href*='mercadolivre.com.br/'][href*='MLB']"
            ]
            
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    href = element.get_attribute('href')
                    if href and ('MLB' in href) and href not in links:
                        # Filter out non-product links
                        if any(exclude in href for exclude in ['lista.mercadolivre', 'search', 'category']):
                            continue
                        links.append(href)
            
            # Remove duplicates while preserving order
            unique_links = []
            seen = set()
            for link in links:
                if link not in seen:
                    unique_links.append(link)
                    seen.add(link)
            
            logger.info(f"Found {len(unique_links)} unique product links")
            return unique_links[:10]  # Limit to 10 products per page
            
        except Exception as e:
            logger.error(f"Error extracting product links: {e}")
            return []
    
    def extract_product_details_v2(self, product_url: str) -> Optional[Dict]:
        """
        Extract detailed information from a product page with updated selectors
        
        Args:
            product_url (str): URL of the product page
            
        Returns:
            Optional[Dict]: Product data dictionary or None if extraction fails
        """
        try:
            self.driver.get(product_url)
            time.sleep(5)
            
            # Wait for product details to load
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.TAG_NAME, "h1")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='product-title']")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-pdp-title"))
                    )
                )
            except TimeoutException:
                logger.warning(f"Timeout loading product page: {product_url}")
                return None
            
            # Extract product information using multiple fallback selectors
            product_data = {
                'url': product_url,
                'platform': 'Mercado Livre',
                'scraped_at': datetime.now().isoformat(),
                'title': self.extract_title_v2(),
                'price': self.extract_price_v2(),
                'seller_name': self.extract_seller_v2(),
                'seller_reputation': self.extract_seller_reputation_v2(),
                'reviews_count': self.extract_reviews_count_v2(),
                'rating': self.extract_rating_v2(),
                'description': self.extract_description_v2(),
                'specifications': self.extract_specifications_v2(),
                'images': self.extract_images_v2(),
                'availability': self.extract_availability_v2(),
                'shipping_info': self.extract_shipping_v2()
            }
            
            # Clean and validate data
            product_data = self.clean_product_data(product_data)
            
            logger.info(f"Successfully extracted data for: {product_data.get('title', 'Unknown')[:50]}...")
            return product_data
            
        except Exception as e:
            logger.error(f"Error extracting product details from {product_url}: {e}")
            return None
    
    def extract_title_v2(self) -> str:
        """Extract product title with multiple fallback selectors"""
        selectors = [
            "h1",
            "[data-testid='product-title']",
            ".ui-pdp-title",
            ".item-title",
            "h1.ui-pdp-title"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                title = element.text.strip()
                if title and len(title) > 5:  # Valid title should be substantial
                    return title
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_price_v2(self) -> str:
        """Extract price with multiple fallback selectors"""
        # Try to find price elements
        price_selectors = [
            ".andes-money-amount__fraction",
            "[data-testid='price-part']",
            ".price-tag-fraction",
            ".ui-pdp-price__second-line .price-tag-fraction",
            "span[class*='price']",
            "span[class*='amount']"
        ]
        
        for selector in price_selectors:
            try:
                price_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                price_text = price_element.text.strip()
                
                if price_text and any(char.isdigit() for char in price_text):
                    # Try to find currency symbol
                    try:
                        currency_element = self.driver.find_element(By.CSS_SELECTOR, ".andes-money-amount__currency-symbol")
                        currency = currency_element.text.strip()
                        return f"{currency} {price_text}"
                    except NoSuchElementException:
                        return f"R$ {price_text}"
                        
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_seller_v2(self) -> str:
        """Extract seller name with multiple fallback selectors"""
        selectors = [
            "[data-testid='seller-info'] span",
            ".ui-pdp-seller__header__title",
            "span[class*='seller']",
            "a[class*='seller']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                seller = element.text.strip()
                if seller and len(seller) > 2:
                    return seller
            except NoSuchElementException:
                continue
        
        # Try to find seller in page text
        try:
            page_text = self.driver.page_source
            if "Vendido por" in page_text:
                # Extract seller name after "Vendido por"
                import re
                match = re.search(r'Vendido por\s*([^<\n]+)', page_text)
                if match:
                    return match.group(1).strip()
        except:
            pass
        
        return ""
    
    def extract_seller_reputation_v2(self) -> str:
        """Extract seller reputation"""
        selectors = [
            "[data-testid='seller-reputation']",
            ".ui-pdp-seller__reputation",
            "span[class*='reputation']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                return element.text.strip()
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_reviews_count_v2(self) -> str:
        """Extract number of reviews"""
        selectors = [
            "[data-testid='reviews-summary']",
            ".ui-pdp-review__amount",
            "span[class*='review']",
            "span[class*='opinion']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                # Extract numbers from text like "(3317)" or "3317 opiniões"
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    return numbers[0]
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_rating_v2(self) -> str:
        """Extract product rating"""
        selectors = [
            "[data-testid='rating-average']",
            ".ui-pdp-review__rating",
            "span[class*='rating']",
            "[class*='star']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                rating = element.get_attribute('data-rating') or element.text.strip()
                if rating and any(char.isdigit() for char in rating):
                    return rating
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_description_v2(self) -> str:
        """Extract product description"""
        description_parts = []
        
        # Try different description selectors
        selectors = [
            "[data-testid='product-description']",
            ".ui-pdp-description__content",
            ".item-description",
            ".ui-pdp-description",
            "div[class*='description']"
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and len(text) > 20:  # Only add substantial text
                        description_parts.append(text)
            except Exception:
                continue
        
        # Also try to get text from page source
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            # Look for product specifications or descriptions
            lines = page_text.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['impressora', 'cartucho', 'tinta', 'hp', 'compatível']):
                    if len(line.strip()) > 30 and len(line.strip()) < 200:
                        description_parts.append(line.strip())
        except:
            pass
        
        return " | ".join(description_parts[:3])  # Limit to first 3 substantial parts
    
    def extract_specifications_v2(self) -> str:
        """Extract product specifications"""
        specs = []
        
        try:
            # Look for specification tables or lists
            spec_selectors = [
                "[data-testid='specifications'] tr",
                ".ui-pdp-specs__table tr",
                "table tr",
                "dl dt, dl dd"
            ]
            
            for selector in spec_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and len(text) > 5 and len(text) < 100:
                        specs.append(text)
        
        except Exception as e:
            logger.debug(f"Error extracting specifications: {e}")
        
        return " | ".join(specs[:8])  # Limit to first 8 specs
    
    def extract_images_v2(self) -> str:
        """Extract product image URLs"""
        images = []
        
        try:
            img_selectors = [
                "[data-testid='product-image'] img",
                ".ui-pdp-gallery img",
                "img[src*='http']"
            ]
            
            for selector in img_selectors:
                img_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for img in img_elements[:3]:  # Limit to first 3 images
                    src = img.get_attribute('src')
                    if src and 'http' in src and 'mercadolibre' in src:
                        images.append(src)
        
        except Exception as e:
            logger.debug(f"Error extracting images: {e}")
        
        return " | ".join(images)
    
    def extract_availability_v2(self) -> str:
        """Extract availability information"""
        selectors = [
            "[data-testid='stock-info']",
            ".ui-pdp-buybox__quantity__available",
            "span[class*='stock']",
            "span[class*='available']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                return element.text.strip()
            except NoSuchElementException:
                continue
        
        return ""
    
    def extract_shipping_v2(self) -> str:
        """Extract shipping information"""
        selectors = [
            "[data-testid='shipping-info']",
            ".ui-pdp-shipping",
            "div[class*='shipping']",
            "span[class*='frete']"
        ]
        
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                return element.text.strip()
            except NoSuchElementException:
                continue
        
        return ""
    
    def go_to_next_page_v2(self) -> bool:
        """Navigate to the next page of search results"""
        try:
            # Try different next button selectors
            next_selectors = [
                ".andes-pagination__button--next",
                "a[title='Siguiente']",
                "a[aria-label='Siguiente']",
                ".ui-search-pagination .ui-search-pagination__button--next"
            ]
            
            for selector in next_selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if next_button.is_enabled() and next_button.is_displayed():
                        next_button.click()
                        time.sleep(5)
                        return True
                except NoSuchElementException:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error navigating to next page: {e}")
        
        return False
    
    def clean_product_data(self, data: Dict) -> Dict:
        """Clean and normalize product data"""
        # Clean price
        if data.get('price'):
            data['price'] = re.sub(r'\s+', ' ', data['price']).strip()
        
        # Clean title
        if data.get('title'):
            data['title'] = re.sub(r'\s+', ' ', data['title']).strip()
        
        # Clean description
        if data.get('description'):
            data['description'] = re.sub(r'\s+', ' ', data['description']).strip()[:500]
        
        # Clean specifications
        if data.get('specifications'):
            data['specifications'] = re.sub(r'\s+', ' ', data['specifications']).strip()[:1000]
        
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
    
    def run_scraping(self, search_terms: List[str] = None, max_pages: int = 2):
        """
        Main method to run the scraping process
        
        Args:
            search_terms (List[str]): Search terms to use
            max_pages (int): Maximum pages per search term
        """
        if search_terms is None:
            search_terms = [
                "cartucho hp 664",
                "cartucho hp 122",
                "toner hp original"
            ]
        
        logger.info("Starting HP product scraping (Version 2)...")
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
    """Main function to run the updated scraper"""
    # Configuration
    SEARCH_TERMS = [
        "cartucho hp 664",
        "cartucho hp 122"
    ]
    
    MAX_PAGES = 1  # Limit pages to avoid being blocked
    
    # Initialize and run scraper
    scraper = HPProductScraperV2(headless=True)
    
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

