from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import pandas as pd
import time
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("americanas_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AmericanasScraper:
    def __init__(self):
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Sets up the Chrome WebDriver."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=options)
            logger.info("Chrome WebDriver initialized successfully.")
        except WebDriverException as e:
            logger.error("Error initializing WebDriver: %s" % e)
            if "executable needs to be in PATH" in str(e):
                logger.error("Please ensure chromedriver is installed and in your system PATH.")
            raise

    def close_driver(self):
        """Closes the WebDriver."""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed.")

    def accept_cookies(self):
        """Attempts to accept cookies if the banner is present."""
        try:
            # Look for common cookie banner elements
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, "cookie-notice-container"))
            )
            accept_button = self.driver.find_element(By.ID, "aceitar-cookies") # Example ID, might vary
            accept_button.click()
            logger.info("Accepted cookies.")
            time.sleep(2) # Give time for banner to disappear
        except TimeoutException:
            logger.info("No cookie banner found or it disappeared quickly.")
        except NoSuchElementException:
            logger.warning("Cookie accept button not found with expected ID.")
        except Exception as e:
            logger.error("Error accepting cookies: %s" % e)

    def search_americanas(self, search_term: str):
        """Navigates to Americanas and performs a search."""
        url = "https://www.americanas.com.br/busca/" + search_term.replace(" ", "-")
        logger.info("Navigating to %s" % url)
        self.driver.get(url)
        self.accept_cookies()
        time.sleep(3) # Wait for page to load

    def extract_product_links(self) -> list:
        """Extracts product links from the search results page."""
        links = []
        try:
            # Wait for product cards to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-testid=\"product-card\"] a[href*=\"/produto/\"]"))
            )
            product_elements = self.driver.find_elements(By.CSS_SELECTOR, "div[data-testid=\"product-card\"] a[href*=\"/produto/\"]")
            for element in product_elements:
                href = element.get_attribute("href")
                if href and "/produto/" in href and "/busca/" not in href:
                    links.append(href)
            logger.info("Found %d product links on the page." % len(links))
        except TimeoutException:
            logger.warning("Timeout waiting for product links on Americanas.")
        except Exception as e:
            logger.error("Error extracting product links: %s" % e)
        return list(set(links)) # Return unique links

    def extract_product_details(self, product_url: str) -> dict:
        """Extracts details from a single product page."""
        logger.info("Extracting details from: %s" % product_url)
        self.driver.get(product_url)
        time.sleep(3) # Wait for page to load

        details = {
            "title": None,
            "price": None,
            "seller_name": None,
            "reviews_count": None,
            "rating": None,
            "description": None,
            "specifications": None,
            "images": None,
            "availability": None,
            "shipping_info": None,
            "url": product_url,
            "platform": "Americanas"
        }

        try:
            # Title
            try:
                details["title"] = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1[data-testid=\"product-title\"]"))
                ).text.strip()
            except:
                logger.warning("Title not found for %s" % product_url)

            # Price
            try:
                price_element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid=\"price-value\"]"))
                )
                details["price"] = price_element.text.strip()
            except:
                logger.warning("Price not found for %s" % product_url)

            # Seller Name
            try:
                seller_element = self.driver.find_element(By.CSS_SELECTOR, "a[data-testid=\"seller-link\"]")
                details["seller_name"] = seller_element.text.strip()
            except NoSuchElementException:
                logger.warning("Seller name not found for %s" % product_url)

            # Reviews Count and Rating
            try:
                reviews_text = self.driver.find_element(By.CSS_SELECTOR, "span[data-testid=\"rating-count\"]").text
                details["reviews_count"] = re.search(r"\d+", reviews_text).group(0) if re.search(r"\d+", reviews_text) else None
                rating_text = self.driver.find_element(By.CSS_SELECTOR, "span[data-testid=\"rating-value\"]").text
                details["rating"] = rating_text.replace(",", ".")
            except NoSuchElementException:
                logger.warning("Reviews/Rating not found for %s" % product_url)

            # Description
            try:
                description_element = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid=\"product-description\"]")
                details["description"] = description_element.text.strip()
            except NoSuchElementException:
                logger.warning("Description not found for %s" % product_url)

            # Specifications (often in a table or list)
            try:
                spec_elements = self.driver.find_elements(By.CSS_SELECTOR, "table[data-testid=\"product-specs\"] tr, ul[data-testid=\"product-specs\"] li")
                specs = []
                for spec in spec_elements:
                    specs.append(spec.text.strip())
                details["specifications"] = " | ".join(specs)
            except NoSuchElementException:
                logger.warning("Specifications not found for %s" % product_url)

            # Images (get first image URL)
            try:
                img_element = self.driver.find_element(By.CSS_SELECTOR, "img[data-testid=\"product-image\"]")
                details["images"] = img_element.get_attribute("src")
            except NoSuchElementException:
                logger.warning("Image not found for %s" % product_url)

            # Availability (simple check for div[data-testid=\"product-availability\"]")
                details["availability"] = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid=\"product-availability\"]").text.strip()
            except NoSuchElementException:
                logger.warning("Availability not found for %s" % product_url)

            # Shipping Info
            try:
                shipping_element = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid=\"shipping-info\"]")
                details["shipping_info"] = shipping_element.text.strip()
            except NoSuchElementException:
                logger.warning("Shipping info not found for %s" % product_url)

        except Exception as e:
            logger.error("Error extracting details for %s: %s" % (product_url, e))

        return details

    def scrape_products(self, search_term: str, max_products: int = 5):
        """Main function to scrape products from Americanas."""
        self.search_americanas(search_term)
        product_links = self.extract_product_links()
        
        all_products_data = []
        for i, link in enumerate(product_links):
            if len(all_products_data) >= max_products:
                logger.info("Reached max_products limit (%d). Stopping." % max_products)
                break
            try:
                product_data = self.extract_product_details(link)
                if product_data["title"]:
                    all_products_data.append(product_data)
                    logger.info("Successfully scraped product %d/%d: %s" % (i+1, len(product_links), product_data.get("title", "N/A")))
                else:
                    logger.warning("Skipping product %d/%d due to missing title: %s" % (i+1, len(product_links), link))
            except Exception as e:
                logger.error("Failed to scrape product %s: %s" % (link, e))
            time.sleep(2) # Be polite

        return pd.DataFrame(all_products_data)

if __name__ == "__main__":
    scraper = AmericanasScraper()
    try:
        search_term = "cartucho hp 664"
        max_products_to_scrape = 3 # Limit for testing
        df_americanas = scraper.scrape_products(search_term, max_products_to_scrape)
        
        if not df_americanas.empty:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = "hp_products_americanas_%s.csv" % timestamp
            df_americanas.to_csv(output_filename, index=False, encoding="utf-8")
            logger.info("Scraped data saved to %s" % output_filename)
            print("\n✅ Americanas scraping complete. Data saved to: %s" % output_filename)
        else:
            print("\n❌ No data scraped from Americanas.")
            
    except Exception as e:
        logger.critical("An error occurred during the scraping process: %s" % e)
    finally:
        scraper.close_driver()


