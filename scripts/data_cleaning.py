#!/usr/bin/env python3
"""
Script for Data Cleaning and Standardization (Entregável 2 - Phase 2)
"""

import pandas as pd
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def clean_price(price_str: str) -> float:
    """Cleans and converts price string to a float."""
    if pd.isna(price_str):
        return None
    # Remove currency symbols, thousands separators, and replace comma with dot for decimals
    cleaned_price = re.sub(r'[^0-9,]', '', str(price_str)).replace(',', '.')
    try:
        return float(cleaned_price)
    except ValueError:
        logger.warning(f"Could not convert price \'{price_str}\' to float. Returning None.")
        return None

def normalize_text(text: str) -> str:
    """Normalizes text by removing extra spaces and line breaks."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces/newlines with single space
    return text

def clean_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning and standardization to the DataFrame."""
    logger.info("Starting data cleaning and standardization...")
    
    # Handle duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows.")

    # Clean and convert 'price' column
    df['price_cleaned'] = df['price'].apply(clean_price)
    
    # Normalize text fields
    text_columns = ['title', 'description', 'specifications', 'seller_name', 'seller_reputation', 'shipping_info']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
            
    # Handle missing values (simple fillna for demonstration, more complex logic might be needed)
    for col in ['reviews_count', 'rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numeric, coerce errors to NaN
            df[col].fillna(0, inplace=True) # Fill NaN with 0 or appropriate value
            
    logger.info("Data cleaning and standardization completed.")
    return df

def save_data(df: pd.DataFrame, filepath: str):
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Successfully saved cleaned data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")

if __name__ == "__main__":
    # Find the latest generated demo CSV file
    import glob
    import os
    
    list_of_files = glob.glob('hp_products_demo_*.csv')
    if not list_of_files:
        logger.error("No demo CSV file found. Please run generate_demo_data.py first.")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Using latest demo file: {latest_file}")
        
        df = load_data(latest_file)
        if not df.empty:
            cleaned_df = clean_and_standardize_data(df)
            
            # Save the cleaned data with a new name
            output_filename = latest_file.replace("hp_products_demo_", "hp_products_cleaned_")
            save_data(cleaned_df, output_filename)
            
            print(f"\n✅ Data cleaning and standardization complete. Cleaned data saved to: {output_filename}")
            print("Check data_cleaning.log for details.")


