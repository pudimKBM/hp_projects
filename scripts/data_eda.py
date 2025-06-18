#!/usr/bin/env python3
"""
Script for Exploratory Data Analysis (EDA) (Entregável 2 - Phase 4)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import logging
import glob
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_eda.log"),
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

def descriptive_statistics(df: pd.DataFrame):
    """Generates descriptive statistics for numerical columns."""
    logger.info("Generating descriptive statistics...")
    print("\n--- Descriptive Statistics ---")
    if 'price_cleaned' in df.columns:
        print("Price (cleaned):")
        print(df['price_cleaned'].describe())
    if 'reviews_count' in df.columns:
        print("\nReviews Count:")
        print(df['reviews_count'].describe())
    if 'rating' in df.columns:
        print("\nRating:")
        print(df['rating'].describe())
    logger.info("Descriptive statistics generated.")

def identify_outliers(df: pd.DataFrame):
    """Identifies outliers using IQR method for numerical columns."""
    logger.info("Identifying outliers...")
    print("\n--- Outlier Identification (IQR Method) ---")
    numerical_cols = ['price_cleaned', 'reviews_count', 'rating']
    for col in numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                print(f"\nOutliers in {col}:\n{outliers[[col, 'title', 'url']]}")
            else:
                print(f"\nNo significant outliers found in {col}.")
    logger.info("Outlier identification completed.")

def lexical_analysis(df: pd.DataFrame, column: str = 'description', top_n: int = 20):
    """Performs lexical analysis on a text column (e.g., description)."""
    logger.info(f"Performing lexical analysis on \'{column}\' column...")
    all_words = []
    for text in df[column].dropna():
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(top_n)
    
    print(f"\n--- Top {top_n} Most Common Words in {column} ---")
    for word, count in common_words:
        print(f"{word}: {count}")
    logger.info("Lexical analysis completed.")

def visualize_data(df: pd.DataFrame):
    """Generates basic visualizations."""
    logger.info("Generating visualizations...")
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Price Distribution
    if 'price_cleaned' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['price_cleaned'].dropna(), kde=True, bins=10)
        plt.title('Distribution of Product Prices')
        plt.xlabel('Price (R$)')
        plt.ylabel('Frequency')
        plt.savefig('price_distribution.png')
        plt.close()
        logger.info("Saved price_distribution.png")

    # Rating Distribution
    if 'rating' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='rating', data=df.dropna(subset=['rating']))
        plt.title('Distribution of Product Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig('rating_distribution.png')
        plt.close()
        logger.info("Saved rating_distribution.png")

    # Product Classification Distribution
    if 'classificacao_produto' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='classificacao_produto', data=df)
        plt.title('Product Classification Distribution')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.savefig('product_classification.png')
        plt.close()
        logger.info("Saved product_classification.png")

    logger.info("Visualizations generated.")

if __name__ == "__main__":
    list_of_files = glob.glob("hp_products_enriched_*.csv")
    if not list_of_files:
        logger.error("No enriched CSV file found. Please run data_enrichment.py first.")
    else:
        latest_enriched_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Using latest enriched file: {latest_enriched_file}")
        
        df = load_data(latest_enriched_file)
        if not df.empty:
            descriptive_statistics(df)
            identify_outliers(df)
            lexical_analysis(df, column='description')
            visualize_data(df)
            
            print("\n✅ Exploratory Data Analysis complete. Check data_eda.log for details and generated PNG files for visualizations.")


