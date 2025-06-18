#!/usr/bin/env python3
"""
Script for Exploratory Data Analysis (EDA) on HP product data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud
import os
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eda.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

def get_latest_csv_file(pattern):
    """Get the latest CSV file matching the pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully from %s" % file_path)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s" % file_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Error loading data from %s: %s" % (file_path, e))
        return pd.DataFrame()

def analyze_price_distribution(df: pd.DataFrame, output_dir: str):
    """Analyzes and visualizes the distribution of prices."""
    if "price_numeric" in df.columns and not df["price_numeric"].isnull().all():
        plt.figure(figsize=(10, 6))
        sns.histplot(df["price_numeric"].dropna(), kde=True, bins=30)
        plt.title("Distribuição de Preços")
        plt.xlabel("Preço (R$)")
        plt.ylabel("Frequência")        plt.grid(axis='y', alpha=0.75)# Corrected line
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "price_distribution_eda.png"))
        plt.close()
        logger.info("Price distribution plot saved.")
    else:
        logger.warning("Price data not available for distribution analysis.")

def generate_wordclouds_and_ngrams(df: pd.DataFrame, output_dir: str):
    """Generates wordclouds and analyzes n-grams for titles and descriptions."""
    stop_words = set(stopwords.words("portuguese"))
    
    # Combine all text for wordcloud
    all_titles = " ".join(df["title"].dropna().astype(str))
    all_descriptions = " ".join(df["description"].dropna().astype(str))
    
    # WordCloud for Titles
    if all_titles:
        wordcloud_titles = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(all_titles)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_titles, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - Títulos de Produtos")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_titles.png"))
        plt.close()
        logger.info("Word cloud for titles saved.")
    else:
        logger.warning("No title data for word cloud generation.")

    # WordCloud for Descriptions
    if all_descriptions:
        wordcloud_descriptions = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(all_descriptions)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_descriptions, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - Descrições de Produtos")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_descriptions.png"))
        plt.close()
        logger.info("Word cloud for descriptions saved.")
    else:
        logger.warning("No description data for word cloud generation.")

    # N-grams for Titles
    def get_ngrams(text, n):
        # Explicitly specify language for word_tokenize
        tokens = word_tokenize(text.lower(), language='portuguese')
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return list(ngrams(tokens, n))

    all_title_ngrams = []
    for title in df["title"].dropna().astype(str):
        all_title_ngrams.extend(get_ngrams(title, 2)) # Bigrams

    if all_title_ngrams:
        bigram_counts = Counter(all_title_ngrams)
        most_common_bigrams = bigram_counts.most_common(10)
        logger.info("Most common bigrams in titles: %s" % most_common_bigrams)
        # You can also visualize these if needed
    else:
        logger.warning("No bigrams found for titles.")

def analyze_correlations(df: pd.DataFrame, output_dir: str):
    """Analyzes and visualizes correlations between numerical variables."""
    numeric_cols = ["price_numeric", "rating_numeric", "reviews_count"]
    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.dropna()

    if not df_numeric.empty and len(df_numeric.columns) > 1:
        correlation_matrix = df_numeric.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matriz de Correlação entre Variáveis Numéricas")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
        logger.info("Correlation matrix plot saved.")
    else:
        logger.warning("Insufficient numerical data for correlation analysis.")

def analyze_label_distribution(df: pd.DataFrame, output_dir: str):
    """Analyzes and visualizes the distribution of product labels."""
    if "label" in df.columns and not df["label"].isnull().all():
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x="label", palette="viridis")
        plt.title("Distribuição de Anúncios por Rótulo")
        plt.xlabel("Rótulo")
        plt.ylabel("Contagem")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "label_distribution.png"))
        plt.close()
        logger.info("Label distribution plot saved.")
    else:
        logger.warning("Label data not available for distribution analysis.")

def identify_useful_features(df: pd.DataFrame):
    """Identifies and logs potentially useful features for modeling."""
    logger.info("\n--- Potenciais Features Úteis para Modelagem ---")
    
    # Features from heuristic labeling
    logger.info("Features from heuristic labeling: seller_name, price_numeric, rating_numeric, reviews_count, is_original, is_xl, product_type, has_sealed_info, has_invoice_info, has_warranty_info.")

    # Textual features (for NLP)
    logger.info("Textual features (for NLP): title, description.")
    
    # Interaction features (example)
    if "price_numeric" in df.columns and "rating_numeric" in df.columns:
        logger.info("Potential interaction: price_numeric * rating_numeric (e.g., value for money).")

    # Categorical features to be one-hot encoded
    if "platform" in df.columns:
        logger.info("Categorical feature for one-hot encoding: platform.")
    if "product_type" in df.columns:
        logger.info("Categorical feature for one-hot encoding: product_type.")

    logger.info("--------------------------------------------------")

if __name__ == "__main__":
    output_directory = "."
    latest_structured_dataset = get_latest_csv_file("hp_products_structured_dataset_mock_*.csv")
    latest_labeled_dataset = get_latest_csv_file("hp_products_labeled_mock_*.csv")

    if latest_structured_dataset:
        df_structured = load_data(latest_structured_dataset)
        if not df_structured.empty:
            logger.info("Starting EDA on structured dataset.")
            analyze_price_distribution(df_structured, output_directory)
            analyze_correlations(df_structured, output_directory)
            identify_useful_features(df_structured)
        else:
            logger.error("Structured dataset is empty. Cannot perform EDA.")
    else:
        logger.error("No structured dataset found for EDA.")

    if latest_labeled_dataset:
        df_labeled = load_data(latest_labeled_dataset)
        if not df_labeled.empty:
            logger.info("Starting EDA on labeled dataset for text analysis and label distribution.")
            generate_wordclouds_and_ngrams(df_labeled, output_directory)
            analyze_label_distribution(df_labeled, output_directory)
        else:
            logger.error("Labeled dataset is empty. Cannot perform text analysis or label distribution.")
    else:
        logger.error("No labeled dataset found for text analysis or label distribution.")

    print("\n✅ EDA script finished. Check eda.log for details and generated PNGs for visualizations.")


