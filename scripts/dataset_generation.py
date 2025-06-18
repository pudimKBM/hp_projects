#!/usr/bin/env python3
"""
Script to generate a structured dataset with relevant features and binary labels.
"""
import pandas as pd
import os
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_latest_csv_file(pattern):
    """Get the latest CSV file matching the pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def generate_structured_dataset(input_file: str, output_dir: str = "."):
    """Generates a structured dataset with selected features and binary labels."""
    try:
        df = pd.read_csv(input_file)
        logger.info("Loaded data from %s" % input_file)
    except FileNotFoundError:
        logger.error("Input file not found: %s" % input_file)
        return
    except Exception as e:
        logger.error("Error loading data: %s" % e)
        return

    # Define relevant features
    # These are chosen based on their potential to differentiate between original and suspicious products.
    # 'title', 'description', 'seller_name' are textual features.
    # 'price_numeric', 'rating_numeric', 'reviews_count' are numerical features.
    # 'platform' can be a categorical feature.
    # 'is_original', 'is_xl', 'product_type', 'has_sealed_info', 'has_invoice_info', 'has_warranty_info' are derived features.
    # 'label' is the target feature.
    
    features = [
        "title", "description", "seller_name", "platform",
        "price_numeric", "rating_numeric", "reviews_count",
        "is_original", "is_xl", "product_type",
        "has_sealed_info", "has_invoice_info", "has_warranty_info",
        "label" # This will be our target variable
    ]

    # Ensure all features exist in the DataFrame, fill missing with None/NaN
    for col in features:
        if col not in df.columns:
            df[col] = None

    df_structured = df[features].copy()

    # Convert 'label' to binary target: 1 for 'original', 0 for 'pirate/suspicious' or 'mixed_signals' or 'unknown'
    # This assumes a binary classification task. If multi-class is needed, this step changes.
    df_structured["target_is_original"] = df_structured["label"].apply(lambda x: 1 if x == "original" else 0)
    
    # Drop the original 'label' column if 'target_is_original' is sufficient
    df_structured = df_structured.drop(columns=["label"])

    # Handle NaN values in numerical columns that might be used for ML
    numeric_cols = ["price_numeric", "rating_numeric", "reviews_count"]
    for col in numeric_cols:
        if col in df_structured.columns:
            df_structured[col] = df_structured[col].fillna(0) # Fill NaN with 0 or a more appropriate strategy

    # Fill NaN in textual/categorical columns with empty string or 'unknown'
    text_cat_cols = ["title", "description", "seller_name", "platform", "product_type"]
    for col in text_cat_cols:
        if col in df_structured.columns:
            df_structured[col] = df_structured[col].fillna("")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, "hp_products_structured_dataset_%s.csv" % timestamp)
    df_structured.to_csv(output_filename, index=False, encoding="utf-8")
    logger.info("Structured dataset saved to %s" % output_filename)
    print("\n✅ Structured dataset generation complete. Data saved to: %s" % output_filename)

if __name__ == "__main__":
    latest_labeled_file = get_latest_csv_file("hp_products_labeled_*.csv")

    if latest_labeled_file:
        generate_structured_dataset(latest_labeled_file)
    else:
        logger.error("No labeled data file found for dataset generation.")
        print("\n❌ Structured dataset generation failed: No labeled data file found.")


