#!/usr/bin/env python3
"""
Script to apply heuristic labeling to product data.
"""
import pandas as pd
import re
import os
import glob
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("labeling.log"),
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

def apply_heuristic_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """Applies heuristic rules to label products as 'original' or 'pirate/suspicious'."""
    df["label"] = "unknown"
    df["confidence"] = 0.0

    # Convert price and rating to numeric if not already
    if "price" in df.columns and not pd.api.types.is_numeric_dtype(df["price"]):
        df["price_numeric"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[^\d.,]", "", regex=True).str.replace(",", "."),
            errors=\



            "coerce"
        )
    else:
        df["price_numeric"] = df["price"]

    if "rating" in df.columns and not pd.api.types.is_numeric_dtype(df["rating"]):
        df["rating_numeric"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["rating_numeric"] = df["rating"]

    # Heuristic Rules
    for index, row in df.iterrows():
        is_original = False
        is_suspicious = False
        current_confidence = 0.0

        title = str(row["title"]).lower() if pd.notna(row["title"]) else ""
        description = str(row["description"]).lower() if pd.notna(row["description"]) else ""
        seller_name = str(row["seller_name"]).lower() if pd.notna(row["seller_name"]) else ""
        price = row["price_numeric"]
        rating = row["rating_numeric"]
        reviews_count = row["reviews_count"]
        platform = str(row["platform"]).lower() if pd.notna(row["platform"]) else ""

        # Rule 1: Official Store Names (strong indicator for original)
        official_stores = ["hp oficial", "loja oficial hp", "hp store", "hp do brasil"]
        if any(store in seller_name for store in official_stores):
            is_original = True
            current_confidence += 0.4

        # Rule 2: Keywords in Title/Description for Originality
        original_keywords = ["original", "genuíno", "autêntico", "direto da fábrica", "com nota fiscal", "lacrado"]
        if any(keyword in title for keyword in original_keywords) or any(keyword in description for keyword in original_keywords):
            is_original = True
            current_confidence += 0.3

        # Rule 3: Keywords in Title/Description for Suspicious/Pirate
        suspicious_keywords = ["compatível", "similar", "genérico", "remanufaturado", "alternativo", "paralelo", "recarga"]
        if any(keyword in title for keyword in suspicious_keywords) or any(keyword in description for keyword in suspicious_keywords):
            is_suspicious = True
            current_confidence += 0.4

        # Rule 4: Price Deviation (requires a baseline, using a simple threshold for now)
        # This is a simplified rule. A more robust approach would involve dynamic price ranges.
        if pd.notna(price):
            if price < 30.0: # Arbitrary low price threshold
                is_suspicious = True
                current_confidence += 0.3
            elif price > 200.0: # Arbitrary high price threshold (could indicate bundle or special)
                is_original = True # Could be a high-end original or bundle
                current_confidence += 0.1

        # Rule 5: Low Rating and High Review Count (suspicious)
        if pd.notna(rating) and pd.notna(reviews_count):
            if rating < 3.0 and reviews_count > 50: # Many reviews but low rating
                is_suspicious = True
                current_confidence += 0.2

        # Rule 6: Grammatical Errors (suspicious)
        # This is a very basic check and would need a more sophisticated NLP model for accuracy.
        # For demonstration, we'll look for common informalities or obvious typos.
        common_typos = ["cartucho hp", "cartucho h p", "cartucho h.p."]
        if any(typo in title for typo in common_typos) and "cartucho hp" not in title:
            is_suspicious = True
            current_confidence += 0.1

        # Rule 7: Unknown/Generic Seller Names (suspicious)
        generic_seller_patterns = ["eletrônicos xyz", "distribuidora abc", "loja online"]
        if any(pattern in seller_name for pattern in generic_seller_patterns):
            is_suspicious = True
            current_confidence += 0.1

        # Final Labeling Logic
        if is_original and not is_suspicious:
            df.at[index, "label"] = "original"
            df.at[index, "confidence"] = min(current_confidence, 1.0)
        elif is_suspicious and not is_original:
            df.at[index, "label"] = "pirate/suspicious"
            df.at[index, "confidence"] = min(current_confidence, 1.0)
        elif is_original and is_suspicious: # Conflicting indicators
            df.at[index, "label"] = "mixed_signals"
            df.at[index, "confidence"] = 0.5 # Neutral confidence
        else:
            df.at[index, "label"] = "unknown"
            df.at[index, "confidence"] = 0.0

    return df

if __name__ == "__main__":
    latest_combined_file = get_latest_csv_file("hp_products_combined_*.csv")

    if latest_combined_file:
        logger.info("Loading latest combined data from %s" % latest_combined_file)
        df_combined = pd.read_csv(latest_combined_file)
        
        df_labeled = apply_heuristic_labeling(df_combined)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_filename = "hp_products_labeled_%s.csv" % timestamp
        df_labeled.to_csv(output_filename, index=False, encoding="utf-8")
        logger.info("Labeled data saved to %s" % output_filename)
        print("\n✅ Heuristic labeling complete. Labeled data saved to: %s" % output_filename)
    else:
        logger.error("No combined data file found for labeling.")
        print("\n❌ Heuristic labeling failed: No combined data file found.")


