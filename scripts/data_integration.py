#!/usr/bin/env python3
"""
Script to integrate and consolidate data from Mercado Livre and Americanas.
"""
import pandas as pd
import os
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def integrate_data(mercado_livre_file: str, americanas_file: str, output_dir: str = "."):
    """Integrates data from Mercado Livre and Americanas CSVs."""
    try:
        df_ml = pd.read_csv(mercado_livre_file)
        logger.info("Loaded Mercado Livre data from %s" % mercado_livre_file)
    except FileNotFoundError:
        logger.error("Mercado Livre file not found: %s" % mercado_livre_file)
        return
    except Exception as e:
        logger.error("Error loading Mercado Livre data: %s" % e)
        return

    try:
        df_am = pd.read_csv(americanas_file)
        logger.info("Loaded Americanas data from %s" % americanas_file)
    except FileNotFoundError:
        logger.error("Americanas file not found: %s" % americanas_file)
        return
    except Exception as e:
        logger.error("Error loading Americanas data: %s" % e)
        return

    # Standardize column names for concatenation
    # Assuming similar columns exist or can be mapped
    # For this example, let's assume 'title', 'price', 'url', 'platform' are common
    # and other columns might be unique or require specific handling.
    # In a real scenario, a more robust mapping would be needed.
    common_columns = [
        "title", "price", "seller_name", "reviews_count", "rating",
        "description", "specifications", "images", "availability",
        "shipping_info", "url", "platform"
    ]

    # Ensure all common columns exist in both DataFrames, fill missing with None/NaN
    for col in common_columns:
        if col not in df_ml.columns:
            df_ml[col] = None
        if col not in df_am.columns:
            df_am[col] = None

    df_ml = df_ml[common_columns]
    df_am = df_am[common_columns]

    # Concatenate the dataframes
    df_combined = pd.concat([df_ml, df_am], ignore_index=True)
    logger.info("Combined dataframes successfully. Total rows: %d" % len(df_combined))

    # Deduplicate based on a combination of relevant columns (e.g., title and price)
    # A more sophisticated deduplication might involve fuzzy matching for titles.
    df_combined.drop_duplicates(subset=["title", "price"], inplace=True)
    logger.info("Deduplicated data. Remaining rows: %d" % len(df_combined))

    # Save the combined data
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, "hp_products_combined_%s.csv" % timestamp)
    df_combined.to_csv(output_filename, index=False, encoding="utf-8")
    logger.info("Combined data saved to %s" % output_filename)
    print("\n✅ Data integration complete. Combined data saved to: %s" % output_filename)

if __name__ == "__main__":
    # Find the latest Mercado Livre and Americanas demo CSVs
    mercado_livre_files = glob.glob("hp_products_demo_*.csv")
    americanas_files = glob.glob("hp_products_americanas_*.csv")

    latest_ml_file = max(mercado_livre_files, key=os.path.getctime) if mercado_livre_files else None
    latest_am_file = max(americanas_files, key=os.path.getctime) if americanas_files else None

    if latest_ml_file and latest_am_file:
        integrate_data(latest_ml_file, latest_am_file)
    else:
        logger.error("Could not find both Mercado Livre and Americanas CSV files for integration.")
        print("\n❌ Data integration failed: Missing CSV files.")


