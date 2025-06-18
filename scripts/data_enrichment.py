import pandas as pd
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_enrichment.log"),
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

def classify_product(title: str, description: str) -> str:
    """Classifies product as Original, Compatible, or Suspicious."""
    text = (str(title) + " " + str(description)).lower()
    if "original" in text and "compativel" not in text:
        return "Original"
    elif "compativel" in text:
        return "Compatível"
    elif "recarregado" in text or "generico" in text or "similar" in text:
        return "Suspeito"
    else:
        return "Não Classificado"

def extract_technical_attributes(description: str, specifications: str) -> dict:
    """Extracts technical attributes using regex."""
    attributes = {}
    text = (str(description) + " " + str(specifications)).lower()

    # Example: Extracting 'Modelo'
    model_match = re.search(r"modelo:?\s*([a-z0-9\-]+)", text)
    if model_match:
        attributes["modelo"] = model_match.group(1).upper()

    # Example: Extracting 'Cor'
    color_match = re.search(r"cor:?\s*(preto|colorido|tricolor|magenta|ciano|amarelo)", text)
    if color_match:
        attributes["cor"] = color_match.group(1).capitalize()

    # Example: Extracting 'Rendimento'
    yield_match = re.search(r"rendimento:?\s*(\d+)\s*paginas", text)
    if yield_match:
        attributes["rendimento_paginas"] = int(yield_match.group(1))

    return attributes

def check_reliability_indicators(title: str, description: str) -> dict:
    """Checks for reliability indicators."""
    indicators = {
        "tem_lacrado": False,
        "tem_nota_fiscal": False,
        "tem_garantia": False
    }
    text = (str(title) + " " + str(description)).lower()

    if "lacrado" in text:
        indicators["tem_lacrado"] = True
    if "nota fiscal" in text or "nf-e" in text:
        indicators["tem_nota_fiscal"] = True
    if "garantia" in text:
        indicators["tem_garantia"] = True

    return indicators

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches the DataFrame with new derived columns."""
    logger.info("Starting data enrichment...")
    
    # Product Classification
    df["classificacao_produto"] = df.apply(lambda row: classify_product(row["title"], row["description"]), axis=1)
    
    # Technical Attributes Extraction
    df["atributos_tecnicos"] = df.apply(lambda row: extract_technical_attributes(row["description"], row["specifications"]), axis=1)
    
    # Expand technical attributes into separate columns
    tech_attrs_df = pd.json_normalize(df["atributos_tecnicos"])
    df = pd.concat([df.drop(columns=["atributos_tecnicos"]), tech_attrs_df], axis=1)
    
    # Reliability Indicators
    reliability_df = df.apply(lambda row: check_reliability_indicators(row["title"], row["description"]), axis=1)
    reliability_df = pd.json_normalize(reliability_df)
    df = pd.concat([df, reliability_df], axis=1)

    logger.info("Data enrichment completed.")
    return df

def save_data(df: pd.DataFrame, filepath: str):
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Successfully saved enriched data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")

if __name__ == "__main__":
    import glob
    import os
    
    list_of_files = glob.glob("hp_products_cleaned_*.csv")
    if not list_of_files:
        logger.error("No cleaned CSV file found. Please run data_cleaning.py first.")
    else:
        latest_cleaned_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Using latest cleaned file: {latest_cleaned_file}")
        
        df = load_data(latest_cleaned_file)
        if not df.empty:
            enriched_df = enrich_data(df)
            
            # Save the enriched data with a new name
            output_filename = latest_cleaned_file.replace("hp_products_cleaned_", "hp_products_enriched_")
            save_data(enriched_df, output_filename)
            
            print(f"\n✅ Data enrichment complete. Enriched data saved to: {output_filename}")
            print("Check data_enrichment.log for details.")


