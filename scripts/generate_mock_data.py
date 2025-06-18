#!/usr/bin/env python3
"""
Script to generate mock data for EDA.
"""
import pandas as pd
import numpy as np
import datetime
import random

def generate_mock_data():
    num_records = 50
    data = {
        "title": [f"Cartucho HP {random.choice([664, 122, 901])} {random.choice(['Preto', 'Colorido', 'Kit'])} Original" if random.random() > 0.2 else f"Cartucho Compatível {random.choice([664, 122, 901])} {random.choice(['Preto', 'Colorido'])}" for _ in range(num_records)],
        "description": [f"Descrição detalhada do cartucho {i}. Alta qualidade e rendimento." if random.random() > 0.3 else f"Cartucho {i} compatível com sua impressora." for i in range(num_records)],
        "price_numeric": [round(random.uniform(20, 300), 2) for _ in range(num_records)],
        "seller_name": [random.choice(["HP Oficial", "Loja Americanas", "Eshop", "Tec Print", "Distribuidora ABC"]) for _ in range(num_records)],
        "rating_numeric": [round(random.uniform(2.5, 5.0), 1) for _ in range(num_records)],
        "reviews_count": [random.randint(0, 500) for _ in range(num_records)],
        "platform": [random.choice(["Mercado Livre", "Americanas"]) for _ in range(num_records)],
        "product_type": [random.choice(["Cartucho", "Toner", "Kit Tinta"]) for _ in range(num_records)],
        "label": [random.choice(["original", "suspeito", "sinais_mistos", "desconhecido"]) for _ in range(num_records)],
        "is_original": [random.choice([True, False]) for _ in range(num_records)],
        "is_xl": [random.choice([True, False]) for _ in range(num_records)],
        "has_sealed_info": [random.choice([True, False]) for _ in range(num_records)],
        "has_invoice_info": [random.choice([True, False]) for _ in range(num_records)],
        "has_warranty_info": [random.choice([True, False]) for _ in range(num_records)],
        "confidence": [round(random.uniform(0.1, 1.0), 2) for _ in range(num_records)],
        "target_is_original": [random.randint(0, 1) for _ in range(num_records)]
    }
    
    df_structured = pd.DataFrame(data)
    df_labeled = df_structured.copy() # For simplicity, use the same data for labeled
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    structured_filename = f"hp_products_structured_dataset_mock_{timestamp}.csv"
    labeled_filename = f"hp_products_labeled_mock_{timestamp}.csv"
    
    df_structured.to_csv(structured_filename, index=False)
    df_labeled.to_csv(labeled_filename, index=False)
    
    print(f"Generated mock structured dataset: {structured_filename}")
    print(f"Generated mock labeled dataset: {labeled_filename}")
    
    return structured_filename, labeled_filename

if __name__ == "__main__":
    generate_mock_data()

