#!/usr/bin/env python3
"""
Demonstrates how to generate a mock CSV file for Americanas products.
This script creates a CSV with sample data that mimics the structure
of scraped data from Americanas, including relevant fields for HP cartridges.
"""
import pandas as pd
import numpy as np
import datetime
import os

def generate_americanas_mock_data(num_products=5):
    """Generates a DataFrame with mock product data for Americanas."""
    data = []
    for i in range(num_products):
        product_id = 1000 + i
        title = "Cartucho HP 664 %s Original Americanas" % ("Preto" if i % 2 == 0 else "Colorido")
        price = round(np.random.uniform(50.0, 150.0), 2)
        seller_name = "Loja Americanas %d" % np.random.randint(1, 5)
        reviews_count = np.random.randint(0, 200)
        rating = round(np.random.uniform(3.0, 5.0), 1)
        description = "Cartucho original HP para impressoras Deskjet. Alta qualidade de impressão."
        specifications = "Tipo: %s | Marca: HP | Modelo: 664" % ("Tinta" if i % 2 == 0 else "Colorido")
        images = "https://example.com/americanas_image_%d.jpg" % product_id
        availability = "Em estoque" if np.random.rand() > 0.1 else "Esgotado"
        shipping_info = "Frete Grátis" if np.random.rand() > 0.5 else "Calcular Frete"
        url = "https://www.americanas.com.br/produto/%d" % product_id
        platform = "Americanas"

        data.append({
            "title": title,
            "price": price,
            "seller_name": seller_name,
            "reviews_count": reviews_count,
            "rating": rating,
            "description": description,
            "specifications": specifications,
            "images": images,
            "availability": availability,
            "shipping_info": shipping_info,
            "url": url,
            "platform": platform
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df_mock = generate_americanas_mock_data(num_products=5)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = "hp_products_americanas_mock_%s.csv" % timestamp
    
    df_mock.to_csv(output_filename, index=False, encoding="utf-8")
    print("Mock Americanas data generated and saved to: %s" % output_filename)


