#!/usr/bin/env python3
"""
Demo script to generate sample HP product data
This demonstrates the expected output format of the scraper
"""

import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample HP product data for demonstration"""
    
    sample_products = [
        {
            'title': 'Cartucho de tinta preta HP Advantage 664 de 2 ml',
            'price': 'R$ 66,90',
            'seller_name': 'OBERO INFORMATICA',
            'seller_reputation': 'MercadoLíder +500mil vendas',
            'reviews_count': '3317',
            'rating': '4.7',
            'url': 'https://www.mercadolivre.com.br/cartucho-de-tinta-preta-hp-advantage-664-de-2-ml/p/MLB36751629',
            'platform': 'Mercado Livre',
            'description': 'Imprima documentos do dia a dia com alta qualidade e a um preço acessível com cartuchos de tinta HP de baixo custo, projetados com proteção contra fraudes e alertas inteligentes de pouca tinta para oferecer desempenho sem preocupações e resultados consistentes.',
            'specifications': 'Impressão de cores consumíveis: Preto | Gota de tinta: 22 pl | Tipos de tinta: à base de pigmento | Rendimento de páginas: 120 páginas | Temperatura operacional: 15 a 32°C',
            'images': 'https://http2.mlstatic.com/D_NQ_NP_2X_664-MLA36751629_1234-F.webp',
            'availability': '+50 disponíveis',
            'shipping_info': 'Frete grátis',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': 'Kit 3 Cartuchos Hp 664 Preto + Colorido 2136 2676 3636 3776',
            'price': 'R$ 242,49',
            'seller_name': 'Loja Oficial HP',
            'seller_reputation': 'Loja oficial',
            'reviews_count': '29',
            'rating': '4.8',
            'url': 'https://www.mercadolivre.com.br/kit-3-cartuchos-hp-664-preto-colorido-2136-2676-3636-3776/p/MLB123456789',
            'platform': 'Mercado Livre',
            'description': 'Kit com 3 cartuchos HP 664 originais, sendo 2 pretos e 1 colorido. Compatível com impressoras HP DeskJet 2136, 2676, 3636, 3776 e outras da linha Ink Advantage.',
            'specifications': 'Marca: HP | Modelo: 664 | Tipo: Original | Cores: Preto e Tricolor | Rendimento: 120 páginas preto, 100 páginas colorido',
            'images': 'https://http2.mlstatic.com/D_NQ_NP_2X_123-MLA123456789_1234-F.webp',
            'availability': '15 disponíveis',
            'shipping_info': 'Frete grátis',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': 'Cartucho HP 664 Colorido(F6V28AB) Para Ink Advantage',
            'price': 'R$ 167,79',
            'seller_name': 'Tec Print',
            'seller_reputation': 'Vendedor confiável',
            'reviews_count': '4',
            'rating': '3.5',
            'url': 'https://www.mercadolivre.com.br/cartucho-hp-664-colorido-f6v28ab-para-ink-advantage/p/MLB987654321',
            'platform': 'Mercado Livre',
            'description': 'Cartucho HP 664 colorido original para impressoras HP Ink Advantage. Código F6V28AB. Cores ciano, magenta e amarelo em um único cartucho.',
            'specifications': 'Marca: HP | Modelo: F6V28AB | Tipo: Original | Cores: Tricolor (CMY) | Rendimento: 100 páginas coloridas',
            'images': 'https://http2.mlstatic.com/D_NQ_NP_2X_987-MLA987654321_1234-F.webp',
            'availability': '6 disponíveis',
            'shipping_info': 'Frete grátis',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': 'Cartucho De Tinta Hp 664 Negro y 664 Tricolor 2 Unidades',
            'price': 'R$ 153,10',
            'seller_name': 'Eshop',
            'seller_reputation': 'Vendedor',
            'reviews_count': '307',
            'rating': '4.6',
            'url': 'https://www.mercadolivre.com.br/cartucho-de-tinta-hp-664-negro-y-664-tricolor-2-unidades/p/MLB456789123',
            'platform': 'Mercado Livre',
            'description': 'Kit com 2 cartuchos HP 664, um preto e um tricolor. Compatível com impressoras HP DeskJet série 1000, 2000, 3000 e 4000.',
            'specifications': 'Marca: HP | Modelo: 664 | Tipo: Original | Kit: 1 Preto + 1 Tricolor | Rendimento: 120 páginas preto, 100 páginas colorido',
            'images': 'https://http2.mlstatic.com/D_NQ_NP_2X_456-MLA456789123_1234-F.webp',
            'availability': '25 disponíveis',
            'shipping_info': 'Frete grátis',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': 'Cartucho HP 122 Preto Original CH561HB Para 1000 2050 3050',
            'price': 'R$ 89,90',
            'seller_name': 'Cartuchos Online',
            'seller_reputation': 'MercadoLíder',
            'reviews_count': '156',
            'rating': '4.5',
            'url': 'https://www.mercadolivre.com.br/cartucho-hp-122-preto-original-ch561hb-para-1000-2050-3050/p/MLB789123456',
            'platform': 'Mercado Livre',
            'description': 'Cartucho HP 122 preto original CH561HB. Compatível com impressoras HP DeskJet 1000, 2050, 3050 e outras da série 122.',
            'specifications': 'Marca: HP | Modelo: CH561HB | Tipo: Original | Cor: Preto | Rendimento: 120 páginas | Compatibilidade: HP DeskJet 1000, 2050, 3050',
            'images': 'https://http2.mlstatic.com/D_NQ_NP_2X_789-MLA789123456_1234-F.webp',
            'availability': '8 disponíveis',
            'shipping_info': 'Frete R$ 15,90',
            'scraped_at': datetime.now().isoformat()
        }
    ]
    
    return sample_products

def save_sample_csv():
    """Save sample data to CSV file"""
    try:
        products = generate_sample_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(products)
        
        # Reorder columns for better readability
        column_order = [
            'title', 'price', 'seller_name', 'seller_reputation', 
            'reviews_count', 'rating', 'url', 'platform', 
            'description', 'specifications', 'images', 
            'availability', 'shipping_info', 'scraped_at'
        ]
        
        df = df[column_order]
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hp_products_demo_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        
        logger.info(f"Sample data saved to {filename}")
        logger.info(f"Generated {len(products)} sample products")
        
        # Print summary
        print(f"\n✅ Demo CSV generated successfully!")
        print(f"📁 File: {filename}")
        print(f"📊 Products: {len(products)}")
        print(f"🏪 Platforms: {df['platform'].unique()}")
        print(f"💰 Price range: {df['price'].min()} - {df['price'].max()}")
        
        return filename
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return None

if __name__ == "__main__":
    save_sample_csv()

