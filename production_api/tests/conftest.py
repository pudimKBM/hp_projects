"""
Pytest configuration and fixtures for production API tests
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import app components
from production_api.app import create_app
from production_api.app.models import db, Product, Classification, ScrapingJob
from production_api.config.config import TestingConfig


@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    app = create_app('testing')
    
    # Create temporary database
    db_fd, app.config['DATABASE_PATH'] = tempfile.mkstemp()
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{app.config['DATABASE_PATH']}"
    app.config['TESTING'] = True
    
    with app.app_context():
        db.create_all()
        yield app
        
    # Cleanup
    os.close(db_fd)
    os.unlink(app.config['DATABASE_PATH'])


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def app_context(app):
    """Create application context"""
    with app.app_context():
        yield app


@pytest.fixture
def db_session(app_context):
    """Create database session for testing"""
    # Start transaction
    connection = db.engine.connect()
    transaction = connection.begin()
    
    # Configure session to use transaction
    session = sessionmaker(bind=connection)()
    db.session = session
    
    yield session
    
    # Rollback transaction
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_product_data():
    """Sample product data for testing"""
    return {
        'title': 'Cartucho HP 664 Original Preto',
        'description': 'Cartucho de tinta original HP 664 preto para impressoras HP DeskJet',
        'price': 89.90,
        'seller_name': 'HP Store Oficial',
        'rating': 4.8,
        'reviews_count': 150,
        'url': 'https://mercadolivre.com.br/cartucho-hp-664-original-preto',
        'platform': 'mercadolivre'
    }


@pytest.fixture
def sample_product(db_session, sample_product_data):
    """Create sample product in database"""
    product = Product(
        title=sample_product_data['title'],
        description=sample_product_data['description'],
        price_numeric=sample_product_data['price'],
        seller_name=sample_product_data['seller_name'],
        rating_numeric=sample_product_data['rating'],
        reviews_count=sample_product_data['reviews_count'],
        platform=sample_product_data['platform'],
        product_type='cartucho',
        url=sample_product_data['url'],
        raw_data=sample_product_data
    )
    
    db_session.add(product)
    db_session.commit()
    return product


@pytest.fixture
def sample_classification(db_session, sample_product):
    """Create sample classification in database"""
    classification = Classification(
        product_id=sample_product.id,
        prediction='original',
        confidence_score=0.92,
        feature_importance={'has_hp_keyword': 0.35, 'seller_reputation': 0.28},
        explanation={'reasoning': 'High confidence original product'},
        model_version='test_model_v1.0',
        processing_time_ms=145
    )
    
    db_session.add(classification)
    db_session.commit()
    return classification


@pytest.fixture
def sample_scraping_job(db_session):
    """Create sample scraping job in database"""
    job = ScrapingJob(
        job_type='manual',
        status='completed',
        search_terms=['cartucho hp', 'toner hp'],
        products_found=25,
        products_processed=23,
        errors=None
    )
    
    db_session.add(job)
    db_session.commit()
    return job


@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing"""
    model = Mock()
    model.predict.return_value = np.array(['original'])
    model.predict_proba.return_value = np.array([[0.08, 0.92]])
    model.feature_names_in_ = ['has_hp_keyword', 'price_numeric', 'seller_reputation']
    return model


@pytest.fixture
def mock_feature_pipeline():
    """Mock feature engineering pipeline"""
    pipeline = Mock()
    pipeline.transform.return_value = pd.DataFrame({
        'has_hp_keyword': [1],
        'price_numeric': [89.90],
        'seller_reputation': [0.8]
    })
    return pipeline


@pytest.fixture
def mock_model_loader():
    """Mock model loader"""
    loader = Mock()
    loader.load_model.return_value = {
        'load_successful': True,
        'model': Mock(),
        'metadata': Mock(),
        'validation_results': {'validation_successful': True}
    }
    loader.check_model_integrity.return_value = {
        'integrity_ok': True,
        'issues': []
    }
    loader.list_available_models.return_value = [
        {'name': 'test_model', 'version': 'v1.0', 'type': 'RandomForest'}
    ]
    return loader


@pytest.fixture
def mock_scraper():
    """Mock HP scraper for testing"""
    scraper = Mock()
    scraper.search_mercado_livre.return_value = [
        {
            'title': 'Cartucho HP 664 Original',
            'description': 'Cartucho original HP',
            'price': 'R$ 89,90',
            'seller_name': 'HP Store',
            'rating': '4.8',
            'reviews_count': '150 avaliações',
            'url': 'https://mercadolivre.com.br/product/123',
            'platform': 'mercadolivre'
        }
    ]
    scraper.close.return_value = None
    return scraper


@pytest.fixture
def mock_html_response():
    """Mock HTML response for scraper testing"""
    return """
    <html>
        <body>
            <div class="ui-search-result">
                <h2 class="ui-search-item__title">Cartucho HP 664 Original</h2>
                <span class="price-tag-fraction">89</span>
                <span class="price-tag-cents">90</span>
                <div class="ui-seller-info__title">HP Store Oficial</div>
                <span class="ui-review-rating">4.8</span>
                <span class="ui-review-count">150 opiniões</span>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def performance_metrics_data():
    """Sample performance metrics data"""
    return {
        'classification_metrics': {
            'total_classifications': 1000,
            'avg_processing_time_ms': 145,
            'accuracy_rate': 0.89,
            'confidence_distribution': {
                'high_confidence': 750,
                'medium_confidence': 200,
                'low_confidence': 50
            }
        },
        'scraper_metrics': {
            'total_jobs': 50,
            'successful_jobs': 48,
            'avg_products_per_job': 25,
            'avg_job_duration_minutes': 15
        },
        'system_metrics': {
            'uptime_hours': 168,
            'memory_usage_mb': 512,
            'cpu_usage_percent': 25
        }
    }


# Helper functions for tests
def create_test_products(db_session, count=5):
    """Create multiple test products"""
    products = []
    for i in range(count):
        product = Product(
            title=f'Test Product {i+1}',
            description=f'Test description {i+1}',
            price_numeric=50.0 + i * 10,
            seller_name=f'Test Seller {i+1}',
            rating_numeric=4.0 + (i * 0.1),
            reviews_count=100 + i * 10,
            platform='mercadolivre',
            product_type='cartucho',
            url=f'https://test.com/product/{i+1}',
            raw_data={'test': True}
        )
        products.append(product)
        db_session.add(product)
    
    db_session.commit()
    return products


def create_test_classifications(db_session, products):
    """Create classifications for test products"""
    classifications = []
    predictions = ['original', 'suspicious']
    
    for i, product in enumerate(products):
        classification = Classification(
            product_id=product.id,
            prediction=predictions[i % 2],
            confidence_score=0.7 + (i * 0.05),
            feature_importance={'test_feature': 0.5},
            explanation={'test': 'explanation'},
            model_version='test_v1.0',
            processing_time_ms=100 + i * 10
        )
        classifications.append(classification)
        db_session.add(classification)
    
    db_session.commit()
    return classifications