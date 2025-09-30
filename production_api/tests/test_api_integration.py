"""
API Integration Tests

This module contains integration tests for API endpoints with real database operations,
testing complete request-response cycles and data persistence.

Requirements covered: 2.1
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask

from production_api.app.models import db, Product, Classification, ScrapingJob, SystemHealth
from production_api.app.services.ml_service import MLService
from production_api.app.services.classification_service import ClassificationEngine


class TestAPIEndpointIntegration:
    """Test API endpoints with complete database integration"""
    
    def test_classify_endpoint_full_workflow(self, client, db_session, mock_ml_model, mock_feature_pipeline):
        """
        Test complete classification endpoint workflow with database persistence.
        
        Requirements: 2.1 - Complete API classification workflow
        """
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            mock_loader.return_value.load_model.return_value = {
                'load_successful': True,
                'model': mock_ml_model,
                'metadata': Mock(),
                'validation_results': {'validation_successful': True}
            }
            
            with patch('production_api.app.services.feature_service.FeatureEngineeringPipeline') as mock_fe:
                mock_fe.return_value = mock_feature_pipeline
                
                # Setup mock responses
                mock_ml_model.predict.return_value = ['original']
                mock_ml_model.predict_proba.return_value = [[0.15, 0.85]]
                mock_feature_pipeline.transform.return_value = {
                    'has_hp_keyword': [1],
                    'price_numeric': [89.90],
                    'seller_reputation': [0.8]
                }
                
                # Test data
                product_data = {
                    'title': 'Cartucho HP 664 Original Preto Integração',
                    'description': 'Cartucho de tinta original HP 664 preto para teste de integração',
                    'price': 89.90,
                    'seller_name': 'HP Store Oficial Integração',
                    'rating': 4.8,
                    'reviews_count': 150,
                    'url': 'https://mercadolivre.com.br/integration-test-product'
                }
                
                # Count initial records
                initial_product_count = db_session.query(Product).count()
                initial_classification_count = db_session.query(Classification).count()
                
                # Make API request
                response = client.post('/api/classify',
                                     json=product_data,
                                     content_type='application/json')
                
                # Verify API response
                assert response.status_code == 200
                data = json.loads(response.data)
                
                assert data['prediction'] == 'original'
                assert data['confidence_score'] == 0.85
                assert 'product_id' in data
                assert 'explanation' in data
                assert 'processing_time_ms' in data
                
                product_id = data['product_id']
                
                # Verify database changes
                final_product_count = db_session.query(Product).count()
                final_classification_count = db_session.query(Classification).count()
                
                assert final_product_count == initial_product_count + 1
                assert final_classification_count == initial_classification_count + 1
                
                # Verify product data in database
                product = db_session.query(Product).get(product_id)
                assert product is not None
                assert product.title == product_data['title']
                assert product.description == product_data['description']
                assert product.price_numeric == product_data['price']
                assert product.seller_name == product_data['seller_name']
                assert product.rating_numeric == product_data['rating']
                assert product.reviews_count == product_data['reviews_count']
                assert product.url == product_data['url']
                assert product.platform == 'api'  # Should be set for API requests
                
                # Verify classification data in database
                classification = db_session.query(Classification).filter_by(
                    product_id=product_id
                ).first()
                assert classification is not None
                assert classification.prediction == 'original'
                assert classification.confidence_score == 0.85
                assert classification.model_version is not None
                assert classification.processing_time_ms > 0
                assert classification.classified_at is not None
                assert classification.feature_importance is not None
                assert classification.explanation is not None
    
    def test_products_endpoint_with_complex_filtering(self, client, db_session):
        """
        Test products endpoint with complex filtering and database queries.
        
        Requirements: 2.1 - API data retrieval with complex filtering
        """
        # Create comprehensive test dataset
        products_data = []
        classifications_data = []
        
        # Create products with varied attributes
        base_date = datetime.utcnow() - timedelta(days=30)
        
        for i in range(30):
            product = Product(
                title=f'Produto HP Teste {i+1}',
                description=f'Descrição do produto de teste {i+1}',
                price_numeric=25.0 + i * 5,  # Prices from 25 to 170
                seller_name=f'Vendedor {(i % 5) + 1}',  # 5 different sellers
                rating_numeric=2.0 + (i % 5),  # Ratings from 2.0 to 6.0
                reviews_count=10 + i * 3,
                platform='mercadolivre',
                product_type='cartucho',
                url=f'https://test-integration.com/produto/{i+1}',
                raw_data={'integration_test': True, 'batch': 'filtering'},
                scraped_at=base_date + timedelta(days=i)
            )
            products_data.append(product)
            db_session.add(product)
        
        db_session.commit()
        
        # Create classifications with varied predictions and confidence
        for i, product in enumerate(products_data):
            prediction = 'original' if i % 3 == 0 else 'suspicious'
            confidence = 0.5 + (i % 5) * 0.1  # Confidence from 0.5 to 0.9
            
            classification = Classification(
                product_id=product.id,
                prediction=prediction,
                confidence_score=confidence,
                feature_importance={
                    'has_hp_keyword': 0.3 + (i % 3) * 0.1,
                    'price_range': 0.2 + (i % 2) * 0.1,
                    'seller_reputation': 0.25 + (i % 4) * 0.05
                },
                explanation={
                    'reasoning': f'Classificação de teste {i+1}',
                    'confidence_level': 'high' if confidence > 0.7 else 'medium'
                },
                model_version='integration_test_v1.0',
                processing_time_ms=100 + i * 5,
                classified_at=base_date + timedelta(days=i, hours=1)
            )
            classifications_data.append(classification)
            db_session.add(classification)
        
        db_session.commit()
        
        # Test 1: Basic pagination
        response = client.get('/api/products?page=1&limit=10')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert len(data['products']) == 10
        assert data['pagination']['total_items'] >= 30  # At least our test data
        assert data['pagination']['page'] == 1
        assert data['pagination']['limit'] == 10
        
        # Test 2: Filter by prediction
        response = client.get('/api/products?prediction=original')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        for product in data['products']:
            # Verify all returned products have 'original' prediction
            product_obj = db_session.query(Product).get(product['id'])
            if product_obj and product_obj.raw_data and 'integration_test' in str(product_obj.raw_data):
                classification = db_session.query(Classification).filter_by(
                    product_id=product['id']
                ).first()
                if classification:
                    assert classification.prediction == 'original'
        
        # Test 3: Filter by confidence score
        response = client.get('/api/products?min_confidence=0.8')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        for product in data['products']:
            product_obj = db_session.query(Product).get(product['id'])
            if product_obj and product_obj.raw_data and 'integration_test' in str(product_obj.raw_data):
                classification = db_session.query(Classification).filter_by(
                    product_id=product['id']
                ).first()
                if classification:
                    assert classification.confidence_score >= 0.8
        
        # Test 4: Date range filtering
        from_date = (base_date + timedelta(days=10)).isoformat()
        to_date = (base_date + timedelta(days=20)).isoformat()
        
        response = client.get(f'/api/products?date_from={from_date}&date_to={to_date}')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify date filtering
        for product in data['products']:
            product_obj = db_session.query(Product).get(product['id'])
            if product_obj and product_obj.raw_data and 'integration_test' in str(product_obj.raw_data):
                assert base_date + timedelta(days=10) <= product_obj.scraped_at <= base_date + timedelta(days=20)
        
        # Test 5: Combined filters
        response = client.get('/api/products?prediction=suspicious&min_confidence=0.6&limit=5')
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert len(data['products']) <= 5
        for product in data['products']:
            product_obj = db_session.query(Product).get(product['id'])
            if product_obj and product_obj.raw_data and 'integration_test' in str(product_obj.raw_data):
                classification = db_session.query(Classification).filter_by(
                    product_id=product['id']
                ).first()
                if classification:
                    assert classification.prediction == 'suspicious'
                    assert classification.confidence_score >= 0.6
    
    def test_product_detail_endpoint_with_complete_data(self, client, db_session):
        """
        Test product detail endpoint with complete relationship data.
        
        Requirements: 2.1 - Complete product detail retrieval
        """
        # Create detailed test product
        product = Product(
            title='Cartucho HP 664 Original Preto Detalhado',
            description='Cartucho de tinta original HP 664 preto com descrição completa para teste de detalhes',
            price_numeric=95.50,
            seller_name='HP Store Oficial Detalhes',
            rating_numeric=4.9,
            reviews_count=275,
            platform='mercadolivre',
            product_type='cartucho',
            url='https://mercadolivre.com.br/cartucho-hp-664-detalhado',
            raw_data={
                'detailed_test': True,
                'original_data': {
                    'price_text': 'R$ 95,50',
                    'rating_text': '4.9 estrelas',
                    'reviews_text': '275 avaliações',
                    'seller_info': {
                        'name': 'HP Store Oficial Detalhes',
                        'reputation': 'Platinum',
                        'sales_count': 10000
                    }
                }
            },
            scraped_at=datetime.utcnow() - timedelta(hours=2)
        )
        db_session.add(product)
        db_session.commit()
        
        # Create detailed classification
        classification = Classification(
            product_id=product.id,
            prediction='original',
            confidence_score=0.94,
            feature_importance={
                'has_hp_keyword': 0.35,
                'seller_reputation': 0.28,
                'price_range': 0.22,
                'rating_score': 0.15
            },
            explanation={
                'reasoning': 'Produto altamente confiável baseado em vendedor oficial e características consistentes',
                'confidence_level': 'very_high',
                'key_factors': [
                    'Vendedor oficial HP',
                    'Preço dentro da faixa esperada',
                    'Alta avaliação dos compradores',
                    'Descrição detalhada e profissional'
                ],
                'risk_factors': []
            },
            model_version='detailed_test_v2.1',
            processing_time_ms=185,
            classified_at=datetime.utcnow() - timedelta(hours=1, minutes=30)
        )
        db_session.add(classification)
        db_session.commit()
        
        # Test product detail retrieval
        response = client.get(f'/api/products/{product.id}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Verify complete product data
        product_data = data['product']
        assert product_data['id'] == product.id
        assert product_data['title'] == product.title
        assert product_data['description'] == product.description
        assert product_data['price_numeric'] == product.price_numeric
        assert product_data['seller_name'] == product.seller_name
        assert product_data['rating_numeric'] == product.rating_numeric
        assert product_data['reviews_count'] == product.reviews_count
        assert product_data['platform'] == product.platform
        assert product_data['product_type'] == product.product_type
        assert product_data['url'] == product.url
        assert 'scraped_at' in product_data
        
        # Verify complete classification data
        classification_data = data['classification']
        assert classification_data['prediction'] == classification.prediction
        assert classification_data['confidence_score'] == classification.confidence_score
        assert classification_data['model_version'] == classification.model_version
        assert classification_data['processing_time_ms'] == classification.processing_time_ms
        assert 'classified_at' in classification_data
        
        # Verify feature importance
        feature_importance = classification_data['feature_importance']
        assert feature_importance['has_hp_keyword'] == 0.35
        assert feature_importance['seller_reputation'] == 0.28
        assert feature_importance['price_range'] == 0.22
        assert feature_importance['rating_score'] == 0.15
        
        # Verify explanation
        explanation = classification_data['explanation']
        assert explanation['reasoning'] == classification.explanation['reasoning']
        assert explanation['confidence_level'] == 'very_high'
        assert len(explanation['key_factors']) == 4
        assert len(explanation['risk_factors']) == 0
    
    def test_health_endpoint_with_real_system_status(self, client, db_session):
        """
        Test health endpoint with real system component status.
        
        Requirements: 2.1 - System health monitoring via API
        """
        # Create some system health records
        health_records = [
            SystemHealth(
                component_name='scraper',
                status='healthy',
                metrics={
                    'last_run': datetime.utcnow().isoformat(),
                    'products_scraped': 45,
                    'success_rate': 0.98,
                    'avg_response_time_ms': 1200
                },
                recorded_at=datetime.utcnow() - timedelta(minutes=5)
            ),
            SystemHealth(
                component_name='ml_model',
                status='healthy',
                metrics={
                    'model_version': 'integration_test_v1.0',
                    'avg_prediction_time_ms': 120,
                    'predictions_per_hour': 2400,
                    'accuracy_last_24h': 0.89
                },
                recorded_at=datetime.utcnow() - timedelta(minutes=3)
            ),
            SystemHealth(
                component_name='database',
                status='healthy',
                metrics={
                    'connection_pool_active': 8,
                    'connection_pool_size': 10,
                    'query_avg_time_ms': 25,
                    'total_products': 1250,
                    'total_classifications': 1200
                },
                recorded_at=datetime.utcnow() - timedelta(minutes=1)
            )
        ]
        
        for record in health_records:
            db_session.add(record)
        db_session.commit()
        
        # Test health endpoint
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Verify overall status
        assert 'status' in data
        assert 'timestamp' in data
        assert 'components' in data
        
        # Verify component status
        components = data['components']
        
        # Check scraper component
        if 'scraper' in components:
            scraper = components['scraper']
            assert scraper['status'] == 'healthy'
            assert 'last_run' in scraper
            assert 'products_scraped' in scraper
            assert 'success_rate' in scraper
        
        # Check ML model component
        if 'ml_model' in components:
            ml_model = components['ml_model']
            assert ml_model['status'] == 'healthy'
            assert 'model_version' in ml_model
            assert 'avg_prediction_time_ms' in ml_model
        
        # Check database component
        if 'database' in components:
            database = components['database']
            assert database['status'] == 'healthy'
            assert 'connection_pool' in database or 'total_products' in database


class TestAPIConcurrencyIntegration:
    """Test API behavior under concurrent load"""
    
    def test_concurrent_classification_requests(self, client, db_session, mock_ml_model, mock_feature_pipeline):
        """
        Test concurrent classification requests maintain data integrity.
        
        Requirements: 2.1 - Concurrent API request handling
        """
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            mock_loader.return_value.load_model.return_value = {
                'load_successful': True,
                'model': mock_ml_model,
                'metadata': Mock(),
                'validation_results': {'validation_successful': True}
            }
            
            with patch('production_api.app.services.feature_service.FeatureEngineeringPipeline') as mock_fe:
                mock_fe.return_value = mock_feature_pipeline
                
                # Setup mock responses
                mock_ml_model.predict.return_value = ['original']
                mock_ml_model.predict_proba.return_value = [[0.1, 0.9]]
                
                # Prepare test data for concurrent requests
                test_products = []
                for i in range(10):
                    test_products.append({
                        'title': f'Produto Concorrente {i+1}',
                        'description': f'Descrição do produto concorrente {i+1}',
                        'price': 50.0 + i * 5,
                        'seller_name': f'Vendedor Concorrente {i+1}',
                        'rating': 4.0 + (i % 5) * 0.2,
                        'reviews_count': 100 + i * 10,
                        'url': f'https://concurrent-test.com/produto/{i+1}'
                    })
                
                # Count initial records
                initial_product_count = db_session.query(Product).count()
                initial_classification_count = db_session.query(Classification).count()
                
                # Execute concurrent requests
                def make_classification_request(product_data):
                    """Make a classification request"""
                    response = client.post('/api/classify',
                                         json=product_data,
                                         content_type='application/json')
                    return response.status_code, json.loads(response.data) if response.status_code == 200 else None
                
                # Use ThreadPoolExecutor for concurrent requests
                results = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_product = {
                        executor.submit(make_classification_request, product): product
                        for product in test_products
                    }
                    
                    for future in as_completed(future_to_product):
                        product = future_to_product[future]
                        try:
                            status_code, response_data = future.result()
                            results.append((status_code, response_data, product))
                        except Exception as exc:
                            results.append((500, None, product))
                
                # Verify all requests succeeded
                successful_requests = [r for r in results if r[0] == 200]
                assert len(successful_requests) == len(test_products)
                
                # Verify database integrity
                final_product_count = db_session.query(Product).count()
                final_classification_count = db_session.query(Classification).count()
                
                assert final_product_count == initial_product_count + len(test_products)
                assert final_classification_count == initial_classification_count + len(test_products)
                
                # Verify no duplicate products were created
                product_titles = [r[2]['title'] for r in successful_requests]
                created_products = db_session.query(Product).filter(
                    Product.title.in_(product_titles)
                ).all()
                
                assert len(created_products) == len(test_products)
                
                # Verify each product has exactly one classification
                for product in created_products:
                    classification_count = db_session.query(Classification).filter_by(
                        product_id=product.id
                    ).count()
                    assert classification_count == 1
    
    def test_concurrent_product_listing_requests(self, client, db_session):
        """
        Test concurrent product listing requests perform consistently.
        
        Requirements: 2.1 - Concurrent API data retrieval
        """
        # Create test data
        products = []
        for i in range(50):
            product = Product(
                title=f'Produto Lista Concorrente {i+1}',
                description=f'Descrição {i+1}',
                price_numeric=30.0 + i,
                seller_name=f'Vendedor {i+1}',
                rating_numeric=3.0 + (i % 5),
                reviews_count=50 + i,
                platform='mercadolivre',
                product_type='cartucho',
                url=f'https://concurrent-list-test.com/produto/{i+1}',
                raw_data={'concurrent_list_test': True}
            )
            products.append(product)
            db_session.add(product)
        
        db_session.commit()
        
        # Create classifications
        for i, product in enumerate(products):
            classification = Classification(
                product_id=product.id,
                prediction='original' if i % 2 == 0 else 'suspicious',
                confidence_score=0.6 + (i % 4) * 0.1,
                feature_importance={'test': 0.5},
                explanation={'test': 'concurrent'},
                model_version='concurrent_test_v1.0',
                processing_time_ms=100 + i
            )
            db_session.add(classification)
        
        db_session.commit()
        
        # Define concurrent request functions
        def get_products_page(page):
            """Get products page"""
            response = client.get(f'/api/products?page={page}&limit=10')
            return response.status_code, json.loads(response.data) if response.status_code == 200 else None
        
        def get_products_filtered(prediction):
            """Get filtered products"""
            response = client.get(f'/api/products?prediction={prediction}')
            return response.status_code, json.loads(response.data) if response.status_code == 200 else None
        
        # Execute concurrent requests
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit various types of requests
            futures = []
            
            # Page requests
            for page in range(1, 6):  # Pages 1-5
                futures.append(executor.submit(get_products_page, page))
            
            # Filter requests
            for prediction in ['original', 'suspicious']:
                futures.append(executor.submit(get_products_filtered, prediction))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    status_code, response_data = future.result()
                    results.append((status_code, response_data))
                except Exception as exc:
                    results.append((500, None))
        
        # Verify all requests succeeded
        successful_results = [r for r in results if r[0] == 200]
        assert len(successful_results) == len(futures)
        
        # Verify response consistency
        for status_code, data in successful_results:
            assert 'products' in data
            assert 'pagination' in data or len(data['products']) >= 0
            
            # Verify product data structure
            for product in data['products']:
                assert 'id' in product
                assert 'title' in product
                assert 'prediction' in product or 'price_numeric' in product


class TestAPIErrorHandlingIntegration:
    """Test API error handling with database operations"""
    
    def test_api_validation_errors_with_rollback(self, client, db_session):
        """
        Test API validation errors trigger proper database rollback.
        
        Requirements: 2.1 - API error handling with database integrity
        """
        # Count initial records
        initial_product_count = db_session.query(Product).count()
        initial_classification_count = db_session.query(Classification).count()
        
        # Test various invalid requests
        invalid_requests = [
            # Missing required fields
            {
                'description': 'Produto sem título',
                'price': 50.0
            },
            # Invalid data types
            {
                'title': 'Produto Válido',
                'description': 'Descrição válida',
                'price': 'preço_inválido',  # Should be numeric
                'seller_name': 'Vendedor'
            },
            # Negative values where inappropriate
            {
                'title': 'Produto Válido',
                'description': 'Descrição válida',
                'price': -50.0,  # Negative price
                'seller_name': 'Vendedor',
                'rating': 4.5,
                'reviews_count': -10  # Negative reviews
            }
        ]
        
        for i, invalid_data in enumerate(invalid_requests):
            response = client.post('/api/classify',
                                 json=invalid_data,
                                 content_type='application/json')
            
            # Should return error status
            assert response.status_code in [400, 422], f"Request {i+1} should have failed validation"
            
            # Verify no data was created in database
            current_product_count = db_session.query(Product).count()
            current_classification_count = db_session.query(Classification).count()
            
            assert current_product_count == initial_product_count, f"Products created after invalid request {i+1}"
            assert current_classification_count == initial_classification_count, f"Classifications created after invalid request {i+1}"
    
    def test_api_service_errors_with_proper_responses(self, client, db_session):
        """
        Test API service errors return proper HTTP responses.
        
        Requirements: 2.1 - API error response handling
        """
        # Test ML service unavailable
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            mock_loader.return_value.load_model.return_value = {
                'load_successful': False,
                'error': 'Model file not found'
            }
            
            valid_product_data = {
                'title': 'Produto Teste Erro',
                'description': 'Produto para teste de erro de serviço',
                'price': 75.0,
                'seller_name': 'Vendedor Teste',
                'rating': 4.2,
                'reviews_count': 120,
                'url': 'https://error-test.com/produto'
            }
            
            response = client.post('/api/classify',
                                 json=valid_product_data,
                                 content_type='application/json')
            
            # Should return service unavailable
            assert response.status_code == 503
            
            data = json.loads(response.data)
            assert 'error' in data
            assert 'service unavailable' in data['error'].lower() or 'model' in data['error'].lower()
        
        # Test database connection error simulation
        # Note: This is a conceptual test - actual implementation would depend on specific error handling
        
        # Test non-existent product detail
        response = client.get('/api/products/99999')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['error'].lower()


# Helper functions for API integration tests
def make_concurrent_requests(client, requests, max_workers=5):
    """Make multiple requests concurrently"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for method, url, data in requests:
            if method.upper() == 'GET':
                future = executor.submit(client.get, url)
            elif method.upper() == 'POST':
                future = executor.submit(client.post, url, json=data, content_type='application/json')
            else:
                continue
            
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                response = future.result()
                results.append((response.status_code, response.data))
            except Exception as exc:
                results.append((500, str(exc)))
    
    return results


def verify_api_response_structure(data, expected_fields):
    """Verify API response has expected structure"""
    for field in expected_fields:
        if '.' in field:
            # Nested field
            parts = field.split('.')
            current = data
            for part in parts:
                assert part in current, f"Missing nested field: {field}"
                current = current[part]
        else:
            assert field in data, f"Missing field: {field}"
    
    return True


def create_api_test_data(db_session, count=10):
    """Create test data for API integration tests"""
    products = []
    classifications = []
    
    for i in range(count):
        product = Product(
            title=f'API Test Product {i+1}',
            description=f'API integration test product {i+1}',
            price_numeric=40.0 + i * 3,
            seller_name=f'API Seller {i+1}',
            rating_numeric=3.5 + (i % 3) * 0.5,
            reviews_count=75 + i * 8,
            platform='api_test',
            product_type='cartucho',
            url=f'https://api-test.com/product/{i+1}',
            raw_data={'api_integration_test': True, 'index': i}
        )
        products.append(product)
        db_session.add(product)
    
    db_session.commit()
    
    for i, product in enumerate(products):
        classification = Classification(
            product_id=product.id,
            prediction='original' if i % 3 == 0 else 'suspicious',
            confidence_score=0.65 + (i % 4) * 0.08,
            feature_importance={
                'api_test_feature': 0.4 + (i % 3) * 0.1,
                'secondary_feature': 0.3 + (i % 2) * 0.1
            },
            explanation={
                'reasoning': f'API test classification {i+1}',
                'test_mode': True
            },
            model_version='api_test_v1.0',
            processing_time_ms=90 + i * 8
        )
        classifications.append(classification)
        db_session.add(classification)
    
    db_session.commit()
    return products, classifications