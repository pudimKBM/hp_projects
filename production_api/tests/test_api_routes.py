"""
Unit tests for API Routes

Tests API endpoints with various request scenarios.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from production_api.app.models import Product, Classification, ScrapingJob


class TestHealthEndpoint:
    """Test cases for /api/health endpoint"""
    
    def test_health_check_basic(self, client):
        """Test basic health check"""
        with patch('production_api.app.services.health_service.HealthService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            mock_health_data = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'components': {
                    'database': {'status': 'healthy'},
                    'ml_model': {'status': 'healthy'}
                }
            }
            mock_service.get_system_health.return_value = mock_health_data
            
            response = client.get('/api/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'healthy'
            assert 'api_info' in data
            assert 'timestamp' in data
    
    def test_health_check_with_force_refresh(self, client):
        """Test health check with force refresh parameter"""
        with patch('production_api.app.services.health_service.HealthService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_system_health.return_value = {'status': 'healthy'}
            
            response = client.get('/api/health?force_refresh=true')
            
            assert response.status_code == 200
            mock_service.get_system_health.assert_called_once_with(True)
    
    def test_health_check_specific_component(self, client):
        """Test health check for specific component"""
        with patch('production_api.app.services.health_service.check_component_health') as mock_check:
            mock_check.return_value = {
                'component': 'database',
                'status': 'healthy',
                'details': {}
            }
            
            response = client.get('/api/health?component=database')
            
            assert response.status_code == 200
            mock_check.assert_called_once_with('database')
    
    def test_health_check_invalid_component(self, client):
        """Test health check with invalid component"""
        response = client.get('/api/health?component=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Component must be one of' in data['message']
    
    def test_health_check_service_error(self, client):
        """Test health check when service is unavailable"""
        with patch('production_api.app.services.health_service.HealthService') as mock_service_class:
            mock_service_class.side_effect = Exception("Service unavailable")
            
            response = client.get('/api/health')
            
            assert response.status_code == 503
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'Service unavailable' in data['message']


class TestClassifyEndpoint:
    """Test cases for /api/classify endpoint"""
    
    def test_classify_product_success(self, client):
        """Test successful product classification"""
        with patch('production_api.app.services.classification_service_wrapper.ClassificationService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            mock_result = {
                'success': True,
                'prediction': 'original',
                'confidence_score': 0.85,
                'explanation': 'High confidence original product'
            }
            mock_service.classify_single_product.return_value = mock_result
            
            product_data = {
                'title': 'Cartucho HP 664 Original',
                'description': 'Cartucho original HP',
                'price': 89.90,
                'seller_name': 'HP Store',
                'rating': 4.8,
                'reviews_count': 150,
                'url': 'https://test.com/product'
            }
            
            response = client.post('/api/classify', 
                                 data=json.dumps(product_data),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] is True
            assert data['prediction'] == 'original'
            assert data['confidence_score'] == 0.85
            
            mock_service.classify_single_product.assert_called_once_with(product_data)
    
    def test_classify_product_missing_required_fields(self, client):
        """Test classification with missing required fields"""
        product_data = {
            'description': 'Product description'
            # Missing title and price
        }
        
        response = client.post('/api/classify',
                             data=json.dumps(product_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Missing required fields' in data['message']
        assert 'title' in data['message']
        assert 'price' in data['message']
    
    def test_classify_product_invalid_data_types(self, client):
        """Test classification with invalid data types"""
        product_data = {
            'title': 'Test Product',
            'price': 'invalid_price',  # Should be numeric
            'rating': 'invalid_rating'  # Should be numeric
        }
        
        response = client.post('/api/classify',
                             data=json.dumps(product_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Price must be a number' in data['message']
    
    def test_classify_product_non_json_request(self, client):
        """Test classification with non-JSON request"""
        response = client.post('/api/classify',
                             data='not json',
                             content_type='text/plain')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Request must be JSON' in data['message']
    
    def test_classify_product_service_error(self, client):
        """Test classification when service fails"""
        with patch('production_api.app.services.classification_service_wrapper.ClassificationService') as mock_service_class:
            mock_service_class.side_effect = Exception("Service unavailable")
            
            product_data = {
                'title': 'Test Product',
                'price': 89.90
            }
            
            response = client.post('/api/classify',
                                 data=json.dumps(product_data),
                                 content_type='application/json')
            
            assert response.status_code == 503
            data = json.loads(response.data)
            assert 'Classification service temporarily unavailable' in data['message']


class TestProductsEndpoint:
    """Test cases for /api/products endpoint"""
    
    def test_list_products_basic(self, client, db_session):
        """Test basic product listing"""
        # Create test products with classifications
        products = self._create_test_products_with_classifications(db_session, 5)
        
        response = client.get('/api/products')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'products' in data
        assert 'pagination' in data
        assert len(data['products']) == 5
        assert data['pagination']['total_items'] == 5
    
    def test_list_products_with_pagination(self, client, db_session):
        """Test product listing with pagination"""
        # Create test products
        products = self._create_test_products_with_classifications(db_session, 25)
        
        # Test first page
        response = client.get('/api/products?page=1&limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['products']) == 10
        assert data['pagination']['page'] == 1
        assert data['pagination']['limit'] == 10
        assert data['pagination']['total_items'] == 25
        assert data['pagination']['total_pages'] == 3
        assert data['pagination']['has_next'] is True
        assert data['pagination']['has_prev'] is False
        
        # Test second page
        response = client.get('/api/products?page=2&limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['products']) == 10
        assert data['pagination']['page'] == 2
        assert data['pagination']['has_next'] is True
        assert data['pagination']['has_prev'] is True
    
    def test_list_products_with_prediction_filter(self, client, db_session):
        """Test product listing with prediction filter"""
        products = self._create_test_products_with_classifications(db_session, 10)
        
        response = client.get('/api/products?prediction=original')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # All returned products should have 'original' prediction
        for product in data['products']:
            assert product['prediction'] == 'original'
    
    def test_list_products_with_confidence_filter(self, client, db_session):
        """Test product listing with confidence filter"""
        products = self._create_test_products_with_classifications(db_session, 10)
        
        response = client.get('/api/products?min_confidence=0.8')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # All returned products should have confidence >= 0.8
        for product in data['products']:
            assert product['confidence_score'] >= 0.8
    
    def test_list_products_with_date_filter(self, client, db_session):
        """Test product listing with date filters"""
        products = self._create_test_products_with_classifications(db_session, 5)
        
        # Set different scraped dates
        for i, product in enumerate(products):
            product.scraped_at = datetime.utcnow() - timedelta(days=i)
        db_session.commit()
        
        # Filter for products from last 2 days
        date_from = (datetime.utcnow() - timedelta(days=2)).isoformat() + 'Z'
        
        response = client.get(f'/api/products?date_from={date_from}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['products']) <= 3  # Should include products from days 0, 1, 2
    
    def test_list_products_invalid_parameters(self, client):
        """Test product listing with invalid parameters"""
        # Invalid page number
        response = client.get('/api/products?page=0')
        assert response.status_code == 400
        
        # Invalid limit
        response = client.get('/api/products?limit=1000')
        assert response.status_code == 400
        
        # Invalid prediction
        response = client.get('/api/products?prediction=invalid')
        assert response.status_code == 400
        
        # Invalid confidence
        response = client.get('/api/products?min_confidence=1.5')
        assert response.status_code == 400
        
        # Invalid date format
        response = client.get('/api/products?date_from=invalid-date')
        assert response.status_code == 400
    
    def _create_test_products_with_classifications(self, db_session, count):
        """Helper method to create test products with classifications"""
        products = []
        predictions = ['original', 'suspicious']
        
        for i in range(count):
            # Create product
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
                raw_data={'test': True},
                scraped_at=datetime.utcnow()
            )
            db_session.add(product)
            products.append(product)
        
        db_session.commit()
        
        # Create classifications
        for i, product in enumerate(products):
            classification = Classification(
                product_id=product.id,
                prediction=predictions[i % 2],
                confidence_score=0.7 + (i * 0.02),  # Varying confidence scores
                feature_importance={'test_feature': 0.5},
                explanation={'test': 'explanation'},
                model_version='test_v1.0',
                processing_time_ms=100 + i * 10,
                classified_at=datetime.utcnow()
            )
            db_session.add(classification)
        
        db_session.commit()
        return products


class TestProductDetailEndpoint:
    """Test cases for /api/products/{id} endpoint"""
    
    def test_get_product_details_success(self, client, db_session, sample_product, sample_classification):
        """Test successful product detail retrieval"""
        response = client.get(f'/api/products/{sample_product.id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'product' in data
        assert 'classification' in data
        assert data['product']['id'] == sample_product.id
        assert data['product']['title'] == sample_product.title
        assert data['classification']['prediction'] == sample_classification.prediction
        assert data['classification']['confidence_score'] == sample_classification.confidence_score
    
    def test_get_product_details_not_found(self, client):
        """Test product detail retrieval for non-existent product"""
        response = client.get('/api/products/99999')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'Product with id 99999 not found' in data['message']


class TestMetricsEndpoint:
    """Test cases for /api/metrics endpoint"""
    
    def test_get_metrics_json_format(self, client):
        """Test metrics retrieval in JSON format"""
        with patch('production_api.app.services.performance_service.get_performance_service') as mock_service_func:
            mock_service = Mock()
            mock_service_func.return_value = mock_service
            
            mock_summary = {
                'total_classifications': 1000,
                'avg_processing_time_ms': 145,
                'accuracy_rate': 0.89
            }
            mock_history = {
                'hourly_data': [{'hour': 1, 'classifications': 50}]
            }
            
            mock_service.get_performance_summary.return_value = mock_summary
            mock_service.get_performance_history.return_value = mock_history
            
            response = client.get('/api/metrics')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'current_metrics' in data
            assert 'historical_data' in data
            assert 'metadata' in data
            assert data['current_metrics'] == mock_summary
            assert data['historical_data'] == mock_history
    
    def test_get_metrics_prometheus_format(self, client):
        """Test metrics retrieval in Prometheus format"""
        with patch('production_api.app.services.performance_service.get_performance_service') as mock_service_func:
            mock_service = Mock()
            mock_service_func.return_value = mock_service
            
            mock_metrics = {
                'total_classifications': 1000,
                'avg_processing_time_ms': 145.5,
                'accuracy_rate': 0.89
            }
            mock_service.export_performance_metrics.return_value = mock_metrics
            
            response = client.get('/api/metrics?format=prometheus')
            
            assert response.status_code == 200
            assert response.content_type == 'text/plain; charset=utf-8'
            
            # Check that metrics are in Prometheus format
            text_data = response.data.decode('utf-8')
            assert 'total_classifications 1000' in text_data
            assert 'avg_processing_time_ms 145.5' in text_data
            assert 'accuracy_rate 0.89' in text_data
    
    def test_get_metrics_invalid_format(self, client):
        """Test metrics retrieval with invalid format"""
        response = client.get('/api/metrics?format=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Format must be "json" or "prometheus"' in data['message']
    
    def test_get_metrics_invalid_history_hours(self, client):
        """Test metrics retrieval with invalid history hours"""
        response = client.get('/api/metrics?history_hours=200')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'History hours must be between 1 and 168' in data['message']
    
    def test_get_metrics_service_error(self, client):
        """Test metrics retrieval when service fails"""
        with patch('production_api.app.services.performance_service.get_performance_service') as mock_service_func:
            mock_service_func.side_effect = Exception("Service unavailable")
            
            response = client.get('/api/metrics')
            
            assert response.status_code == 503
            data = json.loads(response.data)
            assert 'Performance metrics service temporarily unavailable' in data['message']


class TestAPIErrorHandlers:
    """Test cases for API error handlers"""
    
    def test_api_not_found(self, client):
        """Test 404 error handler for API routes"""
        response = client.get('/api/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error'] == 'Not Found'
        assert 'API endpoint not found' in data['message']
    
    def test_method_not_allowed(self, client):
        """Test 405 error handler for API routes"""
        # Try POST on GET-only endpoint
        response = client.post('/api/health')
        
        assert response.status_code == 405
        data = json.loads(response.data)
        assert data['error'] == 'Method Not Allowed'
        assert 'HTTP method not allowed' in data['message']


class TestAPIDecorators:
    """Test cases for API decorators"""
    
    def test_validate_json_request_decorator(self, client):
        """Test JSON validation decorator"""
        # Test with non-JSON content type
        response = client.post('/api/classify',
                             data='not json',
                             content_type='text/plain')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Request must be JSON' in data['message']
    
    def test_handle_api_errors_decorator(self, client):
        """Test error handling decorator"""
        # This is tested implicitly through other error scenarios
        # The decorator catches exceptions and returns proper JSON responses
        pass


class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_full_classification_workflow(self, client, db_session):
        """Test complete classification workflow through API"""
        # 1. Check health
        response = client.get('/api/health')
        assert response.status_code == 200
        
        # 2. Classify a product
        with patch('production_api.app.services.classification_service_wrapper.ClassificationService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            mock_result = {
                'success': True,
                'prediction': 'original',
                'confidence_score': 0.92,
                'model_version': 'v1.0'
            }
            mock_service.classify_single_product.return_value = mock_result
            
            product_data = {
                'title': 'Cartucho HP 664 Original',
                'price': 89.90,
                'seller_name': 'HP Store'
            }
            
            response = client.post('/api/classify',
                                 data=json.dumps(product_data),
                                 content_type='application/json')
            
            assert response.status_code == 200
            classification_result = json.loads(response.data)
            assert classification_result['success'] is True
        
        # 3. List products (would be empty in this test since we're mocking)
        response = client.get('/api/products')
        assert response.status_code == 200
        
        # 4. Get metrics
        with patch('production_api.app.services.performance_service.get_performance_service') as mock_service_func:
            mock_service = Mock()
            mock_service_func.return_value = mock_service
            mock_service.get_performance_summary.return_value = {'total': 1}
            mock_service.get_performance_history.return_value = {'data': []}
            
            response = client.get('/api/metrics')
            assert response.status_code == 200