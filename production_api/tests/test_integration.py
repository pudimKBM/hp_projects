"""
Integration Tests for Production Classification API

This module contains comprehensive integration tests that verify end-to-end workflows:
1. Scraping → Classification → Storage workflow
2. API integration tests with real database operations
3. Cron job execution and error handling scenarios

Requirements covered: 1.2, 1.3, 2.1
"""

import pytest
import time
import threading
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from flask import Flask
from sqlalchemy import text

from production_api.app.models import db, Product, Classification, ScrapingJob, SystemHealth
from production_api.app.services.scraper_service import ScraperService
from production_api.app.services.classification_service import ClassificationEngine
from production_api.app.services.batch_processor import BatchProcessor, BatchJobStatus
from production_api.app.services.cron_service import CronJobManager, JobStatus
from production_api.app.services.ml_service import MLService


class TestEndToEndWorkflow:
    """Test complete scraping → classification → storage workflow"""
    
    def test_complete_scraping_to_classification_workflow(self, app_context, db_session, mock_scraper, mock_ml_model, mock_feature_pipeline):
        """
        Test the complete end-to-end workflow from scraping to classification storage.
        
        Requirements: 1.2, 1.3 - Automated classification and storage of scraped products
        """
        # Setup mock data
        scraped_products = [
            {
                'title': 'Cartucho HP 664 Original Preto',
                'description': 'Cartucho de tinta original HP 664 preto',
                'price': 'R$ 89,90',
                'seller_name': 'HP Store Oficial',
                'rating': '4.8',
                'reviews_count': '150 avaliações',
                'url': 'https://mercadolivre.com.br/cartucho-hp-664-original',
                'platform': 'mercadolivre'
            },
            {
                'title': 'Cartucho HP 664 Compatível',
                'description': 'Cartucho compatível para HP',
                'price': 'R$ 25,90',
                'seller_name': 'Loja Genérica',
                'rating': '3.2',
                'reviews_count': '45 avaliações',
                'url': 'https://mercadolivre.com.br/cartucho-hp-664-compativel',
                'platform': 'mercadolivre'
            }
        ]
        
        mock_scraper.search_mercado_livre.return_value = scraped_products
        
        # Setup ML service with mocks
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            mock_loader.return_value.load_model.return_value = {
                'load_successful': True,
                'model': mock_ml_model,
                'metadata': Mock(),
                'validation_results': {'validation_successful': True}
            }
            
            with patch('production_api.app.services.feature_service.FeatureEngineeringPipeline') as mock_fe:
                mock_fe.return_value = mock_feature_pipeline
                
                # Initialize services
                scraper_service = ScraperService()
                ml_service = MLService()
                classification_engine = ClassificationEngine()
                batch_processor = BatchProcessor()
                
                # Step 1: Execute scraping job
                with patch.object(scraper_service, 'scraper', mock_scraper):
                    job = scraper_service.create_scraping_job(
                        job_type='integration_test',
                        search_terms=['cartucho hp 664']
                    )
                    
                    # Execute scraping
                    results = scraper_service.execute_scraping_job(job.id)
                    
                    # Verify scraping results
                    assert results['success'] is True
                    assert results['products_processed'] == 2
                    assert results['products_stored'] == 2
                    
                    # Verify products were stored in database
                    products = db_session.query(Product).filter_by(scraping_job_id=job.id).all()
                    assert len(products) == 2
                    
                    original_product = next(p for p in products if 'Original' in p.title)
                    compatible_product = next(p for p in products if 'Compatível' in p.title)
                    
                    assert original_product.price_numeric == 89.90
                    assert compatible_product.price_numeric == 25.90
                
                # Step 2: Execute batch classification
                mock_ml_model.predict.side_effect = [['original'], ['suspicious']]
                mock_ml_model.predict_proba.side_effect = [[[0.05, 0.95]], [[0.85, 0.15]]]
                
                # Process batch classification
                batch_job = batch_processor.create_batch_job(
                    job_type='post_scraping',
                    product_ids=[p.id for p in products]
                )
                
                batch_results = batch_processor.process_batch(batch_job.id)
                
                # Verify batch processing results
                assert batch_results['status'] == BatchJobStatus.COMPLETED.value
                assert batch_results['products_processed'] == 2
                assert batch_results['classifications_created'] == 2
                
                # Step 3: Verify classifications were stored correctly
                classifications = db_session.query(Classification).join(Product).filter(
                    Product.scraping_job_id == job.id
                ).all()
                
                assert len(classifications) == 2
                
                # Verify original product classification
                original_classification = next(c for c in classifications 
                                             if c.product.title.find('Original') != -1)
                assert original_classification.prediction == 'original'
                assert original_classification.confidence_score >= 0.9
                
                # Verify suspicious product classification
                suspicious_classification = next(c for c in classifications 
                                               if c.product.title.find('Compatível') != -1)
                assert suspicious_classification.prediction == 'suspicious'
                assert suspicious_classification.confidence_score >= 0.8
                
                # Step 4: Verify complete workflow integrity
                # Check that all data relationships are correct
                for classification in classifications:
                    assert classification.product_id is not None
                    assert classification.product.scraping_job_id == job.id
                    assert classification.model_version is not None
                    assert classification.processing_time_ms > 0
                    assert classification.classified_at is not None
    
    def test_workflow_with_scraping_errors(self, app_context, db_session, mock_scraper):
        """
        Test workflow behavior when scraping encounters errors.
        
        Requirements: 1.2 - Error handling during automated processing
        """
        # Setup scraper to fail on some products
        mock_scraper.search_mercado_livre.side_effect = Exception("Network timeout")
        
        scraper_service = ScraperService()
        
        with patch.object(scraper_service, 'scraper', mock_scraper):
            job = scraper_service.create_scraping_job(
                job_type='error_test',
                search_terms=['cartucho hp']
            )
            
            # Execute scraping job that will fail
            results = scraper_service.execute_scraping_job(job.id)
            
            # Verify error handling
            assert results['success'] is False
            assert 'error' in results
            assert 'Network timeout' in str(results['error'])
            
            # Verify job status was updated
            updated_job = db_session.query(ScrapingJob).get(job.id)
            assert updated_job.status == 'failed'
            assert updated_job.errors is not None
            assert updated_job.completed_at is not None
    
    def test_workflow_with_classification_errors(self, app_context, db_session, sample_product):
        """
        Test workflow behavior when classification encounters errors.
        
        Requirements: 1.3 - Error handling during classification processing
        """
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            # Setup ML service to fail
            mock_loader.return_value.load_model.return_value = {
                'load_successful': False,
                'error': 'Model file corrupted'
            }
            
            batch_processor = BatchProcessor()
            
            # Create batch job
            batch_job = batch_processor.create_batch_job(
                job_type='error_test',
                product_ids=[sample_product.id]
            )
            
            # Execute batch processing that will fail
            results = batch_processor.process_batch(batch_job.id)
            
            # Verify error handling
            assert results['status'] == BatchJobStatus.FAILED.value
            assert 'error' in results
            
            # Verify no classifications were created
            classifications = db_session.query(Classification).filter_by(
                product_id=sample_product.id
            ).all()
            assert len(classifications) == 0


class TestAPIIntegration:
    """Test API endpoints with real database operations"""
    
    def test_classify_endpoint_with_database_storage(self, client, db_session, mock_ml_model, mock_feature_pipeline):
        """
        Test classification endpoint stores results in database.
        
        Requirements: 2.1 - API classification with database integration
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
                mock_ml_model.predict_proba.return_value = [[0.08, 0.92]]
                
                # Make API request
                product_data = {
                    'title': 'Cartucho HP 664 Original Preto',
                    'description': 'Cartucho de tinta original HP 664',
                    'price': 89.90,
                    'seller_name': 'HP Store Oficial',
                    'rating': 4.8,
                    'reviews_count': 150,
                    'url': 'https://mercadolivre.com.br/test-product'
                }
                
                response = client.post('/api/classify', 
                                     json=product_data,
                                     content_type='application/json')
                
                # Verify API response
                assert response.status_code == 200
                data = json.loads(response.data)
                
                assert data['prediction'] == 'original'
                assert data['confidence_score'] >= 0.9
                assert 'product_id' in data
                
                # Verify data was stored in database
                product_id = data['product_id']
                
                # Check product was created
                product = db_session.query(Product).get(product_id)
                assert product is not None
                assert product.title == product_data['title']
                assert product.price_numeric == product_data['price']
                
                # Check classification was created
                classification = db_session.query(Classification).filter_by(
                    product_id=product_id
                ).first()
                assert classification is not None
                assert classification.prediction == 'original'
                assert classification.confidence_score >= 0.9
    
    def test_products_endpoint_with_pagination(self, client, db_session):
        """
        Test products listing endpoint with database pagination.
        
        Requirements: 2.1 - API data access with database operations
        """
        # Create test products with classifications
        products = []
        classifications = []
        
        for i in range(25):  # Create more than one page
            product = Product(
                title=f'Test Product {i+1}',
                description=f'Test description {i+1}',
                price_numeric=50.0 + i,
                seller_name=f'Seller {i+1}',
                rating_numeric=4.0 + (i % 5) * 0.2,
                reviews_count=100 + i * 5,
                platform='mercadolivre',
                product_type='cartucho',
                url=f'https://test.com/product/{i+1}',
                raw_data={'test': True}
            )
            products.append(product)
            db_session.add(product)
        
        db_session.commit()
        
        # Create classifications
        for i, product in enumerate(products):
            classification = Classification(
                product_id=product.id,
                prediction='original' if i % 2 == 0 else 'suspicious',
                confidence_score=0.7 + (i % 3) * 0.1,
                feature_importance={'test_feature': 0.5},
                explanation={'reasoning': f'Test explanation {i+1}'},
                model_version='test_v1.0',
                processing_time_ms=100 + i
            )
            classifications.append(classification)
            db_session.add(classification)
        
        db_session.commit()
        
        # Test first page
        response = client.get('/api/products?page=1&limit=10')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data['products']) == 10
        assert data['pagination']['page'] == 1
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
        
        # Test filtering by prediction
        response = client.get('/api/products?prediction=original')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        for product in data['products']:
            # Get classification for verification
            classification = db_session.query(Classification).filter_by(
                product_id=product['id']
            ).first()
            assert classification.prediction == 'original'
    
    def test_product_detail_endpoint_with_relationships(self, client, db_session, sample_product, sample_classification):
        """
        Test product detail endpoint with complete relationship data.
        
        Requirements: 2.1 - API detailed data access with relationships
        """
        response = client.get(f'/api/products/{sample_product.id}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Verify product data
        assert data['product']['id'] == sample_product.id
        assert data['product']['title'] == sample_product.title
        assert data['product']['price_numeric'] == sample_product.price_numeric
        
        # Verify classification data
        assert data['classification']['prediction'] == sample_classification.prediction
        assert data['classification']['confidence_score'] == sample_classification.confidence_score
        assert data['classification']['model_version'] == sample_classification.model_version
        
        # Verify relationships are properly loaded
        assert 'feature_importance' in data['classification']
        assert 'explanation' in data['classification']
    
    def test_api_error_handling_with_database_rollback(self, client, db_session):
        """
        Test API error handling ensures database consistency.
        
        Requirements: 2.1 - API error handling with database integrity
        """
        # Count initial products
        initial_count = db_session.query(Product).count()
        
        # Send invalid data that should cause database error
        invalid_data = {
            'title': None,  # Required field
            'description': 'Test',
            'price': 'invalid_price',  # Invalid type
            'seller_name': 'Test Seller'
        }
        
        response = client.post('/api/classify',
                             json=invalid_data,
                             content_type='application/json')
        
        # Verify error response
        assert response.status_code == 400
        
        # Verify no products were created (database rollback)
        final_count = db_session.query(Product).count()
        assert final_count == initial_count


class TestCronJobIntegration:
    """Test cron job execution and error handling scenarios"""
    
    def test_scheduled_scraping_job_execution(self, app_context, db_session, mock_scraper):
        """
        Test scheduled cron job execution for scraping.
        
        Requirements: 1.2 - Scheduled automated scraping
        """
        # Setup mock scraper
        mock_scraper.search_mercado_livre.return_value = [
            {
                'title': 'Scheduled Scrape Product',
                'description': 'Product found by scheduled job',
                'price': 'R$ 99,90',
                'seller_name': 'Test Seller',
                'rating': '4.5',
                'reviews_count': '200 avaliações',
                'url': 'https://test.com/scheduled-product',
                'platform': 'mercadolivre'
            }
        ]
        
        cron_manager = CronJobManager()
        scraper_service = ScraperService()
        
        # Create scheduled job function
        def scheduled_scraping_task():
            """Scheduled scraping task function"""
            with patch.object(scraper_service, 'scraper', mock_scraper):
                job = scraper_service.create_scraping_job(
                    job_type='scheduled',
                    search_terms=['cartucho hp']
                )
                return scraper_service.execute_scraping_job(job.id)
        
        # Register and execute scheduled job
        job_config = {
            'name': 'test_scheduled_scraping',
            'schedule_expression': 'every 1 minutes',  # For testing
            'function': scheduled_scraping_task,
            'enabled': True,
            'max_retries': 2
        }
        
        cron_manager.add_job(**job_config)
        
        # Execute the job manually (simulating cron execution)
        execution_result = cron_manager.execute_job('test_scheduled_scraping')
        
        # Verify job execution
        assert execution_result['status'] == JobStatus.COMPLETED.value
        assert execution_result['success'] is True
        
        # Verify products were created
        scheduled_jobs = db_session.query(ScrapingJob).filter_by(
            job_type='scheduled'
        ).all()
        assert len(scheduled_jobs) >= 1
        
        latest_job = max(scheduled_jobs, key=lambda j: j.created_at)
        assert latest_job.status == 'completed'
        assert latest_job.products_found >= 1
        
        # Verify products were stored
        products = db_session.query(Product).filter_by(
            scraping_job_id=latest_job.id
        ).all()
        assert len(products) >= 1
        assert products[0].title == 'Scheduled Scrape Product'
    
    def test_cron_job_failure_and_retry(self, app_context, db_session):
        """
        Test cron job failure handling and retry mechanism.
        
        Requirements: 1.2 - Error handling in scheduled jobs
        """
        cron_manager = CronJobManager()
        
        # Create job that will fail
        failure_count = 0
        
        def failing_task():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 attempts
                raise Exception(f"Simulated failure #{failure_count}")
            return {'success': True, 'message': 'Finally succeeded'}
        
        # Register failing job with retries
        job_config = {
            'name': 'test_failing_job',
            'schedule_expression': 'every 1 minutes',
            'function': failing_task,
            'enabled': True,
            'max_retries': 3
        }
        
        cron_manager.add_job(**job_config)
        
        # Execute job (will fail and retry)
        execution_result = cron_manager.execute_job('test_failing_job')
        
        # Verify retry mechanism worked
        assert execution_result['status'] == JobStatus.COMPLETED.value
        assert execution_result['success'] is True
        assert execution_result['retry_count'] == 2  # Failed twice, succeeded on third
        
        # Verify failure history was recorded
        job_history = cron_manager.get_job_history('test_failing_job')
        assert len(job_history) >= 1
        
        latest_execution = job_history[0]
        assert latest_execution['final_status'] == JobStatus.COMPLETED.value
        assert latest_execution['retry_count'] == 2
    
    def test_cron_job_timeout_handling(self, app_context, db_session):
        """
        Test cron job timeout handling.
        
        Requirements: 1.2 - Timeout handling in scheduled jobs
        """
        cron_manager = CronJobManager()
        
        # Create job that will timeout
        def long_running_task():
            time.sleep(5)  # Sleep longer than timeout
            return {'success': True}
        
        # Register job with short timeout
        job_config = {
            'name': 'test_timeout_job',
            'schedule_expression': 'every 1 minutes',
            'function': long_running_task,
            'enabled': True,
            'timeout_minutes': 0.05  # 3 seconds timeout
        }
        
        cron_manager.add_job(**job_config)
        
        # Execute job (will timeout)
        start_time = time.time()
        execution_result = cron_manager.execute_job('test_timeout_job')
        execution_time = time.time() - start_time
        
        # Verify timeout handling
        assert execution_result['status'] == JobStatus.FAILED.value
        assert 'timeout' in execution_result.get('error', '').lower()
        assert execution_time < 4  # Should timeout before 4 seconds
    
    def test_concurrent_cron_job_execution(self, app_context, db_session):
        """
        Test concurrent cron job execution handling.
        
        Requirements: 1.2 - Concurrent job execution management
        """
        cron_manager = CronJobManager()
        
        # Create job that simulates work
        execution_log = []
        
        def concurrent_task(task_id):
            execution_log.append(f'Task {task_id} started')
            time.sleep(1)  # Simulate work
            execution_log.append(f'Task {task_id} completed')
            return {'success': True, 'task_id': task_id}
        
        # Register multiple jobs
        for i in range(3):
            job_config = {
                'name': f'test_concurrent_job_{i}',
                'schedule_expression': 'every 1 minutes',
                'function': lambda task_id=i: concurrent_task(task_id),
                'enabled': True
            }
            cron_manager.add_job(**job_config)
        
        # Execute jobs concurrently
        threads = []
        results = {}
        
        def execute_job(job_name):
            results[job_name] = cron_manager.execute_job(job_name)
        
        # Start all jobs simultaneously
        for i in range(3):
            job_name = f'test_concurrent_job_{i}'
            thread = threading.Thread(target=execute_job, args=(job_name,))
            threads.append(thread)
            thread.start()
        
        # Wait for all jobs to complete
        for thread in threads:
            thread.join()
        
        # Verify all jobs completed successfully
        assert len(results) == 3
        for job_name, result in results.items():
            assert result['status'] == JobStatus.COMPLETED.value
            assert result['success'] is True
        
        # Verify execution log shows concurrent execution
        assert len(execution_log) == 6  # 3 starts + 3 completions
        start_count = len([log for log in execution_log if 'started' in log])
        complete_count = len([log for log in execution_log if 'completed' in log])
        assert start_count == 3
        assert complete_count == 3


class TestSystemHealthIntegration:
    """Test system health monitoring integration"""
    
    def test_health_monitoring_during_workflow(self, app_context, db_session, mock_scraper, mock_ml_model):
        """
        Test health monitoring during complete workflow execution.
        
        Requirements: 1.3 - System health monitoring during operations
        """
        # Setup services
        scraper_service = ScraperService()
        batch_processor = BatchProcessor()
        
        # Setup mocks
        mock_scraper.search_mercado_livre.return_value = [
            {
                'title': 'Health Test Product',
                'description': 'Product for health monitoring test',
                'price': 'R$ 75,50',
                'seller_name': 'Health Test Seller',
                'rating': '4.6',
                'reviews_count': '120 avaliações',
                'url': 'https://test.com/health-product',
                'platform': 'mercadolivre'
            }
        ]
        
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            mock_loader.return_value.load_model.return_value = {
                'load_successful': True,
                'model': mock_ml_model,
                'metadata': Mock(),
                'validation_results': {'validation_successful': True}
            }
            
            mock_ml_model.predict.return_value = ['original']
            mock_ml_model.predict_proba.return_value = [[0.1, 0.9]]
            
            # Record initial health metrics
            initial_health_count = db_session.query(SystemHealth).count()
            
            # Execute workflow with health monitoring
            with patch.object(scraper_service, 'scraper', mock_scraper):
                # Step 1: Scraping with health tracking
                job = scraper_service.create_scraping_job(
                    job_type='health_test',
                    search_terms=['cartucho hp']
                )
                
                scraping_results = scraper_service.execute_scraping_job(job.id)
                assert scraping_results['success'] is True
                
                # Step 2: Classification with health tracking
                products = db_session.query(Product).filter_by(scraping_job_id=job.id).all()
                
                batch_job = batch_processor.create_batch_job(
                    job_type='health_test',
                    product_ids=[p.id for p in products]
                )
                
                batch_results = batch_processor.process_batch(batch_job.id)
                assert batch_results['status'] == 'completed'
                
                # Verify health metrics were recorded
                final_health_count = db_session.query(SystemHealth).count()
                assert final_health_count > initial_health_count
                
                # Verify health metrics contain workflow data
                recent_health = db_session.query(SystemHealth).order_by(
                    SystemHealth.recorded_at.desc()
                ).first()
                
                assert recent_health is not None
                assert recent_health.component_name in ['scraper', 'classifier', 'batch_processor']
                assert recent_health.status == 'healthy'
                assert recent_health.metrics is not None
    
    def test_error_recovery_integration(self, app_context, db_session):
        """
        Test error recovery mechanisms across integrated components.
        
        Requirements: 1.3 - Error recovery in integrated system
        """
        # This test verifies that when one component fails, 
        # the system can recover and continue processing
        
        scraper_service = ScraperService()
        batch_processor = BatchProcessor()
        
        # Create a product manually (simulating successful scraping)
        product = Product(
            title='Error Recovery Test Product',
            description='Product for testing error recovery',
            price_numeric=65.00,
            seller_name='Recovery Test Seller',
            rating_numeric=4.3,
            reviews_count=80,
            platform='mercadolivre',
            product_type='cartucho',
            url='https://test.com/recovery-product',
            raw_data={'test': True}
        )
        db_session.add(product)
        db_session.commit()
        
        # Simulate ML service failure and recovery
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            # First attempt fails
            mock_loader.return_value.load_model.side_effect = [
                {'load_successful': False, 'error': 'Model corrupted'},
                {  # Second attempt succeeds (recovery)
                    'load_successful': True,
                    'model': Mock(),
                    'metadata': Mock(),
                    'validation_results': {'validation_successful': True}
                }
            ]
            
            # Setup successful prediction for recovery attempt
            mock_model = Mock()
            mock_model.predict.return_value = ['original']
            mock_model.predict_proba.return_value = [[0.2, 0.8]]
            mock_loader.return_value.load_model.return_value['model'] = mock_model
            
            # Create batch job
            batch_job = batch_processor.create_batch_job(
                job_type='recovery_test',
                product_ids=[product.id]
            )
            
            # First processing attempt (will fail)
            first_result = batch_processor.process_batch(batch_job.id)
            assert first_result['status'] == 'failed'
            
            # Verify no classification was created
            classifications = db_session.query(Classification).filter_by(
                product_id=product.id
            ).all()
            assert len(classifications) == 0
            
            # Second processing attempt (recovery - will succeed)
            # Reset the mock to succeed on next call
            mock_loader.return_value.load_model.side_effect = None
            mock_loader.return_value.load_model.return_value = {
                'load_successful': True,
                'model': mock_model,
                'metadata': Mock(),
                'validation_results': {'validation_successful': True}
            }
            
            # Create new batch job for recovery
            recovery_batch_job = batch_processor.create_batch_job(
                job_type='recovery_test_2',
                product_ids=[product.id]
            )
            
            recovery_result = batch_processor.process_batch(recovery_batch_job.id)
            assert recovery_result['status'] == 'completed'
            
            # Verify classification was created on recovery
            classifications = db_session.query(Classification).filter_by(
                product_id=product.id
            ).all()
            assert len(classifications) == 1
            assert classifications[0].prediction == 'original'


# Helper functions for integration tests
def wait_for_condition(condition_func, timeout_seconds=10, check_interval=0.1):
    """Wait for a condition to become true within timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if condition_func():
            return True
        time.sleep(check_interval)
    return False


def create_integration_test_data(db_session, count=5):
    """Create test data for integration tests"""
    products = []
    classifications = []
    
    for i in range(count):
        # Create product
        product = Product(
            title=f'Integration Test Product {i+1}',
            description=f'Description for integration test product {i+1}',
            price_numeric=50.0 + i * 10,
            seller_name=f'Integration Seller {i+1}',
            rating_numeric=4.0 + (i % 5) * 0.2,
            reviews_count=100 + i * 20,
            platform='mercadolivre',
            product_type='cartucho',
            url=f'https://integration-test.com/product/{i+1}',
            raw_data={'integration_test': True, 'index': i}
        )
        products.append(product)
        db_session.add(product)
    
    db_session.commit()
    
    # Create classifications
    for i, product in enumerate(products):
        classification = Classification(
            product_id=product.id,
            prediction='original' if i % 2 == 0 else 'suspicious',
            confidence_score=0.7 + (i % 4) * 0.05,
            feature_importance={
                'has_hp_keyword': 0.3 + (i % 3) * 0.1,
                'price_range': 0.2 + (i % 2) * 0.1,
                'seller_reputation': 0.25 + (i % 4) * 0.05
            },
            explanation={
                'reasoning': f'Integration test classification {i+1}',
                'confidence_level': 'high' if i % 2 == 0 else 'medium'
            },
            model_version='integration_test_v1.0',
            processing_time_ms=100 + i * 15
        )
        classifications.append(classification)
        db_session.add(classification)
    
    db_session.commit()
    return products, classifications


@contextmanager
def temporary_config_override(app, **config_overrides):
    """Temporarily override app configuration for testing"""
    original_config = {}
    
    # Store original values
    for key, value in config_overrides.items():
        original_config[key] = app.config.get(key)
        app.config[key] = value
    
    try:
        yield
    finally:
        # Restore original values
        for key, value in original_config.items():
            if value is None:
                app.config.pop(key, None)
            else:
                app.config[key] = value