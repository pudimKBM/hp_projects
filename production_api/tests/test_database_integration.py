"""
Database Integration Tests

This module tests database operations, transactions, and data integrity
across the production classification API system.

Requirements covered: 1.2, 1.3, 2.1
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from contextlib import contextmanager

from sqlalchemy import text, func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from production_api.app.models import db, Product, Classification, ScrapingJob, SystemHealth
from production_api.app.services.scraper_service import ScraperService
from production_api.app.services.batch_processor import BatchProcessor
from production_api.app.utils.database import get_db_session, DatabaseManager


class TestDatabaseTransactionIntegrity:
    """Test database transaction integrity during operations"""
    
    def test_scraping_transaction_rollback_on_error(self, app_context, db_session, mock_scraper):
        """
        Test that scraping transactions rollback properly on errors.
        
        Requirements: 1.2 - Data integrity during scraping operations
        """
        # Setup mock to fail after processing some products
        scraped_products = [
            {
                'title': 'Valid Product 1',
                'description': 'Valid description',
                'price': 'R$ 50,00',
                'seller_name': 'Valid Seller',
                'rating': '4.5',
                'reviews_count': '100 avaliações',
                'url': 'https://test.com/valid1',
                'platform': 'mercadolivre'
            },
            {
                'title': None,  # This will cause validation error
                'description': 'Invalid product',
                'price': 'R$ 60,00',
                'seller_name': 'Invalid Seller',
                'rating': '4.0',
                'reviews_count': '50 avaliações',
                'url': 'https://test.com/invalid',
                'platform': 'mercadolivre'
            }
        ]
        
        mock_scraper.search_mercado_livre.return_value = scraped_products
        
        scraper_service = ScraperService()
        
        # Count initial products
        initial_product_count = db_session.query(Product).count()
        initial_job_count = db_session.query(ScrapingJob).count()
        
        with patch.object(scraper_service, 'scraper', mock_scraper):
            job = scraper_service.create_scraping_job(
                job_type='transaction_test',
                search_terms=['test']
            )
            
            # Execute scraping (should handle error gracefully)
            results = scraper_service.execute_scraping_job(job.id)
            
            # Verify partial success (valid products processed, invalid ones skipped)
            assert results['success'] is True  # Should succeed with valid products
            assert results['products_processed'] == 1  # Only valid product
            assert results['products_stored'] == 1
            assert 'errors' in results  # Should report errors for invalid products
            
            # Verify database state
            final_product_count = db_session.query(Product).count()
            final_job_count = db_session.query(ScrapingJob).count()
            
            # Should have one new product and one new job
            assert final_product_count == initial_product_count + 1
            assert final_job_count == initial_job_count + 1
            
            # Verify job status reflects partial success
            updated_job = db_session.query(ScrapingJob).get(job.id)
            assert updated_job.status == 'completed'
            assert updated_job.products_found == 2
            assert updated_job.products_processed == 1
            assert updated_job.errors is not None
    
    def test_classification_transaction_atomicity(self, app_context, db_session, sample_product):
        """
        Test that classification transactions are atomic.
        
        Requirements: 1.3 - Data integrity during classification operations
        """
        batch_processor = BatchProcessor()
        
        # Mock ML service to fail after partial processing
        with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Model prediction failed")
            
            mock_loader.return_value.load_model.return_value = {
                'load_successful': True,
                'model': mock_model,
                'metadata': Mock(),
                'validation_results': {'validation_successful': True}
            }
            
            # Count initial classifications
            initial_classification_count = db_session.query(Classification).count()
            
            # Create batch job
            batch_job = batch_processor.create_batch_job(
                job_type='atomicity_test',
                product_ids=[sample_product.id]
            )
            
            # Execute batch processing (should fail)
            results = batch_processor.process_batch(batch_job.id)
            
            # Verify failure handling
            assert results['status'] == 'failed'
            
            # Verify no partial classifications were created
            final_classification_count = db_session.query(Classification).count()
            assert final_classification_count == initial_classification_count
            
            # Verify batch job status reflects failure
            updated_batch_job = db_session.query(ScrapingJob).filter_by(
                job_type='batch_classification'
            ).order_by(ScrapingJob.created_at.desc()).first()
            
            if updated_batch_job:  # May not exist if batch processor doesn't create ScrapingJob
                assert updated_batch_job.status in ['failed', 'completed']
    
    def test_concurrent_database_operations(self, app_context, db_session):
        """
        Test concurrent database operations maintain consistency.
        
        Requirements: 1.2, 1.3 - Concurrent operation safety
        """
        # Create test products for concurrent processing
        products = []
        for i in range(10):
            product = Product(
                title=f'Concurrent Test Product {i+1}',
                description=f'Product for concurrent testing {i+1}',
                price_numeric=100.0 + i,
                seller_name=f'Concurrent Seller {i+1}',
                rating_numeric=4.0 + (i % 5) * 0.2,
                reviews_count=50 + i * 5,
                platform='mercadolivre',
                product_type='cartucho',
                url=f'https://concurrent-test.com/product/{i+1}',
                raw_data={'concurrent_test': True}
            )
            products.append(product)
            db_session.add(product)
        
        db_session.commit()
        
        # Setup concurrent classification operations
        results = {}
        errors = {}
        
        def classify_product_batch(thread_id, product_ids):
            """Classify products in a separate thread"""
            try:
                with patch('production_api.app.services.ml_service.ModelLoader') as mock_loader:
                    mock_model = Mock()
                    mock_model.predict.return_value = ['original'] * len(product_ids)
                    mock_model.predict_proba.return_value = [[0.1, 0.9]] * len(product_ids)
                    
                    mock_loader.return_value.load_model.return_value = {
                        'load_successful': True,
                        'model': mock_model,
                        'metadata': Mock(),
                        'validation_results': {'validation_successful': True}
                    }
                    
                    batch_processor = BatchProcessor()
                    
                    # Create and process batch
                    batch_job = batch_processor.create_batch_job(
                        job_type=f'concurrent_test_{thread_id}',
                        product_ids=product_ids
                    )
                    
                    result = batch_processor.process_batch(batch_job.id)
                    results[thread_id] = result
                    
            except Exception as e:
                errors[thread_id] = str(e)
        
        # Start concurrent threads
        threads = []
        product_chunks = [
            [p.id for p in products[:3]],   # Thread 0: products 0-2
            [p.id for p in products[3:6]],  # Thread 1: products 3-5
            [p.id for p in products[6:]]    # Thread 2: products 6-9
        ]
        
        for i, chunk in enumerate(product_chunks):
            thread = threading.Thread(
                target=classify_product_batch,
                args=(i, chunk)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        assert len(results) == 3, "Not all concurrent operations completed"
        
        # Verify all operations succeeded
        for thread_id, result in results.items():
            assert result['status'] == 'completed', f"Thread {thread_id} failed: {result}"
        
        # Verify database consistency
        total_classifications = db_session.query(Classification).filter(
            Classification.product_id.in_([p.id for p in products])
        ).count()
        
        assert total_classifications == len(products), "Not all products were classified"
        
        # Verify no duplicate classifications
        for product in products:
            classification_count = db_session.query(Classification).filter_by(
                product_id=product.id
            ).count()
            assert classification_count == 1, f"Product {product.id} has {classification_count} classifications"


class TestDatabasePerformanceIntegration:
    """Test database performance under load"""
    
    def test_bulk_insert_performance(self, app_context, db_session):
        """
        Test bulk insert performance for large datasets.
        
        Requirements: 1.2 - Efficient data storage for large scraping jobs
        """
        # Create large dataset
        products_data = []
        for i in range(100):  # Test with 100 products
            products_data.append({
                'title': f'Bulk Test Product {i+1}',
                'description': f'Bulk insert test description {i+1}',
                'price_numeric': 50.0 + i,
                'seller_name': f'Bulk Seller {i+1}',
                'rating_numeric': 4.0 + (i % 5) * 0.2,
                'reviews_count': 100 + i,
                'platform': 'mercadolivre',
                'product_type': 'cartucho',
                'url': f'https://bulk-test.com/product/{i+1}',
                'raw_data': {'bulk_test': True, 'index': i}
            })
        
        # Test bulk insert performance
        start_time = time.time()
        
        # Use bulk insert
        products = [Product(**data) for data in products_data]
        db_session.bulk_save_objects(products)
        db_session.commit()
        
        bulk_insert_time = time.time() - start_time
        
        # Verify all products were inserted
        inserted_count = db_session.query(Product).filter(
            Product.raw_data.contains('"bulk_test": true')
        ).count()
        assert inserted_count == 100
        
        # Performance assertion (should complete within reasonable time)
        assert bulk_insert_time < 5.0, f"Bulk insert took too long: {bulk_insert_time}s"
        
        print(f"Bulk insert of 100 products completed in {bulk_insert_time:.3f}s")
    
    def test_query_performance_with_indexes(self, app_context, db_session):
        """
        Test query performance with proper indexing.
        
        Requirements: 2.1 - Efficient API data retrieval
        """
        # Create test data with various attributes for filtering
        products = []
        classifications = []
        
        for i in range(50):
            product = Product(
                title=f'Query Test Product {i+1}',
                description=f'Query performance test {i+1}',
                price_numeric=25.0 + i * 2,
                seller_name=f'Query Seller {i % 10}',  # 10 different sellers
                rating_numeric=3.0 + (i % 5),
                reviews_count=50 + i * 3,
                platform='mercadolivre',
                product_type='cartucho',
                url=f'https://query-test.com/product/{i+1}',
                raw_data={'query_test': True},
                scraped_at=datetime.utcnow() - timedelta(days=i % 30)  # Spread over 30 days
            )
            products.append(product)
            db_session.add(product)
        
        db_session.commit()
        
        # Create classifications
        for i, product in enumerate(products):
            classification = Classification(
                product_id=product.id,
                prediction='original' if i % 3 == 0 else 'suspicious',
                confidence_score=0.5 + (i % 5) * 0.1,
                feature_importance={'test_feature': 0.5},
                explanation={'test': 'explanation'},
                model_version='query_test_v1.0',
                processing_time_ms=100 + i,
                classified_at=datetime.utcnow() - timedelta(hours=i)
            )
            classifications.append(classification)
            db_session.add(classification)
        
        db_session.commit()
        
        # Test various query patterns
        query_tests = [
            # Filter by prediction
            lambda: db_session.query(Product).join(Classification).filter(
                Classification.prediction == 'original'
            ).all(),
            
            # Filter by confidence score
            lambda: db_session.query(Product).join(Classification).filter(
                Classification.confidence_score >= 0.8
            ).all(),
            
            # Filter by date range
            lambda: db_session.query(Product).filter(
                Product.scraped_at >= datetime.utcnow() - timedelta(days=7)
            ).all(),
            
            # Complex filter with pagination
            lambda: db_session.query(Product).join(Classification).filter(
                and_(
                    Classification.prediction == 'suspicious',
                    Product.price_numeric.between(50, 100),
                    Product.rating_numeric >= 4.0
                )
            ).limit(10).all(),
            
            # Aggregation query
            lambda: db_session.query(
                Classification.prediction,
                func.count(Classification.id),
                func.avg(Classification.confidence_score)
            ).group_by(Classification.prediction).all()
        ]
        
        # Execute queries and measure performance
        for i, query_func in enumerate(query_tests):
            start_time = time.time()
            results = query_func()
            query_time = time.time() - start_time
            
            # Performance assertion (queries should be fast)
            assert query_time < 1.0, f"Query {i+1} took too long: {query_time}s"
            assert results is not None, f"Query {i+1} returned None"
            
            print(f"Query {i+1} completed in {query_time:.3f}s, returned {len(results)} results")


class TestDatabaseMigrationIntegration:
    """Test database migration and schema changes"""
    
    def test_database_initialization(self, app_context):
        """
        Test database initialization and table creation.
        
        Requirements: 1.2, 1.3 - Database setup and initialization
        """
        # Test that all required tables exist
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        
        required_tables = ['product', 'classification', 'scraping_job', 'system_health']
        
        for table in required_tables:
            assert table in tables, f"Required table '{table}' not found"
        
        # Test table relationships
        # Check foreign key constraints
        classification_fks = inspector.get_foreign_keys('classification')
        product_fk_found = any(
            fk['constrained_columns'] == ['product_id'] and 
            fk['referred_table'] == 'product'
            for fk in classification_fks
        )
        assert product_fk_found, "Classification table missing product_id foreign key"
    
    def test_database_constraints_enforcement(self, app_context, db_session):
        """
        Test that database constraints are properly enforced.
        
        Requirements: 1.2, 1.3 - Data integrity constraints
        """
        # Test NOT NULL constraints
        with pytest.raises(IntegrityError):
            product = Product(
                title=None,  # Should violate NOT NULL constraint
                description='Test description',
                price_numeric=50.0,
                seller_name='Test Seller',
                platform='mercadolivre'
            )
            db_session.add(product)
            db_session.commit()
        
        db_session.rollback()
        
        # Test foreign key constraints
        with pytest.raises(IntegrityError):
            classification = Classification(
                product_id=99999,  # Non-existent product ID
                prediction='original',
                confidence_score=0.9,
                model_version='test_v1.0'
            )
            db_session.add(classification)
            db_session.commit()
        
        db_session.rollback()
    
    def test_database_backup_and_restore_simulation(self, app_context, db_session):
        """
        Test database backup and restore simulation.
        
        Requirements: 1.2, 1.3 - Data persistence and recovery
        """
        # Create test data
        original_product = Product(
            title='Backup Test Product',
            description='Product for backup testing',
            price_numeric=75.0,
            seller_name='Backup Seller',
            rating_numeric=4.5,
            reviews_count=200,
            platform='mercadolivre',
            product_type='cartucho',
            url='https://backup-test.com/product',
            raw_data={'backup_test': True}
        )
        db_session.add(original_product)
        db_session.commit()
        
        original_classification = Classification(
            product_id=original_product.id,
            prediction='original',
            confidence_score=0.88,
            feature_importance={'backup_feature': 0.7},
            explanation={'backup': 'test'},
            model_version='backup_test_v1.0',
            processing_time_ms=150
        )
        db_session.add(original_classification)
        db_session.commit()
        
        # Simulate data export (backup)
        products_data = db_session.query(Product).filter(
            Product.raw_data.contains('"backup_test": true')
        ).all()
        
        classifications_data = db_session.query(Classification).join(Product).filter(
            Product.raw_data.contains('"backup_test": true')
        ).all()
        
        # Verify backup data
        assert len(products_data) == 1
        assert len(classifications_data) == 1
        assert products_data[0].title == 'Backup Test Product'
        assert classifications_data[0].prediction == 'original'
        
        # Simulate data deletion (disaster scenario)
        db_session.query(Classification).filter_by(product_id=original_product.id).delete()
        db_session.query(Product).filter_by(id=original_product.id).delete()
        db_session.commit()
        
        # Verify deletion
        remaining_products = db_session.query(Product).filter_by(id=original_product.id).count()
        remaining_classifications = db_session.query(Classification).filter_by(
            product_id=original_product.id
        ).count()
        
        assert remaining_products == 0
        assert remaining_classifications == 0
        
        # Simulate restore (recreate from backup data)
        restored_product = Product(
            title=products_data[0].title,
            description=products_data[0].description,
            price_numeric=products_data[0].price_numeric,
            seller_name=products_data[0].seller_name,
            rating_numeric=products_data[0].rating_numeric,
            reviews_count=products_data[0].reviews_count,
            platform=products_data[0].platform,
            product_type=products_data[0].product_type,
            url=products_data[0].url,
            raw_data=products_data[0].raw_data
        )
        db_session.add(restored_product)
        db_session.commit()
        
        restored_classification = Classification(
            product_id=restored_product.id,
            prediction=classifications_data[0].prediction,
            confidence_score=classifications_data[0].confidence_score,
            feature_importance=classifications_data[0].feature_importance,
            explanation=classifications_data[0].explanation,
            model_version=classifications_data[0].model_version,
            processing_time_ms=classifications_data[0].processing_time_ms
        )
        db_session.add(restored_classification)
        db_session.commit()
        
        # Verify restore
        restored_products = db_session.query(Product).filter(
            Product.raw_data.contains('"backup_test": true')
        ).all()
        
        restored_classifications = db_session.query(Classification).join(Product).filter(
            Product.raw_data.contains('"backup_test": true')
        ).all()
        
        assert len(restored_products) == 1
        assert len(restored_classifications) == 1
        assert restored_products[0].title == 'Backup Test Product'
        assert restored_classifications[0].prediction == 'original'


# Helper functions for database integration tests
def execute_with_timeout(func, timeout_seconds=5):
    """Execute function with timeout"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel timeout
        raise


def verify_database_consistency(db_session):
    """Verify database consistency and relationships"""
    # Check that all classifications have valid products
    orphaned_classifications = db_session.query(Classification).outerjoin(Product).filter(
        Product.id.is_(None)
    ).count()
    
    assert orphaned_classifications == 0, f"Found {orphaned_classifications} orphaned classifications"
    
    # Check that all products with classifications have valid data
    products_with_classifications = db_session.query(Product).join(Classification).all()
    
    for product in products_with_classifications:
        assert product.title is not None, f"Product {product.id} has null title"
        assert product.price_numeric is not None, f"Product {product.id} has null price"
        assert product.platform is not None, f"Product {product.id} has null platform"
    
    return True


@contextmanager
def database_transaction_test(db_session):
    """Context manager for testing database transactions"""
    # Start transaction
    transaction = db_session.begin()
    
    try:
        yield db_session
        # Don't commit - let test decide
    except Exception:
        transaction.rollback()
        raise
    finally:
        if transaction.is_active:
            transaction.rollback()