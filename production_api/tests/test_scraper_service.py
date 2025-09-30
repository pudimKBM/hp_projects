"""
Unit tests for Scraper Service

Tests scraping functionality with mock HTML responses and database integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from flask import Flask

from production_api.app.services.scraper_service import ScraperService, run_manual_scraping, run_scheduled_scraping
from production_api.app.models import Product, ScrapingJob


class TestScraperService:
    """Test cases for ScraperService class"""
    
    def test_init(self):
        """Test ScraperService initialization"""
        service = ScraperService()
        
        assert service.scraper is None
        assert service.current_job is None
    
    def test_create_scraping_job(self, app_context, db_session):
        """Test creating a scraping job"""
        service = ScraperService()
        
        job = service.create_scraping_job(
            job_type="manual",
            search_terms=["cartucho hp", "toner hp"]
        )
        
        assert job.job_type == "manual"
        assert job.status == "running"
        assert job.search_terms == ["cartucho hp", "toner hp"]
        assert job.id is not None
    
    def test_create_scraping_job_with_defaults(self, app_context, db_session):
        """Test creating scraping job with default values"""
        with patch('flask.current_app') as mock_app:
            mock_app.config = {'DEFAULT_SEARCH_TERMS': ['default term']}
            
            service = ScraperService()
            job = service.create_scraping_job()
            
            assert job.job_type == "manual"
            assert job.search_terms == ['default term']
    
    @patch('production_api.app.services.scraper_service.HPProductScraper')
    @patch('production_api.app.services.scraper_service.time.sleep')
    def test_run_scraping_job_success(self, mock_sleep, mock_scraper_class, app_context, db_session):
        """Test successful scraping job execution"""
        # Setup mocks
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        
        mock_products = [
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
        
        mock_scraper.search_mercado_livre.return_value = mock_products
        
        with patch('flask.current_app') as mock_app:
            mock_app.config = {
                'DEFAULT_SEARCH_TERMS': ['cartucho hp'],
                'MAX_PAGES_PER_TERM': 2,
                'SCRAPER_DELAY': 1
            }
            
            service = ScraperService()
            
            with patch.object(service, '_store_products', return_value=1) as mock_store:
                products_found, products_stored, errors = service.run_scraping_job(
                    job_type="manual",
                    search_terms=['cartucho hp'],
                    max_pages=2
                )
                
                assert products_found == 1
                assert products_stored == 1
                assert len(errors) == 0
                
                # Verify scraper was called correctly
                mock_scraper.search_mercado_livre.assert_called_once()
                mock_store.assert_called_once_with(mock_products, [])
                mock_scraper.close.assert_called_once()
    
    @patch('production_api.app.services.scraper_service.HPProductScraper')
    def test_run_scraping_job_scraper_init_failure(self, mock_scraper_class, app_context, db_session):
        """Test scraping job with scraper initialization failure"""
        # Setup mock to fail initialization
        mock_scraper_class.side_effect = Exception("Browser initialization failed")
        
        with patch('flask.current_app') as mock_app:
            mock_app.config = {
                'DEFAULT_SEARCH_TERMS': ['cartucho hp'],
                'MAX_PAGES_PER_TERM': 2
            }
            
            service = ScraperService()
            products_found, products_stored, errors = service.run_scraping_job()
            
            assert products_found == 0
            assert products_stored == 0
            assert len(errors) > 0
            assert "Browser initialization failed" in str(errors)
    
    @patch('production_api.app.services.scraper_service.HPProductScraper')
    def test_scrape_search_term_with_retry(self, mock_scraper_class, app_context):
        """Test scraping search term with retry logic"""
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        
        # First call fails, second succeeds
        mock_scraper.search_mercado_livre.side_effect = [
            Exception("Network error"),
            [{'title': 'Test Product'}]
        ]
        
        service = ScraperService()
        service.scraper = mock_scraper
        
        with patch('time.sleep'):
            products = service._scrape_search_term("test term", 2, 3)
            
            assert len(products) == 1
            assert products[0]['title'] == 'Test Product'
            assert mock_scraper.search_mercado_livre.call_count == 2
    
    def test_clean_product_data(self, app_context):
        """Test product data cleaning"""
        service = ScraperService()
        
        raw_data = {
            'title': '  Cartucho   HP   664   Original  ',
            'description': 'Cartucho\n\noriginal\tHP',
            'price': 'R$ 1.234,56',
            'seller_name': '  HP Store Oficial  ',
            'rating': '4,8 estrelas',
            'reviews_count': '150 avaliações',
            'platform': 'mercadolivre',
            'url': 'https://test.com'
        }
        
        cleaned = service._clean_product_data(raw_data)
        
        assert cleaned['title'] == 'Cartucho HP 664 Original'
        assert cleaned['description'] == 'Cartucho original HP'
        assert cleaned['price_numeric'] == 1234.56
        assert cleaned['seller_name'] == 'HP Store Oficial'
        assert cleaned['rating_numeric'] == 4.8
        assert cleaned['reviews_count'] == 150
        assert cleaned['platform'] == 'mercadolivre'
        assert cleaned['url'] == 'https://test.com'
    
    def test_extract_numeric_price(self, app_context):
        """Test price extraction from various formats"""
        service = ScraperService()
        
        # Test various price formats
        assert service._extract_numeric_price('R$ 89,90') == 89.90
        assert service._extract_numeric_price('R$ 1.234,56') == 1234.56
        assert service._extract_numeric_price('1,234.56') == 1234.56
        assert service._extract_numeric_price('89.90') == 89.90
        assert service._extract_numeric_price('1234') == 1234.0
        assert service._extract_numeric_price('invalid') is None
        assert service._extract_numeric_price('') is None
    
    def test_extract_numeric_rating(self, app_context):
        """Test rating extraction from various formats"""
        service = ScraperService()
        
        assert service._extract_numeric_rating('4.8') == 4.8
        assert service._extract_numeric_rating('4,8 estrelas') == 4.8
        assert service._extract_numeric_rating('Rating: 3.5/5') == 3.5
        assert service._extract_numeric_rating('6.0') == 5.0  # Capped at 5
        assert service._extract_numeric_rating('invalid') is None
        assert service._extract_numeric_rating('') is None
    
    def test_extract_reviews_count(self, app_context):
        """Test reviews count extraction"""
        service = ScraperService()
        
        assert service._extract_reviews_count('150 avaliações') == 150
        assert service._extract_reviews_count('1.234 reviews') == 1234
        assert service._extract_reviews_count('No reviews') == 0
        assert service._extract_reviews_count('') == 0
    
    def test_extract_product_type(self, app_context):
        """Test product type extraction from title"""
        service = ScraperService()
        
        assert service._extract_product_type('Cartucho HP 664') == 'cartucho'
        assert service._extract_product_type('Toner HP LaserJet') == 'toner'
        assert service._extract_product_type('Tinta HP Original') == 'tinta'
        assert service._extract_product_type('Impressora HP') == 'outros'
    
    def test_validate_product_data(self, app_context):
        """Test product data validation"""
        service = ScraperService()
        
        # Valid product
        valid_data = {
            'title': 'Cartucho HP 664 Original',
            'url': 'https://test.com/product'
        }
        assert service._validate_product_data(valid_data) is True
        
        # Missing title
        invalid_data = {
            'url': 'https://test.com/product'
        }
        assert service._validate_product_data(invalid_data) is False
        
        # Missing URL
        invalid_data = {
            'title': 'Cartucho HP 664 Original'
        }
        assert service._validate_product_data(invalid_data) is False
        
        # No HP keywords
        invalid_data = {
            'title': 'Canon Cartridge',
            'url': 'https://test.com/product'
        }
        assert service._validate_product_data(invalid_data) is False
    
    def test_is_duplicate_product(self, app_context, db_session, sample_product):
        """Test duplicate product detection"""
        service = ScraperService()
        
        # Test duplicate by URL
        duplicate_data = {'url': sample_product.url}
        assert service._is_duplicate_product(duplicate_data) is True
        
        # Test duplicate by title and seller
        duplicate_data = {
            'title': sample_product.title,
            'seller_name': sample_product.seller_name
        }
        assert service._is_duplicate_product(duplicate_data) is True
        
        # Test non-duplicate
        unique_data = {
            'title': 'Different Product',
            'seller_name': 'Different Seller',
            'url': 'https://different.com'
        }
        assert service._is_duplicate_product(unique_data) is False
    
    def test_store_products(self, app_context, db_session):
        """Test storing products in database"""
        service = ScraperService()
        
        products_data = [
            {
                'title': 'Cartucho HP 664 Original',
                'description': 'Cartucho original HP',
                'price_numeric': 89.90,
                'seller_name': 'HP Store',
                'rating_numeric': 4.8,
                'reviews_count': 150,
                'platform': 'mercadolivre',
                'url': 'https://test.com/product1'
            },
            {
                'title': 'Toner HP LaserJet',
                'description': 'Toner original HP',
                'price_numeric': 199.90,
                'seller_name': 'HP Store',
                'rating_numeric': 4.5,
                'reviews_count': 75,
                'platform': 'mercadolivre',
                'url': 'https://test.com/product2'
            }
        ]
        
        errors = []
        stored_count = service._store_products(products_data, errors)
        
        assert stored_count == 2
        assert len(errors) == 0
        
        # Verify products were stored
        products = Product.query.all()
        assert len(products) == 2
        assert products[0].title == 'Cartucho HP 664 Original'
        assert products[0].product_type == 'cartucho'
        assert products[1].title == 'Toner HP LaserJet'
        assert products[1].product_type == 'toner'
    
    def test_store_products_with_invalid_data(self, app_context, db_session):
        """Test storing products with some invalid data"""
        service = ScraperService()
        
        products_data = [
            {
                'title': 'Valid HP Product',
                'price_numeric': 89.90,
                'url': 'https://test.com/valid'
            },
            {
                'title': '',  # Invalid - empty title
                'price_numeric': 99.90,
                'url': 'https://test.com/invalid'
            }
        ]
        
        with patch.object(service, '_validate_product_data') as mock_validate:
            mock_validate.side_effect = [True, False]
            
            errors = []
            stored_count = service._store_products(products_data, errors)
            
            assert stored_count == 1  # Only valid product stored
            assert len(Product.query.all()) == 1
    
    def test_get_job_status(self, app_context, db_session, sample_scraping_job):
        """Test getting job status"""
        service = ScraperService()
        
        # Test existing job
        status = service.get_job_status(sample_scraping_job.id)
        assert status is not None
        assert status['id'] == sample_scraping_job.id
        assert status['status'] == sample_scraping_job.status
        
        # Test non-existent job
        status = service.get_job_status(99999)
        assert status is None
    
    def test_get_recent_jobs(self, app_context, db_session):
        """Test getting recent jobs"""
        service = ScraperService()
        
        # Create test jobs
        jobs = []
        for i in range(5):
            job = ScrapingJob(
                job_type='test',
                status='completed',
                search_terms=[f'term{i}'],
                products_found=i * 10,
                products_processed=i * 8
            )
            db_session.add(job)
            jobs.append(job)
        
        db_session.commit()
        
        recent_jobs = service.get_recent_jobs(limit=3)
        
        assert len(recent_jobs) == 3
        # Should be ordered by most recent first
        assert recent_jobs[0]['id'] == jobs[-1].id
        assert recent_jobs[1]['id'] == jobs[-2].id
        assert recent_jobs[2]['id'] == jobs[-3].id
    
    def test_cleanup_old_jobs(self, app_context, db_session):
        """Test cleaning up old jobs"""
        service = ScraperService()
        
        # Create old and new jobs
        old_date = datetime.utcnow() - timedelta(days=35)
        new_date = datetime.utcnow() - timedelta(days=5)
        
        old_job = ScrapingJob(
            job_type='old',
            status='completed',
            search_terms=['old'],
            started_at=old_date
        )
        
        new_job = ScrapingJob(
            job_type='new',
            status='completed',
            search_terms=['new'],
            started_at=new_date
        )
        
        db_session.add(old_job)
        db_session.add(new_job)
        db_session.commit()
        
        # Cleanup jobs older than 30 days
        deleted_count = service.cleanup_old_jobs(days_to_keep=30)
        
        assert deleted_count == 1
        
        # Verify only new job remains
        remaining_jobs = ScrapingJob.query.all()
        assert len(remaining_jobs) == 1
        assert remaining_jobs[0].job_type == 'new'


class TestScraperServiceConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('production_api.app.services.scraper_service.ScraperService')
    def test_run_manual_scraping(self, mock_service_class):
        """Test run_manual_scraping function"""
        # Setup mock
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        expected_result = (10, 8, [])
        mock_service.run_scraping_job.return_value = expected_result
        
        # Test
        result = run_manual_scraping(['test term'], 5)
        
        # Assertions
        assert result == expected_result
        mock_service.run_scraping_job.assert_called_once_with("manual", ['test term'], 5)
    
    @patch('production_api.app.services.scraper_service.ScraperService')
    def test_run_scheduled_scraping(self, mock_service_class):
        """Test run_scheduled_scraping function"""
        # Setup mock
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        expected_result = (15, 12, [])
        mock_service.run_scraping_job.return_value = expected_result
        
        # Test
        result = run_scheduled_scraping()
        
        # Assertions
        assert result == expected_result
        mock_service.run_scraping_job.assert_called_once_with("scheduled")


class TestScraperServiceIntegration:
    """Integration tests for ScraperService"""
    
    @patch('production_api.app.services.scraper_service.HPProductScraper')
    @patch('production_api.app.services.scraper_service.time.sleep')
    def test_full_scraping_workflow(self, mock_sleep, mock_scraper_class, app_context, db_session):
        """Test complete scraping workflow"""
        # Setup mock scraper
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        
        mock_products = [
            {
                'title': 'Cartucho HP 664 Original Preto',
                'description': 'Cartucho de tinta original HP 664 preto',
                'price': 'R$ 89,90',
                'seller_name': 'HP Store Oficial',
                'rating': '4.8',
                'reviews_count': '150 avaliações',
                'url': 'https://mercadolivre.com.br/cartucho-hp-664-original-preto',
                'platform': 'mercadolivre'
            },
            {
                'title': 'Toner HP LaserJet Original',
                'description': 'Toner original HP LaserJet',
                'price': 'R$ 199,90',
                'seller_name': 'HP Store Oficial',
                'rating': '4.9',
                'reviews_count': '200 avaliações',
                'url': 'https://mercadolivre.com.br/toner-hp-laserjet-original',
                'platform': 'mercadolivre'
            }
        ]
        
        mock_scraper.search_mercado_livre.return_value = mock_products
        
        with patch('flask.current_app') as mock_app:
            mock_app.config = {
                'DEFAULT_SEARCH_TERMS': ['cartucho hp', 'toner hp'],
                'MAX_PAGES_PER_TERM': 2,
                'SCRAPER_DELAY': 0.1
            }
            
            # Mock performance service to avoid import issues
            with patch('production_api.app.services.scraper_service.get_performance_service'):
                with patch('production_api.app.services.scraper_service.log_scraper_event'):
                    service = ScraperService()
                    
                    # Run scraping job
                    products_found, products_stored, errors = service.run_scraping_job(
                        job_type="manual",
                        search_terms=['cartucho hp', 'toner hp'],
                        max_pages=2
                    )
                    
                    # Verify results
                    assert products_found == 4  # 2 products per search term
                    assert products_stored == 4
                    assert len(errors) == 0
                    
                    # Verify products were stored in database
                    stored_products = Product.query.all()
                    assert len(stored_products) == 4
                    
                    # Verify job was created and completed
                    job = ScrapingJob.query.first()
                    assert job is not None
                    assert job.status == 'completed'
                    assert job.products_found == 4
                    assert job.products_processed == 4
                    
                    # Verify scraper was called correctly
                    assert mock_scraper.search_mercado_livre.call_count == 2
                    mock_scraper.close.assert_called_once()
    
    @patch('production_api.app.services.scraper_service.HPProductScraper')
    def test_error_handling_workflow(self, mock_scraper_class, app_context, db_session):
        """Test error handling in scraping workflow"""
        # Setup mock scraper to fail
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper.search_mercado_livre.side_effect = Exception("Network timeout")
        
        with patch('flask.current_app') as mock_app:
            mock_app.config = {
                'DEFAULT_SEARCH_TERMS': ['cartucho hp'],
                'MAX_PAGES_PER_TERM': 2,
                'SCRAPER_DELAY': 0.1
            }
            
            service = ScraperService()
            
            products_found, products_stored, errors = service.run_scraping_job(
                job_type="manual",
                search_terms=['cartucho hp'],
                max_pages=2,
                max_retries=2
            )
            
            # Verify error handling
            assert products_found == 0
            assert products_stored == 0
            assert len(errors) > 0
            assert "Network timeout" in str(errors)
            
            # Verify job was marked as failed
            job = ScrapingJob.query.first()
            assert job is not None
            assert job.status == 'failed'
            assert job.errors is not None


class TestScraperServiceEdgeCases:
    """Test edge cases and error scenarios for ScraperService"""
    
    def test_store_products_database_error(self, app_context, db_session):
        """Test storing products when database error occurs"""
        service = ScraperService()
        
        products_data = [
            {
                'title': 'Valid HP Product',
                'price_numeric': 89.90,
                'url': 'https://test.com/valid'
            }
        ]
        
        # Mock database session to raise error on commit
        with patch.object(db_session, 'commit', side_effect=Exception("Database error")):
            errors = []
            stored_count = service._store_products(products_data, errors)
            
            assert stored_count == 0
            assert len(errors) > 0
            assert "Database error" in str(errors)
    
    def test_extract_numeric_price_edge_cases(self, app_context):
        """Test price extraction edge cases"""
        service = ScraperService()
        
        # Test edge cases
        assert service._extract_numeric_price('R$ 0,00') == 0.0
        assert service._extract_numeric_price('FREE') is None
        assert service._extract_numeric_price('R$ 1.000.000,99') == 1000000.99
        assert service._extract_numeric_price('$1,234.56') == 1234.56
        assert service._extract_numeric_price('1.234') == 1234.0  # No comma, treat as thousands
        assert service._extract_numeric_price('1,234') == 1234.0  # Ambiguous case
        assert service._extract_numeric_price('') is None
        assert service._extract_numeric_price(None) is None
    
    def test_extract_numeric_rating_edge_cases(self, app_context):
        """Test rating extraction edge cases"""
        service = ScraperService()
        
        # Test edge cases
        assert service._extract_numeric_rating('0') == 0.0
        assert service._extract_numeric_rating('10.0') == 5.0  # Capped at 5
        assert service._extract_numeric_rating('-1.0') == 0.0  # Minimum 0
        assert service._extract_numeric_rating('4.5 de 5 estrelas') == 4.5
        assert service._extract_numeric_rating('No rating') is None
        assert service._extract_numeric_rating('') is None
        assert service._extract_numeric_rating(None) is None
    
    def test_extract_reviews_count_edge_cases(self, app_context):
        """Test reviews count extraction edge cases"""
        service = ScraperService()
        
        # Test edge cases
        assert service._extract_reviews_count('0 reviews') == 0
        assert service._extract_reviews_count('1 review') == 1
        assert service._extract_reviews_count('1,234 avaliações') == 1234
        assert service._extract_reviews_count('No reviews yet') == 0
        assert service._extract_reviews_count('') == 0
        assert service._extract_reviews_count(None) == 0
    
    def test_clean_product_data_edge_cases(self, app_context):
        """Test product data cleaning edge cases"""
        service = ScraperService()
        
        # Test with minimal data
        raw_data = {
            'title': '',
            'price': '',
            'rating': '',
            'reviews_count': ''
        }
        
        cleaned = service._clean_product_data(raw_data)
        
        assert cleaned['title'] == ''
        assert cleaned['price_numeric'] is None
        assert cleaned['rating_numeric'] is None
        assert cleaned['reviews_count'] == 0
        
        # Test with very long strings
        long_title = 'A' * 1000
        raw_data = {
            'title': long_title,
            'description': 'B' * 3000,
            'seller_name': 'C' * 500
        }
        
        cleaned = service._clean_product_data(raw_data)
        
        assert len(cleaned['title']) <= 500
        assert len(cleaned['description']) <= 2000
        assert len(cleaned['seller_name']) <= 200
    
    def test_validate_product_data_edge_cases(self, app_context):
        """Test product data validation edge cases"""
        service = ScraperService()
        
        # Test with HP variations
        valid_variations = [
            {'title': 'hp cartucho', 'url': 'https://test.com'},
            {'title': 'HEWLETT PACKARD toner', 'url': 'https://test.com'},
            {'title': 'Packard Bell printer', 'url': 'https://test.com'}
        ]
        
        for data in valid_variations:
            assert service._validate_product_data(data) is True
        
        # Test invalid cases
        invalid_cases = [
            {'title': '', 'url': 'https://test.com'},  # Empty title
            {'title': 'Canon printer', 'url': 'https://test.com'},  # No HP keywords
            {'title': 'HP printer', 'url': ''},  # Empty URL
            {'url': 'https://test.com'},  # Missing title
            {'title': 'HP printer'}  # Missing URL
        ]
        
        for data in invalid_cases:
            assert service._validate_product_data(data) is False
    
    def test_scraper_initialization_retry_logic(self, app_context, db_session):
        """Test scraper initialization with retry logic"""
        service = ScraperService()
        
        with patch('production_api.app.services.scraper_service.HPProductScraper') as mock_scraper_class:
            # First two attempts fail, third succeeds
            mock_scraper_class.side_effect = [
                Exception("Browser not found"),
                Exception("Port already in use"),
                Mock()  # Success on third attempt
            ]
            
            with patch('flask.current_app') as mock_app:
                mock_app.config = {
                    'DEFAULT_SEARCH_TERMS': ['test'],
                    'MAX_PAGES_PER_TERM': 1,
                    'SCRAPER_DELAY': 0.1
                }
                
                with patch('time.sleep'):  # Speed up test
                    with patch.object(service, '_store_products', return_value=0):
                        products_found, products_stored, errors = service.run_scraping_job(
                            max_retries=3
                        )
                        
                        # Should succeed after retries
                        assert len(errors) == 0  # No final errors
                        assert mock_scraper_class.call_count == 3
    
    def test_scraper_search_term_failure_continues(self, app_context, db_session):
        """Test that failure on one search term doesn't stop others"""
        service = ScraperService()
        
        with patch('production_api.app.services.scraper_service.HPProductScraper') as mock_scraper_class:
            mock_scraper = Mock()
            mock_scraper_class.return_value = mock_scraper
            
            # First search term fails, second succeeds
            mock_scraper.search_mercado_livre.side_effect = [
                Exception("Network timeout"),
                [{'title': 'Success Product'}]
            ]
            
            with patch('flask.current_app') as mock_app:
                mock_app.config = {
                    'DEFAULT_SEARCH_TERMS': ['term1', 'term2'],
                    'MAX_PAGES_PER_TERM': 1,
                    'SCRAPER_DELAY': 0.1
                }
                
                with patch('time.sleep'):
                    with patch.object(service, '_store_products', return_value=1) as mock_store:
                        products_found, products_stored, errors = service.run_scraping_job(
                            search_terms=['term1', 'term2']
                        )
                        
                        # Should have one success and one error
                        assert products_found == 1
                        assert products_stored == 1
                        assert len(errors) == 1
                        assert "Network timeout" in str(errors)
                        
                        # Store should be called with successful products
                        mock_store.assert_called_once_with([{'title': 'Success Product'}], errors)
    
    def test_duplicate_detection_by_url_and_title(self, app_context, db_session):
        """Test comprehensive duplicate detection"""
        service = ScraperService()
        
        # Create existing product
        existing_product = Product(
            title='Existing HP Product',
            seller_name='Test Seller',
            url='https://existing.com/product',
            price_numeric=99.90,
            platform='mercadolivre',
            product_type='cartucho'
        )
        db_session.add(existing_product)
        db_session.commit()
        
        # Test duplicate by URL
        duplicate_by_url = {'url': 'https://existing.com/product'}
        assert service._is_duplicate_product(duplicate_by_url) is True
        
        # Test duplicate by title and seller
        duplicate_by_title_seller = {
            'title': 'Existing HP Product',
            'seller_name': 'Test Seller'
        }
        assert service._is_duplicate_product(duplicate_by_title_seller) is True
        
        # Test non-duplicate
        unique_product = {
            'title': 'New HP Product',
            'seller_name': 'Different Seller',
            'url': 'https://new.com/product'
        }
        assert service._is_duplicate_product(unique_product) is False
        
        # Test with database error (should return False to be safe)
        with patch('production_api.app.models.Product.query') as mock_query:
            mock_query.filter_by.side_effect = Exception("Database error")
            
            test_data = {'url': 'https://test.com'}
            assert service._is_duplicate_product(test_data) is False
    
    def test_batch_commit_logic(self, app_context, db_session):
        """Test batch commit logic in store_products"""
        service = ScraperService()
        
        # Create 25 products to test batch commits (every 10)
        products_data = []
        for i in range(25):
            products_data.append({
                'title': f'HP Product {i}',
                'price_numeric': 50.0 + i,
                'seller_name': f'Seller {i}',
                'url': f'https://test.com/product{i}',
                'platform': 'mercadolivre'
            })
        
        # Mock commit to track calls
        original_commit = db_session.commit
        commit_calls = []
        
        def mock_commit():
            commit_calls.append(len(Product.query.all()))
            return original_commit()
        
        with patch.object(db_session, 'commit', side_effect=mock_commit):
            errors = []
            stored_count = service._store_products(products_data, errors)
            
            assert stored_count == 25
            assert len(errors) == 0
            
            # Should have commits at 10, 20, and final
            assert len(commit_calls) >= 3
    
    def test_performance_tracking_integration(self, app_context, db_session):
        """Test integration with performance tracking services"""
        service = ScraperService()
        
        with patch('production_api.app.services.scraper_service.HPProductScraper') as mock_scraper_class:
            mock_scraper = Mock()
            mock_scraper_class.return_value = mock_scraper
            mock_scraper.search_mercado_livre.return_value = [{'title': 'Test Product'}]
            
            # Mock performance service
            mock_perf_service = Mock()
            
            with patch('production_api.app.services.scraper_service.get_performance_service') as mock_get_perf:
                mock_get_perf.return_value = mock_perf_service
                
                with patch('production_api.app.services.scraper_service.log_scraper_event') as mock_log:
                    with patch('flask.current_app') as mock_app:
                        mock_app.config = {
                            'DEFAULT_SEARCH_TERMS': ['test'],
                            'MAX_PAGES_PER_TERM': 1,
                            'SCRAPER_DELAY': 0.1
                        }
                        
                        with patch.object(service, '_store_products', return_value=1):
                            products_found, products_stored, errors = service.run_scraping_job()
                            
                            # Verify performance tracking was called
                            mock_perf_service.record_scraper_performance.assert_called_once()
                            mock_log.assert_called_once()
                            
                            # Check the parameters passed to performance tracking
                            perf_call_args = mock_perf_service.record_scraper_performance.call_args[1]
                            assert 'job_duration_seconds' in perf_call_args
                            assert perf_call_args['products_found'] == 1
                            assert perf_call_args['products_processed'] == 1
    
    def test_performance_tracking_failure_handling(self, app_context, db_session):
        """Test that performance tracking failures don't break scraping"""
        service = ScraperService()
        
        with patch('production_api.app.services.scraper_service.HPProductScraper') as mock_scraper_class:
            mock_scraper = Mock()
            mock_scraper_class.return_value = mock_scraper
            mock_scraper.search_mercado_livre.return_value = [{'title': 'Test Product'}]
            
            # Mock performance service to fail
            with patch('production_api.app.services.scraper_service.get_performance_service') as mock_get_perf:
                mock_get_perf.side_effect = Exception("Performance service unavailable")
                
                with patch('flask.current_app') as mock_app:
                    mock_app.config = {
                        'DEFAULT_SEARCH_TERMS': ['test'],
                        'MAX_PAGES_PER_TERM': 1,
                        'SCRAPER_DELAY': 0.1
                    }
                    
                    with patch.object(service, '_store_products', return_value=1):
                        # Should complete successfully despite performance tracking failure
                        products_found, products_stored, errors = service.run_scraping_job()
                        
                        assert products_found == 1
                        assert products_stored == 1
                        # Main scraping should succeed even if performance tracking fails
    
    def test_scraper_cleanup_on_exception(self, app_context, db_session):
        """Test that scraper is properly cleaned up even when exceptions occur"""
        service = ScraperService()
        
        with patch('production_api.app.services.scraper_service.HPProductScraper') as mock_scraper_class:
            mock_scraper = Mock()
            mock_scraper_class.return_value = mock_scraper
            
            # Make scraping fail after scraper is initialized
            mock_scraper.search_mercado_livre.side_effect = Exception("Scraping failed")
            
            with patch('flask.current_app') as mock_app:
                mock_app.config = {
                    'DEFAULT_SEARCH_TERMS': ['test'],
                    'MAX_PAGES_PER_TERM': 1
                }
                
                products_found, products_stored, errors = service.run_scraping_job(max_retries=1)
                
                # Verify scraper.close() was called even though scraping failed
                mock_scraper.close.assert_called_once()
                
                # Verify job was marked as failed
                job = ScrapingJob.query.first()
                assert job.status == 'failed'
    
    def test_job_status_serialization(self, app_context, db_session):
        """Test that job status can be properly serialized"""
        service = ScraperService()
        
        # Create a job with complex data
        job = ScrapingJob(
            job_type='test',
            status='completed',
            search_terms=['term1', 'term2'],
            products_found=10,
            products_processed=8,
            errors=['Error 1', 'Error 2']
        )
        db_session.add(job)
        db_session.commit()
        
        # Get job status
        status = service.get_job_status(job.id)
        
        # Verify all fields are present and serializable
        assert status is not None
        assert status['job_type'] == 'test'
        assert status['status'] == 'completed'
        assert status['search_terms'] == ['term1', 'term2']
        assert status['products_found'] == 10
        assert status['products_processed'] == 8
        assert status['errors'] == ['Error 1', 'Error 2']
        
        # Verify it can be JSON serialized
        import json
        json_str = json.dumps(status, default=str)  # default=str for datetime objects
        assert json_str is not None