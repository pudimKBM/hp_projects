"""
Unit tests for Classification Service

Tests ML classification service with mock models and data.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from production_api.app.services.classification_service import (
    ClassificationEngine, 
    create_classification_engine
)


class TestClassificationEngine:
    """Test cases for ClassificationEngine class"""
    
    def test_init(self):
        """Test ClassificationEngine initialization"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(
            ml_service=mock_ml_service,
            feature_service=mock_feature_service,
            confidence_threshold=0.8
        )
        
        assert engine.ml_service == mock_ml_service
        assert engine.feature_service == mock_feature_service
        assert engine.confidence_threshold == 0.8
        assert engine.class_names == ['suspicious', 'original']
        assert engine.interpretation_pipeline is None
        assert engine.classification_stats['total_classifications'] == 0
    
    def test_setup_interpretation_pipeline(self):
        """Test interpretation pipeline setup"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        feature_names = ['has_hp_keyword', 'price_numeric', 'seller_reputation']
        
        with patch('production_api.app.services.classification_service.InterpretationPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            result = engine.setup_interpretation_pipeline(feature_names)
            
            assert result['success'] is True
            assert result['feature_count'] == 3
            assert engine.interpretation_pipeline == mock_pipeline
            
            # Verify pipeline was created with correct parameters
            mock_pipeline_class.assert_called_once()
            call_args = mock_pipeline_class.call_args
            assert call_args[1]['feature_names'] == feature_names
            assert call_args[1]['class_names'] == ['suspicious', 'original']
    
    def test_classify_product_success(self):
        """Test successful product classification"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # 'original'
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        mock_model.classes_ = np.array([0, 1])
        
        mock_active_model = {
            'model': mock_model,
            'metadata': {'model_version': 'v1.0'}
        }
        
        mock_ml_service.get_active_model.return_value = mock_active_model
        
        # Setup mock feature service
        mock_features = np.array([[1, 89.90, 0.8]])
        mock_feature_names = ['has_hp_keyword', 'price_numeric', 'seller_reputation']
        
        mock_feature_service.prepare_features_for_single_product.return_value = {
            'success': True,
            'features': mock_features,
            'feature_names': mock_feature_names,
            'validation_result': {'valid': True}
        }
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        product_data = {
            'title': 'Cartucho HP 664 Original',
            'price': 89.90,
            'seller_name': 'HP Store'
        }
        
        result = engine.classify_product(product_data, 'test_model')
        
        # Assertions
        assert result['success'] is True
        assert result['prediction'] == 1
        assert result['prediction_label'] == 'original'
        assert result['confidence_score'] == 0.85
        assert result['confidence_level'] == 'High'
        assert result['model_name'] == 'test_model'
        assert 'processing_time_ms' in result
        assert 'timestamp' in result
        
        # Verify service calls
        mock_ml_service.get_active_model.assert_called_once_with('test_model')
        mock_feature_service.prepare_features_for_single_product.assert_called_once_with(product_data)
    
    def test_classify_product_model_not_loaded(self):
        """Test classification when model is not loaded"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        mock_ml_service.get_active_model.return_value = None
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        result = engine.classify_product({'title': 'Test'}, 'nonexistent_model')
        
        assert result['success'] is False
        assert 'Model nonexistent_model not loaded' in result['error']
        assert result['prediction'] is None
        assert result['confidence_score'] == 0.0
    
    def test_classify_product_feature_preparation_failure(self):
        """Test classification when feature preparation fails"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        # Setup mock model
        mock_active_model = {'model': Mock()}
        mock_ml_service.get_active_model.return_value = mock_active_model
        
        # Setup feature service to fail
        mock_feature_service.prepare_features_for_single_product.return_value = {
            'success': False,
            'error': 'Invalid product data'
        }
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        result = engine.classify_product({'title': 'Test'}, 'test_model')
        
        assert result['success'] is False
        assert 'Feature preparation failed' in result['error']
        assert result['prediction'] is None
    
    def test_classify_batch_success(self):
        """Test successful batch classification"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Mock classify_product to return success for all products
        mock_results = [
            {
                'success': True,
                'prediction': 1,
                'prediction_label': 'original',
                'confidence_score': 0.85
            },
            {
                'success': True,
                'prediction': 0,
                'prediction_label': 'suspicious',
                'confidence_score': 0.75
            }
        ]
        
        with patch.object(engine, 'classify_product') as mock_classify:
            mock_classify.side_effect = mock_results
            
            products_data = [
                {'title': 'Product 1'},
                {'title': 'Product 2'}
            ]
            
            result = engine.classify_batch(products_data, 'test_model')
            
            assert result['success'] is True
            assert result['total_products'] == 2
            assert result['successful_classifications'] == 2
            assert result['failed_classifications'] == 0
            assert result['success_rate'] == 1.0
            assert len(result['results']) == 2
            
            # Verify batch indices were added
            assert result['results'][0]['batch_index'] == 0
            assert result['results'][1]['batch_index'] == 1
    
    def test_classify_batch_partial_failure(self):
        """Test batch classification with some failures"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Mock classify_product with mixed results
        mock_results = [
            {
                'success': True,
                'prediction': 1,
                'confidence_score': 0.85
            },
            {
                'success': False,
                'error': 'Classification failed'
            }
        ]
        
        with patch.object(engine, 'classify_product') as mock_classify:
            mock_classify.side_effect = mock_results
            
            products_data = [
                {'title': 'Product 1'},
                {'title': 'Product 2'}
            ]
            
            result = engine.classify_batch(products_data, 'test_model')
            
            assert result['success'] is True
            assert result['total_products'] == 2
            assert result['successful_classifications'] == 1
            assert result['failed_classifications'] == 1
            assert result['success_rate'] == 0.5
    
    def test_make_prediction_success(self):
        """Test successful prediction making"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_model.classes_ = np.array([0, 1])
        
        features = np.array([1, 89.90, 0.8])
        feature_names = ['has_hp_keyword', 'price_numeric', 'seller_reputation']
        
        result = engine._make_prediction(mock_model, features, feature_names)
        
        assert result['success'] is True
        assert result['prediction'] == 1
        assert result['prediction_label'] == 'original'
        assert result['confidence_score'] == 0.8
        assert result['probabilities'] == [0.2, 0.8]
    
    def test_make_prediction_no_proba(self):
        """Test prediction with model that doesn't support predict_proba"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Setup mock model without predict_proba
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        # Remove predict_proba method
        del mock_model.predict_proba
        
        features = np.array([1, 89.90, 0.8])
        feature_names = ['has_hp_keyword', 'price_numeric', 'seller_reputation']
        
        result = engine._make_prediction(mock_model, features, feature_names)
        
        assert result['success'] is True
        assert result['prediction'] == 0
        assert result['prediction_label'] == 'suspicious'
        assert result['confidence_score'] == 0.8  # Default confidence
        assert result['probabilities'] is None
    
    def test_generate_explanation(self):
        """Test explanation generation"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Setup interpretation pipeline
        mock_pipeline = Mock()
        engine.interpretation_pipeline = mock_pipeline
        
        # Setup mock model and data
        mock_model = Mock()
        features = np.array([[1, 89.90, 0.8]])
        feature_names = ['has_hp_keyword', 'price_numeric', 'seller_reputation']
        prediction_result = {
            'prediction_label': 'original',
            'confidence_score': 0.85
        }
        
        # Mock explanation function
        mock_explanation = {
            'top_features': [
                {'feature': 'has_hp_keyword', 'importance': 0.4},
                {'feature': 'seller_reputation', 'importance': 0.3}
            ],
            'feature_importance': {
                'has_hp_keyword': 0.4,
                'seller_reputation': 0.3
            }
        }
        
        with patch('production_api.app.services.classification_service.explain_prediction') as mock_explain:
            mock_explain.return_value = mock_explanation
            
            result = engine._generate_explanation(
                mock_model, features, feature_names, prediction_result, 5
            )
            
            assert 'technical_explanation' in result
            assert 'business_explanation' in result
            assert 'top_features' in result
            assert len(result['top_features']) <= 5
            
            # Verify explain_prediction was called correctly
            mock_explain.assert_called_once_with(
                model=mock_model,
                X_sample=features,
                feature_names=feature_names,
                class_names=['suspicious', 'original'],
                top_n=5
            )
    
    def test_create_business_explanation(self):
        """Test business-friendly explanation creation"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        technical_explanation = {
            'top_features': [
                {'feature': 'has_hp_keyword', 'importance': 0.4},
                {'feature': 'seller_reputation', 'importance': -0.2}
            ]
        }
        
        prediction_result = {
            'prediction_label': 'original',
            'confidence_score': 0.85
        }
        
        explanation = engine._create_business_explanation(
            technical_explanation, prediction_result
        )
        
        assert 'original' in explanation
        assert '85%' in explanation or '0.85' in explanation
        assert 'high confidence' in explanation.lower()
        assert 'supports' in explanation
        assert 'contradicts' in explanation
    
    def test_get_confidence_level(self):
        """Test confidence level mapping"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        assert engine._get_confidence_level(0.95) == "Very High"
        assert engine._get_confidence_level(0.85) == "High"
        assert engine._get_confidence_level(0.75) == "Moderate"
        assert engine._get_confidence_level(0.65) == "Low"
        assert engine._get_confidence_level(0.45) == "Very Low"
    
    def test_update_classification_stats(self):
        """Test classification statistics update"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Update stats with successful classification
        engine._update_classification_stats(0.85, 150, True)
        
        stats = engine.classification_stats
        assert stats['total_classifications'] == 1
        assert stats['successful_classifications'] == 1
        assert stats['failed_classifications'] == 0
        assert stats['avg_processing_time_ms'] == 150
        assert len(stats['confidence_distribution']) == 1
        assert stats['confidence_distribution'][0] == 0.85
        
        # Update with failed classification
        engine._update_classification_stats(0.0, 200, False)
        
        stats = engine.classification_stats
        assert stats['total_classifications'] == 2
        assert stats['successful_classifications'] == 1
        assert stats['failed_classifications'] == 1
        assert stats['avg_processing_time_ms'] == 175  # (150 + 200) / 2
    
    def test_get_classification_stats(self):
        """Test getting classification statistics"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Add some test data
        engine.classification_stats['confidence_distribution'] = [0.8, 0.9, 0.7]
        engine.classification_stats['total_classifications'] = 5
        engine.classification_stats['successful_classifications'] = 4
        
        stats = engine.get_classification_stats()
        
        assert stats['avg_confidence'] == pytest.approx(0.8, rel=1e-2)
        assert stats['min_confidence'] == 0.7
        assert stats['max_confidence'] == 0.9
        assert stats['success_rate'] == 0.8
        assert 'confidence_std' in stats
    
    def test_reset_stats(self):
        """Test resetting classification statistics"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        # Add some data
        engine.classification_stats['total_classifications'] = 10
        engine.classification_stats['confidence_distribution'] = [0.8, 0.9]
        
        # Reset
        engine.reset_stats()
        
        stats = engine.classification_stats
        assert stats['total_classifications'] == 0
        assert stats['successful_classifications'] == 0
        assert stats['failed_classifications'] == 0
        assert stats['avg_processing_time_ms'] == 0
        assert len(stats['confidence_distribution']) == 0
    
    def test_create_business_feature_mapping(self):
        """Test business feature mapping creation"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = ClassificationEngine(mock_ml_service, mock_feature_service)
        
        feature_names = [
            'title_length',
            'description_keywords',
            'price_numeric',
            'rating_avg',
            'reviews_count',
            'seller_reputation',
            'platform_mercadolivre',
            'has_hp_keyword',
            'has_original_keyword',
            'tfidf_cartucho'
        ]
        
        mapping = engine._create_business_feature_mapping(feature_names)
        
        assert 'title_length' in mapping
        assert 'Product Title Content' in mapping.values()
        assert 'Product Description Content' in mapping.values()
        assert 'Price Information' in mapping.values()
        assert 'Customer Rating' in mapping.values()
        assert 'HP Brand Indicators' in mapping.values()
        assert 'Text Analysis: Cartucho' in mapping.values()


class TestClassificationEngineIntegration:
    """Integration tests for ClassificationEngine"""
    
    def test_full_classification_workflow(self):
        """Test complete classification workflow"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
        mock_model.classes_ = np.array([0, 1])
        
        mock_active_model = {
            'model': mock_model,
            'metadata': {'model_version': 'v1.0'}
        }
        
        mock_ml_service.get_active_model.return_value = mock_active_model
        
        # Setup mock feature service
        mock_features = np.array([[1, 89.90, 0.8, 4.5, 150]])
        mock_feature_names = ['has_hp_keyword', 'price_numeric', 'seller_reputation', 'rating', 'reviews']
        
        mock_feature_service.prepare_features_for_single_product.return_value = {
            'success': True,
            'features': mock_features,
            'feature_names': mock_feature_names,
            'validation_result': {'valid': True}
        }
        
        # Create engine and setup interpretation
        engine = ClassificationEngine(mock_ml_service, mock_feature_service, confidence_threshold=0.7)
        
        # Setup interpretation pipeline
        setup_result = engine.setup_interpretation_pipeline(mock_feature_names)
        assert setup_result['success'] is True
        
        # Mock explanation
        mock_explanation = {
            'top_features': [
                {'feature': 'has_hp_keyword', 'importance': 0.4},
                {'feature': 'seller_reputation', 'importance': 0.3}
            ],
            'feature_importance': {'has_hp_keyword': 0.4}
        }
        
        with patch('production_api.app.services.classification_service.explain_prediction') as mock_explain:
            mock_explain.return_value = mock_explanation
            
            # Classify product
            product_data = {
                'title': 'Cartucho HP 664 Original',
                'description': 'Cartucho original HP',
                'price': 89.90,
                'seller_name': 'HP Store Oficial',
                'rating': 4.8,
                'reviews_count': 150
            }
            
            result = engine.classify_product(product_data, 'hp_classifier', include_explanation=True)
            
            # Verify complete result
            assert result['success'] is True
            assert result['prediction'] == 1
            assert result['prediction_label'] == 'original'
            assert result['confidence_score'] == 0.75
            assert result['confidence_level'] == 'Moderate'
            assert result['model_name'] == 'hp_classifier'
            assert result['explanation'] is not None
            assert 'business_explanation' in result['explanation']
            assert 'technical_explanation' in result['explanation']
            
            # Verify statistics were updated
            stats = engine.get_classification_stats()
            assert stats['total_classifications'] == 1
            assert stats['successful_classifications'] == 1
            assert stats['success_rate'] == 1.0


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_classification_engine(self):
        """Test create_classification_engine function"""
        mock_ml_service = Mock()
        mock_feature_service = Mock()
        
        engine = create_classification_engine(
            ml_service=mock_ml_service,
            feature_service=mock_feature_service,
            confidence_threshold=0.8
        )
        
        assert isinstance(engine, ClassificationEngine)
        assert engine.ml_service == mock_ml_service
        assert engine.feature_service == mock_feature_service
        assert engine.confidence_threshold == 0.8