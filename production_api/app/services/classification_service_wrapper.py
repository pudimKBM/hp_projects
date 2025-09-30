"""
Classification Service Wrapper

Provides a simplified interface for the API endpoints to use the ClassificationEngine.
This service handles the integration between the API layer and the ML classification system.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from .classification_service import ClassificationEngine
from .ml_service import MLService
from .feature_service import FeaturePreparationService

logger = logging.getLogger(__name__)


class ClassificationService:
    """
    Simplified classification service for API endpoints
    """
    
    def __init__(self):
        """Initialize the classification service"""
        self._ml_service = None
        self._feature_service = None
        self._classification_engine = None
        self._default_model_name = 'best_model'
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize ML and feature services"""
        try:
            # Initialize ML service
            self._ml_service = MLService()
            
            # Initialize feature service
            self._feature_service = FeaturePreparationService()
            
            # Initialize classification engine
            self._classification_engine = ClassificationEngine(
                ml_service=self._ml_service,
                feature_service=self._feature_service,
                confidence_threshold=0.7
            )
            
            logger.info("Classification service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing classification service: {str(e)}")
            raise
    
    def classify_single_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single product and return API-formatted response
        
        Args:
            product_data: Product data dictionary with fields like title, price, etc.
        
        Returns:
            Dictionary formatted for API response
        """
        from production_api.app.services.performance_service import get_performance_service
        from production_api.app.utils.logging import log_classification_event
        
        start_time = datetime.now()
        
        try:
            # Validate input data
            validation_result = self._validate_product_data(product_data)
            if not validation_result['valid']:
                return {
                    'error': 'Validation Error',
                    'message': validation_result['message'],
                    'status_code': 400
                }
            
            # Perform classification with performance tracking
            classification_result = self._classification_engine.classify_product(
                product_data=product_data,
                model_name=self._default_model_name,
                include_explanation=True,
                top_features=5
            )
            
            if not classification_result['success']:
                return {
                    'error': 'Classification Error',
                    'message': classification_result.get('error', 'Classification failed'),
                    'status_code': 503
                }
            
            # Calculate total processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            classification_result['processing_time_ms'] = processing_time_ms
            
            # Track performance metrics
            try:
                perf_service = get_performance_service()
                perf_service.record_classification_performance(
                    processing_time_ms=processing_time_ms,
                    confidence_score=classification_result['confidence_score'],
                    prediction=classification_result['prediction_label']
                )
            except Exception as e:
                logger.warning(f"Failed to track classification performance: {e}")
            
            # Format response for API
            api_response = self._format_api_response(classification_result, product_data)
            
            # Log classification event
            try:
                log_classification_event(
                    product_id=0,  # Will be set when stored in database
                    prediction=classification_result['prediction_label'],
                    confidence_score=classification_result['confidence_score'],
                    processing_time_ms=processing_time_ms,
                    model_version=classification_result.get('model_version', 'unknown')
                )
            except Exception as e:
                logger.warning(f"Failed to log classification event: {e}")
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error in classify_single_product: {str(e)}")
            return {
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred during classification',
                'status_code': 500
            }
    
    def _validate_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate product data for classification
        
        Args:
            product_data: Product data dictionary
        
        Returns:
            Dictionary with validation results
        """
        try:
            # Required fields
            required_fields = ['title']
            missing_fields = [field for field in required_fields if field not in product_data or not product_data[field]]
            
            if missing_fields:
                return {
                    'valid': False,
                    'message': f'Missing required fields: {", ".join(missing_fields)}'
                }
            
            # Validate data types
            if 'price' in product_data:
                try:
                    float(product_data['price'])
                except (ValueError, TypeError):
                    return {
                        'valid': False,
                        'message': 'Price must be a valid number'
                    }
            
            if 'rating' in product_data:
                try:
                    rating = float(product_data['rating'])
                    if rating < 0 or rating > 5:
                        return {
                            'valid': False,
                            'message': 'Rating must be between 0 and 5'
                        }
                except (ValueError, TypeError):
                    return {
                        'valid': False,
                        'message': 'Rating must be a valid number'
                    }
            
            if 'reviews_count' in product_data:
                try:
                    reviews_count = int(product_data['reviews_count'])
                    if reviews_count < 0:
                        return {
                            'valid': False,
                            'message': 'Reviews count must be non-negative'
                        }
                except (ValueError, TypeError):
                    return {
                        'valid': False,
                        'message': 'Reviews count must be a valid integer'
                    }
            
            # Validate string lengths
            if len(product_data['title']) > 500:
                return {
                    'valid': False,
                    'message': 'Title must be less than 500 characters'
                }
            
            if 'description' in product_data and len(product_data['description']) > 2000:
                return {
                    'valid': False,
                    'message': 'Description must be less than 2000 characters'
                }
            
            return {'valid': True, 'message': 'Validation passed'}
            
        except Exception as e:
            logger.error(f"Error validating product data: {str(e)}")
            return {
                'valid': False,
                'message': f'Validation error: {str(e)}'
            }
    
    def _format_api_response(self, classification_result: Dict[str, Any], product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format classification result for API response
        
        Args:
            classification_result: Result from classification engine
            product_data: Original product data
        
        Returns:
            API-formatted response dictionary
        """
        try:
            # Extract explanation data
            explanation = classification_result.get('explanation', {})
            top_features = explanation.get('top_features', [])
            
            # Format feature importance for API
            feature_importance = []
            for feature_info in top_features[:5]:  # Top 5 features
                feature_importance.append({
                    'feature': feature_info.get('feature', 'Unknown'),
                    'importance': round(feature_info.get('importance', 0), 3),
                    'business_name': self._get_business_feature_name(feature_info.get('feature', ''))
                })
            
            # Create API response
            api_response = {
                'product_id': None,  # Will be set when stored in database
                'prediction': classification_result['prediction_label'],
                'confidence_score': round(classification_result['confidence_score'], 3),
                'explanation': {
                    'top_features': feature_importance,
                    'reasoning': explanation.get('business_explanation', 'Classification completed successfully'),
                    'confidence_level': classification_result.get('confidence_level', 'Unknown')
                },
                'processing_time_ms': classification_result['processing_time_ms'],
                'model_version': classification_result.get('model_version', 'unknown'),
                'classified_at': classification_result['timestamp']
            }
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error formatting API response: {str(e)}")
            return {
                'error': 'Response Formatting Error',
                'message': 'Error formatting classification response',
                'status_code': 500
            }
    
    def _get_business_feature_name(self, technical_name: str) -> str:
        """
        Convert technical feature name to business-friendly name
        
        Args:
            technical_name: Technical feature name
        
        Returns:
            Business-friendly feature name
        """
        # Simple mapping for common features
        mapping = {
            'title_length': 'Title Length',
            'price_numeric': 'Product Price',
            'rating_numeric': 'Customer Rating',
            'reviews_count': 'Number of Reviews',
            'seller_name': 'Seller Information',
            'has_hp_keyword': 'HP Brand Keywords',
            'has_original_keyword': 'Authenticity Keywords',
            'description_length': 'Description Length'
        }
        
        # Check for TF-IDF features
        if 'tfidf' in technical_name.lower():
            word = technical_name.split('_')[-1] if '_' in technical_name else technical_name
            return f'Text Analysis: {word.title()}'
        
        return mapping.get(technical_name, technical_name.replace('_', ' ').title())
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of the classification service
        
        Returns:
            Dictionary with service health information
        """
        try:
            health_status = {
                'status': 'healthy',
                'ml_service_loaded': self._ml_service is not None,
                'feature_service_loaded': self._feature_service is not None,
                'classification_engine_loaded': self._classification_engine is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get classification statistics if available
            if self._classification_engine:
                stats = self._classification_engine.get_classification_stats()
                health_status['classification_stats'] = stats
            
            # Check if models are loaded
            if self._ml_service:
                loaded_models = self._ml_service.list_loaded_models()
                health_status['loaded_models'] = loaded_models
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting service health: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def is_service_ready(self) -> bool:
        """
        Check if the service is ready to handle requests
        
        Returns:
            True if service is ready, False otherwise
        """
        try:
            return (
                self._ml_service is not None and
                self._feature_service is not None and
                self._classification_engine is not None
            )
        except Exception:
            return False