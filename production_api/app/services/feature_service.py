"""
Feature Preparation Service

Adapts the existing feature engineering pipeline for single product processing
and provides data validation and preprocessing for API inputs.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from scipy.sparse import csr_matrix, hstack

# Import from existing feature engineering module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.feature_engineering import FeatureEngineeringPipeline
from src.feature_engineering.text_features import extract_all_text_features
from src.feature_engineering.numerical_features import create_numerical_features
from src.feature_engineering.categorical_features import create_categorical_features, handle_unseen_categories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePreparationService:
    """
    Service for preparing features from single product data for classification
    """
    
    def __init__(self, pipeline_path: Optional[str] = None):
        """
        Initialize Feature Preparation Service
        
        Args:
            pipeline_path: Path to saved feature engineering pipeline
        """
        self.pipeline_path = pipeline_path
        self.fitted_pipeline: Optional[FeatureEngineeringPipeline] = None
        self.is_ready = False
        
        # Expected input schema
        self.expected_columns = [
            'title', 'description', 'price_numeric', 'rating_numeric', 
            'reviews_count', 'platform', 'product_type', 'seller_name'
        ]
        
        # Default values for missing fields
        self.default_values = {
            'title': '',
            'description': '',
            'price_numeric': 0.0,
            'rating_numeric': 0.0,
            'reviews_count': 0,
            'platform': 'unknown',
            'product_type': 'unknown',
            'seller_name': 'unknown'
        }
        
        logger.info("FeaturePreparationService initialized")
    
    def load_fitted_pipeline(self, pipeline_path: str) -> Dict[str, Any]:
        """
        Load a pre-fitted feature engineering pipeline
        
        Args:
            pipeline_path: Path to the saved pipeline file
        
        Returns:
            Dictionary with load results
        """
        try:
            logger.info(f"Loading fitted pipeline from: {pipeline_path}")
            
            if not Path(pipeline_path).exists():
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            
            # Load the fitted pipeline
            self.fitted_pipeline = FeatureEngineeringPipeline.load_pipeline(pipeline_path)
            self.pipeline_path = pipeline_path
            self.is_ready = True
            
            logger.info("Pipeline loaded successfully")
            
            return {
                'success': True,
                'pipeline_path': pipeline_path,
                'feature_count': len(self.fitted_pipeline.get_feature_names()),
                'feature_metadata': self.fitted_pipeline.get_feature_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            self.is_ready = False
            return {
                'success': False,
                'error': str(e),
                'pipeline_path': pipeline_path
            }
    
    def validate_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input product data for API requests
        
        Args:
            product_data: Dictionary containing product information
        
        Returns:
            Dictionary with validation results and cleaned data
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_data': {},
            'missing_fields': [],
            'invalid_fields': []
        }
        
        try:
            # Check for required fields and apply defaults
            cleaned_data = {}
            
            for column in self.expected_columns:
                if column in product_data:
                    value = product_data[column]
                    
                    # Validate and clean the value
                    cleaned_value = self._validate_and_clean_field(column, value)
                    
                    if cleaned_value is None:
                        validation_result['invalid_fields'].append(column)
                        validation_result['warnings'].append(f"Invalid value for {column}, using default")
                        cleaned_data[column] = self.default_values[column]
                    else:
                        cleaned_data[column] = cleaned_value
                else:
                    # Missing field, use default
                    validation_result['missing_fields'].append(column)
                    cleaned_data[column] = self.default_values[column]
            
            # Additional validation checks
            self._perform_business_validation(cleaned_data, validation_result)
            
            validation_result['cleaned_data'] = cleaned_data
            
            # Log validation results
            if validation_result['missing_fields']:
                logger.info(f"Missing fields filled with defaults: {validation_result['missing_fields']}")
            
            if validation_result['invalid_fields']:
                logger.warning(f"Invalid fields corrected: {validation_result['invalid_fields']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating product data: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def prepare_features_for_single_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features for a single product using the fitted pipeline
        
        Args:
            product_data: Dictionary containing product information
        
        Returns:
            Dictionary with prepared features and metadata
        """
        if not self.is_ready or not self.fitted_pipeline:
            return {
                'success': False,
                'error': 'Pipeline not loaded. Call load_fitted_pipeline() first.',
                'features': None
            }
        
        try:
            # Validate input data
            validation_result = self.validate_product_data(product_data)
            
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error': f"Data validation failed: {validation_result['errors']}",
                    'validation_result': validation_result,
                    'features': None
                }
            
            # Convert to DataFrame (single row)
            df = pd.DataFrame([validation_result['cleaned_data']])
            
            # Transform using fitted pipeline
            feature_matrix = self._transform_single_product(df)
            
            # Get feature names
            feature_names = self.fitted_pipeline.get_feature_names()
            
            # Convert sparse matrix to dense for easier handling
            if hasattr(feature_matrix, 'toarray'):
                features_array = feature_matrix.toarray()[0]  # Get first (and only) row
            else:
                features_array = feature_matrix[0] if len(feature_matrix.shape) > 1 else feature_matrix
            
            return {
                'success': True,
                'features': features_array,
                'feature_names': feature_names,
                'feature_count': len(feature_names),
                'validation_result': validation_result,
                'original_data': product_data,
                'cleaned_data': validation_result['cleaned_data']
            }
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'features': None
            }
    
    def _transform_single_product(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform single product DataFrame using fitted pipeline components
        
        Args:
            df: Single-row DataFrame with product data
        
        Returns:
            Transformed feature matrix
        """
        # Get the fitted components from the pipeline
        text_vectorizers = self.fitted_pipeline.text_vectorizers
        numerical_scaler = self.fitted_pipeline.numerical_scaler
        categorical_encoders = self.fitted_pipeline.categorical_encoders
        
        # Get column configurations
        text_columns = self.fitted_pipeline.text_columns
        numerical_columns = self.fitted_pipeline.numerical_columns
        onehot_columns = self.fitted_pipeline.onehot_columns
        label_columns = self.fitted_pipeline.label_columns
        
        feature_matrices = []
        
        # 1. Transform text features
        if text_vectorizers:
            text_features = []
            for col in text_columns:
                if col in text_vectorizers:
                    vectorizer = text_vectorizers[col]
                    text_vector = vectorizer.transform(df[col])
                    text_features.append(text_vector)
            
            if text_features:
                text_matrix = hstack(text_features)
                feature_matrices.append(text_matrix)
        
        # 2. Transform numerical features
        if numerical_scaler and numerical_columns:
            numerical_data = df[numerical_columns].values
            scaled_numerical = numerical_scaler.transform(numerical_data)
            feature_matrices.append(csr_matrix(scaled_numerical))
        
        # 3. Transform categorical features
        if categorical_encoders:
            categorical_matrices = []
            
            # One-hot encoded features
            for col in onehot_columns:
                if col in categorical_encoders:
                    encoder = categorical_encoders[col]
                    # Handle unseen categories
                    try:
                        encoded = encoder.transform(df[[col]])
                        categorical_matrices.append(csr_matrix(encoded))
                    except ValueError:
                        # Handle unseen category by creating zero vector
                        logger.warning(f"Unseen category in {col}: {df[col].iloc[0]}")
                        zero_vector = csr_matrix((1, len(encoder.get_feature_names_out())))
                        categorical_matrices.append(zero_vector)
            
            # Label encoded features
            for col in label_columns:
                if col in categorical_encoders:
                    encoder = categorical_encoders[col]
                    try:
                        encoded = encoder.transform(df[col])
                        categorical_matrices.append(csr_matrix(encoded.reshape(-1, 1)))
                    except ValueError:
                        # Handle unseen category
                        logger.warning(f"Unseen category in {col}: {df[col].iloc[0]}")
                        # Use a default value (e.g., 0 or most frequent class)
                        default_encoded = np.array([0])
                        categorical_matrices.append(csr_matrix(default_encoded.reshape(-1, 1)))
            
            if categorical_matrices:
                categorical_matrix = hstack(categorical_matrices)
                feature_matrices.append(categorical_matrix)
        
        # Combine all feature matrices
        if feature_matrices:
            combined_matrix = hstack(feature_matrices)
            
            # Apply correlation filtering if available
            if hasattr(self.fitted_pipeline, 'correlation_info'):
                correlation_info = self.fitted_pipeline.correlation_info
                if 'filtered_features' in correlation_info:
                    # Apply the same feature selection as during training
                    selected_indices = correlation_info['filtered_features'].get('selected_indices')
                    if selected_indices is not None:
                        combined_matrix = combined_matrix[:, selected_indices]
            
            return combined_matrix
        else:
            # Return empty matrix if no features
            return csr_matrix((1, 0))
    
    def _validate_and_clean_field(self, field_name: str, value: Any) -> Any:
        """
        Validate and clean individual field values
        
        Args:
            field_name: Name of the field
            value: Value to validate
        
        Returns:
            Cleaned value or None if invalid
        """
        try:
            if field_name in ['title', 'description', 'platform', 'product_type', 'seller_name']:
                # String fields
                if value is None:
                    return ''
                return str(value).strip()
            
            elif field_name in ['price_numeric', 'rating_numeric']:
                # Float fields
                if value is None or value == '':
                    return 0.0
                return float(value)
            
            elif field_name == 'reviews_count':
                # Integer field
                if value is None or value == '':
                    return 0
                return int(float(value))  # Handle string numbers
            
            else:
                # Unknown field, return as-is
                return value
                
        except (ValueError, TypeError):
            # Return None for invalid values
            return None
    
    def _perform_business_validation(self, data: Dict[str, Any], validation_result: Dict[str, Any]):
        """
        Perform business logic validation on the cleaned data
        
        Args:
            data: Cleaned product data
            validation_result: Validation result dictionary to update
        """
        # Price validation
        if data['price_numeric'] < 0:
            validation_result['warnings'].append("Negative price detected, setting to 0")
            data['price_numeric'] = 0.0
        
        # Rating validation
        if data['rating_numeric'] < 0 or data['rating_numeric'] > 5:
            validation_result['warnings'].append("Rating out of range [0-5], clamping")
            data['rating_numeric'] = max(0, min(5, data['rating_numeric']))
        
        # Reviews count validation
        if data['reviews_count'] < 0:
            validation_result['warnings'].append("Negative reviews count, setting to 0")
            data['reviews_count'] = 0
        
        # Text field validation
        for text_field in ['title', 'description']:
            if len(data[text_field]) > 10000:  # Reasonable limit
                validation_result['warnings'].append(f"{text_field} too long, truncating")
                data[text_field] = data[text_field][:10000]
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get service status information
        
        Returns:
            Dictionary with service status
        """
        status = {
            'service_ready': self.is_ready,
            'pipeline_loaded': self.fitted_pipeline is not None,
            'pipeline_path': self.pipeline_path,
            'expected_columns': self.expected_columns
        }
        
        if self.fitted_pipeline:
            try:
                status['feature_count'] = len(self.fitted_pipeline.get_feature_names())
                status['feature_metadata'] = self.fitted_pipeline.get_feature_metadata()
            except Exception as e:
                status['pipeline_error'] = str(e)
        
        return status
    
    def get_feature_schema(self) -> Dict[str, Any]:
        """
        Get the expected input schema for product data
        
        Returns:
            Dictionary describing expected input format
        """
        return {
            'required_fields': self.expected_columns,
            'field_types': {
                'title': 'string',
                'description': 'string',
                'price_numeric': 'float',
                'rating_numeric': 'float (0-5)',
                'reviews_count': 'integer',
                'platform': 'string',
                'product_type': 'string',
                'seller_name': 'string'
            },
            'default_values': self.default_values,
            'validation_rules': {
                'price_numeric': 'Must be >= 0',
                'rating_numeric': 'Must be between 0 and 5',
                'reviews_count': 'Must be >= 0',
                'title': 'Max length 10000 characters',
                'description': 'Max length 10000 characters'
            }
        }


# Convenience functions
def create_feature_service(pipeline_path: Optional[str] = None) -> FeaturePreparationService:
    """
    Create and return a FeaturePreparationService instance
    
    Args:
        pipeline_path: Path to saved pipeline
    
    Returns:
        Configured FeaturePreparationService instance
    """
    service = FeaturePreparationService(pipeline_path)
    if pipeline_path:
        service.load_fitted_pipeline(pipeline_path)
    return service


def prepare_single_product_features(
    product_data: Dict[str, Any], 
    pipeline_path: str
) -> Dict[str, Any]:
    """
    Convenience function to prepare features for a single product
    
    Args:
        product_data: Product data dictionary
        pipeline_path: Path to fitted pipeline
    
    Returns:
        Dictionary with prepared features
    """
    service = FeaturePreparationService()
    load_result = service.load_fitted_pipeline(pipeline_path)
    
    if not load_result['success']:
        return load_result
    
    return service.prepare_features_for_single_product(product_data)