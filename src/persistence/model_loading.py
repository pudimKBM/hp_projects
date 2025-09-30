"""
Model Loading and Validation

Implements model deserialization with error handling, data schema compatibility
checking, and model performance verification after loading.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .metadata import ModelMetadata
from .model_saving import ModelSaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading of trained models with validation and compatibility checking
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize ModelLoader
        
        Args:
            base_path: Base directory where models are stored
        """
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "trained_models"
        self.metadata_dir = self.base_path / "metadata"
        self.pipelines_dir = self.base_path / "pipelines"
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"Models directory not found: {self.base_path}")
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        validate_performance: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Load model with metadata and optional validation
        
        Args:
            model_name: Name of the model to load
            version: Specific version to load (loads latest if None)
            validate_performance: Whether to validate model performance
            validation_data: Tuple of (X, y) for performance validation
        
        Returns:
            Dictionary containing model, metadata, and validation results
        """
        try:
            # Get model information from registry
            saver = ModelSaver(str(self.base_path))
            model_info = saver.get_model_info(model_name, version)
            
            if not model_info:
                raise FileNotFoundError(f"Model '{model_name}' version '{version}' not found")
            
            # Load model
            model_path = Path(model_info['model_file'])
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            
            # Load metadata
            metadata_path = Path(model_info['metadata_file'])
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = ModelMetadata.from_dict(metadata_dict)
            logger.info(f"Metadata loaded from: {metadata_path}")
            
            # Validate metadata
            if not metadata.validate():
                logger.warning("Loaded metadata failed validation")
            
            result = {
                'model': model,
                'metadata': metadata,
                'model_info': model_info,
                'load_successful': True,
                'validation_results': None
            }
            
            # Perform validation if requested
            if validate_performance and validation_data is not None:
                validation_results = self._validate_model_performance(
                    model, metadata, validation_data
                )
                result['validation_results'] = validation_results
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
            return {
                'model': None,
                'metadata': None,
                'model_info': None,
                'load_successful': False,
                'error': str(e),
                'validation_results': None
            }
    
    def _validate_model_performance(
        self,
        model: Any,
        metadata: ModelMetadata,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Validate model performance against stored metrics
        
        Args:
            model: Loaded model
            metadata: Model metadata
            validation_data: Tuple of (X, y) for validation
        
        Returns:
            Dictionary with validation results
        """
        X_val, y_val = validation_data
        
        try:
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate current metrics
            current_metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
            }
            
            # Compare with stored metrics
            stored_metrics = metadata.performance_metrics
            metric_differences = {}
            performance_degraded = False
            
            for metric, current_value in current_metrics.items():
                if metric in stored_metrics:
                    stored_value = stored_metrics[metric]
                    difference = current_value - stored_value
                    metric_differences[metric] = {
                        'current': current_value,
                        'stored': stored_value,
                        'difference': difference,
                        'degraded': difference < -0.05  # 5% threshold
                    }
                    
                    if difference < -0.05:
                        performance_degraded = True
            
            validation_results = {
                'validation_successful': True,
                'current_metrics': current_metrics,
                'stored_metrics': stored_metrics,
                'metric_differences': metric_differences,
                'performance_degraded': performance_degraded,
                'validation_data_size': len(y_val)
            }
            
            if performance_degraded:
                logger.warning("Model performance has degraded compared to stored metrics")
            else:
                logger.info("Model performance validation passed")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating model performance: {str(e)}")
            return {
                'validation_successful': False,
                'error': str(e),
                'current_metrics': None,
                'stored_metrics': stored_metrics,
                'metric_differences': None,
                'performance_degraded': None
            }
    
    def validate_data_schema(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        metadata: ModelMetadata
    ) -> Dict[str, Any]:
        """
        Validate data schema compatibility with model
        
        Args:
            data: Input data to validate
            metadata: Model metadata containing expected schema
        
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'schema_compatible': True,
                'issues': [],
                'warnings': []
            }
            
            expected_features = metadata.feature_names
            expected_count = len(expected_features)
            
            if isinstance(data, pd.DataFrame):
                actual_features = list(data.columns)
                actual_count = len(actual_features)
            elif isinstance(data, np.ndarray):
                actual_count = data.shape[1] if len(data.shape) > 1 else 1
                actual_features = [f"feature_{i}" for i in range(actual_count)]
            else:
                validation_results['schema_compatible'] = False
                validation_results['issues'].append("Unsupported data type")
                return validation_results
            
            # Check feature count
            if actual_count != expected_count:
                validation_results['schema_compatible'] = False
                validation_results['issues'].append(
                    f"Feature count mismatch: expected {expected_count}, got {actual_count}"
                )
            
            # Check feature names (for DataFrames)
            if isinstance(data, pd.DataFrame):
                missing_features = set(expected_features) - set(actual_features)
                extra_features = set(actual_features) - set(expected_features)
                
                if missing_features:
                    validation_results['schema_compatible'] = False
                    validation_results['issues'].append(
                        f"Missing features: {list(missing_features)}"
                    )
                
                if extra_features:
                    validation_results['warnings'].append(
                        f"Extra features (will be ignored): {list(extra_features)}"
                    )
            
            # Check data types and ranges (basic validation)
            if isinstance(data, pd.DataFrame):
                for feature in expected_features:
                    if feature in data.columns:
                        if data[feature].isnull().any():
                            validation_results['warnings'].append(
                                f"Feature '{feature}' contains null values"
                            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data schema: {str(e)}")
            return {
                'schema_compatible': False,
                'issues': [f"Schema validation error: {str(e)}"],
                'warnings': []
            }
    
    def load_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Load preprocessing pipeline
        
        Args:
            pipeline_name: Name of the pipeline to load
        
        Returns:
            Dictionary containing pipeline and metadata
        """
        try:
            # Find pipeline files
            pipeline_files = list(self.pipelines_dir.glob(f"{pipeline_name}_pipeline_*.joblib"))
            
            if not pipeline_files:
                raise FileNotFoundError(f"No pipeline found with name: {pipeline_name}")
            
            # Load latest pipeline (by filename timestamp)
            latest_pipeline_file = sorted(pipeline_files)[-1]
            pipeline = joblib.load(latest_pipeline_file)
            
            # Load metadata
            metadata_file = latest_pipeline_file.with_suffix('').with_suffix('_metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    pipeline_metadata = json.load(f)
            else:
                pipeline_metadata = None
                logger.warning(f"No metadata found for pipeline: {pipeline_name}")
            
            logger.info(f"Pipeline loaded from: {latest_pipeline_file}")
            
            return {
                'pipeline': pipeline,
                'metadata': pipeline_metadata,
                'file_path': str(latest_pipeline_file),
                'load_successful': True
            }
            
        except Exception as e:
            logger.error(f"Error loading pipeline '{pipeline_name}': {str(e)}")
            return {
                'pipeline': None,
                'metadata': None,
                'file_path': None,
                'load_successful': False,
                'error': str(e)
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Returns:
            List of available models with their information
        """
        saver = ModelSaver(str(self.base_path))
        return saver.list_saved_models()
    
    def check_model_integrity(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Check integrity of model files
        
        Args:
            model_name: Name of the model
            version: Specific version to check
        
        Returns:
            Dictionary with integrity check results
        """
        try:
            saver = ModelSaver(str(self.base_path))
            model_info = saver.get_model_info(model_name, version)
            
            if not model_info:
                return {
                    'integrity_ok': False,
                    'issues': [f"Model '{model_name}' version '{version}' not found in registry"]
                }
            
            issues = []
            
            # Check model file
            model_path = Path(model_info['model_file'])
            if not model_path.exists():
                issues.append(f"Model file missing: {model_path}")
            else:
                try:
                    # Try to load model
                    joblib.load(model_path)
                except Exception as e:
                    issues.append(f"Model file corrupted: {str(e)}")
            
            # Check metadata file
            metadata_path = Path(model_info['metadata_file'])
            if not metadata_path.exists():
                issues.append(f"Metadata file missing: {metadata_path}")
            else:
                try:
                    # Try to load and validate metadata
                    with open(metadata_path, 'r') as f:
                        metadata_dict = json.load(f)
                    metadata = ModelMetadata.from_dict(metadata_dict)
                    if not metadata.validate():
                        issues.append("Metadata validation failed")
                except Exception as e:
                    issues.append(f"Metadata file corrupted: {str(e)}")
            
            return {
                'integrity_ok': len(issues) == 0,
                'issues': issues,
                'model_info': model_info
            }
            
        except Exception as e:
            return {
                'integrity_ok': False,
                'issues': [f"Integrity check error: {str(e)}"],
                'model_info': None
            }


def load_model(
    model_name: str,
    version: Optional[str] = None,
    base_path: str = "models",
    validate_performance: bool = False,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Convenience function to load a model
    
    Args:
        model_name: Name of the model to load
        version: Specific version to load
        base_path: Base directory where models are stored
        validate_performance: Whether to validate model performance
        validation_data: Tuple of (X, y) for performance validation
    
    Returns:
        Dictionary containing model, metadata, and validation results
    """
    loader = ModelLoader(base_path)
    return loader.load_model(
        model_name=model_name,
        version=version,
        validate_performance=validate_performance,
        validation_data=validation_data
    )


def load_pipeline(pipeline_name: str, base_path: str = "models") -> Dict[str, Any]:
    """
    Convenience function to load a preprocessing pipeline
    
    Args:
        pipeline_name: Name of the pipeline to load
        base_path: Base directory where pipelines are stored
    
    Returns:
        Dictionary containing pipeline and metadata
    """
    loader = ModelLoader(base_path)
    return loader.load_pipeline(pipeline_name)