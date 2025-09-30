"""
ML Classification Service

Provides model loading, validation, and classification functionality for the production API.
Integrates with the existing persistence module to load trained models and handle fallbacks.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Import from existing persistence module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.persistence import ModelLoader, ModelMetadata
from src.persistence.model_loading import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    """
    ML Classification Service that handles model loading, validation, and fallback mechanisms
    """
    
    def __init__(self, models_base_path: str = "models"):
        """
        Initialize ML Service
        
        Args:
            models_base_path: Base path to models directory
        """
        self.models_base_path = Path(models_base_path)
        self.model_loader = ModelLoader(str(self.models_base_path))
        
        # Model cache and metadata
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self._model_versions: Dict[str, str] = {}
        self._fallback_models: List[str] = []
        
        # Service status
        self._service_healthy = True
        self._last_health_check = None
        
        logger.info(f"MLService initialized with models path: {self.models_base_path}")
    
    def load_primary_model(
        self, 
        model_name: str, 
        version: Optional[str] = None,
        validate_performance: bool = True
    ) -> Dict[str, Any]:
        """
        Load the primary classification model
        
        Args:
            model_name: Name of the model to load
            version: Specific version to load (latest if None)
            validate_performance: Whether to validate model performance
        
        Returns:
            Dictionary with load results and model information
        """
        try:
            logger.info(f"Loading primary model: {model_name}, version: {version}")
            
            # Load model using existing persistence module
            load_result = self.model_loader.load_model(
                model_name=model_name,
                version=version,
                validate_performance=validate_performance
            )
            
            if load_result['load_successful']:
                # Cache the loaded model
                cache_key = f"{model_name}_{version or 'latest'}"
                self._loaded_models[cache_key] = load_result
                self._model_versions[model_name] = version or 'latest'
                
                logger.info(f"Successfully loaded model: {model_name}")
                return {
                    'success': True,
                    'model_name': model_name,
                    'version': self._get_model_version(load_result),
                    'load_time': datetime.now().isoformat(),
                    'metadata': load_result['metadata'].to_dict() if load_result['metadata'] else None,
                    'validation_passed': self._check_validation_results(load_result.get('validation_results'))
                }
            else:
                logger.error(f"Failed to load model {model_name}: {load_result.get('error')}")
                return {
                    'success': False,
                    'model_name': model_name,
                    'error': load_result.get('error'),
                    'fallback_available': len(self._fallback_models) > 0
                }
                
        except Exception as e:
            logger.error(f"Exception loading model {model_name}: {str(e)}")
            return {
                'success': False,
                'model_name': model_name,
                'error': str(e),
                'fallback_available': len(self._fallback_models) > 0
            }
    
    def setup_fallback_models(self, fallback_model_names: List[str]) -> Dict[str, Any]:
        """
        Setup fallback models for redundancy
        
        Args:
            fallback_model_names: List of model names to use as fallbacks
        
        Returns:
            Dictionary with setup results
        """
        logger.info(f"Setting up fallback models: {fallback_model_names}")
        
        successful_fallbacks = []
        failed_fallbacks = []
        
        for model_name in fallback_model_names:
            try:
                # Check if model exists and can be loaded
                integrity_check = self.model_loader.check_model_integrity(model_name)
                
                if integrity_check['integrity_ok']:
                    successful_fallbacks.append(model_name)
                    logger.info(f"Fallback model validated: {model_name}")
                else:
                    failed_fallbacks.append({
                        'model_name': model_name,
                        'issues': integrity_check['issues']
                    })
                    logger.warning(f"Fallback model failed validation: {model_name}")
                    
            except Exception as e:
                failed_fallbacks.append({
                    'model_name': model_name,
                    'issues': [str(e)]
                })
                logger.error(f"Error validating fallback model {model_name}: {str(e)}")
        
        self._fallback_models = successful_fallbacks
        
        return {
            'successful_fallbacks': successful_fallbacks,
            'failed_fallbacks': failed_fallbacks,
            'total_fallbacks_available': len(successful_fallbacks)
        }
    
    def get_active_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the currently active model
        
        Args:
            model_name: Name of the model to retrieve
        
        Returns:
            Active model information or None if not loaded
        """
        version = self._model_versions.get(model_name, 'latest')
        cache_key = f"{model_name}_{version}"
        
        return self._loaded_models.get(cache_key)
    
    def validate_model_health(self, model_name: str) -> Dict[str, Any]:
        """
        Validate the health of a loaded model
        
        Args:
            model_name: Name of the model to validate
        
        Returns:
            Dictionary with health check results
        """
        try:
            active_model = self.get_active_model(model_name)
            
            if not active_model:
                return {
                    'healthy': False,
                    'issues': ['Model not loaded'],
                    'last_check': datetime.now().isoformat()
                }
            
            issues = []
            
            # Check if model object exists and is callable
            model_obj = active_model.get('model')
            if not model_obj:
                issues.append('Model object is None')
            elif not hasattr(model_obj, 'predict'):
                issues.append('Model object missing predict method')
            
            # Check metadata
            metadata = active_model.get('metadata')
            if not metadata:
                issues.append('Model metadata missing')
            elif not metadata.validate():
                issues.append('Model metadata validation failed')
            
            # Check validation results if available
            validation_results = active_model.get('validation_results')
            if validation_results and validation_results.get('performance_degraded'):
                issues.append('Model performance has degraded')
            
            health_status = {
                'healthy': len(issues) == 0,
                'issues': issues,
                'last_check': datetime.now().isoformat(),
                'model_version': self._get_model_version(active_model),
                'load_time': active_model.get('load_time')
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error validating model health for {model_name}: {str(e)}")
            return {
                'healthy': False,
                'issues': [f'Health check error: {str(e)}'],
                'last_check': datetime.now().isoformat()
            }
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a loaded model
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model metadata dictionary or None
        """
        active_model = self.get_active_model(model_name)
        if active_model and active_model.get('metadata'):
            return active_model['metadata'].to_dict()
        return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in the models directory
        
        Returns:
            List of available models with their information
        """
        try:
            return self.model_loader.list_available_models()
        except Exception as e:
            logger.error(f"Error listing available models: {str(e)}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get overall service status
        
        Returns:
            Dictionary with service status information
        """
        loaded_models_status = {}
        
        for model_name in self._model_versions.keys():
            health = self.validate_model_health(model_name)
            loaded_models_status[model_name] = {
                'healthy': health['healthy'],
                'version': self._model_versions[model_name],
                'issues': health['issues']
            }
        
        overall_healthy = all(
            status['healthy'] for status in loaded_models_status.values()
        ) if loaded_models_status else False
        
        return {
            'service_healthy': overall_healthy,
            'loaded_models': loaded_models_status,
            'fallback_models_available': len(self._fallback_models),
            'fallback_models': self._fallback_models,
            'last_health_check': datetime.now().isoformat(),
            'models_base_path': str(self.models_base_path)
        }
    
    def reload_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Reload a model (useful for updates or recovery)
        
        Args:
            model_name: Name of the model to reload
            version: Specific version to load
        
        Returns:
            Dictionary with reload results
        """
        logger.info(f"Reloading model: {model_name}, version: {version}")
        
        # Clear from cache first
        old_version = self._model_versions.get(model_name, 'latest')
        old_cache_key = f"{model_name}_{old_version}"
        if old_cache_key in self._loaded_models:
            del self._loaded_models[old_cache_key]
        
        # Load fresh copy
        return self.load_primary_model(model_name, version)
    
    def _get_model_version(self, model_result: Dict[str, Any]) -> str:
        """
        Extract model version from load result
        
        Args:
            model_result: Result from model loading
        
        Returns:
            Model version string
        """
        if model_result.get('metadata'):
            return model_result['metadata'].model_version
        elif model_result.get('model_info'):
            return model_result['model_info'].get('version', 'unknown')
        return 'unknown'
    
    def _check_validation_results(self, validation_results: Optional[Dict[str, Any]]) -> bool:
        """
        Check if validation results indicate the model is performing well
        
        Args:
            validation_results: Validation results from model loading
        
        Returns:
            True if validation passed, False otherwise
        """
        if not validation_results:
            return True  # No validation performed, assume OK
        
        return (
            validation_results.get('validation_successful', False) and
            not validation_results.get('performance_degraded', True)
        )


# Convenience functions for easy access
def create_ml_service(models_path: str = "models") -> MLService:
    """
    Create and return an MLService instance
    
    Args:
        models_path: Path to models directory
    
    Returns:
        Configured MLService instance
    """
    return MLService(models_path)


def load_production_model(
    model_name: str, 
    version: Optional[str] = None,
    models_path: str = "models"
) -> Dict[str, Any]:
    """
    Convenience function to load a production model
    
    Args:
        model_name: Name of the model to load
        version: Specific version to load
        models_path: Path to models directory
    
    Returns:
        Dictionary with load results
    """
    service = MLService(models_path)
    return service.load_primary_model(model_name, version)