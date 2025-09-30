"""
Model Saving Functionality

Implements joblib-based model serialization with metadata storage,
file naming conventions, and organization system.
"""

import os
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .metadata import ModelMetadata, create_model_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSaver:
    """
    Handles saving of trained models with comprehensive metadata
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize ModelSaver
        
        Args:
            base_path: Base directory for saving models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.base_path / "trained_models"
        self.metadata_dir = self.base_path / "metadata"
        self.pipelines_dir = self.base_path / "pipelines"
        
        for directory in [self.models_dir, self.metadata_dir, self.pipelines_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_model_filename(self, model_name: str, algorithm: str, version: str = None) -> str:
        """
        Generate standardized model filename
        
        Args:
            model_name: Name of the model
            algorithm: Algorithm used
            version: Model version (optional)
        
        Returns:
            Standardized filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if version:
            filename = f"{model_name}_{algorithm}_v{version}_{timestamp}.joblib"
        else:
            filename = f"{model_name}_{algorithm}_{timestamp}.joblib"
        
        return filename
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        algorithm: str,
        performance_metrics: Dict[str, float],
        feature_names: List[str],
        cv_scores: Dict[str, List[float]],
        preprocessing_steps: List[str],
        training_data_size: Optional[int] = None,
        class_distribution: Optional[Dict[str, int]] = None,
        version: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save model with comprehensive metadata
        
        Args:
            model: Trained model object
            model_name: Name identifier for the model
            algorithm: Algorithm used
            performance_metrics: Dictionary of performance metrics
            feature_names: List of feature names
            cv_scores: Cross-validation scores
            preprocessing_steps: List of preprocessing steps
            training_data_size: Size of training dataset
            class_distribution: Class distribution in training data
            version: Model version
            additional_metadata: Additional metadata to store
        
        Returns:
            Dictionary with file paths of saved components
        """
        try:
            # Generate filename
            model_filename = self._generate_model_filename(model_name, algorithm, version)
            model_path = self.models_dir / model_filename
            
            # Create metadata
            metadata = create_model_metadata(
                model_name=model_name,
                algorithm=algorithm,
                model=model,
                performance_metrics=performance_metrics,
                feature_names=feature_names,
                cv_scores=cv_scores,
                preprocessing_steps=preprocessing_steps,
                training_data_size=training_data_size,
                class_distribution=class_distribution
            )
            
            if version:
                metadata.model_version = version
            
            # Add additional metadata if provided
            if additional_metadata:
                for key, value in additional_metadata.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)
            
            # Validate metadata
            if not metadata.validate():
                raise ValueError("Invalid metadata - missing required fields")
            
            # Save model
            joblib.dump(model, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            # Save metadata
            metadata_filename = model_filename.replace('.joblib', '_metadata.json')
            metadata_path = self.metadata_dir / metadata_filename
            
            with open(metadata_path, 'w') as f:
                f.write(metadata.to_json())
            
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Create model registry entry
            registry_entry = {
                'model_name': model_name,
                'algorithm': algorithm,
                'version': metadata.model_version,
                'model_file': str(model_path),
                'metadata_file': str(metadata_path),
                'saved_date': metadata.training_date,
                'performance_summary': {
                    'accuracy': performance_metrics.get('accuracy', 0),
                    'f1_score': performance_metrics.get('f1_score', 0),
                    'roc_auc': performance_metrics.get('roc_auc', 0)
                }
            }
            
            self._update_model_registry(registry_entry)
            
            return {
                'model_path': str(model_path),
                'metadata_path': str(metadata_path),
                'model_name': model_name,
                'version': metadata.model_version
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def _update_model_registry(self, registry_entry: Dict[str, Any]):
        """
        Update the model registry with new model information
        
        Args:
            registry_entry: Dictionary with model information
        """
        registry_path = self.base_path / "model_registry.json"
        
        # Load existing registry or create new one
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'models': []}
        
        # Add new entry
        registry['models'].append(registry_entry)
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Model registry updated: {registry_path}")
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models with their metadata
        
        Returns:
            List of model information dictionaries
        """
        registry_path = self.base_path / "model_registry.json"
        
        if not registry_path.exists():
            return []
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        return registry.get('models', [])
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            version: Specific version (optional, returns latest if not specified)
        
        Returns:
            Model information dictionary or None if not found
        """
        models = self.list_saved_models()
        
        # Filter by model name
        matching_models = [m for m in models if m['model_name'] == model_name]
        
        if not matching_models:
            return None
        
        if version:
            # Find specific version
            for model in matching_models:
                if model['version'] == version:
                    return model
            return None
        else:
            # Return latest version (last saved)
            return matching_models[-1]


def save_model(
    model: Any,
    model_name: str,
    algorithm: str,
    performance_metrics: Dict[str, float],
    feature_names: List[str],
    cv_scores: Dict[str, List[float]],
    preprocessing_steps: List[str],
    base_path: str = "models",
    **kwargs
) -> Dict[str, str]:
    """
    Convenience function to save a model with metadata
    
    Args:
        model: Trained model object
        model_name: Name identifier for the model
        algorithm: Algorithm used
        performance_metrics: Dictionary of performance metrics
        feature_names: List of feature names
        cv_scores: Cross-validation scores
        preprocessing_steps: List of preprocessing steps
        base_path: Base directory for saving models
        **kwargs: Additional arguments passed to ModelSaver.save_model
    
    Returns:
        Dictionary with file paths of saved components
    """
    saver = ModelSaver(base_path)
    return saver.save_model(
        model=model,
        model_name=model_name,
        algorithm=algorithm,
        performance_metrics=performance_metrics,
        feature_names=feature_names,
        cv_scores=cv_scores,
        preprocessing_steps=preprocessing_steps,
        **kwargs
    )


def save_pipeline(
    pipeline: Any,
    pipeline_name: str,
    pipeline_steps: List[str],
    base_path: str = "models"
) -> str:
    """
    Save preprocessing pipeline
    
    Args:
        pipeline: Preprocessing pipeline object
        pipeline_name: Name for the pipeline
        pipeline_steps: List of pipeline steps
        base_path: Base directory for saving
    
    Returns:
        Path to saved pipeline file
    """
    saver = ModelSaver(base_path)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_filename = f"{pipeline_name}_pipeline_{timestamp}.joblib"
    pipeline_path = saver.pipelines_dir / pipeline_filename
    
    # Save pipeline
    joblib.dump(pipeline, pipeline_path)
    
    # Save pipeline metadata
    pipeline_metadata = {
        'pipeline_name': pipeline_name,
        'steps': pipeline_steps,
        'saved_date': datetime.now().isoformat(),
        'file_path': str(pipeline_path)
    }
    
    metadata_filename = pipeline_filename.replace('.joblib', '_metadata.json')
    metadata_path = saver.pipelines_dir / metadata_filename
    
    with open(metadata_path, 'w') as f:
        json.dump(pipeline_metadata, f, indent=2)
    
    logger.info(f"Pipeline saved to: {pipeline_path}")
    logger.info(f"Pipeline metadata saved to: {metadata_path}")
    
    return str(pipeline_path)