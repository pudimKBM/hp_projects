"""
Model Metadata Management

Handles creation, storage, and validation of model metadata including
performance metrics, training parameters, and data schema information.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ModelMetadata:
    """
    Comprehensive metadata for trained models
    """
    model_name: str
    algorithm: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    cross_validation_scores: Dict[str, List[float]]
    data_schema_version: str
    preprocessing_steps: List[str]
    model_version: str = "1.0.0"
    training_data_size: Optional[int] = None
    feature_count: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string"""
        # Handle numpy types in the data
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        data = self.to_dict()
        # Convert numpy types recursively
        data = json.loads(json.dumps(data, default=convert_numpy))
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Create metadata from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> bool:
        """Validate metadata completeness and consistency"""
        required_fields = [
            'model_name', 'algorithm', 'training_date',
            'performance_metrics', 'feature_names'
        ]
        
        for field in required_fields:
            if not getattr(self, field):
                return False
        
        # Validate performance metrics
        if not isinstance(self.performance_metrics, dict):
            return False
        
        # Validate feature names
        if not isinstance(self.feature_names, list) or len(self.feature_names) == 0:
            return False
        
        return True


def create_model_metadata(
    model_name: str,
    algorithm: str,
    model,
    performance_metrics: Dict[str, float],
    feature_names: List[str],
    cv_scores: Dict[str, List[float]],
    preprocessing_steps: List[str],
    training_data_size: Optional[int] = None,
    class_distribution: Optional[Dict[str, int]] = None
) -> ModelMetadata:
    """
    Create comprehensive model metadata
    
    Args:
        model_name: Name identifier for the model
        algorithm: Algorithm used (e.g., 'RandomForest', 'SVM')
        model: Trained model object
        performance_metrics: Dictionary of performance metrics
        feature_names: List of feature names used in training
        cv_scores: Cross-validation scores
        preprocessing_steps: List of preprocessing steps applied
        training_data_size: Size of training dataset
        class_distribution: Distribution of classes in training data
    
    Returns:
        ModelMetadata object with all relevant information
    """
    
    # Extract hyperparameters from model
    hyperparameters = {}
    if hasattr(model, 'get_params'):
        hyperparameters = model.get_params()
    
    # Create metadata
    metadata = ModelMetadata(
        model_name=model_name,
        algorithm=algorithm,
        training_date=datetime.now().isoformat(),
        performance_metrics=performance_metrics,
        feature_names=feature_names,
        hyperparameters=hyperparameters,
        cross_validation_scores=cv_scores,
        data_schema_version="1.0",
        preprocessing_steps=preprocessing_steps,
        training_data_size=training_data_size,
        feature_count=len(feature_names),
        class_distribution=class_distribution
    )
    
    return metadata


def compare_metadata(metadata1: ModelMetadata, metadata2: ModelMetadata) -> Dict[str, Any]:
    """
    Compare two model metadata objects
    
    Args:
        metadata1: First metadata object
        metadata2: Second metadata object
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'same_algorithm': metadata1.algorithm == metadata2.algorithm,
        'same_features': metadata1.feature_names == metadata2.feature_names,
        'same_preprocessing': metadata1.preprocessing_steps == metadata2.preprocessing_steps,
        'performance_diff': {},
        'feature_count_diff': metadata1.feature_count - metadata2.feature_count if metadata1.feature_count and metadata2.feature_count else None
    }
    
    # Compare performance metrics
    common_metrics = set(metadata1.performance_metrics.keys()) & set(metadata2.performance_metrics.keys())
    for metric in common_metrics:
        comparison['performance_diff'][metric] = (
            metadata1.performance_metrics[metric] - metadata2.performance_metrics[metric]
        )
    
    return comparison