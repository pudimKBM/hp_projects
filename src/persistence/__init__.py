"""
Model Persistence Module

This module provides functionality for saving, loading, and versioning machine learning models
and their associated preprocessing pipelines.
"""

from .model_saving import ModelSaver, save_model, save_pipeline
from .model_loading import ModelLoader, load_model, load_pipeline
from .versioning import ModelVersioningSystem, ModelVersion, create_model_version, register_model_version
from .pipeline_persistence import PipelinePersistence, save_pipeline_with_versioning, load_pipeline_with_validation
from .metadata import ModelMetadata, create_model_metadata, compare_metadata

__all__ = [
    'ModelSaver',
    'ModelLoader', 
    'ModelVersioningSystem',
    'ModelVersion',
    'PipelinePersistence',
    'ModelMetadata',
    'save_model',
    'load_model',
    'save_pipeline',
    'save_pipeline_with_versioning',
    'load_pipeline',
    'load_pipeline_with_validation',
    'create_model_version',
    'register_model_version',
    'create_model_metadata',
    'compare_metadata'
]