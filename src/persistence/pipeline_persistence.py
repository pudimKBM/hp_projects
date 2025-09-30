"""
Preprocessing Pipeline Persistence

Handles saving and loading of feature engineering pipelines with versioning,
tracking, and compatibility validation.
"""

import os
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelinePersistence:
    """
    Handles comprehensive preprocessing pipeline persistence with versioning
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize PipelinePersistence
        
        Args:
            base_path: Base directory for saving pipelines
        """
        self.base_path = Path(base_path)
        self.pipelines_dir = self.base_path / "pipelines"
        self.pipeline_metadata_dir = self.base_path / "pipeline_metadata"
        
        # Create directories
        for directory in [self.pipelines_dir, self.pipeline_metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_pipeline_hash(self, pipeline_steps: List[str], feature_names: List[str]) -> str:
        """
        Generate hash for pipeline configuration
        
        Args:
            pipeline_steps: List of pipeline steps
            feature_names: List of feature names
        
        Returns:
            Hash string for the pipeline configuration
        """
        config_str = json.dumps({
            'steps': sorted(pipeline_steps),
            'features': sorted(feature_names)
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save_pipeline(
        self,
        pipeline: Any,
        pipeline_name: str,
        pipeline_steps: List[str],
        feature_names: List[str],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        version: Optional[str] = None,
        description: Optional[str] = None,
        dependencies: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Save preprocessing pipeline with comprehensive metadata
        
        Args:
            pipeline: Preprocessing pipeline object
            pipeline_name: Name for the pipeline
            pipeline_steps: List of pipeline steps
            feature_names: List of feature names produced
            input_schema: Schema of input data
            output_schema: Schema of output data
            version: Pipeline version
            description: Description of the pipeline
            dependencies: Dictionary of package dependencies
        
        Returns:
            Dictionary with file paths of saved components
        """
        try:
            # Generate version if not provided
            if not version:
                pipeline_hash = self._generate_pipeline_hash(pipeline_steps, feature_names)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                version = f"v1.0.0_{pipeline_hash}_{timestamp}"
            
            # Generate filename
            pipeline_filename = f"{pipeline_name}_pipeline_{version}.joblib"
            pipeline_path = self.pipelines_dir / pipeline_filename
            
            # Save pipeline
            joblib.dump(pipeline, pipeline_path)
            logger.info(f"Pipeline saved to: {pipeline_path}")
            
            # Create comprehensive metadata
            pipeline_metadata = {
                'pipeline_name': pipeline_name,
                'version': version,
                'steps': pipeline_steps,
                'feature_names': feature_names,
                'input_schema': input_schema,
                'output_schema': output_schema,
                'description': description or f"Preprocessing pipeline for {pipeline_name}",
                'created_date': datetime.now().isoformat(),
                'file_path': str(pipeline_path),
                'dependencies': dependencies or {},
                'pipeline_hash': self._generate_pipeline_hash(pipeline_steps, feature_names),
                'feature_count': len(feature_names),
                'step_count': len(pipeline_steps)
            }
            
            # Save metadata
            metadata_filename = f"{pipeline_name}_pipeline_{version}_metadata.json"
            metadata_path = self.pipeline_metadata_dir / metadata_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(pipeline_metadata, f, indent=2)
            
            logger.info(f"Pipeline metadata saved to: {metadata_path}")
            
            # Update pipeline registry
            self._update_pipeline_registry(pipeline_metadata)
            
            return {
                'pipeline_path': str(pipeline_path),
                'metadata_path': str(metadata_path),
                'pipeline_name': pipeline_name,
                'version': version
            }
            
        except Exception as e:
            logger.error(f"Error saving pipeline: {str(e)}")
            raise
    
    def load_pipeline(
        self,
        pipeline_name: str,
        version: Optional[str] = None,
        validate_compatibility: bool = True,
        expected_input_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load preprocessing pipeline with validation
        
        Args:
            pipeline_name: Name of the pipeline to load
            version: Specific version to load (loads latest if None)
            validate_compatibility: Whether to validate schema compatibility
            expected_input_schema: Expected input schema for validation
        
        Returns:
            Dictionary containing pipeline, metadata, and validation results
        """
        try:
            # Get pipeline information
            pipeline_info = self.get_pipeline_info(pipeline_name, version)
            
            if not pipeline_info:
                raise FileNotFoundError(f"Pipeline '{pipeline_name}' version '{version}' not found")
            
            # Load pipeline
            pipeline_path = Path(pipeline_info['file_path'])
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            
            pipeline = joblib.load(pipeline_path)
            logger.info(f"Pipeline loaded from: {pipeline_path}")
            
            result = {
                'pipeline': pipeline,
                'metadata': pipeline_info,
                'load_successful': True,
                'compatibility_check': None
            }
            
            # Validate compatibility if requested
            if validate_compatibility and expected_input_schema:
                compatibility_check = self.validate_pipeline_compatibility(
                    pipeline_info, expected_input_schema
                )
                result['compatibility_check'] = compatibility_check
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading pipeline '{pipeline_name}': {str(e)}")
            return {
                'pipeline': None,
                'metadata': None,
                'load_successful': False,
                'error': str(e),
                'compatibility_check': None
            }
    
    def validate_pipeline_compatibility(
        self,
        pipeline_metadata: Dict[str, Any],
        expected_input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate pipeline compatibility with expected input schema
        
        Args:
            pipeline_metadata: Pipeline metadata
            expected_input_schema: Expected input data schema
        
        Returns:
            Dictionary with compatibility validation results
        """
        try:
            stored_input_schema = pipeline_metadata.get('input_schema', {})
            
            compatibility_results = {
                'compatible': True,
                'issues': [],
                'warnings': []
            }
            
            # Check feature names
            expected_features = set(expected_input_schema.get('feature_names', []))
            stored_features = set(stored_input_schema.get('feature_names', []))
            
            missing_features = stored_features - expected_features
            extra_features = expected_features - stored_features
            
            if missing_features:
                compatibility_results['compatible'] = False
                compatibility_results['issues'].append(
                    f"Missing required features: {list(missing_features)}"
                )
            
            if extra_features:
                compatibility_results['warnings'].append(
                    f"Extra features in input (will be ignored): {list(extra_features)}"
                )
            
            # Check data types
            expected_dtypes = expected_input_schema.get('dtypes', {})
            stored_dtypes = stored_input_schema.get('dtypes', {})
            
            for feature in stored_features & expected_features:
                if feature in expected_dtypes and feature in stored_dtypes:
                    if expected_dtypes[feature] != stored_dtypes[feature]:
                        compatibility_results['warnings'].append(
                            f"Data type mismatch for '{feature}': expected {expected_dtypes[feature]}, "
                            f"pipeline expects {stored_dtypes[feature]}"
                        )
            
            return compatibility_results
            
        except Exception as e:
            logger.error(f"Error validating pipeline compatibility: {str(e)}")
            return {
                'compatible': False,
                'issues': [f"Compatibility validation error: {str(e)}"],
                'warnings': []
            }
    
    def _update_pipeline_registry(self, pipeline_metadata: Dict[str, Any]):
        """
        Update the pipeline registry with new pipeline information
        
        Args:
            pipeline_metadata: Dictionary with pipeline information
        """
        registry_path = self.base_path / "pipeline_registry.json"
        
        # Load existing registry or create new one
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'pipelines': []}
        
        # Add new entry
        registry['pipelines'].append(pipeline_metadata)
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Pipeline registry updated: {registry_path}")
    
    def get_pipeline_info(self, pipeline_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific pipeline
        
        Args:
            pipeline_name: Name of the pipeline
            version: Specific version (optional, returns latest if not specified)
        
        Returns:
            Pipeline information dictionary or None if not found
        """
        registry_path = self.base_path / "pipeline_registry.json"
        
        if not registry_path.exists():
            return None
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        pipelines = registry.get('pipelines', [])
        
        # Filter by pipeline name
        matching_pipelines = [p for p in pipelines if p['pipeline_name'] == pipeline_name]
        
        if not matching_pipelines:
            return None
        
        if version:
            # Find specific version
            for pipeline in matching_pipelines:
                if pipeline['version'] == version:
                    return pipeline
            return None
        else:
            # Return latest version (last saved)
            return matching_pipelines[-1]
    
    def list_pipeline_versions(self, pipeline_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a specific pipeline
        
        Args:
            pipeline_name: Name of the pipeline
        
        Returns:
            List of pipeline version information
        """
        registry_path = self.base_path / "pipeline_registry.json"
        
        if not registry_path.exists():
            return []
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        pipelines = registry.get('pipelines', [])
        
        # Filter by pipeline name and sort by creation date
        matching_pipelines = [p for p in pipelines if p['pipeline_name'] == pipeline_name]
        matching_pipelines.sort(key=lambda x: x['created_date'])
        
        return matching_pipelines
    
    def compare_pipeline_versions(
        self,
        pipeline_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a pipeline
        
        Args:
            pipeline_name: Name of the pipeline
            version1: First version to compare
            version2: Second version to compare
        
        Returns:
            Dictionary with comparison results
        """
        pipeline1 = self.get_pipeline_info(pipeline_name, version1)
        pipeline2 = self.get_pipeline_info(pipeline_name, version2)
        
        if not pipeline1 or not pipeline2:
            return {
                'comparison_successful': False,
                'error': 'One or both pipeline versions not found'
            }
        
        comparison = {
            'comparison_successful': True,
            'pipeline_name': pipeline_name,
            'version1': version1,
            'version2': version2,
            'differences': {
                'steps_changed': pipeline1['steps'] != pipeline2['steps'],
                'features_changed': pipeline1['feature_names'] != pipeline2['feature_names'],
                'schema_changed': pipeline1['input_schema'] != pipeline2['input_schema'],
                'step_count_diff': pipeline1['step_count'] - pipeline2['step_count'],
                'feature_count_diff': pipeline1['feature_count'] - pipeline2['feature_count']
            },
            'details': {
                'version1_steps': pipeline1['steps'],
                'version2_steps': pipeline2['steps'],
                'version1_features': pipeline1['feature_names'],
                'version2_features': pipeline2['feature_names']
            }
        }
        
        return comparison
    
    def delete_pipeline_version(self, pipeline_name: str, version: str) -> bool:
        """
        Delete a specific pipeline version
        
        Args:
            pipeline_name: Name of the pipeline
            version: Version to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            pipeline_info = self.get_pipeline_info(pipeline_name, version)
            
            if not pipeline_info:
                logger.warning(f"Pipeline '{pipeline_name}' version '{version}' not found")
                return False
            
            # Delete pipeline file
            pipeline_path = Path(pipeline_info['file_path'])
            if pipeline_path.exists():
                pipeline_path.unlink()
                logger.info(f"Deleted pipeline file: {pipeline_path}")
            
            # Delete metadata file
            metadata_filename = f"{pipeline_name}_pipeline_{version}_metadata.json"
            metadata_path = self.pipeline_metadata_dir / metadata_filename
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted metadata file: {metadata_path}")
            
            # Update registry
            registry_path = self.base_path / "pipeline_registry.json"
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                
                # Remove the pipeline from registry
                registry['pipelines'] = [
                    p for p in registry['pipelines']
                    if not (p['pipeline_name'] == pipeline_name and p['version'] == version)
                ]
                
                with open(registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
                
                logger.info(f"Removed pipeline from registry: {pipeline_name} v{version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting pipeline version: {str(e)}")
            return False


def save_pipeline_with_versioning(
    pipeline: Any,
    pipeline_name: str,
    pipeline_steps: List[str],
    feature_names: List[str],
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    base_path: str = "models",
    **kwargs
) -> Dict[str, str]:
    """
    Convenience function to save a pipeline with versioning
    
    Args:
        pipeline: Preprocessing pipeline object
        pipeline_name: Name for the pipeline
        pipeline_steps: List of pipeline steps
        feature_names: List of feature names produced
        input_schema: Schema of input data
        output_schema: Schema of output data
        base_path: Base directory for saving
        **kwargs: Additional arguments passed to save_pipeline
    
    Returns:
        Dictionary with file paths of saved components
    """
    persistence = PipelinePersistence(base_path)
    return persistence.save_pipeline(
        pipeline=pipeline,
        pipeline_name=pipeline_name,
        pipeline_steps=pipeline_steps,
        feature_names=feature_names,
        input_schema=input_schema,
        output_schema=output_schema,
        **kwargs
    )


def load_pipeline_with_validation(
    pipeline_name: str,
    version: Optional[str] = None,
    base_path: str = "models",
    expected_input_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to load a pipeline with validation
    
    Args:
        pipeline_name: Name of the pipeline to load
        version: Specific version to load
        base_path: Base directory where pipelines are stored
        expected_input_schema: Expected input schema for validation
    
    Returns:
        Dictionary containing pipeline, metadata, and validation results
    """
    persistence = PipelinePersistence(base_path)
    return persistence.load_pipeline(
        pipeline_name=pipeline_name,
        version=version,
        validate_compatibility=expected_input_schema is not None,
        expected_input_schema=expected_input_schema
    )