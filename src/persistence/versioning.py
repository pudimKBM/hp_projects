"""
Model Versioning System

Implements model version tracking, comparison, performance history,
and rollback functionality for comprehensive model lifecycle management.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import semver

from .model_saving import ModelSaver
from .model_loading import ModelLoader
from .metadata import ModelMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """
    Represents a specific version of a model
    """
    model_name: str
    version: str
    algorithm: str
    created_date: str
    performance_metrics: Dict[str, float]
    model_path: str
    metadata_path: str
    is_active: bool = False
    is_production: bool = False
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelVersioningSystem:
    """
    Comprehensive model versioning and lifecycle management system
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize ModelVersioningSystem
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.versions_dir = self.base_path / "versions"
        self.history_dir = self.base_path / "history"
        
        # Create directories
        for directory in [self.versions_dir, self.history_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.saver = ModelSaver(str(self.base_path))
        self.loader = ModelLoader(str(self.base_path))
    
    def create_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
        auto_increment: bool = True
    ) -> str:
        """
        Create a new version for a model
        
        Args:
            model_name: Name of the model
            version: Specific version string (auto-generated if None)
            parent_version: Parent version for tracking lineage
            auto_increment: Whether to auto-increment version numbers
        
        Returns:
            Version string for the new version
        """  
      try:
            if not version:
                if auto_increment:
                    version = self._generate_next_version(model_name, parent_version)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    version = f"1.0.0_{timestamp}"
            
            # Validate version format
            if not self._validate_version_format(version):
                raise ValueError(f"Invalid version format: {version}")
            
            # Check if version already exists
            if self._version_exists(model_name, version):
                raise ValueError(f"Version {version} already exists for model {model_name}")
            
            logger.info(f"Created new version {version} for model {model_name}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating version: {str(e)}")
            raise
    
    def _generate_next_version(self, model_name: str, parent_version: Optional[str] = None) -> str:
        """
        Generate the next version number for a model
        
        Args:
            model_name: Name of the model
            parent_version: Parent version for incremental versioning
        
        Returns:
            Next version string
        """
        versions = self.list_model_versions(model_name)
        
        if not versions:
            return "1.0.0"
        
        if parent_version:
            # Increment from parent version
            try:
                return semver.bump_patch(parent_version)
            except ValueError:
                # Fallback if parent version is not semver compatible
                return f"{parent_version}.1"
        else:
            # Find latest version and increment
            latest_version = versions[-1].version
            try:
                return semver.bump_patch(latest_version)
            except ValueError:
                # Fallback for non-semver versions
                return f"{latest_version}.1"
    
    def _validate_version_format(self, version: str) -> bool:
        """
        Validate version string format
        
        Args:
            version: Version string to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try semver validation first
            semver.VersionInfo.parse(version)
            return True
        except ValueError:
            # Allow custom version formats with basic validation
            return len(version) > 0 and not version.isspace()
    
    def _version_exists(self, model_name: str, version: str) -> bool:
        """
        Check if a version already exists for a model
        
        Args:
            model_name: Name of the model
            version: Version to check
        
        Returns:
            True if version exists, False otherwise
        """
        versions = self.list_model_versions(model_name)
        return any(v.version == version for v in versions)
    
    def register_model_version(
        self,
        model_name: str,
        version: str,
        algorithm: str,
        performance_metrics: Dict[str, float],
        model_path: str,
        metadata_path: str,
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Register a new model version in the versioning system
        
        Args:
            model_name: Name of the model
            version: Version string
            algorithm: Algorithm used
            performance_metrics: Performance metrics
            model_path: Path to model file
            metadata_path: Path to metadata file
            parent_version: Parent version for lineage tracking
            tags: Tags for the version
        
        Returns:
            ModelVersion object
        """  
      try:
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                algorithm=algorithm,
                created_date=datetime.now().isoformat(),
                performance_metrics=performance_metrics,
                model_path=model_path,
                metadata_path=metadata_path,
                parent_version=parent_version,
                tags=tags or []
            )
            
            # Save version information
            self._save_version_info(model_version)
            
            # Update version history
            self._update_version_history(model_version)
            
            logger.info(f"Registered model version: {model_name} v{version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model version: {str(e)}")
            raise
    
    def _save_version_info(self, model_version: ModelVersion):
        """
        Save version information to file
        
        Args:
            model_version: ModelVersion object to save
        """
        version_file = self.versions_dir / f"{model_version.model_name}_v{model_version.version}.json"
        
        with open(version_file, 'w') as f:
            json.dump(asdict(model_version), f, indent=2)
    
    def _update_version_history(self, model_version: ModelVersion):
        """
        Update the version history for a model
        
        Args:
            model_version: ModelVersion object to add to history
        """
        history_file = self.history_dir / f"{model_version.model_name}_history.json"
        
        # Load existing history or create new
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {
                'model_name': model_version.model_name,
                'versions': [],
                'performance_timeline': []
            }
        
        # Add new version
        version_entry = asdict(model_version)
        history['versions'].append(version_entry)
        
        # Add performance timeline entry
        performance_entry = {
            'version': model_version.version,
            'date': model_version.created_date,
            'metrics': model_version.performance_metrics
        }
        history['performance_timeline'].append(performance_entry)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """
        List all versions of a specific model
        
        Args:
            model_name: Name of the model
        
        Returns:
            List of ModelVersion objects sorted by creation date
        """
        history_file = self.history_dir / f"{model_name}_history.json"
        
        if not history_file.exists():
            return []
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        versions = []
        for version_data in history.get('versions', []):
            versions.append(ModelVersion(**version_data))
        
        # Sort by creation date
        versions.sort(key=lambda v: v.created_date)
        return versions
    
    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """
        Get specific model version
        
        Args:
            model_name: Name of the model
            version: Version to retrieve
        
        Returns:
            ModelVersion object or None if not found
        """
        versions = self.list_model_versions(model_name)
        
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a model
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
        
        Returns:
            Dictionary with detailed comparison results
        """   
     v1 = self.get_model_version(model_name, version1)
        v2 = self.get_model_version(model_name, version2)
        
        if not v1 or not v2:
            return {
                'comparison_successful': False,
                'error': 'One or both versions not found'
            }
        
        # Calculate performance differences
        performance_diff = {}
        common_metrics = set(v1.performance_metrics.keys()) & set(v2.performance_metrics.keys())
        
        for metric in common_metrics:
            diff = v2.performance_metrics[metric] - v1.performance_metrics[metric]
            performance_diff[metric] = {
                'version1_value': v1.performance_metrics[metric],
                'version2_value': v2.performance_metrics[metric],
                'difference': diff,
                'improvement': diff > 0
            }
        
        comparison = {
            'comparison_successful': True,
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'algorithm_changed': v1.algorithm != v2.algorithm,
            'performance_differences': performance_diff,
            'time_difference': (
                datetime.fromisoformat(v2.created_date) - 
                datetime.fromisoformat(v1.created_date)
            ).total_seconds(),
            'version1_info': asdict(v1),
            'version2_info': asdict(v2)
        }
        
        return comparison
    
    def get_performance_history(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance history for a model across all versions
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with performance timeline and statistics
        """
        history_file = self.history_dir / f"{model_name}_history.json"
        
        if not history_file.exists():
            return {
                'model_name': model_name,
                'timeline': [],
                'statistics': {}
            }
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        timeline = history.get('performance_timeline', [])
        
        # Calculate statistics
        statistics = {}
        if timeline:
            # Get all metrics
            all_metrics = set()
            for entry in timeline:
                all_metrics.update(entry['metrics'].keys())
            
            for metric in all_metrics:
                values = [entry['metrics'].get(metric, 0) for entry in timeline if metric in entry['metrics']]
                if values:
                    statistics[metric] = {
                        'best': max(values),
                        'worst': min(values),
                        'latest': values[-1],
                        'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'declining',
                        'improvement_rate': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
                    }
        
        return {
            'model_name': model_name,
            'timeline': timeline,
            'statistics': statistics
        }
    
    def set_active_version(self, model_name: str, version: str) -> bool:
        """
        Set a specific version as the active version
        
        Args:
            model_name: Name of the model
            version: Version to set as active
        
        Returns:
            True if successful, False otherwise
        """
        try:
            versions = self.list_model_versions(model_name)
            
            # Update all versions to inactive
            for v in versions:
                v.is_active = False
                self._save_version_info(v)
            
            # Set specified version as active
            target_version = self.get_model_version(model_name, version)
            if target_version:
                target_version.is_active = True
                self._save_version_info(target_version)
                
                # Update history
                self._update_version_history(target_version)
                
                logger.info(f"Set {model_name} v{version} as active version")
                return True
            else:
                logger.error(f"Version {version} not found for model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting active version: {str(e)}")
            return False
    
    def set_production_version(self, model_name: str, version: str) -> bool:
        """
        Set a specific version as the production version
        
        Args:
            model_name: Name of the model
            version: Version to set as production
        
        Returns:
            True if successful, False otherwise
        """     
   try:
            versions = self.list_model_versions(model_name)
            
            # Update all versions to non-production
            for v in versions:
                v.is_production = False
                self._save_version_info(v)
            
            # Set specified version as production
            target_version = self.get_model_version(model_name, version)
            if target_version:
                target_version.is_production = True
                self._save_version_info(target_version)
                
                # Update history
                self._update_version_history(target_version)
                
                logger.info(f"Set {model_name} v{version} as production version")
                return True
            else:
                logger.error(f"Version {version} not found for model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting production version: {str(e)}")
            return False
    
    def rollback_to_version(self, model_name: str, target_version: str) -> Dict[str, Any]:
        """
        Rollback to a previous version of the model
        
        Args:
            model_name: Name of the model
            target_version: Version to rollback to
        
        Returns:
            Dictionary with rollback results
        """
        try:
            # Get current active version
            current_version = self.get_active_version(model_name)
            
            # Get target version
            target = self.get_model_version(model_name, target_version)
            
            if not target:
                return {
                    'rollback_successful': False,
                    'error': f"Target version {target_version} not found"
                }
            
            # Set target version as active
            success = self.set_active_version(model_name, target_version)
            
            if success:
                rollback_info = {
                    'rollback_successful': True,
                    'model_name': model_name,
                    'previous_version': current_version.version if current_version else None,
                    'new_active_version': target_version,
                    'rollback_date': datetime.now().isoformat()
                }
                
                # Log rollback event
                self._log_rollback_event(rollback_info)
                
                logger.info(f"Successfully rolled back {model_name} to version {target_version}")
                return rollback_info
            else:
                return {
                    'rollback_successful': False,
                    'error': 'Failed to set target version as active'
                }
                
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")
            return {
                'rollback_successful': False,
                'error': str(e)
            }
    
    def _log_rollback_event(self, rollback_info: Dict[str, Any]):
        """
        Log rollback event for audit trail
        
        Args:
            rollback_info: Information about the rollback
        """
        rollback_log_file = self.history_dir / "rollback_log.json"
        
        # Load existing log or create new
        if rollback_log_file.exists():
            with open(rollback_log_file, 'r') as f:
                log = json.load(f)
        else:
            log = {'rollback_events': []}
        
        # Add new rollback event
        log['rollback_events'].append(rollback_info)
        
        # Save updated log
        with open(rollback_log_file, 'w') as f:
            json.dump(log, f, indent=2)
    
    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get the currently active version of a model
        
        Args:
            model_name: Name of the model
        
        Returns:
            ModelVersion object or None if no active version
        """
        versions = self.list_model_versions(model_name)
        
        for version in versions:
            if version.is_active:
                return version
        
        return None
    
    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get the currently production version of a model
        
        Args:
            model_name: Name of the model
        
        Returns:
            ModelVersion object or None if no production version
        """
        versions = self.list_model_versions(model_name)
        
        for version in versions:
            if version.is_production:
                return version
        
        return None
    
    def add_version_tag(self, model_name: str, version: str, tag: str) -> bool:
        """
        Add a tag to a specific version
        
        Args:
            model_name: Name of the model
            version: Version to tag
            tag: Tag to add
        
        Returns:
            True if successful, False otherwise
        """ 
       try:
            model_version = self.get_model_version(model_name, version)
            
            if not model_version:
                return False
            
            if tag not in model_version.tags:
                model_version.tags.append(tag)
                self._save_version_info(model_version)
                self._update_version_history(model_version)
                
                logger.info(f"Added tag '{tag}' to {model_name} v{version}")
                return True
            
            return True  # Tag already exists
            
        except Exception as e:
            logger.error(f"Error adding tag: {str(e)}")
            return False
    
    def remove_version_tag(self, model_name: str, version: str, tag: str) -> bool:
        """
        Remove a tag from a specific version
        
        Args:
            model_name: Name of the model
            version: Version to remove tag from
            tag: Tag to remove
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_version = self.get_model_version(model_name, version)
            
            if not model_version:
                return False
            
            if tag in model_version.tags:
                model_version.tags.remove(tag)
                self._save_version_info(model_version)
                self._update_version_history(model_version)
                
                logger.info(f"Removed tag '{tag}' from {model_name} v{version}")
                return True
            
            return True  # Tag doesn't exist
            
        except Exception as e:
            logger.error(f"Error removing tag: {str(e)}")
            return False
    
    def find_versions_by_tag(self, model_name: str, tag: str) -> List[ModelVersion]:
        """
        Find all versions with a specific tag
        
        Args:
            model_name: Name of the model
            tag: Tag to search for
        
        Returns:
            List of ModelVersion objects with the specified tag
        """
        versions = self.list_model_versions(model_name)
        return [v for v in versions if tag in v.tags]
    
    def delete_version(self, model_name: str, version: str, force: bool = False) -> bool:
        """
        Delete a specific version (with safety checks)
        
        Args:
            model_name: Name of the model
            version: Version to delete
            force: Force deletion even if version is active/production
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_version = self.get_model_version(model_name, version)
            
            if not model_version:
                logger.warning(f"Version {version} not found for model {model_name}")
                return False
            
            # Safety checks
            if not force:
                if model_version.is_active:
                    logger.error(f"Cannot delete active version {version}. Use force=True to override.")
                    return False
                
                if model_version.is_production:
                    logger.error(f"Cannot delete production version {version}. Use force=True to override.")
                    return False
            
            # Delete model files
            model_path = Path(model_version.model_path)
            if model_path.exists():
                model_path.unlink()
            
            metadata_path = Path(model_version.metadata_path)
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Delete version info file
            version_file = self.versions_dir / f"{model_name}_v{version}.json"
            if version_file.exists():
                version_file.unlink()
            
            # Update history (remove from versions list)
            history_file = self.history_dir / f"{model_name}_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Remove version from history
                history['versions'] = [
                    v for v in history['versions'] if v['version'] != version
                ]
                history['performance_timeline'] = [
                    p for p in history['performance_timeline'] if p['version'] != version
                ]
                
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
            
            logger.info(f"Deleted version {version} of model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting version: {str(e)}")
            return False


# Convenience functions
def create_model_version(model_name: str, base_path: str = "models", **kwargs) -> str:
    """
    Convenience function to create a new model version
    """
    versioning = ModelVersioningSystem(base_path)
    return versioning.create_version(model_name, **kwargs)


def register_model_version(
    model_name: str,
    version: str,
    algorithm: str,
    performance_metrics: Dict[str, float],
    model_path: str,
    metadata_path: str,
    base_path: str = "models",
    **kwargs
) -> ModelVersion:
    """
    Convenience function to register a model version
    """
    versioning = ModelVersioningSystem(base_path)
    return versioning.register_model_version(
        model_name, version, algorithm, performance_metrics,
        model_path, metadata_path, **kwargs
    )