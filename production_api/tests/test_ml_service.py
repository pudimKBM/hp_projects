"""
Unit tests for ML Service

Tests model loading, validation, and fallback mechanisms.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from production_api.app.services.ml_service import MLService, create_ml_service, load_production_model


class TestMLService:
    """Test cases for MLService class"""
    
    def test_init(self):
        """Test MLService initialization"""
        service = MLService("test_models")
        
        assert service.models_base_path.name == "test_models"
        assert service._loaded_models == {}
        assert service._model_versions == {}
        assert service._fallback_models == []
        assert service._service_healthy is True
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_load_primary_model_success(self, mock_loader_class, mock_model_loader):
        """Test successful model loading"""
        # Setup mock
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_metadata = Mock()
        mock_metadata.model_version = "v1.0"
        mock_metadata.to_dict.return_value = {"version": "v1.0"}
        
        mock_loader.load_model.return_value = {
            'load_successful': True,
            'model': Mock(),
            'metadata': mock_metadata,
            'validation_results': {'validation_successful': True}
        }
        
        # Test
        service = MLService("test_models")
        result = service.load_primary_model("test_model", "v1.0")
        
        # Assertions
        assert result['success'] is True
        assert result['model_name'] == "test_model"
        assert result['version'] == "v1.0"
        assert result['validation_passed'] is True
        assert "test_model_v1.0" in service._loaded_models
        
        mock_loader.load_model.assert_called_once_with(
            model_name="test_model",
            version="v1.0",
            validate_performance=True
        )
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_load_primary_model_failure(self, mock_loader_class):
        """Test model loading failure"""
        # Setup mock
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_loader.load_model.return_value = {
            'load_successful': False,
            'error': 'Model file not found'
        }
        
        # Test
        service = MLService("test_models")
        result = service.load_primary_model("nonexistent_model")
        
        # Assertions
        assert result['success'] is False
        assert result['model_name'] == "nonexistent_model"
        assert result['error'] == 'Model file not found'
        assert result['fallback_available'] is False
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_setup_fallback_models(self, mock_loader_class):
        """Test fallback model setup"""
        # Setup mock
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_loader.check_model_integrity.side_effect = [
            {'integrity_ok': True, 'issues': []},
            {'integrity_ok': False, 'issues': ['Missing file']},
            {'integrity_ok': True, 'issues': []}
        ]
        
        # Test
        service = MLService("test_models")
        result = service.setup_fallback_models(['model1', 'model2', 'model3'])
        
        # Assertions
        assert len(result['successful_fallbacks']) == 2
        assert 'model1' in result['successful_fallbacks']
        assert 'model3' in result['successful_fallbacks']
        assert len(result['failed_fallbacks']) == 1
        assert result['failed_fallbacks'][0]['model_name'] == 'model2'
        assert service._fallback_models == ['model1', 'model3']
    
    def test_get_active_model(self, mock_model_loader):
        """Test getting active model"""
        service = MLService("test_models")
        
        # No model loaded
        result = service.get_active_model("test_model")
        assert result is None
        
        # Load a model
        mock_model_data = {'model': Mock(), 'metadata': Mock()}
        service._loaded_models["test_model_latest"] = mock_model_data
        service._model_versions["test_model"] = "latest"
        
        result = service.get_active_model("test_model")
        assert result == mock_model_data
    
    def test_validate_model_health_no_model(self, mock_model_loader):
        """Test model health validation when no model is loaded"""
        service = MLService("test_models")
        
        result = service.validate_model_health("nonexistent_model")
        
        assert result['healthy'] is False
        assert 'Model not loaded' in result['issues']
        assert 'last_check' in result
    
    def test_validate_model_health_success(self, mock_model_loader):
        """Test successful model health validation"""
        service = MLService("test_models")
        
        # Setup mock model
        mock_model = Mock()
        mock_model.predict = Mock()
        
        mock_metadata = Mock()
        mock_metadata.validate.return_value = True
        
        mock_model_data = {
            'model': mock_model,
            'metadata': mock_metadata,
            'validation_results': {'validation_successful': True}
        }
        
        service._loaded_models["test_model_latest"] = mock_model_data
        service._model_versions["test_model"] = "latest"
        
        result = service.validate_model_health("test_model")
        
        assert result['healthy'] is True
        assert len(result['issues']) == 0
    
    def test_validate_model_health_issues(self, mock_model_loader):
        """Test model health validation with issues"""
        service = MLService("test_models")
        
        # Setup mock model with issues
        mock_metadata = Mock()
        mock_metadata.validate.return_value = False
        
        mock_model_data = {
            'model': None,  # Missing model
            'metadata': mock_metadata,
            'validation_results': {'performance_degraded': True}
        }
        
        service._loaded_models["test_model_latest"] = mock_model_data
        service._model_versions["test_model"] = "latest"
        
        result = service.validate_model_health("test_model")
        
        assert result['healthy'] is False
        assert 'Model object is None' in result['issues']
        assert 'Model metadata validation failed' in result['issues']
        assert 'Model performance has degraded' in result['issues']
    
    def test_get_model_metadata(self, mock_model_loader):
        """Test getting model metadata"""
        service = MLService("test_models")
        
        # No model loaded
        result = service.get_model_metadata("test_model")
        assert result is None
        
        # Model with metadata
        mock_metadata = Mock()
        mock_metadata.to_dict.return_value = {"version": "v1.0", "accuracy": 0.95}
        
        mock_model_data = {'metadata': mock_metadata}
        service._loaded_models["test_model_latest"] = mock_model_data
        service._model_versions["test_model"] = "latest"
        
        result = service.get_model_metadata("test_model")
        assert result == {"version": "v1.0", "accuracy": 0.95}
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_list_available_models(self, mock_loader_class):
        """Test listing available models"""
        # Setup mock
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        expected_models = [
            {'name': 'model1', 'version': 'v1.0', 'type': 'RandomForest'},
            {'name': 'model2', 'version': 'v2.0', 'type': 'LogisticRegression'}
        ]
        mock_loader.list_available_models.return_value = expected_models
        
        # Test
        service = MLService("test_models")
        result = service.list_available_models()
        
        assert result == expected_models
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_list_available_models_error(self, mock_loader_class):
        """Test listing available models with error"""
        # Setup mock
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader.list_available_models.side_effect = Exception("Directory not found")
        
        # Test
        service = MLService("test_models")
        result = service.list_available_models()
        
        assert result == []
    
    def test_get_service_status(self, mock_model_loader):
        """Test getting service status"""
        service = MLService("test_models")
        
        # Setup some loaded models
        service._model_versions = {"model1": "v1.0", "model2": "v2.0"}
        service._fallback_models = ["fallback1", "fallback2"]
        
        # Mock health validation
        with patch.object(service, 'validate_model_health') as mock_health:
            mock_health.side_effect = [
                {'healthy': True, 'issues': []},
                {'healthy': False, 'issues': ['Some issue']}
            ]
            
            result = service.get_service_status()
            
            assert result['service_healthy'] is False  # One model unhealthy
            assert len(result['loaded_models']) == 2
            assert result['fallback_models_available'] == 2
            assert result['fallback_models'] == ["fallback1", "fallback2"]
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_reload_model(self, mock_loader_class):
        """Test model reloading"""
        # Setup mock
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        mock_metadata = Mock()
        mock_metadata.model_version = "v2.0"
        mock_metadata.to_dict.return_value = {"version": "v2.0"}
        
        mock_loader.load_model.return_value = {
            'load_successful': True,
            'model': Mock(),
            'metadata': mock_metadata,
            'validation_results': {'validation_successful': True}
        }
        
        # Test
        service = MLService("test_models")
        
        # Load initial model
        service._loaded_models["test_model_v1.0"] = {'old': 'data'}
        service._model_versions["test_model"] = "v1.0"
        
        result = service.reload_model("test_model", "v2.0")
        
        # Assertions
        assert result['success'] is True
        assert result['version'] == "v2.0"
        assert "test_model_v1.0" not in service._loaded_models  # Old version removed
        assert "test_model_v2.0" in service._loaded_models  # New version added
    
    def test_get_model_version(self, mock_model_loader):
        """Test extracting model version"""
        service = MLService("test_models")
        
        # Test with metadata
        mock_metadata = Mock()
        mock_metadata.model_version = "v1.5"
        result_with_metadata = {'metadata': mock_metadata}
        
        version = service._get_model_version(result_with_metadata)
        assert version == "v1.5"
        
        # Test with model_info
        result_with_info = {'model_info': {'version': 'v2.0'}}
        version = service._get_model_version(result_with_info)
        assert version == "v2.0"
        
        # Test with no version info
        result_empty = {}
        version = service._get_model_version(result_empty)
        assert version == "unknown"
    
    def test_check_validation_results(self, mock_model_loader):
        """Test validation results checking"""
        service = MLService("test_models")
        
        # No validation results
        assert service._check_validation_results(None) is True
        
        # Successful validation
        good_results = {
            'validation_successful': True,
            'performance_degraded': False
        }
        assert service._check_validation_results(good_results) is True
        
        # Failed validation
        bad_results = {
            'validation_successful': False,
            'performance_degraded': True
        }
        assert service._check_validation_results(bad_results) is False


class TestMLServiceConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_ml_service(self):
        """Test create_ml_service function"""
        service = create_ml_service("custom_path")
        
        assert isinstance(service, MLService)
        assert service.models_base_path.name == "custom_path"
    
    @patch('production_api.app.services.ml_service.MLService')
    def test_load_production_model(self, mock_service_class):
        """Test load_production_model function"""
        # Setup mock
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        expected_result = {'success': True, 'model_name': 'prod_model'}
        mock_service.load_primary_model.return_value = expected_result
        
        # Test
        result = load_production_model("prod_model", "v1.0", "models")
        
        # Assertions
        assert result == expected_result
        mock_service_class.assert_called_once_with("models")
        mock_service.load_primary_model.assert_called_once_with("prod_model", "v1.0")


class TestMLServiceIntegration:
    """Integration tests for MLService"""
    
    @patch('production_api.app.services.ml_service.ModelLoader')
    def test_full_workflow(self, mock_loader_class):
        """Test complete ML service workflow"""
        # Setup mock loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Mock successful model loading
        mock_metadata = Mock()
        mock_metadata.model_version = "v1.0"
        mock_metadata.to_dict.return_value = {"version": "v1.0"}
        mock_metadata.validate.return_value = True
        
        mock_model = Mock()
        mock_model.predict = Mock()
        
        mock_loader.load_model.return_value = {
            'load_successful': True,
            'model': mock_model,
            'metadata': mock_metadata,
            'validation_results': {'validation_successful': True}
        }
        
        # Mock fallback setup
        mock_loader.check_model_integrity.return_value = {
            'integrity_ok': True,
            'issues': []
        }
        
        # Mock available models
        mock_loader.list_available_models.return_value = [
            {'name': 'main_model', 'version': 'v1.0'},
            {'name': 'fallback_model', 'version': 'v1.0'}
        ]
        
        # Test workflow
        service = MLService("test_models")
        
        # 1. Load primary model
        load_result = service.load_primary_model("main_model", "v1.0")
        assert load_result['success'] is True
        
        # 2. Setup fallbacks
        fallback_result = service.setup_fallback_models(["fallback_model"])
        assert len(fallback_result['successful_fallbacks']) == 1
        
        # 3. Check health
        health_result = service.validate_model_health("main_model")
        assert health_result['healthy'] is True
        
        # 4. Get service status
        status = service.get_service_status()
        assert status['service_healthy'] is True
        assert status['fallback_models_available'] == 1
        
        # 5. List available models
        models = service.list_available_models()
        assert len(models) == 2