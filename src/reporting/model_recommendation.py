"""
Model Recommendation System Module

Provides automated model selection based on performance metrics,
business constraints, and deployment readiness assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BusinessConstraints:
    """
    Business constraints and preferences for model selection
    """
    primary_metric: str = 'f1_score'
    min_accuracy: float = 0.8
    min_precision: float = 0.7
    min_recall: float = 0.7
    interpretability_weight: float = 0.1
    speed_weight: float = 0.1
    stability_weight: float = 0.2
    max_training_time: Optional[float] = None
    max_prediction_time: Optional[float] = None
    require_probabilities: bool = False
    
    def validate(self) -> bool:
        """Validate constraint values"""
        return (0 <= self.min_accuracy <= 1 and 
                0 <= self.min_precision <= 1 and 
                0 <= self.min_recall <= 1 and
                0 <= self.interpretability_weight <= 1 and
                0 <= self.speed_weight <= 1 and
                0 <= self.stability_weight <= 1)


@dataclass
class ModelCharacteristics:
    """
    Model characteristics for recommendation scoring
    """
    interpretability_score: float  # 0-1 scale
    speed_score: float  # 0-1 scale (higher = faster)
    stability_score: float  # 0-1 scale (based on CV variance)
    complexity_score: float  # 0-1 scale (higher = more complex)
    memory_usage_score: float  # 0-1 scale (higher = more memory efficient)

de
f get_model_characteristics(model_name: str, 
                            cv_results: Optional[Dict[str, Any]] = None,
                            training_time: Optional[float] = None) -> ModelCharacteristics:
    """
    Get model characteristics based on algorithm type and performance data.
    
    Args:
        model_name: Name of the model
        cv_results: Cross-validation results for stability assessment
        training_time: Training time in seconds
        
    Returns:
        ModelCharacteristics object with scored characteristics
    """
    model_lower = model_name.lower()
    
    # Default characteristics based on model type
    characteristics = {
        'interpretability_score': 0.5,
        'speed_score': 0.5,
        'stability_score': 0.5,
        'complexity_score': 0.5,
        'memory_usage_score': 0.5
    }
    
    # Interpretability scoring
    if any(keyword in model_lower for keyword in ['logistic', 'linear']):
        characteristics['interpretability_score'] = 0.9
    elif any(keyword in model_lower for keyword in ['tree', 'forest']):
        characteristics['interpretability_score'] = 0.7
    elif any(keyword in model_lower for keyword in ['svm', 'support']):
        characteristics['interpretability_score'] = 0.3
    elif any(keyword in model_lower for keyword in ['neural', 'mlp', 'deep']):
        characteristics['interpretability_score'] = 0.1
    
    # Speed scoring (inverse of typical training/prediction time)
    if any(keyword in model_lower for keyword in ['logistic', 'linear', 'naive']):
        characteristics['speed_score'] = 0.9
    elif any(keyword in model_lower for keyword in ['tree', 'forest']):
        characteristics['speed_score'] = 0.6
    elif any(keyword in model_lower for keyword in ['svm', 'support']):
        characteristics['speed_score'] = 0.4
    elif any(keyword in model_lower for keyword in ['neural', 'mlp', 'boost']):
        characteristics['speed_score'] = 0.3
    
    # Complexity scoring
    if any(keyword in model_lower for keyword in ['logistic', 'linear', 'naive']):
        characteristics['complexity_score'] = 0.2
    elif any(keyword in model_lower for keyword in ['tree']):
        characteristics['complexity_score'] = 0.4
    elif any(keyword in model_lower for keyword in ['forest', 'boost']):
        characteristics['complexity_score'] = 0.7
    elif any(keyword in model_lower for keyword in ['svm', 'neural']):
        characteristics['complexity_score'] = 0.8
    
    # Memory usage scoring (inverse of typical memory requirements)
    if any(keyword in model_lower for keyword in ['logistic', 'linear', 'naive']):
        characteristics['memory_usage_score'] = 0.9
    elif any(keyword in model_lower for keyword in ['tree', 'svm']):
        characteristics['memory_usage_score'] = 0.7
    elif any(keyword in model_lower for keyword in ['forest', 'boost']):
        characteristics['memory_usage_score'] = 0.5
    elif any(keyword in model_lower for keyword in ['neural', 'mlp']):
        characteristics['memory_usage_score'] = 0.3
    
    # Stability scoring based on CV results
    if cv_results and 'metrics' in cv_results:
        cv_stds = []
        for metric_name, metric_data in cv_results['metrics'].items():
            if isinstance(metric_data, dict) and 'test_std' in metric_data:
                cv_stds.append(metric_data['test_std'])
        
        if cv_stds:
            avg_std = np.mean(cv_stds)
            # Convert std to stability score (lower std = higher stability)
            characteristics['stability_score'] = max(0.0, 1.0 - (avg_std * 10))
    
    return ModelCharacteristics(**characteristics)
def a
ssess_deployment_readiness(performance_metrics: Dict[str, float],
                              model_characteristics: ModelCharacteristics,
                              business_constraints: BusinessConstraints,
                              cv_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Assess model readiness for deployment based on multiple criteria.
    
    Args:
        performance_metrics: Dictionary of performance metrics
        model_characteristics: Model characteristics
        business_constraints: Business constraints
        cv_results: Cross-validation results for additional analysis
        
    Returns:
        Dictionary with deployment readiness assessment
    """
    assessment = {
        'ready_for_deployment': False,
        'readiness_score': 0.0,
        'passed_checks': [],
        'failed_checks': [],
        'warnings': [],
        'recommendations': []
    }
    
    checks_passed = 0
    total_checks = 0
    
    # Performance checks
    total_checks += 4
    
    if performance_metrics.get('accuracy', 0) >= business_constraints.min_accuracy:
        checks_passed += 1
        assessment['passed_checks'].append(f"Accuracy ≥ {business_constraints.min_accuracy}")
    else:
        assessment['failed_checks'].append(f"Accuracy below threshold ({performance_metrics.get('accuracy', 0):.3f} < {business_constraints.min_accuracy})")
    
    if performance_metrics.get('precision', 0) >= business_constraints.min_precision:
        checks_passed += 1
        assessment['passed_checks'].append(f"Precision ≥ {business_constraints.min_precision}")
    else:
        assessment['failed_checks'].append(f"Precision below threshold ({performance_metrics.get('precision', 0):.3f} < {business_constraints.min_precision})")
    
    if performance_metrics.get('recall', 0) >= business_constraints.min_recall:
        checks_passed += 1
        assessment['passed_checks'].append(f"Recall ≥ {business_constraints.min_recall}")
    else:
        assessment['failed_checks'].append(f"Recall below threshold ({performance_metrics.get('recall', 0):.3f} < {business_constraints.min_recall})")
    
    # F1-score check (should be reasonable)
    f1_threshold = 0.6  # Minimum reasonable F1-score
    if performance_metrics.get('f1_score', 0) >= f1_threshold:
        checks_passed += 1
        assessment['passed_checks'].append(f"F1-score ≥ {f1_threshold}")
    else:
        assessment['failed_checks'].append(f"F1-score below threshold ({performance_metrics.get('f1_score', 0):.3f} < {f1_threshold})")
    
    # Calculate readiness score
    assessment['readiness_score'] = checks_passed / total_checks if total_checks > 0 else 0.0
    
    # Determine deployment readiness
    assessment['ready_for_deployment'] = (
        assessment['readiness_score'] >= 0.8 and 
        len(assessment['failed_checks']) <= 1
    )
    
    # Add warnings
    if assessment['readiness_score'] < 0.6:
        assessment['warnings'].append("Low overall readiness score - significant improvements needed")
    elif assessment['readiness_score'] < 0.8:
        assessment['warnings'].append("Moderate readiness score - some improvements recommended")
    
    if not assessment['ready_for_deployment']:
        assessment['recommendations'].append("Address failed checks before deployment")
        assessment['recommendations'].append("Consider A/B testing with current production model")
    
    return assessment
def cre
ate_model_recommendation_system(models_results: Dict[str, Dict[str, np.ndarray]],
                                     cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
                                     business_constraints: Optional[BusinessConstraints] = None,
                                     training_times: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Complete model recommendation system with automated selection.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        cv_results: Cross-validation results for each model
        business_constraints: Business constraints and preferences
        training_times: Training times for each model
        
    Returns:
        Comprehensive recommendation report
    """
    from ..validation.metrics import calculate_comprehensive_metrics
    
    if business_constraints is None:
        business_constraints = BusinessConstraints()
    
    if not business_constraints.validate():
        raise ValueError("Invalid business constraints")
    
    recommendations = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        # Calculate performance metrics
        performance_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        # Get model characteristics
        model_cv_results = cv_results.get(model_name) if cv_results else None
        training_time = training_times.get(model_name) if training_times else None
        model_characteristics = get_model_characteristics(model_name, model_cv_results, training_time)
        
        # Calculate simple recommendation score based on primary metric
        primary_metric = business_constraints.primary_metric
        rec_score = performance_metrics.get(primary_metric, 0.0)
        
        # Assess deployment readiness
        deployment_assessment = assess_deployment_readiness(
            performance_metrics, model_characteristics, business_constraints, model_cv_results
        )
        
        recommendations.append({
            'model_name': model_name,
            'recommendation_score': rec_score,
            'performance_metrics': performance_metrics,
            'model_characteristics': model_characteristics,
            'deployment_assessment': deployment_assessment,
            'reasons': [f"{primary_metric}: {rec_score:.3f}"]
        })
    
    # Sort by recommendation score
    recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
    
    # Generate final recommendation
    best_model = recommendations[0]
    
    # Create comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'business_constraints': business_constraints,
        'recommended_model': best_model['model_name'],
        'recommendation_score': best_model['recommendation_score'],
        'deployment_ready': best_model['deployment_assessment']['ready_for_deployment'],
        'all_models': recommendations,
        'summary': f"Based on the analysis of {len(recommendations)} models, {best_model['model_name']} is recommended with a score of {best_model['recommendation_score']:.3f}.",
        'next_steps': ["Proceed with deployment" if best_model['deployment_assessment']['ready_for_deployment'] else "Address deployment issues"]
    }
    
    return report


class ModelRecommendationSystem:
    """
    Main class for model recommendation system functionality.
    
    This class provides a unified interface for all model recommendation
    capabilities including automated selection, deployment assessment,
    and report generation.
    """
    
    def __init__(self, business_constraints: Optional[BusinessConstraints] = None):
        """
        Initialize the recommendation system.
        
        Args:
            business_constraints: Business constraints and preferences
        """
        self.business_constraints = business_constraints or BusinessConstraints()
        
    def recommend_model(self, 
                       models_results: Dict[str, Dict[str, np.ndarray]],
                       cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
                       training_times: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate model recommendation based on performance and constraints.
        
        Args:
            models_results: Dictionary with model names as keys and results as values
            cv_results: Cross-validation results for each model
            training_times: Training times for each model
            
        Returns:
            Comprehensive recommendation report
        """
        return create_model_recommendation_system(
            models_results, cv_results, self.business_constraints, training_times
        )
    
    def assess_deployment_readiness(self,
                                  model_name: str,
                                  performance_metrics: Dict[str, float],
                                  cv_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess deployment readiness for a specific model.
        
        Args:
            model_name: Name of the model
            performance_metrics: Performance metrics dictionary
            cv_results: Cross-validation results
            
        Returns:
            Deployment readiness assessment
        """
        model_characteristics = get_model_characteristics(model_name, cv_results)
        return assess_deployment_readiness(
            performance_metrics, model_characteristics, self.business_constraints, cv_results
        )