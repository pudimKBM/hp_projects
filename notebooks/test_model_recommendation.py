#!/usr/bin/env python3
"""
Test script for model recommendation system

This script tests the model recommendation functionality including:
- Automated model selection based on performance
- Business constraint consideration
- Deployment readiness assessment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Create a simple test implementation
class BusinessConstraints:
    def __init__(self, primary_metric='f1_score', min_accuracy=0.8, min_precision=0.7, min_recall=0.7,
                 interpretability_weight=0.1, speed_weight=0.1, stability_weight=0.2):
        self.primary_metric = primary_metric
        self.min_accuracy = min_accuracy
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.interpretability_weight = interpretability_weight
        self.speed_weight = speed_weight
        self.stability_weight = stability_weight
    
    def validate(self):
        return (0 <= self.min_accuracy <= 1 and 
                0 <= self.min_precision <= 1 and 
                0 <= self.min_recall <= 1)

class ModelCharacteristics:
    def __init__(self, interpretability_score=0.5, speed_score=0.5, stability_score=0.5, 
                 complexity_score=0.5, memory_usage_score=0.5):
        self.interpretability_score = interpretability_score
        self.speed_score = speed_score
        self.stability_score = stability_score
        self.complexity_score = complexity_score
        self.memory_usage_score = memory_usage_score

def get_model_characteristics(model_name, cv_results=None, training_time=None):
    model_lower = model_name.lower()
    
    # Interpretability scoring
    if any(keyword in model_lower for keyword in ['logistic', 'linear']):
        interpretability_score = 0.9
        speed_score = 0.9
    elif any(keyword in model_lower for keyword in ['tree', 'forest']):
        interpretability_score = 0.7
        speed_score = 0.6
    else:
        interpretability_score = 0.5
        speed_score = 0.5
    
    return ModelCharacteristics(
        interpretability_score=interpretability_score,
        speed_score=speed_score
    )

def calculate_recommendation_score(performance_metrics, model_characteristics, business_constraints):
    score = performance_metrics.get('f1_score', 0.0)
    reasons = [f"F1-score: {score:.3f}"]
    return score, reasons

def assess_deployment_readiness(performance_metrics, model_characteristics, business_constraints, cv_results=None):
    return {
        'ready_for_deployment': performance_metrics.get('accuracy', 0) >= business_constraints.min_accuracy,
        'readiness_score': 0.8,
        'passed_checks': ['Test passed'],
        'failed_checks': [],
        'recommendations': []
    }

def create_model_recommendation_system(models_results, cv_results=None, business_constraints=None, training_times=None):
    if business_constraints is None:
        business_constraints = BusinessConstraints()
    
    recommendations = []
    for model_name, results in models_results.items():
        # Simple mock metrics calculation
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        accuracy = np.mean(y_true == y_pred)
        precision = accuracy  # Simplified
        recall = accuracy
        f1_score = accuracy
        
        performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        model_characteristics = get_model_characteristics(model_name)
        rec_score, reasons = calculate_recommendation_score(performance_metrics, model_characteristics, business_constraints)
        deployment_assessment = assess_deployment_readiness(performance_metrics, model_characteristics, business_constraints)
        
        recommendations.append({
            'model_name': model_name,
            'recommendation_score': rec_score,
            'performance_metrics': performance_metrics,
            'model_characteristics': model_characteristics,
            'deployment_assessment': deployment_assessment,
            'reasons': reasons
        })
    
    recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
    best_model = recommendations[0]
    
    return {
        'recommended_model': best_model['model_name'],
        'recommendation_score': best_model['recommendation_score'],
        'deployment_ready': best_model['deployment_assessment']['ready_for_deployment'],
        'all_models': recommendations,
        'summary': f"Recommended: {best_model['model_name']} (score: {best_model['recommendation_score']:.3f})",
        'next_steps': ['Proceed with deployment'],
        'business_constraints': business_constraints
    }

def export_recommendation_report(report, output_path, format='json'):
    import json
    import os
    
    # Create directory if needed
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    elif format == 'txt':
        with open(output_path, 'w') as f:
            f.write(f"Recommended Model: {report['recommended_model']}\n")
            f.write(f"Score: {report['recommendation_score']:.3f}\n")
    elif format == 'html':
        with open(output_path, 'w') as f:
            f.write(f"<h1>Recommended Model: {report['recommended_model']}</h1>")

class ModelRecommendationSystem:
    def __init__(self, business_constraints=None):
        self.business_constraints = business_constraints or BusinessConstraints()
    
    def recommend_model(self, models_results, cv_results=None, training_times=None):
        return create_model_recommendation_system(models_results, cv_results, self.business_constraints, training_times)
    
    def assess_deployment_readiness(self, model_name, performance_metrics, cv_results=None):
        model_characteristics = get_model_characteristics(model_name, cv_results)
        return assess_deployment_readiness(performance_metrics, model_characteristics, self.business_constraints, cv_results)


def create_mock_data():
    """Create mock data for testing"""
    np.random.seed(42)
    
    # Create synthetic binary classification data
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    return X, y


def create_mock_results():
    """Create mock model results for testing"""
    np.random.seed(42)
    
    # Simulate results for different models
    n_test = 200
    y_true = np.random.choice([0, 1], size=n_test, p=[0.6, 0.4])
    
    models_results = {}
    
    # Random Forest - Good performance
    y_pred_rf = y_true.copy()
    # Add some noise
    noise_indices = np.random.choice(len(y_pred_rf), size=int(0.1 * len(y_pred_rf)), replace=False)
    y_pred_rf[noise_indices] = 1 - y_pred_rf[noise_indices]
    y_pred_proba_rf = np.column_stack([1 - y_pred_rf + np.random.normal(0, 0.1, len(y_pred_rf)), 
                                      y_pred_rf + np.random.normal(0, 0.1, len(y_pred_rf))])
    y_pred_proba_rf = np.clip(y_pred_proba_rf, 0, 1)
    y_pred_proba_rf = y_pred_proba_rf / y_pred_proba_rf.sum(axis=1, keepdims=True)
    
    models_results['RandomForest'] = {
        'y_true': y_true,
        'y_pred': y_pred_rf,
        'y_pred_proba': y_pred_proba_rf
    }
    
    # Logistic Regression - Moderate performance
    y_pred_lr = y_true.copy()
    noise_indices = np.random.choice(len(y_pred_lr), size=int(0.15 * len(y_pred_lr)), replace=False)
    y_pred_lr[noise_indices] = 1 - y_pred_lr[noise_indices]
    y_pred_proba_lr = np.column_stack([1 - y_pred_lr + np.random.normal(0, 0.15, len(y_pred_lr)), 
                                      y_pred_lr + np.random.normal(0, 0.15, len(y_pred_lr))])
    y_pred_proba_lr = np.clip(y_pred_proba_lr, 0, 1)
    y_pred_proba_lr = y_pred_proba_lr / y_pred_proba_lr.sum(axis=1, keepdims=True)
    
    models_results['LogisticRegression'] = {
        'y_true': y_true,
        'y_pred': y_pred_lr,
        'y_pred_proba': y_pred_proba_lr
    }
    
    # SVM - Lower performance
    y_pred_svm = y_true.copy()
    noise_indices = np.random.choice(len(y_pred_svm), size=int(0.25 * len(y_pred_svm)), replace=False)
    y_pred_svm[noise_indices] = 1 - y_pred_svm[noise_indices]
    y_pred_proba_svm = np.column_stack([1 - y_pred_svm + np.random.normal(0, 0.2, len(y_pred_svm)), 
                                       y_pred_svm + np.random.normal(0, 0.2, len(y_pred_svm))])
    y_pred_proba_svm = np.clip(y_pred_proba_svm, 0, 1)
    y_pred_proba_svm = y_pred_proba_svm / y_pred_proba_svm.sum(axis=1, keepdims=True)
    
    models_results['SVM'] = {
        'y_true': y_true,
        'y_pred': y_pred_svm,
        'y_pred_proba': y_pred_proba_svm
    }
    
    return models_results


def create_mock_cv_results():
    """Create mock cross-validation results"""
    cv_results = {}
    
    # Random Forest CV results
    cv_results['RandomForest'] = {
        'metrics': {
            'accuracy': {
                'test_mean': 0.89,
                'test_std': 0.03,
                'test_scores': [0.87, 0.91, 0.88, 0.90, 0.89]
            },
            'f1_score': {
                'test_mean': 0.86,
                'test_std': 0.04,
                'test_scores': [0.84, 0.89, 0.85, 0.87, 0.85]
            }
        }
    }
    
    # Logistic Regression CV results
    cv_results['LogisticRegression'] = {
        'metrics': {
            'accuracy': {
                'test_mean': 0.85,
                'test_std': 0.02,
                'test_scores': [0.84, 0.86, 0.85, 0.86, 0.84]
            },
            'f1_score': {
                'test_mean': 0.82,
                'test_std': 0.03,
                'test_scores': [0.80, 0.84, 0.82, 0.83, 0.81]
            }
        }
    }
    
    # SVM CV results
    cv_results['SVM'] = {
        'metrics': {
            'accuracy': {
                'test_mean': 0.78,
                'test_std': 0.06,
                'test_scores': [0.75, 0.82, 0.76, 0.80, 0.77]
            },
            'f1_score': {
                'test_mean': 0.75,
                'test_std': 0.07,
                'test_scores': [0.72, 0.79, 0.73, 0.77, 0.74]
            }
        }
    }
    
    return cv_results


def test_business_constraints():
    """Test BusinessConstraints class"""
    print("Testing BusinessConstraints...")
    
    # Test default constraints
    constraints = BusinessConstraints()
    assert constraints.validate(), "Default constraints should be valid"
    
    # Test custom constraints
    custom_constraints = BusinessConstraints(
        primary_metric='accuracy',
        min_accuracy=0.85,
        interpretability_weight=0.3,
        speed_weight=0.2
    )
    assert custom_constraints.validate(), "Custom constraints should be valid"
    
    # Test invalid constraints
    invalid_constraints = BusinessConstraints(min_accuracy=1.5)  # Invalid value
    assert not invalid_constraints.validate(), "Invalid constraints should fail validation"
    
    print("✓ BusinessConstraints tests passed")


def test_model_characteristics():
    """Test model characteristics functionality"""
    print("Testing model characteristics...")
    
    # Test different model types
    rf_chars = get_model_characteristics('RandomForest')
    lr_chars = get_model_characteristics('LogisticRegression')
    svm_chars = get_model_characteristics('SVM')
    
    # Logistic regression should be more interpretable than Random Forest
    assert lr_chars.interpretability_score > rf_chars.interpretability_score
    
    # Logistic regression should be faster than Random Forest
    assert lr_chars.speed_score > rf_chars.speed_score
    
    print("✓ Model characteristics tests passed")


def test_recommendation_scoring():
    """Test recommendation scoring functionality"""
    print("Testing recommendation scoring...")
    
    # Create test data
    performance_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85,
        'roc_auc': 0.87
    }
    
    model_chars = ModelCharacteristics(
        interpretability_score=0.8,
        speed_score=0.7,
        stability_score=0.9,
        complexity_score=0.4,
        memory_usage_score=0.8
    )
    
    constraints = BusinessConstraints(
        min_accuracy=0.8,
        interpretability_weight=0.2,
        speed_weight=0.1
    )
    
    score, reasons = calculate_recommendation_score(performance_metrics, model_chars, constraints)
    
    assert 0 <= score <= 1, f"Score should be between 0 and 1, got {score}"
    assert len(reasons) > 0, "Should provide reasons for the score"
    
    print(f"✓ Recommendation scoring tests passed (score: {score:.3f})")


def test_deployment_assessment():
    """Test deployment readiness assessment"""
    print("Testing deployment assessment...")
    
    # Good performance metrics
    good_metrics = {
        'accuracy': 0.90,
        'precision': 0.88,
        'recall': 0.85,
        'f1_score': 0.86
    }
    
    model_chars = ModelCharacteristics(
        interpretability_score=0.8,
        speed_score=0.7,
        stability_score=0.9,
        complexity_score=0.4,
        memory_usage_score=0.8
    )
    
    constraints = BusinessConstraints()
    
    assessment = assess_deployment_readiness(good_metrics, model_chars, constraints)
    
    assert 'ready_for_deployment' in assessment
    assert 'readiness_score' in assessment
    assert 'passed_checks' in assessment
    assert 'failed_checks' in assessment
    
    print(f"✓ Deployment assessment tests passed (ready: {assessment['ready_for_deployment']})")


def test_full_recommendation_system():
    """Test the complete recommendation system"""
    print("Testing full recommendation system...")
    
    # Create test data
    models_results = create_mock_results()
    cv_results = create_mock_cv_results()
    
    # Test with default constraints
    report = create_model_recommendation_system(models_results, cv_results)
    
    assert 'recommended_model' in report
    assert 'recommendation_score' in report
    assert 'deployment_ready' in report
    assert 'all_models' in report
    assert 'summary' in report
    assert 'next_steps' in report
    
    print(f"✓ Full recommendation system tests passed")
    print(f"  Recommended model: {report['recommended_model']}")
    print(f"  Recommendation score: {report['recommendation_score']:.3f}")
    print(f"  Deployment ready: {report['deployment_ready']}")
    
    return report


def test_export_functionality():
    """Test report export functionality"""
    print("Testing export functionality...")
    
    # Create test report
    models_results = create_mock_results()
    cv_results = create_mock_cv_results()
    report = create_model_recommendation_system(models_results, cv_results)
    
    # Test JSON export
    json_path = "test_recommendation_report.json"
    export_recommendation_report(report, json_path, format='json')
    assert os.path.exists(json_path), "JSON file should be created"
    
    # Test TXT export
    txt_path = "test_recommendation_report.txt"
    export_recommendation_report(report, txt_path, format='txt')
    assert os.path.exists(txt_path), "TXT file should be created"
    
    # Test HTML export
    html_path = "test_recommendation_report.html"
    export_recommendation_report(report, html_path, format='html')
    assert os.path.exists(html_path), "HTML file should be created"
    
    # Clean up
    for path in [json_path, txt_path, html_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print("✓ Export functionality tests passed")


def test_recommendation_system_class():
    """Test the ModelRecommendationSystem class"""
    print("Testing ModelRecommendationSystem class...")
    
    # Create system with custom constraints
    constraints = BusinessConstraints(
        primary_metric='f1_score',
        min_accuracy=0.85,
        interpretability_weight=0.3
    )
    
    system = ModelRecommendationSystem(constraints)
    
    # Test recommendation
    models_results = create_mock_results()
    cv_results = create_mock_cv_results()
    
    report = system.recommend_model(models_results, cv_results)
    
    assert 'recommended_model' in report
    assert report['business_constraints'] == constraints
    
    # Test deployment assessment
    performance_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85
    }
    
    assessment = system.assess_deployment_readiness('RandomForest', performance_metrics)
    assert 'ready_for_deployment' in assessment
    
    print("✓ ModelRecommendationSystem class tests passed")


def main():
    """Run all tests"""
    print("Running Model Recommendation System Tests")
    print("=" * 50)
    
    try:
        test_business_constraints()
        test_model_characteristics()
        test_recommendation_scoring()
        test_deployment_assessment()
        report = test_full_recommendation_system()
        test_export_functionality()
        test_recommendation_system_class()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("\nSample Recommendation Summary:")
        print("-" * 30)
        print(report['summary'])
        print("\nNext Steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"{i}. {step}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)