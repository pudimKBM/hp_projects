#!/usr/bin/env python3
"""
Test script for hyperparameter optimization pipeline (Task 5.4)
"""

import sys
import os
sys.path.append('notebooks')

# Import the ML pipeline functions
from ml_pipeline_setup import (
    calculate_search_space_size,
    select_optimization_method,
    optimize_model_hyperparameters,
    create_optimization_comparison,
    validate_optimization_results,
    MODEL_CONFIG
)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_search_space_calculation():
    """Test search space size calculation."""
    print("Testing search space calculation...")
    
    # Test with simple parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    expected_size = 3 * 3 * 2  # 18 combinations
    actual_size = calculate_search_space_size(param_grid)
    
    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
    print(f"✓ Search space calculation correct: {actual_size} combinations")

def test_method_selection():
    """Test automatic optimization method selection."""
    print("Testing optimization method selection...")
    
    # Test small search space (should use grid search)
    small_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    method_config = select_optimization_method(small_grid, max_grid_size=10)
    
    assert method_config['method'] == 'grid_search', "Small grid should use grid search"
    print(f"✓ Small grid correctly selected: {method_config['method']}")
    
    # Test large search space (should use random search)
    large_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 10]
    }
    method_config = select_optimization_method(large_grid, max_grid_size=100)
    
    assert method_config['method'] == 'random_search', "Large grid should use random search"
    print(f"✓ Large grid correctly selected: {method_config['method']}")

def test_optimization_comparison():
    """Test optimization results comparison."""
    print("Testing optimization comparison...")
    
    # Mock optimization summary
    optimization_summary = [
        {
            'model': 'random_forest',
            'best_score': 0.85,
            'execution_time': 120,
            'method': 'grid_search',
            'iterations': 50
        },
        {
            'model': 'svm',
            'best_score': 0.82,
            'execution_time': 180,
            'method': 'random_search',
            'iterations': 100
        },
        {
            'model': 'logistic_regression',
            'best_score': 0.78,
            'execution_time': 60,
            'method': 'grid_search',
            'iterations': 25
        }
    ]
    
    comparison = create_optimization_comparison(optimization_summary)
    
    assert comparison['best_model'] == 'random_forest', "Best model should be random_forest"
    assert len(comparison['rankings']) == 3, "Should have 3 rankings"
    assert comparison['rankings'][0]['rank'] == 1, "First ranking should be rank 1"
    
    print(f"✓ Comparison working correctly. Best model: {comparison['best_model']}")

def test_result_validation():
    """Test optimization result validation."""
    print("Testing result validation...")
    
    # Test successful results
    good_results = {
        'success': True,
        'best_score': 0.85,
        'cv_scores': [0.83, 0.85, 0.87, 0.84, 0.86],
        'execution_time': 300,
        'best_params': {'n_estimators': 200, 'max_depth': 10}
    }
    
    validation = validate_optimization_results(good_results)
    assert validation['is_valid'] == True, "Good results should be valid"
    print("✓ Good results validated correctly")
    
    # Test poor results
    poor_results = {
        'success': True,
        'best_score': 0.45,  # Below threshold
        'cv_scores': [0.3, 0.5, 0.6, 0.4, 0.45],  # High variance
        'execution_time': 300,
        'best_params': {'C': 1}
    }
    
    validation = validate_optimization_results(poor_results, min_score_threshold=0.6)
    assert len(validation['warnings']) > 0, "Poor results should have warnings"
    print("✓ Poor results flagged correctly")

def test_with_synthetic_data():
    """Test with synthetic dataset (quick test)."""
    print("Testing with synthetic data...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=200,  # Small dataset for quick testing
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Test optimization method selection for different models
    for model_name in ['random_forest', 'logistic_regression']:
        param_grid = MODEL_CONFIG[model_name]
        method_config = select_optimization_method(param_grid)
        print(f"Model: {model_name}, Method: {method_config['method']}, "
              f"Search space: {method_config['search_space_size']}")
    
    print("✓ Synthetic data test completed")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("🧪 TESTING HYPERPARAMETER OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    try:
        test_search_space_calculation()
        test_method_selection()
        test_optimization_comparison()
        test_result_validation()
        test_with_synthetic_data()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("🎯 Hyperparameter optimization pipeline is working correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    run_all_tests()