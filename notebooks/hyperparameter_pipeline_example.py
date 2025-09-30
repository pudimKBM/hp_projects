#!/usr/bin/env python3
"""
Example usage of the hyperparameter optimization pipeline (Task 5.4)
This demonstrates the unified interface for hyperparameter optimization.
"""

import sys
import os
sys.path.append('notebooks')

# Import the ML pipeline functions
from ml_pipeline_setup import (
    optimize_model_hyperparameters,
    optimize_multiple_models,
    print_optimization_summary,
    save_optimization_pipeline_results
)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("Creating sample dataset...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        class_sep=0.8,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset created: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test

def example_single_model_optimization():
    """Example: Optimize a single model using the unified interface."""
    print("\n" + "=" * 70)
    print("🎯 EXAMPLE 1: SINGLE MODEL OPTIMIZATION")
    print("=" * 70)
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_sample_dataset()
    
    # Optimize Random Forest with automatic method selection
    print("\n🌳 Optimizing Random Forest with automatic method selection...")
    rf_results = optimize_model_hyperparameters(
        model_name='random_forest',
        X_train=X_train,
        y_train=y_train,
        search_budget='auto',  # Automatic budget selection
        time_budget_minutes=5,  # 5 minute time limit
        cv=3,  # 3-fold CV for faster execution
        scoring='f1'
    )
    
    print(f"\n📊 Results Summary:")
    print(f"Best Score: {rf_results['best_score']:.4f}")
    print(f"Best Parameters: {rf_results['best_params']}")
    print(f"Optimization Method: {rf_results['optimization_method']}")
    print(f"Execution Time: {rf_results['execution_time']:.1f} seconds")
    
    # Save results
    save_path = save_optimization_pipeline_results(rf_results, 'random_forest')
    print(f"Results saved to: {save_path}")
    
    return rf_results

def example_multi_model_optimization():
    """Example: Optimize multiple models and compare results."""
    print("\n" + "=" * 70)
    print("🎯 EXAMPLE 2: MULTI-MODEL OPTIMIZATION")
    print("=" * 70)
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_sample_dataset()
    
    # Define models to optimize
    models_to_optimize = ['random_forest', 'svm', 'logistic_regression']
    
    # Optimize multiple models
    print(f"\n🚀 Optimizing {len(models_to_optimize)} models...")
    multi_results = optimize_multiple_models(
        model_list=models_to_optimize,
        X_train=X_train,
        y_train=y_train,
        search_budget='medium',  # Medium search budget
        time_budget_per_model=3,  # 3 minutes per model
        cv=3,  # 3-fold CV for faster execution
        scoring='f1'
    )
    
    # Results are automatically printed by the function
    # But we can also access individual results
    print(f"\n📈 Individual Model Results:")
    for model_name, results in multi_results['individual_results'].items():
        if results['success']:
            print(f"{model_name}: {results['best_score']:.4f} "
                  f"({results['optimization_method']}, {results['execution_time']:.1f}s)")
        else:
            print(f"{model_name}: FAILED - {results.get('error', 'Unknown error')}")
    
    return multi_results

def example_custom_optimization():
    """Example: Custom optimization with specific parameters."""
    print("\n" + "=" * 70)
    print("🎯 EXAMPLE 3: CUSTOM OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_sample_dataset()
    
    # Define custom parameter grid
    custom_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    print(f"\n🔧 Using custom parameter grid with {np.prod([len(v) for v in custom_param_grid.values()])} combinations")
    
    # Optimize with custom parameters
    custom_results = optimize_model_hyperparameters(
        model_name='random_forest',
        X_train=X_train,
        y_train=y_train,
        search_budget=custom_param_grid,  # Use custom grid
        time_budget_minutes=5,
        cv=3,
        scoring='f1'
    )
    
    print(f"\n📊 Custom Optimization Results:")
    print(f"Best Score: {custom_results['best_score']:.4f}")
    print(f"Best Parameters: {custom_results['best_params']}")
    print(f"Method Used: {custom_results['optimization_method']}")
    
    return custom_results

def demonstrate_pipeline_features():
    """Demonstrate key features of the optimization pipeline."""
    print("\n" + "=" * 70)
    print("🎯 PIPELINE FEATURES DEMONSTRATION")
    print("=" * 70)
    
    print("\n✨ Key Features of the Hyperparameter Optimization Pipeline:")
    print("\n1. 🤖 Automatic Method Selection:")
    print("   - Small search spaces (≤100 combinations) → GridSearchCV")
    print("   - Large search spaces (>100 combinations) → RandomizedSearchCV")
    print("   - Time budget consideration for method selection")
    
    print("\n2. 🎯 Unified Interface:")
    print("   - Single function for all models: optimize_model_hyperparameters()")
    print("   - Multi-model optimization: optimize_multiple_models()")
    print("   - Consistent parameter interface across all algorithms")
    
    print("\n3. 📊 Comprehensive Comparison:")
    print("   - Performance ranking across models")
    print("   - Efficiency analysis (score per time)")
    print("   - Statistical significance testing")
    print("   - Business-friendly recommendations")
    
    print("\n4. ✅ Validation and Quality Assurance:")
    print("   - Automatic result validation")
    print("   - Overfitting detection")
    print("   - Performance threshold checking")
    print("   - Execution time monitoring")
    
    print("\n5. 💾 Result Management:")
    print("   - Automatic result saving with metadata")
    print("   - Version tracking and comparison")
    print("   - Comprehensive reporting")
    print("   - Easy result retrieval and analysis")
    
    print("\n6. 🔧 Flexibility and Customization:")
    print("   - Custom parameter grids")
    print("   - Configurable time budgets")
    print("   - Multiple scoring metrics")
    print("   - Cross-validation configuration")

def main():
    """Run all examples."""
    print("🚀 HYPERPARAMETER OPTIMIZATION PIPELINE EXAMPLES")
    print("=" * 70)
    print("This script demonstrates the unified hyperparameter optimization pipeline")
    print("that combines GridSearchCV and RandomizedSearchCV with automatic method selection.")
    
    try:
        # Run examples
        example_single_model_optimization()
        example_multi_model_optimization()
        example_custom_optimization()
        demonstrate_pipeline_features()
        
        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("🎯 The hyperparameter optimization pipeline is ready for production use.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ EXAMPLE FAILED: {str(e)}")
        print("=" * 70)
        raise

if __name__ == "__main__":
    main()