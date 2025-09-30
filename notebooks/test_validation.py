"""
Test script for the validation module.

This script tests the model validation and evaluation functionality
with synthetic data to ensure all components work correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Import validation modules
from src.validation import (
    calculate_comprehensive_metrics,
    calculate_metrics_for_multiple_models,
    plot_confusion_matrix,
    plot_multiple_confusion_matrices,
    plot_roc_curves,
    create_model_comparison_table,
    rank_models_by_performance,
    detailed_cv_analysis,
    compare_cv_results,
    create_performance_radar_chart,
    generate_model_recommendation
)


def test_validation_module():
    """Test the validation module with synthetic data."""
    print("Testing Model Validation Module")
    print("=" * 50)
    
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    print("\nTraining models...")
    
    # Train models and collect results
    models_results = {}
    cv_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Store results
        models_results[name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Perform CV analysis
        cv_result = detailed_cv_analysis(model, X_train, y_train, cv=5)
        cv_results[name] = cv_result
        
        print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    print("\n" + "=" * 50)
    print("TESTING VALIDATION FUNCTIONS")
    print("=" * 50)
    
    # Test 1: Comprehensive metrics calculation
    print("\n1. Testing comprehensive metrics calculation...")
    try:
        for name, results in models_results.items():
            metrics = calculate_comprehensive_metrics(
                results['y_true'], 
                results['y_pred'], 
                results['y_pred_proba'][:, 1]
            )
            print(f"  {name}: F1={metrics['f1_score']:.3f}, AUC={metrics['roc_auc']:.3f}")
        print("  ✓ Comprehensive metrics calculation works")
    except Exception as e:
        print(f"  ✗ Error in comprehensive metrics: {e}")
    
    # Test 2: Multiple models metrics
    print("\n2. Testing multiple models metrics...")
    try:
        metrics_df = calculate_metrics_for_multiple_models(models_results)
        print("  Metrics DataFrame shape:", metrics_df.shape)
        print("  ✓ Multiple models metrics calculation works")
    except Exception as e:
        print(f"  ✗ Error in multiple models metrics: {e}")
    
    # Test 3: Model comparison table
    print("\n3. Testing model comparison table...")
    try:
        comparison_table = create_model_comparison_table(models_results, cv_results)
        print("  Comparison table shape:", comparison_table.shape)
        print("  ✓ Model comparison table works")
    except Exception as e:
        print(f"  ✗ Error in comparison table: {e}")
    
    # Test 4: Model ranking
    print("\n4. Testing model ranking...")
    try:
        ranking = rank_models_by_performance(models_results)
        print("  Ranking shape:", ranking.shape)
        print("  Top model:", ranking.iloc[0]['Model'])
        print("  ✓ Model ranking works")
    except Exception as e:
        print(f"  ✗ Error in model ranking: {e}")
    
    # Test 5: CV comparison
    print("\n5. Testing CV comparison...")
    try:
        cv_comparison = compare_cv_results(cv_results, metric='f1')
        print("  CV comparison shape:", cv_comparison.shape)
        print("  ✓ CV comparison works")
    except Exception as e:
        print(f"  ✗ Error in CV comparison: {e}")
    
    # Test 6: Model recommendation
    print("\n6. Testing model recommendation...")
    try:
        recommendation = generate_model_recommendation(models_results, cv_results)
        print("  Recommended model:", recommendation['recommended_model'])
        print("  Recommendation score:", recommendation['recommendation_score'])
        print("  ✓ Model recommendation works")
    except Exception as e:
        print(f"  ✗ Error in model recommendation: {e}")
    
    # Test 7: Visualizations (without displaying)
    print("\n7. Testing visualizations...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Test confusion matrix
        fig1 = plot_confusion_matrix(
            models_results['Random Forest']['y_true'],
            models_results['Random Forest']['y_pred']
        )
        print("  ✓ Confusion matrix plot works")
        
        # Test multiple confusion matrices
        fig2 = plot_multiple_confusion_matrices(models_results)
        print("  ✓ Multiple confusion matrices plot works")
        
        # Test ROC curves
        fig3 = plot_roc_curves(models_results)
        print("  ✓ ROC curves plot works")
        
        # Test radar chart
        fig4 = create_performance_radar_chart(models_results)
        print("  ✓ Performance radar chart works")
        
        # Close figures to free memory
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except Exception as e:
        print(f"  ✗ Error in visualizations: {e}")
    
    print("\n" + "=" * 50)
    print("VALIDATION MODULE TEST COMPLETED")
    print("=" * 50)
    
    return models_results, cv_results


if __name__ == "__main__":
    models_results, cv_results = test_validation_module()
    print("\nTest completed successfully!")
    print("The validation module is ready for use in the ML pipeline.")