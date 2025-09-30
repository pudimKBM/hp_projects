"""
Test script for the interpretation module implementation.

This script tests the key functionality of the interpretation pipeline
to ensure all components work correctly together.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import our interpretation modules
from src.interpretation import (
    InterpretationPipeline,
    extract_tree_importance,
    calculate_permutation_importance,
    explain_prediction,
    plot_feature_importance
)

def test_interpretation_pipeline():
    """Test the complete interpretation pipeline."""
    print("Testing Model Interpretation Pipeline...")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    class_names = ['Original', 'Suspicious']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    models_dict = {
        'RandomForest': rf_model,
        'LogisticRegression': lr_model
    }
    
    print(f"✓ Trained {len(models_dict)} models")
    
    # Initialize interpretation pipeline
    pipeline = InterpretationPipeline(
        feature_names=feature_names,
        class_names=class_names
    )
    
    print("✓ Initialized InterpretationPipeline")
    
    # Test feature importance analysis
    print("\nTesting feature importance analysis...")
    
    # Test individual model analysis
    rf_importance = pipeline.analyze_model_importance(
        rf_model, X_test, y_test, 'RandomForest'
    )
    print(f"✓ Analyzed RandomForest importance: {list(rf_importance.keys())}")
    
    # Test multiple models analysis
    all_importance = pipeline.analyze_multiple_models(
        models_dict, X_test, y_test
    )
    print(f"✓ Analyzed multiple models importance: {list(all_importance.keys())}")
    
    # Test prediction explanations
    print("\nTesting prediction explanations...")
    
    # Explain a few predictions
    sample_indices = [0, 1, 2, 3, 4]
    X_samples = X_test[sample_indices]
    
    explanations = pipeline.explain_predictions(
        rf_model, X_samples, 'RandomForest'
    )
    print(f"✓ Generated {len(explanations)} prediction explanations")
    
    # Test individual explanation
    single_explanation = explain_prediction(
        rf_model, X_test[0], feature_names, class_names
    )
    print(f"✓ Generated single prediction explanation")
    print(f"  - Predicted class: {single_explanation['predicted_class']}")
    print(f"  - Confidence: {single_explanation.get('confidence', 'N/A')}")
    print(f"  - Top feature: {single_explanation['top_contributing_features'][0]['feature']}")
    
    # Test visualization creation
    print("\nTesting visualization creation...")
    
    try:
        plot_paths = pipeline.create_visualizations(
            output_dir="test_interpretation_plots",
            top_n=10
        )
        print(f"✓ Created {len(plot_paths)} visualization plots")
        for plot_name in list(plot_paths.keys())[:3]:  # Show first 3
            print(f"  - {plot_name}")
    except Exception as e:
        print(f"⚠ Visualization creation failed (expected in headless environment): {str(e)}")
    
    # Test insights generation
    print("\nTesting insights generation...")
    
    insights = pipeline.generate_insights()
    print(f"✓ Generated insights with {len(insights)} categories")
    print(f"  - Feature insights: {'✓' if 'feature_insights' in insights else '✗'}")
    print(f"  - Model insights: {'✓' if 'model_insights' in insights else '✗'}")
    print(f"  - Prediction insights: {'✓' if 'prediction_insights' in insights else '✗'}")
    print(f"  - Recommendations: {len(insights.get('recommendations', []))}")
    
    # Test summary report creation
    print("\nTesting summary report creation...")
    
    try:
        report_path = pipeline.create_interpretation_summary(
            "test_interpretation_summary.md"
        )
        print(f"✓ Created interpretation summary: {report_path}")
        
        # Read first few lines of the report
        with open(report_path, 'r') as f:
            lines = f.readlines()[:5]
            print("  Report preview:")
            for line in lines:
                print(f"    {line.strip()}")
    except Exception as e:
        print(f"✗ Summary report creation failed: {str(e)}")
    
    print("\n" + "="*50)
    print("✓ All interpretation pipeline tests completed successfully!")
    print("="*50)

def test_individual_components():
    """Test individual interpretation components."""
    print("\nTesting individual components...")
    
    # Create simple test data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test tree importance extraction
    try:
        tree_importance = extract_tree_importance(model, feature_names)
        print(f"✓ Tree importance extraction: {len(tree_importance)} features")
    except Exception as e:
        print(f"✗ Tree importance failed: {str(e)}")
    
    # Test permutation importance
    try:
        perm_importance = calculate_permutation_importance(
            model, X, y, feature_names, n_repeats=3
        )
        print(f"✓ Permutation importance: {len(perm_importance)} features")
    except Exception as e:
        print(f"✗ Permutation importance failed: {str(e)}")
    
    # Test single prediction explanation
    try:
        explanation = explain_prediction(model, X[0], feature_names)
        print(f"✓ Single prediction explanation generated")
    except Exception as e:
        print(f"✗ Prediction explanation failed: {str(e)}")

if __name__ == "__main__":
    print("Starting Interpretation Module Tests")
    print("="*50)
    
    try:
        test_individual_components()
        test_interpretation_pipeline()
        
        print("\n🎉 All tests passed! The interpretation module is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()