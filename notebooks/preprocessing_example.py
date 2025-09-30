#!/usr/bin/env python3
"""
Example usage of data preprocessing and splitting functionality.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the notebooks directory to the path to import ml_pipeline_setup
sys.path.append('notebooks')

# Import the ML pipeline setup
from ml_pipeline_setup import (
    detect_class_imbalance,
    create_stratified_train_test_split,
    handle_class_imbalance,
    create_complete_preprocessing_pipeline
)

def main():
    """Example usage of preprocessing functions."""
    
    print("=" * 60)
    print("DATA PREPROCESSING EXAMPLE")
    print("=" * 60)
    
    # Load data
    print("Loading HP products data...")
    df = pd.read_csv('data/hp_products_labeled_mock_20250617_144244.csv')
    
    # Prepare features and target
    feature_cols = ['price_numeric', 'rating_numeric', 'reviews_count', 'confidence']
    X = df[feature_cols].fillna(0)
    y = df['target_is_original']
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Example 1: Basic class imbalance detection
    print(f"\n" + "=" * 40)
    print("EXAMPLE 1: CLASS IMBALANCE DETECTION")
    print("=" * 40)
    
    imbalance_info = detect_class_imbalance(y, threshold=0.3)
    
    if imbalance_info['is_imbalanced']:
        print(f"⚠️  Dataset is imbalanced!")
        print(f"   Minority class {imbalance_info['minority_class']}: {imbalance_info['minority_ratio']:.1%}")
        print(f"   Imbalance ratio: {imbalance_info['imbalance_ratio']:.1f}:1")
    else:
        print(f"✓ Dataset is reasonably balanced")
    
    # Example 2: Train-test split with validation
    print(f"\n" + "=" * 40)
    print("EXAMPLE 2: STRATIFIED TRAIN-TEST SPLIT")
    print("=" * 40)
    
    split_results = create_stratified_train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        validate_split=True
    )
    
    print(f"✓ Split completed:")
    print(f"   Training set: {split_results['X_train'].shape}")
    print(f"   Test set: {split_results['X_test'].shape}")
    print(f"   Split is valid: {split_results['validation_results']['is_valid']}")
    
    # Example 3: Handle class imbalance with different methods
    print(f"\n" + "=" * 40)
    print("EXAMPLE 3: CLASS IMBALANCE HANDLING")
    print("=" * 40)
    
    # Method 1: Class weights (always works)
    print("Method 1: Class Weights")
    weights_result = handle_class_imbalance(
        split_results['X_train'], 
        split_results['y_train'], 
        method='class_weights'
    )
    
    if weights_result['class_weights']:
        class_weights = weights_result['class_weights']['normalized_weights']
        print(f"✓ Class weights calculated: {class_weights}")
    
    # Method 2: SMOTE (if applicable)
    print(f"\nMethod 2: SMOTE Oversampling")
    smote_result = handle_class_imbalance(
        split_results['X_train'], 
        split_results['y_train'], 
        method='smote'
    )
    
    if smote_result['smote_info'] and 'samples_added' in smote_result['smote_info']:
        print(f"✓ SMOTE applied: {smote_result['smote_info']['samples_added']} samples added")
    else:
        print("ℹ️  SMOTE not applied (dataset already balanced)")
    
    # Example 4: Complete preprocessing pipeline
    print(f"\n" + "=" * 40)
    print("EXAMPLE 4: COMPLETE PREPROCESSING PIPELINE")
    print("=" * 40)
    
    # This combines splitting and imbalance handling in one step
    preprocessing_results = create_complete_preprocessing_pipeline(
        X, y,
        test_size=0.2,
        imbalance_method='class_weights',  # Use class_weights as it always works
        random_state=42
    )
    
    print(f"✓ Complete preprocessing finished:")
    print(f"   Final training set: {preprocessing_results['X_train'].shape}")
    print(f"   Final test set: {preprocessing_results['X_test'].shape}")
    
    # Show how to access the results
    X_train = preprocessing_results['X_train']
    X_test = preprocessing_results['X_test']
    y_train = preprocessing_results['y_train']
    y_test = preprocessing_results['y_test']
    
    # Access class weights if calculated
    if preprocessing_results['imbalance_info']['class_weights']:
        class_weights = preprocessing_results['imbalance_info']['class_weights']['normalized_weights']
        print(f"   Class weights for model training: {class_weights}")
    
    print(f"\n" + "=" * 60)
    print("PREPROCESSING EXAMPLE COMPLETE!")
    print("The data is now ready for model training.")
    print("=" * 60)
    
    return preprocessing_results

if __name__ == "__main__":
    results = main()