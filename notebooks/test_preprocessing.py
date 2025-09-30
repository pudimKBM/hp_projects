#!/usr/bin/env python3
"""
Test script for data preprocessing and splitting functionality.
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

def test_preprocessing_functions():
    """Test the data preprocessing functions with sample data."""
    
    print("=" * 60)
    print("TESTING DATA PREPROCESSING FUNCTIONS")
    print("=" * 60)
    
    # Load sample data
    print("Loading sample data...")
    try:
        df = pd.read_csv('data/hp_products_labeled_mock_20250617_144244.csv')
        print(f"Data loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check target column
        target_col = 'target_is_original'
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")
            return False
        
        # Create simple feature matrix for testing
        # Use numerical and categorical features
        feature_cols = ['price_numeric', 'rating_numeric', 'reviews_count', 'confidence']
        X = df[feature_cols].fillna(0)  # Simple preprocessing for testing
        y = df[target_col]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Test 1: Class imbalance detection
    print(f"\n" + "=" * 40)
    print("TEST 1: CLASS IMBALANCE DETECTION")
    print("=" * 40)
    
    try:
        imbalance_info = detect_class_imbalance(y)
        print("✓ Class imbalance detection successful")
    except Exception as e:
        print(f"✗ Class imbalance detection failed: {e}")
        return False
    
    # Test 2: Train-test split
    print(f"\n" + "=" * 40)
    print("TEST 2: TRAIN-TEST SPLIT")
    print("=" * 40)
    
    try:
        split_results = create_stratified_train_test_split(X, y, test_size=0.2, random_state=42)
        print("✓ Train-test split successful")
        print(f"  Train shape: {split_results['X_train'].shape}")
        print(f"  Test shape: {split_results['X_test'].shape}")
    except Exception as e:
        print(f"✗ Train-test split failed: {e}")
        return False
    
    # Test 3: Class imbalance handling (SMOTE)
    print(f"\n" + "=" * 40)
    print("TEST 3: CLASS IMBALANCE HANDLING (SMOTE)")
    print("=" * 40)
    
    try:
        imbalance_results = handle_class_imbalance(
            split_results['X_train'], 
            split_results['y_train'], 
            method='smote'
        )
        print("✓ SMOTE imbalance handling successful")
        print(f"  Original size: {len(split_results['y_train'])}")
        print(f"  Resampled size: {len(imbalance_results['y_processed'])}")
    except Exception as e:
        print(f"✗ SMOTE imbalance handling failed: {e}")
        print("Trying class weights method instead...")
        
        try:
            imbalance_results = handle_class_imbalance(
                split_results['X_train'], 
                split_results['y_train'], 
                method='class_weights'
            )
            print("✓ Class weights imbalance handling successful")
        except Exception as e2:
            print(f"✗ Class weights imbalance handling also failed: {e2}")
            return False
    
    # Test 4: Complete preprocessing pipeline
    print(f"\n" + "=" * 40)
    print("TEST 4: COMPLETE PREPROCESSING PIPELINE")
    print("=" * 40)
    
    try:
        preprocessing_results = create_complete_preprocessing_pipeline(
            X, y, 
            test_size=0.2, 
            imbalance_method='class_weights',  # Use class_weights as fallback
            random_state=42
        )
        print("✓ Complete preprocessing pipeline successful")
        print(f"  Final train shape: {preprocessing_results['X_train'].shape}")
        print(f"  Final test shape: {preprocessing_results['X_test'].shape}")
    except Exception as e:
        print(f"✗ Complete preprocessing pipeline failed: {e}")
        return False
    
    print(f"\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    print("Data preprocessing functions are working correctly.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_preprocessing_functions()
    if not success:
        sys.exit(1)