# Data Splitting Module

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def detect_class_imbalance(y, threshold=0.1):
    """
    Detect class imbalance in target variable.
    
    Parameters:
    -----------
    y : pandas.Series or numpy.array
        Target variable
    threshold : float
        Minimum ratio for minority class to be considered balanced
    
    Returns:
    --------
    dict : Class imbalance analysis results
    """
    # Calculate class distribution
    if hasattr(y, 'value_counts'):
        class_counts = y.value_counts()
    else:
        unique, counts = np.unique(y, return_counts=True)
        class_counts = pd.Series(counts, index=unique)
    
    total_samples = len(y)
    class_ratios = class_counts / total_samples
    
    # Identify minority and majority classes
    minority_class = class_ratios.idxmin()
    majority_class = class_ratios.idxmax()
    minority_ratio = class_ratios.min()
    
    # Determine if imbalanced
    is_imbalanced = minority_ratio < threshold
    
    imbalance_info = {
        'is_imbalanced': is_imbalanced,
        'class_counts': class_counts.to_dict(),
        'class_ratios': class_ratios.to_dict(),
        'minority_class': minority_class,
        'majority_class': majority_class,
        'minority_ratio': minority_ratio,
        'imbalance_ratio': class_ratios.max() / class_ratios.min(),
        'total_samples': total_samples
    }
    
    print(f"Class Imbalance Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Class ratios: {dict(zip(class_ratios.index, [f'{r:.3f}' for r in class_ratios.values]))}")
    print(f"Minority class: {minority_class} ({minority_ratio:.3f})")
    print(f"Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1")
    print(f"Is imbalanced (threshold={threshold}): {is_imbalanced}")
    
    return imbalance_info


def validate_train_test_split(X_train, X_test, y_train, y_test, original_y):
    """
    Validate that train-test split maintains class distribution.
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame or numpy.array
        Training and test feature sets
    y_train, y_test : pandas.Series or numpy.array
        Training and test target sets
    original_y : pandas.Series or numpy.array
        Original target variable before splitting
    
    Returns:
    --------
    dict : Validation results
    """
    # Calculate distributions
    if hasattr(original_y, 'value_counts'):
        original_dist = original_y.value_counts(normalize=True)
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
    else:
        original_unique, original_counts = np.unique(original_y, return_counts=True)
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        
        original_dist = pd.Series(original_counts / len(original_y), index=original_unique)
        train_dist = pd.Series(train_counts / len(y_train), index=train_unique)
        test_dist = pd.Series(test_counts / len(y_test), index=test_unique)
    
    # Calculate distribution differences
    train_diff = abs(original_dist - train_dist).max()
    test_diff = abs(original_dist - test_dist).max()
    
    # Validation criteria
    max_allowed_diff = 0.05  # 5% maximum difference
    is_valid = (train_diff <= max_allowed_diff) and (test_diff <= max_allowed_diff)
    
    validation_results = {
        'is_valid': is_valid,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_ratio': len(X_train) / (len(X_train) + len(X_test)),
        'original_distribution': original_dist.to_dict(),
        'train_distribution': train_dist.to_dict(),
        'test_distribution': test_dist.to_dict(),
        'train_difference': train_diff,
        'test_difference': test_diff,
        'max_allowed_difference': max_allowed_diff
    }
    
    print(f"\nTrain-Test Split Validation:")
    print(f"Train size: {len(X_train)} ({validation_results['train_ratio']:.1%})")
    print(f"Test size: {len(X_test)} ({1-validation_results['train_ratio']:.1%})")
    print(f"Original distribution: {dict(zip(original_dist.index, [f'{r:.3f}' for r in original_dist.values]))}")
    print(f"Train distribution: {dict(zip(train_dist.index, [f'{r:.3f}' for r in train_dist.values]))}")
    print(f"Test distribution: {dict(zip(test_dist.index, [f'{r:.3f}' for r in test_dist.values]))}")
    print(f"Max distribution difference: Train={train_diff:.3f}, Test={test_diff:.3f}")
    print(f"Split is valid: {is_valid}")
    
    return validation_results


def create_stratified_train_test_split(X, y, test_size=0.2, random_state=42, validate_split=True):
    """
    Create stratified train-test split with validation.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Feature matrix
    y : pandas.Series or numpy.array
        Target variable
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    validate_split : bool
        Whether to validate the split maintains class distribution
    
    Returns:
    --------
    dict : Dictionary containing split data and validation results
    """
    print(f"Creating stratified train-test split...")
    print(f"Original data shape: X={X.shape}, y={len(y)}")
    print(f"Test size: {test_size} ({test_size*100:.0f}%)")
    print(f"Random state: {random_state}")
    
    # Detect class imbalance before splitting
    imbalance_info = detect_class_imbalance(y)
    
    # Perform stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
            shuffle=True
        )
        
        print(f"Stratified split successful!")
        
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to regular train_test_split...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
    
    print(f"Split results: Train={X_train.shape}, Test={X_test.shape}")
    
    # Validate split if requested
    validation_results = None
    if validate_split:
        validation_results = validate_train_test_split(X_train, X_test, y_train, y_test, y)
    
    # Create split summary
    split_info = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_ratio': test_size,
        'random_state': random_state,
        'imbalance_info': imbalance_info,
        'validation_results': validation_results
    }
    
    return split_info