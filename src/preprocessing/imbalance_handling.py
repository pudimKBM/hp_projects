# Class Imbalance Handling Module

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from .data_splitting import detect_class_imbalance


def apply_smote_oversampling(X_train, y_train, k_neighbors=5, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.array
        Training feature matrix
    y_train : pandas.Series or numpy.array
        Training target variable
    k_neighbors : int
        Number of nearest neighbors for SMOTE
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing resampled data and SMOTE information
    """
    print(f"Applying SMOTE oversampling...")
    
    # Original class distribution
    original_counts = pd.Series(y_train).value_counts()
    print(f"Original class distribution: {original_counts.to_dict()}")
    
    # Convert sparse matrices to dense for SMOTE
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train
    
    # Initialize SMOTE
    try:
        smote = SMOTE(
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Apply SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_train_dense, y_train)
        
        # New class distribution
        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"Resampled class distribution: {resampled_counts.to_dict()}")
        
        # Calculate oversampling statistics
        total_original = len(y_train)
        total_resampled = len(y_resampled)
        samples_added = total_resampled - total_original
        
        smote_info = {
            'method': 'SMOTE',
            'k_neighbors': k_neighbors,
            'random_state': random_state,
            'original_counts': original_counts.to_dict(),
            'resampled_counts': resampled_counts.to_dict(),
            'original_size': total_original,
            'resampled_size': total_resampled,
            'samples_added': samples_added,
            'oversampling_ratio': total_resampled / total_original
        }
        
        print(f"SMOTE completed successfully!")
        print(f"Original size: {total_original} -> Resampled size: {total_resampled}")
        print(f"Samples added: {samples_added} ({(samples_added/total_original)*100:.1f}% increase)")
        
        return {
            'X_resampled': X_resampled,
            'y_resampled': y_resampled,
            'smote_info': smote_info,
            'success': True
        }
        
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Returning original data...")
        
        return {
            'X_resampled': X_train_dense,
            'y_resampled': y_train,
            'smote_info': {'method': 'None', 'error': str(e)},
            'success': False
        }


def calculate_class_weights(y_train, method='balanced'):
    """
    Calculate class weights for algorithms that support class weighting.
    
    Parameters:
    -----------
    y_train : pandas.Series or numpy.array
        Training target variable
    method : str
        Method for calculating weights ('balanced', 'balanced_subsample', 'custom')
    
    Returns:
    --------
    dict : Dictionary containing class weights and weight information
    """
    print(f"Calculating class weights using method: {method}")
    
    # Get class distribution
    if hasattr(y_train, 'value_counts'):
        class_counts = y_train.value_counts()
    else:
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = pd.Series(counts, index=unique)
    
    total_samples = len(y_train)
    n_classes = len(class_counts)
    
    if method == 'balanced':
        # Sklearn's balanced method: n_samples / (n_classes * np.bincount(y))
        class_weights = {}
        for class_label, count in class_counts.items():
            class_weights[class_label] = total_samples / (n_classes * count)
    
    elif method == 'balanced_subsample':
        # Similar to balanced but with subsampling consideration
        class_weights = {}
        for class_label, count in class_counts.items():
            class_weights[class_label] = total_samples / (n_classes * count)
    
    elif method == 'custom':
        # Custom inverse frequency weighting
        class_weights = {}
        for class_label, count in class_counts.items():
            class_weights[class_label] = 1.0 / (count / total_samples)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights so they sum to number of classes
    weight_sum = sum(class_weights.values())
    normalized_weights = {k: (v / weight_sum) * n_classes for k, v in class_weights.items()}
    
    weight_info = {
        'method': method,
        'class_counts': class_counts.to_dict(),
        'raw_weights': class_weights,
        'normalized_weights': normalized_weights,
        'total_samples': total_samples,
        'n_classes': n_classes
    }
    
    print(f"Class counts: {class_counts.to_dict()}")
    print(f"Raw weights: {dict(zip(class_weights.keys(), [f'{w:.3f}' for w in class_weights.values()]))}")
    print(f"Normalized weights: {dict(zip(normalized_weights.keys(), [f'{w:.3f}' for w in normalized_weights.values()]))}")
    
    return weight_info


def validate_balanced_dataset(X_resampled, y_resampled, original_y, balance_threshold=0.1):
    """
    Validate that the resampled dataset is properly balanced.
    
    Parameters:
    -----------
    X_resampled : pandas.DataFrame or numpy.array
        Resampled feature matrix
    y_resampled : pandas.Series or numpy.array
        Resampled target variable
    original_y : pandas.Series or numpy.array
        Original target variable before resampling
    balance_threshold : float
        Maximum allowed difference between class ratios for balanced dataset
    
    Returns:
    --------
    dict : Validation results
    """
    print(f"Validating balanced dataset...")
    
    # Calculate distributions
    if hasattr(y_resampled, 'value_counts'):
        resampled_counts = y_resampled.value_counts()
        original_counts = original_y.value_counts()
    else:
        resampled_unique, resampled_counts_arr = np.unique(y_resampled, return_counts=True)
        original_unique, original_counts_arr = np.unique(original_y, return_counts=True)
        resampled_counts = pd.Series(resampled_counts_arr, index=resampled_unique)
        original_counts = pd.Series(original_counts_arr, index=original_unique)
    
    # Calculate ratios
    resampled_ratios = resampled_counts / len(y_resampled)
    original_ratios = original_counts / len(original_y)
    
    # Check balance
    ratio_diff = abs(resampled_ratios.max() - resampled_ratios.min())
    is_balanced = ratio_diff <= balance_threshold
    
    # Calculate improvement
    original_imbalance = abs(original_ratios.max() - original_ratios.min())
    balance_improvement = original_imbalance - ratio_diff
    
    validation_results = {
        'is_balanced': is_balanced,
        'balance_threshold': balance_threshold,
        'original_size': len(original_y),
        'resampled_size': len(y_resampled),
        'original_counts': original_counts.to_dict(),
        'resampled_counts': resampled_counts.to_dict(),
        'original_ratios': original_ratios.to_dict(),
        'resampled_ratios': resampled_ratios.to_dict(),
        'original_imbalance': original_imbalance,
        'resampled_imbalance': ratio_diff,
        'balance_improvement': balance_improvement,
        'improvement_percentage': (balance_improvement / original_imbalance) * 100 if original_imbalance > 0 else 0
    }
    
    print(f"Original distribution: {dict(zip(original_ratios.index, [f'{r:.3f}' for r in original_ratios.values]))}")
    print(f"Resampled distribution: {dict(zip(resampled_ratios.index, [f'{r:.3f}' for r in resampled_ratios.values]))}")
    print(f"Original imbalance: {original_imbalance:.3f}")
    print(f"Resampled imbalance: {ratio_diff:.3f}")
    print(f"Balance improvement: {balance_improvement:.3f} ({validation_results['improvement_percentage']:.1f}%)")
    print(f"Dataset is balanced (threshold={balance_threshold}): {is_balanced}")
    
    return validation_results


def handle_class_imbalance(X_train, y_train, method='smote', **kwargs):
    """
    Comprehensive class imbalance handling with multiple methods.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.array
        Training feature matrix
    y_train : pandas.Series or numpy.array
        Training target variable
    method : str
        Method to handle imbalance ('smote', 'class_weights', 'both', 'none')
    **kwargs : dict
        Additional parameters for specific methods
    
    Returns:
    --------
    dict : Dictionary containing processed data and imbalance handling results
    """
    print(f"=" * 50)
    print(f"HANDLING CLASS IMBALANCE - Method: {method.upper()}")
    print(f"=" * 50)
    
    # Initial imbalance analysis
    initial_imbalance = detect_class_imbalance(y_train)
    
    results = {
        'method': method,
        'initial_imbalance': initial_imbalance,
        'X_processed': X_train,
        'y_processed': y_train,
        'class_weights': None,
        'smote_info': None,
        'validation_results': None
    }
    
    if method == 'none' or not initial_imbalance['is_imbalanced']:
        print("No imbalance handling applied.")
        return results
    
    elif method == 'smote':
        # Apply SMOTE oversampling
        smote_results = apply_smote_oversampling(
            X_train, y_train,
            k_neighbors=kwargs.get('k_neighbors', 5),
            random_state=kwargs.get('random_state', 42)
        )
        
        if smote_results['success']:
            results['X_processed'] = smote_results['X_resampled']
            results['y_processed'] = smote_results['y_resampled']
            results['smote_info'] = smote_results['smote_info']
            
            # Validate balanced dataset
            validation = validate_balanced_dataset(
                smote_results['X_resampled'],
                smote_results['y_resampled'],
                y_train
            )
            results['validation_results'] = validation
    
    elif method == 'class_weights':
        # Calculate class weights
        weight_info = calculate_class_weights(
            y_train,
            method=kwargs.get('weight_method', 'balanced')
        )
        results['class_weights'] = weight_info
        
        print("Class weights calculated. Use these weights in model training.")
    
    elif method == 'both':
        # Apply both SMOTE and calculate class weights
        print("Applying both SMOTE and class weights...")
        
        # First apply SMOTE
        smote_results = apply_smote_oversampling(
            X_train, y_train,
            k_neighbors=kwargs.get('k_neighbors', 5),
            random_state=kwargs.get('random_state', 42)
        )
        
        if smote_results['success']:
            results['X_processed'] = smote_results['X_resampled']
            results['y_processed'] = smote_results['y_resampled']
            results['smote_info'] = smote_results['smote_info']
            
            # Then calculate class weights on resampled data
            weight_info = calculate_class_weights(
                smote_results['y_resampled'],
                method=kwargs.get('weight_method', 'balanced')
            )
            results['class_weights'] = weight_info
            
            # Validate balanced dataset
            validation = validate_balanced_dataset(
                smote_results['X_resampled'],
                smote_results['y_resampled'],
                y_train
            )
            results['validation_results'] = validation
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Final summary
    print(f"\n" + "=" * 50)
    print(f"CLASS IMBALANCE HANDLING COMPLETE")
    print(f"=" * 50)
    print(f"Method used: {method}")
    print(f"Original size: {len(y_train)}")
    print(f"Final size: {len(results['y_processed'])}")
    
    if results['smote_info']:
        print(f"SMOTE applied: {results['smote_info']['samples_added']} samples added")
    
    if results['class_weights']:
        print(f"Class weights calculated: {results['class_weights']['method']} method")
    
    if results['validation_results']:
        print(f"Balance validation: {'PASSED' if results['validation_results']['is_balanced'] else 'FAILED'}")
    
    return results