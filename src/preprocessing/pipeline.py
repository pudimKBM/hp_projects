# Complete Preprocessing Pipeline Module

from .data_splitting import create_stratified_train_test_split
from .imbalance_handling import handle_class_imbalance


def create_complete_preprocessing_pipeline(X, y, test_size=0.2, imbalance_method='smote', random_state=42):
    """
    Complete data preprocessing pipeline including splitting and imbalance handling.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Feature matrix
    y : pandas.Series or numpy.array
        Target variable
    test_size : float
        Proportion of data for test set
    imbalance_method : str
        Method to handle class imbalance
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing all preprocessing results
    """
    print("=" * 60)
    print("COMPLETE DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Train-test split
    print("\n1. TRAIN-TEST SPLIT")
    print("-" * 30)
    split_results = create_stratified_train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        validate_split=True
    )
    
    # Step 2: Handle class imbalance on training data only
    print(f"\n2. CLASS IMBALANCE HANDLING")
    print("-" * 30)
    imbalance_results = handle_class_imbalance(
        split_results['X_train'],
        split_results['y_train'],
        method=imbalance_method,
        random_state=random_state
    )
    
    # Combine results
    preprocessing_results = {
        'X_train': imbalance_results['X_processed'],
        'X_test': split_results['X_test'],
        'y_train': imbalance_results['y_processed'],
        'y_test': split_results['y_test'],
        'split_info': split_results,
        'imbalance_info': imbalance_results,
        'preprocessing_config': {
            'test_size': test_size,
            'imbalance_method': imbalance_method,
            'random_state': random_state
        }
    }
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"PREPROCESSING PIPELINE COMPLETE")
    print(f"=" * 60)
    print(f"Final training set: {preprocessing_results['X_train'].shape}")
    print(f"Final test set: {preprocessing_results['X_test'].shape}")
    print(f"Imbalance method: {imbalance_method}")
    print(f"Ready for model training!")
    
    return preprocessing_results