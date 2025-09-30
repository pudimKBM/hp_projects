# Model Training Module
"""
Model training components for ML pipeline
"""

from .random_forest import (
    train_random_forest,
    extract_rf_feature_importance,
    cross_validate_random_forest
)

from .svm import (
    train_svm,
    train_svm_multiple_kernels,
    cross_validate_svm,
    analyze_svm_support_vectors
)

from .logistic_regression import (
    train_logistic_regression,
    extract_lr_feature_importance,
    train_lr_multiple_penalties,
    cross_validate_logistic_regression,
    analyze_lr_regularization_path
)

from .gradient_boosting import (
    train_gradient_boosting,
    train_gb_with_early_stopping,
    extract_gb_feature_importance,
    train_gb_multiple_configs,
    cross_validate_gradient_boosting
)

__all__ = [
    'train_random_forest',
    'extract_rf_feature_importance',
    'cross_validate_random_forest',
    'train_svm',
    'train_svm_multiple_kernels',
    'cross_validate_svm',
    'analyze_svm_support_vectors',
    'train_logistic_regression',
    'extract_lr_feature_importance',
    'train_lr_multiple_penalties',
    'cross_validate_logistic_regression',
    'analyze_lr_regularization_path',
    'train_gradient_boosting',
    'train_gb_with_early_stopping',
    'extract_gb_feature_importance',
    'train_gb_multiple_configs',
    'cross_validate_gradient_boosting'
]