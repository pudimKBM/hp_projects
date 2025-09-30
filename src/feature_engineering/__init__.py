# Feature Engineering Module
"""
Feature engineering components for ML pipeline
"""

from .text_features import (
    create_tfidf_features,
    create_text_length_features,
    create_keyword_features,
    combine_text_features,
    extract_all_text_features
)

from .numerical_features import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    handle_outliers,
    create_numerical_features,
    validate_numerical_features
)

from .categorical_features import (
    create_onehot_encoding,
    create_label_encoding,
    create_frequency_encoding,
    create_categorical_features,
    handle_unseen_categories
)

from .correlation_analysis import (
    calculate_correlation_matrix,
    find_highly_correlated_features,
    select_features_to_remove,
    remove_correlated_features,
    analyze_feature_correlations,
    create_correlation_report
)

from .pipeline import (
    FeatureEngineeringPipeline,
    engineer_features,
    validate_input_data
)

__all__ = [
    'create_tfidf_features',
    'create_text_length_features', 
    'create_keyword_features',
    'combine_text_features',
    'extract_all_text_features',
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'handle_outliers',
    'create_numerical_features',
    'validate_numerical_features',
    'create_onehot_encoding',
    'create_label_encoding',
    'create_frequency_encoding',
    'create_categorical_features',
    'handle_unseen_categories',
    'calculate_correlation_matrix',
    'find_highly_correlated_features',
    'select_features_to_remove',
    'remove_correlated_features',
    'analyze_feature_correlations',
    'create_correlation_report',
    'FeatureEngineeringPipeline',
    'engineer_features',
    'validate_input_data'
]