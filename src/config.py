# ML Pipeline Configuration
# Configuration dictionaries for models, features, and validation parameters

import numpy as np

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'text_features': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.95,
        'ngram_range': (1, 2),
        'stop_words': 'english'
    },
    'numerical_scaling': 'StandardScaler',
    'categorical_encoding': 'OneHotEncoder',
    'correlation_threshold': 0.95,
    'feature_selection': {
        'method': 'correlation',
        'threshold': 0.95
    }
}

# Validation Configuration
VALIDATION_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': RANDOM_STATE,
    'stratify': True,
    'min_accuracy_threshold': 0.8,
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'scoring': 'f1',  # Primary scoring metric for hyperparameter tuning
    'n_jobs': -1  # Use all available cores
}

# Class Imbalance Handling Configuration
IMBALANCE_CONFIG = {
    'method': 'smote',  # Options: 'smote', 'class_weight', 'both'
    'smote_k_neighbors': 5,
    'smote_random_state': RANDOM_STATE
}

# Model Persistence Configuration
PERSISTENCE_CONFIG = {
    'model_dir': 'models',
    'save_format': 'joblib',  # Options: 'joblib', 'pickle'
    'include_metadata': True,
    'versioning': True
}

# Report Generation Configuration
REPORT_CONFIG = {
    'output_dir': 'reports/ml_pipeline',
    'formats': ['html', 'pdf'],
    'include_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300
}