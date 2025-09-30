# Modular ML Pipeline Setup
# Clean, maintainable version using modular structure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime

# Import our modular components
import sys
sys.path.append('../src')

from config import (
    RANDOM_STATE, MODEL_CONFIG, FEATURE_CONFIG, VALIDATION_CONFIG,
    IMBALANCE_CONFIG, PERSISTENCE_CONFIG, REPORT_CONFIG
)

from feature_engineering import (
    FeatureEngineeringPipeline,
    engineer_features,
    extract_all_text_features,
    create_numerical_features,
    create_categorical_features,
    analyze_feature_correlations
)

from preprocessing import (
    create_complete_preprocessing_pipeline,
    detect_class_imbalance,
    handle_class_imbalance,
    create_stratified_train_test_split
)

from models import (
    train_random_forest,
    train_svm,
    train_logistic_regression,
    train_gradient_boosting
)

from hyperparameter_optimization import (
    optimize_model_hyperparameters,
    optimize_multiple_models,
    run_comprehensive_grid_search,
    create_comprehensive_param_grids
)

print("=" * 60)
print("MODULAR ML PIPELINE SETUP COMPLETE!")
print("=" * 60)
print(f"Random state set to: {RANDOM_STATE}")
print("All modular components imported successfully")

# Create necessary directories
os.makedirs(PERSISTENCE_CONFIG['model_dir'], exist_ok=True)
os.makedirs(REPORT_CONFIG['output_dir'], exist_ok=True)
os.makedirs('visualizations/ml_pipeline', exist_ok=True)

print(f"\nDirectories created:")
print(f"- Models: {PERSISTENCE_CONFIG['model_dir']}")
print(f"- Reports: {REPORT_CONFIG['output_dir']}")
print(f"- Visualizations: visualizations/ml_pipeline")

print("\n" + "=" * 60)
print("AVAILABLE PIPELINE COMPONENTS:")
print("=" * 60)

print("\n🔧 FEATURE ENGINEERING:")
print("- FeatureEngineeringPipeline: Complete feature engineering pipeline")
print("- engineer_features(): Quick feature engineering function")
print("- extract_all_text_features(): Text feature extraction")
print("- create_numerical_features(): Numerical feature processing")
print("- create_categorical_features(): Categorical feature encoding")

print("\n📊 DATA PREPROCESSING:")
print("- create_complete_preprocessing_pipeline(): End-to-end preprocessing")
print("- create_stratified_train_test_split(): Train-test splitting")
print("- handle_class_imbalance(): SMOTE and class weighting")
print("- detect_class_imbalance(): Class distribution analysis")

print("\n🤖 MODEL TRAINING:")
print("- train_random_forest(): Random Forest training")
print("- train_svm(): SVM training")
print("- train_logistic_regression(): Logistic Regression training")
print("- train_gradient_boosting(): Gradient Boosting training")

print("\n🎯 HYPERPARAMETER OPTIMIZATION:")
print("- optimize_model_hyperparameters(): Single model optimization")
print("- optimize_multiple_models(): Multi-model optimization")
print("- run_comprehensive_grid_search(): GridSearchCV for all models")
print("- create_comprehensive_param_grids(): Parameter grid generation")

print("\n" + "=" * 60)
print("EXAMPLE USAGE:")
print("=" * 60)

example_code = '''
# 1. Feature Engineering
pipeline = FeatureEngineeringPipeline()
X, y = pipeline.fit_transform(df, target_column='target_is_original')

# 2. Data Preprocessing
preprocessing_results = create_complete_preprocessing_pipeline(
    X, y, test_size=0.2, imbalance_method='smote'
)

# 3. Model Training
rf_results = train_random_forest(
    preprocessing_results['X_train'], 
    preprocessing_results['y_train'],
    preprocessing_results['X_test'],
    preprocessing_results['y_test']
)

# 4. Hyperparameter Optimization
optimization_results = optimize_multiple_models(
    ['random_forest', 'svm', 'logistic_regression'],
    preprocessing_results['X_train'],
    preprocessing_results['y_train']
)
'''

print(example_code)

print("=" * 60)
print("Ready to build ML models with clean, modular code!")
print("=" * 60)