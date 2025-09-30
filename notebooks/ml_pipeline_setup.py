# ML Pipeline Infrastructure and Configuration
# Task 1: Set up ML pipeline infrastructure and configuration

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

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)                                               

print("ML Pipeline Infrastructure Setup Complete!")
print(f"Random state set to: {RANDOM_STATE}")
print("Required libraries imported successfully")

# Configuration dictionaries for models, features, and validation parameters

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

print("\nConfiguration Setup Complete!")
print("=" * 50)
print("MODEL_CONFIG keys:", list(MODEL_CONFIG.keys()))
print("FEATURE_CONFIG keys:", list(FEATURE_CONFIG.keys()))
print("VALIDATION_CONFIG:", VALIDATION_CONFIG)
print("IMBALANCE_CONFIG:", IMBALANCE_CONFIG)
print("PERSISTENCE_CONFIG:", PERSISTENCE_CONFIG)
print("REPORT_CONFIG:", REPORT_CONFIG)

# Create necessary directories
os.makedirs(PERSISTENCE_CONFIG['model_dir'], exist_ok=True)
os.makedirs(REPORT_CONFIG['output_dir'], exist_ok=True)
os.makedirs('visualizations/ml_pipeline', exist_ok=True)

print(f"\nDirectories created:")
print(f"- Models: {PERSISTENCE_CONFIG['model_dir']}")
print(f"- Reports: {REPORT_CONFIG['output_dir']}")
print(f"- Visualizations: visualizations/ml_pipeline")

print("\n" + "=" * 50)
print("ML Pipeline Infrastructure Setup Complete!")
print("Ready to proceed with feature engineering and model training.")
print("=" * 50)

import joblib
from scipy.sparse import hstack, csr_matrix
import re

# Set random seeds for reproducibility
np.random.seed(42)

# Configuration dictionaries
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'random_state': 42
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
        'random_state': 42
    },
    'logistic_regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'random_state': 42
    }
}

FEATURE_CONFIG = {
    'text_features': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.95,
        'ngram_range': (1, 2)
    },
    'numerical_scaling': 'StandardScaler',
    'categorical_encoding': 'OneHotEncoder',
    'correlation_threshold': 0.95
}

VALIDATION_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'stratify': True,
    'min_accuracy_threshold': 0.8,
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
}

# ============================================================================
# TASK 2.1: TEXT FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def create_tfidf_features(df, text_columns=['title', 'description'], max_features=1000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from text columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing text columns
    text_columns : list
        List of column names containing text data
    max_features : int
        Maximum number of features for TF-IDF
    ngram_range : tuple
        Range of n-grams to extract
    
    Returns:
    --------
    dict : Dictionary containing TF-IDF matrices and vectorizers
    """
    tfidf_features = {}
    
    for col in text_columns:
        if col in df.columns:
            # Handle missing values
            text_data = df[col].fillna('').astype(str)
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.95,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True
            )
            
            # Fit and transform text data
            tfidf_matrix = vectorizer.fit_transform(text_data)
            
            # Store results
            tfidf_features[f'{col}_tfidf'] = {
                'matrix': tfidf_matrix,
                'vectorizer': vectorizer,
                'feature_names': [f'{col}_tfidf_{name}' for name in vectorizer.get_feature_names_out()]
            }
            
            print(f"Created TF-IDF features for '{col}': {tfidf_matrix.shape[1]} features")
    
    return tfidf_features

def create_text_length_features(df, text_columns=['title', 'description']):
    """
    Create text length and basic text statistics features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing text columns
    text_columns : list
        List of column names containing text data
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with text length features
    """
    text_features = pd.DataFrame(index=df.index)
    
    for col in text_columns:
        if col in df.columns:
            # Handle missing values
            text_data = df[col].fillna('').astype(str)
            
            # Character count
            text_features[f'{col}_char_count'] = text_data.str.len()
            
            # Word count
            text_features[f'{col}_word_count'] = text_data.str.split().str.len()
            
            # Average word length
            text_features[f'{col}_avg_word_length'] = text_data.apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            
            # Number of uppercase words
            text_features[f'{col}_uppercase_count'] = text_data.apply(
                lambda x: sum(1 for word in x.split() if word.isupper())
            )
            
            print(f"Created text length features for '{col}': {text_features.filter(regex=f'^{col}_').shape[1]} features")
    
    return text_features

def create_keyword_features(df, text_columns=['title', 'description']):
    """
    Create keyword detection features for HP product authenticity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing text columns
    text_columns : list
        List of column names containing text data
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with keyword features
    """
    keyword_features = pd.DataFrame(index=df.index)
    
    # Define keyword patterns for authenticity detection
    authenticity_keywords = {
        'original': ['original', 'genuine', 'authentic', 'official'],
        'hp_brand': ['hp', 'hewlett', 'packard'],
        'quality': ['quality', 'premium', 'professional', 'high-quality'],
        'compatibility': ['compatible', 'compatível', 'replacement'],
        'warranty': ['warranty', 'garantia', 'guarantee'],
        'sealed': ['sealed', 'lacrado', 'new', 'novo'],
        'invoice': ['invoice', 'nota fiscal', 'receipt', 'nf']
    }
    
    for col in text_columns:
        if col in df.columns:
            # Handle missing values and convert to lowercase
            text_data = df[col].fillna('').astype(str).str.lower()
            
            # Create keyword features
            for keyword_type, keywords in authenticity_keywords.items():
                feature_name = f'{col}_has_{keyword_type}'
                keyword_features[feature_name] = text_data.apply(
                    lambda x: any(keyword in x for keyword in keywords)
                ).astype(int)
            
            # Count total authenticity keywords
            all_keywords = [kw for kw_list in authenticity_keywords.values() for kw in kw_list]
            keyword_features[f'{col}_authenticity_keyword_count'] = text_data.apply(
                lambda x: sum(1 for keyword in all_keywords if keyword in x)
            )
            
            print(f"Created keyword features for '{col}': {keyword_features.filter(regex=f'^{col}_').shape[1]} features")
    
    return keyword_features

def combine_text_features(tfidf_features, text_length_features, keyword_features):
    """
    Combine all text features into a single feature matrix.
    
    Parameters:
    -----------
    tfidf_features : dict
        Dictionary containing TF-IDF matrices and metadata
    text_length_features : pandas.DataFrame
        DataFrame with text length features
    keyword_features : pandas.DataFrame
        DataFrame with keyword features
    
    Returns:
    --------
    dict : Dictionary containing combined feature matrix and feature names
    """
    # Start with non-TF-IDF features
    dense_features = pd.concat([text_length_features, keyword_features], axis=1)
    
    # Convert to sparse matrix for memory efficiency
    dense_sparse = csr_matrix(dense_features.values)
    
    # Collect all feature names
    all_feature_names = list(dense_features.columns)
    
    # Add TF-IDF features
    sparse_matrices = [dense_sparse]
    
    for tfidf_name, tfidf_data in tfidf_features.items():
        sparse_matrices.append(tfidf_data['matrix'])
        all_feature_names.extend(tfidf_data['feature_names'])
    
    # Combine all sparse matrices
    combined_matrix = hstack(sparse_matrices)
    
    print(f"Combined text features: {combined_matrix.shape[1]} total features")
    print(f"Feature breakdown:")
    print(f"  - Text length features: {text_length_features.shape[1]}")
    print(f"  - Keyword features: {keyword_features.shape[1]}")
    for tfidf_name, tfidf_data in tfidf_features.items():
        print(f"  - {tfidf_name}: {tfidf_data['matrix'].shape[1]}")
    
    return {
        'matrix': combined_matrix,
        'feature_names': all_feature_names,
        'tfidf_vectorizers': {name: data['vectorizer'] for name, data in tfidf_features.items()}
    }

def extract_all_text_features(df, text_columns=['title', 'description']):
    """
    Extract all text features from the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing text columns
    text_columns : list
        List of column names containing text data
    
    Returns:
    --------
    dict : Dictionary containing all text features and metadata
    """
    print("Starting text feature extraction...")
    
    # Create TF-IDF features
    print("\n1. Creating TF-IDF features...")
    tfidf_features = create_tfidf_features(
        df, 
        text_columns=text_columns,
        max_features=FEATURE_CONFIG['text_features']['max_features'],
        ngram_range=FEATURE_CONFIG['text_features']['ngram_range']
    )
    
    # Create text length features
    print("\n2. Creating text length features...")
    text_length_features = create_text_length_features(df, text_columns=text_columns)
    
    # Create keyword features
    print("\n3. Creating keyword features...")
    keyword_features = create_keyword_features(df, text_columns=text_columns)
    
    # Combine all text features
    print("\n4. Combining text features...")
    combined_features = combine_text_features(tfidf_features, text_length_features, keyword_features)
    
    print("\nText feature extraction completed!")
    
    return combined_features
# ============================================================================
# TASK 2.2: NUMERICAL FEATURE PREPROCESSING
# ============================================================================

def detect_outliers_iqr(series, multiplier=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    series : pandas.Series
        Numerical series to analyze
    multiplier : float
        IQR multiplier for outlier detection
    
    Returns:
    --------
    pandas.Series : Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers

def detect_outliers_zscore(series, threshold=3):
    """
    Detect outliers using Z-score method.
    
    Parameters:
    -----------
    series : pandas.Series
        Numerical series to analyze
    threshold : float
        Z-score threshold for outlier detection
    
    Returns:
    --------
    pandas.Series : Boolean series indicating outliers
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    outliers = z_scores > threshold
    return outliers

def handle_outliers(df, numerical_columns, method='cap', outlier_method='iqr'):
    """
    Handle outliers in numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numerical_columns : list
        List of numerical column names
    method : str
        Method to handle outliers ('cap', 'remove', 'transform')
    outlier_method : str
        Method to detect outliers ('iqr', 'zscore')
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with outliers handled
    """
    df_processed = df.copy()
    outlier_info = {}
    
    for col in numerical_columns:
        if col in df.columns:
            # Detect outliers
            if outlier_method == 'iqr':
                outliers = detect_outliers_iqr(df[col])
            else:
                outliers = detect_outliers_zscore(df[col])
            
            outlier_count = outliers.sum()
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100
            }
            
            if outlier_count > 0:
                if method == 'cap':
                    # Cap outliers to 5th and 95th percentiles
                    lower_cap = df[col].quantile(0.05)
                    upper_cap = df[col].quantile(0.95)
                    df_processed[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                    
                elif method == 'remove':
                    # Remove outlier rows (not recommended for small datasets)
                    df_processed = df_processed[~outliers]
                    
                elif method == 'transform':
                    # Log transform to reduce outlier impact
                    df_processed[col] = np.log1p(df[col])
            
            print(f"Column '{col}': {outlier_count} outliers ({outlier_info[col]['percentage']:.1f}%) - {method}")
    
    return df_processed, outlier_info

def create_numerical_features(df, numerical_columns=['price_numeric', 'rating_numeric', 'reviews_count']):
    """
    Create and preprocess numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numerical_columns : list
        List of numerical column names
    
    Returns:
    --------
    dict : Dictionary containing processed numerical features and scaler
    """
    print("Processing numerical features...")
    
    # Handle missing values
    df_clean = df.copy()
    for col in numerical_columns:
        if col in df.columns:
            # Fill missing values with median
            median_val = df[col].median()
            df_clean[col] = df[col].fillna(median_val)
            print(f"Filled {df[col].isnull().sum()} missing values in '{col}' with median: {median_val:.2f}")
    
    # Handle outliers
    df_processed, outlier_info = handle_outliers(df_clean, numerical_columns, method='cap')
    
    # Create derived numerical features
    derived_features = pd.DataFrame(index=df.index)
    
    # Price-based features
    if 'price_numeric' in df.columns:
        price_col = df_processed['price_numeric']
        
        # Price categories
        derived_features['price_category_low'] = (price_col <= price_col.quantile(0.33)).astype(int)
        derived_features['price_category_medium'] = ((price_col > price_col.quantile(0.33)) & 
                                                   (price_col <= price_col.quantile(0.67))).astype(int)
        derived_features['price_category_high'] = (price_col > price_col.quantile(0.67)).astype(int)
        
        # Log price for better distribution
        derived_features['log_price'] = np.log1p(price_col)
    
    # Rating-based features
    if 'rating_numeric' in df.columns:
        rating_col = df_processed['rating_numeric']
        
        # Rating categories
        derived_features['high_rating'] = (rating_col >= 4.0).astype(int)
        derived_features['low_rating'] = (rating_col <= 2.5).astype(int)
        derived_features['rating_squared'] = rating_col ** 2
    
    # Review count features
    if 'reviews_count' in df.columns:
        reviews_col = df_processed['reviews_count']
        
        # Review count categories
        derived_features['high_reviews'] = (reviews_col >= reviews_col.quantile(0.75)).astype(int)
        derived_features['low_reviews'] = (reviews_col <= reviews_col.quantile(0.25)).astype(int)
        derived_features['log_reviews'] = np.log1p(reviews_col)
    
    # Cross-feature ratios
    if 'price_numeric' in df.columns and 'rating_numeric' in df.columns:
        # Price per rating point
        derived_features['price_per_rating'] = df_processed['price_numeric'] / (df_processed['rating_numeric'] + 0.1)
    
    if 'price_numeric' in df.columns and 'reviews_count' in df.columns:
        # Price per review
        derived_features['price_per_review'] = df_processed['price_numeric'] / (df_processed['reviews_count'] + 1)
    
    if 'rating_numeric' in df.columns and 'reviews_count' in df.columns:
        # Rating reliability (rating weighted by review count)
        derived_features['rating_reliability'] = df_processed['rating_numeric'] * np.log1p(df_processed['reviews_count'])
    
    # Combine original and derived features
    all_numerical_features = pd.concat([
        df_processed[numerical_columns],
        derived_features
    ], axis=1)
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_numerical_features)
    
    # Create DataFrame with scaled features
    scaled_df = pd.DataFrame(
        scaled_features,
        columns=all_numerical_features.columns,
        index=df.index
    )
    
    print(f"Created {len(numerical_columns)} original + {derived_features.shape[1]} derived = {all_numerical_features.shape[1]} numerical features")
    print(f"Derived features: {list(derived_features.columns)}")
    
    return {
        'features': scaled_df,
        'scaler': scaler,
        'feature_names': list(all_numerical_features.columns),
        'outlier_info': outlier_info,
        'original_columns': numerical_columns
    }

def validate_numerical_features(numerical_data, feature_names):
    """
    Validate numerical features for quality and distribution.
    
    Parameters:
    -----------
    numerical_data : pandas.DataFrame or numpy.array
        Numerical feature data
    feature_names : list
        List of feature names
    
    Returns:
    --------
    dict : Validation results
    """
    if isinstance(numerical_data, np.ndarray):
        numerical_data = pd.DataFrame(numerical_data, columns=feature_names)
    
    validation_results = {}
    
    for col in numerical_data.columns:
        col_data = numerical_data[col]
        
        validation_results[col] = {
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'null_count': col_data.isnull().sum(),
            'infinite_count': np.isinf(col_data).sum(),
            'zero_variance': col_data.std() == 0
        }
    
    # Summary statistics
    total_features = len(feature_names)
    zero_variance_features = sum(1 for v in validation_results.values() if v['zero_variance'])
    features_with_nulls = sum(1 for v in validation_results.values() if v['null_count'] > 0)
    features_with_inf = sum(1 for v in validation_results.values() if v['infinite_count'] > 0)
    
    print(f"\nNumerical Features Validation:")
    print(f"Total features: {total_features}")
    print(f"Zero variance features: {zero_variance_features}")
    print(f"Features with null values: {features_with_nulls}")
    print(f"Features with infinite values: {features_with_inf}")
    
    if zero_variance_features > 0:
        zero_var_cols = [col for col, v in validation_results.items() if v['zero_variance']]
        print(f"Zero variance columns: {zero_var_cols}")
    
    return validation_results

# ============================================================================
# TASK 2.3: CATEGORICAL FEATURE ENCODING
# ============================================================================

def create_onehot_encoding(df, categorical_columns, handle_unknown='ignore'):
    """
    Create one-hot encoding for categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    categorical_columns : list
        List of categorical column names for one-hot encoding
    handle_unknown : str
        How to handle unknown categories ('ignore', 'error')
    
    Returns:
    --------
    dict : Dictionary containing encoded features and encoder
    """
    from sklearn.preprocessing import OneHotEncoder
    
    # Handle missing values
    df_clean = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            df_clean[col] = df[col].fillna('unknown').astype(str)
    
    # Initialize and fit encoder
    encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=True, dtype=np.int32)
    
    # Fit and transform data
    encoded_data = encoder.fit_transform(df_clean[categorical_columns])
    
    # Create feature names
    feature_names = []
    for i, col in enumerate(categorical_columns):
        categories = encoder.categories_[i]
        for category in categories:
            feature_names.append(f'{col}_{category}')
    
    print(f"One-hot encoded {len(categorical_columns)} columns into {encoded_data.shape[1]} features")
    print(f"Columns: {categorical_columns}")
    
    return {
        'features': encoded_data,
        'encoder': encoder,
        'feature_names': feature_names,
        'original_columns': categorical_columns
    }

def create_label_encoding(df, categorical_columns, handle_unknown='use_encoded_value', unknown_value=-1):
    """
    Create label encoding for categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    categorical_columns : list
        List of categorical column names for label encoding
    handle_unknown : str
        How to handle unknown categories
    unknown_value : int
        Value to use for unknown categories
    
    Returns:
    --------
    dict : Dictionary containing encoded features and encoders
    """
    encoded_features = pd.DataFrame(index=df.index)
    encoders = {}
    feature_names = []
    
    for col in categorical_columns:
        if col in df.columns:
            # Handle missing values
            col_data = df[col].fillna('unknown').astype(str)
            
            # Initialize encoder
            encoder = LabelEncoder()
            
            # Fit encoder
            encoder.fit(col_data)
            
            # Transform data
            encoded_col = encoder.transform(col_data)
            
            # Store results
            encoded_features[f'{col}_encoded'] = encoded_col
            encoders[col] = encoder
            feature_names.append(f'{col}_encoded')
            
            print(f"Label encoded '{col}': {len(encoder.classes_)} unique categories")
    
    return {
        'features': encoded_features,
        'encoders': encoders,
        'feature_names': feature_names,
        'original_columns': categorical_columns
    }

def create_frequency_encoding(df, categorical_columns):
    """
    Create frequency encoding for categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    categorical_columns : list
        List of categorical column names for frequency encoding
    
    Returns:
    --------
    dict : Dictionary containing encoded features and frequency maps
    """
    encoded_features = pd.DataFrame(index=df.index)
    frequency_maps = {}
    feature_names = []
    
    for col in categorical_columns:
        if col in df.columns:
            # Handle missing values
            col_data = df[col].fillna('unknown').astype(str)
            
            # Calculate frequency mapping
            freq_map = col_data.value_counts().to_dict()
            
            # Apply frequency encoding
            encoded_col = col_data.map(freq_map)
            
            # Store results
            encoded_features[f'{col}_freq'] = encoded_col
            frequency_maps[col] = freq_map
            feature_names.append(f'{col}_freq')
            
            print(f"Frequency encoded '{col}': {len(freq_map)} unique categories")
    
    return {
        'features': encoded_features,
        'frequency_maps': frequency_maps,
        'feature_names': feature_names,
        'original_columns': categorical_columns
    }

# ============================================================================
# TASK 5.1: CREATE PARAMETER GRID DEFINITIONS
# ============================================================================

def create_comprehensive_param_grids():
    """
    Define comprehensive parameter grids for each algorithm.
    
    Returns:
    --------
    dict : Dictionary containing parameter grids for each model type
    """
    
    # Random Forest parameter grid
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 10],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'random_state': [42]
    }
    
    # SVM parameter grid
    svm_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'degree': [2, 3, 4, 5],  # Only for poly kernel
        'coef0': [0.0, 0.1, 0.5, 1.0],  # For poly and sigmoid
        'class_weight': [None, 'balanced'],
        'probability': [True],  # Enable probability estimates
        'random_state': [42]
    }
    
    # Logistic Regression parameter grid
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # Only for elasticnet
        'max_iter': [100, 500, 1000, 2000, 5000],
        'class_weight': [None, 'balanced'],
        'random_state': [42]
    }
    
    # Gradient Boosting parameter grid
    gb_param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 10],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'random_state': [42]
    }
    
    # Compact grids for faster tuning (reduced parameter space)
    compact_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'class_weight': [None, 'balanced'],
            'probability': [True],
            'random_state': [42]
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [42]
        }
    }
    
    # Full comprehensive grids
    comprehensive_grids = {
        'random_forest': rf_param_grid,
        'svm': svm_param_grid,
        'logistic_regression': lr_param_grid,
        'gradient_boosting': gb_param_grid
    }
    
    return {
        'comprehensive': comprehensive_grids,
        'compact': compact_grids
    }

def generate_parameter_combinations(param_grid, max_combinations=None):
    """
    Generate parameter combinations from a parameter grid.
    
    Parameters:
    -----------
    param_grid : dict
        Parameter grid dictionary
    max_combinations : int, optional
        Maximum number of combinations to generate
    
    Returns:
    --------
    list : List of parameter dictionaries
    """
    from itertools import product
    
    # Get all parameter names and values
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all combinations
    combinations = []
    for combo in product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
        
        if max_combinations and len(combinations) >= max_combinations:
            break
    
    print(f"Generated {len(combinations)} parameter combinations")
    if max_combinations and len(combinations) == max_combinations:
        print(f"Limited to maximum of {max_combinations} combinations")
    
    return combinations

def validate_parameter_constraints(param_dict, model_type):
    """
    Validate parameter constraints and compatibility.
    
    Parameters:
    -----------
    param_dict : dict
        Parameter dictionary to validate
    model_type : str
        Type of model ('random_forest', 'svm', 'logistic_regression', 'gradient_boosting')
    
    Returns:
    --------
    tuple : (is_valid, error_message)
    """
    
    if model_type == 'svm':
        # SVM-specific constraints
        if param_dict.get('kernel') == 'linear' and 'gamma' in param_dict:
            if param_dict['gamma'] not in ['scale', 'auto']:
                return False, "Linear kernel doesn't use gamma parameter (except 'scale' or 'auto')"
        
        if param_dict.get('kernel') not in ['poly', 'sigmoid'] and 'coef0' in param_dict:
            if param_dict['coef0'] != 0.0:
                return False, "coef0 is only used for poly and sigmoid kernels"
        
        if param_dict.get('kernel') != 'poly' and 'degree' in param_dict:
            if param_dict['degree'] != 3:
                return False, "degree is only used for poly kernel"
    
    elif model_type == 'logistic_regression':
        # Logistic Regression constraints
        penalty = param_dict.get('penalty')
        solver = param_dict.get('solver')
        
        # Solver-penalty compatibility
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            return False, "L1 penalty only supported by liblinear and saga solvers"
        
        if penalty == 'elasticnet' and solver != 'saga':
            return False, "Elasticnet penalty only supported by saga solver"
        
        if penalty == 'elasticnet' and 'l1_ratio' not in param_dict:
            return False, "l1_ratio must be specified for elasticnet penalty"
        
        if penalty != 'elasticnet' and 'l1_ratio' in param_dict:
            return False, "l1_ratio only used with elasticnet penalty"
        
        if penalty is None and solver == 'liblinear':
            return False, "liblinear solver doesn't support no penalty"
    
    elif model_type == 'random_forest':
        # Random Forest constraints
        if param_dict.get('bootstrap') is False and param_dict.get('class_weight') == 'balanced_subsample':
            return False, "balanced_subsample class_weight requires bootstrap=True"
    
    elif model_type == 'gradient_boosting':
        # Gradient Boosting constraints
        if param_dict.get('subsample') <= 0 or param_dict.get('subsample') > 1:
            return False, "subsample must be in (0, 1] range"
        
        if param_dict.get('learning_rate') <= 0:
            return False, "learning_rate must be positive"
    
    return True, "Valid parameters"

def filter_valid_parameters(param_combinations, model_type):
    """
    Filter parameter combinations to keep only valid ones.
    
    Parameters:
    -----------
    param_combinations : list
        List of parameter dictionaries
    model_type : str
        Type of model
    
    Returns:
    --------
    list : List of valid parameter dictionaries
    """
    valid_combinations = []
    invalid_count = 0
    
    for param_dict in param_combinations:
        is_valid, error_msg = validate_parameter_constraints(param_dict, model_type)
        if is_valid:
            valid_combinations.append(param_dict)
        else:
            invalid_count += 1
    
    print(f"Filtered {len(param_combinations)} combinations:")
    print(f"  Valid: {len(valid_combinations)}")
    print(f"  Invalid: {invalid_count}")
    
    return valid_combinations

def estimate_search_time(param_combinations, cv_folds=5, base_time_per_fit=10):
    """
    Estimate time required for hyperparameter search.
    
    Parameters:
    -----------
    param_combinations : list or int
        Number of parameter combinations or list of combinations
    cv_folds : int
        Number of cross-validation folds
    base_time_per_fit : float
        Estimated time per model fit in seconds
    
    Returns:
    --------
    dict : Time estimates in different units
    """
    if isinstance(param_combinations, list):
        n_combinations = len(param_combinations)
    else:
        n_combinations = param_combinations
    
    total_fits = n_combinations * cv_folds
    total_seconds = total_fits * base_time_per_fit
    
    time_estimates = {
        'total_fits': total_fits,
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'total_hours': total_seconds / 3600,
        'combinations': n_combinations,
        'cv_folds': cv_folds
    }
    
    print(f"Hyperparameter Search Time Estimate:")
    print(f"  Parameter combinations: {n_combinations}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Total model fits: {total_fits}")
    print(f"  Estimated time: {time_estimates['total_minutes']:.1f} minutes ({time_estimates['total_hours']:.2f} hours)")
    
    return time_estimates

def create_adaptive_param_grid(model_type, search_budget='medium'):
    """
    Create adaptive parameter grid based on search budget.
    
    Parameters:
    -----------
    model_type : str
        Type of model
    search_budget : str
        Search budget ('small', 'medium', 'large', 'comprehensive')
    
    Returns:
    --------
    dict : Adaptive parameter grid
    """
    all_grids = create_comprehensive_param_grids()
    
    if search_budget == 'small':
        # Very limited grid for quick testing
        small_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'random_state': [42]
            },
            'svm': {
                'C': [1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
                'random_state': [42]
            },
            'logistic_regression': {
                'C': [1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'random_state': [42]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
                'random_state': [42]
            }
        }
        return small_grids.get(model_type, {})
    
    elif search_budget == 'medium':
        return all_grids['compact'].get(model_type, {})
    
    elif search_budget == 'large':
        # Expanded compact grid
        base_grid = all_grids['compact'].get(model_type, {})
        # Add more parameter values for larger search
        if model_type == 'random_forest':
            base_grid['n_estimators'] = [50, 100, 200, 300, 500]
            base_grid['max_depth'] = [5, 10, 15, 20, None]
        elif model_type == 'svm':
            base_grid['C'] = [0.01, 0.1, 1, 10, 100, 1000]
            base_grid['gamma'] = ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        return base_grid
    
    elif search_budget == 'comprehensive':
        return all_grids['comprehensive'].get(model_type, {})
    
    else:
        raise ValueError(f"Unknown search budget: {search_budget}")

print("Parameter grid definitions created successfully!")
print("Available functions:")
print("- create_comprehensive_param_grids()")
print("- generate_parameter_combinations()")
print("- validate_parameter_constraints()")
print("- filter_valid_parameters()")
print("- estimate_search_time()")
print("- create_adaptive_param_grid()")

def handle_unseen_categories(new_data, encoders, encoding_type='onehot'):
    """
    Handle unseen categories in new data using fitted encoders.
    
    Parameters:
    -----------
    new_data : pandas.DataFrame
        New data with potential unseen categories
    encoders : dict
        Dictionary containing fitted encoders
    encoding_type : str
        Type of encoding ('onehot', 'label', 'frequency')
    
    Returns:
    --------
    encoded_data : array-like
        Encoded data with unseen categories handled
    """
    if encoding_type == 'onehot':
        # Handle missing values
        new_data_clean = new_data.copy()
        for col in encoders['original_columns']:
            new_data_clean[col] = new_data[col].fillna('unknown').astype(str)
        
        # Transform using fitted encoder
        encoded_data = encoders['encoder'].transform(new_data_clean[encoders['original_columns']])
        return encoded_data
    
    elif encoding_type == 'label':
        encoded_features = pd.DataFrame(index=new_data.index)
        
        for col in encoders['original_columns']:
            if col in new_data.columns:
                col_data = new_data[col].fillna('unknown').astype(str)
                encoder = encoders['encoders'][col]
                
                # Handle unseen categories
                encoded_col = []
                for value in col_data:
                    if value in encoder.classes_:
                        encoded_col.append(encoder.transform([value])[0])
                    else:
                        encoded_col.append(-1)  # Unknown category value
                
                encoded_features[f'{col}_encoded'] = encoded_col
        
        return encoded_features
    
    elif encoding_type == 'frequency':
        encoded_features = pd.DataFrame(index=new_data.index)
        
        for col in encoders['original_columns']:
            if col in new_data.columns:
                col_data = new_data[col].fillna('unknown').astype(str)
                freq_map = encoders['frequency_maps'][col]
                
                # Handle unseen categories with frequency 0
                encoded_col = col_data.map(freq_map).fillna(0)
                encoded_features[f'{col}_freq'] = encoded_col
        
        return encoded_features

def create_categorical_features(df, onehot_columns=['platform', 'product_type'], 
                              label_columns=['seller_name'], frequency_columns=None):
    """
    Create all categorical features using different encoding methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    onehot_columns : list
        Columns for one-hot encoding
    label_columns : list
        Columns for label encoding
    frequency_columns : list
        Columns for frequency encoding
    
    Returns:
    --------
    dict : Dictionary containing all categorical features and encoders
    """
    print("Processing categorical features...")
    
    categorical_features = {}
    all_feature_names = []
    
    # One-hot encoding
    if onehot_columns:
        available_onehot = [col for col in onehot_columns if col in df.columns]
        if available_onehot:
            onehot_result = create_onehot_encoding(df, available_onehot)
            categorical_features['onehot'] = onehot_result
            all_feature_names.extend(onehot_result['feature_names'])
    
    # Label encoding
    if label_columns:
        available_label = [col for col in label_columns if col in df.columns]
        if available_label:
            label_result = create_label_encoding(df, available_label)
            categorical_features['label'] = label_result
            all_feature_names.extend(label_result['feature_names'])
    
    # Frequency encoding
    if frequency_columns:
        available_freq = [col for col in frequency_columns if col in df.columns]
        if available_freq:
            freq_result = create_frequency_encoding(df, available_freq)
            categorical_features['frequency'] = freq_result
            all_feature_names.extend(freq_result['feature_names'])
    
    # Combine all categorical features
    feature_matrices = []
    
    # Add one-hot features (sparse)
    if 'onehot' in categorical_features:
        feature_matrices.append(categorical_features['onehot']['features'])
    
    # Add label features (dense, convert to sparse)
    if 'label' in categorical_features:
        label_sparse = csr_matrix(categorical_features['label']['features'].values)
        feature_matrices.append(label_sparse)
    
    # Add frequency features (dense, convert to sparse)
    if 'frequency' in categorical_features:
        freq_sparse = csr_matrix(categorical_features['frequency']['features'].values)
        feature_matrices.append(freq_sparse)
    
    # Combine all features
    if feature_matrices:
        combined_matrix = hstack(feature_matrices)
    else:
        # Create empty sparse matrix if no categorical features
        combined_matrix = csr_matrix((len(df), 0))
    
    print(f"Created {combined_matrix.shape[1]} categorical features")
    print(f"Feature breakdown:")
    if 'onehot' in categorical_features:
        print(f"  - One-hot features: {categorical_features['onehot']['features'].shape[1]}")
    if 'label' in categorical_features:
        print(f"  - Label features: {categorical_features['label']['features'].shape[1]}")
    if 'frequency' in categorical_features:
        print(f"  - Frequency features: {categorical_features['frequency']['features'].shape[1]}")
    
    return {
        'matrix': combined_matrix,
        'feature_names': all_feature_names,
        'encoders': categorical_features,
        'encoding_info': {
            'onehot_columns': onehot_columns,
            'label_columns': label_columns,
            'frequency_columns': frequency_columns or []
        }
    }

# ============================================================================
# TASK 2.4: FEATURE CORRELATION ANALYSIS AND REMOVAL
# ============================================================================

def calculate_correlation_matrix(feature_matrix, feature_names, method='pearson'):
    """
    Calculate correlation matrix for features.
    
    Parameters:
    -----------
    feature_matrix : numpy.array or pandas.DataFrame or sparse matrix
        Feature matrix
    feature_names : list
        List of feature names
    method : str
        Correlation method ('pearson', 'spearman')
    
    Returns:
    --------
    pandas.DataFrame : Correlation matrix
    """
    # Convert sparse matrix to dense if needed
    if hasattr(feature_matrix, 'toarray'):
        feature_matrix = feature_matrix.toarray()
    
    # Create DataFrame if needed
    if not isinstance(feature_matrix, pd.DataFrame):
        df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    else:
        df_features = feature_matrix
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df_features.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df_features.corr(method='spearman')
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    print(f"Calculated {method} correlation matrix for {len(feature_names)} features")
    
    return corr_matrix

def find_highly_correlated_features(corr_matrix, threshold=0.95):
    """
    Find pairs of highly correlated features.
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    threshold : float
        Correlation threshold for identifying highly correlated features
    
    Returns:
    --------
    list : List of tuples containing highly correlated feature pairs
    """
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find highly correlated pairs
    highly_correlated = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_value = upper_triangle.loc[idx, col]
            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                highly_correlated.append((idx, col, corr_value))
    
    print(f"Found {len(highly_correlated)} feature pairs with correlation >= {threshold}")
    
    return highly_correlated

def select_features_to_remove(highly_correlated_pairs, feature_importance=None):
    """
    Select which features to remove from highly correlated pairs.
    
    Parameters:
    -----------
    highly_correlated_pairs : list
        List of tuples with highly correlated feature pairs
    feature_importance : dict, optional
        Dictionary mapping feature names to importance scores
    
    Returns:
    --------
    set : Set of feature names to remove
    """
    features_to_remove = set()
    
    for feat1, feat2, corr_value in highly_correlated_pairs:
        # If we have feature importance, keep the more important feature
        if feature_importance:
            imp1 = feature_importance.get(feat1, 0)
            imp2 = feature_importance.get(feat2, 0)
            
            if imp1 >= imp2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        else:
            # Default: remove the second feature (arbitrary choice)
            features_to_remove.add(feat2)
    
    print(f"Selected {len(features_to_remove)} features for removal")
    
    return features_to_remove

def remove_correlated_features(feature_matrix, feature_names, features_to_remove):
    """
    Remove highly correlated features from feature matrix.
    
    Parameters:
    -----------
    feature_matrix : numpy.array or sparse matrix
        Feature matrix
    feature_names : list
        List of feature names
    features_to_remove : set
        Set of feature names to remove
    
    Returns:
    --------
    dict : Dictionary containing filtered features and names
    """
    # Find indices of features to keep
    indices_to_keep = [i for i, name in enumerate(feature_names) if name not in features_to_remove]
    remaining_feature_names = [name for name in feature_names if name not in features_to_remove]
    
    # Filter feature matrix
    if hasattr(feature_matrix, 'toarray'):  # Sparse matrix
        filtered_matrix = feature_matrix[:, indices_to_keep]
    else:  # Dense matrix
        filtered_matrix = feature_matrix[:, indices_to_keep]
    
    print(f"Removed {len(features_to_remove)} features, kept {len(remaining_feature_names)} features")
    
    return {
        'matrix': filtered_matrix,
        'feature_names': remaining_feature_names,
        'removed_features': list(features_to_remove),
        'kept_indices': indices_to_keep
    }

def analyze_feature_correlations(feature_matrix, feature_names, threshold=0.95, 
                               feature_importance=None, create_heatmap=True):
    """
    Comprehensive feature correlation analysis.
    
    Parameters:
    -----------
    feature_matrix : numpy.array or sparse matrix
        Feature matrix
    feature_names : list
        List of feature names
    threshold : float
        Correlation threshold for removal
    feature_importance : dict, optional
        Feature importance scores
    create_heatmap : bool
        Whether to create correlation heatmap
    
    Returns:
    --------
    dict : Dictionary containing correlation analysis results
    """
    print(f"Starting correlation analysis for {len(feature_names)} features...")
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(feature_matrix, feature_names)
    
    # Find highly correlated features
    highly_correlated = find_highly_correlated_features(corr_matrix, threshold)
    
    # Select features to remove
    features_to_remove = select_features_to_remove(highly_correlated, feature_importance)
    
    # Remove correlated features
    filtered_result = remove_correlated_features(feature_matrix, feature_names, features_to_remove)
    
    # Create heatmap if requested and feasible
    heatmap_created = False
    if create_heatmap and len(feature_names) <= 50:  # Only for manageable number of features
        try:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title(f'Feature Correlation Matrix (threshold={threshold})')
            plt.tight_layout()
            plt.show()
            heatmap_created = True
            print("Created correlation heatmap")
        except Exception as e:
            print(f"Could not create heatmap: {e}")
    
    # Summary statistics
    correlation_stats = {
        'total_features': len(feature_names),
        'highly_correlated_pairs': len(highly_correlated),
        'features_removed': len(features_to_remove),
        'features_remaining': len(filtered_result['feature_names']),
        'max_correlation': corr_matrix.abs().max().max() if not corr_matrix.empty else 0,
        'mean_correlation': corr_matrix.abs().mean().mean() if not corr_matrix.empty else 0
    }
    
    print(f"\nCorrelation Analysis Summary:")
    print(f"Total features: {correlation_stats['total_features']}")
    print(f"Highly correlated pairs (>={threshold}): {correlation_stats['highly_correlated_pairs']}")
    print(f"Features removed: {correlation_stats['features_removed']}")
    print(f"Features remaining: {correlation_stats['features_remaining']}")
    print(f"Max correlation: {correlation_stats['max_correlation']:.3f}")
    print(f"Mean correlation: {correlation_stats['mean_correlation']:.3f}")
    
    return {
        'correlation_matrix': corr_matrix,
        'highly_correlated_pairs': highly_correlated,
        'filtered_features': filtered_result,
        'correlation_stats': correlation_stats,
        'heatmap_created': heatmap_created
    }

def create_correlation_report(correlation_results, save_path=None):
    """
    Create a detailed correlation analysis report.
    
    Parameters:
    -----------
    correlation_results : dict
        Results from analyze_feature_correlations
    save_path : str, optional
        Path to save the report
    
    Returns:
    --------
    str : Correlation analysis report
    """
    stats = correlation_results['correlation_stats']
    highly_corr = correlation_results['highly_correlated_pairs']
    
    report = f"""
# Feature Correlation Analysis Report

## Summary Statistics
- **Total Features**: {stats['total_features']}
- **Highly Correlated Pairs**: {stats['highly_correlated_pairs']}
- **Features Removed**: {stats['features_removed']}
- **Features Remaining**: {stats['features_remaining']}
- **Maximum Correlation**: {stats['max_correlation']:.3f}
- **Mean Correlation**: {stats['mean_correlation']:.3f}

## Highly Correlated Feature Pairs
"""
    
    if highly_corr:
        report += "\n| Feature 1 | Feature 2 | Correlation |\n"
        report += "|-----------|-----------|-------------|\n"
        for feat1, feat2, corr in highly_corr[:20]:  # Show top 20
            report += f"| {feat1} | {feat2} | {corr:.3f} |\n"
        
        if len(highly_corr) > 20:
            report += f"\n... and {len(highly_corr) - 20} more pairs\n"
    else:
        report += "\nNo highly correlated feature pairs found.\n"
    
    report += f"""
## Removed Features
{correlation_results['filtered_features']['removed_features']}

## Recommendations
- {'✓' if stats['features_removed'] > 0 else '✗'} Multicollinearity reduction: {stats['features_removed']} features removed
- {'✓' if stats['mean_correlation'] < 0.3 else '⚠️'} Average correlation level: {stats['mean_correlation']:.3f}
- {'✓' if stats['max_correlation'] < 0.95 else '⚠️'} Maximum correlation: {stats['max_correlation']:.3f}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Correlation report saved to: {save_path}")
    
    return report

# ============================================================================
# TASK 2.5: COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================

def validate_input_data(df, required_columns=None):
    """
    Validate input data quality and structure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to validate
    required_columns : list, optional
        List of required column names
    
    Returns:
    --------
    dict : Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'data_info': {}
    }
    
    # Basic data info
    validation_results['data_info'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Check for required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            validation_results['is_valid'] = False
    
    # Check for empty dataframe
    if df.empty:
        validation_results['errors'].append("Dataframe is empty")
        validation_results['is_valid'] = False
    
    # Check for excessive missing values
    high_null_cols = []
    for col, null_count in validation_results['data_info']['null_counts'].items():
        null_percentage = (null_count / len(df)) * 100
        if null_percentage > 50:
            high_null_cols.append(f"{col} ({null_percentage:.1f}%)")
    
    if high_null_cols:
        validation_results['warnings'].append(f"Columns with >50% missing values: {high_null_cols}")
    
    # Check for duplicate rows
    if validation_results['data_info']['duplicate_rows'] > 0:
        dup_percentage = (validation_results['data_info']['duplicate_rows'] / len(df)) * 100
        validation_results['warnings'].append(f"Found {validation_results['data_info']['duplicate_rows']} duplicate rows ({dup_percentage:.1f}%)")
    
    print(f"Data validation completed:")
    print(f"  Shape: {validation_results['data_info']['shape']}")
    print(f"  Errors: {len(validation_results['errors'])}")
    print(f"  Warnings: {len(validation_results['warnings'])}")
    
    if validation_results['errors']:
        print("  Errors found:")
        for error in validation_results['errors']:
            print(f"    - {error}")
    
    if validation_results['warnings']:
        print("  Warnings:")
        for warning in validation_results['warnings']:
            print(f"    - {warning}")
    
    return validation_results

class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for HP product classification.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature engineering pipeline.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary for feature engineering
        """
        self.config = config or FEATURE_CONFIG
        self.is_fitted = False
        
        # Store fitted components
        self.text_vectorizers = {}
        self.numerical_scaler = None
        self.categorical_encoders = {}
        self.feature_names = []
        self.correlation_info = {}
        
        # Feature metadata
        self.feature_metadata = {
            'text_features': [],
            'numerical_features': [],
            'categorical_features': [],
            'total_features': 0
        }
    
    def fit(self, df, target_column='target_is_original', 
            text_columns=['title', 'description'],
            numerical_columns=['price_numeric', 'rating_numeric', 'reviews_count'],
            onehot_columns=['platform', 'product_type'],
            label_columns=['seller_name']):
        """
        Fit the feature engineering pipeline on training data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Training dataframe
        target_column : str
            Name of target column
        text_columns : list
            Text columns for feature extraction
        numerical_columns : list
            Numerical columns for preprocessing
        onehot_columns : list
            Categorical columns for one-hot encoding
        label_columns : list
            Categorical columns for label encoding
        
        Returns:
        --------
        self : FeatureEngineeringPipeline
        """
        print("=" * 60)
        print("FITTING FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        # Validate input data
        required_cols = text_columns + numerical_columns + onehot_columns + label_columns + [target_column]
        validation_results = validate_input_data(df, required_cols)
        
        if not validation_results['is_valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Store column configurations
        self.text_columns = text_columns
        self.numerical_columns = numerical_columns
        self.onehot_columns = onehot_columns
        self.label_columns = label_columns
        self.target_column = target_column
        
        # 1. Extract text features
        print("\n1. EXTRACTING TEXT FEATURES")
        print("-" * 40)
        text_features = extract_all_text_features(df, text_columns)
        self.text_vectorizers = text_features['tfidf_vectorizers']
        self.feature_metadata['text_features'] = text_features['feature_names']
        
        # 2. Process numerical features
        print("\n2. PROCESSING NUMERICAL FEATURES")
        print("-" * 40)
        numerical_features = create_numerical_features(df, numerical_columns)
        self.numerical_scaler = numerical_features['scaler']
        self.feature_metadata['numerical_features'] = numerical_features['feature_names']
        
        # 3. Encode categorical features
        print("\n3. ENCODING CATEGORICAL FEATURES")
        print("-" * 40)
        categorical_features = create_categorical_features(
            df, onehot_columns, label_columns
        )
        self.categorical_encoders = categorical_features['encoders']
        self.feature_metadata['categorical_features'] = categorical_features['feature_names']
        
        # 4. Combine all features
        print("\n4. COMBINING ALL FEATURES")
        print("-" * 40)
        all_matrices = [
            text_features['matrix'],
            csr_matrix(numerical_features['features'].values),
            categorical_features['matrix']
        ]
        
        combined_matrix = hstack(all_matrices)
        all_feature_names = (text_features['feature_names'] + 
                           numerical_features['feature_names'] + 
                           categorical_features['feature_names'])
        
        # 5. Correlation analysis and feature selection
        print("\n5. CORRELATION ANALYSIS AND FEATURE SELECTION")
        print("-" * 40)
        correlation_results = analyze_feature_correlations(
            combined_matrix, 
            all_feature_names,
            threshold=self.config['correlation_threshold'],
            create_heatmap=False  # Skip heatmap for large feature sets
        )
        
        # Store final features
        self.final_feature_matrix = correlation_results['filtered_features']['matrix']
        self.feature_names = correlation_results['filtered_features']['feature_names']
        self.correlation_info = correlation_results
        
        # Update metadata
        self.feature_metadata['total_features'] = len(self.feature_names)
        self.feature_metadata['removed_features'] = correlation_results['filtered_features']['removed_features']
        
        # Extract target variable
        self.target = df[target_column].values
        
        # Mark as fitted
        self.is_fitted = True
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE FITTED SUCCESSFULLY")
        print("=" * 60)
        print(f"Final feature matrix shape: {self.final_feature_matrix.shape}")
        print(f"Total features: {self.feature_metadata['total_features']}")
        print(f"Text features: {len(self.feature_metadata['text_features'])}")
        print(f"Numerical features: {len(self.feature_metadata['numerical_features'])}")
        print(f"Categorical features: {len(self.feature_metadata['categorical_features'])}")
        print(f"Removed features: {len(self.feature_metadata['removed_features'])}")
        
        return self
    
    def transform(self, df):
        """
        Transform new data using fitted pipeline.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            New dataframe to transform
        
        Returns:
        --------
        scipy.sparse.csr_matrix : Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        print("Transforming new data using fitted pipeline...")
        
        # 1. Transform text features
        text_matrices = []
        text_feature_names = []
        
        for col in self.text_columns:
            if col in df.columns:
                # Text length and keyword features
                text_length = create_text_length_features(df, [col])
                keyword_features = create_keyword_features(df, [col])
                dense_text = pd.concat([text_length, keyword_features], axis=1)
                text_matrices.append(csr_matrix(dense_text.values))
                text_feature_names.extend(dense_text.columns)
                
                # TF-IDF features
                if f'{col}_tfidf' in self.text_vectorizers:
                    text_data = df[col].fillna('').astype(str)
                    tfidf_matrix = self.text_vectorizers[f'{col}_tfidf'].transform(text_data)
                    text_matrices.append(tfidf_matrix)
                    tfidf_names = [f'{col}_tfidf_{name}' for name in self.text_vectorizers[f'{col}_tfidf'].get_feature_names_out()]
                    text_feature_names.extend(tfidf_names)
        
        # 2. Transform numerical features
        # Handle missing values
        df_clean = df.copy()
        for col in self.numerical_columns:
            if col in df.columns:
                median_val = df[col].median()
                df_clean[col] = df[col].fillna(median_val)
        
        # Handle outliers (cap method)
        df_processed, _ = handle_outliers(df_clean, self.numerical_columns, method='cap')
        
        # Create derived features (same logic as in fit)
        derived_features = pd.DataFrame(index=df.index)
        
        if 'price_numeric' in df.columns:
            price_col = df_processed['price_numeric']
            price_quantiles = np.quantile(price_col, [0.33, 0.67])
            derived_features['price_category_low'] = (price_col <= price_quantiles[0]).astype(int)
            derived_features['price_category_medium'] = ((price_col > price_quantiles[0]) & 
                                                       (price_col <= price_quantiles[1])).astype(int)
            derived_features['price_category_high'] = (price_col > price_quantiles[1]).astype(int)
            derived_features['log_price'] = np.log1p(price_col)
        
        if 'rating_numeric' in df.columns:
            rating_col = df_processed['rating_numeric']
            derived_features['high_rating'] = (rating_col >= 4.0).astype(int)
            derived_features['low_rating'] = (rating_col <= 2.5).astype(int)
            derived_features['rating_squared'] = rating_col ** 2
        
        if 'reviews_count' in df.columns:
            reviews_col = df_processed['reviews_count']
            reviews_quantiles = np.quantile(reviews_col, [0.25, 0.75])
            derived_features['high_reviews'] = (reviews_col >= reviews_quantiles[1]).astype(int)
            derived_features['low_reviews'] = (reviews_col <= reviews_quantiles[0]).astype(int)
            derived_features['log_reviews'] = np.log1p(reviews_col)
        
        # Cross-feature ratios
        if 'price_numeric' in df.columns and 'rating_numeric' in df.columns:
            derived_features['price_per_rating'] = df_processed['price_numeric'] / (df_processed['rating_numeric'] + 0.1)
        
        if 'price_numeric' in df.columns and 'reviews_count' in df.columns:
            derived_features['price_per_review'] = df_processed['price_numeric'] / (df_processed['reviews_count'] + 1)
        
        if 'rating_numeric' in df.columns and 'reviews_count' in df.columns:
            derived_features['rating_reliability'] = df_processed['rating_numeric'] * np.log1p(df_processed['reviews_count'])
        
        # Combine and scale numerical features
        all_numerical = pd.concat([df_processed[self.numerical_columns], derived_features], axis=1)
        scaled_numerical = self.numerical_scaler.transform(all_numerical)
        
        # 3. Transform categorical features
        categorical_matrices = []
        
        # One-hot encoding
        if 'onehot' in self.categorical_encoders:
            onehot_data = handle_unseen_categories(df, self.categorical_encoders['onehot'], 'onehot')
            categorical_matrices.append(onehot_data)
        
        # Label encoding
        if 'label' in self.categorical_encoders:
            label_data = handle_unseen_categories(df, self.categorical_encoders['label'], 'label')
            categorical_matrices.append(csr_matrix(label_data.values))
        
        # 4. Combine all features
        all_matrices = text_matrices + [csr_matrix(scaled_numerical)] + categorical_matrices
        combined_matrix = hstack(all_matrices)
        
        # 5. Apply correlation-based feature selection
        kept_indices = self.correlation_info['filtered_features']['kept_indices']
        final_matrix = combined_matrix[:, kept_indices]
        
        print(f"Transformed data shape: {final_matrix.shape}")
        
        return final_matrix
    
    def fit_transform(self, df, **fit_params):
        """
        Fit pipeline and transform data in one step.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Training dataframe
        **fit_params : dict
            Parameters for fit method
        
        Returns:
        --------
        tuple : (X, y) - feature matrix and target vector
        """
        self.fit(df, **fit_params)
        X = self.final_feature_matrix
        y = self.target
        return X, y
    
    def get_feature_names(self):
        """Get names of final features."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        return self.feature_names.copy()
    
    def get_feature_metadata(self):
        """Get detailed feature metadata."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        return self.feature_metadata.copy()
    
    def save_pipeline(self, filepath):
        """Save fitted pipeline to file."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'config': self.config,
            'text_vectorizers': self.text_vectorizers,
            'numerical_scaler': self.numerical_scaler,
            'categorical_encoders': self.categorical_encoders,
            'feature_names': self.feature_names,
            'correlation_info': self.correlation_info,
            'feature_metadata': self.feature_metadata,
            'column_configs': {
                'text_columns': self.text_columns,
                'numerical_columns': self.numerical_columns,
                'onehot_columns': self.onehot_columns,
                'label_columns': self.label_columns,
                'target_column': self.target_column
            }
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline saved to: {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath):
        """Load fitted pipeline from file."""
        pipeline_data = joblib.load(filepath)
        
        # Create new instance
        pipeline = cls(config=pipeline_data['config'])
        
        # Restore fitted state
        pipeline.text_vectorizers = pipeline_data['text_vectorizers']
        pipeline.numerical_scaler = pipeline_data['numerical_scaler']
        pipeline.categorical_encoders = pipeline_data['categorical_encoders']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.correlation_info = pipeline_data['correlation_info']
        pipeline.feature_metadata = pipeline_data['feature_metadata']
        
        # Restore column configurations
        col_configs = pipeline_data['column_configs']
        pipeline.text_columns = col_configs['text_columns']
        pipeline.numerical_columns = col_configs['numerical_columns']
        pipeline.onehot_columns = col_configs['onehot_columns']
        pipeline.label_columns = col_configs['label_columns']
        pipeline.target_column = col_configs['target_column']
        
        pipeline.is_fitted = True
        
        print(f"Pipeline loaded from: {filepath}")
        return pipeline

# Convenience function for quick feature engineering
def engineer_features(df, target_column='target_is_original', save_pipeline_path=None):
    """
    Quick feature engineering function using default configuration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Target column name
    save_pipeline_path : str, optional
        Path to save fitted pipeline
    
    Returns:
    --------
    tuple : (X, y, pipeline) - features, target, and fitted pipeline
    """
    pipeline = FeatureEngineeringPipeline()
    X, y = pipeline.fit_transform(df, target_column=target_column)
    
    if save_pipeline_path:
        pipeline.save_pipeline(save_pipeline_path)
    
    return X, y, pipeline

print("\n" + "=" * 60)
print("FEATURE ENGINEERING MODULE IMPLEMENTATION COMPLETE")
print("=" * 60)
print("Available functions:")
print("- extract_all_text_features()")
print("- create_numerical_features()")
print("- create_categorical_features()")
print("- analyze_feature_correlations()")
print("- FeatureEngineeringPipeline class")
print("- engineer_features() - convenience function")
print("\nExample usage:")
print("pipeline = FeatureEngineeringPipeline()")
print("X, y = pipeline.fit_transform(df)")
print("# or")
print("X, y, pipeline = engineer_features(df)")

# ============================================================================
# TASK 3: DATA PREPROCESSING AND SPLITTING
# ============================================================================

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

print("\n" + "=" * 50)
print("TASK 3.1: TRAIN-TEST SPLIT FUNCTIONALITY COMPLETE!")
print("Functions implemented:")
print("- detect_class_imbalance(): Analyze class distribution")
print("- validate_train_test_split(): Validate split maintains distribution")
print("- create_stratified_train_test_split(): Main splitting function")
print("=" * 50)

# ============================================================================
# TASK 3.2: CLASS IMBALANCE HANDLING
# ============================================================================

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

print("\n" + "=" * 50)
print("TASK 3.2: CLASS IMBALANCE HANDLING COMPLETE!")
print("Functions implemented:")
print("- apply_smote_oversampling(): SMOTE oversampling implementation")
print("- calculate_class_weights(): Class weight calculation")
print("- validate_balanced_dataset(): Validate resampling results")
print("- handle_class_imbalance(): Comprehensive imbalance handling")
print("- create_complete_preprocessing_pipeline(): End-to-end preprocessing")
print("=" * 50)

# ============================================================================
# TASK 4.1: RANDOM FOREST TRAINING FUNCTION
# ============================================================================

def train_random_forest(X_train, y_train, X_test=None, y_test=None, 
                       n_estimators=100, max_depth=None, min_samples_split=2,
                       min_samples_leaf=1, max_features='sqrt', random_state=42,
                       cv_folds=5, return_feature_importance=True):
    """
    Train Random Forest classifier with cross-validation scoring.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    min_samples_split : int
        Minimum samples required to split internal node
    min_samples_leaf : int
        Minimum samples required to be at leaf node
    max_features : str or int
        Number of features to consider for best split
    random_state : int
        Random state for reproducibility
    cv_folds : int
        Number of cross-validation folds
    return_feature_importance : bool
        Whether to calculate feature importance
    
    Returns:
    --------
    dict : Dictionary containing trained model, scores, and feature importance
    """
    print("Training Random Forest Classifier...")
    
    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        rf_model, X_train, y_train, 
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=-1
    )
    
    # Calculate additional CV metrics
    cv_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(
            rf_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring=metric,
            n_jobs=-1
        )
        cv_metrics[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Train final model on full training set
    print("Training final model on full training set...")
    rf_model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = rf_model.predict(X_train)
    y_train_proba = rf_model.predict_proba(X_train)[:, 1]
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    # Calculate test metrics if test data provided
    test_metrics = None
    if X_test is not None and y_test is not None:
        print("Evaluating on test set...")
        y_test_pred = rf_model.predict(X_test)
        y_test_proba = rf_model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
    
    # Extract feature importance
    feature_importance = None
    if return_feature_importance:
        feature_importance = {
            'importance_scores': rf_model.feature_importances_,
            'importance_std': np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
        }
    
    # Print results summary
    print(f"\nRandom Forest Training Results:")
    print(f"Cross-validation F1 Score: {cv_metrics['f1']['mean']:.4f} (+/- {cv_metrics['f1']['std']*2:.4f})")
    print(f"Training F1 Score: {train_metrics['f1']:.4f}")
    if test_metrics:
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    
    return {
        'model': rf_model,
        'cv_metrics': cv_metrics,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'model_params': rf_model.get_params()
    }

def extract_rf_feature_importance(rf_results, feature_names=None, top_n=20):
    """
    Extract and visualize Random Forest feature importance.
    
    Parameters:
    -----------
    rf_results : dict
        Results from train_random_forest function
    feature_names : list, optional
        List of feature names
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with feature importance rankings
    """
    if rf_results['feature_importance'] is None:
        print("Feature importance not available in results")
        return None
    
    importance_scores = rf_results['feature_importance']['importance_scores']
    importance_std = rf_results['feature_importance']['importance_std']
    
    # Create feature importance DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores,
        'importance_std': importance_std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Display top features
    print(f"\nTop {top_n} Random Forest Feature Importances:")
    print("=" * 60)
    for i, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:<40} {row['importance']:.4f} (+/- {row['importance_std']:.4f})")
    
    return importance_df

def cross_validate_random_forest(X, y, param_grid=None, cv_folds=5, scoring='f1', random_state=42):
    """
    Perform comprehensive cross-validation for Random Forest with different parameters.
    
    Parameters:
    -----------
    X : array-like or sparse matrix
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Parameter grid for testing different configurations
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for evaluation
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing CV results for different parameter combinations
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    
    print("Performing Random Forest cross-validation with parameter grid...")
    print(f"Parameter grid: {param_grid}")
    
    # Initialize model
    rf_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Extract results
    cv_results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
        'grid_search_object': grid_search
    }
    
    print(f"\nBest Random Forest Parameters:")
    for param, value in cv_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best CV Score ({scoring}): {cv_results['best_score']:.4f}")
    
    return cv_results

print("Random Forest training functions implemented successfully!")

# ============================================================================
# TASK 4.2: SVM TRAINING FUNCTION
# ============================================================================

def train_svm(X_train, y_train, X_test=None, y_test=None,
              C=1.0, kernel='rbf', gamma='scale', probability=True,
              random_state=42, cv_folds=5, class_weight=None):
    """
    Train Support Vector Machine classifier with RBF and linear kernels.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    C : float
        Regularization parameter
    kernel : str
        Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
    gamma : str or float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    probability : bool
        Whether to enable probability estimates
    random_state : int
        Random state for reproducibility
    cv_folds : int
        Number of cross-validation folds
    class_weight : dict or str, optional
        Weights associated with classes
    
    Returns:
    --------
    dict : Dictionary containing trained model, scores, and confidence metrics
    """
    print(f"Training SVM Classifier (kernel={kernel}, C={C})...")
    
    # Initialize SVM
    svm_model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
        class_weight=class_weight
    )
    
    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        svm_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=-1
    )
    
    # Calculate additional CV metrics
    cv_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(
            svm_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring=metric,
            n_jobs=-1
        )
        cv_metrics[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Train final model on full training set
    print("Training final model on full training set...")
    svm_model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = svm_model.predict(X_train)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred)
    }
    
    # Add probability-based metrics if enabled
    if probability:
        y_train_proba = svm_model.predict_proba(X_train)[:, 1]
        train_metrics['roc_auc'] = roc_auc_score(y_train, y_train_proba)
        
        # Calculate prediction confidence statistics
        confidence_stats = {
            'mean_confidence': np.mean(np.max(svm_model.predict_proba(X_train), axis=1)),
            'std_confidence': np.std(np.max(svm_model.predict_proba(X_train), axis=1)),
            'min_confidence': np.min(np.max(svm_model.predict_proba(X_train), axis=1)),
            'max_confidence': np.max(np.max(svm_model.predict_proba(X_train), axis=1))
        }
    else:
        confidence_stats = None
    
    # Calculate test metrics if test data provided
    test_metrics = None
    test_confidence_stats = None
    if X_test is not None and y_test is not None:
        print("Evaluating on test set...")
        y_test_pred = svm_model.predict(X_test)
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }
        
        if probability:
            y_test_proba = svm_model.predict_proba(X_test)[:, 1]
            test_metrics['roc_auc'] = roc_auc_score(y_test, y_test_proba)
            
            test_confidence_stats = {
                'mean_confidence': np.mean(np.max(svm_model.predict_proba(X_test), axis=1)),
                'std_confidence': np.std(np.max(svm_model.predict_proba(X_test), axis=1)),
                'min_confidence': np.min(np.max(svm_model.predict_proba(X_test), axis=1)),
                'max_confidence': np.max(np.max(svm_model.predict_proba(X_test), axis=1))
            }
    
    # Print results summary
    print(f"\nSVM Training Results:")
    print(f"Cross-validation F1 Score: {cv_metrics['f1']['mean']:.4f} (+/- {cv_metrics['f1']['std']*2:.4f})")
    print(f"Training F1 Score: {train_metrics['f1']:.4f}")
    if test_metrics:
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    if confidence_stats:
        print(f"Mean Prediction Confidence: {confidence_stats['mean_confidence']:.4f}")
    
    return {
        'model': svm_model,
        'cv_metrics': cv_metrics,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confidence_stats': confidence_stats,
        'test_confidence_stats': test_confidence_stats,
        'model_params': svm_model.get_params()
    }

def train_svm_multiple_kernels(X_train, y_train, X_test=None, y_test=None,
                              kernels=['rbf', 'linear'], C_values=[0.1, 1, 10],
                              cv_folds=5, random_state=42):
    """
    Train SVM with multiple kernels and compare performance.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    kernels : list
        List of kernel types to test
    C_values : list
        List of C values to test
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing results for all kernel/C combinations
    """
    print("Training SVM with multiple kernels and C values...")
    
    results = {}
    best_score = 0
    best_config = None
    
    for kernel in kernels:
        for C in C_values:
            config_name = f"{kernel}_C{C}"
            print(f"\nTraining SVM with kernel={kernel}, C={C}")
            
            # Train SVM with current configuration
            svm_results = train_svm(
                X_train, y_train, X_test, y_test,
                C=C, kernel=kernel, cv_folds=cv_folds, random_state=random_state
            )
            
            results[config_name] = svm_results
            
            # Track best configuration
            cv_f1_score = svm_results['cv_metrics']['f1']['mean']
            if cv_f1_score > best_score:
                best_score = cv_f1_score
                best_config = config_name
    
    # Create comparison summary
    comparison_df = pd.DataFrame({
        config: {
            'CV_F1_Mean': results[config]['cv_metrics']['f1']['mean'],
            'CV_F1_Std': results[config]['cv_metrics']['f1']['std'],
            'Train_F1': results[config]['train_metrics']['f1'],
            'Test_F1': results[config]['test_metrics']['f1'] if results[config]['test_metrics'] else None
        }
        for config in results.keys()
    }).T
    
    print(f"\nSVM Kernel Comparison Results:")
    print("=" * 70)
    print(comparison_df.round(4))
    print(f"\nBest Configuration: {best_config} (CV F1: {best_score:.4f})")
    
    return {
        'results': results,
        'comparison': comparison_df,
        'best_config': best_config,
        'best_model': results[best_config]['model']
    }

def cross_validate_svm(X, y, param_grid=None, cv_folds=5, scoring='f1', random_state=42):
    """
    Perform comprehensive cross-validation for SVM with different parameters.
    
    Parameters:
    -----------
    X : array-like or sparse matrix
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Parameter grid for testing different configurations
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for evaluation
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing CV results for different parameter combinations
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    
    print("Performing SVM cross-validation with parameter grid...")
    print(f"Parameter grid: {param_grid}")
    
    # Initialize model
    svm_model = SVC(probability=True, random_state=random_state)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        svm_model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Extract results
    cv_results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
        'grid_search_object': grid_search
    }
    
    print(f"\nBest SVM Parameters:")
    for param, value in cv_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best CV Score ({scoring}): {cv_results['best_score']:.4f}")
    
    return cv_results

def analyze_svm_support_vectors(svm_results, X_train, y_train):
    """
    Analyze SVM support vectors and decision boundary characteristics.
    
    Parameters:
    -----------
    svm_results : dict
        Results from train_svm function
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    
    Returns:
    --------
    dict : Dictionary containing support vector analysis
    """
    svm_model = svm_results['model']
    
    # Support vector analysis
    n_support_vectors = svm_model.n_support_
    support_vector_indices = svm_model.support_
    
    # Calculate support vector statistics
    total_samples = len(y_train)
    total_support_vectors = len(support_vector_indices)
    support_vector_ratio = total_support_vectors / total_samples
    
    # Support vectors per class
    class_0_support = n_support_vectors[0]
    class_1_support = n_support_vectors[1]
    
    analysis = {
        'total_support_vectors': total_support_vectors,
        'support_vector_ratio': support_vector_ratio,
        'class_0_support_vectors': class_0_support,
        'class_1_support_vectors': class_1_support,
        'support_vector_indices': support_vector_indices
    }
    
    print(f"\nSVM Support Vector Analysis:")
    print(f"Total Support Vectors: {total_support_vectors} ({support_vector_ratio:.2%} of training data)")
    print(f"Class 0 Support Vectors: {class_0_support}")
    print(f"Class 1 Support Vectors: {class_1_support}")
    
    return analysis

print("SVM training functions implemented successfully!")

# ============================================================================
# TASK 4.3: LOGISTIC REGRESSION TRAINING FUNCTION
# ============================================================================

def train_logistic_regression(X_train, y_train, X_test=None, y_test=None,
                             C=1.0, penalty='l2', solver='liblinear', max_iter=1000,
                             random_state=42, cv_folds=5, class_weight=None,
                             return_coefficients=True):
    """
    Train Logistic Regression classifier with L1 and L2 regularization.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    C : float
        Inverse of regularization strength
    penalty : str
        Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
    solver : str
        Algorithm to use in optimization problem
    max_iter : int
        Maximum number of iterations for solver convergence
    random_state : int
        Random state for reproducibility
    cv_folds : int
        Number of cross-validation folds
    class_weight : dict or str, optional
        Weights associated with classes
    return_coefficients : bool
        Whether to extract and analyze coefficients
    
    Returns:
    --------
    dict : Dictionary containing trained model, scores, and coefficient analysis
    """
    print(f"Training Logistic Regression (penalty={penalty}, C={C})...")
    
    # Initialize Logistic Regression
    lr_model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    
    # Perform cross-validation
    print(f"Performing {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        lr_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring='f1',
        n_jobs=-1
    )
    
    # Calculate additional CV metrics
    cv_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(
            lr_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring=metric,
            n_jobs=-1
        )
        cv_metrics[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Train final model on full training set
    print("Training final model on full training set...")
    lr_model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = lr_model.predict(X_train)
    y_train_proba = lr_model.predict_proba(X_train)[:, 1]
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    # Calculate prediction confidence statistics
    confidence_stats = {
        'mean_confidence': np.mean(np.max(lr_model.predict_proba(X_train), axis=1)),
        'std_confidence': np.std(np.max(lr_model.predict_proba(X_train), axis=1)),
        'min_confidence': np.min(np.max(lr_model.predict_proba(X_train), axis=1)),
        'max_confidence': np.max(np.max(lr_model.predict_proba(X_train), axis=1))
    }
    
    # Calculate test metrics if test data provided
    test_metrics = None
    test_confidence_stats = None
    if X_test is not None and y_test is not None:
        print("Evaluating on test set...")
        y_test_pred = lr_model.predict(X_test)
        y_test_proba = lr_model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        test_confidence_stats = {
            'mean_confidence': np.mean(np.max(lr_model.predict_proba(X_test), axis=1)),
            'std_confidence': np.std(np.max(lr_model.predict_proba(X_test), axis=1)),
            'min_confidence': np.min(np.max(lr_model.predict_proba(X_test), axis=1)),
            'max_confidence': np.max(np.max(lr_model.predict_proba(X_test), axis=1))
        }
    
    # Extract coefficients for feature importance
    coefficient_analysis = None
    if return_coefficients:
        coefficients = lr_model.coef_[0]  # For binary classification
        intercept = lr_model.intercept_[0]
        
        coefficient_analysis = {
            'coefficients': coefficients,
            'intercept': intercept,
            'abs_coefficients': np.abs(coefficients),
            'coefficient_stats': {
                'mean': np.mean(coefficients),
                'std': np.std(coefficients),
                'min': np.min(coefficients),
                'max': np.max(coefficients),
                'n_positive': np.sum(coefficients > 0),
                'n_negative': np.sum(coefficients < 0),
                'n_zero': np.sum(coefficients == 0)  # For L1 regularization
            }
        }
    
    # Print results summary
    print(f"\nLogistic Regression Training Results:")
    print(f"Cross-validation F1 Score: {cv_metrics['f1']['mean']:.4f} (+/- {cv_metrics['f1']['std']*2:.4f})")
    print(f"Training F1 Score: {train_metrics['f1']:.4f}")
    if test_metrics:
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Mean Prediction Confidence: {confidence_stats['mean_confidence']:.4f}")
    if coefficient_analysis:
        print(f"Non-zero Coefficients: {coefficient_analysis['coefficient_stats']['n_positive'] + coefficient_analysis['coefficient_stats']['n_negative']}")
        if penalty == 'l1':
            print(f"Zero Coefficients (L1 sparsity): {coefficient_analysis['coefficient_stats']['n_zero']}")
    
    return {
        'model': lr_model,
        'cv_metrics': cv_metrics,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confidence_stats': confidence_stats,
        'test_confidence_stats': test_confidence_stats,
        'coefficient_analysis': coefficient_analysis,
        'model_params': lr_model.get_params()
    }

def extract_lr_feature_importance(lr_results, feature_names=None, top_n=20):
    """
    Extract and visualize Logistic Regression coefficient-based feature importance.
    
    Parameters:
    -----------
    lr_results : dict
        Results from train_logistic_regression function
    feature_names : list, optional
        List of feature names
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with coefficient-based feature importance
    """
    if lr_results['coefficient_analysis'] is None:
        print("Coefficient analysis not available in results")
        return None
    
    coefficients = lr_results['coefficient_analysis']['coefficients']
    abs_coefficients = lr_results['coefficient_analysis']['abs_coefficients']
    
    # Create feature importance DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(coefficients))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': abs_coefficients,
        'importance_rank': range(1, len(coefficients) + 1)
    })
    
    # Sort by absolute coefficient value
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    importance_df['importance_rank'] = range(1, len(importance_df) + 1)
    
    # Display top features
    print(f"\nTop {top_n} Logistic Regression Feature Importances (by |coefficient|):")
    print("=" * 80)
    print(f"{'Feature':<40} {'Coefficient':<12} {'|Coefficient|':<12} {'Effect'}")
    print("-" * 80)
    
    for i, row in importance_df.head(top_n).iterrows():
        effect = "Positive" if row['coefficient'] > 0 else "Negative"
        print(f"{row['feature']:<40} {row['coefficient']:<12.4f} {row['abs_coefficient']:<12.4f} {effect}")
    
    return importance_df

def train_lr_multiple_penalties(X_train, y_train, X_test=None, y_test=None,
                               penalties=['l1', 'l2'], C_values=[0.01, 0.1, 1, 10],
                               cv_folds=5, random_state=42):
    """
    Train Logistic Regression with multiple penalties and C values.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    penalties : list
        List of penalty types to test
    C_values : list
        List of C values to test
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing results for all penalty/C combinations
    """
    print("Training Logistic Regression with multiple penalties and C values...")
    
    results = {}
    best_score = 0
    best_config = None
    
    for penalty in penalties:
        # Choose appropriate solver for penalty
        if penalty == 'l1':
            solver = 'liblinear'
        elif penalty == 'l2':
            solver = 'liblinear'
        else:
            solver = 'saga'  # For elasticnet
        
        for C in C_values:
            config_name = f"{penalty}_C{C}"
            print(f"\nTraining Logistic Regression with penalty={penalty}, C={C}")
            
            # Train Logistic Regression with current configuration
            lr_results = train_logistic_regression(
                X_train, y_train, X_test, y_test,
                C=C, penalty=penalty, solver=solver,
                cv_folds=cv_folds, random_state=random_state
            )
            
            results[config_name] = lr_results
            
            # Track best configuration
            cv_f1_score = lr_results['cv_metrics']['f1']['mean']
            if cv_f1_score > best_score:
                best_score = cv_f1_score
                best_config = config_name
    
    # Create comparison summary
    comparison_df = pd.DataFrame({
        config: {
            'CV_F1_Mean': results[config]['cv_metrics']['f1']['mean'],
            'CV_F1_Std': results[config]['cv_metrics']['f1']['std'],
            'Train_F1': results[config]['train_metrics']['f1'],
            'Test_F1': results[config]['test_metrics']['f1'] if results[config]['test_metrics'] else None,
            'Non_Zero_Coef': (results[config]['coefficient_analysis']['coefficient_stats']['n_positive'] + 
                             results[config]['coefficient_analysis']['coefficient_stats']['n_negative'])
        }
        for config in results.keys()
    }).T
    
    print(f"\nLogistic Regression Penalty Comparison Results:")
    print("=" * 80)
    print(comparison_df.round(4))
    print(f"\nBest Configuration: {best_config} (CV F1: {best_score:.4f})")
    
    return {
        'results': results,
        'comparison': comparison_df,
        'best_config': best_config,
        'best_model': results[best_config]['model']
    }

def cross_validate_logistic_regression(X, y, param_grid=None, cv_folds=5, scoring='f1', random_state=42):
    """
    Perform comprehensive cross-validation for Logistic Regression.
    
    Parameters:
    -----------
    X : array-like or sparse matrix
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Parameter grid for testing different configurations
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for evaluation
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing CV results for different parameter combinations
    """
    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000, 2000]
        }
    
    print("Performing Logistic Regression cross-validation with parameter grid...")
    print(f"Parameter grid: {param_grid}")
    
    # Initialize model
    lr_model = LogisticRegression(random_state=random_state, n_jobs=-1)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        lr_model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Extract results
    cv_results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': pd.DataFrame(grid_search.cv_results_),
        'grid_search_object': grid_search
    }
    
    print(f"\nBest Logistic Regression Parameters:")
    for param, value in cv_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best CV Score ({scoring}): {cv_results['best_score']:.4f}")
    
    return cv_results

def analyze_lr_regularization_path(X_train, y_train, penalty='l1', C_values=None, cv_folds=5):
    """
    Analyze regularization path for Logistic Regression to understand feature selection.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    penalty : str
        Regularization penalty ('l1' or 'l2')
    C_values : list, optional
        List of C values to test
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    dict : Dictionary containing regularization path analysis
    """
    if C_values is None:
        C_values = np.logspace(-4, 2, 20)  # From 0.0001 to 100
    
    print(f"Analyzing {penalty} regularization path...")
    
    coefficients_path = []
    cv_scores_path = []
    n_features_path = []
    
    solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
    
    for C in C_values:
        # Train model with current C
        lr_model = LogisticRegression(
            C=C, penalty=penalty, solver=solver, 
            random_state=42, max_iter=2000
        )
        
        # Cross-validation score
        cv_scores = cross_val_score(
            lr_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1'
        )
        cv_scores_path.append(cv_scores.mean())
        
        # Fit model to get coefficients
        lr_model.fit(X_train, y_train)
        coefficients_path.append(lr_model.coef_[0])
        
        # Count non-zero features (for L1)
        if penalty == 'l1':
            n_features_path.append(np.sum(lr_model.coef_[0] != 0))
        else:
            n_features_path.append(len(lr_model.coef_[0]))
    
    # Find optimal C
    best_idx = np.argmax(cv_scores_path)
    optimal_C = C_values[best_idx]
    best_cv_score = cv_scores_path[best_idx]
    
    regularization_analysis = {
        'C_values': C_values,
        'cv_scores': cv_scores_path,
        'coefficients_path': np.array(coefficients_path),
        'n_features_path': n_features_path,
        'optimal_C': optimal_C,
        'best_cv_score': best_cv_score,
        'penalty': penalty
    }
    
    print(f"Optimal C for {penalty} regularization: {optimal_C:.4f}")
    print(f"Best CV F1 Score: {best_cv_score:.4f}")
    if penalty == 'l1':
        print(f"Number of selected features at optimal C: {n_features_path[best_idx]}")
    
    return regularization_analysis

print("Logistic Regression training functions implemented successfully!")

# ============================================================================
# TASK 4.4: GRADIENT BOOSTING TRAINING FUNCTION
# ============================================================================

def train_gradient_boosting(X_train, y_train, X_test=None, y_test=None,
                           n_estimators=100, learning_rate=0.1, max_depth=3,
                           min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                           random_state=42, cv_folds=5, class_weight=None,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=5):
    """
    Train Gradient Boosting classifier with comprehensive evaluation.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    n_estimators : int
        Number of boosting stages
    learning_rate : float
        Learning rate shrinks the contribution of each tree
    max_depth : int
        Maximum depth of individual regression estimators
    min_samples_split : int
        Minimum number of samples required to split an internal node
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node
    subsample : float
        Fraction of samples used for fitting individual base learners
    random_state : int
        Random state for reproducibility
    cv_folds : int
        Number of cross-validation folds
    class_weight : str or dict, optional
        Weights associated with classes
    early_stopping : bool
        Whether to use early stopping to prevent overfitting
    validation_fraction : float
        Fraction of training data to use for early stopping validation
    n_iter_no_change : int
        Number of iterations with no improvement to wait before early stopping
    
    Returns:
    --------
    dict : Dictionary containing model, metrics, and analysis results
    """
    print("Training Gradient Boosting Classifier...")
    print(f"Parameters: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
    
    # Initialize Gradient Boosting model
    gb_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'subsample': subsample,
        'random_state': random_state
    }
    
    # Add early stopping parameters if enabled
    if early_stopping:
        gb_params.update({
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'tol': 1e-4
        })
    
    gb_model = GradientBoostingClassifier(**gb_params)
    
    # Train the model
    print("Fitting Gradient Boosting model...")
    gb_model.fit(X_train, y_train)
    
    # Get training predictions and probabilities
    y_train_pred = gb_model.predict(X_train)
    y_train_proba = gb_model.predict_proba(X_train)[:, 1]
    
    # Calculate training metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='binary'),
        'recall': recall_score(y_train, y_train_pred, average='binary'),
        'f1': f1_score(y_train, y_train_pred, average='binary'),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    print(f"Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Test set evaluation (if provided)
    test_metrics = None
    if X_test is not None and y_test is not None:
        y_test_pred = gb_model.predict(X_test)
        y_test_proba = gb_model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average='binary'),
            'recall': recall_score(y_test, y_test_pred, average='binary'),
            'f1': f1_score(y_test, y_test_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        print(f"\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Cross-validation evaluation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_model = GradientBoostingClassifier(**gb_params)
    
    cv_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        cv_scores = cross_val_score(
            cv_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring=metric
        )
        cv_metrics[metric] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        print(f"  CV {metric.upper()}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance analysis
    feature_importance = gb_model.feature_importances_
    
    # Create feature importance DataFrame (if feature names available)
    try:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print("-" * 50)
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    except Exception as e:
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(feature_importance))],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(f"Feature importance calculated (generic names used)")
    
    # Training progress analysis (if early stopping was used)
    training_analysis = {
        'n_estimators_used': gb_model.n_estimators_,
        'train_score_': getattr(gb_model, 'train_score_', None),
        'oob_improvement_': getattr(gb_model, 'oob_improvement_', None),
        'feature_importances_': gb_model.feature_importances_
    }
    
    if early_stopping and hasattr(gb_model, 'train_score_'):
        print(f"\nTraining Progress:")
        print(f"  Estimators used: {gb_model.n_estimators_} / {n_estimators}")
        if gb_model.n_estimators_ < n_estimators:
            print(f"  Early stopping triggered after {gb_model.n_estimators_} iterations")
    
    # Model complexity analysis
    complexity_metrics = {
        'n_estimators_final': gb_model.n_estimators_,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'total_nodes': sum(tree[0].tree_.node_count for tree in gb_model.estimators_),
        'avg_nodes_per_tree': np.mean([tree[0].tree_.node_count for tree in gb_model.estimators_])
    }
    
    print(f"\nModel Complexity:")
    print(f"  Total trees: {complexity_metrics['n_estimators_final']}")
    print(f"  Total nodes: {complexity_metrics['total_nodes']}")
    print(f"  Average nodes per tree: {complexity_metrics['avg_nodes_per_tree']:.1f}")
    
    return {
        'model': gb_model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_metrics': cv_metrics,
        'feature_importance': importance_df,
        'training_analysis': training_analysis,
        'complexity_metrics': complexity_metrics,
        'parameters': gb_params
    }

def train_gb_with_early_stopping(X_train, y_train, X_val=None, y_val=None,
                                 n_estimators=1000, learning_rate=0.1, max_depth=3,
                                 min_samples_split=2, min_samples_leaf=1,
                                 subsample=0.8, random_state=42,
                                 n_iter_no_change=10, tol=1e-4):
    """
    Train Gradient Boosting with explicit early stopping validation.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_val : array-like or sparse matrix, optional
        Validation feature matrix for early stopping
    y_val : array-like, optional
        Validation target vector for early stopping
    n_estimators : int
        Maximum number of boosting stages
    learning_rate : float
        Learning rate
    max_depth : int
        Maximum depth of trees
    min_samples_split : int
        Minimum samples to split
    min_samples_leaf : int
        Minimum samples at leaf
    subsample : float
        Subsample ratio
    random_state : int
        Random state
    n_iter_no_change : int
        Iterations without improvement before stopping
    tol : float
        Tolerance for improvement
    
    Returns:
    --------
    dict : Dictionary containing model and training history
    """
    print("Training Gradient Boosting with explicit early stopping...")
    
    # If no validation set provided, split training data
    if X_val is None or y_val is None:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
    else:
        X_train_split, y_train_split = X_train, y_train
    
    # Initialize model with early stopping
    gb_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        validation_fraction=0.0,  # We provide explicit validation set
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        random_state=random_state
    )
    
    # Train model
    gb_model.fit(X_train_split, y_train_split)
    
    # Manual early stopping evaluation
    train_scores = []
    val_scores = []
    best_score = -np.inf
    best_iteration = 0
    no_improvement_count = 0
    
    # Evaluate at each stage
    for i in range(1, gb_model.n_estimators_ + 1):
        # Create model with i estimators
        temp_model = GradientBoostingClassifier(
            n_estimators=i,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state
        )
        temp_model.fit(X_train_split, y_train_split)
        
        # Calculate scores
        train_pred = temp_model.predict_proba(X_train_split)[:, 1]
        val_pred = temp_model.predict_proba(X_val)[:, 1]
        
        train_score = roc_auc_score(y_train_split, train_pred)
        val_score = roc_auc_score(y_val, val_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        # Check for improvement
        if val_score > best_score + tol:
            best_score = val_score
            best_iteration = i
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Early stopping check
        if no_improvement_count >= n_iter_no_change:
            print(f"Early stopping at iteration {i} (best: {best_iteration})")
            break
    
    # Train final model with optimal number of estimators
    final_model = GradientBoostingClassifier(
        n_estimators=best_iteration,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )
    final_model.fit(X_train, y_train)
    
    training_history = {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'best_iteration': best_iteration,
        'best_val_score': best_score,
        'total_iterations': len(train_scores)
    }
    
    print(f"Best validation AUC: {best_score:.4f} at iteration {best_iteration}")
    
    return {
        'model': final_model,
        'training_history': training_history,
        'best_iteration': best_iteration
    }

def extract_gb_feature_importance(gb_model, feature_names=None, top_n=20):
    """
    Extract and analyze feature importance from Gradient Boosting model.
    
    Parameters:
    -----------
    gb_model : GradientBoostingClassifier
        Trained Gradient Boosting model
    feature_names : list, optional
        List of feature names
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with feature importance analysis
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(gb_model.feature_importances_))]
    
    # Get feature importances
    importances = gb_model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_pct': (importances / importances.sum()) * 100
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance_pct'].cumsum()
    
    # Feature importance statistics
    importance_stats = {
        'total_features': len(feature_names),
        'non_zero_importance': (importances > 0).sum(),
        'top_10_cumulative': importance_df.head(10)['importance_pct'].sum(),
        'top_20_cumulative': importance_df.head(20)['importance_pct'].sum(),
        'features_for_80pct': (importance_df['cumulative_importance'] <= 80).sum(),
        'features_for_90pct': (importance_df['cumulative_importance'] <= 90).sum()
    }
    
    print(f"Gradient Boosting Feature Importance Analysis:")
    print("=" * 60)
    print(f"Total features: {importance_stats['total_features']}")
    print(f"Features with non-zero importance: {importance_stats['non_zero_importance']}")
    print(f"Top 10 features explain: {importance_stats['top_10_cumulative']:.1f}% of importance")
    print(f"Top 20 features explain: {importance_stats['top_20_cumulative']:.1f}% of importance")
    print(f"Features needed for 80% importance: {importance_stats['features_for_80pct']}")
    print(f"Features needed for 90% importance: {importance_stats['features_for_90pct']}")
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    print(f"{'Feature':<40} {'Importance':<12} {'Percentage':<12} {'Cumulative':<12}")
    print("-" * 60)
    
    for i, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:<40} {row['importance']:<12.4f} {row['importance_pct']:<12.1f}% {row['cumulative_importance']:<12.1f}%")
    
    return importance_df

def train_gb_multiple_configs(X_train, y_train, X_test=None, y_test=None,
                             n_estimators_list=[100, 200, 300],
                             learning_rates=[0.01, 0.1, 0.2],
                             max_depths=[3, 5, 7],
                             cv_folds=5, random_state=42):
    """
    Train Gradient Boosting with multiple configurations for comparison.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like or sparse matrix, optional
        Test feature matrix
    y_test : array-like, optional
        Test target vector
    n_estimators_list : list
        List of n_estimators values to test
    learning_rates : list
        List of learning rates to test
    max_depths : list
        List of max depths to test
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing results for all configurations
    """
    print("Training Gradient Boosting with multiple configurations...")
    
    results = {}
    best_score = 0
    best_config = None
    
    total_configs = len(n_estimators_list) * len(learning_rates) * len(max_depths)
    config_count = 0
    
    for n_est in n_estimators_list:
        for lr in learning_rates:
            for depth in max_depths:
                config_count += 1
                config_name = f"n{n_est}_lr{lr}_d{depth}"
                
                print(f"\n[{config_count}/{total_configs}] Training GB: n_estimators={n_est}, lr={lr}, max_depth={depth}")
                
                # Train Gradient Boosting with current configuration
                gb_results = train_gradient_boosting(
                    X_train, y_train, X_test, y_test,
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth,
                    cv_folds=cv_folds,
                    random_state=random_state
                )
                
                results[config_name] = gb_results
                
                # Track best configuration based on CV F1 score
                cv_f1_score = gb_results['cv_metrics']['f1']['mean']
                if cv_f1_score > best_score:
                    best_score = cv_f1_score
                    best_config = config_name
    
    # Create comparison summary
    comparison_df = pd.DataFrame({
        config: {
            'N_Estimators': results[config]['parameters']['n_estimators'],
            'Learning_Rate': results[config]['parameters']['learning_rate'],
            'Max_Depth': results[config]['parameters']['max_depth'],
            'CV_F1_Mean': results[config]['cv_metrics']['f1']['mean'],
            'CV_F1_Std': results[config]['cv_metrics']['f1']['std'],
            'CV_AUC_Mean': results[config]['cv_metrics']['roc_auc']['mean'],
            'Train_F1': results[config]['train_metrics']['f1'],
            'Test_F1': results[config]['test_metrics']['f1'] if results[config]['test_metrics'] else None,
            'Complexity': results[config]['complexity_metrics']['total_nodes']
        }
        for config in results.keys()
    }).T
    
    print(f"\nGradient Boosting Configuration Comparison:")
    print("=" * 100)
    print(comparison_df.round(4))
    print(f"\nBest Configuration: {best_config}")
    print(f"Best CV F1 Score: {best_score:.4f}")
    
    return {
        'results': results,
        'comparison': comparison_df,
        'best_config': best_config,
        'best_model': results[best_config]['model']
    }

def cross_validate_gradient_boosting(X, y, param_grid=None, cv_folds=5, scoring='f1', random_state=42):
    """
    Perform comprehensive cross-validation for Gradient Boosting.
    
    Parameters:
    -----------
    X : array-like or sparse matrix
        Feature matrix
    y : array-like
        Target vector
    param_grid : dict, optional
        Parameter grid for testing different configurations
    cv_folds : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for evaluation
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing CV results for different parameter combinations
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    print("Performing Gradient Boosting cross-validation with parameter grid...")
    print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    
    # Initialize model
    gb_model = GradientBoostingClassifier(random_state=random_state)
    
    # Perform randomized search (more efficient for large parameter spaces)
    from sklearn.model_selection import RandomizedSearchCV
    
    random_search = RandomizedSearchCV(
        gb_model,
        param_grid,
        n_iter=50,  # Limit to 50 random combinations
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    # Fit randomized search
    random_search.fit(X, y)
    
    # Extract results
    cv_results = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'best_estimator': random_search.best_estimator_,
        'cv_results': pd.DataFrame(random_search.cv_results_),
        'search_object': random_search
    }
    
    print(f"\nBest Gradient Boosting Parameters:")
    for param, value in cv_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best CV Score ({scoring}): {cv_results['best_score']:.4f}")
    
    return cv_results

print("Gradient Boosting training functions implemented successfully!")

# ============================================================================
# TASK 5.2: IMPLEMENT GRIDSEARCHCV OPTIMIZATION
# ============================================================================

def create_gridsearch_wrapper(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1):
    """
    Create GridSearchCV wrapper for a model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Base model to optimize
    param_grid : dict
        Parameter grid for search
    cv : int or cross-validation generator
        Cross-validation strategy
    scoring : str or callable
        Scoring metric for optimization
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    
    Returns:
    --------
    GridSearchCV : Configured GridSearchCV object
    """
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        error_score='raise'
    )
    
    return grid_search

def optimize_random_forest_grid(X_train, y_train, search_budget='medium', cv=5, scoring='f1'):
    """
    Optimize Random Forest using GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_budget : str
        Search budget ('small', 'medium', 'large', 'comprehensive')
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting Random Forest GridSearchCV optimization (budget: {search_budget})...")
    
    # Get parameter grid
    param_grid = create_adaptive_param_grid('random_forest', search_budget)
    
    # Estimate search time
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    
    time_estimate = estimate_search_time(n_combinations, cv)
    
    # Create base model
    rf_model = RandomForestClassifier()
    
    # Create GridSearchCV
    grid_search = create_gridsearch_wrapper(
        model=rf_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit grid search
    print(f"Fitting GridSearchCV with {n_combinations} parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'random_forest',
        'search_method': 'grid_search',
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'n_combinations_tested': len(grid_search.cv_results_['params']),
        'optimization_time_seconds': optimization_time,
        'search_budget': search_budget,
        'scoring_metric': scoring
    }
    
    print(f"Random Forest GridSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def optimize_svm_grid(X_train, y_train, search_budget='medium', cv=5, scoring='f1'):
    """
    Optimize SVM using GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_budget : str
        Search budget
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting SVM GridSearchCV optimization (budget: {search_budget})...")
    
    # Get parameter grid
    param_grid = create_adaptive_param_grid('svm', search_budget)
    
    # Filter valid parameter combinations
    all_combinations = generate_parameter_combinations(param_grid)
    valid_combinations = filter_valid_parameters(all_combinations, 'svm')
    
    # Convert back to grid format for GridSearchCV
    filtered_param_grid = {}
    for param_name in param_grid.keys():
        unique_values = list(set(combo[param_name] for combo in valid_combinations if param_name in combo))
        filtered_param_grid[param_name] = unique_values
    
    # Estimate search time
    time_estimate = estimate_search_time(len(valid_combinations), cv)
    
    # Create base model
    svm_model = SVC(probability=True)
    
    # Create GridSearchCV
    grid_search = create_gridsearch_wrapper(
        model=svm_model,
        param_grid=filtered_param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit grid search
    print(f"Fitting GridSearchCV with {len(valid_combinations)} valid parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'svm',
        'search_method': 'grid_search',
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'n_combinations_tested': len(grid_search.cv_results_['params']),
        'optimization_time_seconds': optimization_time,
        'search_budget': search_budget,
        'scoring_metric': scoring
    }
    
    print(f"SVM GridSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def optimize_logistic_regression_grid(X_train, y_train, search_budget='medium', cv=5, scoring='f1'):
    """
    Optimize Logistic Regression using GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_budget : str
        Search budget
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting Logistic Regression GridSearchCV optimization (budget: {search_budget})...")
    
    # Get parameter grid
    param_grid = create_adaptive_param_grid('logistic_regression', search_budget)
    
    # Filter valid parameter combinations
    all_combinations = generate_parameter_combinations(param_grid)
    valid_combinations = filter_valid_parameters(all_combinations, 'logistic_regression')
    
    # Convert back to grid format for GridSearchCV
    filtered_param_grid = {}
    for param_name in param_grid.keys():
        unique_values = list(set(combo[param_name] for combo in valid_combinations if param_name in combo))
        filtered_param_grid[param_name] = unique_values
    
    # Estimate search time
    time_estimate = estimate_search_time(len(valid_combinations), cv)
    
    # Create base model
    lr_model = LogisticRegression()
    
    # Create GridSearchCV
    grid_search = create_gridsearch_wrapper(
        model=lr_model,
        param_grid=filtered_param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit grid search
    print(f"Fitting GridSearchCV with {len(valid_combinations)} valid parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'logistic_regression',
        'search_method': 'grid_search',
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'n_combinations_tested': len(grid_search.cv_results_['params']),
        'optimization_time_seconds': optimization_time,
        'search_budget': search_budget,
        'scoring_metric': scoring
    }
    
    print(f"Logistic Regression GridSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def optimize_gradient_boosting_grid(X_train, y_train, search_budget='medium', cv=5, scoring='f1'):
    """
    Optimize Gradient Boosting using GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_budget : str
        Search budget
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting Gradient Boosting GridSearchCV optimization (budget: {search_budget})...")
    
    # Get parameter grid
    param_grid = create_adaptive_param_grid('gradient_boosting', search_budget)
    
    # Estimate search time
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    
    time_estimate = estimate_search_time(n_combinations, cv)
    
    # Create base model
    gb_model = GradientBoostingClassifier()
    
    # Create GridSearchCV
    grid_search = create_gridsearch_wrapper(
        model=gb_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit grid search
    print(f"Fitting GridSearchCV with {n_combinations} parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'gradient_boosting',
        'search_method': 'grid_search',
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'n_combinations_tested': len(grid_search.cv_results_['params']),
        'optimization_time_seconds': optimization_time,
        'search_budget': search_budget,
        'scoring_metric': scoring
    }
    
    print(f"Gradient Boosting GridSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def track_optimization_progress(grid_search, update_interval=10):
    """
    Track and display optimization progress.
    
    Parameters:
    -----------
    grid_search : GridSearchCV
        GridSearchCV object being fitted
    update_interval : int
        Seconds between progress updates
    
    Returns:
    --------
    None
    """
    import time
    import threading
    
    def progress_tracker():
        start_time = time.time()
        while not hasattr(grid_search, 'best_score_'):
            elapsed = time.time() - start_time
            print(f"GridSearchCV running... Elapsed time: {elapsed:.1f}s")
            time.sleep(update_interval)
    
    # Start progress tracking in separate thread
    progress_thread = threading.Thread(target=progress_tracker)
    progress_thread.daemon = True
    progress_thread.start()

class GridSearchProgressCallback:
    """
    Enhanced progress tracking for GridSearchCV with detailed metrics.
    """
    
    def __init__(self, total_combinations, model_name="Model"):
        self.total_combinations = total_combinations
        self.model_name = model_name
        self.start_time = None
        self.completed_fits = 0
        
    def start_tracking(self):
        """Start progress tracking."""
        self.start_time = datetime.now()
        print(f"\n🚀 Starting {self.model_name} GridSearchCV optimization...")
        print(f"📊 Total parameter combinations: {self.total_combinations}")
        print(f"⏰ Started at: {self.start_time.strftime('%H:%M:%S')}")
        print("-" * 50)
    
    def update_progress(self, current_fit=None):
        """Update progress during optimization."""
        if self.start_time is None:
            return
            
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if current_fit is not None:
            self.completed_fits = current_fit
            
        progress_pct = (self.completed_fits / self.total_combinations) * 100 if self.total_combinations > 0 else 0
        
        # Estimate remaining time
        if self.completed_fits > 0:
            avg_time_per_fit = elapsed / self.completed_fits
            remaining_fits = self.total_combinations - self.completed_fits
            eta_seconds = remaining_fits * avg_time_per_fit
            eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
        else:
            eta_str = "calculating..."
        
        print(f"⏳ Progress: {self.completed_fits}/{self.total_combinations} ({progress_pct:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}")
    
    def finish_tracking(self, best_score=None):
        """Finish progress tracking."""
        if self.start_time is None:
            return
            
        total_time = (datetime.now() - self.start_time).total_seconds()
        print("-" * 50)
        print(f"✅ {self.model_name} optimization completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        if best_score is not None:
            print(f"🎯 Best score achieved: {best_score:.4f}")
        print()

def create_enhanced_gridsearch_wrapper(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1, 
                                     progress_callback=None):
    """
    Create enhanced GridSearchCV wrapper with progress tracking.
    
    Parameters:
    -----------
    model : sklearn estimator
        Base model to optimize
    param_grid : dict
        Parameter grid for search
    cv : int or cross-validation generator
        Cross-validation strategy
    scoring : str or callable
        Scoring metric for optimization
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    progress_callback : GridSearchProgressCallback, optional
        Progress tracking callback
    
    Returns:
    --------
    GridSearchCV : Configured GridSearchCV object
    """
    from sklearn.model_selection import GridSearchCV
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    # Initialize progress callback if not provided
    if progress_callback is None:
        progress_callback = GridSearchProgressCallback(
            total_combinations, 
            model.__class__.__name__
        )
    
    # Start progress tracking
    progress_callback.start_tracking()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        error_score='raise'
    )
    
    # Store callback for later use
    grid_search._progress_callback = progress_callback
    
    return grid_search

def extract_best_parameters(grid_search_results):
    """
    Extract and format best parameters from GridSearchCV results.
    
    Parameters:
    -----------
    grid_search_results : dict
        Results from GridSearchCV optimization
    
    Returns:
    --------
    dict : Formatted best parameters with metadata
    """
    best_params_info = {
        'model_type': grid_search_results['model_type'],
        'best_parameters': grid_search_results['best_params'],
        'best_cv_score': grid_search_results['best_score'],
        'optimization_method': grid_search_results['search_method'],
        'search_budget': grid_search_results['search_budget'],
        'combinations_tested': grid_search_results['n_combinations_tested'],
        'optimization_time': grid_search_results['optimization_time_seconds'],
        'scoring_metric': grid_search_results['scoring_metric']
    }
    
    return best_params_info

def save_grid_search_results(results, filename=None):
    """
    Save GridSearchCV results to file.
    
    Parameters:
    -----------
    results : dict
        GridSearchCV optimization results
    filename : str, optional
        Custom filename for saving results
    
    Returns:
    --------
    str : Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gridsearch_{results['model_type']}_{timestamp}.joblib"
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    
    # Save results
    joblib.dump(results, filepath)
    
    print(f"GridSearchCV results saved to: {filepath}")
    return filepath

def run_comprehensive_grid_search(X_train, y_train, models=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting'], 
                                 search_budget='medium', cv=5, scoring='f1', save_results=True):
    """
    Run comprehensive GridSearchCV optimization for multiple models.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    models : list
        List of model types to optimize
    search_budget : str
        Search budget for each model
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    save_results : bool
        Whether to save results to files
    
    Returns:
    --------
    dict : Dictionary containing optimization results for all models
    """
    print("=" * 60)
    print("COMPREHENSIVE GRIDSEARCHCV OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    optimization_functions = {
        'random_forest': optimize_random_forest_grid,
        'svm': optimize_svm_grid,
        'logistic_regression': optimize_logistic_regression_grid,
        'gradient_boosting': optimize_gradient_boosting_grid
    }
    
    all_results = {}
    total_start_time = datetime.now()
    
    for i, model_type in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Optimizing {model_type.replace('_', ' ').title()}...")
        print("-" * 40)
        
        if model_type in optimization_functions:
            try:
                # Run optimization
                results = optimization_functions[model_type](
                    X_train, y_train, 
                    search_budget=search_budget, 
                    cv=cv, 
                    scoring=scoring
                )
                
                all_results[model_type] = results
                
                # Save results if requested
                if save_results:
                    save_grid_search_results(results)
                
                print(f"✓ {model_type} optimization completed successfully!")
                
            except Exception as e:
                print(f"✗ Error optimizing {model_type}: {str(e)}")
                all_results[model_type] = {'error': str(e)}
        else:
            print(f"✗ Unknown model type: {model_type}")
            all_results[model_type] = {'error': f'Unknown model type: {model_type}'}
    
    total_end_time = datetime.now()
    total_time = (total_end_time - total_start_time).total_seconds()
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for model_type, results in all_results.items():
        if 'error' not in results:
            successful_models.append(model_type)
            print(f"✓ {model_type}: Best Score = {results['best_score']:.4f}")
        else:
            failed_models.append(model_type)
            print(f"✗ {model_type}: {results['error']}")
    
    print(f"\nTotal optimization time: {total_time:.1f} seconds")
    print(f"Successful optimizations: {len(successful_models)}/{len(models)}")
    
    # Find best overall model
    if successful_models:
        best_model = max(successful_models, key=lambda m: all_results[m]['best_score'])
        best_score = all_results[best_model]['best_score']
        print(f"Best performing model: {best_model} (Score: {best_score:.4f})")
    
    return all_results

def compare_grid_search_results(results_dict):
    """
    Compare GridSearchCV results across multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from multiple model optimizations
    
    Returns:
    --------
    pandas.DataFrame : Comparison table of model performances
    """
    comparison_data = []
    
    for model_type, results in results_dict.items():
        if 'error' not in results:
            comparison_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'Best_Score': results['best_score'],
                'Combinations_Tested': results['n_combinations_tested'],
                'Optimization_Time_Seconds': results['optimization_time_seconds'],
                'Search_Budget': results['search_budget'],
                'Scoring_Metric': results['scoring_metric']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Best_Score', ascending=False)
        
        print("\nModel Performance Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    else:
        print("No successful optimizations to compare.")
        return pd.DataFrame()

def extract_best_models_from_grid_search(results_dict):
    """
    Extract the best trained models from GridSearchCV results.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from multiple model optimizations
    
    Returns:
    --------
    dict : Dictionary containing the best trained models
    """
    best_models = {}
    
    for model_type, results in results_dict.items():
        if 'error' not in results and 'best_estimator' in results:
            best_models[model_type] = {
                'model': results['best_estimator'],
                'params': results['best_params'],
                'score': results['best_score'],
                'cv_results': results['cv_results']
            }
    
    print(f"Extracted {len(best_models)} best models from GridSearchCV results")
    return best_models

print("GridSearchCV optimization functions created successfully!")
print("Available functions:")
print("- create_gridsearch_wrapper()")
print("- optimize_random_forest_grid()")
print("- optimize_svm_grid()")
print("- optimize_logistic_regression_grid()")
print("- optimize_gradient_boosting_grid()")
print("- run_comprehensive_grid_search()")
print("- compare_grid_search_results()")
print("- extract_best_models_from_grid_search()")
print("- track_optimization_progress()")
print("- extract_best_parameters()")
print("- save_grid_search_results()")

# ============================================================================
# TASK 5.3: IMPLEMENT RANDOMIZEDSEARCHCV OPTIMIZATION
# ============================================================================

def create_randomized_search_wrapper(model, param_distributions, n_iter=100, cv=5, scoring='f1', n_jobs=-1, verbose=1, random_state=42):
    """
    Create RandomizedSearchCV wrapper for a model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Base model to optimize
    param_distributions : dict
        Parameter distributions for search
    n_iter : int
        Number of parameter settings sampled
    cv : int or cross-validation generator
        Cross-validation strategy
    scoring : str or callable
        Scoring metric for optimization
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    RandomizedSearchCV : Configured RandomizedSearchCV object
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
        error_score='raise'
    )
    
    return random_search

def create_parameter_distributions():
    """
    Create parameter distributions for RandomizedSearchCV.
    Uses continuous and discrete distributions for intelligent sampling.
    
    Returns:
    --------
    dict : Dictionary containing parameter distributions for each model type
    """
    from scipy.stats import randint, uniform, loguniform
    
    # Random Forest parameter distributions
    rf_param_dist = {
        'n_estimators': randint(50, 501),  # 50 to 500
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': randint(2, 21),  # 2 to 20
        'min_samples_leaf': randint(1, 11),  # 1 to 10
        'max_features': ['sqrt', 'log2', None] + list(uniform(0.1, 0.9).rvs(10)),
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'random_state': [42]
    }
    
    # SVM parameter distributions
    svm_param_dist = {
        'C': loguniform(0.001, 1000),  # Log-uniform from 0.001 to 1000
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'] + list(loguniform(0.001, 10).rvs(20)),
        'degree': randint(2, 6),  # 2 to 5 for poly kernel
        'coef0': uniform(-1, 2),  # -1 to 1 for poly and sigmoid
        'class_weight': [None, 'balanced'],
        'probability': [True],
        'random_state': [42]
    }
    
    # Logistic Regression parameter distributions
    lr_param_dist = {
        'C': loguniform(0.001, 1000),
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'],
        'l1_ratio': uniform(0.1, 0.8),  # 0.1 to 0.9 for elasticnet
        'max_iter': randint(100, 5001),  # 100 to 5000
        'class_weight': [None, 'balanced'],
        'random_state': [42]
    }
    
    # Gradient Boosting parameter distributions
    gb_param_dist = {
        'n_estimators': randint(50, 501),
        'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
        'max_depth': randint(3, 11),  # 3 to 10
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 11),
        'max_features': ['sqrt', 'log2', None] + list(uniform(0.1, 0.9).rvs(10)),
        'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
        'random_state': [42]
    }
    
    return {
        'random_forest': rf_param_dist,
        'svm': svm_param_dist,
        'logistic_regression': lr_param_dist,
        'gradient_boosting': gb_param_dist
    }

def intelligent_parameter_sampling(param_distributions, model_type, n_samples=100):
    """
    Intelligently sample parameters from distributions with constraint validation.
    
    Parameters:
    -----------
    param_distributions : dict
        Parameter distributions
    model_type : str
        Type of model for constraint validation
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    list : List of valid parameter dictionaries
    """
    import random
    from scipy.stats import uniform, randint, loguniform
    
    valid_samples = []
    attempts = 0
    max_attempts = n_samples * 10  # Prevent infinite loops
    
    while len(valid_samples) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample parameters
        sample = {}
        for param_name, distribution in param_distributions.items():
            if isinstance(distribution, list):
                # Discrete choice
                sample[param_name] = random.choice(distribution)
            elif hasattr(distribution, 'rvs'):
                # Scipy distribution
                if isinstance(distribution, (randint,)):
                    sample[param_name] = int(distribution.rvs())
                else:
                    sample[param_name] = distribution.rvs()
            else:
                # Fallback to random choice
                sample[param_name] = random.choice(distribution)
        
        # Validate constraints
        is_valid, _ = validate_parameter_constraints(sample, model_type)
        if is_valid:
            valid_samples.append(sample)
    
    print(f"Generated {len(valid_samples)} valid parameter samples from {attempts} attempts")
    return valid_samples

def optimize_random_forest_randomized(X_train, y_train, n_iter=100, cv=5, scoring='f1'):
    """
    Optimize Random Forest using RandomizedSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_iter : int
        Number of parameter settings to sample
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting Random Forest RandomizedSearchCV optimization ({n_iter} iterations)...")
    
    # Get parameter distributions
    param_distributions = create_parameter_distributions()['random_forest']
    
    # Estimate search time
    time_estimate = estimate_search_time(n_iter, cv)
    
    # Create base model
    rf_model = RandomForestClassifier()
    
    # Create RandomizedSearchCV
    random_search = create_randomized_search_wrapper(
        model=rf_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit randomized search
    print(f"Fitting RandomizedSearchCV with {n_iter} parameter samples...")
    random_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'random_forest',
        'search_method': 'randomized_search',
        'best_estimator': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'n_iterations': n_iter,
        'optimization_time_seconds': optimization_time,
        'scoring_metric': scoring
    }
    
    print(f"Random Forest RandomizedSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def optimize_svm_randomized(X_train, y_train, n_iter=100, cv=5, scoring='f1'):
    """
    Optimize SVM using RandomizedSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_iter : int
        Number of parameter settings to sample
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting SVM RandomizedSearchCV optimization ({n_iter} iterations)...")
    
    # Get parameter distributions
    param_distributions = create_parameter_distributions()['svm']
    
    # Generate valid parameter samples
    valid_samples = intelligent_parameter_sampling(param_distributions, 'svm', n_iter)
    
    # Convert samples back to distributions for RandomizedSearchCV
    # Use the original distributions but limit iterations to valid samples
    actual_n_iter = min(n_iter, len(valid_samples))
    
    # Estimate search time
    time_estimate = estimate_search_time(actual_n_iter, cv)
    
    # Create base model
    svm_model = SVC(probability=True)
    
    # Create RandomizedSearchCV
    random_search = create_randomized_search_wrapper(
        model=svm_model,
        param_distributions=param_distributions,
        n_iter=actual_n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit randomized search
    print(f"Fitting RandomizedSearchCV with {actual_n_iter} parameter samples...")
    random_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'svm',
        'search_method': 'randomized_search',
        'best_estimator': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'n_iterations': actual_n_iter,
        'optimization_time_seconds': optimization_time,
        'scoring_metric': scoring
    }
    
    print(f"SVM RandomizedSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def optimize_logistic_regression_randomized(X_train, y_train, n_iter=100, cv=5, scoring='f1'):
    """
    Optimize Logistic Regression using RandomizedSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_iter : int
        Number of parameter settings to sample
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting Logistic Regression RandomizedSearchCV optimization ({n_iter} iterations)...")
    
    # Get parameter distributions
    param_distributions = create_parameter_distributions()['logistic_regression']
    
    # Generate valid parameter samples
    valid_samples = intelligent_parameter_sampling(param_distributions, 'logistic_regression', n_iter)
    actual_n_iter = min(n_iter, len(valid_samples))
    
    # Estimate search time
    time_estimate = estimate_search_time(actual_n_iter, cv)
    
    # Create base model
    lr_model = LogisticRegression()
    
    # Create RandomizedSearchCV
    random_search = create_randomized_search_wrapper(
        model=lr_model,
        param_distributions=param_distributions,
        n_iter=actual_n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit randomized search
    print(f"Fitting RandomizedSearchCV with {actual_n_iter} parameter samples...")
    random_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'logistic_regression',
        'search_method': 'randomized_search',
        'best_estimator': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'n_iterations': actual_n_iter,
        'optimization_time_seconds': optimization_time,
        'scoring_metric': scoring
    }
    
    print(f"Logistic Regression RandomizedSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def optimize_gradient_boosting_randomized(X_train, y_train, n_iter=100, cv=5, scoring='f1'):
    """
    Optimize Gradient Boosting using RandomizedSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_iter : int
        Number of parameter settings to sample
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict : Optimization results
    """
    print(f"Starting Gradient Boosting RandomizedSearchCV optimization ({n_iter} iterations)...")
    
    # Get parameter distributions
    param_distributions = create_parameter_distributions()['gradient_boosting']
    
    # Estimate search time
    time_estimate = estimate_search_time(n_iter, cv)
    
    # Create base model
    gb_model = GradientBoostingClassifier()
    
    # Create RandomizedSearchCV
    random_search = create_randomized_search_wrapper(
        model=gb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Track optimization time
    start_time = datetime.now()
    
    # Fit randomized search
    print(f"Fitting RandomizedSearchCV with {n_iter} parameter samples...")
    random_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    optimization_time = (end_time - start_time).total_seconds()
    
    # Extract results
    results = {
        'model_type': 'gradient_boosting',
        'search_method': 'randomized_search',
        'best_estimator': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'n_iterations': n_iter,
        'optimization_time_seconds': optimization_time,
        'scoring_metric': scoring
    }
    
    print(f"Gradient Boosting RandomizedSearchCV completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimization time: {optimization_time:.1f} seconds")
    
    return results

def compare_grid_vs_random_search(grid_results, random_results):
    """
    Compare results between GridSearchCV and RandomizedSearchCV.
    
    Parameters:
    -----------
    grid_results : dict
        Results from GridSearchCV
    random_results : dict
        Results from RandomizedSearchCV
    
    Returns:
    --------
    dict : Comparison analysis
    """
    comparison = {
        'model_type': grid_results['model_type'],
        'grid_search': {
            'best_score': grid_results['best_score'],
            'best_params': grid_results['best_params'],
            'optimization_time': grid_results['optimization_time_seconds'],
            'combinations_tested': grid_results.get('n_combinations_tested', 'N/A')
        },
        'random_search': {
            'best_score': random_results['best_score'],
            'best_params': random_results['best_params'],
            'optimization_time': random_results['optimization_time_seconds'],
            'iterations': random_results['n_iterations']
        }
    }
    
    # Calculate performance difference
    score_diff = random_results['best_score'] - grid_results['best_score']
    time_ratio = random_results['optimization_time_seconds'] / grid_results['optimization_time_seconds']
    
    comparison['analysis'] = {
        'score_difference': score_diff,
        'random_better': score_diff > 0,
        'time_ratio_random_vs_grid': time_ratio,
        'random_faster': time_ratio < 1,
        'efficiency_score': score_diff / time_ratio  # Score improvement per time unit
    }
    
    print(f"\nGrid vs Random Search Comparison for {comparison['model_type']}:")
    print(f"Grid Search - Score: {grid_results['best_score']:.4f}, Time: {grid_results['optimization_time_seconds']:.1f}s")
    print(f"Random Search - Score: {random_results['best_score']:.4f}, Time: {random_results['optimization_time_seconds']:.1f}s")
    print(f"Score difference: {score_diff:+.4f} ({'Random' if score_diff > 0 else 'Grid'} is better)")
    print(f"Time ratio: {time_ratio:.2f}x ({'Random' if time_ratio < 1 else 'Grid'} is faster)")
    
    return comparison

def save_randomized_search_results(results, filename=None):
    """
    Save RandomizedSearchCV results to file.
    
    Parameters:
    -----------
    results : dict
        RandomizedSearchCV optimization results
    filename : str, optional
        Custom filename for saving results
    
    Returns:
    --------
    str : Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"randomized_search_{results['model_type']}_{timestamp}.joblib"
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    
    # Save results
    joblib.dump(results, filepath)
    
    print(f"RandomizedSearchCV results saved to: {filepath}")
    return filepath

print("RandomizedSearchCV optimization functions created successfully!")
print("Available functions:")
print("- create_randomized_search_wrapper()")
print("- create_parameter_distributions()")
print("- intelligent_parameter_sampling()")
print("- optimize_random_forest_randomized()")
print("- optimize_svm_randomized()")
print("- optimize_logistic_regression_randomized()")
print("- optimize_gradient_boosting_randomized()")
print("- compare_grid_vs_random_search()")
print("- save_randomized_search_results()")

# ============================================================================
# TASK 5.4: CREATE HYPERPARAMETER OPTIMIZATION PIPELINE
# ============================================================================

def calculate_search_space_size(param_grid):
    """
    Calculate the total size of the parameter search space.
    
    Parameters:
    -----------
    param_grid : dict
        Parameter grid dictionary
    
    Returns:
    --------
    int : Total number of parameter combinations
    """
    total_combinations = 1
    for param_values in param_grid.values():
        if isinstance(param_values, list):
            total_combinations *= len(param_values)
        else:
            total_combinations *= 1
    
    return total_combinations

def select_optimization_method(param_grid, max_grid_size=100, time_budget_minutes=30):
    """
    Automatically select optimization method based on search space size and time budget.
    
    Parameters:
    -----------
    param_grid : dict
        Parameter grid dictionary
    max_grid_size : int
        Maximum search space size for GridSearchCV
    time_budget_minutes : int
        Available time budget in minutes
    
    Returns:
    --------
    dict : Optimization method configuration
    """
    search_space_size = calculate_search_space_size(param_grid)
    
    # Decision logic for method selection
    if search_space_size <= max_grid_size:
        method = 'grid_search'
        n_iter = search_space_size  # Use all combinations
        reason = f"Small search space ({search_space_size} combinations)"
    elif search_space_size <= 500:
        method = 'random_search'
        n_iter = min(100, search_space_size // 2)
        reason = f"Medium search space ({search_space_size} combinations)"
    else:
        method = 'random_search'
        # Estimate iterations based on time budget
        estimated_time_per_iter = 0.5  # minutes per iteration (rough estimate)
        max_iter_by_time = int(time_budget_minutes / estimated_time_per_iter)
        n_iter = min(200, max_iter_by_time, search_space_size // 5)
        reason = f"Large search space ({search_space_size} combinations)"
    
    return {
        'method': method,
        'n_iter': n_iter,
        'search_space_size': search_space_size,
        'reason': reason
    }

def optimize_model_hyperparameters(model_name, X_train, y_train, 
                                 search_budget='auto', cv=5, scoring='f1',
                                 time_budget_minutes=30, n_jobs=-1, verbose=1):
    """
    Unified hyperparameter optimization interface for any model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model ('random_forest', 'svm', 'logistic_regression', 'gradient_boosting')
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_budget : str or dict
        Search budget ('auto', 'small', 'medium', 'large') or custom config
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for optimization
    time_budget_minutes : int
        Time budget for optimization in minutes
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    
    Returns:
    --------
    dict : Comprehensive optimization results
    """
    print(f"\n🎯 Starting hyperparameter optimization for {model_name}")
    print(f"📊 Search budget: {search_budget}")
    print(f"⏰ Time budget: {time_budget_minutes} minutes")
    print("=" * 60)
    
    # Get parameter grid for the model
    if search_budget == 'auto':
        param_grid = create_adaptive_param_grid(model_name, search_budget='medium')
    elif isinstance(search_budget, str):
        param_grid = create_adaptive_param_grid(model_name, search_budget=search_budget)
    else:
        param_grid = search_budget
    
    # Automatically select optimization method
    method_config = select_optimization_method(
        param_grid, 
        max_grid_size=100, 
        time_budget_minutes=time_budget_minutes
    )
    
    print(f"🔍 Selected method: {method_config['method']}")
    print(f"📈 Search space size: {method_config['search_space_size']}")
    print(f"💡 Reason: {method_config['reason']}")
    print(f"🔢 Iterations: {method_config['n_iter']}")
    
    # Execute optimization based on selected method
    start_time = datetime.now()
    
    try:
        if method_config['method'] == 'grid_search':
            # Use GridSearchCV - handle custom parameter grids directly
            results = run_direct_grid_search(
                model_name=model_name,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
        
        else:  # random_search
            # Use RandomizedSearchCV
            if model_name == 'random_forest':
                results = optimize_random_forest_randomized(
                    X_train, y_train, n_iter=method_config['n_iter'], cv=cv, scoring=scoring
                )
            elif model_name == 'svm':
                results = optimize_svm_randomized(
                    X_train, y_train, n_iter=method_config['n_iter'], cv=cv, scoring=scoring
                )
            elif model_name == 'logistic_regression':
                results = optimize_logistic_regression_randomized(
                    X_train, y_train, n_iter=method_config['n_iter'], cv=cv, scoring=scoring
                )
            elif model_name == 'gradient_boosting':
                results = optimize_gradient_boosting_randomized(
                    X_train, y_train, n_iter=method_config['n_iter'], cv=cv, scoring=scoring
                )
            else:
                raise ValueError(f"Unsupported model for random search: {model_name}")
        
        # Add method information to results
        results['optimization_method'] = method_config['method']
        results['search_space_size'] = method_config['search_space_size']
        results['iterations_performed'] = method_config['n_iter']
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        return {
            'model_name': model_name,
            'optimization_method': method_config['method'],
            'success': False,
            'error': str(e),
            'execution_time': (datetime.now() - start_time).total_seconds()
        }
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()
    results['execution_time'] = execution_time
    results['success'] = True
    
    print(f"\n✅ Optimization completed successfully!")
    print(f"⏱️  Total time: {execution_time:.1f} seconds")
    print(f"🏆 Best score: {results['best_score']:.4f}")
    
    return results

def optimize_multiple_models(model_list, X_train, y_train, 
                           search_budget='auto', cv=5, scoring='f1',
                           time_budget_per_model=30, n_jobs=-1, verbose=1):
    """
    Optimize hyperparameters for multiple models and compare results.
    
    Parameters:
    -----------
    model_list : list
        List of model names to optimize
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_budget : str or dict
        Search budget for each model
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for optimization
    time_budget_per_model : int
        Time budget per model in minutes
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
    
    Returns:
    --------
    dict : Results for all models with comparison
    """
    print(f"\n🚀 Starting multi-model hyperparameter optimization")
    print(f"🎯 Models: {model_list}")
    print(f"📊 Search budget: {search_budget}")
    print(f"⏰ Time budget per model: {time_budget_per_model} minutes")
    print("=" * 70)
    
    all_results = {}
    optimization_summary = []
    
    total_start_time = datetime.now()
    
    for i, model_name in enumerate(model_list, 1):
        print(f"\n📈 Optimizing model {i}/{len(model_list)}: {model_name}")
        print("-" * 50)
        
        # Optimize individual model
        model_results = optimize_model_hyperparameters(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            search_budget=search_budget,
            cv=cv,
            scoring=scoring,
            time_budget_minutes=time_budget_per_model,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Store results
        all_results[model_name] = model_results
        
        # Add to summary
        if model_results['success']:
            optimization_summary.append({
                'model': model_name,
                'best_score': model_results['best_score'],
                'execution_time': model_results['execution_time'],
                'method': model_results['optimization_method'],
                'iterations': model_results.get('iterations_performed', 'N/A')
            })
        else:
            optimization_summary.append({
                'model': model_name,
                'best_score': 0.0,
                'execution_time': model_results['execution_time'],
                'method': 'failed',
                'iterations': 0,
                'error': model_results.get('error', 'Unknown error')
            })
    
    # Calculate total execution time
    total_execution_time = (datetime.now() - total_start_time).total_seconds()
    
    # Create comparison summary
    comparison_results = create_optimization_comparison(optimization_summary)
    
    # Final results
    final_results = {
        'individual_results': all_results,
        'comparison_summary': comparison_results,
        'optimization_summary': optimization_summary,
        'total_execution_time': total_execution_time,
        'models_optimized': len(model_list),
        'successful_optimizations': sum(1 for r in all_results.values() if r['success'])
    }
    
    # Print final summary
    print_optimization_summary(final_results)
    
    return final_results

def create_optimization_comparison(optimization_summary):
    """
    Create a comprehensive comparison of optimization results.
    
    Parameters:
    -----------
    optimization_summary : list
        List of optimization summary dictionaries
    
    Returns:
    --------
    dict : Comparison results and rankings
    """
    # Convert to DataFrame for easier analysis
    df_summary = pd.DataFrame(optimization_summary)
    
    # Filter successful optimizations
    successful_df = df_summary[df_summary['best_score'] > 0].copy()
    
    if len(successful_df) == 0:
        return {
            'best_model': None,
            'rankings': [],
            'performance_comparison': {},
            'efficiency_analysis': {},
            'recommendations': ["No successful optimizations to compare"]
        }
    
    # Rank by performance
    successful_df = successful_df.sort_values('best_score', ascending=False)
    
    # Performance comparison
    performance_comparison = {
        'best_model': successful_df.iloc[0]['model'],
        'best_score': successful_df.iloc[0]['best_score'],
        'score_range': {
            'min': successful_df['best_score'].min(),
            'max': successful_df['best_score'].max(),
            'std': successful_df['best_score'].std()
        }
    }
    
    # Efficiency analysis (score per minute)
    successful_df['efficiency'] = successful_df['best_score'] / (successful_df['execution_time'] / 60)
    efficiency_ranking = successful_df.sort_values('efficiency', ascending=False)
    
    efficiency_analysis = {
        'most_efficient': efficiency_ranking.iloc[0]['model'],
        'efficiency_score': efficiency_ranking.iloc[0]['efficiency'],
        'time_comparison': {
            'fastest': successful_df.loc[successful_df['execution_time'].idxmin(), 'model'],
            'slowest': successful_df.loc[successful_df['execution_time'].idxmax(), 'model'],
            'avg_time': successful_df['execution_time'].mean()
        }
    }
    
    # Create rankings
    rankings = []
    for i, (_, row) in enumerate(successful_df.iterrows(), 1):
        rankings.append({
            'rank': i,
            'model': row['model'],
            'score': row['best_score'],
            'time': row['execution_time'],
            'method': row['method'],
            'efficiency': row['efficiency']
        })
    
    # Generate recommendations
    recommendations = generate_optimization_recommendations(
        performance_comparison, efficiency_analysis, rankings
    )
    
    return {
        'best_model': performance_comparison['best_model'],
        'rankings': rankings,
        'performance_comparison': performance_comparison,
        'efficiency_analysis': efficiency_analysis,
        'recommendations': recommendations
    }

def generate_optimization_recommendations(performance_comparison, efficiency_analysis, rankings):
    """
    Generate actionable recommendations based on optimization results.
    
    Parameters:
    -----------
    performance_comparison : dict
        Performance comparison results
    efficiency_analysis : dict
        Efficiency analysis results
    rankings : list
        Model rankings
    
    Returns:
    --------
    list : List of recommendation strings
    """
    recommendations = []
    
    # Performance recommendations
    best_model = performance_comparison['best_model']
    best_score = performance_comparison['best_score']
    score_std = performance_comparison['score_range']['std']
    
    recommendations.append(f"🏆 Best performing model: {best_model} (score: {best_score:.4f})")
    
    if score_std < 0.05:
        recommendations.append("📊 Models show similar performance - consider efficiency and interpretability")
    else:
        recommendations.append(f"📈 Significant performance differences detected (std: {score_std:.4f})")
    
    # Efficiency recommendations
    most_efficient = efficiency_analysis['most_efficient']
    if most_efficient != best_model:
        recommendations.append(f"⚡ Most efficient model: {most_efficient} (best score/time ratio)")
        recommendations.append("💡 Consider efficiency vs. performance trade-off for production deployment")
    
    # Method recommendations
    method_counts = {}
    for ranking in rankings:
        method = ranking['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    if 'grid_search' in method_counts and 'random_search' in method_counts:
        recommendations.append("🔍 Mixed optimization methods used - automatic selection working well")
    
    # Score threshold recommendations
    if best_score < 0.8:
        recommendations.append("⚠️  Best score below 80% - consider feature engineering or data quality improvements")
    elif best_score > 0.95:
        recommendations.append("🎯 Excellent performance achieved - validate against overfitting")
    
    # Model-specific recommendations
    top_3_models = [r['model'] for r in rankings[:3]]
    if 'random_forest' in top_3_models:
        recommendations.append("🌳 Random Forest in top 3 - good baseline with interpretability")
    if 'gradient_boosting' in top_3_models:
        recommendations.append("🚀 Gradient Boosting in top 3 - consider ensemble methods")
    
    return recommendations

def print_optimization_summary(results):
    """
    Print a comprehensive summary of optimization results.
    
    Parameters:
    -----------
    results : dict
        Complete optimization results
    """
    print("\n" + "=" * 70)
    print("🎯 HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    # Overall statistics
    print(f"📊 Models optimized: {results['models_optimized']}")
    print(f"✅ Successful optimizations: {results['successful_optimizations']}")
    print(f"⏱️  Total execution time: {results['total_execution_time']:.1f} seconds")
    
    # Best model information
    if results['comparison_summary']['best_model']:
        best_model = results['comparison_summary']['best_model']
        best_score = results['comparison_summary']['performance_comparison']['best_score']
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"🎯 Best Score: {best_score:.4f}")
        
        # Print rankings
        print(f"\n📈 MODEL RANKINGS:")
        print("-" * 50)
        for ranking in results['comparison_summary']['rankings']:
            print(f"{ranking['rank']}. {ranking['model']:<20} "
                  f"Score: {ranking['score']:.4f} "
                  f"Time: {ranking['time']:.1f}s "
                  f"Method: {ranking['method']}")
        
        # Print recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        for i, rec in enumerate(results['comparison_summary']['recommendations'], 1):
            print(f"{i}. {rec}")
    
    else:
        print("\n❌ No successful optimizations to summarize")
    
    print("\n" + "=" * 70)

def validate_optimization_results(results, min_score_threshold=0.6):
    """
    Validate optimization results and flag potential issues.
    
    Parameters:
    -----------
    results : dict
        Optimization results to validate
    min_score_threshold : float
        Minimum acceptable score threshold
    
    Returns:
    --------
    dict : Validation results and warnings
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check if optimization was successful
    if not results.get('success', False):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Optimization failed to complete")
        return validation_results
    
    # Check score threshold
    best_score = results.get('best_score', 0)
    if best_score < min_score_threshold:
        validation_results['warnings'].append(
            f"Best score ({best_score:.4f}) below threshold ({min_score_threshold})"
        )
        validation_results['recommendations'].append(
            "Consider feature engineering, data quality improvements, or different algorithms"
        )
    
    # Check for overfitting indicators
    cv_scores = results.get('cv_scores', [])
    if len(cv_scores) > 0:
        cv_std = np.std(cv_scores)
        if cv_std > 0.1:
            validation_results['warnings'].append(
                f"High CV score variance ({cv_std:.4f}) - possible overfitting"
            )
            validation_results['recommendations'].append(
                "Consider regularization or simpler models"
            )
    
    # Check execution time
    execution_time = results.get('execution_time', 0)
    if execution_time > 1800:  # 30 minutes
        validation_results['warnings'].append(
            f"Long optimization time ({execution_time/60:.1f} minutes)"
        )
        validation_results['recommendations'].append(
            "Consider reducing search space or using RandomizedSearchCV"
        )
    
    # Check parameter diversity
    best_params = results.get('best_params', {})
    if len(best_params) == 0:
        validation_results['warnings'].append("No parameters were optimized")
    
    return validation_results

def save_optimization_pipeline_results(results, model_name, filename=None):
    """
    Save complete optimization pipeline results to file.
    
    Parameters:
    -----------
    results : dict
        Complete optimization results
    model_name : str
        Name of the model
    filename : str, optional
        Custom filename for saving results
    
    Returns:
    --------
    str : Path to saved file
    """
    # Create filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_pipeline_{model_name}_{timestamp}.joblib"
    
    # Ensure directory exists
    os.makedirs('models/optimization_results', exist_ok=True)
    filepath = os.path.join('models/optimization_results', filename)
    
    # Add metadata
    results['metadata'] = {
        'model_name': model_name,
        'save_timestamp': datetime.now().isoformat(),
        'pipeline_version': '1.0',
        'validation_results': validate_optimization_results(results)
    }
    
    # Save results
    joblib.dump(results, filepath)
    
    print(f"Optimization pipeline results saved to: {filepath}")
    return filepath

print("\n" + "=" * 70)
print("✅ HYPERPARAMETER OPTIMIZATION PIPELINE CREATED SUCCESSFULLY!")
print("=" * 70)
print("Available functions:")
print("- calculate_search_space_size()")
print("- select_optimization_method()")
print("- optimize_model_hyperparameters()")
print("- optimize_multiple_models()")
print("- create_optimization_comparison()")
print("- generate_optimization_recommendations()")
print("- print_optimization_summary()")
print("- validate_optimization_results()")
print("- save_optimization_pipeline_results()")
print("\n🎯 The pipeline provides:")
print("✓ Unified interface for all optimization methods")
print("✓ Automatic method selection based on search space size")
print("✓ Best model selection and validation")
print("✓ Multi-model comparison and recommendations")
print("✓ Comprehensive result validation and reporting")
print("=" * 70)