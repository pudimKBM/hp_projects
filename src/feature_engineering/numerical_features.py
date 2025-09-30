# Numerical Feature Engineering Module

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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