# Categorical Feature Engineering Module

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import csr_matrix, hstack


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