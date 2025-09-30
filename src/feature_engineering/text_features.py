# Text Feature Engineering Module

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import re

from ..config import FEATURE_CONFIG


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