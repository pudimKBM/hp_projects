"""
Feature importance analysis for model interpretation.

This module provides functions to extract and analyze feature importance
from different types of machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)


def extract_tree_importance(
    model: Union[RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model (RandomForest, GradientBoosting, DecisionTree)
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and their importance scores
        
    Raises:
        ValueError: If model doesn't have feature_importances_ attribute
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {type(model).__name__} doesn't have feature_importances_ attribute")
        
        importance_scores = model.feature_importances_
        
        if len(importance_scores) != len(feature_names):
            raise ValueError(f"Number of importance scores ({len(importance_scores)}) "
                           f"doesn't match number of features ({len(feature_names)})")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores,
            'importance_type': 'tree_based'
        })
        
        # Sort by importance descending
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"Extracted tree-based importance for {len(feature_names)} features")
        return importance_df
        
    except Exception as e:
        logger.error(f"Error extracting tree importance: {str(e)}")
        raise


def calculate_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = 'accuracy'
) -> pd.DataFrame:
    """
    Calculate permutation importance for any model type.
    
    Args:
        model: Trained model (any sklearn-compatible model)
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        random_state: Random state for reproducibility
        scoring: Scoring metric for importance calculation
        
    Returns:
        DataFrame with features and their permutation importance scores
    """
    try:
        logger.info(f"Calculating permutation importance with {n_repeats} repeats")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'importance_type': 'permutation'
        })
        
        # Sort by importance descending
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"Calculated permutation importance for {len(feature_names)} features")
        return importance_df
        
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {str(e)}")
        raise


def rank_feature_importance(
    importance_df: pd.DataFrame,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Rank features by importance and optionally return top N features.
    
    Args:
        importance_df: DataFrame with feature importance scores
        top_n: Number of top features to return (None for all)
        
    Returns:
        DataFrame with ranked features
    """
    try:
        # Add rank column
        ranked_df = importance_df.copy()
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        # Calculate relative importance (percentage)
        total_importance = ranked_df['importance'].sum()
        if total_importance > 0:
            ranked_df['relative_importance'] = (ranked_df['importance'] / total_importance) * 100
        else:
            ranked_df['relative_importance'] = 0
        
        # Calculate cumulative importance
        ranked_df['cumulative_importance'] = ranked_df['relative_importance'].cumsum()
        
        # Return top N if specified
        if top_n is not None:
            ranked_df = ranked_df.head(top_n)
        
        logger.info(f"Ranked {len(ranked_df)} features by importance")
        return ranked_df
        
    except Exception as e:
        logger.error(f"Error ranking feature importance: {str(e)}")
        raise


def compare_feature_importance(
    importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare feature importance across multiple models.
    
    Args:
        importance_dict: Dictionary with model names as keys and importance DataFrames as values
        top_n: Number of top features to include in comparison
        
    Returns:
        DataFrame with feature importance comparison across models
    """
    try:
        if not importance_dict:
            raise ValueError("importance_dict cannot be empty")
        
        # Get all unique features
        all_features = set()
        for df in importance_dict.values():
            all_features.update(df['feature'].tolist())
        
        # Create comparison DataFrame
        comparison_data = []
        
        for feature in all_features:
            feature_data = {'feature': feature}
            
            for model_name, importance_df in importance_dict.items():
                # Find importance for this feature
                feature_row = importance_df[importance_df['feature'] == feature]
                if not feature_row.empty:
                    importance_score = feature_row['importance'].iloc[0]
                    rank = feature_row.index[0] + 1  # 1-based rank
                else:
                    importance_score = 0
                    rank = len(importance_df) + 1
                
                feature_data[f'{model_name}_importance'] = importance_score
                feature_data[f'{model_name}_rank'] = rank
            
            comparison_data.append(feature_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate average importance and rank across models
        importance_cols = [col for col in comparison_df.columns if col.endswith('_importance')]
        rank_cols = [col for col in comparison_df.columns if col.endswith('_rank')]
        
        comparison_df['avg_importance'] = comparison_df[importance_cols].mean(axis=1)
        comparison_df['avg_rank'] = comparison_df[rank_cols].mean(axis=1)
        
        # Sort by average importance
        comparison_df = comparison_df.sort_values('avg_importance', ascending=False).reset_index(drop=True)
        
        # Return top N features
        result_df = comparison_df.head(top_n)
        
        logger.info(f"Compared feature importance across {len(importance_dict)} models")
        return result_df
        
    except Exception as e:
        logger.error(f"Error comparing feature importance: {str(e)}")
        raise


def get_feature_importance_summary(
    importance_df: pd.DataFrame,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Generate summary statistics for feature importance.
    
    Args:
        importance_df: DataFrame with feature importance scores
        threshold: Minimum importance threshold for "important" features
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        total_features = len(importance_df)
        important_features = len(importance_df[importance_df['importance'] >= threshold])
        
        summary = {
            'total_features': total_features,
            'important_features': important_features,
            'important_features_pct': (important_features / total_features) * 100 if total_features > 0 else 0,
            'max_importance': importance_df['importance'].max(),
            'min_importance': importance_df['importance'].min(),
            'mean_importance': importance_df['importance'].mean(),
            'std_importance': importance_df['importance'].std(),
            'top_feature': importance_df.iloc[0]['feature'] if not importance_df.empty else None,
            'top_importance': importance_df.iloc[0]['importance'] if not importance_df.empty else 0
        }
        
        # Features contributing to 80% of total importance
        if 'cumulative_importance' in importance_df.columns:
            features_80pct = len(importance_df[importance_df['cumulative_importance'] <= 80])
            summary['features_80pct'] = features_80pct
        
        logger.info("Generated feature importance summary")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating feature importance summary: {str(e)}")
        raise


def filter_important_features(
    importance_df: pd.DataFrame,
    method: str = 'threshold',
    threshold: float = 0.01,
    top_n: Optional[int] = None,
    cumulative_threshold: float = 95.0
) -> List[str]:
    """
    Filter features based on importance criteria.
    
    Args:
        importance_df: DataFrame with feature importance scores
        method: Filtering method ('threshold', 'top_n', 'cumulative')
        threshold: Minimum importance threshold
        top_n: Number of top features to select
        cumulative_threshold: Cumulative importance threshold (percentage)
        
    Returns:
        List of important feature names
    """
    try:
        if method == 'threshold':
            important_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
        
        elif method == 'top_n':
            if top_n is None:
                raise ValueError("top_n must be specified when using 'top_n' method")
            important_features = importance_df.head(top_n)['feature'].tolist()
        
        elif method == 'cumulative':
            if 'cumulative_importance' not in importance_df.columns:
                # Calculate cumulative importance if not present
                total_importance = importance_df['importance'].sum()
                cumulative_pct = (importance_df['importance'].cumsum() / total_importance) * 100
                importance_df = importance_df.copy()
                importance_df['cumulative_importance'] = cumulative_pct
            
            important_features = importance_df[
                importance_df['cumulative_importance'] <= cumulative_threshold
            ]['feature'].tolist()
        
        else:
            raise ValueError(f"Unknown filtering method: {method}")
        
        logger.info(f"Filtered {len(important_features)} important features using {method} method")
        return important_features
        
    except Exception as e:
        logger.error(f"Error filtering important features: {str(e)}")
        raise