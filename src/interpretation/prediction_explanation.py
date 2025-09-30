"""
Prediction explanation functionality for model interpretation.

This module provides functions to explain individual predictions,
calculate confidence scores, and identify contributing features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)


def explain_prediction(
    model: BaseEstimator,
    X_sample: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Explain a single prediction by identifying contributing features.
    
    Args:
        model: Trained model
        X_sample: Single sample to explain (1D array)
        feature_names: List of feature names
        class_names: List of class names (optional)
        top_n: Number of top contributing features to return
        
    Returns:
        Dictionary with prediction explanation
    """
    try:
        # Ensure X_sample is 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = model.predict(X_sample)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_sample)[0]
        else:
            probabilities = None
        
        # Calculate feature contributions based on model type
        feature_contributions = _calculate_feature_contributions(
            model, X_sample, feature_names
        )
        
        # Get top contributing features
        top_features = _get_top_contributing_features(
            feature_contributions, top_n
        )
        
        # Prepare class information
        if class_names is None:
            if hasattr(model, 'classes_'):
                class_names = model.classes_.tolist()
            else:
                class_names = [f"Class_{i}" for i in range(len(probabilities) if probabilities is not None else 2)]
        
        explanation = {
            'prediction': prediction,
            'predicted_class': class_names[prediction] if isinstance(prediction, (int, np.integer)) else str(prediction),
            'probabilities': dict(zip(class_names, probabilities)) if probabilities is not None else None,
            'confidence': calculate_prediction_confidence(probabilities) if probabilities is not None else None,
            'feature_contributions': feature_contributions,
            'top_contributing_features': top_features,
            'sample_values': dict(zip(feature_names, X_sample[0]))
        }
        
        logger.info(f"Generated prediction explanation for sample")
        return explanation
        
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise


def calculate_prediction_confidence(
    probabilities: np.ndarray,
    method: str = 'max_prob'
) -> float:
    """
    Calculate confidence score for a prediction.
    
    Args:
        probabilities: Array of class probabilities
        method: Method for calculating confidence ('max_prob', 'entropy', 'margin')
        
    Returns:
        Confidence score (0-1)
    """
    try:
        if probabilities is None or len(probabilities) == 0:
            return 0.0
        
        if method == 'max_prob':
            # Maximum probability as confidence
            confidence = np.max(probabilities)
        
        elif method == 'entropy':
            # 1 - normalized entropy as confidence
            # Higher entropy = lower confidence
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        elif method == 'margin':
            # Margin between top two probabilities
            sorted_probs = np.sort(probabilities)[::-1]
            if len(sorted_probs) >= 2:
                confidence = sorted_probs[0] - sorted_probs[1]
            else:
                confidence = sorted_probs[0]
        
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        return float(confidence)
        
    except Exception as e:
        logger.error(f"Error calculating prediction confidence: {str(e)}")
        raise


def get_top_contributing_features(
    feature_contributions: Dict[str, float],
    top_n: int = 10,
    abs_values: bool = True
) -> List[Tuple[str, float]]:
    """
    Get top contributing features for a prediction.
    
    Args:
        feature_contributions: Dictionary of feature contributions
        top_n: Number of top features to return
        abs_values: Whether to sort by absolute values
        
    Returns:
        List of tuples (feature_name, contribution)
    """
    try:
        if abs_values:
            # Sort by absolute contribution values
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        else:
            # Sort by raw contribution values
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True
            )
        
        return sorted_features[:top_n]
        
    except Exception as e:
        logger.error(f"Error getting top contributing features: {str(e)}")
        raise


def explain_prediction_batch(
    model: BaseEstimator,
    X_batch: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Explain multiple predictions in batch.
    
    Args:
        model: Trained model
        X_batch: Batch of samples to explain
        feature_names: List of feature names
        class_names: List of class names (optional)
        top_n: Number of top contributing features per prediction
        
    Returns:
        List of prediction explanations
    """
    try:
        explanations = []
        
        for i, sample in enumerate(X_batch):
            try:
                explanation = explain_prediction(
                    model, sample, feature_names, class_names, top_n
                )
                explanation['sample_index'] = i
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Failed to explain sample {i}: {str(e)}")
                # Add placeholder explanation
                explanations.append({
                    'sample_index': i,
                    'error': str(e),
                    'prediction': None
                })
        
        logger.info(f"Generated explanations for {len(explanations)} samples")
        return explanations
        
    except Exception as e:
        logger.error(f"Error explaining prediction batch: {str(e)}")
        raise


def _calculate_feature_contributions(
    model: BaseEstimator,
    X_sample: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculate feature contributions based on model type.
    
    Args:
        model: Trained model
        X_sample: Single sample (2D array)
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature contributions
    """
    try:
        if isinstance(model, LogisticRegression):
            # Use coefficients for linear models
            contributions = _linear_model_contributions(model, X_sample, feature_names)
        
        elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            # Use feature importance weighted by feature values
            contributions = _tree_model_contributions(model, X_sample, feature_names)
        
        else:
            # Fallback: use feature values weighted by feature importance if available
            if hasattr(model, 'feature_importances_'):
                contributions = _tree_model_contributions(model, X_sample, feature_names)
            else:
                # Simple contribution based on feature values
                contributions = dict(zip(feature_names, X_sample[0]))
        
        return contributions
        
    except Exception as e:
        logger.error(f"Error calculating feature contributions: {str(e)}")
        # Return zero contributions as fallback
        return {name: 0.0 for name in feature_names}


def _linear_model_contributions(
    model: LogisticRegression,
    X_sample: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """Calculate contributions for linear models using coefficients."""
    try:
        # Get coefficients (for binary classification, use first class)
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            
            # Calculate contributions as coefficient * feature_value
            contributions = coef * X_sample[0]
            
            return dict(zip(feature_names, contributions))
        else:
            return {name: 0.0 for name in feature_names}
            
    except Exception as e:
        logger.error(f"Error calculating linear model contributions: {str(e)}")
        return {name: 0.0 for name in feature_names}


def _tree_model_contributions(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    X_sample: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """Calculate contributions for tree-based models using feature importance."""
    try:
        if hasattr(model, 'feature_importances_'):
            # Weight feature values by their importance
            importance = model.feature_importances_
            contributions = importance * X_sample[0]
            
            return dict(zip(feature_names, contributions))
        else:
            return {name: 0.0 for name in feature_names}
            
    except Exception as e:
        logger.error(f"Error calculating tree model contributions: {str(e)}")
        return {name: 0.0 for name in feature_names}


def _get_top_contributing_features(
    feature_contributions: Dict[str, float],
    top_n: int
) -> List[Dict[str, Any]]:
    """Get top contributing features with additional information."""
    try:
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_features = []
        for i, (feature, contribution) in enumerate(sorted_features[:top_n]):
            top_features.append({
                'rank': i + 1,
                'feature': feature,
                'contribution': contribution,
                'abs_contribution': abs(contribution),
                'direction': 'positive' if contribution > 0 else 'negative' if contribution < 0 else 'neutral'
            })
        
        return top_features
        
    except Exception as e:
        logger.error(f"Error getting top contributing features: {str(e)}")
        return []


def analyze_prediction_patterns(
    explanations: List[Dict[str, Any]],
    class_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze patterns in prediction explanations.
    
    Args:
        explanations: List of prediction explanations
        class_filter: Filter by specific predicted class (optional)
        
    Returns:
        Dictionary with pattern analysis
    """
    try:
        if class_filter:
            explanations = [exp for exp in explanations 
                          if exp.get('predicted_class') == class_filter]
        
        if not explanations:
            return {'error': 'No explanations to analyze'}
        
        # Analyze confidence patterns
        confidences = [exp.get('confidence', 0) for exp in explanations if exp.get('confidence') is not None]
        
        # Analyze top features across predictions
        all_top_features = []
        for exp in explanations:
            top_features = exp.get('top_contributing_features', [])
            for feature_info in top_features:
                all_top_features.append(feature_info['feature'])
        
        # Count feature frequency
        feature_counts = pd.Series(all_top_features).value_counts()
        
        analysis = {
            'total_predictions': len(explanations),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'most_important_features': feature_counts.head(10).to_dict(),
            'class_distribution': pd.Series([exp.get('predicted_class') for exp in explanations]).value_counts().to_dict()
        }
        
        logger.info(f"Analyzed patterns for {len(explanations)} predictions")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing prediction patterns: {str(e)}")
        raise