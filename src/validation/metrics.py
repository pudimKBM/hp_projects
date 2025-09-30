"""
Comprehensive metrics calculation for model validation.

This module implements functions to calculate various classification metrics
including precision, recall, F1-score, AUC-ROC, accuracy, and balanced accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
from typing import Dict, List, Tuple, Any, Optional
import warnings


def calculate_comprehensive_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  y_pred_proba: Optional[np.ndarray] = None,
                                  average: str = 'binary',
                                  pos_label: int = 1) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for a single model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for AUC-ROC)
        average: Averaging strategy for multi-class ('binary', 'macro', 'weighted')
        pos_label: Positive class label for binary classification
        
    Returns:
        Dictionary containing all calculated metrics
        
    Requirements: 3.1, 3.4
    """
    metrics = {}
    
    try:
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-score
        metrics['precision'] = precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        
        # AUC-ROC (if probabilities are provided)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    # Use probabilities for positive class
                    if y_pred_proba.ndim == 2:
                        y_proba_pos = y_pred_proba[:, 1]
                    else:
                        y_proba_pos = y_pred_proba
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
            except ValueError as e:
                warnings.warn(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan
            
        # Additional metrics for binary classification
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Specificity (True Negative Rate)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Sensitivity (same as recall, but explicit)
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Positive Predictive Value (same as precision, but explicit)
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Negative Predictive Value
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # Matthews Correlation Coefficient
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if mcc_denominator != 0:
                metrics['mcc'] = (tp * tn - fp * fn) / mcc_denominator
            else:
                metrics['mcc'] = 0.0
                
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return default metrics in case of error
        metrics = {
            'accuracy': 0.0, 'balanced_accuracy': 0.0, 'precision': 0.0,
            'recall': 0.0, 'f1_score': 0.0, 'roc_auc': np.nan
        }
    
    return metrics


def calculate_metrics_for_multiple_models(models_results: Dict[str, Dict[str, np.ndarray]],
                                        average: str = 'binary') -> pd.DataFrame:
    """
    Calculate comprehensive metrics for multiple models.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
                       Each result should contain 'y_true', 'y_pred', 'y_pred_proba'
        average: Averaging strategy for multi-class metrics
        
    Returns:
        DataFrame with models as rows and metrics as columns
        
    Requirements: 3.1, 3.4
    """
    all_metrics = {}
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        metrics = calculate_comprehensive_metrics(
            y_true, y_pred, y_pred_proba, average=average
        )
        all_metrics[model_name] = metrics
    
    # Convert to DataFrame for easy comparison
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Round to 4 decimal places for readability
    metrics_df = metrics_df.round(4)
    
    # Sort by F1-score (or another primary metric)
    if 'f1_score' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
    
    return metrics_df


def get_classification_report_dict(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 target_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get detailed classification report as dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        Classification report as dictionary
        
    Requirements: 3.1
    """
    try:
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        return report
    except Exception as e:
        print(f"Error generating classification report: {e}")
        return {}


def calculate_precision_recall_curve(y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        Tuple of (precision, recall, thresholds)
        
    Requirements: 3.1
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        return precision, recall, thresholds
    except Exception as e:
        print(f"Error calculating precision-recall curve: {e}")
        return np.array([]), np.array([]), np.array([])


def calculate_roc_curve(y_true: np.ndarray, 
                       y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve and AUC for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        Tuple of (fpr, tpr, thresholds, auc_score)
        
    Requirements: 3.1, 3.3
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_score
    except Exception as e:
        print(f"Error calculating ROC curve: {e}")
        return np.array([]), np.array([]), np.array([]), 0.0


def validate_predictions(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        y_pred_proba: Optional[np.ndarray] = None) -> bool:
    """
    Validate that predictions are in correct format and compatible with true labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        True if predictions are valid, False otherwise
        
    Requirements: 3.4
    """
    try:
        # Check array shapes
        if len(y_true) != len(y_pred):
            print(f"Shape mismatch: y_true {len(y_true)} vs y_pred {len(y_pred)}")
            return False
            
        if y_pred_proba is not None and len(y_true) != len(y_pred_proba):
            print(f"Shape mismatch: y_true {len(y_true)} vs y_pred_proba {len(y_pred_proba)}")
            return False
        
        # Check for valid label values
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        # Check if predicted labels are subset of true labels (allowing for missing classes in predictions)
        if not np.all(np.isin(unique_pred, unique_true)):
            print(f"Invalid predicted labels. True labels: {unique_true}, Predicted: {unique_pred}")
            return False
        
        # Check probability values if provided
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 1:
                # Binary classification probabilities
                if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
                    print("Probabilities must be between 0 and 1")
                    return False
            elif y_pred_proba.ndim == 2:
                # Multi-class probabilities
                if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
                    print("Probabilities must be between 0 and 1")
                    return False
                # Check if probabilities sum to 1 (with tolerance)
                prob_sums = np.sum(y_pred_proba, axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-3):
                    print("Multi-class probabilities should sum to 1")
                    return False
        
        return True
        
    except Exception as e:
        print(f"Error validating predictions: {e}")
        return False


def get_metric_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all available metrics.
    
    Returns:
        Dictionary mapping metric names to their descriptions
    """
    return {
        'accuracy': 'Overall accuracy: (TP + TN) / (TP + TN + FP + FN)',
        'balanced_accuracy': 'Balanced accuracy: average of recall for each class',
        'precision': 'Precision: TP / (TP + FP) - proportion of positive predictions that are correct',
        'recall': 'Recall/Sensitivity: TP / (TP + FN) - proportion of actual positives identified',
        'f1_score': 'F1-score: harmonic mean of precision and recall',
        'roc_auc': 'Area Under ROC Curve: measure of separability between classes',
        'specificity': 'Specificity: TN / (TN + FP) - proportion of actual negatives identified',
        'sensitivity': 'Sensitivity: same as recall, TP / (TP + FN)',
        'ppv': 'Positive Predictive Value: same as precision, TP / (TP + FP)',
        'npv': 'Negative Predictive Value: TN / (TN + FN)',
        'mcc': 'Matthews Correlation Coefficient: correlation between observed and predicted'
    }