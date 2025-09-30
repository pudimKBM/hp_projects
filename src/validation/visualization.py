"""
Visualization functions for model validation and evaluation.

This module implements visualization functions for confusion matrices,
ROC curves, and cross-validation score distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Optional, Tuple, Any
import warnings


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (8, 6),
                         cmap: str = 'Blues',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix with seaborn/matplotlib.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes for labeling
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        figsize: Figure size (width, height)
        cmap: Colormap for the heatmap
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.2
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate labels if they are long
    if class_names and any(len(name) > 10 for name in class_names):
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiple_confusion_matrices(models_results: Dict[str, Dict[str, np.ndarray]],
                                   class_names: Optional[List[str]] = None,
                                   normalize: bool = False,
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrices for multiple models in a grid layout.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
                       Each result should contain 'y_true' and 'y_pred'
        class_names: Names of classes for labeling
        normalize: Whether to normalize confusion matrices
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.2
    """
    n_models = len(models_results)
    
    # Calculate grid dimensions
    n_cols = min(3, n_models)  # Maximum 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_models > 1 else axes
    
    for idx, (model_name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Plot on corresponding subplot
        ax = axes_flat[idx]
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True)
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices Comparison' + (' (Normalized)' if normalize else ''), 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(models_results: Dict[str, Dict[str, np.ndarray]],
                   title: str = 'ROC Curves Comparison',
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same plot.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
                       Each result should contain 'y_true' and 'y_pred_proba'
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.1, 3.3
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for (model_name, results), color in zip(models_results.items(), colors):
        y_true = results['y_true']
        y_pred_proba = results.get('y_pred_proba', None)
        
        if y_pred_proba is None:
            print(f"Warning: No probabilities available for {model_name}, skipping ROC curve")
            continue
        
        try:
            # Handle binary classification probabilities
            if y_pred_proba.ndim == 2:
                y_proba_pos = y_pred_proba[:, 1]  # Probabilities for positive class
            else:
                y_proba_pos = y_pred_proba
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
            
        except Exception as e:
            print(f"Error plotting ROC curve for {model_name}: {e}")
            continue
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
           label='Random Classifier (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cv_scores_distribution(cv_results: Dict[str, List[float]],
                               metric_name: str = 'F1-Score',
                               title: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cross-validation score distributions for multiple models.
    
    Args:
        cv_results: Dictionary with model names as keys and CV scores as values
        metric_name: Name of the metric being plotted
        title: Title for the plot (auto-generated if None)
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 2.3, 3.4
    """
    if title is None:
        title = f'Cross-Validation {metric_name} Distribution'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data for plotting
    model_names = list(cv_results.keys())
    scores_data = [cv_results[model] for model in model_names]
    
    # Box plot
    box_plot = ax1.boxplot(scores_data, labels=model_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title(f'{metric_name} Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(metric_name, fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if they are long
    if any(len(name) > 8 for name in model_names):
        ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot for more detailed distribution
    parts = ax2.violinplot(scores_data, positions=range(1, len(model_names) + 1))
    
    # Color the violin plots
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_title(f'{metric_name} Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(metric_name, fontsize=11)
    ax2.set_xticks(range(1, len(model_names) + 1))
    ax2.set_xticklabels(model_names)
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if they are long
    if any(len(name) > 8 for name in model_names):
        ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curves(models_results: Dict[str, Dict[str, np.ndarray]],
                                title: str = 'Precision-Recall Curves',
                                figsize: Tuple[int, int] = (10, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.1, 3.3
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for (model_name, results), color in zip(models_results.items(), colors):
        y_true = results['y_true']
        y_pred_proba = results.get('y_pred_proba', None)
        
        if y_pred_proba is None:
            print(f"Warning: No probabilities available for {model_name}, skipping PR curve")
            continue
        
        try:
            # Handle binary classification probabilities
            if y_pred_proba.ndim == 2:
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba
            
            # Calculate Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
            avg_precision = average_precision_score(y_true, y_proba_pos)
            
            # Plot PR curve
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{model_name} (AP = {avg_precision:.3f})')
            
        except Exception as e:
            print(f"Error plotting PR curve for {model_name}: {e}")
            continue
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance_comparison(feature_importance_dict: Dict[str, Dict[str, float]],
                                     top_n: int = 15,
                                     figsize: Tuple[int, int] = (12, 8),
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance comparison across multiple models.
    
    Args:
        feature_importance_dict: Dictionary with model names as keys and 
                               feature importance dictionaries as values
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    # Combine all feature importances
    all_features = set()
    for model_importance in feature_importance_dict.values():
        all_features.update(model_importance.keys())
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame(index=list(all_features))
    
    for model_name, importance in feature_importance_dict.items():
        importance_df[model_name] = pd.Series(importance)
    
    # Fill NaN with 0 and get top features by average importance
    importance_df = importance_df.fillna(0)
    importance_df['avg_importance'] = importance_df.mean(axis=1)
    top_features = importance_df.nlargest(top_n, 'avg_importance').drop('avg_importance', axis=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grouped bar chart
    top_features.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(f'Top {top_n} Feature Importance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig