"""
ROC curve analysis for binary classification models.

This module provides comprehensive ROC curve analysis including
AUC calculation, optimal threshold selection, and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any
import warnings


def calculate_roc_metrics(y_true: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive ROC metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary containing ROC metrics and curve data
        
    Requirements: 3.1, 3.3
    """
    try:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        # Calculate additional metrics
        specificity = 1 - fpr
        sensitivity = tpr
        
        # Find threshold for 95% sensitivity
        sens_95_idx = np.where(tpr >= 0.95)[0]
        if len(sens_95_idx) > 0:
            sens_95_threshold = thresholds[sens_95_idx[0]]
            sens_95_fpr = fpr[sens_95_idx[0]]
        else:
            sens_95_threshold = np.nan
            sens_95_fpr = np.nan
        
        # Find threshold for 95% specificity
        spec_95_idx = np.where(specificity >= 0.95)[0]
        if len(spec_95_idx) > 0:
            spec_95_threshold = thresholds[spec_95_idx[-1]]
            spec_95_tpr = tpr[spec_95_idx[-1]]
        else:
            spec_95_threshold = np.nan
            spec_95_tpr = np.nan
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_fpr': optimal_fpr,
            'optimal_tpr': optimal_tpr,
            'optimal_specificity': 1 - optimal_fpr,
            'optimal_sensitivity': optimal_tpr,
            'youden_j': j_scores[optimal_idx],
            'sens_95_threshold': sens_95_threshold,
            'sens_95_fpr': sens_95_fpr,
            'spec_95_threshold': spec_95_threshold,
            'spec_95_tpr': spec_95_tpr
        }
        
    except Exception as e:
        print(f"Error calculating ROC metrics: {e}")
        return {}


def compare_roc_curves(models_results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Compare ROC curves and AUC scores across multiple models.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
                       Each result should contain 'y_true' and 'y_pred_proba'
        
    Returns:
        DataFrame with ROC comparison metrics
        
    Requirements: 3.1, 3.3
    """
    comparison_data = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred_proba = results.get('y_pred_proba', None)
        
        if y_pred_proba is None:
            print(f"Warning: No probabilities available for {model_name}")
            continue
        
        try:
            # Handle binary classification probabilities
            if y_pred_proba.ndim == 2:
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba
            
            # Calculate ROC metrics
            roc_metrics = calculate_roc_metrics(y_true, y_proba_pos)
            
            if roc_metrics:
                comparison_data.append({
                    'Model': model_name,
                    'AUC': roc_metrics['auc'],
                    'Optimal_Threshold': roc_metrics['optimal_threshold'],
                    'Optimal_Sensitivity': roc_metrics['optimal_sensitivity'],
                    'Optimal_Specificity': roc_metrics['optimal_specificity'],
                    'Youden_J': roc_metrics['youden_j'],
                    'Sens_95_Threshold': roc_metrics['sens_95_threshold'],
                    'Sens_95_FPR': roc_metrics['sens_95_fpr'],
                    'Spec_95_Threshold': roc_metrics['spec_95_threshold'],
                    'Spec_95_TPR': roc_metrics['spec_95_tpr']
                })
                
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('AUC', ascending=False)
        return df.round(4)
    else:
        return pd.DataFrame()


def plot_detailed_roc_analysis(models_results: Dict[str, Dict[str, np.ndarray]],
                              title: str = 'Detailed ROC Analysis',
                              figsize: Tuple[int, int] = (15, 10),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create detailed ROC analysis plot with multiple subplots.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        title: Main title for the figure
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.1, 3.3
    """
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # ROC curves
    ax2 = fig.add_subplot(gs[0, 1])  # AUC comparison
    ax3 = fig.add_subplot(gs[1, 0])  # Threshold analysis
    ax4 = fig.add_subplot(gs[1, 1])  # Sensitivity vs Specificity
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    auc_scores = []
    model_names = []
    
    # Plot 1: ROC Curves
    for (model_name, results), color in zip(models_results.items(), colors):
        y_true = results['y_true']
        y_pred_proba = results.get('y_pred_proba', None)
        
        if y_pred_proba is None:
            continue
        
        try:
            # Handle binary classification probabilities
            if y_pred_proba.ndim == 2:
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba
            
            # Calculate ROC metrics
            roc_metrics = calculate_roc_metrics(y_true, y_proba_pos)
            
            if roc_metrics:
                fpr, tpr = roc_metrics['fpr'], roc_metrics['tpr']
                roc_auc = roc_metrics['auc']
                
                # Plot ROC curve
                ax1.plot(fpr, tpr, color=color, lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                # Mark optimal point
                ax1.plot(roc_metrics['optimal_fpr'], roc_metrics['optimal_tpr'],
                        'o', color=color, markersize=8)
                
                auc_scores.append(roc_auc)
                model_names.append(model_name)
                
                # Plot 3: Threshold analysis (J-statistic)
                j_scores = tpr - fpr
                ax3.plot(roc_metrics['thresholds'], j_scores, color=color, 
                        label=f'{model_name}', alpha=0.7)
                
                # Plot 4: Sensitivity vs Specificity
                specificity = 1 - fpr
                ax4.plot(specificity, tpr, color=color, lw=2,
                        label=f'{model_name}')
                
        except Exception as e:
            print(f"Error plotting for {model_name}: {e}")
            continue
    
    # Finalize Plot 1: ROC Curves
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC Comparison Bar Chart
    if auc_scores:
        bars = ax2.bar(range(len(model_names)), auc_scores, color=colors[:len(model_names)])
        ax2.set_xlabel('Models')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('AUC Comparison')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Finalize Plot 3: Threshold Analysis
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Youden J Statistic')
    ax3.set_title('Optimal Threshold Analysis')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Finalize Plot 4: Sensitivity vs Specificity
    ax4.set_xlabel('Specificity')
    ax4.set_ylabel('Sensitivity')
    ax4.set_title('Sensitivity vs Specificity')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.0])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def find_optimal_threshold(y_true: np.ndarray, 
                          y_pred_proba: np.ndarray,
                          method: str = 'youden') -> Dict[str, float]:
    """
    Find optimal classification threshold using different methods.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        method: Method to use ('youden', 'closest_to_topleft', 'f1_optimal')
        
    Returns:
        Dictionary with optimal threshold and corresponding metrics
        
    Requirements: 3.1, 3.3
    """
    from sklearn.metrics import f1_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    if method == 'youden':
        # Youden's J statistic: maximize (sensitivity + specificity - 1)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
    elif method == 'closest_to_topleft':
        # Point closest to top-left corner (0, 1)
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        optimal_idx = np.argmin(distances)
        
    elif method == 'f1_optimal':
        # Threshold that maximizes F1-score
        f1_scores = []
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # Calculate predictions with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    return {
        'threshold': optimal_threshold,
        'fpr': optimal_fpr,
        'tpr': optimal_tpr,
        'sensitivity': optimal_tpr,
        'specificity': 1 - optimal_fpr,
        'method': method,
        'predictions': y_pred_optimal
    }


def bootstrap_auc_confidence_interval(y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray,
                                    n_bootstrap: int = 1000,
                                    confidence_level: float = 0.95,
                                    random_state: int = 42) -> Dict[str, float]:
    """
    Calculate confidence interval for AUC using bootstrap sampling.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with AUC statistics and confidence interval
        
    Requirements: 3.1, 3.3
    """
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        
        # Calculate AUC for bootstrap sample
        try:
            auc_boot = roc_auc_score(y_true_boot, y_pred_proba_boot)
            bootstrap_aucs.append(auc_boot)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_aucs, lower_percentile)
    ci_upper = np.percentile(bootstrap_aucs, upper_percentile)
    
    # Original AUC
    original_auc = roc_auc_score(y_true, y_pred_proba)
    
    return {
        'auc': original_auc,
        'auc_mean': np.mean(bootstrap_aucs),
        'auc_std': np.std(bootstrap_aucs),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': len(bootstrap_aucs)
    }


def roc_analysis_summary(models_results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Generate comprehensive ROC analysis summary for multiple models.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        
    Returns:
        DataFrame with comprehensive ROC analysis results
        
    Requirements: 3.1, 3.3
    """
    summary_data = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred_proba = results.get('y_pred_proba', None)
        
        if y_pred_proba is None:
            continue
        
        try:
            # Handle binary classification probabilities
            if y_pred_proba.ndim == 2:
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba
            
            # Basic ROC metrics
            roc_metrics = calculate_roc_metrics(y_true, y_proba_pos)
            
            # Confidence interval
            ci_results = bootstrap_auc_confidence_interval(y_true, y_proba_pos)
            
            # Optimal thresholds using different methods
            youden_optimal = find_optimal_threshold(y_true, y_proba_pos, 'youden')
            f1_optimal = find_optimal_threshold(y_true, y_proba_pos, 'f1_optimal')
            
            summary_data.append({
                'Model': model_name,
                'AUC': roc_metrics['auc'],
                'AUC_CI_Lower': ci_results['ci_lower'],
                'AUC_CI_Upper': ci_results['ci_upper'],
                'Youden_Threshold': youden_optimal['threshold'],
                'Youden_Sensitivity': youden_optimal['sensitivity'],
                'Youden_Specificity': youden_optimal['specificity'],
                'F1_Optimal_Threshold': f1_optimal['threshold'],
                'F1_Optimal_Sensitivity': f1_optimal['sensitivity'],
                'F1_Optimal_Specificity': f1_optimal['specificity']
            })
            
        except Exception as e:
            print(f"Error in ROC analysis for {model_name}: {e}")
            continue
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.sort_values('AUC', ascending=False)
        return df.round(4)
    else:
        return pd.DataFrame()