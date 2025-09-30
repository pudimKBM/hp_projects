"""
Cross-validation analysis for model validation.

This module provides comprehensive cross-validation analysis including
detailed CV score analysis, statistical significance testing, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


def detailed_cv_analysis(model, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        cv: int = 5,
                        scoring: Union[str, List[str]] = None,
                        random_state: int = 42) -> Dict[str, Any]:
    """
    Perform detailed cross-validation analysis for a single model.
    
    Args:
        model: Scikit-learn model object
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: Scoring metrics to use
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing detailed CV results
        
    Requirements: 2.3, 3.4
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Ensure scoring is a list
    if isinstance(scoring, str):
        scoring = [scoring]
    
    # Create stratified k-fold
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    try:
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, 
            cv=cv_splitter, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics for each metric
        results = {
            'cv_folds': cv,
            'metrics': {},
            'raw_scores': cv_results
        }
        
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results['metrics'][metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_min': np.min(test_scores),
                'test_max': np.max(test_scores),
                'test_scores': test_scores,
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'train_min': np.min(train_scores),
                'train_max': np.max(train_scores),
                'train_scores': train_scores,
                'overfitting': np.mean(train_scores) - np.mean(test_scores)
            }
        
        return results
        
    except Exception as e:
        print(f"Error in cross-validation analysis: {e}")
        return {}


def cv_score_statistics(cv_scores: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for cross-validation scores.
    
    Args:
        cv_scores: Array of cross-validation scores
        
    Returns:
        Dictionary with statistical measures
        
    Requirements: 2.3, 3.4
    """
    return {
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'min': np.min(cv_scores),
        'max': np.max(cv_scores),
        'median': np.median(cv_scores),
        'q25': np.percentile(cv_scores, 25),
        'q75': np.percentile(cv_scores, 75),
        'iqr': np.percentile(cv_scores, 75) - np.percentile(cv_scores, 25),
        'cv_coefficient': np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) != 0 else np.inf,
        'confidence_interval_95': stats.t.interval(
            0.95, len(cv_scores) - 1, 
            loc=np.mean(cv_scores), 
            scale=stats.sem(cv_scores)
        )
    }


def compare_cv_results(models_cv_results: Dict[str, Dict[str, Any]],
                      metric: str = 'f1') -> pd.DataFrame:
    """
    Compare cross-validation results across multiple models.
    
    Args:
        models_cv_results: Dictionary with model names as keys and CV results as values
        metric: Metric to compare
        
    Returns:
        DataFrame with comparison statistics
        
    Requirements: 2.3, 3.4
    """
    comparison_data = []
    
    for model_name, cv_results in models_cv_results.items():
        if metric in cv_results.get('metrics', {}):
            metric_data = cv_results['metrics'][metric]
            
            # Calculate confidence interval
            test_scores = metric_data['test_scores']
            ci_95 = stats.t.interval(
                0.95, len(test_scores) - 1,
                loc=np.mean(test_scores),
                scale=stats.sem(test_scores)
            )
            
            comparison_data.append({
                'Model': model_name,
                'Mean': metric_data['test_mean'],
                'Std': metric_data['test_std'],
                'Min': metric_data['test_min'],
                'Max': metric_data['test_max'],
                'CV_Coefficient': metric_data['test_std'] / metric_data['test_mean'] if metric_data['test_mean'] != 0 else np.inf,
                'CI_Lower': ci_95[0],
                'CI_Upper': ci_95[1],
                'Overfitting': metric_data['overfitting']
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Mean', ascending=False)
        return df.round(4)
    else:
        return pd.DataFrame()


def statistical_significance_test(cv_scores_1: np.ndarray, 
                                cv_scores_2: np.ndarray,
                                test_type: str = 'paired_ttest') -> Dict[str, Any]:
    """
    Perform statistical significance test between two sets of CV scores.
    
    Args:
        cv_scores_1: CV scores for first model
        cv_scores_2: CV scores for second model
        test_type: Type of test ('paired_ttest', 'wilcoxon', 'mann_whitney')
        
    Returns:
        Dictionary with test results
        
    Requirements: 2.3, 3.4
    """
    try:
        if test_type == 'paired_ttest':
            # Paired t-test (assumes same CV folds)
            statistic, p_value = stats.ttest_rel(cv_scores_1, cv_scores_2)
            test_name = 'Paired t-test'
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric paired test)
            statistic, p_value = stats.wilcoxon(cv_scores_1, cv_scores_2)
            test_name = 'Wilcoxon signed-rank test'
            
        elif test_type == 'mann_whitney':
            # Mann-Whitney U test (independent samples)
            statistic, p_value = stats.mannwhitneyu(cv_scores_1, cv_scores_2, alternative='two-sided')
            test_name = 'Mann-Whitney U test'
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Effect size (Cohen's d for t-test)
        if test_type == 'paired_ttest':
            diff = cv_scores_1 - cv_scores_2
            effect_size = np.mean(diff) / np.std(diff, ddof=1)
        else:
            # For non-parametric tests, use rank-biserial correlation as effect size
            effect_size = np.nan
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': effect_size,
            'mean_diff': np.mean(cv_scores_1) - np.mean(cv_scores_2)
        }
        
    except Exception as e:
        print(f"Error in statistical test: {e}")
        return {}


def pairwise_significance_tests(models_cv_results: Dict[str, Dict[str, Any]],
                               metric: str = 'f1',
                               test_type: str = 'paired_ttest') -> pd.DataFrame:
    """
    Perform pairwise statistical significance tests between all models.
    
    Args:
        models_cv_results: Dictionary with model names as keys and CV results as values
        metric: Metric to test
        test_type: Type of statistical test
        
    Returns:
        DataFrame with pairwise test results
        
    Requirements: 2.3, 3.4
    """
    model_names = list(models_cv_results.keys())
    n_models = len(model_names)
    
    # Initialize results matrix
    p_values = np.ones((n_models, n_models))
    effect_sizes = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_1 = model_names[i]
            model_2 = model_names[j]
            
            # Get CV scores
            scores_1 = models_cv_results[model_1]['metrics'][metric]['test_scores']
            scores_2 = models_cv_results[model_2]['metrics'][metric]['test_scores']
            
            # Perform test
            test_result = statistical_significance_test(scores_1, scores_2, test_type)
            
            if test_result:
                p_values[i, j] = test_result['p_value']
                p_values[j, i] = test_result['p_value']  # Symmetric
                effect_sizes[i, j] = test_result['effect_size']
                effect_sizes[j, i] = -test_result['effect_size']  # Opposite sign
    
    # Create DataFrames
    p_value_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
    effect_size_df = pd.DataFrame(effect_sizes, index=model_names, columns=model_names)
    
    return {
        'p_values': p_value_df.round(4),
        'effect_sizes': effect_size_df.round(4),
        'significant_pairs': p_value_df < 0.05
    }


def plot_cv_results(models_cv_results: Dict[str, Dict[str, Any]],
                   metric: str = 'f1',
                   figsize: Tuple[int, int] = (12, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cross-validation results for multiple models.
    
    Args:
        models_cv_results: Dictionary with model names as keys and CV results as values
        metric: Metric to plot
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 2.3, 3.4
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    model_names = list(models_cv_results.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
    
    # Prepare data
    test_scores_data = []
    train_scores_data = []
    overfitting_data = []
    mean_scores = []
    
    for model_name in model_names:
        if metric in models_cv_results[model_name].get('metrics', {}):
            metric_data = models_cv_results[model_name]['metrics'][metric]
            test_scores_data.append(metric_data['test_scores'])
            train_scores_data.append(metric_data['train_scores'])
            overfitting_data.append(metric_data['overfitting'])
            mean_scores.append(metric_data['test_mean'])
    
    # Plot 1: Box plot of test scores
    if test_scores_data:
        box_plot = ax1.boxplot(test_scores_data, labels=model_names, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title(f'CV {metric.upper()} Test Scores Distribution')
        ax1.set_ylabel(f'{metric.upper()} Score')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Train vs Test scores
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        if metric in models_cv_results[model_name].get('metrics', {}):
            metric_data = models_cv_results[model_name]['metrics'][metric]
            ax2.scatter([metric_data['train_mean']], [metric_data['test_mean']], 
                       color=color, s=100, label=model_name, alpha=0.7)
    
    # Add diagonal line
    min_score = min([min(test_scores_data[i]) for i in range(len(test_scores_data))] + 
                   [min(train_scores_data[i]) for i in range(len(train_scores_data))])
    max_score = max([max(test_scores_data[i]) for i in range(len(test_scores_data))] + 
                   [max(train_scores_data[i]) for i in range(len(train_scores_data))])
    ax2.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5)
    
    ax2.set_xlabel(f'Train {metric.upper()} Score')
    ax2.set_ylabel(f'Test {metric.upper()} Score')
    ax2.set_title('Train vs Test Performance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overfitting analysis
    if overfitting_data:
        bars = ax3.bar(range(len(model_names)), overfitting_data, color=colors)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Overfitting (Train - Test)')
        ax3.set_title('Overfitting Analysis')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, overfitting_data):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: CV score variability
    if test_scores_data:
        std_scores = [np.std(scores) for scores in test_scores_data]
        bars = ax4.bar(range(len(model_names)), std_scores, color=colors)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('CV Score Standard Deviation')
        ax4.set_title('CV Score Variability')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, std_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Cross-Validation Analysis - {metric.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cv_learning_curves(model, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           train_sizes: np.ndarray = None,
                           cv: int = 5,
                           scoring: str = 'f1',
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning curves showing performance vs training set size.
    
    Args:
        model: Scikit-learn model object
        X: Feature matrix
        y: Target vector
        train_sizes: Array of training set sizes to use
        cv: Number of cross-validation folds
        scoring: Scoring metric
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 2.3, 3.4
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, 
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot learning curves
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        ax.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(f'{scoring.upper()} Score')
        ax.set_title('Learning Curves')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        print(f"Error plotting learning curves: {e}")
        return plt.figure(figsize=figsize)


def cv_analysis_report(models_cv_results: Dict[str, Dict[str, Any]],
                      metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive cross-validation analysis report.
    
    Args:
        models_cv_results: Dictionary with model names as keys and CV results as values
        metrics: List of metrics to analyze
        
    Returns:
        Dictionary with analysis results as DataFrames
        
    Requirements: 2.3, 3.4
    """
    if metrics is None:
        # Get all available metrics from first model
        first_model = list(models_cv_results.values())[0]
        metrics = list(first_model.get('metrics', {}).keys())
    
    report = {}
    
    for metric in metrics:
        # Comparison table
        comparison_df = compare_cv_results(models_cv_results, metric)
        report[f'{metric}_comparison'] = comparison_df
        
        # Statistical significance tests
        if len(models_cv_results) > 1:
            significance_results = pairwise_significance_tests(models_cv_results, metric)
            report[f'{metric}_significance'] = significance_results
    
    return report