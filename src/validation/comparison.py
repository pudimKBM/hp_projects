"""
Model performance comparison and ranking system.

This module provides comprehensive model comparison capabilities including
side-by-side performance tables, statistical tests, and ranking systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


def create_model_comparison_table(models_results: Dict[str, Dict[str, np.ndarray]],
                                 cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
                                 metrics: List[str] = None) -> pd.DataFrame:
    """
    Create comprehensive side-by-side model performance comparison table.
    
    Args:
        models_results: Dictionary with model names as keys and test results as values
        cv_results: Optional CV results for additional statistics
        metrics: List of metrics to include in comparison
        
    Returns:
        DataFrame with comprehensive model comparison
        
    Requirements: 3.3, 3.5
    """
    from .metrics import calculate_comprehensive_metrics
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    comparison_data = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        # Calculate test metrics
        test_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        # Prepare row data
        row_data = {'Model': model_name}
        
        # Add test metrics
        for metric in metrics:
            if metric in test_metrics:
                row_data[f'Test_{metric.title()}'] = test_metrics[metric]
        
        # Add CV metrics if available
        if cv_results and model_name in cv_results:
            cv_data = cv_results[model_name]
            for metric in metrics:
                if metric in cv_data.get('metrics', {}):
                    cv_metric = cv_data['metrics'][metric]
                    row_data[f'CV_{metric.title()}_Mean'] = cv_metric['test_mean']
                    row_data[f'CV_{metric.title()}_Std'] = cv_metric['test_std']
                    row_data[f'Overfitting_{metric.title()}'] = cv_metric['overfitting']
        
        comparison_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric (F1-score if available, otherwise first metric)
    primary_metric = 'Test_F1_Score' if 'Test_F1_Score' in df.columns else df.columns[1]
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False)
    
    return df.round(4)


def rank_models_by_performance(models_results: Dict[str, Dict[str, np.ndarray]],
                              ranking_metrics: List[str] = None,
                              weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Rank models based on multiple performance metrics with optional weighting.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        ranking_metrics: List of metrics to use for ranking
        weights: Dictionary with metric weights (must sum to 1.0)
        
    Returns:
        DataFrame with model rankings and scores
        
    Requirements: 3.3, 3.5
    """
    from .metrics import calculate_comprehensive_metrics
    
    if ranking_metrics is None:
        ranking_metrics = ['f1_score', 'roc_auc', 'accuracy', 'precision', 'recall']
    
    if weights is None:
        # Equal weights
        weights = {metric: 1.0 / len(ranking_metrics) for metric in ranking_metrics}
    
    # Validate weights
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    
    ranking_data = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        # Calculate weighted score
        weighted_score = 0.0
        metric_scores = {}
        
        for metric in ranking_metrics:
            if metric in metrics and not np.isnan(metrics[metric]):
                score = metrics[metric]
                weight = weights.get(metric, 0.0)
                weighted_score += score * weight
                metric_scores[metric] = score
            else:
                metric_scores[metric] = 0.0
        
        ranking_data.append({
            'Model': model_name,
            'Weighted_Score': weighted_score,
            **metric_scores
        })
    
    # Create DataFrame and sort by weighted score
    df = pd.DataFrame(ranking_data)
    df = df.sort_values('Weighted_Score', ascending=False)
    
    # Add rank column
    df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Model', 'Weighted_Score'] + ranking_metrics
    df = df[cols]
    
    return df.round(4)


def perform_statistical_tests(models_results: Dict[str, Dict[str, np.ndarray]],
                             cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
                             metric: str = 'f1_score',
                             alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform statistical significance tests between model performances.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        cv_results: Optional CV results for paired tests
        metric: Metric to test
        alpha: Significance level
        
    Returns:
        Dictionary with statistical test results
        
    Requirements: 3.3, 3.5
    """
    from .cross_validation import statistical_significance_test, pairwise_significance_tests
    
    results = {
        'metric': metric,
        'alpha': alpha,
        'tests': {}
    }
    
    model_names = list(models_results.keys())
    
    if cv_results and len(model_names) > 1:
        # Use CV results for paired tests
        try:
            pairwise_results = pairwise_significance_tests(cv_results, metric)
            results['pairwise_tests'] = pairwise_results
            
            # Summary of significant differences
            p_values = pairwise_results['p_values']
            significant_pairs = []
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_1 = model_names[i]
                    model_2 = model_names[j]
                    p_val = p_values.loc[model_1, model_2]
                    
                    if p_val < alpha:
                        # Determine which model is better
                        mean_1 = cv_results[model_1]['metrics'][metric]['test_mean']
                        mean_2 = cv_results[model_2]['metrics'][metric]['test_mean']
                        better_model = model_1 if mean_1 > mean_2 else model_2
                        worse_model = model_2 if mean_1 > mean_2 else model_1
                        
                        significant_pairs.append({
                            'better_model': better_model,
                            'worse_model': worse_model,
                            'p_value': p_val,
                            'mean_difference': abs(mean_1 - mean_2)
                        })
            
            results['significant_differences'] = significant_pairs
            
        except Exception as e:
            print(f"Error in pairwise statistical tests: {e}")
    
    # Overall ANOVA test if we have CV results
    if cv_results and len(model_names) > 2:
        try:
            # Collect all CV scores
            all_scores = []
            group_labels = []
            
            for model_name in model_names:
                if metric in cv_results[model_name].get('metrics', {}):
                    scores = cv_results[model_name]['metrics'][metric]['test_scores']
                    all_scores.extend(scores)
                    group_labels.extend([model_name] * len(scores))
            
            if len(set(group_labels)) > 2:
                # Perform one-way ANOVA
                groups = [cv_results[name]['metrics'][metric]['test_scores'] 
                         for name in model_names 
                         if metric in cv_results[name].get('metrics', {})]
                
                f_stat, p_val = stats.f_oneway(*groups)
                
                results['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'significant': p_val < alpha,
                    'interpretation': 'At least one model significantly different' if p_val < alpha else 'No significant differences'
                }
                
        except Exception as e:
            print(f"Error in ANOVA test: {e}")
    
    return results


def create_performance_radar_chart(models_results: Dict[str, Dict[str, np.ndarray]],
                                  metrics: List[str] = None,
                                  figsize: Tuple[int, int] = (10, 10),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create radar chart comparing model performance across multiple metrics.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        metrics: List of metrics to include in radar chart
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.3, 3.5
    """
    from .metrics import calculate_comprehensive_metrics
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Calculate metrics for all models
    model_metrics = {}
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        calculated_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        model_metrics[model_name] = [calculated_metrics.get(metric, 0.0) for metric in metrics]
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_metrics)))
    
    for (model_name, values), color in zip(model_metrics.items(), colors):
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Model Performance Comparison', size=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_performance_heatmap(models_results: Dict[str, Dict[str, np.ndarray]],
                              metrics: List[str] = None,
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing model performance across different metrics.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        metrics: List of metrics to include
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
        
    Requirements: 3.3, 3.5
    """
    from .metrics import calculate_comprehensive_metrics
    
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Calculate metrics for all models
    heatmap_data = []
    model_names = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        calculated_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        metric_values = [calculated_metrics.get(metric, 0.0) for metric in metrics]
        
        heatmap_data.append(metric_values)
        model_names.append(model_name)
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(heatmap_data, 
                     index=model_names, 
                     columns=[metric.replace('_', ' ').title() for metric in metrics])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0.5, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                fmt='.3f', ax=ax)
    
    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Models', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_model_recommendation(models_results: Dict[str, Dict[str, np.ndarray]],
                                 cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
                                 business_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate model recommendation based on performance and business requirements.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        cv_results: Optional CV results for stability analysis
        business_requirements: Dictionary with business constraints and preferences
        
    Returns:
        Dictionary with recommendation and reasoning
        
    Requirements: 3.3, 3.5
    """
    from .metrics import calculate_comprehensive_metrics
    
    # Default business requirements
    if business_requirements is None:
        business_requirements = {
            'primary_metric': 'f1_score',
            'min_accuracy': 0.8,
            'interpretability_important': False,
            'speed_important': False,
            'stability_important': True
        }
    
    recommendations = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results.get('y_pred_proba', None)
        
        # Calculate performance metrics
        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        # Initialize recommendation score
        score = 0.0
        reasons = []
        
        # Primary metric performance
        primary_metric = business_requirements.get('primary_metric', 'f1_score')
        if primary_metric in metrics:
            primary_score = metrics[primary_metric]
            score += primary_score * 0.4  # 40% weight
            reasons.append(f"{primary_metric}: {primary_score:.3f}")
        
        # Minimum accuracy requirement
        min_accuracy = business_requirements.get('min_accuracy', 0.8)
        if metrics.get('accuracy', 0) >= min_accuracy:
            score += 0.2  # 20% bonus for meeting requirement
            reasons.append(f"Meets accuracy requirement (≥{min_accuracy})")
        else:
            score -= 0.3  # Penalty for not meeting requirement
            reasons.append(f"Below accuracy requirement ({metrics.get('accuracy', 0):.3f} < {min_accuracy})")
        
        # Stability (CV results)
        if cv_results and model_name in cv_results and business_requirements.get('stability_important', True):
            cv_data = cv_results[model_name]
            if primary_metric in cv_data.get('metrics', {}):
                cv_std = cv_data['metrics'][primary_metric]['test_std']
                if cv_std < 0.05:  # Low variability
                    score += 0.15
                    reasons.append(f"High stability (CV std: {cv_std:.3f})")
                elif cv_std > 0.1:  # High variability
                    score -= 0.1
                    reasons.append(f"Lower stability (CV std: {cv_std:.3f})")
        
        # Interpretability bonus (heuristic based on model name)
        if business_requirements.get('interpretability_important', False):
            interpretable_models = ['logistic', 'linear', 'tree', 'forest']
            if any(keyword in model_name.lower() for keyword in interpretable_models):
                score += 0.1
                reasons.append("Interpretable model type")
        
        # Speed bonus (heuristic based on model name)
        if business_requirements.get('speed_important', False):
            fast_models = ['logistic', 'linear', 'naive_bayes']
            if any(keyword in model_name.lower() for keyword in fast_models):
                score += 0.1
                reasons.append("Fast model type")
        
        recommendations.append({
            'model': model_name,
            'score': score,
            'reasons': reasons,
            'metrics': metrics
        })
    
    # Sort by recommendation score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # Generate final recommendation
    best_model = recommendations[0]
    
    recommendation = {
        'recommended_model': best_model['model'],
        'recommendation_score': best_model['score'],
        'reasons': best_model['reasons'],
        'all_rankings': recommendations,
        'summary': f"Based on the analysis, {best_model['model']} is recommended with a score of {best_model['score']:.3f}."
    }
    
    # Add warnings if needed
    warnings = []
    if best_model['score'] < 0.7:
        warnings.append("Low overall recommendation score - consider model improvement")
    
    if best_model['metrics'].get('accuracy', 0) < business_requirements.get('min_accuracy', 0.8):
        warnings.append("Recommended model does not meet minimum accuracy requirement")
    
    if warnings:
        recommendation['warnings'] = warnings
    
    return recommendation


def create_comprehensive_comparison_report(models_results: Dict[str, Dict[str, np.ndarray]],
                                         cv_results: Optional[Dict[str, Dict[str, Any]]] = None,
                                         save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create comprehensive model comparison report with all analyses.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        cv_results: Optional CV results
        save_dir: Directory to save plots (optional)
        
    Returns:
        Dictionary with all comparison results
        
    Requirements: 3.3, 3.5
    """
    report = {}
    
    # Performance comparison table
    report['comparison_table'] = create_model_comparison_table(models_results, cv_results)
    
    # Model ranking
    report['model_ranking'] = rank_models_by_performance(models_results)
    
    # Statistical tests
    if cv_results:
        report['statistical_tests'] = perform_statistical_tests(models_results, cv_results)
    
    # Model recommendation
    report['recommendation'] = generate_model_recommendation(models_results, cv_results)
    
    # Generate visualizations
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Radar chart
        radar_fig = create_performance_radar_chart(models_results)
        radar_fig.savefig(os.path.join(save_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.close(radar_fig)
        
        # Heatmap
        heatmap_fig = create_performance_heatmap(models_results)
        heatmap_fig.savefig(os.path.join(save_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close(heatmap_fig)
        
        report['visualizations_saved'] = True
    
    return report