"""
Model validation and evaluation module.

This module provides comprehensive model validation capabilities including:
- Metrics calculation (precision, recall, F1-score, AUC-ROC, accuracy)
- Confusion matrix visualization
- ROC curve analysis
- Cross-validation analysis
- Model performance comparison
"""

from .metrics import (
    calculate_comprehensive_metrics,
    calculate_metrics_for_multiple_models,
    get_classification_report_dict
)

from .visualization import (
    plot_confusion_matrix,
    plot_multiple_confusion_matrices,
    plot_roc_curves,
    plot_cv_scores_distribution
)

from .comparison import (
    create_model_comparison_table,
    perform_statistical_tests,
    rank_models_by_performance,
    create_performance_radar_chart,
    create_performance_heatmap,
    generate_model_recommendation,
    create_comprehensive_comparison_report
)

from .cross_validation import (
    detailed_cv_analysis,
    cv_score_statistics,
    plot_cv_results,
    compare_cv_results,
    statistical_significance_test,
    pairwise_significance_tests,
    plot_cv_learning_curves,
    cv_analysis_report
)

from .roc_analysis import (
    calculate_roc_metrics,
    compare_roc_curves,
    plot_detailed_roc_analysis,
    find_optimal_threshold,
    bootstrap_auc_confidence_interval,
    roc_analysis_summary
)

__all__ = [
    # Metrics
    'calculate_comprehensive_metrics',
    'calculate_metrics_for_multiple_models', 
    'get_classification_report_dict',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_multiple_confusion_matrices',
    'plot_roc_curves',
    'plot_cv_scores_distribution',
    'plot_precision_recall_curves',
    'plot_feature_importance_comparison',
    
    # Comparison
    'create_model_comparison_table',
    'perform_statistical_tests',
    'rank_models_by_performance',
    'create_performance_radar_chart',
    'create_performance_heatmap',
    'generate_model_recommendation',
    'create_comprehensive_comparison_report',
    
    # Cross-validation
    'detailed_cv_analysis',
    'cv_score_statistics',
    'plot_cv_results',
    'compare_cv_results',
    'statistical_significance_test',
    'pairwise_significance_tests',
    'plot_cv_learning_curves',
    'cv_analysis_report',
    
    # ROC Analysis
    'calculate_roc_metrics',
    'compare_roc_curves',
    'plot_detailed_roc_analysis',
    'find_optimal_threshold',
    'bootstrap_auc_confidence_interval',
    'roc_analysis_summary'
]