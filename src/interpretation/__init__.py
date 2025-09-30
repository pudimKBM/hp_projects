"""
Model interpretation and explainability module.

This module provides functionality for:
- Feature importance analysis
- Prediction explanation
- Model interpretability visualizations
- Comprehensive interpretation pipelines
"""

from .feature_importance import (
    extract_tree_importance,
    calculate_permutation_importance,
    rank_feature_importance,
    compare_feature_importance
)

from .prediction_explanation import (
    explain_prediction,
    calculate_prediction_confidence,
    get_top_contributing_features,
    explain_prediction_batch
)

from .visualization import (
    plot_feature_importance,
    plot_importance_heatmap,
    create_interactive_importance_plot,
    plot_prediction_explanation
)

from .pipeline import (
    InterpretationPipeline,
    map_feature_names
)

__all__ = [
    'extract_tree_importance',
    'calculate_permutation_importance', 
    'rank_feature_importance',
    'compare_feature_importance',
    'explain_prediction',
    'calculate_prediction_confidence',
    'get_top_contributing_features',
    'explain_prediction_batch',
    'plot_feature_importance',
    'plot_importance_heatmap',
    'create_interactive_importance_plot',
    'plot_prediction_explanation',
    'InterpretationPipeline',
    'map_feature_names'
]