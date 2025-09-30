"""
Visualization functions for model interpretation.

This module provides functions to create various visualizations for
feature importance, prediction explanations, and model interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create horizontal bar plot for feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Get top N features
        plot_data = importance_df.head(top_n).copy()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(
            range(len(plot_data)),
            plot_data['importance'],
            color=plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
        )
        
        # Customize plot
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, plot_data['importance'])):
            ax.text(
                bar.get_width() + max(plot_data['importance']) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{importance:.4f}',
                va='center',
                fontsize=9
            )
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        
        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        logger.info(f"Created feature importance plot for {len(plot_data)} features")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        raise


def plot_importance_heatmap(
    comparison_df: pd.DataFrame,
    title: str = "Feature Importance Comparison",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create heatmap comparing feature importance across models.
    
    Args:
        comparison_df: DataFrame with features and importance scores for multiple models
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Extract importance columns
        importance_cols = [col for col in comparison_df.columns if col.endswith('_importance')]
        
        if not importance_cols:
            raise ValueError("No importance columns found in comparison_df")
        
        # Prepare data for heatmap
        heatmap_data = comparison_df[['feature'] + importance_cols].set_index('feature')
        
        # Rename columns to remove '_importance' suffix
        heatmap_data.columns = [col.replace('_importance', '') for col in heatmap_data.columns]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data.T,  # Transpose to have models on y-axis
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance Score'},
            ax=ax
        )
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Models')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved importance heatmap to {save_path}")
        
        logger.info(f"Created importance heatmap for {len(heatmap_data.columns)} models")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating importance heatmap: {str(e)}")
        raise


def create_interactive_importance_plot(
    importance_df: pd.DataFrame,
    title: str = "Interactive Feature Importance",
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Create interactive plot data for feature importance (for Plotly or similar).
    
    Args:
        importance_df: DataFrame with feature importance
        title: Plot title
        top_n: Number of top features to include
        
    Returns:
        Dictionary with plot data and configuration
    """
    try:
        # Get top N features
        plot_data = importance_df.head(top_n).copy()
        
        # Prepare interactive plot data
        plot_config = {
            'data': {
                'x': plot_data['importance'].tolist(),
                'y': plot_data['feature'].tolist(),
                'type': 'bar',
                'orientation': 'h',
                'marker': {
                    'color': plot_data['importance'].tolist(),
                    'colorscale': 'Viridis'
                },
                'text': [f'{imp:.4f}' for imp in plot_data['importance']],
                'textposition': 'outside'
            },
            'layout': {
                'title': title,
                'xaxis': {'title': 'Importance Score'},
                'yaxis': {'title': 'Features', 'autorange': 'reversed'},
                'height': max(400, len(plot_data) * 25),
                'margin': {'l': 200, 'r': 50, 't': 50, 'b': 50}
            }
        }
        
        logger.info(f"Created interactive plot config for {len(plot_data)} features")
        return plot_config
        
    except Exception as e:
        logger.error(f"Error creating interactive importance plot: {str(e)}")
        raise


def plot_prediction_explanation(
    explanation: Dict[str, Any],
    title: str = "Prediction Explanation",
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize explanation for a single prediction.
    
    Args:
        explanation: Prediction explanation dictionary
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    try:
        top_features = explanation.get('top_contributing_features', [])[:top_n]
        
        if not top_features:
            raise ValueError("No top contributing features found in explanation")
        
        # Extract data for plotting
        features = [f['feature'] for f in top_features]
        contributions = [f['contribution'] for f in top_features]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Feature contributions
        colors = ['green' if c > 0 else 'red' for c in contributions]
        bars = ax1.barh(range(len(features)), contributions, color=colors, alpha=0.7)
        
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Contribution')
        ax1.set_title('Feature Contributions')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.invert_yaxis()
        
        # Add value labels
        for bar, contrib in zip(bars, contributions):
            ax1.text(
                contrib + (max(contributions) - min(contributions)) * 0.02 * (1 if contrib > 0 else -1),
                bar.get_y() + bar.get_height() / 2,
                f'{contrib:.4f}',
                va='center',
                fontsize=9
            )
        
        # Plot 2: Prediction probabilities (if available)
        probabilities = explanation.get('probabilities')
        if probabilities:
            classes = list(probabilities.keys())
            probs = list(probabilities.values())
            
            bars2 = ax2.bar(classes, probs, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
            ax2.set_ylabel('Probability')
            ax2.set_title('Class Probabilities')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bar, prob in zip(bars2, probs):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{prob:.3f}',
                    ha='center',
                    fontsize=10
                )
            
            # Highlight predicted class
            predicted_class = explanation.get('predicted_class')
            if predicted_class in classes:
                pred_idx = classes.index(predicted_class)
                bars2[pred_idx].set_edgecolor('black')
                bars2[pred_idx].set_linewidth(2)
        else:
            ax2.text(0.5, 0.5, 'No probability\ninformation available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Class Probabilities')
        
        # Add overall title
        confidence = explanation.get('confidence')
        if confidence is not None:
            fig.suptitle(f"{title}\nPredicted: {explanation.get('predicted_class', 'Unknown')} "
                        f"(Confidence: {confidence:.3f})", fontsize=12, fontweight='bold')
        else:
            fig.suptitle(f"{title}\nPredicted: {explanation.get('predicted_class', 'Unknown')}", 
                        fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction explanation plot to {save_path}")
        
        logger.info("Created prediction explanation plot")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating prediction explanation plot: {str(e)}")
        raise


def plot_feature_importance_comparison(
    importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create side-by-side comparison of feature importance across models.
    
    Args:
        importance_dict: Dictionary with model names and importance DataFrames
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    try:
        n_models = len(importance_dict)
        if n_models == 0:
            raise ValueError("importance_dict cannot be empty")
        
        # Create subplots
        fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
        if n_models == 1:
            axes = [axes]
        
        # Plot each model's importance
        for i, (model_name, importance_df) in enumerate(importance_dict.items()):
            ax = axes[i]
            plot_data = importance_df.head(top_n)
            
            bars = ax.barh(
                range(len(plot_data)),
                plot_data['importance'],
                color=plt.cm.Set2(i / n_models)
            )
            
            ax.set_yticks(range(len(plot_data)))
            if i == 0:  # Only set y-labels for first subplot
                ax.set_yticklabels(plot_data['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(model_name, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, importance in zip(bars, plot_data['importance']):
                ax.text(
                    bar.get_width() + max(plot_data['importance']) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{importance:.3f}',
                    va='center',
                    fontsize=8
                )
        
        # Add overall title
        fig.suptitle('Feature Importance Comparison Across Models', 
                    fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance comparison to {save_path}")
        
        logger.info(f"Created feature importance comparison for {n_models} models")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance comparison: {str(e)}")
        raise


def plot_confidence_distribution(
    explanations: List[Dict[str, Any]],
    title: str = "Prediction Confidence Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of prediction confidence scores.
    
    Args:
        explanations: List of prediction explanations
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Extract confidence scores
        confidences = [exp.get('confidence') for exp in explanations 
                      if exp.get('confidence') is not None]
        
        if not confidences:
            raise ValueError("No confidence scores found in explanations")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(confidences, vert=True)
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Box Plot')
        ax2.set_xticklabels(['All Predictions'])
        
        # Add statistics
        stats_text = f"""Statistics:
Mean: {np.mean(confidences):.3f}
Median: {np.median(confidences):.3f}
Std: {np.std(confidences):.3f}
Min: {np.min(confidences):.3f}
Max: {np.max(confidences):.3f}"""
        
        ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confidence distribution plot to {save_path}")
        
        logger.info(f"Created confidence distribution plot for {len(confidences)} predictions")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating confidence distribution plot: {str(e)}")
        raise