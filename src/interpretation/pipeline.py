"""
Comprehensive interpretation pipeline for model explainability.

This module provides a unified interface for all interpretation methods,
business-friendly feature mapping, and comprehensive insights generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from datetime import datetime

from .feature_importance import (
    extract_tree_importance,
    calculate_permutation_importance,
    rank_feature_importance,
    compare_feature_importance,
    get_feature_importance_summary
)
from .prediction_explanation import (
    explain_prediction,
    explain_prediction_batch,
    analyze_prediction_patterns
)
from .visualization import (
    plot_feature_importance,
    plot_importance_heatmap,
    plot_prediction_explanation,
    plot_feature_importance_comparison,
    plot_confidence_distribution
)

logger = logging.getLogger(__name__)


class InterpretationPipeline:
    """
    Comprehensive pipeline for model interpretation and explainability.
    
    This class provides a unified interface for feature importance analysis,
    prediction explanation, and visualization generation.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        business_feature_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the interpretation pipeline.
        
        Args:
            feature_names: List of feature names
            class_names: List of class names (optional)
            business_feature_mapping: Mapping from technical to business-friendly names
        """
        self.feature_names = feature_names
        self.class_names = class_names or ['Class_0', 'Class_1']
        self.business_feature_mapping = business_feature_mapping or {}
        
        # Storage for results
        self.feature_importance_results = {}
        self.prediction_explanations = []
        self.visualizations = {}
        self.insights = {}
        
        logger.info(f"Initialized InterpretationPipeline with {len(feature_names)} features")
    
    def analyze_model_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        methods: List[str] = ['tree', 'permutation']
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance for a single model using multiple methods.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            methods: List of importance calculation methods
            
        Returns:
            Dictionary with importance results for each method
        """
        try:
            importance_results = {}
            
            # Tree-based importance
            if 'tree' in methods:
                try:
                    tree_importance = extract_tree_importance(model, self.feature_names)
                    importance_results['tree'] = rank_feature_importance(tree_importance)
                    logger.info(f"Calculated tree-based importance for {model_name}")
                except Exception as e:
                    logger.warning(f"Could not calculate tree importance for {model_name}: {str(e)}")
            
            # Permutation importance
            if 'permutation' in methods:
                try:
                    perm_importance = calculate_permutation_importance(
                        model, X, y, self.feature_names
                    )
                    importance_results['permutation'] = rank_feature_importance(perm_importance)
                    logger.info(f"Calculated permutation importance for {model_name}")
                except Exception as e:
                    logger.warning(f"Could not calculate permutation importance for {model_name}: {str(e)}")
            
            # Store results
            self.feature_importance_results[model_name] = importance_results
            
            return importance_results
            
        except Exception as e:
            logger.error(f"Error analyzing model importance for {model_name}: {str(e)}")
            raise
    
    def analyze_multiple_models(
        self,
        models_dict: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        methods: List[str] = ['tree', 'permutation']
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Analyze feature importance for multiple models.
        
        Args:
            models_dict: Dictionary with model names and trained models
            X: Feature matrix
            y: Target vector
            methods: List of importance calculation methods
            
        Returns:
            Dictionary with importance results for all models
        """
        try:
            all_results = {}
            
            for model_name, model in models_dict.items():
                results = self.analyze_model_importance(model, X, y, model_name, methods)
                all_results[model_name] = results
            
            logger.info(f"Analyzed importance for {len(models_dict)} models")
            return all_results
            
        except Exception as e:
            logger.error(f"Error analyzing multiple models: {str(e)}")
            raise
    
    def explain_predictions(
        self,
        model: Any,
        X_samples: np.ndarray,
        model_name: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            model: Trained model
            X_samples: Samples to explain
            model_name: Name of the model
            top_n: Number of top features per explanation
            
        Returns:
            List of prediction explanations
        """
        try:
            explanations = explain_prediction_batch(
                model, X_samples, self.feature_names, self.class_names, top_n
            )
            
            # Add model name to each explanation
            for exp in explanations:
                exp['model_name'] = model_name
            
            # Store explanations
            self.prediction_explanations.extend(explanations)
            
            logger.info(f"Generated {len(explanations)} explanations for {model_name}")
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining predictions for {model_name}: {str(e)}")
            raise
    
    def create_visualizations(
        self,
        output_dir: str = "interpretation_plots",
        top_n: int = 20
    ) -> Dict[str, str]:
        """
        Create comprehensive visualizations for interpretation results.
        
        Args:
            output_dir: Directory to save plots
            top_n: Number of top features to display
            
        Returns:
            Dictionary with plot names and file paths
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            plot_paths = {}
            
            # Feature importance plots for each model
            for model_name, importance_results in self.feature_importance_results.items():
                for method, importance_df in importance_results.items():
                    plot_name = f"{model_name}_{method}_importance"
                    plot_path = os.path.join(output_dir, f"{plot_name}.png")
                    
                    fig = plot_feature_importance(
                        importance_df,
                        title=f"{model_name} - {method.title()} Importance",
                        top_n=top_n,
                        save_path=plot_path
                    )
                    plt.close(fig)
                    plot_paths[plot_name] = plot_path
            
            # Comparison plots if multiple models
            if len(self.feature_importance_results) > 1:
                # Compare permutation importance across models
                perm_importance_dict = {}
                for model_name, results in self.feature_importance_results.items():
                    if 'permutation' in results:
                        perm_importance_dict[model_name] = results['permutation']
                
                if len(perm_importance_dict) > 1:
                    # Comparison plot
                    comparison_plot_path = os.path.join(output_dir, "importance_comparison.png")
                    fig = plot_feature_importance_comparison(
                        perm_importance_dict,
                        top_n=top_n,
                        save_path=comparison_plot_path
                    )
                    plt.close(fig)
                    plot_paths['importance_comparison'] = comparison_plot_path
                    
                    # Heatmap
                    comparison_df = compare_feature_importance(perm_importance_dict, top_n)
                    heatmap_path = os.path.join(output_dir, "importance_heatmap.png")
                    fig = plot_importance_heatmap(
                        comparison_df,
                        save_path=heatmap_path
                    )
                    plt.close(fig)
                    plot_paths['importance_heatmap'] = heatmap_path
            
            # Prediction explanation plots
            if self.prediction_explanations:
                # Confidence distribution
                confidence_plot_path = os.path.join(output_dir, "confidence_distribution.png")
                fig = plot_confidence_distribution(
                    self.prediction_explanations,
                    save_path=confidence_plot_path
                )
                plt.close(fig)
                plot_paths['confidence_distribution'] = confidence_plot_path
                
                # Individual prediction explanations (first few)
                for i, explanation in enumerate(self.prediction_explanations[:5]):
                    exp_plot_path = os.path.join(output_dir, f"prediction_explanation_{i+1}.png")
                    fig = plot_prediction_explanation(
                        explanation,
                        title=f"Prediction Explanation {i+1}",
                        save_path=exp_plot_path
                    )
                    plt.close(fig)
                    plot_paths[f'prediction_explanation_{i+1}'] = exp_plot_path
            
            self.visualizations = plot_paths
            logger.info(f"Created {len(plot_paths)} visualization plots in {output_dir}")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights from interpretation results.
        
        Returns:
            Dictionary with interpretation insights
        """
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'feature_insights': {},
                'model_insights': {},
                'prediction_insights': {},
                'recommendations': []
            }
            
            # Feature-level insights
            if self.feature_importance_results:
                insights['feature_insights'] = self._analyze_feature_patterns()
            
            # Model-level insights
            if len(self.feature_importance_results) > 1:
                insights['model_insights'] = self._analyze_model_differences()
            
            # Prediction-level insights
            if self.prediction_explanations:
                insights['prediction_insights'] = self._analyze_prediction_patterns()
            
            # Generate recommendations
            insights['recommendations'] = self._generate_recommendations()
            
            self.insights = insights
            logger.info("Generated comprehensive interpretation insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise
    
    def create_interpretation_summary(
        self,
        output_path: str = "interpretation_summary.md"
    ) -> str:
        """
        Create a comprehensive interpretation summary report.
        
        Args:
            output_path: Path to save the summary report
            
        Returns:
            Path to the saved report
        """
        try:
            # Generate insights if not already done
            if not self.insights:
                self.generate_insights()
            
            # Create markdown report
            report_content = self._create_markdown_report()
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Created interpretation summary report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating interpretation summary: {str(e)}")
            raise
    
    def _analyze_feature_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in feature importance across methods and models."""
        try:
            all_features = set()
            feature_scores = {}
            
            # Collect all feature importance scores
            for model_name, results in self.feature_importance_results.items():
                for method, importance_df in results.items():
                    for _, row in importance_df.iterrows():
                        feature = row['feature']
                        importance = row['importance']
                        
                        all_features.add(feature)
                        
                        if feature not in feature_scores:
                            feature_scores[feature] = []
                        feature_scores[feature].append(importance)
            
            # Calculate statistics for each feature
            feature_stats = {}
            for feature, scores in feature_scores.items():
                feature_stats[feature] = {
                    'mean_importance': np.mean(scores),
                    'std_importance': np.std(scores),
                    'consistency': 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0,
                    'appearances': len(scores)
                }
            
            # Find most consistent and important features
            consistent_features = sorted(
                feature_stats.items(),
                key=lambda x: x[1]['consistency'],
                reverse=True
            )[:10]
            
            important_features = sorted(
                feature_stats.items(),
                key=lambda x: x[1]['mean_importance'],
                reverse=True
            )[:10]
            
            return {
                'total_unique_features': len(all_features),
                'most_consistent_features': [f[0] for f in consistent_features],
                'most_important_features': [f[0] for f in important_features],
                'feature_statistics': feature_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feature patterns: {str(e)}")
            return {}
    
    def _analyze_model_differences(self) -> Dict[str, Any]:
        """Analyze differences in feature importance between models."""
        try:
            model_similarities = {}
            
            # Compare models pairwise
            model_names = list(self.feature_importance_results.keys())
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    # Compare permutation importance if available
                    if ('permutation' in self.feature_importance_results[model1] and
                        'permutation' in self.feature_importance_results[model2]):
                        
                        df1 = self.feature_importance_results[model1]['permutation']
                        df2 = self.feature_importance_results[model2]['permutation']
                        
                        # Calculate correlation of importance scores
                        merged = pd.merge(df1[['feature', 'importance']], 
                                        df2[['feature', 'importance']], 
                                        on='feature', suffixes=('_1', '_2'))
                        
                        if len(merged) > 0:
                            correlation = merged['importance_1'].corr(merged['importance_2'])
                            model_similarities[f"{model1}_vs_{model2}"] = correlation
            
            return {
                'model_similarities': model_similarities,
                'most_similar_models': max(model_similarities.items(), key=lambda x: x[1]) if model_similarities else None,
                'most_different_models': min(model_similarities.items(), key=lambda x: x[1]) if model_similarities else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing model differences: {str(e)}")
            return {}
    
    def _analyze_prediction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in prediction explanations."""
        try:
            return analyze_prediction_patterns(self.prediction_explanations)
            
        except Exception as e:
            logger.error(f"Error analyzing prediction patterns: {str(e)}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on interpretation results."""
        recommendations = []
        
        try:
            # Feature-based recommendations
            if 'feature_insights' in self.insights:
                feature_insights = self.insights['feature_insights']
                
                if 'most_important_features' in feature_insights:
                    top_features = feature_insights['most_important_features'][:5]
                    recommendations.append(
                        f"Focus on the top 5 most important features: {', '.join(top_features)}. "
                        "These features consistently drive model predictions."
                    )
                
                if 'most_consistent_features' in feature_insights:
                    consistent_features = feature_insights['most_consistent_features'][:3]
                    recommendations.append(
                        f"The most consistent features across models are: {', '.join(consistent_features)}. "
                        "These features provide reliable predictive power."
                    )
            
            # Model-based recommendations
            if 'model_insights' in self.insights:
                model_insights = self.insights['model_insights']
                
                if 'most_similar_models' in model_insights and model_insights['most_similar_models']:
                    similar_models = model_insights['most_similar_models'][0]
                    recommendations.append(
                        f"Models {similar_models} show similar feature importance patterns. "
                        "Consider ensemble methods or choose the simpler model."
                    )
            
            # Prediction-based recommendations
            if 'prediction_insights' in self.insights:
                pred_insights = self.insights['prediction_insights']
                
                if 'avg_confidence' in pred_insights:
                    avg_conf = pred_insights['avg_confidence']
                    if avg_conf < 0.7:
                        recommendations.append(
                            f"Average prediction confidence is {avg_conf:.3f}, which is relatively low. "
                            "Consider collecting more training data or feature engineering."
                        )
                    elif avg_conf > 0.9:
                        recommendations.append(
                            f"High average confidence ({avg_conf:.3f}) suggests good model performance. "
                            "Monitor for potential overfitting on new data."
                        )
            
            # General recommendations
            recommendations.append(
                "Regularly monitor feature importance to detect data drift and model degradation."
            )
            
            recommendations.append(
                "Use prediction explanations to build trust with stakeholders and identify edge cases."
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations. Please review interpretation results manually."]
    
    def _create_markdown_report(self) -> str:
        """Create a comprehensive markdown report."""
        try:
            report = f"""# Model Interpretation Summary Report

Generated on: {self.insights.get('timestamp', 'Unknown')}

## Executive Summary

This report provides a comprehensive analysis of model interpretability, including feature importance analysis, prediction explanations, and actionable insights.

## Feature Analysis

### Most Important Features
"""
            
            # Add feature insights
            if 'feature_insights' in self.insights:
                feature_insights = self.insights['feature_insights']
                
                if 'most_important_features' in feature_insights:
                    for i, feature in enumerate(feature_insights['most_important_features'][:10], 1):
                        business_name = self.business_feature_mapping.get(feature, feature)
                        report += f"{i}. **{business_name}** ({feature})\n"
                
                report += f"\n### Feature Statistics\n"
                report += f"- Total unique features analyzed: {feature_insights.get('total_unique_features', 'N/A')}\n"
                report += f"- Most consistent features: {', '.join(feature_insights.get('most_consistent_features', [])[:5])}\n"
            
            # Add model insights
            if 'model_insights' in self.insights:
                model_insights = self.insights['model_insights']
                report += f"\n## Model Comparison\n"
                
                if 'model_similarities' in model_insights:
                    report += f"### Model Similarity Analysis\n"
                    for comparison, similarity in model_insights['model_similarities'].items():
                        report += f"- {comparison}: {similarity:.3f}\n"
            
            # Add prediction insights
            if 'prediction_insights' in self.insights:
                pred_insights = self.insights['prediction_insights']
                report += f"\n## Prediction Analysis\n"
                report += f"- Total predictions analyzed: {pred_insights.get('total_predictions', 'N/A')}\n"
                report += f"- Average confidence: {pred_insights.get('avg_confidence', 'N/A'):.3f}\n"
                report += f"- Confidence standard deviation: {pred_insights.get('confidence_std', 'N/A'):.3f}\n"
                
                if 'most_important_features' in pred_insights:
                    report += f"\n### Most Frequently Important Features in Predictions\n"
                    for feature, count in list(pred_insights['most_important_features'].items())[:10]:
                        business_name = self.business_feature_mapping.get(feature, feature)
                        report += f"- **{business_name}**: {count} times\n"
            
            # Add recommendations
            if 'recommendations' in self.insights:
                report += f"\n## Recommendations\n"
                for i, rec in enumerate(self.insights['recommendations'], 1):
                    report += f"{i}. {rec}\n"
            
            # Add visualizations section
            if self.visualizations:
                report += f"\n## Generated Visualizations\n"
                for plot_name, plot_path in self.visualizations.items():
                    report += f"- **{plot_name.replace('_', ' ').title()}**: `{plot_path}`\n"
            
            report += f"\n## Technical Details\n"
            report += f"- Feature names: {len(self.feature_names)} features\n"
            report += f"- Class names: {', '.join(self.class_names)}\n"
            report += f"- Models analyzed: {', '.join(self.feature_importance_results.keys())}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating markdown report: {str(e)}")
            return f"Error creating report: {str(e)}"


def map_feature_names(
    feature_names: List[str],
    mapping_dict: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Create business-friendly feature name mapping.
    
    Args:
        feature_names: List of technical feature names
        mapping_dict: Optional predefined mapping dictionary
        
    Returns:
        Dictionary mapping technical names to business-friendly names
    """
    try:
        if mapping_dict:
            return mapping_dict
        
        # Create default business-friendly names
        business_mapping = {}
        
        for feature in feature_names:
            # Convert technical names to readable format
            business_name = feature.replace('_', ' ').title()
            
            # Handle common patterns
            if 'tfidf' in feature.lower():
                business_name = f"Text Content: {business_name.replace('Tfidf', '').strip()}"
            elif 'scaled' in feature.lower():
                business_name = business_name.replace('Scaled', '(Normalized)')
            elif 'encoded' in feature.lower():
                business_name = business_name.replace('Encoded', '(Category)')
            
            business_mapping[feature] = business_name
        
        logger.info(f"Created business mapping for {len(feature_names)} features")
        return business_mapping
        
    except Exception as e:
        logger.error(f"Error mapping feature names: {str(e)}")
        return {name: name for name in feature_names}  # Fallback to original names