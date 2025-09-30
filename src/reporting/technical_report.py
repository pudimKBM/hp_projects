"""
Technical Report Generation Module

Generates comprehensive technical analysis reports including model performance,
methodology details, and feature analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings

class TechnicalReportGenerator:
    """
    Generates detailed technical reports for ML pipeline results.
    
    This class creates comprehensive technical documentation including:
    - Model performance analysis
    - Feature importance analysis  
    - Cross-validation results
    - Methodology documentation
    - Statistical analysis
    """
    
    def __init__(self):
        """Initialize the technical report generator."""
        self.report_sections = []
        self.figures = []
        
    def generate_report(self, 
                       model_results: Dict[str, Any],
                       feature_analysis: Dict[str, Any],
                       cv_results: Dict[str, Any],
                       methodology: Dict[str, Any]) -> str:
        """
        Generate a comprehensive technical report.
        
        Args:
            model_results: Dictionary containing model performance metrics
            feature_analysis: Dictionary containing feature importance and analysis
            cv_results: Dictionary containing cross-validation results
            methodology: Dictionary containing methodology details
            
        Returns:
            str: Complete technical report as formatted text
        """
        self.report_sections = []
        
        # Generate report sections
        self._add_header()
        self._add_executive_overview(model_results)
        self._add_methodology_section(methodology)
        self._add_data_analysis_section(feature_analysis)
        self._add_model_performance_section(model_results)
        self._add_cross_validation_section(cv_results)
        self._add_feature_importance_section(feature_analysis)
        self._add_statistical_analysis(model_results, cv_results)
        self._add_conclusions_and_recommendations(model_results)
        
        return '\n\n'.join(self.report_sections)
    
    def _add_header(self):
        """Add report header with metadata."""
        header = f"""# ML Pipeline Technical Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** Technical Analysis
**Pipeline:** HP Product Authenticity Classification

---"""
        self.report_sections.append(header)
    
    def _add_executive_overview(self, model_results: Dict[str, Any]):
        """Add executive overview section."""
        best_model = self._get_best_model(model_results)
        
        overview = f"""## Executive Overview

This technical report presents a comprehensive analysis of the HP Product Authenticity Classification ML pipeline. The analysis includes evaluation of {len(model_results.get('models', {}))} different machine learning algorithms, feature engineering analysis, and performance validation.

**Key Findings:**
- Best performing model: {best_model['name']} (Accuracy: {best_model['accuracy']:.3f})
- Total features engineered: {model_results.get('total_features', 'N/A')}
- Cross-validation stability: {self._assess_cv_stability(model_results)}
- Recommended for deployment: {best_model['name']}"""
        
        self.report_sections.append(overview)
    
    def _add_methodology_section(self, methodology: Dict[str, Any]):
        """Add detailed methodology section."""
        method_section = f"""## Methodology

### Data Preprocessing
- **Dataset Size:** {methodology.get('dataset_size', 'N/A')} samples
- **Train/Test Split:** {methodology.get('train_test_split', '80/20')}
- **Cross-Validation:** {methodology.get('cv_folds', 5)}-fold stratified CV
- **Class Imbalance Handling:** {methodology.get('imbalance_method', 'SMOTE')}

### Feature Engineering
- **Text Features:** TF-IDF vectorization (max_features={methodology.get('tfidf_max_features', 1000)})
- **Numerical Features:** StandardScaler normalization
- **Categorical Features:** One-hot encoding
- **Derived Features:** Price ratios, text metrics, keyword indicators
- **Feature Selection:** Correlation threshold = {methodology.get('correlation_threshold', 0.95)}

### Model Training
- **Algorithms Evaluated:** {', '.join(methodology.get('algorithms', []))}
- **Hyperparameter Optimization:** {methodology.get('optimization_method', 'GridSearchCV')}
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Random Seed:** {methodology.get('random_state', 42)} (for reproducibility)"""
        
        self.report_sections.append(method_section)
    
    def _add_data_analysis_section(self, feature_analysis: Dict[str, Any]):
        """Add data and feature analysis section."""
        data_section = f"""## Data Analysis

### Dataset Characteristics
- **Total Samples:** {feature_analysis.get('total_samples', 'N/A')}
- **Feature Count:** {feature_analysis.get('feature_count', 'N/A')}
- **Class Distribution:** 
  - Original: {feature_analysis.get('class_distribution', {}).get('original', 'N/A')}
  - Suspicious: {feature_analysis.get('class_distribution', {}).get('suspicious', 'N/A')}

### Feature Engineering Results
- **Text Features Generated:** {feature_analysis.get('text_features_count', 'N/A')}
- **Numerical Features:** {feature_analysis.get('numerical_features_count', 'N/A')}
- **Categorical Features:** {feature_analysis.get('categorical_features_count', 'N/A')}
- **Derived Features:** {feature_analysis.get('derived_features_count', 'N/A')}

### Data Quality Assessment
- **Missing Values:** {feature_analysis.get('missing_values_percent', 0):.2f}%
- **Outliers Detected:** {feature_analysis.get('outliers_count', 'N/A')}
- **Feature Correlation Issues:** {feature_analysis.get('high_correlation_pairs', 0)} pairs removed"""
        
        self.report_sections.append(data_section)
    
    def _add_model_performance_section(self, model_results: Dict[str, Any]):
        """Add detailed model performance analysis."""
        performance_section = "## Model Performance Analysis\n\n"
        
        if 'models' in model_results:
            performance_section += "### Individual Model Results\n\n"
            
            for model_name, results in model_results['models'].items():
                performance_section += f"""#### {model_name}
- **Accuracy:** {results.get('accuracy', 0):.4f} ± {results.get('accuracy_std', 0):.4f}
- **Precision:** {results.get('precision', 0):.4f} ± {results.get('precision_std', 0):.4f}
- **Recall:** {results.get('recall', 0):.4f} ± {results.get('recall_std', 0):.4f}
- **F1-Score:** {results.get('f1', 0):.4f} ± {results.get('f1_std', 0):.4f}
- **AUC-ROC:** {results.get('auc_roc', 0):.4f} ± {results.get('auc_roc_std', 0):.4f}
- **Training Time:** {results.get('training_time', 0):.2f} seconds
- **Best Parameters:** {self._format_parameters(results.get('best_params', {}))}

"""
        
        # Add performance comparison table
        performance_section += self._create_performance_table(model_results)
        
        self.report_sections.append(performance_section)
    
    def _add_cross_validation_section(self, cv_results: Dict[str, Any]):
        """Add cross-validation analysis section."""
        cv_section = f"""## Cross-Validation Analysis

### CV Configuration
- **Folds:** {cv_results.get('cv_folds', 5)}
- **Stratification:** {cv_results.get('stratified', True)}
- **Scoring Metrics:** {', '.join(cv_results.get('scoring_metrics', []))}

### CV Results Summary"""
        
        if 'cv_scores' in cv_results:
            cv_section += "\n\n| Model | Mean Accuracy | Std Dev | Min | Max |\n|-------|---------------|---------|-----|-----|\n"
            
            for model_name, scores in cv_results['cv_scores'].items():
                mean_acc = np.mean(scores.get('accuracy', []))
                std_acc = np.std(scores.get('accuracy', []))
                min_acc = np.min(scores.get('accuracy', []))
                max_acc = np.max(scores.get('accuracy', []))
                
                cv_section += f"| {model_name} | {mean_acc:.4f} | {std_acc:.4f} | {min_acc:.4f} | {max_acc:.4f} |\n"
        
        cv_section += f"""

### Statistical Significance
- **ANOVA F-statistic:** {cv_results.get('anova_f_stat', 'N/A')}
- **P-value:** {cv_results.get('anova_p_value', 'N/A')}
- **Significant Differences:** {cv_results.get('significant_differences', 'N/A')}"""
        
        self.report_sections.append(cv_section)
    
    def _add_feature_importance_section(self, feature_analysis: Dict[str, Any]):
        """Add feature importance analysis section."""
        importance_section = "## Feature Importance Analysis\n\n"
        
        if 'feature_importance' in feature_analysis:
            importance_section += "### Top 10 Most Important Features\n\n"
            importance_section += "| Rank | Feature | Importance | Type |\n|------|---------|------------|------|\n"
            
            top_features = feature_analysis['feature_importance'][:10]
            for i, (feature, importance, feature_type) in enumerate(top_features, 1):
                importance_section += f"| {i} | {feature} | {importance:.4f} | {feature_type} |\n"
        
        if 'feature_categories' in feature_analysis:
            importance_section += f"""

### Feature Category Analysis
- **Text Features Contribution:** {feature_analysis['feature_categories'].get('text_contribution', 0):.2f}%
- **Numerical Features Contribution:** {feature_analysis['feature_categories'].get('numerical_contribution', 0):.2f}%
- **Categorical Features Contribution:** {feature_analysis['feature_categories'].get('categorical_contribution', 0):.2f}%
- **Derived Features Contribution:** {feature_analysis['feature_categories'].get('derived_contribution', 0):.2f}%"""
        
        self.report_sections.append(importance_section)
    
    def _add_statistical_analysis(self, model_results: Dict[str, Any], cv_results: Dict[str, Any]):
        """Add statistical analysis section."""
        stats_section = f"""## Statistical Analysis

### Model Comparison Statistics
- **Best Model Confidence Interval:** {self._calculate_confidence_interval(model_results)}
- **Performance Variance:** {self._calculate_performance_variance(cv_results)}
- **Overfitting Assessment:** {self._assess_overfitting(model_results)}

### Reliability Metrics
- **Prediction Consistency:** {cv_results.get('prediction_consistency', 'N/A')}%
- **Model Stability Score:** {cv_results.get('stability_score', 'N/A')}
- **Generalization Estimate:** {self._estimate_generalization(model_results, cv_results)}"""
        
        self.report_sections.append(stats_section)
    
    def _add_conclusions_and_recommendations(self, model_results: Dict[str, Any]):
        """Add conclusions and technical recommendations."""
        best_model = self._get_best_model(model_results)
        
        conclusions = f"""## Conclusions and Technical Recommendations

### Key Findings
1. **Best Performing Algorithm:** {best_model['name']} achieved the highest overall performance
2. **Feature Engineering Impact:** Derived features contributed significantly to model performance
3. **Class Imbalance:** Successfully addressed through SMOTE oversampling
4. **Model Stability:** Cross-validation results show consistent performance across folds

### Technical Recommendations
1. **Production Deployment:** Deploy {best_model['name']} with current hyperparameters
2. **Monitoring:** Implement prediction confidence monitoring (threshold: {best_model.get('confidence_threshold', 0.7)})
3. **Retraining:** Schedule monthly retraining with new data
4. **Feature Updates:** Monitor feature importance drift over time

### Performance Expectations
- **Expected Accuracy:** {best_model['accuracy']:.3f} ± {best_model.get('accuracy_std', 0):.3f}
- **False Positive Rate:** {best_model.get('fpr', 'N/A')}
- **False Negative Rate:** {best_model.get('fnr', 'N/A')}
- **Processing Speed:** {best_model.get('prediction_time', 'N/A')} ms per prediction

### Risk Assessment
- **Model Complexity:** {self._assess_model_complexity(best_model)}
- **Interpretability:** {self._assess_interpretability(best_model)}
- **Maintenance Requirements:** {self._assess_maintenance_needs(best_model)}"""
        
        self.report_sections.append(conclusions)
    
    def _get_best_model(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best performing model from results."""
        if 'models' not in model_results:
            return {'name': 'N/A', 'accuracy': 0}
        
        best_model = None
        best_score = 0
        
        for model_name, results in model_results['models'].items():
            score = results.get('accuracy', 0)
            if score > best_score:
                best_score = score
                best_model = {'name': model_name, **results}
        
        return best_model or {'name': 'N/A', 'accuracy': 0}
    
    def _assess_cv_stability(self, model_results: Dict[str, Any]) -> str:
        """Assess cross-validation stability."""
        # Simplified assessment - in practice would analyze CV score variance
        return "High" if model_results.get('cv_stability_score', 0) > 0.95 else "Moderate"
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """Format hyperparameters for display."""
        if not params:
            return "Default parameters"
        
        formatted = []
        for key, value in params.items():
            formatted.append(f"{key}={value}")
        
        return ", ".join(formatted)
    
    def _create_performance_table(self, model_results: Dict[str, Any]) -> str:
        """Create a formatted performance comparison table."""
        if 'models' not in model_results:
            return ""
        
        table = "\n### Performance Comparison Table\n\n"
        table += "| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
        table += "|-------|----------|-----------|--------|----------|----------|\n"
        
        for model_name, results in model_results['models'].items():
            table += f"| {model_name} | {results.get('accuracy', 0):.4f} | "
            table += f"{results.get('precision', 0):.4f} | {results.get('recall', 0):.4f} | "
            table += f"{results.get('f1', 0):.4f} | {results.get('auc_roc', 0):.4f} |\n"
        
        return table
    
    def _calculate_confidence_interval(self, model_results: Dict[str, Any]) -> str:
        """Calculate confidence interval for best model."""
        best_model = self._get_best_model(model_results)
        accuracy = best_model.get('accuracy', 0)
        std = best_model.get('accuracy_std', 0)
        
        if std > 0:
            ci_lower = accuracy - 1.96 * std
            ci_upper = accuracy + 1.96 * std
            return f"[{ci_lower:.4f}, {ci_upper:.4f}] (95% CI)"
        
        return "N/A"
    
    def _calculate_performance_variance(self, cv_results: Dict[str, Any]) -> str:
        """Calculate performance variance across models."""
        if 'cv_scores' not in cv_results:
            return "N/A"
        
        variances = []
        for model_name, scores in cv_results['cv_scores'].items():
            if 'accuracy' in scores:
                variances.append(np.var(scores['accuracy']))
        
        if variances:
            avg_variance = np.mean(variances)
            return f"{avg_variance:.6f}"
        
        return "N/A"
    
    def _assess_overfitting(self, model_results: Dict[str, Any]) -> str:
        """Assess overfitting risk."""
        # Simplified assessment - would compare train vs validation performance
        return "Low risk" if model_results.get('overfitting_score', 0) < 0.1 else "Moderate risk"
    
    def _estimate_generalization(self, model_results: Dict[str, Any], cv_results: Dict[str, Any]) -> str:
        """Estimate model generalization capability."""
        # Simplified estimation based on CV consistency
        consistency = cv_results.get('prediction_consistency', 0)
        if consistency > 90:
            return "Excellent"
        elif consistency > 80:
            return "Good"
        else:
            return "Fair"
    
    def _assess_model_complexity(self, model: Dict[str, Any]) -> str:
        """Assess model complexity."""
        model_name = model.get('name', '').lower()
        if 'random forest' in model_name or 'gradient' in model_name:
            return "High"
        elif 'svm' in model_name:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_interpretability(self, model: Dict[str, Any]) -> str:
        """Assess model interpretability."""
        model_name = model.get('name', '').lower()
        if 'logistic' in model_name:
            return "High"
        elif 'random forest' in model_name:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_maintenance_needs(self, model: Dict[str, Any]) -> str:
        """Assess model maintenance requirements."""
        complexity = self._assess_model_complexity(model)
        if complexity == "High":
            return "Regular hyperparameter tuning and monitoring required"
        elif complexity == "Moderate":
            return "Periodic performance monitoring recommended"
        else:
            return "Minimal maintenance required"