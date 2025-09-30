# Model Interpretation Summary Report

Generated on: 2025-09-30T14:19:49.550114

## Executive Summary

This report provides a comprehensive analysis of model interpretability, including feature importance analysis, prediction explanations, and actionable insights.

## Feature Analysis

### Most Important Features
1. **feature_14** (feature_14)
2. **feature_11** (feature_11)
3. **feature_15** (feature_15)
4. **feature_2** (feature_2)
5. **feature_17** (feature_17)
6. **feature_7** (feature_7)
7. **feature_18** (feature_18)
8. **feature_16** (feature_16)
9. **feature_3** (feature_3)
10. **feature_9** (feature_9)

### Feature Statistics
- Total unique features analyzed: 20
- Most consistent features: feature_11, feature_7, feature_14, feature_17, feature_18

## Model Comparison
### Model Similarity Analysis
- RandomForest_vs_LogisticRegression: 0.597

## Prediction Analysis
- Total predictions analyzed: 5
- Average confidence: 0.772
- Confidence standard deviation: 0.155

### Most Frequently Important Features in Predictions
- **feature_14**: 5 times
- **feature_1**: 5 times
- **feature_17**: 4 times
- **feature_16**: 4 times
- **feature_2**: 4 times
- **feature_15**: 4 times
- **feature_11**: 3 times
- **feature_12**: 3 times
- **feature_9**: 3 times
- **feature_8**: 3 times

## Recommendations
1. Regularly monitor feature importance to detect data drift and model degradation.
2. Use prediction explanations to build trust with stakeholders and identify edge cases.

## Generated Visualizations
- **Randomforest Tree Importance**: `test_interpretation_plots\RandomForest_tree_importance.png`
- **Randomforest Permutation Importance**: `test_interpretation_plots\RandomForest_permutation_importance.png`
- **Logisticregression Permutation Importance**: `test_interpretation_plots\LogisticRegression_permutation_importance.png`
- **Importance Comparison**: `test_interpretation_plots\importance_comparison.png`
- **Importance Heatmap**: `test_interpretation_plots\importance_heatmap.png`
- **Confidence Distribution**: `test_interpretation_plots\confidence_distribution.png`
- **Prediction Explanation 1**: `test_interpretation_plots\prediction_explanation_1.png`
- **Prediction Explanation 2**: `test_interpretation_plots\prediction_explanation_2.png`
- **Prediction Explanation 3**: `test_interpretation_plots\prediction_explanation_3.png`
- **Prediction Explanation 4**: `test_interpretation_plots\prediction_explanation_4.png`
- **Prediction Explanation 5**: `test_interpretation_plots\prediction_explanation_5.png`

## Technical Details
- Feature names: 20 features
- Class names: Original, Suspicious
- Models analyzed: RandomForest, LogisticRegression
