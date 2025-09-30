# Feature Correlation Analysis Module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_correlation_matrix(feature_matrix, feature_names, method='pearson'):
    """
    Calculate correlation matrix for features.
    
    Parameters:
    -----------
    feature_matrix : numpy.array or pandas.DataFrame or sparse matrix
        Feature matrix
    feature_names : list
        List of feature names
    method : str
        Correlation method ('pearson', 'spearman')
    
    Returns:
    --------
    pandas.DataFrame : Correlation matrix
    """
    # Convert sparse matrix to dense if needed
    if hasattr(feature_matrix, 'toarray'):
        feature_matrix = feature_matrix.toarray()
    
    # Create DataFrame if needed
    if not isinstance(feature_matrix, pd.DataFrame):
        df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    else:
        df_features = feature_matrix
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df_features.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df_features.corr(method='spearman')
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    print(f"Calculated {method} correlation matrix for {len(feature_names)} features")
    
    return corr_matrix


def find_highly_correlated_features(corr_matrix, threshold=0.95):
    """
    Find pairs of highly correlated features.
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    threshold : float
        Correlation threshold for identifying highly correlated features
    
    Returns:
    --------
    list : List of tuples containing highly correlated feature pairs
    """
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find highly correlated pairs
    highly_correlated = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_value = upper_triangle.loc[idx, col]
            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                highly_correlated.append((idx, col, corr_value))
    
    print(f"Found {len(highly_correlated)} feature pairs with correlation >= {threshold}")
    
    return highly_correlated


def select_features_to_remove(highly_correlated_pairs, feature_importance=None):
    """
    Select which features to remove from highly correlated pairs.
    
    Parameters:
    -----------
    highly_correlated_pairs : list
        List of tuples with highly correlated feature pairs
    feature_importance : dict, optional
        Dictionary mapping feature names to importance scores
    
    Returns:
    --------
    set : Set of feature names to remove
    """
    features_to_remove = set()
    
    for feat1, feat2, corr_value in highly_correlated_pairs:
        # If we have feature importance, keep the more important feature
        if feature_importance:
            imp1 = feature_importance.get(feat1, 0)
            imp2 = feature_importance.get(feat2, 0)
            
            if imp1 >= imp2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        else:
            # Default: remove the second feature (arbitrary choice)
            features_to_remove.add(feat2)
    
    print(f"Selected {len(features_to_remove)} features for removal")
    
    return features_to_remove


def remove_correlated_features(feature_matrix, feature_names, features_to_remove):
    """
    Remove highly correlated features from feature matrix.
    
    Parameters:
    -----------
    feature_matrix : numpy.array or sparse matrix
        Feature matrix
    feature_names : list
        List of feature names
    features_to_remove : set
        Set of feature names to remove
    
    Returns:
    --------
    dict : Dictionary containing filtered features and names
    """
    # Find indices of features to keep
    indices_to_keep = [i for i, name in enumerate(feature_names) if name not in features_to_remove]
    remaining_feature_names = [name for name in feature_names if name not in features_to_remove]
    
    # Filter feature matrix
    if hasattr(feature_matrix, 'toarray'):  # Sparse matrix
        filtered_matrix = feature_matrix[:, indices_to_keep]
    else:  # Dense matrix
        filtered_matrix = feature_matrix[:, indices_to_keep]
    
    print(f"Removed {len(features_to_remove)} features, kept {len(remaining_feature_names)} features")
    
    return {
        'matrix': filtered_matrix,
        'feature_names': remaining_feature_names,
        'removed_features': list(features_to_remove),
        'kept_indices': indices_to_keep
    }


def analyze_feature_correlations(feature_matrix, feature_names, threshold=0.95, 
                               feature_importance=None, create_heatmap=True):
    """
    Comprehensive feature correlation analysis.
    
    Parameters:
    -----------
    feature_matrix : numpy.array or sparse matrix
        Feature matrix
    feature_names : list
        List of feature names
    threshold : float
        Correlation threshold for removal
    feature_importance : dict, optional
        Feature importance scores
    create_heatmap : bool
        Whether to create correlation heatmap
    
    Returns:
    --------
    dict : Dictionary containing correlation analysis results
    """
    print(f"Starting correlation analysis for {len(feature_names)} features...")
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(feature_matrix, feature_names)
    
    # Find highly correlated features
    highly_correlated = find_highly_correlated_features(corr_matrix, threshold)
    
    # Select features to remove
    features_to_remove = select_features_to_remove(highly_correlated, feature_importance)
    
    # Remove correlated features
    filtered_result = remove_correlated_features(feature_matrix, feature_names, features_to_remove)
    
    # Create heatmap if requested and feasible
    heatmap_created = False
    if create_heatmap and len(feature_names) <= 50:  # Only for manageable number of features
        try:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title(f'Feature Correlation Matrix (threshold={threshold})')
            plt.tight_layout()
            plt.show()
            heatmap_created = True
            print("Created correlation heatmap")
        except Exception as e:
            print(f"Could not create heatmap: {e}")
    
    # Summary statistics
    correlation_stats = {
        'total_features': len(feature_names),
        'highly_correlated_pairs': len(highly_correlated),
        'features_removed': len(features_to_remove),
        'features_remaining': len(filtered_result['feature_names']),
        'max_correlation': corr_matrix.abs().max().max() if not corr_matrix.empty else 0,
        'mean_correlation': corr_matrix.abs().mean().mean() if not corr_matrix.empty else 0
    }
    
    print(f"\nCorrelation Analysis Summary:")
    print(f"Total features: {correlation_stats['total_features']}")
    print(f"Highly correlated pairs (>={threshold}): {correlation_stats['highly_correlated_pairs']}")
    print(f"Features removed: {correlation_stats['features_removed']}")
    print(f"Features remaining: {correlation_stats['features_remaining']}")
    print(f"Max correlation: {correlation_stats['max_correlation']:.3f}")
    print(f"Mean correlation: {correlation_stats['mean_correlation']:.3f}")
    
    return {
        'correlation_matrix': corr_matrix,
        'highly_correlated_pairs': highly_correlated,
        'filtered_features': filtered_result,
        'correlation_stats': correlation_stats,
        'heatmap_created': heatmap_created
    }


def create_correlation_report(correlation_results, save_path=None):
    """
    Create a detailed correlation analysis report.
    
    Parameters:
    -----------
    correlation_results : dict
        Results from analyze_feature_correlations
    save_path : str, optional
        Path to save the report
    
    Returns:
    --------
    str : Correlation analysis report
    """
    stats = correlation_results['correlation_stats']
    highly_corr = correlation_results['highly_correlated_pairs']
    
    report = f"""
# Feature Correlation Analysis Report

## Summary Statistics
- **Total Features**: {stats['total_features']}
- **Highly Correlated Pairs**: {stats['highly_correlated_pairs']}
- **Features Removed**: {stats['features_removed']}
- **Features Remaining**: {stats['features_remaining']}
- **Maximum Correlation**: {stats['max_correlation']:.3f}
- **Mean Correlation**: {stats['mean_correlation']:.3f}

## Highly Correlated Feature Pairs
"""
    
    if highly_corr:
        report += "\n| Feature 1 | Feature 2 | Correlation |\n"
        report += "|-----------|-----------|-------------|\n"
        for feat1, feat2, corr in highly_corr[:20]:  # Show top 20
            report += f"| {feat1} | {feat2} | {corr:.3f} |\n"
        
        if len(highly_corr) > 20:
            report += f"\n... and {len(highly_corr) - 20} more pairs\n"
    else:
        report += "\nNo highly correlated feature pairs found.\n"
    
    report += f"""
## Removed Features
{correlation_results['filtered_features']['removed_features']}

## Recommendations
- {'✓' if stats['features_removed'] > 0 else '✗'} Multicollinearity reduction: {stats['features_removed']} features removed
- {'✓' if stats['mean_correlation'] < 0.3 else '⚠️'} Average correlation level: {stats['mean_correlation']:.3f}
- {'✓' if stats['max_correlation'] < 0.95 else '⚠️'} Maximum correlation: {stats['max_correlation']:.3f}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Correlation report saved to: {save_path}")
    
    return report