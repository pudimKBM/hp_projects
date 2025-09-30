# ML Pipeline Source Code

This directory contains the modular machine learning pipeline for HP product authenticity classification. The pipeline is designed with a clean, modular architecture that separates concerns and enables easy testing, maintenance, and extension.

## 🏗️ Architecture Overview

The pipeline follows a data flow architecture where each module has a specific responsibility:

```
Raw Data → Feature Engineering → Preprocessing → Model Training → Hyperparameter Optimization → Validation → Interpretation → Persistence → Reporting
```

## 📁 Module Structure

### 🔧 `feature_engineering/`
Transforms raw product data into ML-ready features.

**Key Components:**
- `text_features.py` - TF-IDF vectorization, text length metrics, keyword extraction
- `numerical_features.py` - Scaling, normalization, outlier handling
- `categorical_features.py` - One-hot encoding, label encoding, frequency encoding
- `correlation_analysis.py` - Feature correlation analysis and removal
- `pipeline.py` - Complete feature engineering pipeline

**Usage:**
```python
from src.feature_engineering import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline()
X, y = pipeline.fit_transform(df, target_column='target_is_original')
```

### 🔄 `preprocessing/`
Handles data splitting and class imbalance correction.

**Key Components:**
- `data_splitting.py` - Stratified train-test splitting with validation
- `imbalance_handling.py` - SMOTE oversampling and class weight calculation
- `pipeline.py` - Complete preprocessing pipeline

**Usage:**
```python
from src.preprocessing import create_complete_preprocessing_pipeline

results = create_complete_preprocessing_pipeline(
    X, y, test_size=0.2, imbalance_method='smote'
)
```

### 🤖 `models/`
Model training implementations with consistent interfaces.

**Key Components:**
- Individual model training functions for each algorithm
- Consistent parameter handling across models
- Cross-validation integration
- Model metadata tracking

**Usage:**
```python
from src.models import train_random_forest

model, cv_results = train_random_forest(
    X_train, y_train, cv=5, random_state=42
)
```

### ⚡ `hyperparameter_optimization/`
Automated hyperparameter tuning with intelligent method selection.

**Key Components:**
- Automatic selection between GridSearchCV and RandomizedSearchCV
- Budget management and time estimation
- Multi-model optimization
- Progress tracking and result validation

**Usage:**
```python
from src.hyperparameter_optimization import optimize_multiple_models

results = optimize_multiple_models(
    model_list=['random_forest', 'svm', 'logistic_regression'],
    X_train=X_train, y_train=y_train,
    search_budget='medium'
)
```

### 📊 `validation/`
Comprehensive model evaluation and comparison.

**Key Components:**
- `metrics.py` - Performance metrics calculation (precision, recall, F1, AUC-ROC)
- `comparison.py` - Model comparison tables and statistical tests
- `cross_validation.py` - Cross-validation analysis and overfitting detection
- `roc_analysis.py` - ROC curve analysis and AUC calculation
- `visualization.py` - Performance visualization plots

**Usage:**
```python
from src.validation import create_model_comparison_table, plot_roc_curves

comparison = create_model_comparison_table(models_results, cv_results)
roc_fig = plot_roc_curves(models_results)
```

### 🔍 `interpretation/`
Model explainability and feature importance analysis.

**Key Components:**
- `feature_importance.py` - Tree-based and permutation importance
- `prediction_explanation.py` - Individual prediction explanations
- `visualization.py` - Interactive importance plots and explanations
- `pipeline.py` - Complete interpretation pipeline with insights generation

**Usage:**
```python
from src.interpretation import InterpretationPipeline

interpreter = InterpretationPipeline(feature_names, class_names)
importance = interpreter.analyze_feature_importance(model, X_train, y_train)
explanations = interpreter.explain_predictions(model, X_samples, 'RandomForest')
```

### 💾 `persistence/`
Model and pipeline serialization with versioning.

**Key Components:**
- `model_saving.py` - Model serialization with metadata
- `model_loading.py` - Model deserialization with validation
- `metadata.py` - Model metadata management
- `versioning.py` - Model version tracking
- `pipeline_persistence.py` - Complete pipeline serialization

**Usage:**
```python
from src.persistence import ModelSaver, ModelLoader

# Save model
saver = ModelSaver()
saver.save_model(model, 'random_forest_v1', performance_metrics, feature_names)

# Load model
loader = ModelLoader()
loaded_model = loader.load_model('random_forest_v1')
```

### 📈 `reporting/`
Automated report generation and model recommendations.

**Key Components:**
- `technical_report.py` - Detailed technical analysis reports
- `executive_summary.py` - Business-friendly summaries
- `model_recommendation.py` - Automated model selection with business constraints
- `export_functionality.py` - Multi-format report export (PDF, HTML)

**Usage:**
```python
from src.reporting import ModelRecommendationSystem, ReportExporter

# Model recommendation
recommender = ModelRecommendationSystem(business_constraints)
recommendation = recommender.recommend_model(models_results, cv_results)

# Export reports
exporter = ReportExporter()
exporter.export_to_pdf(report_content, 'technical_report.pdf')
```

## 🔧 Configuration

The pipeline is configured through `config.py`:

```python
# Feature Engineering Configuration
FEATURE_CONFIG = {
    'text_features': {
        'max_features': 1000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95
    },
    'correlation_threshold': 0.95,
    'numerical_scaling': 'StandardScaler'
}

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    # ... other models
}

# Validation Configuration
VALIDATION_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'min_accuracy_threshold': 0.8,
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
}
```

## 🚀 Quick Start Examples

### Complete Pipeline Example
```python
from src.feature_engineering import FeatureEngineeringPipeline
from src.preprocessing import create_complete_preprocessing_pipeline
from src.hyperparameter_optimization import optimize_multiple_models
from src.validation import create_model_comparison_table
from src.interpretation import InterpretationPipeline
from src.reporting import ModelRecommendationSystem

# 1. Feature Engineering
feature_pipeline = FeatureEngineeringPipeline()
X, y = feature_pipeline.fit_transform(df)

# 2. Preprocessing
preprocessing_results = create_complete_preprocessing_pipeline(X, y)
X_train, y_train = preprocessing_results['X_train'], preprocessing_results['y_train']
X_test, y_test = preprocessing_results['X_test'], preprocessing_results['y_test']

# 3. Model Training & Optimization
models = ['random_forest', 'svm', 'logistic_regression']
optimization_results = optimize_multiple_models(models, X_train, y_train)

# 4. Model Validation
comparison = create_model_comparison_table(
    optimization_results['models_results'], 
    optimization_results['cv_results']
)

# 5. Model Interpretation
interpreter = InterpretationPipeline(
    feature_names=feature_pipeline.get_feature_names(),
    class_names=['Suspicious', 'Original']
)
importance = interpreter.analyze_feature_importance(best_model, X_train, y_train)

# 6. Model Recommendation
recommender = ModelRecommendationSystem()
recommendation = recommender.recommend_model(
    optimization_results['models_results'],
    optimization_results['cv_results']
)

print(f"Recommended Model: {recommendation['recommended_model']}")
print(f"Deployment Ready: {recommendation['deployment_ready']}")
```

### Individual Module Usage

#### Feature Engineering Only
```python
from src.feature_engineering import create_tfidf_features, create_numerical_features

# Text features
tfidf_features = create_tfidf_features(df, ['title', 'description'])

# Numerical features
numerical_features = create_numerical_features(df, ['price_numeric', 'rating_numeric'])
```

#### Model Training Only
```python
from src.models import train_random_forest, train_svm

# Train individual models
rf_model, rf_cv = train_random_forest(X_train, y_train)
svm_model, svm_cv = train_svm(X_train, y_train)
```

#### Validation Only
```python
from src.validation import calculate_comprehensive_metrics, plot_confusion_matrix

# Calculate metrics
metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)

# Create visualizations
cm_fig = plot_confusion_matrix(y_true, y_pred, class_names=['Suspicious', 'Original'])
```

## 🧪 Testing

Each module includes comprehensive tests. Run tests from the project root:

```bash
# Test individual modules
python notebooks/test_preprocessing.py
python notebooks/test_hyperparameter_pipeline.py
python notebooks/test_validation.py
python notebooks/test_interpretation.py
python notebooks/test_model_recommendation.py
```

## 📊 Module Dependencies

```
feature_engineering → preprocessing → models → hyperparameter_optimization
                                        ↓
validation ← interpretation ← persistence ← reporting
```

**External Dependencies:**
- `scikit-learn` - Core ML algorithms and utilities
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib`, `seaborn` - Visualizations
- `joblib` - Model serialization
- `imbalanced-learn` - SMOTE implementation
- `reportlab` - PDF generation

## 🔄 Data Flow

1. **Raw Data** → Feature Engineering → **Feature Matrix (X) + Target (y)**
2. **X, y** → Preprocessing → **X_train, X_test, y_train, y_test (balanced)**
3. **Training Data** → Model Training → **Trained Models**
4. **Models** → Hyperparameter Optimization → **Optimized Models**
5. **Models + Test Data** → Validation → **Performance Metrics**
6. **Models + Data** → Interpretation → **Feature Importance + Explanations**
7. **Models + Metadata** → Persistence → **Saved Models + Versions**
8. **All Results** → Reporting → **Technical Reports + Executive Summaries**

## 🎯 Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Consistency**: Uniform interfaces across all modules
3. **Testability**: Comprehensive test coverage for all components
4. **Configurability**: Extensive configuration options
5. **Extensibility**: Easy to add new models, features, or metrics
6. **Robustness**: Comprehensive error handling and validation
7. **Documentation**: Clear docstrings and usage examples

## 🚀 Performance Considerations

- **Memory Efficiency**: Sparse matrix support for text features
- **Computational Efficiency**: Optimized algorithms and parallel processing
- **Scalability**: Batch processing capabilities for large datasets
- **Caching**: Intermediate results caching to avoid recomputation

## 🔧 Extending the Pipeline

### Adding a New Model
1. Create training function in `models/`
2. Add model configuration to `config.py`
3. Update hyperparameter grids
4. Add tests

### Adding New Features
1. Implement feature extraction in appropriate `feature_engineering/` module
2. Update `FeatureEngineeringPipeline`
3. Add configuration options
4. Update tests

### Adding New Metrics
1. Implement metric calculation in `validation/metrics.py`
2. Update visualization functions
3. Add to default metrics list in config
4. Update comparison functions

This modular architecture ensures the pipeline is maintainable, testable, and extensible while providing comprehensive ML capabilities for HP product authenticity classification.