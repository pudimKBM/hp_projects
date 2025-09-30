"""
Classification Engine Service

Provides classification functionality with confidence scores and explanations.
Integrates ML models, feature preparation, and interpretation pipelines.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

# Import from existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.interpretation import (
    explain_prediction,
    calculate_prediction_confidence,
    get_top_contributing_features,
    InterpretationPipeline
)

from .ml_service import MLService
from .feature_service import FeaturePreparationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationEngine:
    """
    Classification engine that generates predictions with confidence scores and explanations
    """
    
    def __init__(
        self,
        ml_service: MLService,
        feature_service: FeaturePreparationService,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize Classification Engine
        
        Args:
            ml_service: ML service for model management
            feature_service: Feature preparation service
            confidence_threshold: Minimum confidence threshold for predictions
        """
        self.ml_service = ml_service
        self.feature_service = feature_service
        self.confidence_threshold = confidence_threshold
        
        # Classification metadata
        self.class_names = ['suspicious', 'original']  # Based on HP product classification
        self.interpretation_pipeline: Optional[InterpretationPipeline] = None
        
        # Performance tracking
        self.classification_stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'avg_processing_time_ms': 0,
            'confidence_distribution': []
        }
        
        logger.info("ClassificationEngine initialized")
    
    def setup_interpretation_pipeline(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Setup interpretation pipeline for prediction explanations
        
        Args:
            feature_names: List of feature names from the model
        
        Returns:
            Dictionary with setup results
        """
        try:
            # Create business-friendly feature mapping
            business_mapping = self._create_business_feature_mapping(feature_names)
            
            # Initialize interpretation pipeline
            self.interpretation_pipeline = InterpretationPipeline(
                feature_names=feature_names,
                class_names=self.class_names,
                business_feature_mapping=business_mapping
            )
            
            logger.info(f"Interpretation pipeline setup with {len(feature_names)} features")
            
            return {
                'success': True,
                'feature_count': len(feature_names),
                'business_mapping_count': len(business_mapping),
                'class_names': self.class_names
            }
            
        except Exception as e:
            logger.error(f"Error setting up interpretation pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def classify_product(
        self,
        product_data: Dict[str, Any],
        model_name: str,
        include_explanation: bool = True,
        top_features: int = 10
    ) -> Dict[str, Any]:
        """
        Classify a single product with confidence scores and explanations
        
        Args:
            product_data: Product data dictionary
            model_name: Name of the model to use for classification
            include_explanation: Whether to include prediction explanation
            top_features: Number of top features to include in explanation
        
        Returns:
            Dictionary with classification results
        """
        start_time = time.time()
        
        try:
            # Get active model
            active_model = self.ml_service.get_active_model(model_name)
            if not active_model:
                return {
                    'success': False,
                    'error': f'Model {model_name} not loaded',
                    'prediction': None,
                    'confidence_score': 0.0
                }
            
            model = active_model['model']
            
            # Prepare features
            feature_result = self.feature_service.prepare_features_for_single_product(product_data)
            
            if not feature_result['success']:
                return {
                    'success': False,
                    'error': f"Feature preparation failed: {feature_result['error']}",
                    'prediction': None,
                    'confidence_score': 0.0
                }
            
            features = feature_result['features']
            feature_names = feature_result['feature_names']
            
            # Make prediction
            prediction_result = self._make_prediction(model, features, feature_names)
            
            if not prediction_result['success']:
                return prediction_result
            
            # Generate explanation if requested
            explanation = None
            if include_explanation and self.interpretation_pipeline:
                explanation = self._generate_explanation(
                    model, features, feature_names, prediction_result, top_features
                )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Update statistics
            self._update_classification_stats(
                prediction_result['confidence_score'], 
                processing_time_ms, 
                success=True
            )
            
            # Prepare final result
            result = {
                'success': True,
                'prediction': prediction_result['prediction'],
                'prediction_label': prediction_result['prediction_label'],
                'confidence_score': prediction_result['confidence_score'],
                'confidence_level': self._get_confidence_level(prediction_result['confidence_score']),
                'processing_time_ms': processing_time_ms,
                'model_name': model_name,
                'model_version': active_model.get('metadata', {}).get('model_version', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'feature_validation': feature_result['validation_result'],
                'explanation': explanation
            }
            
            logger.info(f"Successfully classified product: {prediction_result['prediction_label']} "
                       f"(confidence: {prediction_result['confidence_score']:.3f})")
            
            return result
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self._update_classification_stats(0.0, processing_time_ms, success=False)
            
            logger.error(f"Error classifying product: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence_score': 0.0,
                'processing_time_ms': processing_time_ms
            }
    
    def classify_batch(
        self,
        products_data: List[Dict[str, Any]],
        model_name: str,
        include_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Classify multiple products in batch
        
        Args:
            products_data: List of product data dictionaries
            model_name: Name of the model to use
            include_explanations: Whether to include explanations
        
        Returns:
            Dictionary with batch classification results
        """
        start_time = time.time()
        
        try:
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, product_data in enumerate(products_data):
                try:
                    result = self.classify_product(
                        product_data, 
                        model_name, 
                        include_explanation=include_explanations
                    )
                    
                    result['batch_index'] = i
                    results.append(result)
                    
                    if result['success']:
                        successful_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    results.append({
                        'success': False,
                        'error': str(e),
                        'batch_index': i,
                        'prediction': None,
                        'confidence_score': 0.0
                    })
            
            total_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'total_products': len(products_data),
                'successful_classifications': successful_count,
                'failed_classifications': failed_count,
                'success_rate': successful_count / len(products_data) if products_data else 0,
                'total_processing_time_ms': total_time_ms,
                'avg_time_per_product_ms': total_time_ms / len(products_data) if products_data else 0,
                'results': results,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch classification: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_products': len(products_data),
                'results': []
            }
    
    def _make_prediction(
        self, 
        model: Any, 
        features: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Make prediction using the model
        
        Args:
            model: Trained model
            features: Feature array
            feature_names: List of feature names
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Reshape features if needed (ensure 2D array)
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence_score = float(np.max(probabilities))
                
                # Map prediction to class name
                if hasattr(model, 'classes_'):
                    class_mapping = {i: cls for i, cls in enumerate(model.classes_)}
                    prediction_label = class_mapping.get(prediction, str(prediction))
                else:
                    prediction_label = self.class_names[int(prediction)] if int(prediction) < len(self.class_names) else str(prediction)
            else:
                # For models without probability prediction
                confidence_score = 0.8  # Default confidence
                prediction_label = self.class_names[int(prediction)] if int(prediction) < len(self.class_names) else str(prediction)
            
            return {
                'success': True,
                'prediction': int(prediction),
                'prediction_label': prediction_label,
                'confidence_score': confidence_score,
                'probabilities': probabilities.tolist() if 'probabilities' in locals() else None
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence_score': 0.0
            }
    
    def _generate_explanation(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction_result: Dict[str, Any],
        top_features: int
    ) -> Dict[str, Any]:
        """
        Generate explanation for the prediction
        
        Args:
            model: Trained model
            features: Feature array
            feature_names: List of feature names
            prediction_result: Prediction result dictionary
            top_features: Number of top features to include
        
        Returns:
            Dictionary with explanation
        """
        try:
            # Reshape features if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Generate explanation using interpretation module
            explanation = explain_prediction(
                model=model,
                X_sample=features,
                feature_names=feature_names,
                class_names=self.class_names,
                top_n=top_features
            )
            
            # Add business-friendly explanations
            business_explanation = self._create_business_explanation(
                explanation, prediction_result
            )
            
            return {
                'technical_explanation': explanation,
                'business_explanation': business_explanation,
                'top_features': explanation.get('top_features', [])[:top_features],
                'feature_importance': explanation.get('feature_importance', {}),
                'confidence_factors': self._analyze_confidence_factors(explanation, prediction_result)
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                'error': f"Explanation generation failed: {str(e)}",
                'technical_explanation': None,
                'business_explanation': "Unable to generate explanation"
            }
    
    def _create_business_explanation(
        self, 
        technical_explanation: Dict[str, Any], 
        prediction_result: Dict[str, Any]
    ) -> str:
        """
        Create business-friendly explanation
        
        Args:
            technical_explanation: Technical explanation from interpretation module
            prediction_result: Prediction result
        
        Returns:
            Business-friendly explanation string
        """
        try:
            prediction_label = prediction_result['prediction_label']
            confidence = prediction_result['confidence_score']
            
            # Start with prediction summary
            confidence_text = self._get_confidence_level(confidence).lower()
            explanation = f"This product is classified as '{prediction_label}' with {confidence_text} confidence ({confidence:.1%})."
            
            # Add top contributing factors
            top_features = technical_explanation.get('top_features', [])
            if top_features:
                explanation += f"\n\nKey factors influencing this classification:"
                
                for i, feature_info in enumerate(top_features[:5], 1):
                    feature_name = feature_info.get('feature', 'Unknown')
                    importance = feature_info.get('importance', 0)
                    
                    # Convert to business-friendly name
                    business_name = self._get_business_feature_name(feature_name)
                    
                    # Determine impact direction
                    impact = "supports" if importance > 0 else "contradicts"
                    
                    explanation += f"\n{i}. {business_name} {impact} the '{prediction_label}' classification"
            
            # Add confidence interpretation
            if confidence < self.confidence_threshold:
                explanation += f"\n\nNote: The confidence score is below the threshold ({self.confidence_threshold:.1%}). Consider manual review."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error creating business explanation: {str(e)}")
            return f"Classification: {prediction_result.get('prediction_label', 'Unknown')} (Confidence: {prediction_result.get('confidence_score', 0):.1%})"
    
    def _analyze_confidence_factors(
        self, 
        explanation: Dict[str, Any], 
        prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze factors affecting prediction confidence
        
        Args:
            explanation: Technical explanation
            prediction_result: Prediction result
        
        Returns:
            Dictionary with confidence analysis
        """
        try:
            confidence = prediction_result['confidence_score']
            top_features = explanation.get('top_features', [])
            
            # Analyze feature contributions
            positive_contributions = [f for f in top_features if f.get('importance', 0) > 0]
            negative_contributions = [f for f in top_features if f.get('importance', 0) < 0]
            
            # Calculate contribution strength
            total_positive = sum(f.get('importance', 0) for f in positive_contributions)
            total_negative = abs(sum(f.get('importance', 0) for f in negative_contributions))
            
            return {
                'confidence_level': self._get_confidence_level(confidence),
                'positive_factors_count': len(positive_contributions),
                'negative_factors_count': len(negative_contributions),
                'positive_contribution_strength': total_positive,
                'negative_contribution_strength': total_negative,
                'contribution_balance': total_positive - total_negative,
                'uncertainty_indicators': self._identify_uncertainty_indicators(explanation, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing confidence factors: {str(e)}")
            return {'error': str(e)}
    
    def _identify_uncertainty_indicators(
        self, 
        explanation: Dict[str, Any], 
        confidence: float
    ) -> List[str]:
        """
        Identify indicators of prediction uncertainty
        
        Args:
            explanation: Technical explanation
            confidence: Confidence score
        
        Returns:
            List of uncertainty indicators
        """
        indicators = []
        
        try:
            # Low confidence
            if confidence < 0.6:
                indicators.append("Low overall confidence score")
            
            # Conflicting features
            top_features = explanation.get('top_features', [])
            positive_count = sum(1 for f in top_features if f.get('importance', 0) > 0)
            negative_count = sum(1 for f in top_features if f.get('importance', 0) < 0)
            
            if positive_count > 0 and negative_count > 0:
                indicators.append("Conflicting feature contributions detected")
            
            # Weak feature importance
            if top_features:
                max_importance = max(abs(f.get('importance', 0)) for f in top_features)
                if max_importance < 0.1:
                    indicators.append("Weak feature importance values")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error identifying uncertainty indicators: {str(e)}")
            return ["Error analyzing uncertainty"]
    
    def _create_business_feature_mapping(self, feature_names: List[str]) -> Dict[str, str]:
        """
        Create mapping from technical to business-friendly feature names
        
        Args:
            feature_names: List of technical feature names
        
        Returns:
            Dictionary mapping technical to business names
        """
        mapping = {}
        
        for feature in feature_names:
            # Convert technical names to business-friendly names
            business_name = feature.replace('_', ' ').title()
            
            # Handle specific patterns for HP product classification
            if 'title' in feature.lower():
                business_name = "Product Title Content"
            elif 'description' in feature.lower():
                business_name = "Product Description Content"
            elif 'price' in feature.lower():
                business_name = "Price Information"
            elif 'rating' in feature.lower():
                business_name = "Customer Rating"
            elif 'reviews' in feature.lower():
                business_name = "Number of Reviews"
            elif 'seller' in feature.lower():
                business_name = "Seller Information"
            elif 'platform' in feature.lower():
                business_name = "Sales Platform"
            elif 'hp' in feature.lower():
                business_name = "HP Brand Indicators"
            elif 'original' in feature.lower():
                business_name = "Authenticity Indicators"
            elif 'tfidf' in feature.lower():
                business_name = f"Text Analysis: {feature.split('_')[-1].title()}"
            
            mapping[feature] = business_name
        
        return mapping
    
    def _get_business_feature_name(self, technical_name: str) -> str:
        """
        Get business-friendly name for a feature
        
        Args:
            technical_name: Technical feature name
        
        Returns:
            Business-friendly feature name
        """
        if self.interpretation_pipeline and self.interpretation_pipeline.business_feature_mapping:
            return self.interpretation_pipeline.business_feature_mapping.get(
                technical_name, technical_name
            )
        return technical_name.replace('_', ' ').title()
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Convert confidence score to descriptive level
        
        Args:
            confidence: Confidence score (0-1)
        
        Returns:
            Descriptive confidence level
        """
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Moderate"
        elif confidence >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def _update_classification_stats(
        self, 
        confidence: float, 
        processing_time_ms: int, 
        success: bool
    ):
        """
        Update classification statistics
        
        Args:
            confidence: Confidence score
            processing_time_ms: Processing time in milliseconds
            success: Whether classification was successful
        """
        try:
            self.classification_stats['total_classifications'] += 1
            
            if success:
                self.classification_stats['successful_classifications'] += 1
                self.classification_stats['confidence_distribution'].append(confidence)
            else:
                self.classification_stats['failed_classifications'] += 1
            
            # Update average processing time
            total_time = (self.classification_stats['avg_processing_time_ms'] * 
                         (self.classification_stats['total_classifications'] - 1) + processing_time_ms)
            self.classification_stats['avg_processing_time_ms'] = (
                total_time / self.classification_stats['total_classifications']
            )
            
        except Exception as e:
            logger.error(f"Error updating classification stats: {str(e)}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification performance statistics
        
        Returns:
            Dictionary with classification statistics
        """
        try:
            stats = self.classification_stats.copy()
            
            # Calculate additional metrics
            if stats['confidence_distribution']:
                stats['avg_confidence'] = np.mean(stats['confidence_distribution'])
                stats['confidence_std'] = np.std(stats['confidence_distribution'])
                stats['min_confidence'] = np.min(stats['confidence_distribution'])
                stats['max_confidence'] = np.max(stats['confidence_distribution'])
            else:
                stats['avg_confidence'] = 0.0
                stats['confidence_std'] = 0.0
                stats['min_confidence'] = 0.0
                stats['max_confidence'] = 0.0
            
            # Calculate success rate
            if stats['total_classifications'] > 0:
                stats['success_rate'] = (
                    stats['successful_classifications'] / stats['total_classifications']
                )
            else:
                stats['success_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting classification stats: {str(e)}")
            return {'error': str(e)}
    
    def reset_stats(self):
        """Reset classification statistics"""
        self.classification_stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'avg_processing_time_ms': 0,
            'confidence_distribution': []
        }
        logger.info("Classification statistics reset")


# Convenience functions
def create_classification_engine(
    ml_service: MLService,
    feature_service: FeaturePreparationService,
    confidence_threshold: float = 0.7
) -> ClassificationEngine:
    """
    Create and return a ClassificationEngine instance
    
    Args:
        ml_service: ML service instance
        feature_service: Feature service instance
        confidence_threshold: Confidence threshold
    
    Returns:
        Configured ClassificationEngine instance
    """
    return ClassificationEngine(ml_service, feature_service, confidence_threshold)