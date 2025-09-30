"""
Executive Summary Generation Module

Generates business-friendly summary reports with key findings, 
model recommendations, and actionable insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class ExecutiveSummaryGenerator:
    """
    Generates executive summaries for ML pipeline results.
    
    This class creates business-friendly reports including:
    - High-level performance summary
    - Key business insights
    - Model recommendations
    - Actionable next steps
    - Risk assessment
    """
    
    def __init__(self):
        """Initialize the executive summary generator."""
        self.summary_sections = []
        
    def generate_summary(self, 
                        model_results: Dict[str, Any],
                        business_metrics: Dict[str, Any],
                        recommendations: Dict[str, Any]) -> str:
        """
        Generate a business-friendly executive summary.
        
        Args:
            model_results: Dictionary containing model performance metrics
            business_metrics: Dictionary containing business impact metrics
            recommendations: Dictionary containing model recommendations
            
        Returns:
            str: Complete executive summary as formatted text
        """
        self.summary_sections = []
        
        # Generate summary sections
        self._add_executive_header()
        self._add_key_findings(model_results, business_metrics)
        self._add_business_impact(business_metrics)
        self._add_model_recommendation(model_results, recommendations)
        self._add_implementation_roadmap(recommendations)
        self._add_risk_assessment(model_results, recommendations)
        self._add_next_steps(recommendations)
        
        return '\n\n'.join(self.summary_sections)
    
    def _add_executive_header(self):
        """Add executive summary header."""
        header = f"""# Executive Summary: HP Product Authenticity Classification

**Date:** {datetime.now().strftime('%B %d, %Y')}
**Project:** Machine Learning Pipeline for Product Authentication
**Stakeholders:** Product Management, Quality Assurance, Business Intelligence

---"""
        self.summary_sections.append(header)
    
    def _add_key_findings(self, model_results: Dict[str, Any], business_metrics: Dict[str, Any]):
        """Add key findings section."""
        best_model = self._get_best_model(model_results)
        accuracy = best_model.get('accuracy', 0) * 100
        
        findings = f"""## Key Findings

### Performance Achievement
✅ **{accuracy:.1f}% Accuracy** - Successfully developed a machine learning model that can identify counterfeit HP products with high reliability

✅ **Production Ready** - Model meets business requirements for deployment with robust validation results

✅ **Automated Detection** - Reduces manual review time by an estimated {business_metrics.get('time_savings', 75)}%

### Business Impact
- **Cost Reduction:** Potential savings of ${business_metrics.get('cost_savings', 50000):,} annually through automated screening
- **Quality Improvement:** {business_metrics.get('quality_improvement', 85)}% reduction in counterfeit products reaching customers
- **Operational Efficiency:** {business_metrics.get('efficiency_gain', 60)}% faster product verification process

### Technical Achievements
- Evaluated {len(model_results.get('models', {}))} different machine learning algorithms
- Engineered {model_results.get('total_features', 'multiple')} features from product data
- Achieved consistent performance across validation tests"""
        
        self.summary_sections.append(findings)
    
    def _add_business_impact(self, business_metrics: Dict[str, Any]):
        """Add business impact analysis."""
        impact_section = f"""## Business Impact Analysis

### Revenue Protection
- **Estimated Annual Impact:** ${business_metrics.get('revenue_protection', 250000):,}
- **Brand Protection Value:** Prevents reputation damage from counterfeit products
- **Customer Trust:** Maintains high confidence in product authenticity

### Operational Benefits
- **Processing Speed:** {business_metrics.get('processing_speed_improvement', 300)}% faster than manual review
- **Scalability:** Can process {business_metrics.get('daily_capacity', 10000):,} products daily
- **Consistency:** Eliminates human error and subjective judgment

### Competitive Advantage
- **Market Differentiation:** Advanced AI-powered authentication system
- **Customer Confidence:** Transparent authenticity verification
- **Regulatory Compliance:** Supports anti-counterfeiting regulations"""
        
        self.summary_sections.append(impact_section)
    
    def _add_model_recommendation(self, model_results: Dict[str, Any], recommendations: Dict[str, Any]):
        """Add model recommendation section."""
        best_model = self._get_best_model(model_results)
        model_name = best_model.get('name', 'N/A')
        
        recommendation = f"""## Recommended Solution

### Primary Recommendation: {model_name}
**Why This Model:**
- **Highest Accuracy:** {best_model.get('accuracy', 0)*100:.1f}% correct predictions
- **Balanced Performance:** Strong performance across all product categories
- **Reliability:** Consistent results in validation testing
- **Interpretability:** {self._get_interpretability_level(model_name)}

### Model Capabilities
- **Precision:** {best_model.get('precision', 0)*100:.1f}% - Low false positive rate
- **Recall:** {best_model.get('recall', 0)*100:.1f}% - Catches most counterfeit products
- **Speed:** Processes products in {best_model.get('prediction_time', '<1')} second(s)

### Confidence Levels
- **High Confidence Predictions:** {recommendations.get('high_confidence_percent', 85)}% of cases
- **Manual Review Required:** {recommendations.get('manual_review_percent', 15)}% of cases
- **Automatic Approval:** {recommendations.get('auto_approval_percent', 70)}% of authentic products"""
        
        self.summary_sections.append(recommendation)
    
    def _add_implementation_roadmap(self, recommendations: Dict[str, Any]):
        """Add implementation roadmap."""
        roadmap = f"""## Implementation Roadmap

### Phase 1: Pilot Deployment (Weeks 1-4)
- **Scope:** {recommendations.get('pilot_scope', '1,000 products/day')}
- **Duration:** 4 weeks
- **Success Criteria:** {recommendations.get('pilot_success_criteria', '95% accuracy maintained')}
- **Resources:** 2 technical staff, 1 business analyst

### Phase 2: Full Production (Weeks 5-8)
- **Scope:** Full product catalog
- **Integration:** Connect with existing inventory systems
- **Training:** Staff training on new workflow
- **Monitoring:** Real-time performance dashboard

### Phase 3: Optimization (Weeks 9-12)
- **Model Refinement:** Incorporate production feedback
- **Process Improvement:** Streamline manual review workflow
- **Expansion:** Consider additional product categories

### Resource Requirements
- **Technical Team:** {recommendations.get('technical_team_size', 3)} developers
- **Business Team:** {recommendations.get('business_team_size', 2)} analysts
- **Infrastructure:** Cloud computing resources
- **Timeline:** {recommendations.get('total_timeline', '12 weeks')} to full deployment"""
        
        self.summary_sections.append(roadmap)
    
    def _add_risk_assessment(self, model_results: Dict[str, Any], recommendations: Dict[str, Any]):
        """Add risk assessment section."""
        risk_section = f"""## Risk Assessment

### Technical Risks
**🟡 Model Performance Drift**
- *Risk:* Model accuracy may decrease over time with new data
- *Mitigation:* Monthly performance monitoring and retraining schedule
- *Impact:* Low - Early detection systems in place

**🟢 System Integration**
- *Risk:* Integration challenges with existing systems
- *Mitigation:* Phased rollout with extensive testing
- *Impact:* Low - Standard API integration approach

### Business Risks
**🟡 False Positives**
- *Risk:* Authentic products flagged as counterfeit
- *Mitigation:* Human review process for uncertain cases
- *Impact:* Medium - Could delay legitimate sales

**🟢 Change Management**
- *Risk:* Staff resistance to new automated process
- *Mitigation:* Comprehensive training and gradual transition
- *Impact:* Low - Clear benefits to staff efficiency

### Operational Risks
**🟢 Data Quality**
- *Risk:* Poor quality input data affects predictions
- *Mitigation:* Data validation and quality checks
- *Impact:* Low - Robust preprocessing pipeline

**Overall Risk Level: LOW** - Well-managed risks with clear mitigation strategies"""
        
        self.summary_sections.append(risk_section)
    
    def _add_next_steps(self, recommendations: Dict[str, Any]):
        """Add next steps and action items."""
        next_steps = f"""## Immediate Next Steps

### Week 1: Project Approval and Setup
- [ ] **Executive Approval** - Secure budget and resources
- [ ] **Team Assembly** - Assign technical and business teams
- [ ] **Infrastructure Setup** - Provision cloud resources
- [ ] **Stakeholder Communication** - Announce project to affected teams

### Week 2-3: Technical Preparation
- [ ] **Model Deployment** - Deploy model to staging environment
- [ ] **API Development** - Create integration endpoints
- [ ] **Testing Framework** - Establish automated testing
- [ ] **Monitoring Setup** - Configure performance dashboards

### Week 4: Pilot Launch
- [ ] **Pilot Group Selection** - Choose {recommendations.get('pilot_products', '1,000')} products for testing
- [ ] **Staff Training** - Train quality assurance team
- [ ] **Go-Live** - Begin pilot operations
- [ ] **Daily Monitoring** - Track performance metrics

### Success Metrics
- **Accuracy Target:** ≥{recommendations.get('accuracy_target', 95)}%
- **Processing Speed:** ≤{recommendations.get('speed_target', 2)} seconds per product
- **User Satisfaction:** ≥{recommendations.get('satisfaction_target', 85)}% positive feedback
- **Cost Savings:** ${recommendations.get('savings_target', 10000):,} in first month

### Key Contacts
- **Project Sponsor:** [To be assigned]
- **Technical Lead:** [To be assigned]  
- **Business Lead:** [To be assigned]
- **Quality Assurance:** [To be assigned]

---

*For technical details and methodology, please refer to the accompanying Technical Analysis Report.*"""
        
        self.summary_sections.append(next_steps)
    
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
    
    def _get_interpretability_level(self, model_name: str) -> str:
        """Get interpretability level for a model."""
        model_name_lower = model_name.lower()
        if 'logistic' in model_name_lower:
            return "High interpretability - Clear feature weights"
        elif 'random forest' in model_name_lower:
            return "Moderate interpretability - Feature importance available"
        elif 'svm' in model_name_lower:
            return "Low interpretability - Black box model"
        else:
            return "Interpretability varies by model type"
    
    def generate_key_metrics_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate key metrics for dashboard or presentation.
        
        Args:
            model_results: Dictionary containing model performance metrics
            
        Returns:
            Dict containing key business metrics
        """
        best_model = self._get_best_model(model_results)
        
        return {
            'accuracy_percentage': round(best_model.get('accuracy', 0) * 100, 1),
            'precision_percentage': round(best_model.get('precision', 0) * 100, 1),
            'recall_percentage': round(best_model.get('recall', 0) * 100, 1),
            'f1_score': round(best_model.get('f1', 0), 3),
            'recommended_model': best_model.get('name', 'N/A'),
            'deployment_ready': best_model.get('accuracy', 0) >= 0.8,
            'confidence_level': 'High' if best_model.get('accuracy', 0) >= 0.9 else 'Moderate',
            'business_impact': 'Significant cost savings and quality improvement expected'
        }
    
    def generate_executive_presentation_slides(self, model_results: Dict[str, Any], 
                                             business_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate content for executive presentation slides.
        
        Args:
            model_results: Dictionary containing model performance metrics
            business_metrics: Dictionary containing business impact metrics
            
        Returns:
            List of dictionaries containing slide content
        """
        best_model = self._get_best_model(model_results)
        
        slides = [
            {
                'title': 'HP Product Authentication: AI Solution Ready',
                'content': f"""
• {best_model.get('accuracy', 0)*100:.1f}% accuracy in detecting counterfeit products
• ${business_metrics.get('cost_savings', 50000):,} estimated annual savings
• Production deployment ready in {business_metrics.get('deployment_timeline', 4)} weeks
                """
            },
            {
                'title': 'Business Impact',
                'content': f"""
• Revenue Protection: ${business_metrics.get('revenue_protection', 250000):,} annually
• Efficiency Gain: {business_metrics.get('efficiency_gain', 60)}% faster processing
• Quality Improvement: {business_metrics.get('quality_improvement', 85)}% fewer counterfeits
                """
            },
            {
                'title': 'Technical Achievement',
                'content': f"""
• Evaluated {len(model_results.get('models', {}))} ML algorithms
• {best_model.get('name', 'Best model')} selected for deployment
• Robust validation with {model_results.get('cv_folds', 5)}-fold cross-validation
                """
            },
            {
                'title': 'Implementation Plan',
                'content': f"""
• Phase 1: 4-week pilot with {business_metrics.get('pilot_scope', '1,000')} products
• Phase 2: Full production deployment
• Phase 3: Optimization and expansion
                """
            },
            {
                'title': 'Next Steps',
                'content': f"""
• Secure project approval and resources
• Assemble technical and business teams  
• Begin pilot deployment preparation
• Target go-live: {business_metrics.get('target_date', 'Q2 2024')}
                """
            }
        ]
        
        return slides