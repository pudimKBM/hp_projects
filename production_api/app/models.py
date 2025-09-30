"""
Database models for the Production Classification API.

This module defines SQLAlchemy models for:
- Product: Raw product data from scraping
- Classification: ML prediction results
- ScrapingJob: Job execution history and status
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON, Index
import json

db = SQLAlchemy()


class Product(db.Model):
    """Model for storing raw product data from scraping."""
    
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)
    price_numeric = db.Column(db.Float)
    seller_name = db.Column(db.String(200))
    rating_numeric = db.Column(db.Float)
    reviews_count = db.Column(db.Integer, default=0)
    platform = db.Column(db.String(50), default='mercadolivre')
    product_type = db.Column(db.String(100))
    scraped_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    url = db.Column(db.String(1000))
    raw_data = db.Column(JSON)
    
    # Relationship to classifications
    classifications = db.relationship('Classification', backref='product', lazy=True, cascade='all, delete-orphan')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_product_scraped_at', 'scraped_at'),
        Index('idx_product_platform', 'platform'),
        Index('idx_product_type', 'product_type'),
    )
    
    def to_dict(self):
        """Convert product to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'price_numeric': self.price_numeric,
            'seller_name': self.seller_name,
            'rating_numeric': self.rating_numeric,
            'reviews_count': self.reviews_count,
            'platform': self.platform,
            'product_type': self.product_type,
            'scraped_at': self.scraped_at.isoformat() if self.scraped_at else None,
            'url': self.url,
            'raw_data': self.raw_data
        }
    
    def __repr__(self):
        return f'<Product {self.id}: {self.title[:50]}...>'


class Classification(db.Model):
    """Model for storing ML classification results."""
    
    __tablename__ = 'classifications'
    
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)  # "original" or "suspicious"
    confidence_score = db.Column(db.Float, nullable=False)
    feature_importance = db.Column(JSON)
    explanation = db.Column(JSON)
    model_version = db.Column(db.String(50))
    classified_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    processing_time_ms = db.Column(db.Integer)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_classification_product_id', 'product_id'),
        Index('idx_classification_prediction', 'prediction'),
        Index('idx_classification_confidence', 'confidence_score'),
        Index('idx_classification_classified_at', 'classified_at'),
    )
    
    def to_dict(self):
        """Convert classification to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'prediction': self.prediction,
            'confidence_score': self.confidence_score,
            'feature_importance': self.feature_importance,
            'explanation': self.explanation,
            'model_version': self.model_version,
            'classified_at': self.classified_at.isoformat() if self.classified_at else None,
            'processing_time_ms': self.processing_time_ms
        }
    
    def __repr__(self):
        return f'<Classification {self.id}: {self.prediction} ({self.confidence_score:.2f})>'


class ScrapingJob(db.Model):
    """Model for tracking scraping job execution history."""
    
    __tablename__ = 'scraping_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    job_type = db.Column(db.String(20), nullable=False)  # "scheduled" or "manual"
    status = db.Column(db.String(20), nullable=False, default='running')  # "running", "completed", "failed"
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime)
    products_found = db.Column(db.Integer, default=0)
    products_processed = db.Column(db.Integer, default=0)
    errors = db.Column(JSON)
    search_terms = db.Column(JSON)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_scraping_job_status', 'status'),
        Index('idx_scraping_job_type', 'job_type'),
        Index('idx_scraping_job_started_at', 'started_at'),
    )
    
    def to_dict(self):
        """Convert scraping job to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'job_type': self.job_type,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'products_found': self.products_found,
            'products_processed': self.products_processed,
            'errors': self.errors,
            'search_terms': self.search_terms
        }
    
    def mark_completed(self, products_found=0, products_processed=0, errors=None):
        """Mark job as completed with results."""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        self.products_found = products_found
        self.products_processed = products_processed
        if errors:
            self.errors = errors
    
    def mark_failed(self, errors):
        """Mark job as failed with error details."""
        self.status = 'failed'
        self.completed_at = datetime.utcnow()
        self.errors = errors
    
    def __repr__(self):
        return f'<ScrapingJob {self.id}: {self.job_type} - {self.status}>'


class SystemHealth(db.Model):
    """Model for storing system health and performance metrics."""
    
    __tablename__ = 'system_health'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    component = db.Column(db.String(50), nullable=False)  # "scraper", "ml_model", "database", "api"
    status = db.Column(db.String(20), nullable=False)  # "healthy", "degraded", "unhealthy"
    metrics = db.Column(JSON)
    message = db.Column(db.String(500))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_system_health_timestamp', 'timestamp'),
        Index('idx_system_health_component', 'component'),
        Index('idx_system_health_status', 'status'),
    )
    
    def to_dict(self):
        """Convert system health record to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'component': self.component,
            'status': self.status,
            'metrics': self.metrics,
            'message': self.message
        }
    
    def __repr__(self):
        return f'<SystemHealth {self.component}: {self.status}>'