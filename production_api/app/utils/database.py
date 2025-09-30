"""
Database utilities for initialization, migration, and management.

This module provides functions for:
- Database initialization
- Schema creation and migration
- Database connection management
- Data validation and cleanup
"""

import os
import sqlite3
from datetime import datetime
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


def init_database(app, db):
    """
    Initialize the database with the Flask app.
    
    Args:
        app: Flask application instance
        db: SQLAlchemy database instance
    """
    with app.app_context():
        # Create all tables
        db.create_all()
        logger.info("Database tables created successfully")
        
        # Create indexes if they don't exist
        create_indexes(db)
        
        # Insert initial data if needed
        insert_initial_data(db)
        
        logger.info("Database initialization completed")


def create_indexes(db):
    """Create additional indexes for performance optimization."""
    try:
        # Additional composite indexes for common queries
        with db.engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_product_scraped_platform 
                ON products(scraped_at, platform)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_classification_product_prediction 
                ON classifications(product_id, prediction)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_classification_confidence_prediction 
                ON classifications(confidence_score, prediction)
            """))
            conn.commit()
        
        logger.info("Additional database indexes created")
        
    except Exception as e:
        logger.warning(f"Error creating additional indexes: {e}")


def insert_initial_data(db):
    """Insert any required initial data."""
    try:
        # Check if we need to insert initial system health record
        from production_api.app.models import SystemHealth
        
        if SystemHealth.query.count() == 0:
            initial_health = SystemHealth(
                component='system',
                status='healthy',
                metrics={'initialized_at': datetime.utcnow().isoformat()},
                message='System initialized successfully'
            )
            db.session.add(initial_health)
            db.session.commit()
            logger.info("Initial system health record created")
            
    except Exception as e:
        logger.warning(f"Error inserting initial data: {e}")
        db.session.rollback()


def migrate_database(db):
    """
    Perform database migrations for schema updates.
    
    Args:
        db: SQLAlchemy database instance
    """
    try:
        # Check current schema version
        schema_version = get_schema_version(db)
        logger.info(f"Current schema version: {schema_version}")
        
        # Apply migrations based on version
        if schema_version < 1:
            migrate_to_v1(db)
        
        # Add more migration steps as needed
        # if schema_version < 2:
        #     migrate_to_v2(db)
        
        logger.info("Database migration completed")
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise


def get_schema_version(db):
    """Get the current schema version from the database."""
    try:
        # Try to get version from a metadata table
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT version FROM schema_version ORDER BY id DESC LIMIT 1
            """))
            row = result.fetchone()
            return row[0] if row else 0
        
    except Exception:
        # If table doesn't exist, we're at version 0
        return 0


def migrate_to_v1(db):
    """Migrate database to version 1."""
    try:
        # Create schema_version table if it doesn't exist
        with db.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """))
            
            # Insert version 1 record
            conn.execute(text("""
                INSERT INTO schema_version (version, description) 
                VALUES (1, 'Initial schema with products, classifications, scraping_jobs, and system_health tables')
            """))
            conn.commit()
        
        logger.info("Migrated to schema version 1")
        
    except Exception as e:
        logger.error(f"Migration to v1 failed: {e}")
        raise


def backup_database(db_path, backup_dir=None):
    """
    Create a backup of the SQLite database.
    
    Args:
        db_path: Path to the database file
        backup_dir: Directory to store backups (default: same as db_path)
    
    Returns:
        str: Path to the backup file
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    if backup_dir is None:
        backup_dir = os.path.dirname(db_path)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    backup_filename = f"{db_name}_backup_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Create backup using SQLite backup API
    try:
        source = sqlite3.connect(db_path)
        backup = sqlite3.connect(backup_path)
        source.backup(backup)
        backup.close()
        source.close()
        
        logger.info(f"Database backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise


def validate_database_integrity(db):
    """
    Validate database integrity and consistency.
    
    Args:
        db: SQLAlchemy database instance
    
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    try:
        from production_api.app.models import Product, Classification, ScrapingJob
        
        # Check for orphaned classifications
        orphaned_classifications = db.session.query(Classification).filter(
            ~Classification.product_id.in_(
                db.session.query(Product.id)
            )
        ).count()
        
        if orphaned_classifications > 0:
            results['valid'] = False
            results['issues'].append(f"Found {orphaned_classifications} orphaned classifications")
        
        # Check for products without classifications
        unclassified_products = db.session.query(Product).filter(
            ~Product.id.in_(
                db.session.query(Classification.product_id)
            )
        ).count()
        
        # Collect statistics
        results['statistics'] = {
            'total_products': Product.query.count(),
            'total_classifications': Classification.query.count(),
            'total_scraping_jobs': ScrapingJob.query.count(),
            'orphaned_classifications': orphaned_classifications,
            'unclassified_products': unclassified_products
        }
        
        logger.info(f"Database validation completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        results['valid'] = False
        results['issues'].append(f"Validation error: {e}")
        return results


def cleanup_old_data(db, days_to_keep=30):
    """
    Clean up old data from the database.
    
    Args:
        db: SQLAlchemy database instance
        days_to_keep: Number of days of data to retain
    
    Returns:
        dict: Cleanup results
    """
    from datetime import timedelta
    from production_api.app.models import SystemHealth, ScrapingJob
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    results = {'deleted_records': 0, 'errors': []}
    
    try:
        # Clean up old system health records
        old_health_records = SystemHealth.query.filter(
            SystemHealth.timestamp < cutoff_date
        ).count()
        
        SystemHealth.query.filter(
            SystemHealth.timestamp < cutoff_date
        ).delete()
        
        # Clean up old completed scraping jobs
        old_jobs = ScrapingJob.query.filter(
            ScrapingJob.completed_at < cutoff_date,
            ScrapingJob.status.in_(['completed', 'failed'])
        ).count()
        
        ScrapingJob.query.filter(
            ScrapingJob.completed_at < cutoff_date,
            ScrapingJob.status.in_(['completed', 'failed'])
        ).delete()
        
        db.session.commit()
        
        results['deleted_records'] = old_health_records + old_jobs
        logger.info(f"Cleaned up {results['deleted_records']} old records")
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        results['errors'].append(str(e))
        db.session.rollback()
    
    return results