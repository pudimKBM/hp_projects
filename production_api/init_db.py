#!/usr/bin/env python3
"""
Database initialization script for Production Classification API.

This script creates the database schema, applies migrations, and sets up
initial data for the production classification system.

Usage:
    python init_db.py [--reset] [--backup]
    
Options:
    --reset: Drop existing tables and recreate (WARNING: destroys data)
    --backup: Create backup before initialization
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from production_api.app.models import db, Product, Classification, ScrapingJob, SystemHealth
from production_api.app.utils.database import (
    init_database, migrate_database, backup_database, 
    validate_database_integrity
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('production_api/logs/db_init.log', mode='a')
        ]
    )


def create_app():
    """Create Flask application for database initialization."""
    app = Flask(__name__)
    
    # Database configuration
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'production.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database with app
    db.init_app(app)
    
    return app, db_path


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description='Initialize Production Classification API database')
    parser.add_argument('--reset', action='store_true', 
                       help='Drop existing tables and recreate (WARNING: destroys data)')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup before initialization')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting database initialization...")
    
    try:
        # Create Flask app and get database path
        app, db_path = create_app()
        
        with app.app_context():
            # Create backup if requested and database exists
            if args.backup and os.path.exists(db_path):
                logger.info("Creating database backup...")
                backup_path = backup_database(db_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Reset database if requested
            if args.reset:
                logger.warning("Resetting database - all data will be lost!")
                db.drop_all()
                logger.info("Existing tables dropped")
            
            # Initialize database
            logger.info("Initializing database schema...")
            init_database(app, db)
            
            # Run migrations
            logger.info("Running database migrations...")
            migrate_database(db)
            
            # Validate database integrity
            logger.info("Validating database integrity...")
            validation_results = validate_database_integrity(db)
            
            if validation_results['valid']:
                logger.info("Database validation passed")
            else:
                logger.warning(f"Database validation issues: {validation_results['issues']}")
            
            # Print statistics
            stats = validation_results['statistics']
            logger.info(f"Database statistics:")
            logger.info(f"  - Products: {stats.get('total_products', 0)}")
            logger.info(f"  - Classifications: {stats.get('total_classifications', 0)}")
            logger.info(f"  - Scraping Jobs: {stats.get('total_scraping_jobs', 0)}")
            
            logger.info("Database initialization completed successfully!")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()