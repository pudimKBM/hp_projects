#!/usr/bin/env python3
"""
Database migration script for Production Classification API.

This script handles database schema migrations and updates for the
production classification system.

Usage:
    python migrate_db.py [--dry-run] [--backup]
    
Options:
    --dry-run: Show what would be migrated without applying changes
    --backup: Create backup before migration
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from production_api.app.models import db
from production_api.app.utils.database import (
    migrate_database, backup_database, validate_database_integrity,
    get_schema_version
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('production_api/logs/db_migration.log', mode='a')
        ]
    )


def create_app():
    """Create Flask application for database migration."""
    app = Flask(__name__)
    
    # Database configuration
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'production.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database with app
    db.init_app(app)
    
    return app, db_path


def check_migration_needed(db):
    """
    Check if migration is needed.
    
    Returns:
        tuple: (needs_migration, current_version, target_version)
    """
    current_version = get_schema_version(db)
    target_version = 1  # Update this as we add more migrations
    
    return current_version < target_version, current_version, target_version


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate Production Classification API database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be migrated without applying changes')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup before migration')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting database migration check...")
    
    try:
        # Create Flask app and get database path
        app, db_path = create_app()
        
        with app.app_context():
            # Check if migration is needed
            needs_migration, current_version, target_version = check_migration_needed(db)
            
            logger.info(f"Current schema version: {current_version}")
            logger.info(f"Target schema version: {target_version}")
            
            if not needs_migration:
                logger.info("Database is up to date - no migration needed")
                return
            
            if args.dry_run:
                logger.info("DRY RUN: Would migrate database from version "
                           f"{current_version} to {target_version}")
                return
            
            # Create backup if requested
            if args.backup:
                logger.info("Creating database backup...")
                backup_path = backup_database(db_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Validate database before migration
            logger.info("Validating database before migration...")
            pre_validation = validate_database_integrity(db)
            
            if not pre_validation['valid']:
                logger.warning(f"Database validation issues found: {pre_validation['issues']}")
                response = input("Continue with migration? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Migration cancelled by user")
                    return
            
            # Run migration
            logger.info(f"Migrating database from version {current_version} to {target_version}...")
            migrate_database(db)
            
            # Validate database after migration
            logger.info("Validating database after migration...")
            post_validation = validate_database_integrity(db)
            
            if post_validation['valid']:
                logger.info("Database migration completed successfully!")
            else:
                logger.error(f"Database validation failed after migration: {post_validation['issues']}")
                sys.exit(1)
            
            # Print final statistics
            stats = post_validation['statistics']
            logger.info(f"Final database statistics:")
            logger.info(f"  - Products: {stats.get('total_products', 0)}")
            logger.info(f"  - Classifications: {stats.get('total_classifications', 0)}")
            logger.info(f"  - Scraping Jobs: {stats.get('total_scraping_jobs', 0)}")
            
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()