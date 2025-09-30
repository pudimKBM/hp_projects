#!/usr/bin/env python3
"""
Entry point for running the HP Product Classification API server.

This script creates and runs the Flask application with proper configuration
based on the environment.
"""

import os
import sys
from flask import Flask

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add parent directory to path for src imports
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from production_api.app import create_app, db


def main():
    """Main entry point for the API server"""
    
    # Get environment configuration
    config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Create Flask application
    app = create_app(config_name)
    
    # Create database tables if they don't exist
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {str(e)}")
            sys.exit(1)
    
    # Get host and port from environment or config
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = app.config.get('DEBUG', False)
    
    app.logger.info(f"Starting HP Product Classification API on {host}:{port}")
    app.logger.info(f"Environment: {config_name}")
    app.logger.info(f"Debug mode: {debug}")
    
    # Run the application
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        app.logger.info("API server stopped by user")
    except Exception as e:
        app.logger.error(f"Error starting API server: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()