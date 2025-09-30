"""
Flask application package for HP Product Classification API
"""
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import logging
import os

# Initialize extensions
db = SQLAlchemy()

def create_app(config_name='default'):
    """
    Application factory pattern for creating Flask app
    """
    app = Flask(__name__)
    
    # Load configuration
    from production_api.config.config import get_config
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Initialize extensions with app
    db.init_app(app)
    
    # Configure logging
    setup_logging(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

def register_blueprints(app):
    """Register application blueprints"""
    from production_api.app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

def register_error_handlers(app):
    """Register global error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': 'The request could not be understood by the server',
            'status_code': 400
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': 'Method Not Allowed',
            'message': 'The method is not allowed for the requested URL',
            'status_code': 405
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An internal server error occurred',
            'status_code': 500
        }), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        return jsonify({
            'error': 'Service Unavailable',
            'message': 'The service is temporarily unavailable',
            'status_code': 503
        }), 503

def setup_logging(app):
    """Configure comprehensive application logging and performance monitoring"""
    from production_api.app.utils.logging import setup_logging as setup_structured_logging
    from production_api.app.services.performance_service import start_performance_monitoring
    
    # Setup structured logging
    setup_structured_logging(app)
    
    # Start performance monitoring (only in non-testing environments)
    if not app.testing:
        try:
            start_performance_monitoring()
            app.logger.info("Performance monitoring started")
        except Exception as e:
            app.logger.warning(f"Failed to start performance monitoring: {e}")
    
    # Setup cleanup on app teardown
    @app.teardown_appcontext
    def cleanup_performance_monitoring(error):
        """Cleanup performance monitoring on app teardown"""
        if error:
            app.logger.error(f"Application error during request: {error}")
    
    app.logger.info('Production API startup completed')