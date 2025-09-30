"""
API routes for HP Product Classification API

This module contains all REST API endpoints for the classification system.
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import logging
from functools import wraps

# Create blueprint
api_bp = Blueprint('api', __name__)

# Get logger
logger = logging.getLogger(__name__)


def validate_json_request(f):
    """Decorator to validate JSON request data"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request must be JSON',
                'status_code': 400
            }), 400
        return f(*args, **kwargs)
    return decorated_function


def handle_api_errors(f):
    """Decorator to handle common API errors"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Validation Error',
                'message': str(e),
                'status_code': 400
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'status_code': 500
            }), 500
    return decorated_function


@api_bp.route('/health', methods=['GET'])
@handle_api_errors
def health_check():
    """
    Health check endpoint
    GET /api/health
    
    Query parameters:
    - force_refresh: Force refresh of cached health data (default: false)
    - component: Check specific component only ('database', 'ml_model', 'scraper')
    """
    from production_api.app.services.health_service import HealthService, check_component_health
    
    # Get query parameters
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
    component = request.args.get('component')
    
    try:
        if component:
            # Check specific component
            if component not in ['database', 'ml_model', 'scraper']:
                return jsonify({
                    'error': 'Validation Error',
                    'message': 'Component must be one of: database, ml_model, scraper',
                    'status_code': 400
                }), 400
            
            health_data = check_component_health(component)
            return jsonify(health_data)
        else:
            # Get comprehensive system health
            health_service = HealthService()
            health_data = health_service.get_system_health(force_refresh)
            
            # Add API metadata
            health_data['api_info'] = {
                'version': current_app.config.get('API_VERSION', 'v1'),
                'service': current_app.config.get('API_TITLE', 'HP Product Classification API')
            }
            
            return jsonify(health_data)
            
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'error': 'Health check service unavailable',
            'message': str(e),
            'api_info': {
                'version': current_app.config.get('API_VERSION', 'v1'),
                'service': current_app.config.get('API_TITLE', 'HP Product Classification API')
            }
        }), 503


@api_bp.route('/classify', methods=['POST'])
@validate_json_request
@handle_api_errors
def classify_product():
    """
    Classify a single product for authenticity
    POST /api/classify
    
    Expected JSON payload:
    {
        "title": "Product title",
        "description": "Product description", 
        "price": 89.90,
        "seller_name": "Seller name",
        "rating": 4.8,
        "reviews_count": 150,
        "url": "https://example.com/product"
    }
    """
    # Import here to avoid circular imports
    from production_api.app.services.classification_service_wrapper import ClassificationService
    
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['title', 'price']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            'error': 'Validation Error',
            'message': f'Missing required fields: {", ".join(missing_fields)}',
            'status_code': 400
        }), 400
    
    # Validate data types
    try:
        # Ensure price is numeric
        if not isinstance(data.get('price'), (int, float)):
            raise ValueError("Price must be a number")
        
        # Ensure rating is numeric if provided
        if 'rating' in data and not isinstance(data['rating'], (int, float)):
            raise ValueError("Rating must be a number")
        
        # Ensure reviews_count is integer if provided
        if 'reviews_count' in data and not isinstance(data['reviews_count'], int):
            raise ValueError("Reviews count must be an integer")
            
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e),
            'status_code': 400
        }), 400
    
    # Get classification service
    classification_service = ClassificationService()
    
    try:
        # Perform classification
        result = classification_service.classify_single_product(data)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({
            'error': 'Service Error',
            'message': 'Classification service temporarily unavailable',
            'status_code': 503
        }), 503


@api_bp.route('/products', methods=['GET'])
@handle_api_errors
def list_products():
    """
    Get paginated list of classified products
    GET /api/products
    
    Query parameters:
    - page: Page number (default: 1)
    - limit: Items per page (default: 20, max: 100)
    - prediction: Filter by prediction ("original", "suspicious")
    - min_confidence: Minimum confidence score filter
    - date_from: Filter products scraped after date (ISO format)
    - date_to: Filter products scraped before date (ISO format)
    """
    from production_api.app.models import Product, Classification
    from production_api.app import db
    from sqlalchemy import and_, or_
    
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', current_app.config['DEFAULT_PAGE_SIZE'], type=int)
    prediction = request.args.get('prediction')
    min_confidence = request.args.get('min_confidence', type=float)
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    # Validate parameters
    if page < 1:
        return jsonify({
            'error': 'Validation Error',
            'message': 'Page number must be positive',
            'status_code': 400
        }), 400
    
    if limit < 1 or limit > current_app.config['MAX_PAGE_SIZE']:
        return jsonify({
            'error': 'Validation Error',
            'message': f'Limit must be between 1 and {current_app.config["MAX_PAGE_SIZE"]}',
            'status_code': 400
        }), 400
    
    if prediction and prediction not in ['original', 'suspicious']:
        return jsonify({
            'error': 'Validation Error',
            'message': 'Prediction must be "original" or "suspicious"',
            'status_code': 400
        }), 400
    
    if min_confidence and (min_confidence < 0 or min_confidence > 1):
        return jsonify({
            'error': 'Validation Error',
            'message': 'Confidence must be between 0 and 1',
            'status_code': 400
        }), 400
    
    # Build query
    query = db.session.query(Product, Classification).join(
        Classification, Product.id == Classification.product_id
    )
    
    # Apply filters
    filters = []
    
    if prediction:
        filters.append(Classification.prediction == prediction)
    
    if min_confidence:
        filters.append(Classification.confidence_score >= min_confidence)
    
    if date_from:
        try:
            date_from_obj = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            filters.append(Product.scraped_at >= date_from_obj)
        except ValueError:
            return jsonify({
                'error': 'Validation Error',
                'message': 'Invalid date_from format. Use ISO format (YYYY-MM-DDTHH:MM:SSZ)',
                'status_code': 400
            }), 400
    
    if date_to:
        try:
            date_to_obj = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            filters.append(Product.scraped_at <= date_to_obj)
        except ValueError:
            return jsonify({
                'error': 'Validation Error',
                'message': 'Invalid date_to format. Use ISO format (YYYY-MM-DDTHH:MM:SSZ)',
                'status_code': 400
            }), 400
    
    if filters:
        query = query.filter(and_(*filters))
    
    # Order by classification date (most recent first)
    query = query.order_by(Classification.classified_at.desc())
    
    # Paginate
    try:
        paginated = query.paginate(
            page=page,
            per_page=limit,
            error_out=False
        )
    except Exception as e:
        logger.error(f"Database error in list_products: {str(e)}")
        return jsonify({
            'error': 'Database Error',
            'message': 'Error retrieving products',
            'status_code': 500
        }), 500
    
    # Format response
    products = []
    for product, classification in paginated.items:
        products.append({
            'id': product.id,
            'title': product.title,
            'prediction': classification.prediction,
            'confidence_score': classification.confidence_score,
            'scraped_at': product.scraped_at.isoformat() + 'Z' if product.scraped_at else None,
            'classified_at': classification.classified_at.isoformat() + 'Z' if classification.classified_at else None
        })
    
    return jsonify({
        'products': products,
        'pagination': {
            'page': page,
            'limit': limit,
            'total_items': paginated.total,
            'total_pages': paginated.pages,
            'has_next': paginated.has_next,
            'has_prev': paginated.has_prev
        }
    })


@api_bp.route('/products/<int:product_id>', methods=['GET'])
@handle_api_errors
def get_product_details(product_id):
    """
    Get detailed information for a specific product
    GET /api/products/{id}
    """
    from production_api.app.models import Product, Classification
    from production_api.app import db
    
    # Query product with classification
    result = db.session.query(Product, Classification).join(
        Classification, Product.id == Classification.product_id
    ).filter(Product.id == product_id).first()
    
    if not result:
        return jsonify({
            'error': 'Not Found',
            'message': f'Product with id {product_id} not found',
            'status_code': 404
        }), 404
    
    product, classification = result
    
    # Format detailed response
    response = {
        'product': {
            'id': product.id,
            'title': product.title,
            'description': product.description,
            'price_numeric': product.price_numeric,
            'seller_name': product.seller_name,
            'rating_numeric': product.rating_numeric,
            'reviews_count': product.reviews_count,
            'platform': product.platform,
            'url': product.url,
            'scraped_at': product.scraped_at.isoformat() + 'Z' if product.scraped_at else None
        },
        'classification': {
            'prediction': classification.prediction,
            'confidence_score': classification.confidence_score,
            'feature_importance': classification.feature_importance,
            'explanation': classification.explanation,
            'model_version': classification.model_version,
            'classified_at': classification.classified_at.isoformat() + 'Z' if classification.classified_at else None,
            'processing_time_ms': classification.processing_time_ms
        }
    }
    
    return jsonify(response)


# Error handlers specific to API blueprint
@api_bp.errorhandler(404)
def api_not_found(error):
    """Handle 404 errors in API routes"""
    return jsonify({
        'error': 'Not Found',
        'message': 'API endpoint not found',
        'status_code': 404
    }), 404


@api_bp.route('/metrics', methods=['GET'])
@handle_api_errors
def get_performance_metrics():
    """
    Get performance metrics and statistics
    GET /api/metrics
    
    Query parameters:
    - format: Output format ('json' or 'prometheus', default: 'json')
    - history_hours: Hours of history to include (default: 24)
    """
    from production_api.app.services.performance_service import get_performance_service
    
    # Get query parameters
    output_format = request.args.get('format', 'json').lower()
    history_hours = request.args.get('history_hours', 24, type=int)
    
    # Validate parameters
    if output_format not in ['json', 'prometheus']:
        return jsonify({
            'error': 'Validation Error',
            'message': 'Format must be "json" or "prometheus"',
            'status_code': 400
        }), 400
    
    if history_hours < 1 or history_hours > 168:  # Max 1 week
        return jsonify({
            'error': 'Validation Error',
            'message': 'History hours must be between 1 and 168',
            'status_code': 400
        }), 400
    
    try:
        perf_service = get_performance_service()
        
        if output_format == 'prometheus':
            # Export in Prometheus format
            metrics_data = perf_service.export_performance_metrics('prometheus')
            
            # Return as plain text for Prometheus
            from flask import Response
            prometheus_text = '\n'.join([
                f'{key} {value}' for key, value in metrics_data.items()
                if isinstance(value, (int, float))
            ])
            
            return Response(prometheus_text, mimetype='text/plain')
        else:
            # Return JSON format
            performance_summary = perf_service.get_performance_summary()
            performance_history = perf_service.get_performance_history(history_hours)
            
            return jsonify({
                'current_metrics': performance_summary,
                'historical_data': performance_history,
                'metadata': {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'history_hours': history_hours,
                    'format': output_format
                }
            })
            
    except Exception as e:
        logger.error(f"Performance metrics error: {str(e)}")
        return jsonify({
            'error': 'Service Error',
            'message': 'Performance metrics service temporarily unavailable',
            'status_code': 503
        }), 503


@api_bp.errorhandler(405)
def api_method_not_allowed(error):
    """Handle 405 errors in API routes"""
    return jsonify({
        'error': 'Method Not Allowed',
        'message': 'HTTP method not allowed for this endpoint',
        'status_code': 405
    }), 405