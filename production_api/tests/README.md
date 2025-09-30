# Production API Test Suite

This directory contains comprehensive unit tests for the HP Product Classification API services.

## Test Files

### Core Service Tests

1. **test_ml_service.py** - Tests for ML Service
   - Model loading and validation
   - Fallback model setup
   - Health checking
   - Model metadata management
   - Service status monitoring
   - Error handling and recovery

2. **test_scraper_service.py** - Tests for Scraper Service
   - Scraping job creation and execution
   - Product data cleaning and validation
   - Database integration
   - Duplicate detection
   - Error handling and retry logic
   - Performance tracking integration
   - Edge cases and data extraction

3. **test_classification_service.py** - Tests for Classification Service
   - Product classification with confidence scores
   - Batch processing
   - Explanation generation
   - Business-friendly interpretations
   - Statistics tracking
   - Integration with ML and feature services

4. **test_api_routes.py** - Tests for API Endpoints
   - Health check endpoint
   - Product classification endpoint
   - Product listing with pagination and filters
   - Product detail retrieval
   - Performance metrics endpoint
   - Error handling and validation

### Test Configuration

- **conftest.py** - Pytest configuration and fixtures
  - Flask app setup for testing
  - Database fixtures with test data
  - Mock objects for external services
  - Helper functions for test data creation

## Test Coverage

The test suite covers:

### ML Service (test_ml_service.py)
- ✅ Model loading success and failure scenarios
- ✅ Fallback model configuration
- ✅ Model health validation
- ✅ Service status monitoring
- ✅ Model reloading
- ✅ Metadata extraction
- ✅ Integration workflows

### Scraper Service (test_scraper_service.py)
- ✅ Job creation and execution
- ✅ Product data extraction and cleaning
- ✅ Price, rating, and review count parsing
- ✅ Product validation and duplicate detection
- ✅ Database storage with batch commits
- ✅ Error handling and retry mechanisms
- ✅ Performance tracking integration
- ✅ Edge cases and data validation
- ✅ Scraper initialization and cleanup

### Classification Service (test_classification_service.py)
- ✅ Single product classification
- ✅ Batch classification processing
- ✅ Confidence score calculation
- ✅ Explanation generation (technical and business)
- ✅ Feature importance analysis
- ✅ Statistics tracking and reporting
- ✅ Integration with interpretation pipeline

### API Routes (test_api_routes.py)
- ✅ Health check endpoint with component filtering
- ✅ Product classification with validation
- ✅ Product listing with pagination and filters
- ✅ Product detail retrieval
- ✅ Performance metrics in JSON and Prometheus formats
- ✅ Error handling and HTTP status codes
- ✅ Request validation and parameter checking

## Key Test Features

### Mock Integration
- Comprehensive mocking of external dependencies
- Database session mocking for isolated tests
- ML model and scraper mocking
- Performance service integration mocking

### Edge Case Coverage
- Invalid data handling
- Network failures and timeouts
- Database errors and rollbacks
- Service unavailability scenarios
- Data validation edge cases

### Performance Testing Preparation
- Statistics tracking validation
- Batch processing verification
- Resource cleanup testing
- Error recovery mechanisms

## Running Tests

Due to Flask/Werkzeug compatibility issues in the current environment, tests are designed to be run with:

```bash
# Install compatible versions first
pip install flask==2.3.3 werkzeug==2.3.7

# Run all tests
pytest production_api/tests/ -v

# Run specific test file
pytest production_api/tests/test_scraper_service.py -v

# Run with coverage
pytest production_api/tests/ --cov=production_api/app/services
```

## Test Data

Tests use realistic HP product data including:
- Cartucho HP 664 Original products
- Various price formats (R$ 89,90, R$ 1.234,56)
- Brazilian Portuguese product descriptions
- MercadoLivre platform integration
- Seller reputation and review data

## Assertions and Validations

Each test includes comprehensive assertions for:
- Return value validation
- State change verification
- Mock call verification
- Error condition handling
- Data integrity checks
- Performance metrics tracking

## Integration Tests (Task 7.2)

### Integration Test Files

1. **test_integration.py** - End-to-End Workflow Tests
   - Complete scraping → classification → storage workflow
   - Workflow error handling and recovery
   - System health monitoring during operations
   - Cross-component integration validation

2. **test_database_integration.py** - Database Integration Tests
   - Transaction integrity and rollback scenarios
   - Concurrent database operations
   - Bulk insert performance testing
   - Query performance with proper indexing
   - Database migration and constraint enforcement
   - Backup and restore simulation

3. **test_api_integration.py** - API Integration Tests
   - Complete API request-response cycles with database persistence
   - Complex filtering and pagination with real data
   - Concurrent API request handling
   - API error handling with database rollback
   - Real system status monitoring via API

4. **test_integration_runner.py** - Integration Test Runner
   - Environment validation and setup verification
   - Comprehensive test execution with detailed reporting
   - Manual test validation when pytest unavailable
   - Test result aggregation and analysis

### Integration Test Coverage

#### End-to-End Workflows (Requirements: 1.2, 1.3)
- ✅ Complete scraping → classification → storage pipeline
- ✅ Batch processing with ML classification integration
- ✅ Error handling and recovery across components
- ✅ System health monitoring during operations
- ✅ Concurrent operation safety and data integrity

#### Database Integration (Requirements: 1.2, 1.3, 2.1)
- ✅ Transaction atomicity and rollback on errors
- ✅ Concurrent database operations with consistency
- ✅ Bulk insert performance for large datasets
- ✅ Query performance with proper indexing
- ✅ Database constraint enforcement
- ✅ Migration and schema validation
- ✅ Backup and restore procedures

#### API Integration (Requirements: 2.1)
- ✅ Classification endpoint with complete database persistence
- ✅ Product listing with complex filtering and pagination
- ✅ Product detail retrieval with full relationship data
- ✅ Health monitoring with real system component status
- ✅ Concurrent API request handling and data integrity
- ✅ Error handling with proper HTTP responses and database rollback

#### Cron Job Integration (Requirements: 1.2)
- ✅ Scheduled scraping job execution
- ✅ Job failure handling and retry mechanisms
- ✅ Timeout handling for long-running jobs
- ✅ Concurrent job execution management

### Running Integration Tests

```bash
# Run all integration tests with pytest
pytest production_api/tests/test_integration*.py -v

# Run specific integration test file
pytest production_api/tests/test_integration.py -v

# Run with coverage for integration tests
pytest production_api/tests/test_integration*.py --cov=production_api/app

# Use the integration test runner
python production_api/tests/test_integration_runner.py
```

### Integration Test Features

#### Real Database Operations
- Tests use actual database transactions and rollbacks
- Verification of data persistence and relationships
- Concurrent operation testing with real database locks
- Performance testing with realistic data volumes

#### Complete Workflow Testing
- End-to-end pipeline validation from scraping to storage
- Cross-service integration with real ML models and feature engineering
- Error propagation and recovery across system boundaries
- System health monitoring integration

#### API Integration Validation
- Real HTTP request-response cycles with database persistence
- Complex query scenarios with filtering and pagination
- Concurrent request handling with data integrity verification
- Error handling with proper HTTP status codes and database consistency

#### Environment Validation
- Automatic environment setup verification
- Dependency checking and validation
- Test data creation and cleanup
- Performance benchmarking and reporting

## Future Enhancements

The test suite is designed to support:
- Performance and load testing (task 7.3)
- Continuous integration pipelines
- Test data factories for complex scenarios
- Property-based testing for edge cases
- Automated performance regression detection