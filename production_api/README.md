# Production Classification API

A production-ready system that integrates HP product scraping with ML classification to provide automated product authenticity detection.

## Project Structure

```
production_api/
├── app/                    # Flask application
│   ├── api/               # API routes and endpoints
│   ├── services/          # Business logic services
│   ├── utils/             # Utility functions
│   └── models.py          # Database models
├── config/                # Configuration files
│   └── config.py         # Application configuration
├── data/                  # Database files
├── logs/                  # Application logs
├── init_db.py            # Database initialization script
├── migrate_db.py         # Database migration script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Database Models

### Product
Stores raw product data from web scraping:
- Product details (title, description, price, seller)
- Metadata (scraped timestamp, URL, platform)
- Raw scraped data in JSON format

### Classification
Stores ML classification results:
- Prediction (original/suspicious) with confidence score
- Feature importance and explanations
- Model version and processing metrics

### ScrapingJob
Tracks scraping job execution:
- Job status and timing information
- Products found and processed counts
- Error tracking and search parameters

### SystemHealth
Monitors system component health:
- Component status (scraper, ML model, database, API)
- Performance metrics and health messages
- Timestamp tracking for monitoring

## Database Setup

### Initialize Database
```bash
# Create new database with schema
python init_db.py

# Reset database (WARNING: destroys data)
python init_db.py --reset

# Create backup before initialization
python init_db.py --backup
```

### Migrate Database
```bash
# Check and apply migrations
python migrate_db.py

# Dry run to see what would be migrated
python migrate_db.py --dry-run

# Create backup before migration
python migrate_db.py --backup
```

## Configuration

The system supports multiple environments:
- **Development**: Debug enabled, conservative scraping settings
- **Production**: Optimized for performance and security
- **Testing**: In-memory database, fast execution

Set environment with `FLASK_ENV` variable:
```bash
export FLASK_ENV=production  # or development, testing
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Next Steps

After completing this task, the following components need to be implemented:
1. ML classification service integration
2. Scraper service wrapper
3. REST API endpoints
4. Health monitoring system
5. Configuration management
6. Testing suite

See the main tasks.md file for the complete implementation plan.