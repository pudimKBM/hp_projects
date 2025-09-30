# HP Product Classification API - Deployment Guide

This guide covers the deployment and operation of the HP Product Classification API.

## Entry Points and Scripts

### 1. API Server (`run_api.py`)

Starts the Flask API server with proper configuration.

```bash
# Start development server
python run_api.py

# Start with specific environment
FLASK_ENV=production python run_api.py

# Start with custom host/port
FLASK_HOST=0.0.0.0 FLASK_PORT=8000 python run_api.py
```

### 2. Manual Scraper (`run_scraper.py`)

Runs manual scraping jobs with classification (requires Flask app context).

```bash
# Run scraping with default settings
python run_scraper.py run

# Run with custom search terms
python run_scraper.py run --terms "cartucho hp" "toner hp"

# Run without classification
python run_scraper.py run --no-classify

# List recent jobs
python run_scraper.py list

# Show scraper status
python run_scraper.py status
```

### 3. Standalone Scraper (`run_scraper_standalone.py`)

Lightweight scraper for cron jobs (no Flask dependencies).

```bash
# Run standalone scraping
python run_scraper_standalone.py

# Custom configuration
python run_scraper_standalone.py --terms "cartucho hp" --max-pages 3 --delay 1.5

# With logging
python run_scraper_standalone.py --log-file logs/scraper.log --verbose
```

### 4. Cron Setup (`setup_cron.py`)

Configures automated scheduling for scraping jobs.

```bash
# Install cron job with default schedule
python setup_cron.py install

# Install with custom schedule (every 4 hours)
python setup_cron.py install --schedule "0 */4 * * *"

# Install for development environment
python setup_cron.py install --environment development

# Remove cron jobs
python setup_cron.py remove

# List existing jobs
python setup_cron.py list

# Show status and configuration
python setup_cron.py status
```

### 5. Deployment Script (`deploy.py`)

Complete deployment automation.

```bash
# Deploy development environment
python deploy.py development

# Deploy production (with dependency installation and cron setup)
python deploy.py production

# Deploy without dependencies or cron
python deploy.py production --no-deps --no-cron

# Validate configuration only
python deploy.py production --validate-only

# Deploy with custom cron schedule
python deploy.py production --schedule "0 */8 * * *"
```

### 6. Configuration Validation (`config/validate_config.py`)

Validates configuration settings for all environments.

```bash
# Validate all environments
python config/validate_config.py

# Validate specific environment
python config/validate_config.py --environment production

# Show detailed output
python config/validate_config.py --verbose

# Check configuration files and summary
python config/validate_config.py --check-files --summary
```

## Quick Start Deployment

### Development Environment

```bash
# 1. Validate configuration
python config/validate_config.py --environment development

# 2. Deploy development environment
python deploy.py development

# 3. Start API server
python run_api.py

# 4. Test manual scraping
python run_scraper.py run --terms "cartucho hp" --max-pages 1
```

### Production Environment

```bash
# 1. Set production secret key
export SECRET_KEY="your-secure-secret-key-here"

# 2. Validate production configuration
python deploy.py production --validate-only

# 3. Deploy production environment
python deploy.py production

# 4. Start API server
FLASK_ENV=production python run_api.py

# 5. Verify cron jobs are installed
python setup_cron.py status
```

## Environment Configuration

### Environment Files

- `.env.development` - Development settings
- `.env.production` - Production settings  
- `.env.testing` - Testing settings
- `.env.example` - Template with all options

### Key Configuration Variables

```bash
# Required for production
SECRET_KEY=your-secure-secret-key

# Database
DATABASE_URL=sqlite:///data/production.db

# API settings
API_PORT=5000
API_DEBUG=false

# Scraper settings
SCRAPER_DELAY=2.0
MAX_PAGES_PER_TERM=5
SEARCH_TERMS=cartucho hp original,toner hp,cartucho hp 664

# Cron schedule (every 6 hours)
CRON_SCHEDULE=0 */6 * * *

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/production.log
```

## Monitoring and Maintenance

### Log Files

- `logs/production_api.log` - API server logs
- `logs/cron_scraper.log` - Automated scraping logs
- `logs/development.log` - Development logs

### Health Checks

```bash
# Check API health
curl http://localhost:5000/api/health

# Check scraper status
python run_scraper.py status

# Check cron job status
python setup_cron.py status
```

### Database Management

```bash
# Initialize database
python init_db.py

# Run migrations
python migrate_db.py

# Backup database (SQLite)
cp data/production.db data/backup_$(date +%Y%m%d_%H%M%S).db
```

## Troubleshooting

### Common Issues

1. **Flask/Werkzeug compatibility errors**
   - Use `run_scraper_standalone.py` for cron jobs
   - Check requirements.txt versions

2. **Configuration validation failures**
   - Run `python config/validate_config.py --verbose`
   - Check environment variables are set

3. **Cron jobs not running**
   - Check `python setup_cron.py list`
   - Verify log files for errors
   - Test standalone scraper manually

4. **Database connection errors**
   - Check DATABASE_URL configuration
   - Ensure database file/directory exists
   - Run `python init_db.py` to initialize

5. **Model loading errors**
   - Verify models exist in models/ directory
   - Check MODEL_PATH configuration
   - Ensure src/ directory is accessible

### Debug Mode

Enable verbose logging and debug mode:

```bash
# Set debug environment variables
export FLASK_ENV=development
export LOG_LEVEL=DEBUG

# Run with verbose output
python run_scraper.py run --verbose
python setup_cron.py status --verbose
python deploy.py development --validate-only
```

## Security Considerations

### Production Deployment

1. **Set secure SECRET_KEY**
   ```bash
   export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   ```

2. **Use proper database**
   - PostgreSQL or MySQL for production
   - Not SQLite for high-traffic deployments

3. **Configure CORS properly**
   ```bash
   export CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
   ```

4. **Enable rate limiting**
   ```bash
   export RATE_LIMIT_ENABLED=true
   export RATE_LIMIT_PER_MINUTE=60
   ```

5. **Set up monitoring**
   ```bash
   export SENTRY_DSN=your-sentry-dsn
   export METRICS_ENABLED=true
   ```

This deployment guide provides comprehensive instructions for setting up and operating the HP Product Classification API in various environments.