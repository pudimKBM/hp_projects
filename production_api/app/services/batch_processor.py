"""
Batch Classification Processing Service

This service handles batch processing of newly scraped products for classification.
It integrates with the ML classification service and provides progress tracking
and performance monitoring for batch jobs.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import queue

from flask import current_app
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_

from ..models import db, Product, Classification, ScrapingJob, SystemHealth
from .classification_service import ClassificationEngine
from .ml_service import MLService
from .feature_service import FeaturePreparationService


logger = logging.getLogger(__name__)


class BatchJobStatus(Enum):
    """Enumeration for batch job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    batch_size: int = 50
    max_concurrent_batches: int = 2
    retry_attempts: int = 3
    timeout_minutes: int = 30
    confidence_threshold: float = 0.7
    include_explanations: bool = False
    auto_process_new_products: bool = True


class BatchProcessingJob:
    """Represents a batch processing job."""
    
    def __init__(self, job_id: str, product_ids: List[int], config: BatchProcessingConfig):
        """Initialize batch processing job."""
        self.job_id = job_id
        self.product_ids = product_ids
        self.config = config
        
        # Job status
        self.status = BatchJobStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Progress tracking
        self.total_products = len(product_ids)
        self.processed_products = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        
        # Results and errors
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        
        # Performance metrics
        self.processing_time_ms = 0
        self.avg_time_per_product_ms = 0


class BatchProcessor:
    """Service for batch processing of product classifications."""
    
    def __init__(self):
        """Initialize the batch processor."""
        self.ml_service = None
        self.feature_service = None
        self.classification_engine = None
        
        # Job management
        self.active_jobs: Dict[str, BatchProcessingJob] = {}
        self.job_queue = queue.Queue()
        self.worker_threads: List[threading.Thread] = []
        self.is_running = False
        
        # Configuration
        self.config = BatchProcessingConfig()
        
        # Performance tracking
        self.performance_metrics = {
            'total_batches_processed': 0,
            'total_products_classified': 0,
            'avg_batch_processing_time_ms': 0,
            'success_rate': 0.0,
            'last_batch_completed': None
        }
        
        self._lock = threading.Lock()
    
    def initialize_services(self, models_path: str = "models") -> bool:
        """
        Initialize ML services required for batch processing.
        
        Args:
            models_path: Path to ML models directory
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize ML service
            self.ml_service = MLService(models_path)
            
            # Load primary model (assuming best model is available)
            model_load_result = self.ml_service.load_primary_model("best_model")
            if not model_load_result['success']:
                logger.error(f"Failed to load primary model: {model_load_result.get('error')}")
                return False
            
            # Initialize feature service
            self.feature_service = FeaturePreparationService()
            
            # Initialize classification engine
            self.classification_engine = ClassificationEngine(
                ml_service=self.ml_service,
                feature_service=self.feature_service,
                confidence_threshold=self.config.confidence_threshold
            )
            
            # Setup interpretation pipeline
            active_model = self.ml_service.get_active_model("best_model")
            if active_model and active_model.get('metadata'):
                feature_names = active_model['metadata'].get('feature_names', [])
                if feature_names:
                    self.classification_engine.setup_interpretation_pipeline(feature_names)
            
            logger.info("Batch processor services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize batch processor services: {e}")
            return False
    
    def start_processing(self, max_workers: int = 2):
        """
        Start batch processing workers.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        if self.is_running:
            logger.warning("Batch processor is already running")
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(f"worker_{i}",),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started batch processor with {max_workers} workers")
    
    def stop_processing(self):
        """Stop batch processing workers."""
        self.is_running = False
        
        # Cancel active jobs
        with self._lock:
            for job in self.active_jobs.values():
                if job.status == BatchJobStatus.RUNNING:
                    job.status = BatchJobStatus.CANCELLED
                    job.completed_at = datetime.utcnow()
        
        logger.info("Batch processor stopped")
    
    def process_new_products(self, scraping_job_id: Optional[int] = None) -> str:
        """
        Process newly scraped products that haven't been classified yet.
        
        Args:
            scraping_job_id: Optional specific scraping job ID to process
            
        Returns:
            str: Job ID for the batch processing job
        """
        try:
            # Find unclassified products
            query = db.session.query(Product).outerjoin(Classification).filter(
                Classification.id.is_(None)
            )
            
            # Filter by scraping job if specified
            if scraping_job_id:
                query = query.filter(Product.scraped_at >= (
                    db.session.query(ScrapingJob.started_at)
                    .filter(ScrapingJob.id == scraping_job_id)
                    .scalar()
                ))
            
            unclassified_products = query.all()
            
            if not unclassified_products:
                logger.info("No unclassified products found")
                return None
            
            product_ids = [p.id for p in unclassified_products]
            
            # Create batch job
            job_id = f"batch_{int(time.time())}_{len(product_ids)}"
            return self.create_batch_job(job_id, product_ids)
            
        except Exception as e:
            logger.error(f"Error processing new products: {e}")
            raise
    
    def create_batch_job(self, job_id: str, product_ids: List[int], 
                        config: Optional[BatchProcessingConfig] = None) -> str:
        """
        Create a new batch processing job.
        
        Args:
            job_id: Unique identifier for the job
            product_ids: List of product IDs to process
            config: Optional custom configuration
            
        Returns:
            str: Job ID
        """
        try:
            if not product_ids:
                raise ValueError("No product IDs provided")
            
            # Use default config if not provided
            if config is None:
                config = self.config
            
            # Create job
            job = BatchProcessingJob(job_id, product_ids, config)
            
            with self._lock:
                self.active_jobs[job_id] = job
            
            # Add to processing queue
            self.job_queue.put(job_id)
            
            logger.info(f"Created batch job {job_id} with {len(product_ids)} products")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating batch job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Optional[Dict]: Job status information or None if not found
        """
        with self._lock:
            job = self.active_jobs.get(job_id)
            
            if not job:
                return None
            
            return {
                'job_id': job_id,
                'status': job.status.value,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'total_products': job.total_products,
                'processed_products': job.processed_products,
                'successful_classifications': job.successful_classifications,
                'failed_classifications': job.failed_classifications,
                'progress_percentage': (job.processed_products / job.total_products * 100) if job.total_products > 0 else 0,
                'processing_time_ms': job.processing_time_ms,
                'avg_time_per_product_ms': job.avg_time_per_product_ms,
                'errors': job.errors,
                'config': {
                    'batch_size': job.config.batch_size,
                    'confidence_threshold': job.config.confidence_threshold,
                    'include_explanations': job.config.include_explanations
                }
            }
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """Get status for all active jobs."""
        with self._lock:
            return [
                self.get_job_status(job_id)
                for job_id in self.active_jobs.keys()
            ]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a batch processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            bool: True if job was cancelled successfully
        """
        with self._lock:
            job = self.active_jobs.get(job_id)
            
            if not job:
                return False
            
            if job.status == BatchJobStatus.RUNNING:
                job.status = BatchJobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                logger.info(f"Cancelled batch job {job_id}")
                return True
            
            return False
    
    def _worker_loop(self, worker_name: str):
        """
        Main worker loop for processing batch jobs.
        
        Args:
            worker_name: Name of the worker thread
        """
        logger.info(f"Batch worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get job from queue (with timeout)
                job_id = self.job_queue.get(timeout=5)
                
                with self._lock:
                    job = self.active_jobs.get(job_id)
                
                if not job or job.status != BatchJobStatus.PENDING:
                    continue
                
                # Process the job
                self._process_batch_job(job, worker_name)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                time.sleep(1)
        
        logger.info(f"Batch worker {worker_name} stopped")
    
    def _process_batch_job(self, job: BatchProcessingJob, worker_name: str):
        """
        Process a single batch job.
        
        Args:
            job: Batch processing job
            worker_name: Name of the worker processing the job
        """
        job.status = BatchJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        start_time = time.time()
        
        logger.info(f"Worker {worker_name} processing job {job.job_id} with {job.total_products} products")
        
        try:
            # Process products in batches
            batch_size = job.config.batch_size
            
            for i in range(0, len(job.product_ids), batch_size):
                # Check if job was cancelled
                if job.status == BatchJobStatus.CANCELLED:
                    logger.info(f"Job {job.job_id} was cancelled")
                    return
                
                batch_product_ids = job.product_ids[i:i + batch_size]
                
                # Process batch
                batch_results = self._process_product_batch(
                    batch_product_ids, job.config, job.job_id
                )
                
                # Update job progress
                job.processed_products += len(batch_product_ids)
                job.successful_classifications += batch_results['successful_count']
                job.failed_classifications += batch_results['failed_count']
                job.results.extend(batch_results['results'])
                job.errors.extend(batch_results['errors'])
                
                # Log progress
                progress = (job.processed_products / job.total_products) * 100
                logger.info(f"Job {job.job_id} progress: {progress:.1f}% ({job.processed_products}/{job.total_products})")
            
            # Mark job as completed
            job.status = BatchJobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.processing_time_ms = int((time.time() - start_time) * 1000)
            job.avg_time_per_product_ms = job.processing_time_ms / job.total_products if job.total_products > 0 else 0
            
            # Update performance metrics
            self._update_performance_metrics(job)
            
            logger.info(f"Job {job.job_id} completed successfully. "
                       f"Success rate: {job.successful_classifications}/{job.total_products} "
                       f"({job.successful_classifications/job.total_products*100:.1f}%)")
            
        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.processing_time_ms = int((time.time() - start_time) * 1000)
            job.errors.append(f"Job processing failed: {str(e)}")
            
            logger.error(f"Job {job.job_id} failed: {e}")
            
            # Log system health issue
            self._log_batch_failure(job.job_id, str(e))
    
    def _process_product_batch(self, product_ids: List[int], config: BatchProcessingConfig, 
                              job_id: str) -> Dict[str, Any]:
        """
        Process a batch of products for classification.
        
        Args:
            product_ids: List of product IDs to process
            config: Batch processing configuration
            job_id: Parent job ID for logging
            
        Returns:
            Dict: Batch processing results
        """
        results = []
        errors = []
        successful_count = 0
        failed_count = 0
        
        try:
            # Fetch products from database
            products = Product.query.filter(Product.id.in_(product_ids)).all()
            
            if not products:
                return {
                    'results': results,
                    'errors': ['No products found for given IDs'],
                    'successful_count': 0,
                    'failed_count': len(product_ids)
                }
            
            # Process each product
            for product in products:
                try:
                    # Convert product to dictionary for classification
                    product_data = {
                        'title': product.title,
                        'description': product.description,
                        'price': product.price_numeric,
                        'seller_name': product.seller_name,
                        'rating': product.rating_numeric,
                        'reviews_count': product.reviews_count,
                        'platform': product.platform,
                        'url': product.url
                    }
                    
                    # Classify product
                    classification_result = self.classification_engine.classify_product(
                        product_data=product_data,
                        model_name="best_model",
                        include_explanation=config.include_explanations
                    )
                    
                    if classification_result['success']:
                        # Store classification in database
                        classification = Classification(
                            product_id=product.id,
                            prediction=classification_result['prediction_label'],
                            confidence_score=classification_result['confidence_score'],
                            feature_importance=classification_result.get('explanation', {}).get('top_features'),
                            explanation=classification_result.get('explanation', {}).get('business_explanation'),
                            model_version=classification_result.get('model_version'),
                            processing_time_ms=classification_result['processing_time_ms']
                        )
                        
                        db.session.add(classification)
                        successful_count += 1
                        
                        results.append({
                            'product_id': product.id,
                            'success': True,
                            'prediction': classification_result['prediction_label'],
                            'confidence_score': classification_result['confidence_score']
                        })
                        
                    else:
                        failed_count += 1
                        error_msg = f"Classification failed for product {product.id}: {classification_result.get('error')}"
                        errors.append(error_msg)
                        
                        results.append({
                            'product_id': product.id,
                            'success': False,
                            'error': classification_result.get('error')
                        })
                
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Error processing product {product.id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    
                    results.append({
                        'product_id': product.id,
                        'success': False,
                        'error': str(e)
                    })
            
            # Commit database changes
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            error_msg = f"Batch processing error for job {job_id}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return {
            'results': results,
            'errors': errors,
            'successful_count': successful_count,
            'failed_count': failed_count
        }
    
    def _update_performance_metrics(self, job: BatchProcessingJob):
        """Update overall performance metrics."""
        try:
            self.performance_metrics['total_batches_processed'] += 1
            self.performance_metrics['total_products_classified'] += job.successful_classifications
            
            # Update average processing time
            total_batches = self.performance_metrics['total_batches_processed']
            current_avg = self.performance_metrics['avg_batch_processing_time_ms']
            new_avg = ((current_avg * (total_batches - 1)) + job.processing_time_ms) / total_batches
            self.performance_metrics['avg_batch_processing_time_ms'] = new_avg
            
            # Update success rate
            if job.total_products > 0:
                job_success_rate = job.successful_classifications / job.total_products
                total_success_rate = self.performance_metrics['success_rate']
                new_success_rate = ((total_success_rate * (total_batches - 1)) + job_success_rate) / total_batches
                self.performance_metrics['success_rate'] = new_success_rate
            
            self.performance_metrics['last_batch_completed'] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _log_batch_failure(self, job_id: str, error_message: str):
        """Log batch processing failure to system health monitoring."""
        try:
            health_record = SystemHealth(
                component='batch_processor',
                status='unhealthy',
                metrics={'failed_job': job_id},
                message=f"Batch processing job '{job_id}' failed: {error_message}"
            )
            
            db.session.add(health_record)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to log batch failure to system health: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get batch processing performance metrics."""
        return {
            **self.performance_metrics,
            'active_jobs_count': len(self.active_jobs),
            'queue_size': self.job_queue.qsize(),
            'is_running': self.is_running,
            'worker_count': len(self.worker_threads)
        }
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """
        Clean up completed jobs older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for keeping completed jobs
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            with self._lock:
                jobs_to_remove = []
                
                for job_id, job in self.active_jobs.items():
                    if (job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED] and
                        job.completed_at and job.completed_at < cutoff_time):
                        jobs_to_remove.append(job_id)
                
                for job_id in jobs_to_remove:
                    del self.active_jobs[job_id]
                
                if jobs_to_remove:
                    logger.info(f"Cleaned up {len(jobs_to_remove)} old batch jobs")
            
        except Exception as e:
            logger.error(f"Error cleaning up completed jobs: {e}")


# Global batch processor instance
batch_processor = BatchProcessor()


def initialize_batch_processor(models_path: str = "models") -> bool:
    """
    Initialize the global batch processor.
    
    Args:
        models_path: Path to ML models directory
        
    Returns:
        bool: True if initialization successful
    """
    return batch_processor.initialize_services(models_path)


def start_batch_processing(max_workers: int = 2):
    """Start batch processing workers."""
    batch_processor.start_processing(max_workers)


def stop_batch_processing():
    """Stop batch processing workers."""
    batch_processor.stop_processing()


def process_new_products_batch(scraping_job_id: Optional[int] = None) -> Optional[str]:
    """
    Process newly scraped products in batch.
    
    Args:
        scraping_job_id: Optional scraping job ID to process
        
    Returns:
        Optional[str]: Batch job ID or None if no products to process
    """
    return batch_processor.process_new_products(scraping_job_id)


def get_batch_processor_status() -> Dict[str, Any]:
    """Get overall batch processor status."""
    return {
        'performance_metrics': batch_processor.get_performance_metrics(),
        'active_jobs': batch_processor.get_all_jobs_status()
    }