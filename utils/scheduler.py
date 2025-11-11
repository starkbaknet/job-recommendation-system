import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from db.models import Job, Applicant
from db.database import SessionLocal
from utils.vector_utils import compute_job_vectors_batch
import schedule
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorComputationScheduler:
    def __init__(self):
        # Load batch size from environment variable
        self.batch_size = int(os.getenv('VECTOR_COMPUTATION_BATCH_SIZE', '50'))
        self.max_workers = int(os.getenv('VECTOR_COMPUTATION_MAX_WORKERS', '4'))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = False
        self.lock = threading.Lock()
    
    def compute_job_vectors_batch(self):
        """
        Compute vectors for jobs that don't have vectors yet
        """
        logger.info(f"[{datetime.now()}] Starting job vector computation batch...")
        
        db = SessionLocal()
        try:
            processed = compute_job_vectors_batch(db, self.batch_size)
            logger.info(f"[{datetime.now()}] Job vector computation completed. Processed {processed} jobs.")
        except Exception as e:
            logger.error(f"Error during job vector computation: {str(e)}")
        finally:
            db.close()
    
    def start_scheduler(self):
        """
        Start the background scheduler
        """
        logger.info("Starting vector computation scheduler...")
        logger.info(f"Scheduler configured with batch_size: {self.batch_size}, max_workers: {self.max_workers}")
        
        self.running = True
        
        # Schedule the job to run every hour
        schedule.every().hour.do(self.compute_job_vectors_batch)
        
        # Also run once at startup for any existing entries without vectors
        self.compute_job_vectors_batch()
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """
        Stop the background scheduler
        """
        self.running = False
        logger.info("Vector computation scheduler stopped.")

def run_scheduler():
    """
    Function to run the scheduler in a separate thread
    """
    scheduler = VectorComputationScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    # Run scheduler directly if script is executed
    scheduler = VectorComputationScheduler()
    scheduler.start_scheduler()