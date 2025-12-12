import sys
import os
import logging
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.data.pipeline_contrastive import ContrastiveDataPipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline_1h.log")
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting 1H Data Pipeline for Contrastive Agent...")
    
    # Override Config explicitly for this run
    # (Though run_pipeline matches args, fetching inside might use config.TIMEFRAME default)
    # The pipeline.run_pipeline accepts `timeframe` arg, so we pass it there.
    
    pipeline = ContrastiveDataPipeline()
    pipeline.run_pipeline(timeframe='1h')
    
    logger.info("✅ 1H Data Pipeline Completed Successfully.")
