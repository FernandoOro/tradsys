
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.agent2.pipeline import ReversionDataPipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    pipeline = ReversionDataPipeline()
    # High limit to get 10 years of data (100k hours)
    pipeline.run_pipeline(limit=100000)
