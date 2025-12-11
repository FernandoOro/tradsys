
import logging
import pandas as pd
from src.config import config
from src.models.regime.hmm import RegimeDetector

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainHMM")

def train_hmm():
    """
    Trains the Hidden Markov Model (HMM) for market regime detection.
    Uses data from 'data/processed/train.parquet'.
    """
    logger.info("Starting HMM Training Pipeline...")
    
    # 1. Load Data
    data_path = config.PROCESSED_DATA_DIR / "train.parquet"
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}. Please run src/data/pipeline.py first.")
        return
        
    logger.info(f"Loading training data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Check if we have 'close' price (Pipeline preserves it)
    if 'close' not in df.columns:
        logger.error("Column 'close' is missing. Cannot calculate returns/volatility.")
        return

    # 2. Initialize Detector
    # n_components=3 (Bull, Bear, Range) or (Low Vol, High Vol, Panic)
    detector = RegimeDetector(n_components=3, n_iter=1000)
    
    # 3. Fit
    # The detector handles feature extraction (log_ret, vol) internally in 'prepare_data'
    logger.info("Fitting Gaussian HMM...")
    detector.fit(df)
    
    # 4. Interpret States (Heuristic)
    # predictor.means_ contains [mean_ret, mean_vol] for each state? 
    # Need to check prepare_data order: ['log_ret', 'volatility_24h']
    
    logger.info("Analyzing Learned States:")
    for i in range(detector.n_components):
        mean_ret = detector.model.means_[i][0]
        mean_vol = detector.model.means_[i][1]
        logger.info(f"State {i}: Mean Return={mean_ret:.6f}, Mean Volatility={mean_vol:.6f}")
        
    logger.info("✅ HMM Training Complete. Model saved artifact should be in models/hmm_regime.pkl")

if __name__ == "__main__":
    train_hmm()
