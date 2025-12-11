import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Central configuration class for the Smart Spot Trading System.
    Loads variables from .env and enforces strict validation rules.
    """
    
    # Project Structure Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"

    # Create directories if they don't exist
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    # Exchange & Infrastructure
    EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance").lower()
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1") # Default generic region
    
    # Trading Parameters
    SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")
    
    # Credentials
    API_KEY = os.getenv("API_KEY", "")
    SECRET_KEY = os.getenv("SECRET_KEY", "")
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "") # New

    # Feature Flags
    IS_PAPER_TRADING = os.getenv("IS_PAPER_TRADING", "True").lower() == "true" # New
    STRATEGY_PROFILE = os.getenv("STRATEGY_PROFILE", "AUDITED").upper() # SNIPER, AUDITED, RECKLESS

    # Risk Management
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.02")) # New
    MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "1.0")) # New
    STOP_LOSS_ATR_MULT = float(os.getenv("STOP_LOSS_ATR_MULT", "2.0")) # New
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") # New
    
    def __init__(self):
        self._validate_golden_rules()

    def _validate_golden_rules(self):
        """
        Enforce Golden Rules and Critical Constraints.
        """
        # Section 5.R: Network Topology & Colocation
        # Requisito: El VPS de inferencia debe estar en la misma región AWS que el Exchange.
        # Specific check for Binance -> AWS Tokyo (ap-northeast-1)
        if self.EXCHANGE_ID == 'binance' and self.AWS_REGION != 'ap-northeast-1':
            warnings.warn(
                f"LATENCY WARNING: Exchange is '{self.EXCHANGE_ID}' but AWS_REGION is '{self.AWS_REGION}'. "
                f"Optimal region for Binance is 'ap-northeast-1' (Tokyo). "
                f"High latency (>20ms) effectively kills high-freq strategies.",
                UserWarning
            )

    @classmethod
    def get_db_path(cls) -> Path:
        return cls.DATA_DIR / "trading_system.db"

# Instantiate to run validations immediately when imported? 
# Usually better to let user instantiate, but for a global config singleton pattern:
config = Config()
