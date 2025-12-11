import os
import sys
import time
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.normalization import Normalizer
from src.data.frac_diff import FractionalDifferencing
from src.data.sanitation import EventFilter
from src.inference.predictor import Predictor
from src.execution.risk_manager import RiskManager
from src.execution.router import SmartRouter
from src.models.ensemble.meta_labeling import MetaAuditor
from src.models.regime.hmm import RegimeDetector
from src.database.storage import StorageEngine
from src.execution.strategies import StrategyFactory

# Configure Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, mode: str = "paper"):
        self.mode = mode
        logger.info(f"Initializing SYSTEM in {mode.upper()} mode...")
        
        self.storage = StorageEngine()
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.normalizer = Normalizer(window_size=100)
        
        try:
            with open(config.MODELS_DIR / "pca.pkl", "rb") as f:
                self.pca = pickle.load(f)
            logger.info("PCA Model Loaded.")
        except FileNotFoundError:
            logger.error("PCA Model not found! Run pipeline first.")
            sys.exit(1)
        
        self.event_filter = EventFilter() 
        self.predictor = Predictor(model_name="agent1")
        self.auditor = MetaAuditor("auditor_v1")
        # Correctly initialize and force load
        self.regime_detector = RegimeDetector(n_components=3)
        self.regime_detector.load_model() # Loads from hmm_regime.pkl
        
        self.risk_manager = RiskManager()
        
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        self.exchange = exchange_class({
            'apiKey': config.API_KEY,
            'secret': config.SECRET_KEY,
            'enableRateLimit': True
        })
        self.router = SmartRouter(self.exchange)
        
        self.seq_len = 10
        self.feat_dim = self.pca.pca.n_components_
        
        # Load Strategy
        self.strategy = StrategyFactory.get_strategy()
        logger.info(f"Loaded Strategy Profile: {self.strategy.name}")
        
    def run_cycle(self):
        symbol = config.SYMBOL
        timeframe = config.TIMEFRAME
        
        logger.info("--- Starting Cycle ---")
        
        if self.event_filter.is_high_impact_now(datetime.now()):
            logger.warning("HIGH IMPACT EVENT DETECTED. SLEEPING.")
            return

        df = self.loader.fetch_market_context(symbol, timeframe, limit=500)
        if df.empty: return
        
        last_ts = df.index[-1]
        now_utc = datetime.now(timezone.utc)
        time_diff = (now_utc - last_ts).total_seconds()
        
        if time_diff > 300: 
            logger.error(f"CRITICAL: Data Stale! Lag: {time_diff}s. Aborting.")
            return

        df = self.engineer.run(df) 
        
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        exclude_norm = time_cols + ['open', 'high', 'low', 'close', 'volume', 'datetime', 'target']
        df_norm = self.normalizer.apply_normalization(df, exclude_cols=exclude_norm)
        
        if len(df_norm) < self.seq_len:
            return
            
        recent_window = df_norm.iloc[-self.seq_len:]
        
        # PCA Transform
        feature_cols = [c for c in df_norm.select_dtypes(include=['number']).columns if c not in exclude_norm]
        
        try:
             x_pca = self.pca.transform(recent_window[feature_cols])
        except Exception as e:
             logger.error(f"PCA Transform failed: {e}")
             return
             
        x_val = x_pca.values 
        t_val = recent_window[time_cols].values 
        
        x_val = np.expand_dims(x_val, 0)
        t_val = np.expand_dims(t_val, 0)
        
        # REGIME CHECK (Connected)
        regime = -1
        try:
             # Pass the full DF for HMM to feature engineering
             regime = self.regime_detector.predict_state(df) 
             logger.info(f"Market Regime: {regime}")
             
        # DECISION CORE (Strategy Pattern)
        should_execute = self.strategy.should_trade(
            confidence=float(confidence),
            regime=int(regime),
            auditor_vetoed=is_vetoed
        )
        
        if not should_execute:
            logger.info(f"🛑 Trade Rejected by {self.strategy.name} strategy.")
            return

        # EXECUTION
        if self.mode == "paper":
            equity = 10000.0
            size = self.risk_manager.calculate_position_size(equity, current_atr, confidence, current_price)
            if size > 0:
                action = "BUY" if direction > 0 else "SELL"
                logger.info(f"✅ PAPER EXECUTION: {action} {size:.4f} {symbol}")
                self.storage.store_trade(
                    symbol=symbol,
                    side=action,
                    size=size,
                    price=current_price,
                    status="OPEN"
                )
                
        elif self.mode == "live":
            equity = self.exchange.fetch_balance()['total']['USDT']
            size = self.risk_manager.calculate_position_size(equity, current_atr, confidence, current_price)
            if size > 0:
                action = "buy" if direction > 0 else "sell"
                self.router.execute_order(symbol, action, size)
                self.storage.store_trade(
                    symbol=symbol,
                    side=action.upper(),
                    size=size,
                    price=current_price,
                    status="OPEN"
                )

    def start(self):
        logger.info("System Online. PCA Active. Persistence Active. HMM Active.")
        try:
            while True:
                self.run_cycle()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutdown.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["paper", "live"], help="Override .env IS_PAPER_TRADING")
    args = parser.parse_args()
    
    # Determine Mode: CLI > Env > Default(Paper)
    if args.mode:
        final_mode = args.mode
    else:
        final_mode = "paper" if config.IS_PAPER_TRADING else "live"
        
    bot = TradingBot(mode=final_mode)
    bot.start()
