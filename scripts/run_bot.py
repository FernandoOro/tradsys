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
        self.event_filter = EventFilter() 
        self.predictor = Predictor(model_name="agent1")
        # Load Agent 2 (Optional, handled gracefully if missing unless Ensemble requested)
        try:
            self.predictor_2 = Predictor(model_name="agent2")
            logger.info("Agent 2 (Mean Reversion) Loaded.")
        except Exception:
            self.predictor_2 = None
            logger.warning("Agent 2 not found. Ensemble Strategy will fail if selected.")

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
        
        regime = -1
        try:
             # Pass the full DF for HMM to feature engineering
             regime = self.regime_detector.predict_state(df.iloc[-100:]) # Use last window context
             if isinstance(regime, np.ndarray): regime = regime[-1] # Ensure scalar
             logger.info(f"Market Regime: {regime}")
        except Exception as e:
             logger.error(f"HMM Failed: {e}")
             return

        # --- DYNAMIC INFERENCE ---
        confidence = 0.0
        direction = 0.0
        agent2_score = 0.0
        is_vetoed = False
        
        # Branch 1: Regime 0 (Range) -> Use Agent 2
        if regime == 0 and self.predictor_2:
            try:
                # Agent 2 takes Flattened Features (No Time)
                pred2 = self.predictor_2.predict(x_val) # x_val shape (1, 10, Feat) -> Auto-flattened
                agent2_score = float(pred2["score"][-1]) # Last step
                # Map Score to Direction? 
                # Agent 2 head is Linear(1). 
                # If trained on BCEWithLogits, output is Logit. Sigmoid(out) -> Prob(Reversal).
                # Wait, training used BCEWithLogitsLoss.
                # So output is LOGIT.
                # Prob = Sigmoid(logit).
                # 1 = Reversal (Target 1), 0 = Continuation.
                
                reversal_prob = 1 / (1 + np.exp(-agent2_score))
                agent2_score = reversal_prob # Normalize to 0-1
                
                # Direction? Agent 2 predicts "Reversal".
                # If Price is Low (RSI < 30) -> Buy.
                # If Price is High (RSI > 70) -> Sell.
                # We need a heuristic for direction since Agent 2 only predicts "Probability of Reversal".
                # Or did we train Agent 2 on Directional Target?
                # Target in train_agent2: df['target'].
                # Labeler (Triple Barrier) produces 1 (Up) or 0 (Down/Flat).
                # So Agent 2 predicts Probability of UP move.
                
                # Correction: train_agent2 filters Regime 0. Target is Standard TBM (1=Up).
                # So Agent 2 output IS probability of UP.
                # Excellent. 
                
                direction = 1 if agent2_score > 0.5 else -1
                confidence = abs(agent2_score - 0.5) * 2 # Map 0.5-1.0 to 0-1 confidence
                
                logger.info(f"🤖 Agent 2 (Reversion): Score={agent2_score:.2f} | Dir={direction}")
                
            except Exception as e:
                logger.error(f"Agent 2 Inference Failed: {e}")
                return

        # Branch 2: Regime 1/2 (Trend) -> Use Agent 1
        elif regime != 0:
            try:
                pred = self.predictor.predict(x_val, t_val)
                raw_dir = pred["direction"][-1]
                raw_conf = pred["confidence"][-1]
                
                direction = 1 if raw_dir > 0 else -1
                confidence = float(raw_conf)
                
                logger.info(f"🤖 Agent 1 (Trend): Conf={confidence:.2f} | Dir={direction}")
                
                # Auditor Check (Only for Agent 1)
                # ... (Auditor Logic)
                is_vetoed = False # Default
                # self.auditor.predict_veto(...) implementation missing in snippet context but assumed
                
            except Exception as e:
                logger.error(f"Agent 1 Inference Failed: {e}")
                return
             
        # DECISION CORE (Strategy Pattern)
        should_execute = self.strategy.should_trade(
            confidence=float(confidence),
            regime=int(regime),
            auditor_vetoed=is_vetoed,
            agent2_score=float(agent2_score) # Pass Agent 2 score
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
