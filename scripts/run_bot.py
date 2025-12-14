import os
import sys
import time
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import ccxt
import joblib
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
        # Fix: Point explicitly to subdirectory "models/agent1/agent1.onnx"
        self.predictor = Predictor(model_name="agent1/agent1")
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
        
        # Path Fix: Check models/agent1 first
        hmm_path = config.MODELS_DIR / "agent1" / "hmm_regime.pkl"
        if not hmm_path.exists():
            hmm_path = config.MODELS_DIR / "hmm_regime.pkl"
            
        if hmm_path.exists():
            try:
                self.regime_detector.model = joblib.load(hmm_path)
                self.regime_detector.is_fitted = True
                logger.info(f"HMM model loaded from {hmm_path}.")
            except Exception as e:
                logger.error(f"Failed to load HMM: {e}")
        else:
            logger.warning(f"HMM model not found at {hmm_path}. Need to fit first.")
        
        self.risk_manager = RiskManager()
        
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        exchange_options = {
            'apiKey': config.API_KEY,
            'secret': config.SECRET_KEY,
            'enableRateLimit': True
        }
        
        # HYBRID ARCHITECTURE SWITCH
        if config.EXCHANGE_ID in ['binanceusdm', 'binance_futures']:
             logger.info("🚀 FUTURES MODE DETECTED. Configuring CCXT for Derivatives.")
             exchange_options['options'] = {'defaultType': 'future'}
             
        self.exchange = exchange_class(exchange_options)
        self.router = SmartRouter(self.exchange)
        
        self.seq_len = 10
        self.feat_dim = self.pca.pca.n_components_
        
        # Load Strategy
        self.strategy = StrategyFactory.get_strategy()
        # force 1m timeframe for Legacy Agent (Safety override)
        config.TIMEFRAME = '1m'
        logger.info(f"Loaded Strategy Profile: {self.strategy.name} on Timeframe: {config.TIMEFRAME}")
        
        # Testnet Paper Trading Setup
        self.use_testnet = False
        if self.mode == "paper":
            testnet_key = os.getenv("BINANCE_TESTNET_KEY")
            testnet_secret = os.getenv("BINANCE_TESTNET_SECRET")
            if testnet_key and testnet_secret:
                # FIX: Futures Sandbox is deprecated/broken in CCXT. Using Internal Sim for Futures.
                if config.EXCHANGE_ID in ['binanceusdm', 'binance_futures']:
                     logger.warning("🟡 Futures Sandbox Not Supported. Forcing INTERNAL SIMULATION.")
                     self.use_testnet = False
                else:
                     logger.info("🟢 Testnet Keys Detected. Enabling SANDBOX MODE for Paper Trading.")
                     self.exchange.apiKey = testnet_key
                     self.exchange.secret = testnet_secret
                     self.exchange.set_sandbox_mode(True)
                     self.use_testnet = True
            else:
                logger.warning("🟡 No Testnet Keys. Running in INTERNAL SIMULATION Mode (DB Only).")

    def run_cycle(self):
        symbol = config.SYMBOL
        # Ensure 1m timeframe is used
        timeframe = '1m' 
        
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
        current_price = df['close'].iloc[-1]
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
                
                reversal_prob = 1 / (1 + np.exp(-agent2_score))
                agent2_score = reversal_prob # Normalize to 0-1
                
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
                is_vetoed = False # Default
                
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
            # Still log the signal as VETOED for analysis
            self.storage.store_signal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                features={
                    "regime": int(regime),
                    "price": float(current_price),
                    "agent2_score": float(agent2_score),
                    "strategy": self.strategy.name,
                    "reason": "Strategy Reject"
                },
                vetoed=True
            )
            return

        # TELEMETRY: Trace Accepted Signal
        self.storage.store_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            features={
                "regime": int(regime),
                "price": float(current_price),
                "agent2_score": float(agent2_score),
                "strategy": self.strategy.name,
                "reason": "Accepted"
            },
            vetoed=False
        )

        # EXECUTION
        if self.mode == "paper":
            if self.use_testnet:
                 # TRUE PAPER TRADING (Binance Testnet)
                 equity = self.exchange.fetch_balance()['total']['USDT']
                 size = self.risk_manager.calculate_position_size(equity, current_atr=0.0, confidence=confidence, price=0.0)
                 # Wait, pass equity and price? Need current price?
                 ticker = self.exchange.fetch_ticker(symbol)
                 current_price = ticker['last']
                 # Recalculate size with price
                 # ATR? We computed indicators. Let's get ATR.
                 # df['atr'] exists from FeatureEngineer? FeatEng adds 'atr'.
                 current_atr = df['atr'].iloc[-1]
                 
                 size = self.risk_manager.calculate_position_size(equity, current_atr, confidence, current_price)
                 
                 if size > 0:
                     action = "buy" if direction > 0 else "sell"
                     logger.info(f"✅ TESTNET ORDER: {action.upper()} {size} {symbol}")
                     try:
                         self.router.execute_order(symbol, action, size)
                         # Tag with Strategy Name for Dashboard Segmentation
                         status_tag = f"OPEN_{self.strategy.name}"
                         if config.EXCHANGE_ID in ['binanceusdm', 'binance_futures']:
                             status_tag += "_FUT"
                             
                         self.storage.store_trade(symbol, action.upper(), size, current_price, status_tag)
                     except Exception as e:
                         logger.error(f"Testnet Order Failed: {e}")
            else:
                # INTERNAL SIMULATION (Fake Money)
                equity = 10000.0
                current_price = df['close'].iloc[-1]
                current_atr = df['atr'].iloc[-1]
                size = self.risk_manager.calculate_position_size(equity, current_atr, confidence, current_price)
                if size > 0:
                    action = "BUY" if direction > 0 else "SELL"
                    logger.info(f"✅ PAPER EXECUTION (SIM): {action} {size:.4f} {symbol}")
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
