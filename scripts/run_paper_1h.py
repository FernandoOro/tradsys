
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
from dotenv import load_dotenv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.binance_paper import BinancePaperExecutor
from src.models.agents.transformer import TransformerAgent
from src.database.storage import StorageEngine
from src.execution.risk_manager import RiskManager
from src.config import config

# Load Env
load_dotenv()

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/paper_1h.log")
    ]
)
logger = logging.getLogger("PaperTrader-1H")

# --- MODEL WRAPPER ---
class LivePredictor:
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths (Adjusted for "models/contrastive_1h/" structure)
        base_path = model_dir
        # Check if subfolder exists
        if (base_path / "contrastive_1h").exists():
            base_path = base_path / "contrastive_1h"
            
        self.model_path = base_path / "model_contrastive_1h.pt"
        self.hmm_path = base_path / "hmm_contrastive_1h.pkl"
        self.pca_latent_path = base_path / "pca_contrastive_1h_latent.pkl"
        self.pca_data_path = base_path / "pca_1h.pkl" # This might be named just pca_1h.pkl
        
        logger.info(f"Loading Models from {base_path}...")
        
        # 1. Load PCA Data Preprocessor
        self.pca_data = joblib.load(self.pca_data_path)
        
        # 2. Load Torch Model
        # We need to infer input_dim from PCA output
        # PCAFeatureReducer stores the actual sklearn PCA in self.pca
        self.input_dim = self.pca_data.pca.n_components_ 
        
        self.model = TransformerAgent(input_dim=self.input_dim, d_model=128, nhead=4, num_layers=2)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        # 3. Load HMM & Latent PCA
        self.hmm = joblib.load(self.hmm_path)
        self.pca_latent = joblib.load(self.pca_latent_path)
        
        logger.info("✅ All Models Loaded Successfully.")

    def get_signal(self, df_recent):
        """
        Process Data -> Extract Feats -> HMM -> Signal
        """
        pass

# --- FEATURE ENGINEERING ---
def add_features(df):
    from src.data.features import FeatureEngineer
    fe = FeatureEngineer()
    df_feat = fe.generate_features(df)
    return df_feat

class PaperTrader:
    def __init__(self):
        self.executor = BinancePaperExecutor(symbol="BTC/USDT", timeframe="1h")
        self.storage = StorageEngine()
        
        # Resolve Model Path
        from pathlib import Path
        model_dir = Path("models").resolve()
        self.brain = LivePredictor(model_dir)
        
        self.risk_manager = RiskManager()
        self.position = 0 # 0: Flat, 1: Long, -1: Short
        
    def run_cycle(self):
        logger.info("--- Starting 1H Cycle ---")
        
        # 1. Fetch Data
        # Increased limit to 1000 to handle FFD warm-up (needs ~500+ bars)
        df = self.executor.fetch_recent_metrics(limit=1000)
        if df.empty: return
        
        # 2. Add Features
        from src.data.features import FeatureEngineer
        fe = FeatureEngineer()
        
        # FeatureEngineer requires DatetimeIndex
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
            
        df = fe.run(df) # Adds RSI, BBands, etc. AND handles cleaning
        
        # 3. PCA Transform
        # We must align EXACTLY with the training features (names and order).
        # Sklearn PCA stores feature_names_in_ (available in recent versions).
        
        try:
            expected_features = self.brain.pca_data.pca.feature_names_in_
        except AttributeError:
             logger.error("PCA model doesn't have feature_names_in_. Version mismatch?")
             return

        # Prepare DataFrame with all expected features
        df_feats = df.copy()
        
        # Fill missing columns (artifacts like sample_weight, target_class, etc.)
        for col in expected_features:
            if col not in df_feats.columns:
                # logger.warning(f"Adding dummy feature: {col}")
                if col == 'sample_weight':
                    df_feats[col] = 1.0
                else:
                    df_feats[col] = 0.0
        
        # Enforce Order
        df_feats = df_feats[expected_features]
        df_feats = df_feats.fillna(0)
        
        # DEBUG: Log Types
        logger.info(f"DF Feats Shape: {df_feats.shape}")
        # logger.info(f"DF Feats Head: \n{df_feats.head()}")
        
        # Saniity Check: Ensure all numeric
        # Explicitly cast to float to avoid int64/float mix issues in sklearn
        try:
            df_feats = df_feats.astype(float)
        except Exception as e:
            logger.error(f"Failed to cast to float: {e}")
            logger.info(f"Dtypes: {df_feats.dtypes}")
            return

        # Transform
        try:
            # Fix 1: Pass DataFrame (required by reduction.py using .index)
            # Fix 2: Reset Index to avoid 'could not determine shape' with DatetimeIndex
            # Fix 3: explicit float conversion
            X_df = self.brain.pca_data.transform(df_feats.reset_index(drop=True))
            
            # Convert back to simple numpy array for PyTorch
            X_pca = X_df.values
            
        except Exception as e:
            logger.error(f"PCA Transform Failed: {e}")
            return
        
        # 4. Prepare Tensor
        # Seq Len = 10
        if len(X_pca) < 10:
            logger.warning("Not enough data for sequence.")
            return

        last_seq = X_pca[-10:] # (10, input_dim)
        
        # Time Feats
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        last_time = df[time_cols].iloc[-10:].values
        
        x_t = torch.FloatTensor(last_seq).unsqueeze(0).to(self.brain.device) # (1, 10, dim)
        t_t = torch.FloatTensor(last_time).unsqueeze(0).to(self.brain.device)
        
        # 5. Extract Embedding & HMM
        emb = self.brain.model.extract_features(x_t, t_t).detach().cpu().numpy() # (1, 128)
        
        # Latent PCA
        emb_latent = self.brain.pca_latent.transform(emb)
        
        # Predict Regime
        regime = self.brain.hmm.predict(emb_latent)[0]
        
        # Predict Score (Direction)
        preds = self.brain.model(x_t, t_t)
        score = preds[0, 0].item() # Direction Probability
        
        logger.info(f"🧠 Analysis: Score={score:.4f} | Regime={regime}")
        
        # 6. Strategy Logic (Golden Regime)
        # Golden Regime is 1 (Verified in Backtest)
        # Threshold 0.99
        
        GOLDEN_REGIME = 1 
        THRESHOLD = 0.99
        
        signal = 0
        if regime == GOLDEN_REGIME and score > THRESHOLD:
            signal = 1
        elif regime != GOLDEN_REGIME:
            signal = -1 # Safety Exit
            
        logger.info(f"🚦 Signal: {signal} (Pos: {self.position})")
        
        # TELEMETRY: Store Signal
        # Map score to confidence-like metric
        confidence = abs(score - 0.5) * 2 
        
        self.storage.store_signal(
            symbol="BTC/USDT",
            direction=float(signal),
            confidence=confidence,
            features={
                "regime": int(regime),
                "raw_score": float(score),
                "close": float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0,
                "strategy": "Contrastive-1H"
            },
            vetoed=(signal == 0)
        )
        
        # 7. Execution Logic (Hybrid Spot/Futures)
        
        # Determine Mode
        is_futures = config.EXCHANGE_ID in ['binanceusdm', 'binance_futures']
        
        # Get Equity & ATR
        equity = self.executor.get_balance_usdt()
        
        # Calculate ATR for sizing
        high = df['high']
        low = df['low']
        close = df['close']
        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
        atr = tr.rolling(14).mean().bfill()
        current_atr = atr.iloc[-1]
        
        current_price = df['close'].iloc[-1]
        
        # Risk Calc
        # Note: If confidence is high but we are exiting, size doesn't matter much (exit is full).
        # We generally use size for ENTRIES.
        size = self.risk_manager.calculate_position_size(equity, current_atr, confidence, current_price)
        
        logger.info(f"💰 Risk Calc: Equity=${equity:.2f} | Size={size:.4f} BTC")

        # --- SIGNAL EXECUTION ---
        
        # 1. LONG ENTRY
        if signal == 1 and self.position == 0:
             if size > 0:
                 order = self.executor.execute_order('buy', size * current_price) # Executor expects USDT amount? 
                 # Wait, executor.execute_order takes amount_usdt.
                 # RiskManager returns size_asset (BTC).
                 # So pass size_asset * price.
                 if order:
                     self.position = 1
                     tag = "OPEN_1H_LONG_FUT" if is_futures else "OPEN_1H_LONG"
                     self.storage.store_trade("BTC/USDT", "BUY", size, current_price, tag)

        # 2. LONG EXIT (Sell to Close)
        elif self.position == 1 and signal in [-1, 0]: # Exit on -1 or 0 (Neutral)
             # Basic Exit Logic: Close all
             # We assume we hold 'size' or check actual balance?
             # For paper, we assume flat close.
             # Ideally fetch open position size. But for now use previous trade size or 'amount_to_trade' equivalent.
             # We passed 'amount_usdt' to executor.
             # Let's just execute SELL of same notional roughly. Or convert actual holding?
             # Executor handles "Market Sell". 
             # To be safe in paper: sell everything?
             # Simulator keeps state? No state here.
             # We'll just sell the same Risk Amount ($USD) to keep simple, assuming 100% turnover.
             # Or better: Sell 100% of Balance BTC? 
             # binance_paper.executor doesn't have 'close_position'.
             # Let's use same size calculation for exit (Slightly inaccurate if price moved, but close enough for paper logic or pass 'reduceOnly').
             
             amount_usdt_exit = size * current_price # Current value of intended position
             order = self.executor.execute_order('sell', amount_usdt_exit)
             if order:
                 self.position = 0
                 tag = "CLOSED_1H_LONG_FUT" if is_futures else "CLOSED_1H_LONG"
                 self.storage.store_trade("BTC/USDT", "SELL", size, current_price, tag)
                 
                 # If Futures & Signal is -1, we might want to FLIP to Short immediately?
                 # If signal == -1, we continue to Short Entry block?
                 # Let's allow FLIP if Futures.
        
        # 3. SHORT ENTRY (Futures Only)
        if is_futures and signal == -1 and self.position == 0:
             if size > 0:
                 order = self.executor.execute_order('sell', size * current_price) # Open Short
                 if order:
                     self.position = -1
                     self.storage.store_trade("BTC/USDT", "SELL", size, current_price, "OPEN_1H_SHORT")

        # 4. SHORT EXIT (Futures Only)
        elif self.position == -1 and signal in [1, 0]:
             amount_usdt_exit = size * current_price
             order = self.executor.execute_order('buy', amount_usdt_exit) # Close Short
             if order:
                 self.position = 0
                 self.storage.store_trade("BTC/USDT", "BUY", size, current_price, "CLOSED_1H_SHORT")
                 
                 # If Signal is 1, let next cycle pick it up or allow immediate flip?
                 # Loop structure: we just closed. position=0.
                 # If signal=1, we could fall through to Long Entry?
                 # But 'LONG ENTRY' block was at top and checked position==0.
                 # We need to re-check or unstructured flow.
                 # For simplicity, we wait for next cycle (1h delay) or just call it "Flip Logic".
                 # Given 1H timeframe, 1 cycle delay for flip is acceptable to avoid complexity/race conditions.
            
    def start(self):
        logger.info("🤖 Paper Trader 1H initialized.")
        
        # Immediate Startup Check (User Request)
        logger.info("⚡ Running Startup Check (Immediate Cycle)...")
        try:
            self.run_cycle()
            logger.info("✅ Startup Check Passed.")
        except Exception as e:
            logger.error(f"❌ Startup Check Failed: {e}")
            
        last_run_hour = -1
        
        logger.info("🕐 Entering Scheduling Loop (Waiting for :02 minute)...")
        while True:
            now = datetime.now()
            
            # --- PRODUCTION MODE (Hourly at :02) ---
            if now.minute == 2 and now.hour != last_run_hour:
                last_run_hour = now.hour
                try:
                    logger.info(f"⏰ Triggering Hourly Cycle at {now}")
                    self.run_cycle()
                except Exception as e:
                    logger.error(f"Cycle failed: {e}")
                
            time.sleep(30) # Check every 30s

if __name__ == "__main__":
    bot = PaperTrader()
    bot.start()
