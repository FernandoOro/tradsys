
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ReversionLabeler:
    """
    Specialized Labeler for Agent 2 (Mean Reversion).
    Optimized for high-frequency reversal signals (scalping).
    """
    
    def __init__(self):
        pass

    def add_reversion_targets(self, df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """
        Creates labels for Mean Reversion (Agent 2).
        
        RELAXED LOGIC (For Training Viability):
        1. Buy Signal (1): Price < BB_Lower AND Hits Take Profit (+0.5%) BEFORE Stop Loss (-0.5%).
        2. Sell Signal (2): Price > BB_Upper AND Hits Take Profit (-0.5%) BEFORE Stop Loss (+0.5%).
        3. Neutral (0): Else
        
        Changes from original:
        - TP reduced from 1.0% to 0.5% (Easier to hit).
        - Risk/Reward ratio is now 1:1.
        """
        if 'bb_low' not in df.columns:
            logger.warning("Bollinger Bands not found. Please run feature engineering first.")
            return df
            
        n = len(df)
        labels = np.zeros(n)
        
        # Parameters (RELAXED)
        TP_PCT = 0.005  # 0.5% Target (Quick Scalp)
        SL_PCT = 0.005  # 0.5% Stop
        
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        
        # 1. Identify Potential Zones (Triggers)
        # Using .values for speed
        bb_low = df['bb_low'].values
        bb_high = df['bb_high'].values
        
        potential_buy = close < bb_low
        potential_sell = close > bb_high
        
        # 2. Vectorized Barrier Check
        for i in range(n - horizon):
            # BUY LOGIC
            if potential_buy[i]:
                entry = close[i]
                tp_price = entry * (1 + TP_PCT)
                sl_price = entry * (1 - SL_PCT)
                
                # Look ahead
                window_high = high[i+1 : i+1+horizon]
                window_low = low[i+1 : i+1+horizon]
                
                # Did we hit TP?
                tp_hits = np.where(window_high >= tp_price)[0]
                # Did we hit SL?
                sl_hits = np.where(window_low <= sl_price)[0]
                
                first_tp = tp_hits[0] if len(tp_hits) > 0 else horizon + 1
                first_sl = sl_hits[0] if len(sl_hits) > 0 else horizon + 1
                
                if first_tp < first_sl:
                    labels[i] = 1 # Valid Buy
                    
            # SELL LOGIC
            elif potential_sell[i]:
                entry = close[i]
                tp_price = entry * (1 - TP_PCT)
                sl_price = entry * (1 + SL_PCT)
                
                window_high = high[i+1 : i+1+horizon]
                window_low = low[i+1 : i+1+horizon]
                
                tp_hits = np.where(window_low <= tp_price)[0]
                sl_hits = np.where(window_high >= sl_price)[0]
                
                first_tp = tp_hits[0] if len(tp_hits) > 0 else horizon + 1
                first_sl = sl_hits[0] if len(sl_hits) > 0 else horizon + 1
                
                if first_tp < first_sl:
                    labels[i] = 2 # Valid Sell
                    
        df['target_reversion'] = labels
        
        # Count stats
        counts = pd.Series(labels).value_counts()
        logger.info(f"Reversion Labels Generated: {counts.to_dict()}")
        
        return df
