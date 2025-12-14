import pandas as pd
import numpy as np
import ta # Changed from pandas_ta
import logging
from src.data.frac_diff import FractionalDifferencing

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Computes technical indicators and time-based features.
    Updated for Phase 14: Cross-Asset Context.
    Updated for Phase 19: Fractional Differencing.
    Refactored to use 'ta' library.
    """
    
    def __init__(self):
        self.ffd = FractionalDifferencing(d=0.4) 

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds RSI, MACD, ATR, and Log Returns.
        """
        df = df.sort_index()
        
        # 1. Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. RSI (ta lib)
        rsi_ind = ta.momentum.RSIIndicator(close=df["close"], window=14)
        df['rsi'] = rsi_ind.rsi()
        
        # 3. MACD
        macd_ind = ta.trend.MACD(close=df["close"])
        df['MACD_12_26_9'] = macd_ind.macd() # Standard name used in system
        
        # 4. ATR
        atr_ind = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df['atr'] = atr_ind.average_true_range()
        
        # 5. Volatility
        df['volatility_24h'] = df['log_ret'].rolling(window=24).std()
        
        # 6. Bollinger Bands & Distances (Agent 2 Essentials)
        bb_indicator = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df['bb_high'] = bb_indicator.bollinger_hband()
        df['bb_low'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close'] # Relative Width
        df['dist_to_ma'] = (df['close'] - bb_indicator.bollinger_mavg()) / df['close'] # Normalized Dist
        
        return df

    def add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Cross-Asset features if context columns exist (e.g. close_BTC).
        """
        context_cols = [c for c in df.columns if c.startswith("close_") and c != "close"]
        
        for col in context_cols:
            suffix = col.split("_")[1] # e.g. BTC
            
            # 1. Rolling Correlation (30 bars)
            ctx_ret = np.log(df[col] / df[col].shift(1))
            main_ret = df['log_ret'] 
            
            df[f'corr_{suffix}'] = main_ret.rolling(30).corr(ctx_ret)
            
            # 2. Beta / Relative Strength (Ratio)
            df[f'rel_str_{suffix}'] = df['close'] / df[col]
            
        return df

    def add_ffd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Fractionally Differenced Close Price.
        Preserves memory while achieving stationarity.
        """
        # Apply FFD to Close
        # Note: This is computationally expensive, so use fixed window if needed.
        # For simplicity in this pipeline, we use the fixed mapping.
        try:
             df['close_ffd'] = self.ffd.frac_diff_fixed(df['close'])
        except Exception as e:
             logger.warning(f"FFD Calculation failed: {e}")
             
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Cyclical Time Features.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
            
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        return df
    
    def clean_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows due to NaN (Warm-up period).")
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(df)
        df = self.add_ffd_features(df) # NEW: Logic Connected
        df = self.add_context_features(df)
        df = self.add_time_features(df)
        df = self.clean_nan(df)
        return df
