import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Labeler:
    """
    Implements the Triple Barrier Method (TBM) for labeling financial time series.
    Barriers:
    - Upper: Price + k * Volatility (Profit Take)
    - Lower: Price - k * Volatility (Stop Loss)
    - Vertical: t + T (Time limit)
    """
    
    def __init__(self, barrier_width: float = 2.0, time_horizon: int = 24):
        """
        Args:
            barrier_width: Multiplier for volatility (e.g., 2.0 * ATR or StdDev)
            time_horizon: Max bars to hold (Vertical barrier)
        """
        self.barrier_width = barrier_width
        self.time_horizon = time_horizon

    def get_barriers(self, prices: pd.Series, volatility: pd.Series):
        """
        Calculates dynamic barrier levels.
        """
        upper = prices + (volatility * self.barrier_width)
        lower = prices - (volatility * self.barrier_width)
        return upper, lower

    def triple_barrier_method(self, close: pd.Series, volatility: pd.Series) -> pd.DataFrame:
        """
        Computes the labels.
        Returns DataFrame with:
        - target_class: 1 (Hit Upper), -1 (Hit Lower), 0 (Time out)
        - target_ret: Return achieved at barrier hit.
        - t_end: Index where the barrier was hit (for uniqueness calc).
        """
        t_hor = self.time_horizon
        n = len(close)
        
        # Prepare Output
        labels = np.zeros(n)
        returns = np.zeros(n)
        t_ends = np.zeros(n, dtype=int)
        
        p_arr = close.values
        v_arr = volatility.values
        
        for i in range(n - t_hor):
            current_price = p_arr[i]
            upper_barrier = current_price + (v_arr[i] * self.barrier_width)
            lower_barrier = current_price - (v_arr[i] * self.barrier_width)
            
            # Future window
            window = p_arr[i+1 : i+1+t_hor]
            
            # Check for hits
            hit_upper = np.where(window >= upper_barrier)[0]
            hit_lower = np.where(window <= lower_barrier)[0]
            
            first_upper = hit_upper[0] if len(hit_upper) > 0 else t_hor + 1
            first_lower = hit_lower[0] if len(hit_lower) > 0 else t_hor + 1
            
            if first_upper == t_hor + 1 and first_lower == t_hor + 1:
                # Vertical Barrier (Time out)
                labels[i] = 0
                returns[i] = (window[-1] - current_price) / current_price
                t_ends[i] = i + t_hor
            elif first_upper < first_lower:
                # Hit Upper first
                labels[i] = 1
                returns[i] = (window[first_upper] - current_price) / current_price
                t_ends[i] = i + 1 + first_upper
            else:
                # Hit Lower first
                labels[i] = -1
                returns[i] = (window[first_lower] - current_price) / current_price
                t_ends[i] = i + 1 + first_lower
                
        # Handle end of series
        labels[n-t_hor:] = np.nan
        returns[n-t_hor:] = np.nan
        t_ends[n-t_hor:] = 0
        
        df_labels = pd.DataFrame({
            'target_class': labels, 
            'target_ret': returns,
            't_end': t_ends
        }, index=close.index)
        
        return df_labels.dropna()

    def get_sample_weights(self, close: pd.Series, t_ends: pd.Series) -> pd.Series:
        """
        Computes sample weights based on label Uniqueness (De Prado).
        If multiple labels overlap in time, they share the weight.
        """
        # 1. Build Indicator Matrix (Concurrency)
        # For each time t, how many labels are active?
        # A label started at t_start (index) is active until t_end (value).
        
        # We need to map dataframe index to integer index for speed
        idx_map = {idx: i for i, idx in enumerate(close.index)}
        rev_map = {i: idx for i, idx in enumerate(close.index)}
        
        n = len(close)
        concurrency = np.zeros(n)
        
        # t_ends contains the integer index? Or the Series index?
        # Our triple_barrier method returned integer index relative to start 0?
        # Let's check: t_ends[i] = i + horizon. Yes, consistent with array index.
        # But wait, triple_barrier returns DataFrame indexed by time.
        # t_ends in output is likely int? Let's assume so from the code above.
        
        t_starts = np.arange(len(t_ends))
        t_ends_arr = t_ends.values.astype(int)
        
        # Compute Concurrency
        # Optimization: Loop? Or scan line?
        # Simple loop for robustness
        for i in range(len(t_starts)):
            start = t_starts[i]
            end = t_ends_arr[i]
            if end > start: # Valid
                concurrency[start:end] += 1
        
        # Avoid division by zero
        concurrency[concurrency == 0] = 1
        
        # 2. Compute Average Uniqueness per Label
        uniqueness = np.zeros(len(t_starts))
        
        for i in range(len(t_starts)):
            start = t_starts[i]
            end = t_ends_arr[i]
            if end > start:
                # Harmonic average or simple average of (1/concurrency) over the lifespan
                w = 1.0 / concurrency[start:end]
                uniqueness[i] = w.mean()
                
        # 3. Normalize
        # Scale to sum to N (preserve sample size magnitude)
        weights = uniqueness
        weights = weights * (len(weights) / (weights.sum() + 1e-8))
        
        return pd.Series(weights, index=close.index[:len(weights)])

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends targets and weights to dataframe.
        """
        if 'atr' not in df.columns:
            logger.warning("ATR not found for TBM Labeling. Calculating simplistic volatility.")
            vol = df['close'].rolling(20).std()
        else:
            vol = df['atr']
            
        labels_df = self.triple_barrier_method(df['close'], vol)
        
        # Calculate Weights
        # t_end from labels_df corresponds to integer index relative to df?
        # triple_barrier method uses numpy array logic on 'close'.
        # Assuming df and close are aligned.
        
        weights = self.get_sample_weights(df['close'].iloc[:len(labels_df)], labels_df['t_end'])
        labels_df['sample_weight'] = weights
        
        # Align
        df_labeled = df.join(labels_df[['target_class', 'target_ret', 'sample_weight']], how='inner')
        return df_labeled

    def add_reversion_targets(self, df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """
        Creates labels for Mean Reversion (Agent 2).
        STRICT LOGIC:
        1. Buy Signal (1): Price < BB_Lower AND Hits Take Profit (+1%) BEFORE Stop Loss (-0.5%).
        2. Sell Signal (2): Price > BB_Upper AND Hits Take Profit (-1%) BEFORE Stop Loss (+0.5%).
        3. Neutral (0): Else
        """
        if 'bb_low' not in df.columns:
            logger.warning("Bollinger Bands not found. Please run feature engineering first.")
            return df
            
        n = len(df)
        labels = np.zeros(n)
        
        # Parameters
        TP_PCT = 0.010  # Reduced to 1.0% (Reversion is quick)
        SL_PCT = 0.005  # Tight Stop (0.5%) - Risk/Reward 2:1
        
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
        # We need to loop or use advanced numpy to check "First Hit".
        # For simplicity and clarity in python, a numba-like loop is best, 
        # but here we use a simple window loop which is fast enough for 100k rows.
        
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
        return df
