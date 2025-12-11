import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FractionalDifferencing:
    """
    Implements Fractional Differencing (FFD) to make time series stationary 
    while preserving as much memory (correlation) as possible.
    Reference: Marcos Lopez de Prado, 'Advances in Financial Machine Learning'.
    """
    
    def __init__(self, d: float = 0.4, thres: float = 1e-4):
        self.d = d
        self.thres = thres

    def get_weights(self, d: float, size: int):
        """
        Calculates weights for the fractional difference.
        w_k = -w_{k-1} * (d - k + 1) / k
        """
        w = [1.0]
        for k in range(1, size):
            w_ = -w[-1] * (d - k + 1) / k
            w.append(w_)
        
        w = np.array(w[::-1]).reshape(-1, 1) # Reverse to convolve
        return w

    def get_weights_floored(self, d: float, thres: float):
        """
        Calculates weights but cuts off when weight < threshold to save compute.
        """
        w = [1.0]
        k = 1
        while True:
            w_ = -w[-1] * (d - k + 1) / k
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_fixed(self, series: pd.Series, window_len: int = None):
        """
        Applies Fixed-Window Fractional Differencing.
        """
        # 1. Compute Weights
        w = self.get_weights_floored(self.d, self.thres)
        width = len(w) - 1
        
        # 2. Convolve
        # Handle N<width
        if len(series) < width:
            logger.warning("Series length smaller than FFD window width. Returning NaNs.")
            return pd.Series(np.nan, index=series.index)
            
        # Apply filter
        # We can use pd.rolling().apply but getting the dot product is faster with numpy
        # Logic: 
        # For each t, value is dot(weights, series[t-width:t])
        
        df = {}
        # Simple iterative loop for clarity/safety, or stride tricks for speed.
        # Given "Institution Grade", let's be robust.
        # But looping python is slow.
        # Optimization: pandas rolling apply with raw=True
        
        # To vectorise:
        # Weights are constant.
        # This is a convolution.
        
        # Let's use np.convolve if valid, treating series as 1D.
        # Convolve mode 'valid' returns only parts where full overlap exists.
        
        values = series.values
        if np.isnan(values).any():
             # Fill logic?
             # For now assume robust pipeline beforehand.
             pass
             
        # Reverse weights because np.convolve flips them again?
        # np.convolve(a, v, mode='valid')
        # w is already [w_n, ..., w_0] (reversed in get_weights?). 
        # Actually get_weights usually gives [w0, w1, ...] then we reversed it to [... w1, w0].
        # If we use convolution, check alignment.
        # Standard: Yt = w0*Xt + w1*Xt-1 ...
        
        # Implementation using pandas for index alignment
        
        # Optimized:
        # Create a rolling object
        # But rolling takes scalar window. Here window is len(w).
        
        # Fastest robust way:
        # Loop over windows using stride_tricks or just list comp if len < 100k
        
        transformed = []
        # Pre-calc
        w_sq = w.reshape(-1)
        
        for i in range(width, len(values)):
             # Window from i-width to i (inclusive of i if w has current term)
             # w has w0 corresponding to Xt?
             # If w is [w_last ... w_0], and slice is [X_{t-k} ... X_t]
             # Then dot product works.
             window = values[i-width : i+1]
             if len(window) != len(w_sq):
                 # Alignment edge case
                 transformed.append(np.nan)
                 continue
                 
             transformed.append(np.dot(window, w_sq))
             
        # Pad initial steps
        out = [np.nan] * width + transformed
        
        # Check lengths matches (might be off by 1 depending on range)
        diff = len(values) - len(out)
        if diff > 0:
            out += [np.nan] * diff # Should not happen if loop corrects
        elif diff < 0:
             out = out[:len(values)]
             
        return pd.Series(out, index=series.index)

    def transform(self, df: pd.DataFrame, on_cols: list) -> pd.DataFrame:
        df_new = df.copy()
        for col in on_cols:
             if col in df.columns:
                 logger.info(f"Applying FFD (d={self.d}) to {col}...")
                 df_new[col] = self.frac_diff_fixed(df[col])
        return df_new
