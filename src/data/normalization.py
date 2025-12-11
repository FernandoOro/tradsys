import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Normalizer:
    """
    Handles data normalization strictly adhering to 'No Look-Ahead Bias'.
    """
    
    def __init__(self, window_size: int = 200, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon

    def rolling_z_score(self, series: pd.Series) -> pd.Series:
        """
        Computes Dynamic Z-Score: (x - roll_mean) / (roll_std + epsilon)
        Uses only PAST data (window) for statistics.
        """
        roll_mean = series.rolling(window=self.window_size).mean()
        roll_std = series.rolling(window=self.window_size).std()
        
        z_score = (series - roll_mean) / (roll_std + self.epsilon)
        return z_score

    def apply_normalization(self, df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
        """
        Applies rolling z-score to all numerical columns except excluded ones.
        """
        if exclude_cols is None:
            exclude_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'target', 'open', 'high', 'low', 'close', 'volume']
            # We exclude raw price columns usually if we work with log-returns or differences. 
            # However, if the model needs price inputs normalized, valid. 
            # Context says: "Input: Ventana de precios + Time Embeddings". 
            # Typically Transformer inputs are normalized.
            # Let's normalize indicators + features but NOT time embeddings (already -1 to 1).
            # Log returns are already stationary-ish but z-score makes them unit variance.
        
        normalized_df = df.copy()
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            # Apply Rolling Z-Score
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
               normalized_df[col] = self.rolling_z_score(df[col])
        
        # After rolling, we have NaNs for the first 'window_size' rows
        initial_len = len(normalized_df)
        normalized_df = normalized_df.dropna()
        dropped = initial_len - len(normalized_df)
        logger.info(f"Normalization Warm-up: Dropped {dropped} rows.")
        
        return normalized_df
        
    def fractional_differencing(self, df: pd.DataFrame, d: float = 0.4) -> pd.DataFrame:
        """
        Apply Fractional Differencing (FFD) to preserve memory while achieving stationarity.
        Simplified implementation for now as placeholder for full FFD logic provided in libraries.
        For this phase, assuming we stick to Rolling Z-Score on Log Returns as primary.
        This method is a stub/placeholder for future expansion or if we add 'statsmodels' logic here.
        """
        # TODO: Implement full FFD with weights preservation if needed.
        # Check 'statsmodels' usage in requirements -> it's there.
        # But for 'features.py' we used log-returns. 
        # Leaving as pass or basic diff for now to ensure pipeline flows.
        return df

