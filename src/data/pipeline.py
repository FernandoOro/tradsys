import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

from src.config import config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.normalization import Normalizer
from src.data.reduction import PCAFeatureReducer

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Orchestrates the entire ETL process: 
    Raw + Context -> Features -> Normalize -> PCA -> Split -> Save
    """
    
    def __init__(self):
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.normalizer = Normalizer(window_size=100)
        self.pca = PCAFeatureReducer(variance_threshold=0.95)

    def run_pipeline(self, symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME, limit: int = 5000):
        # 1. Fetch with Context
        logger.info("Step 1: Fetching Market Context...")
        df = self.loader.fetch_market_context(symbol, timeframe, context_symbols=['BTC/USDT', 'ETH/USDT'], limit=limit)
        
        if df.empty:
            logger.error("No data fetched.")
            return

        # 2. Features
        logger.info("Step 2: Engineering Features...")
        df = self.engineer.run(df)
        
        # 3. Normalization
        logger.info("Step 3: Normalizing...")
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        exclude = time_cols + ['open', 'high', 'low', 'close', 'volume', 'target'] # Removed datetime
        
        # 3b. Generate Target (Simple Next Step Direction)
        # 1 = Up, 0 = Down/Flat
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        df.dropna(inplace=True) # Drop last row which has no target (though shift(-1) leaves one NaN)
        
        # Apply normalization to all numeric except exclusions
        df_norm = self.normalizer.apply_normalization(df, exclude_cols=exclude)

        # 4. PCA (Orthogonalization)
        # We need to fit PCA only on TRAIN set to avoid leakage, but here we do it before splitting for simplicity in this skeleton?
        # NO. Must split FIRST, then Fit PCA on Train, then Transform all.
        # Let's split first.
        
        self.split_fit_transform_save(df_norm, time_cols)

    def split_fit_transform_save(self, df: pd.DataFrame, time_cols: list):
        """
        Splits data, Fits PCA on Train, Transforms all, Saves.
        """
        n = len(df)
        train_end = int(n * 0.60)
        val_start = int(n * 0.65)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        vault_df = df.iloc[val_end:]
        
        # Identify Feature Columns for PCA (Exclude Time, Price, etc)
        # We want to PCA the indicators.
        # Identify Feature Columns for PCA (Exclude Time, Price, etc)
        # We want to PCA the indicators.
        non_feature_cols = time_cols + ['open', 'high', 'low', 'close', 'volume', 'target'] # Removed datetime
        # Also any string cols if any?
        # Select numeric
        feature_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in non_feature_cols]
        
        logger.info(f"Step 4: PCA Fitness on {len(feature_cols)} features...")
        
        # Fit PCA
        self.pca.fit(train_df[feature_cols])
        
        # Save PCA Model
        pca_path = config.MODELS_DIR / "pca.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(self.pca, f)
        logger.info(f"Saved PCA model to {pca_path}")
        
        # Transform
        def apply_pca(d_subset):
            if d_subset.empty: return d_subset
            # Transform features
            d_pca = self.pca.transform(d_subset[feature_cols])
            # Concatenate back with preserved cols (Time, Price, Target)
            d_preserved = d_subset[non_feature_cols].copy()
            # Reset index to allow concat if needed, but index alignment is better
            # PCA transform returns new DF with same index
            return pd.concat([d_preserved, d_pca], axis=1)

        train_pca = apply_pca(train_df)
        val_pca = apply_pca(val_df)
        vault_pca = apply_pca(vault_df)
        
        # Save
        train_pca.to_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
        val_pca.to_parquet(config.PROCESSED_DATA_DIR / "val.parquet")
        vault_pca.to_parquet(config.PROCESSED_DATA_DIR / "vault.parquet")
        
        logger.info(f"Datasets saved to {config.PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DataPipeline()
    pipeline.run_pipeline()
