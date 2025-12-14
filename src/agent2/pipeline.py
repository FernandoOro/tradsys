
from src.data.pipeline import DataPipeline
from src.config import config
import pickle
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ReversionDataPipeline(DataPipeline):
    """
    Pipeline for Agent 2 (Mean Reversion).
    Overrides labeling and storage paths.
    """
    
    def run_pipeline(self, symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME, limit: int = 100000):
        # 1. Fetch with Context (Same as base)
        logger.info("Step 1 (Agent 2): Fetching Market Context...")
        df = self.loader.fetch_market_context(symbol, timeframe, context_symbols=['BTC/USDT', 'ETH/USDT'], limit=limit)
        
        if df.empty:
            logger.error("No data fetched.")
            return

        # 2. Features (Same as base, includes BB width now)
        logger.info("Step 2 (Agent 2): Engineering Features...")
        df = self.engineer.run(df)
        
        # 3. Normalization (Same as base)
        logger.info("Step 3 (Agent 2): Normalizing...")
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        # Exclude 'target_reversion' instead of 'target'
        exclude = time_cols + ['open', 'high', 'low', 'close', 'volume', 'target', 'target_reversion'] 
        
        # 3b. Generate Target (REVERSION LOGIC)
        logger.info("Step 2b (Agent 2): Generating Reversion Labels (Ping Pong)...")
        from src.data.labeling import Labeler
        labeler = Labeler(barrier_width=1.5, time_horizon=24) 
        
        # USE NEW METHOD
        df = labeler.add_reversion_targets(df, horizon=24)
        
        # We don't need 'target' column from TBM, we use 'target_reversion'
        # Drop rows where target_reversion is NaN?
        # Reversion Logic fills 0s by default, but FixedForwardWindowIndexer might leave NaNs at end.
        df.dropna(inplace=True)
        
        # Apply normalization
        # Note: exclude_cols must match columns present. 'target' might not exist if we didn't run labeler.run
        if 'target' in df.columns:
            exclude.append('target')
            
        df_norm = self.normalizer.apply_normalization(df, exclude_cols=[c for c in exclude if c in df.columns])

        # 4. PCA & Split & Save
        self.split_fit_transform_save(df_norm, time_cols)

    def split_fit_transform_save(self, df: pd.DataFrame, time_cols: list):
        """
        Overrides to save with _reversion suffix and handle target_reversion.
        """
        n = len(df)
        train_end = int(n * 0.60)
        val_start = int(n * 0.65)
        val_end = int(n * 0.85)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        vault_df = df.iloc[val_end:]

        # Identify Feature Columns
        non_feature_cols = time_cols + ['open', 'high', 'low', 'close', 'volume', 'target', 'target_reversion']
        
        feature_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in non_feature_cols]

        logger.info(f"Step 4 (Agent 2): PCA Fitness on {len(feature_cols)} features...")

        # Fit PCA
        self.pca.fit(train_df[feature_cols])

        # Save PCA Model (Isolated)
        pca_path = config.MODELS_DIR / "pca_reversion.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(self.pca, f)
        logger.info(f"Saved PCA model to {pca_path}")

        # Transform
        def apply_pca(d_subset):
            if d_subset.empty: return d_subset
            d_pca = self.pca.transform(d_subset[feature_cols])
            
            if not isinstance(d_pca, pd.DataFrame):
                d_pca = pd.DataFrame(d_pca, index=d_subset.index)
            else:
                d_pca.index = d_subset.index 
                
            # Preserve target_reversion!
            cols_to_preserve = [c for c in non_feature_cols if c in d_subset.columns]
            d_preserved = d_subset[cols_to_preserve].copy()
            
            return pd.concat([d_preserved, d_pca], axis=1)

        train_pca = apply_pca(train_df)
        val_pca = apply_pca(val_df)
        vault_pca = apply_pca(vault_df)

        # Save with _reversion suffix
        train_pca.to_parquet(config.PROCESSED_DATA_DIR / "train_reversion.parquet")
        val_pca.to_parquet(config.PROCESSED_DATA_DIR / "val_reversion.parquet")
        vault_pca.to_parquet(config.PROCESSED_DATA_DIR / "vault_reversion.parquet")

        logger.info(f"Datasets saved to {config.PROCESSED_DATA_DIR} with suffix '_reversion'")
