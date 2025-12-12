from src.data.pipeline import DataPipeline
from src.config import config
import pickle
import pandas as pd
import logging
import numpy as np # Import numpy just in case

logger = logging.getLogger(__name__)

class ContrastiveDataPipeline(DataPipeline):
    """
    Pipeline for Contrastive Agent (1H).
    Overrides storage paths to avoid conflict with Agent 1.
    """
    def split_fit_transform_save(self, df: pd.DataFrame, time_cols: list):
        """
        Overrides to save with _1h suffix.
        """
        n = len(df)
        train_end = int(n * 0.60)
        val_start = int(n * 0.65)
        val_end = int(n * 0.85)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        vault_df = df.iloc[val_end:]

        # Reuse same logic for features
        non_feature_cols = time_cols + ['open', 'high', 'low', 'close', 'volume', 'target']
        # Also any string cols if any?
        
        feature_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in non_feature_cols]

        logger.info(f"Step 4 (Contrastive 1H): PCA Fitness on {len(feature_cols)} features...")

        # Fit PCA
        self.pca.fit(train_df[feature_cols])

        # Save PCA Model (Isolated)
        pca_path = config.MODELS_DIR / "pca_1h.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(self.pca, f)
        logger.info(f"Saved PCA model to {pca_path}")

        # Transform
        def apply_pca(d_subset):
            if d_subset.empty: return d_subset
            # Transform features
            d_pca = self.pca.transform(d_subset[feature_cols])
            
            # d_pca is usually a DataFrame if PCA returns one, or numpy array. 
            # Looking at src/data/reduction.py (PCAFeatureReducer), it likely returns DataFrame?
            # src/data/pipeline.py line 111 implies it returns something that can be concatted.
            # Assuming PCAFeatureReducer returns DataFrame with index or we reset it.
            # Let's verify pipeline.py logic:
            # d_preserved = d_subset[non_feature_cols].copy()
            # return pd.concat([d_preserved, d_pca], axis=1)
            
            # Note: If d_pca has no index, concat with axis=1 aligns by position (dangerous if gap).
            # We should ensure alignment.
            if not isinstance(d_pca, pd.DataFrame):
                d_pca = pd.DataFrame(d_pca, index=d_subset.index)
            else:
                d_pca.index = d_subset.index # Force alignment
                
            d_preserved = d_subset[non_feature_cols].copy()
            return pd.concat([d_preserved, d_pca], axis=1)

        train_pca = apply_pca(train_df)
        val_pca = apply_pca(val_df)
        vault_pca = apply_pca(vault_df)

        # Save (Isolated)
        train_pca.to_parquet(config.PROCESSED_DATA_DIR / "train_1h.parquet")
        val_pca.to_parquet(config.PROCESSED_DATA_DIR / "val_1h.parquet")
        vault_pca.to_parquet(config.PROCESSED_DATA_DIR / "vault_1h.parquet")

        logger.info(f"Datasets saved to {config.PROCESSED_DATA_DIR} with suffix '_1h'")
