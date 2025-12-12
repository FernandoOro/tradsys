
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.config import config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.normalization import Normalizer
from src.data.labeling import Labeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_features():
    logger.info("🚀 Starting Feature Importance Analysis (XGBoost)...")
    
    # 1. Fetch & Prepare Data (Replicating Pipeline logic manually to avoid PCA)
    loader = DataLoader()
    engineer = FeatureEngineer()
    labeler = Labeler(barrier_width=1.5, time_horizon=24) # TBM
    normalizer = Normalizer(window_size=100)
    
    # Fetch
    logger.info("Fetching Market Context...")
    df = loader.fetch_market_context(config.SYMBOL, config.TIMEFRAME, limit=50000)
    
    # Engineer
    logger.info("Engineering Features...")
    df = engineer.run(df)
    
    # Label (Target)
    logger.info("Generating Triple Barrier Target...")
    df = labeler.run(df)
    
    # Prep Target: Analysis focuses on "Long Opportunities" (Class 1)
    df['target'] = np.where(df['target_class'] == 1, 1, 0)
    df.dropna(inplace=True)
    
    # Normalize
    # Note: XGBoost doesn't strictly need normalization, but we do it to match the Agent's view
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    # CRITICAL FIX: Exclude 'target_ret' (Leakage) and 'sample_weight'
    exclude = time_cols + ['open', 'high', 'low', 'close', 'volume', 'target', 'target_class', 'target_ret', 'sample_weight']
    df_norm = normalizer.apply_normalization(df, exclude_cols=exclude)
    
    # Select Features
    feature_cols = [c for c in df_norm.columns if c not in exclude and c not in ['timestamp', 'symbol']]
    X = df_norm[feature_cols]
    y = df_norm['target']
    
    logger.info(f"Analyzing {len(feature_cols)} features with {len(X)} samples.")
    
    # 2. Train XGBoost
    # Use simple params, we just want importance
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=42
    )
    
    model.fit(X, y)
    
    # 3. Extract Importance
    importance = model.feature_importances_
    results = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # 4. Save & Report
    out_path = config.PROCESSED_DATA_DIR / "feature_importance.csv"
    results.to_csv(out_path, index=False)
    
    logger.info(f"\n🏆 Top 20 Features:\n{results.head(20)}")
    logger.info(f"\n🗑️ Bottom 20 Features (Candidates for Removal):\n{results.tail(20)}")
    logger.info(f"Full report saved to {out_path}")
    
    # Suggest Threshold
    # e.g., Features with < 0.005 importance
    noise = results[results['Importance'] < 0.005]
    logger.info(f"Found {len(noise)} features with < 0.5% importance.")

if __name__ == "__main__":
    analyze_features()
