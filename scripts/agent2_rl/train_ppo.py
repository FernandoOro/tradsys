
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports from src
from src.agent2_rl.env import TradingEnv
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_rl():
    logger.info("🚀 Starting Deep RL Training (Agent 2.1 - PPO)...")
    
    # 1. Load Data
    data_path = config.PROCESSED_DATA_DIR / "train_reversion.parquet"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded Data: {df.shape}")
    
    # 2. Split (Train/Val) - RL needs a continuous timeline
    # We'll use the first 80% for training
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:]
    
    # 3. Initialize Environments
    logger.info("Initializing Environments...")
    
    # Training Env (Wrapped for Normalization)
    env_train = DummyVecEnv([lambda: TradingEnv(df_train)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Validation Env (Validation data, reusing statistics from training)
    # We separate validation environment to monitor performance on unseen data
    env_val = DummyVecEnv([lambda: TradingEnv(df_val)])
    env_val = VecNormalize(env_val, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    # Important: Sync normalization stats from train to val
    env_val.obs_rms = env_train.obs_rms
    
    # 4. Setup Callbacks
    eval_callback = EvalCallback(
        env_val, 
        best_model_save_path=str(config.MODELS_DIR / "agent2_rl_best"),
        log_path=str(config.LOGS_DIR / "rl_tensorboard"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # 5. Initialize PPO Agent
    # Policy: MlpPolicy (Dense Layers)
    # Hyperparameters: Tuned for stability
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage exploration
        tensorboard_log=str(config.LOGS_DIR / "rl_tensorboard")
    )
    
    # 6. Train
    TIMESTEPS = 1000000 # 1 Million Candles
    logger.info(f"Training for {TIMESTEPS} steps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, progress_bar=True)
    
    # 7. Save Final Model & Normalization Stats
    model.save(config.MODELS_DIR / "agent2_rl_final")
    env_train.save(str(config.MODELS_DIR / "agent2_rl_vecnormalize.pkl"))
    logger.info("✅ Training Complete. Model & Stats Saved.")

if __name__ == "__main__":
    train_rl()
