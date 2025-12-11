
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.inference.predictor import Predictor
from src.backtesting.simulator import Simulator
from src.models.regime.hmm import RegimeDetector
from src.models.ensemble.meta_labeling import MetaAuditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Comparator")

# === EMBEDDED UTILS (Zero-Copy Dataset) ===
class LazySequenceDataset(Dataset):
    def __init__(self, features, time_features, targets, seq_len):
        self.features = torch.FloatTensor(features)
        self.time_features = torch.FloatTensor(time_features)
        t_dir = np.where(targets > 0.5, 1.0, -1.0)
        t_conf = np.ones_like(t_dir)
        self.targets = torch.FloatTensor(np.stack([t_dir, t_conf], axis=1))
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        t = self.time_features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, t, y
# ==========================================

def run_comparison():
    val_path = config.PROCESSED_DATA_DIR / "val.parquet"
    if not val_path.exists():
        logger.error("Validation data missing.")
        return

    logger.info(f"Loading Val Data: {val_path}")
    df_val = pd.read_parquet(val_path)
    
    # --- 1. PREPARE DATA ---
    non_feat = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', 'datetime', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    feature_cols = [c for c in df_val.columns if c not in non_feat and 'sample_weight' not in c]
    pca_cols = [c for c in feature_cols if c.startswith('PC_')]
    
    if pca_cols:
        final_feats = pca_cols
    else:
        final_feats = feature_cols
    
    logger.info(f"Using Features: {final_feats}")
    
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = df_val[final_feats].values.astype(np.float32)
    time_features = df_val[time_cols].values.astype(np.float32)
    targets = df_val['target'].values
    
    features = np.nan_to_num(features)
    time_features = np.nan_to_num(time_features)
    
    seq_len = 10
    dataset = LazySequenceDataset(features, time_features, targets, seq_len)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)
    
    # --- 2. RUN BASE INFERENCE (Agent 1) ---
    predictor = Predictor(model_name="agent1")
    all_probs = []
    
    logger.info("Running Agent 1 Inference...")
    for batch_x, batch_t, _ in tqdm(loader):
        batch_x_np = batch_x.numpy()
        batch_t_np = batch_t.numpy()
        preds = predictor.predict(batch_x_np, batch_t_np)
        all_probs.extend(preds['direction'])
    
    valid_df = df_val.iloc[seq_len:].copy()
    valid_df['pred_score'] = np.array(all_probs)
    
    # --- 3. RUN HMM ---
    hmm_path = config.MODELS_DIR / "hmm_regime.pkl"
    if hmm_path.exists():
        hmm_model = joblib.load(hmm_path)
        rd = RegimeDetector(n_components=3)
        rd.model = hmm_model
        rd.is_fitted = True
        
        # Prepare Data manually aligned
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We must run it on the FULL df_val to get correct rolling windows, then slice
            hmm_data = rd.prepare_data(df_val)
            states = rd.model.predict(hmm_data.values)
            s_series = pd.Series(states, index=hmm_data.index)
            valid_df['regime'] = s_series.reindex(valid_df.index).fillna(-1)
    else:
        valid_df['regime'] = 1
        
    # --- 4. RUN AUDITOR ---
    auditor = MetaAuditor("auditor_v1")
    # Prepare features for Auditor (Must match training!)
    # Training used: X_features (PC_0...PC_7) + primary_conf + primary_side
    # Let's verify what X_features are in valid_df
    X_audit = valid_df[final_feats].copy()
    p_prob = valid_df['pred_score'] # This is technically direction (-1 to 1). 
    # WAIT. The Auditor in train.py was fed 'prob' (0-1) or 'conf' (0-1)?
    # In train.py: meta_X = ... 'primary_conf', 'primary_side'
    # Here pred_score is output of Tanh (-1 to 1).
    # We need confidence. Predictor returns 'confidence' too but we only grabbed 'direction'.
    # For now, let's treat abs(pred_score) as confidence proxy?
    # No, let's re-run inference correctly? Or just Hack it for the test?
    # Predictor returns {'direction': ..., 'confidence': ...}
    # My Lazy Loop above only grabbed 'direction'.
    # FIX: Need confidence.
    
    # Redo inference loop? No, that takes time.
    # Let's assume Conf = Abs(Dir) if we lack the sigmoid head. 
    # Actually train.py Tanh head output logic:
    # forward() returns x_dir, x_conf.
    # predictor.predict returns BOTH.
    pass

    # Quickfix to get confidence without re-running loop:
    # I need to modify the loop above to store confidence. 
    # For this script generation, I'll rewrite the loop.

    # --- 2b. RE-RUN INFERENCE WITH CONFIDENCE ---
    all_dirs = []
    all_confs = []
    
    logger.info("Running Inference (Dir + Conf)...")
    for batch_x, batch_t, _ in tqdm(loader):
        batch_x_np = batch_x.numpy()
        batch_t_np = batch_t.numpy()
        preds = predictor.predict(batch_x_np, batch_t_np)
        all_dirs.extend(preds['direction'])
        all_confs.extend(preds['confidence'])
        
    valid_df.loc[:, 'pred_score'] = np.array(all_dirs)
    valid_df.loc[:, 'confidence'] = np.array(all_confs)
    
    # Auditor Prediction
    # p_side = sign(pred_score)
    p_side = np.sign(valid_df['pred_score'])
    p_conf = valid_df['confidence']
    
    # X_features Must match training features exactly (PC_0..PC_7)
    X_audit = valid_df[final_feats]
    
    # Get Veto Decisions
    # Threshold 0.5 for Auditor (Standard)
    veto_mask = auditor.predict_veto(X_audit, p_conf, p_side, threshold=0.6)
    valid_df['auditor_approved'] = veto_mask
    
    # --- 5. DEFINE STRATEGIES ---
    
    stats_list = []
    sim = Simulator(fees=0.001, slippage=0.0005)
    
    # A. SNIPER (Baseline)
    # Threshold 0.95, Block Regime 0, Auditor ignored
    s1 = valid_df.copy()
    cond_buy_a = (s1['pred_score'] > 0.95) & (s1['regime'] != 0)
    cond_sell_a = (s1['pred_score'] < 0.0)
    s1['signal'] = np.select([cond_buy_a, cond_sell_a], [1, -1], 0)
    
    pf1, m1 = sim.run_backtest(s1['close'], s1['signal'])
    stats_list.append({"Name": "Sniper (Base)", "Ret %": m1['Total Return [%]'], "Sharpe": m1['Sharpe Ratio'], "Trades": m1['Total Trades'], "Win Rate": m1['Win Rate [%]']})
    
    # B. AUDITED AGGRESSION (Challenger)
    # Threshold 0.75, Block Regime 0, Auditor MUST Approve
    s2 = valid_df.copy()
    cond_buy_b = (s2['pred_score'] > 0.75) & (s2['regime'] != 0) & (s2['auditor_approved'] == True)
    cond_sell_b = (s2['pred_score'] < 0.0) # Exit same as before
    s2['signal'] = np.select([cond_buy_b, cond_sell_b], [1, -1], 0)
    
    pf2, m2 = sim.run_backtest(s2['close'], s2['signal'])
    stats_list.append({"Name": "Audited (New)", "Ret %": m2['Total Return [%]'], "Sharpe": m2['Sharpe Ratio'], "Trades": m2['Total Trades'], "Win Rate": m2['Win Rate [%]']})
    
    # C. RECKLESS (Control)
    # Threshold 0.75, Block Regime 0, NO Auditor (To prove Auditor value)
    s3 = valid_df.copy()
    cond_buy_c = (s3['pred_score'] > 0.75) & (s3['regime'] != 0)
    cond_sell_c = (s3['pred_score'] < 0.0)
    s3['signal'] = np.select([cond_buy_c, cond_sell_c], [1, -1], 0)
    
    pf3, m3 = sim.run_backtest(s3['close'], s3['signal'])
    stats_list.append({"Name": "Reckless (No Audit)", "Ret %": m3['Total Return [%]'], "Sharpe": m3['Sharpe Ratio'], "Trades": m3['Total Trades'], "Win Rate": m3['Win Rate [%]']})
    
    # --- 6. REPORT ---
    results = pd.DataFrame(stats_list)
    print("\n" + "="*60)
    print("      ⚔️  STRATEGY SHOWDOWN  ⚔️      ")
    print("="*60)
    print(results.to_string(index=False))
    print("="*60 + "\n")

if __name__ == "__main__":
    run_comparison()
