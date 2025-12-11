import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import logging
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects Market Regimes (Bull/Bear/Range or High/Low Volatility)
    using Unsupervised Gaussian HMM.
    """
    
    def __init__(self, n_components: int = 3, covariance_type: str = "diag", n_iter: int = 100):
        self.model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
        self.n_components = n_components
        self.is_fitted = False

    def prepare_data(self, df: pd.DataFrame):
        """
        Extracts features for HMM. 
        Usually Log Returns and Volatility (Range).
        """
        # Ensure we have log returns
        if 'log_ret' not in df.columns:
             df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
             
        # Rolling Volatility (Proxy for Variance)
        if 'volatility_24h' not in df.columns:
             df['volatility_24h'] = df['log_ret'].rolling(24).std()
             
        data = df[['log_ret', 'volatility_24h']].dropna()
        return data

    def fit(self, df: pd.DataFrame):
        """
        Fits the HMM to history.
        """
        data = self.prepare_data(df)
        X = data.values
        
        logger.info(f"Fitting HMM with {self.n_components} states on {len(X)} samples...")
        self.model.fit(X)
        self.is_fitted = True
        
        # --- ENFORCE SORTING BY VOLATILITY ---
        # Feature 1 is 'volatility_24h' (variance proxy). Feature 0 is 'log_ret'.
        # We sort states based on the MEAN of Feature 1 (Avg Volatility of that state).
        # Or better: The Variance of Feature 0 (Log Returns).
        # GaussianHMM: covars_ is shape (n_comp, n_features, n_features) or (n_comp, n_features) if diag.
        # Let's use the MEAN of the Volatility Feature (index 1).
        
        # Check feature dims
        # data cols: ['log_ret', 'volatility_24h']
        state_vol_means = self.model.means_[:, 1]
        
        # Get sorted indices (Low Vol -> High Vol)
        sorted_idx = np.argsort(state_vol_means)
        logger.info(f"Sorting States by Volatility: {sorted_idx}")
        
        # Prepare Reordered Parameters (Copying to avoid view issues)
        new_startprob = np.array(self.model.startprob_[sorted_idx])
        new_transmat = np.array(self.model.transmat_[sorted_idx, :][:, sorted_idx])
        new_means = np.array(self.model.means_[sorted_idx])
        new_covars = np.array(self.model.covars_[sorted_idx])
        
        # Atomic Assignment (Bypassing Setters to avoid validation glitch)
        self.model.startprob_ = new_startprob
        self.model.transmat_ = new_transmat
        
        # hmmlearn 0.2.8+ uses properties for means/covars
        # We set private attributes directly to avoid 'ValueError' during sort
        if hasattr(self.model, "_means_"):
            self.model._means_ = new_means
            self.model._covars_ = new_covars
        else:
            # Fallback for older versions
            self.model.means_ = new_means
            self.model.covars_ = new_covars
        
        # Analyze states (After Sort)
        for i in range(self.n_components):
            logger.info(f"State {i}: MeanVol={self.model.means_[i][1]:.6f}, VarRet={np.diag(self.model.covars_[i])[0]:.6f}")
            
        # Save
        self.save_model()

    def predict_state(self, df: pd.DataFrame) -> int:
        """
        Predicts the regime of the LATEST data point.
        """
        if not self.is_fitted:
            self.load_model()
            
        data = self.prepare_data(df)
        X = data.values
        
        # Predict sequence
        hidden_states = self.model.predict(X)
        current_state = hidden_states[-1]
        
        return current_state

    def save_model(self):
        path = config.MODELS_DIR / "hmm_regime.pkl"
        joblib.dump(self.model, path)
        logger.info(f"HMM model saved to {path}")

    def load_model(self):
        path = config.MODELS_DIR / "hmm_regime.pkl"
        if path.exists():
            self.model = joblib.load(path)
            self.is_fitted = True
            logger.info("HMM model loaded.")
        else:
            logger.warning("HMM model not found. Need to fit first.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Dummy data
    dates = pd.date_range("2023-01-01", periods=1000, freq="H")
    close = np.cumsum(np.random.randn(1000)) + 100
    df = pd.DataFrame({'close': close}, index=dates)
    
    detector = RegimeDetector()
    detector.fit(df)
    state = detector.predict_state(df)
    print(f"Current Regime: {state}")
