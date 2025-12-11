import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)

class MetaAuditor:
    """
    The Auditor (Meta-Labeling Model).
    Filters false positives from the Primary Ensemble.
    Uses XGBoost.
    """
    
    def __init__(self, model_name: str = "auditor_v1"):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        self.model_path = config.MODELS_DIR / f"{model_name}.json"
        self.is_fitted = False

    def prepare_meta_features(self, X_features: pd.DataFrame, primary_pred_prob: pd.Series, primary_pred_side: pd.Series):
        """
        Constructs the feature set for the Auditor.
        Includes:
        - Original Market Features (can be subset).
        - Primary Model's Confidence (prob).
        - Primary Model's Decision (Side).
        """
        meta_X = X_features.copy()
        meta_X['primary_conf'] = primary_pred_prob
        meta_X['primary_side'] = primary_pred_side
        return meta_X

    def fit(self, X_features: pd.DataFrame, primary_pred_prob: pd.Series, primary_pred_side: pd.Series, true_outcome: pd.Series):
        """
        Trains the Auditor.
        Args:
            X_features: Original features.
            primary_pred_prob: Confidence from Ensemble (0 to 1).
            primary_pred_side: Direction (-1 or 1).
            true_outcome: 1 if trade was profitable, 0 otherwise.
        """
        logger.info("Training Meta-Auditor (XGBoost)...")
        
        meta_X = self.prepare_meta_features(X_features, primary_pred_prob, primary_pred_side)
        y = true_outcome
        
        self.model.fit(meta_X, y)
        self.is_fitted = True
        
        # Save
        self.model.save_model(str(self.model_path))
        logger.info(f"Auditor saved to {self.model_path}")

    def predict_veto(self, X_features: pd.DataFrame, primary_pred_prob: pd.Series, primary_pred_side: pd.Series, threshold: float = 0.6) -> pd.Series:
        """
        Returns boolean mask: True if approved, False if vetoed.
        """
        if not self.is_fitted:
            if self.model_path.exists():
                self.model.load_model(str(self.model_path))
                self.is_fitted = True
            else:
                logger.warning("Auditor not fitted. Approving all signals (Pass-Through).")
                return pd.Series([True] * len(X_features), index=X_features.index)
        
        meta_X = self.prepare_meta_features(X_features, primary_pred_prob, primary_pred_side)
        
        # Probability that the primary trade is CORRECT
        prob_correct = self.model.predict_proba(meta_X)[:, 1]
        
        # Approve if probability of correctness > threshold
        decisions = prob_correct > threshold
        return pd.Series(decisions, index=X_features.index)

if __name__ == "__main__":
    # Smoke Test
    logging.basicConfig(level=logging.INFO)
    N = 100
    feats = pd.DataFrame(np.random.randn(N, 5), columns=[f'f{i}' for i in range(5)])
    p_prob = pd.Series(np.random.rand(N))
    p_side = pd.Series(np.random.choice([-1, 1], N))
    outcome = pd.Series(np.random.randint(0, 2, N))
    
    auditor = MetaAuditor()
    auditor.fit(feats, p_prob, p_side, outcome)
    decisions = auditor.predict_veto(feats, p_prob, p_side)
    print(decisions.value_counts())
