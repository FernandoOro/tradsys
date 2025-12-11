import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AdversarialValidator:
    """
    Performs Adversarial Validation to detect Covariate Shift between Train and Test sets.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
        
    def check_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list = None) -> float:
        """
        Returns ROC-AUC score.
        AUC ~ 0.5: No Drift (Good).
        AUC > 0.7: Drift Detected (Bad).
        """
        logger.info("Running Adversarial Validation...")
        
        # Select numeric columns if None provided
        if features is None:
            features = train_df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove targets if present
            features = [f for f in features if f not in ['target', 'label', 'open', 'high', 'low', 'close', 'volume']]
            
        logger.info(f"Using features: {features}")
        
        # Prepare Data
        X_train = train_df[features].copy()
        X_test = test_df[features].copy()
        
        # Assign Labels: Train=0, Test=1
        X_train['adv_target'] = 0
        X_test['adv_target'] = 1
        
        # Combine
        combined = pd.concat([X_train, X_test], axis=0).sample(frac=1.0, random_state=42)
        X = combined.drop(columns=['adv_target'])
        y = combined['adv_target']
        
        # Train/Val split for the Adversarial Model
        # (We use a simple split here just to get the metric)
        X_adv_train, X_adv_val, y_adv_train, y_adv_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Classifier
        self.model.fit(X_adv_train, y_adv_train)
        
        # Predict
        probs = self.model.predict_proba(X_adv_val)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y_adv_val, probs)
        
        logger.info(f"Adversarial AUC: {auc:.4f}")
        
        if auc > 0.7:
            logger.warning(f"⚠️ HIGH DRIFT DETECTED (AUC={auc:.2f}). Test set is significantly different from Train set.")
            logger.warning("Feature distributions have shifted. Proceed with CAUTION.")
        else:
            logger.info("✅ Data distribution looks stable (No significant drift).")
            
        return auc
