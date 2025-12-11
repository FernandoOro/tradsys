import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class PCAFeatureReducer:
    """
    Applies Principal Component Analysis (PCA) to orthogonalize features
    and remove multicollinearity (noise).
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.pca = PCA(n_components=variance_threshold)
        
    def fit(self, df: pd.DataFrame):
        """
        Fits PCA on the dataframe (training set).
        Expects ONLY numeric feature columns (no target, no time).
        """
        self.pca.fit(df)
        logger.info(f"PCA Fitted. Components to keep: {self.pca.n_components_}")
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe into Principal Components.
        """
        data_pca = self.pca.transform(df)
        # Create new DF
        cols = [f"PC_{i}" for i in range(data_pca.shape[1])]
        df_pca = pd.DataFrame(data_pca, columns=cols, index=df.index)
        return df_pca
