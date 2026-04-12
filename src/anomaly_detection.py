import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class IsolationForestDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

    def predict_scores(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return scores

    def predict_labels(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
