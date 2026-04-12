import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# =========================================================
# Supervised Theft Classification Model (Random Forest)
# =========================================================

class TheftClassifier:
    def __init__(self, n_estimators=200, random_state=42):
        """
        Wrapper for RandomForestClassifier with built-in scaling.
        """
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced"
        )
        self.is_fitted = False

    # -----------------------------------------------------
    # Fit model on training data
    # -----------------------------------------------------
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

    # -----------------------------------------------------
    # Predict probabilities
    # -----------------------------------------------------
    def predict_proba(self, X):
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    # -----------------------------------------------------
    # Predict labels
    # -----------------------------------------------------
    def predict(self, X, threshold=0.5):
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    # -----------------------------------------------------
    # Save model + scaler
    # -----------------------------------------------------
    def save(self, path_model="models/random_forest.pkl",
                   path_scaler="models/scaler.pkl"):
        joblib.dump(self.model, path_model)
        joblib.dump(self.scaler, path_scaler)

    # -----------------------------------------------------
    # Load saved model + scaler
    # -----------------------------------------------------
    def load(self, path_model="models/random_forest.pkl",
                  path_scaler="models/scaler.pkl"):
        self.model = joblib.load(path_model)
        self.scaler = joblib.load(path_scaler)
        self.is_fitted = True

    # -----------------------------------------------------
    # Internal check
    # -----------------------------------------------------
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


# =========================================================
# Helper training function for notebooks
# =========================================================

def train_random_forest(df_features, save_model=True):
    """
    df_features must include all engineered features + 'label'.
    """

    # Separate X and y
    X = df_features.drop(columns=["label"])
    y = df_features["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Train model
    clf = TheftClassifier()
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    # Save
    if save_model:
        clf.save()

    return clf, X_test, y_test


    def save(self, path_model="../models/random_forest.pkl",
                   path_scaler="../models/scaler.pkl"):
        import os

        # Ensure directories exist
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        os.makedirs(os.path.dirname(path_scaler), exist_ok=True)

        # Save model + scaler
        joblib.dump(self.model, path_model)
        joblib.dump(self.scaler, path_scaler)

