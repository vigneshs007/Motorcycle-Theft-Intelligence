import os
import pandas as pd
import joblib

# =========================================================
# File & Path Utilities
# =========================================================

def ensure_dir(path: str):
    """
    Creates directory if not exists.
    """
    os.makedirs(path, exist_ok=True)


def save_csv(df, path: str):
    """
    Saves a dataframe to a CSV file.
    Automatically creates folders.
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def load_csv(path: str):
    """
    Loads a CSV file into a DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# =========================================================
# Model Save / Load Helpers
# =========================================================

def save_model(model, path: str):
    """
    Saves a model or scaler using joblib.
    """
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)


def load_model(path: str):
    """
    Loads a model/scaler.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# =========================================================
# General Helpers
# =========================================================

def describe_df(df):
    """
    Quick summary of any DataFrame.
    """
    print("\nRows:", len(df))
    print("\nColumns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nMissing Values:\n", df.isnull().sum())
