import numpy as np
import pandas as pd
from datetime import timedelta

# =========================================================
# Sliding Window Feature Extraction
# =========================================================

def sliding_window_features(df, window_size=10, step_size=5):
    """
    Converts raw timestamped sensor data into window-level ML features.

    df: raw dataframe containing speed, accel, vibration, gps, labels
    window_size: size of each window in seconds
    step_size: how much the window slides each step
    """

    df = df.sort_values("ts").reset_index(drop=True)
    features = []
    n = len(df)
    i = 0

    while i < n:
        t0 = df.loc[i, "ts"]
        t1 = t0 + timedelta(seconds=window_size)

        mask = (df["ts"] >= t0) & (df["ts"] < t1)
        w = df[mask]

        if len(w) == 0:
            i += 1
            continue

        # --- SPEED FEATURES ---
        speed_mean = w["speed"].mean()
        speed_std  = w["speed"].std()

        # --- ACCELERATION FEATURES ---
        accel_mag = np.sqrt(w["accel_x"]**2 + w["accel_y"]**2)
        accel_mean = accel_mag.mean()
        accel_std  = accel_mag.std()

        # --- VIBRATION ---
        vib_mean = w["vibration"].mean()

        # --- ENGINE RATIO ---
        engine_ratio = w["engine_on"].mean()

        # --- GPS MOVEMENT ---
        gps_jump = np.sqrt(
            (np.diff(w["lat"]) * 111320)**2 +
            (np.diff(w["lon"]) * 111320)**2
        )
        gps_jump_mean = gps_jump.mean() if len(gps_jump) > 0 else 0

        # --- LABEL (any theft in this window = theft) ---
        label = 1 if w["label"].sum() > 0 else 0

        features.append({
            "t_mid": t0 + timedelta(seconds=window_size / 2),
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "accel_mean": accel_mean,
            "accel_std": accel_std,
            "vib_mean": vib_mean,
            "engine_ratio": engine_ratio,
            "gps_jump_mean": gps_jump_mean,
            "label": label
        })

        i += step_size

    return pd.DataFrame(features)


# =========================================================
# Process Entire Raw Dataset
# =========================================================

def process_raw_dataset(df, window_size=10, step_size=5):
    """
    Takes entire raw dataset (possibly multiple trips) and extracts ML features.
    """
    return sliding_window_features(df, window_size, step_size)
