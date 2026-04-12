import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# ============================
# Helper — Ensure Directory Exists
# ============================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================
# Paths
# ============================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "processed", "features.csv")

VIS_SPEED = os.path.join(ROOT, "visuals", "speed_plots")
VIS_FEATURE = os.path.join(ROOT, "visuals", "feature_plots")
VIS_ANOMALY = os.path.join(ROOT, "visuals", "anomaly_plots")

ensure_dir(VIS_SPEED)
ensure_dir(VIS_FEATURE)
ensure_dir(VIS_ANOMALY)


# ============================
# Load Dataset
# ============================
print(f"Loading dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)


# ============================
# Plot 1 — Speed Histogram
# ============================
plt.figure(figsize=(7, 4))
sns.histplot(df["speed_mean"], kde=True)
plt.title("Speed Mean Distribution")
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(VIS_SPEED, "speed_mean_hist.png"))
plt.close()


# ============================
# Plot 2 — Acceleration Histogram
# ============================
plt.figure(figsize=(7, 4))
sns.histplot(df["accel_mean"], kde=True, color="green")
plt.title("Acceleration Mean Distribution")
plt.savefig(os.path.join(VIS_FEATURE, "accel_mean_hist.png"))
plt.close()


# ============================
# Plot 3 — Vibration Histogram
# ============================
plt.figure(figsize=(7, 4))
sns.histplot(df["vib_mean"], kde=True, color="orange")
plt.title("Vibration Mean Distribution")
plt.savefig(os.path.join(VIS_FEATURE, "vib_mean_hist.png"))
plt.close()


# ============================
# Plot 4 — GPS Jump Histogram
# ============================
if "gps_jump_mean" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(df["gps_jump_mean"], kde=True, color="purple")
    plt.title("GPS Jump Mean Distribution")
    plt.savefig(os.path.join(VIS_FEATURE, "gps_jump_mean_hist.png"))
    plt.close()


# ============================
# Plot 5 — Pairplot (feature relationships)
# ============================
feature_cols = ["speed_mean", "accel_mean", "vib_mean", "gps_jump_mean"]
plt.figure()
sns.pairplot(df[feature_cols])
plt.savefig(os.path.join(VIS_FEATURE, "pairplot_features.png"))
plt.close()


# ============================
# Plot 6 — Correlation Heatmap
# ============================
plt.figure(figsize=(8, 6))
sns.heatmap(df[feature_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(VIS_FEATURE, "correlation_heatmap.png"))
plt.close()


# ============================
# Plot 7 — Anomaly Scatter (if anomaly labels exist)
# ============================
if "anomaly" in df.columns:
    plt.figure(figsize=(7, 5))
    normal = df[df["anomaly"] == 0]
    anomaly = df[df["anomaly"] == 1]

    plt.scatter(normal["speed_mean"], normal["accel_mean"], s=10, alpha=0.5, label="Normal")
    plt.scatter(anomaly["speed_mean"], anomaly["accel_mean"], s=20, alpha=0.9, label="Anomaly", color="red")

    plt.xlabel("Speed Mean")
    plt.ylabel("Acceleration Mean")
    plt.title("Anomaly Scatter Plot")
    plt.legend()
    plt.savefig(os.path.join(VIS_ANOMALY, "anomaly_scatter.png"))
    plt.close()


print("\n✔ Visualization generation completed!")
print("Check the visuals folder for output images.")
