import matplotlib.pyplot as plt

def plot_sensor_line(df, col):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df[col], color="blue")
    ax.set_title(f"{col} Over Time")
    ax.set_xlabel("Samples")
    ax.set_ylabel(col)
    ax.grid(True)
    return fig

def plot_anomaly_scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(6, 4))

    normal = df[df["anomaly"] == 0]
    anomaly = df[df["anomaly"] == 1]

    ax.scatter(normal[x], normal[y], s=10, label="Normal", alpha=0.6)
    ax.scatter(anomaly[x], anomaly[y], s=40, label="Anomaly", alpha=0.9, color="red")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.set_title("Anomaly Scatter Plot")
    return fig
