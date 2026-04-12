import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ---------------------------------------------------------
# Generate a single synthetic motorcycle trip
# ---------------------------------------------------------

def generate_single_trip(
    trip_id: int,
    start_time: datetime,
    duration_seconds: int = 300,
    theft_probability: float = 0.12,
    seed: int = None
):
    """
    Generates a synthetic motorcycle sensor trip.
    Returns a DataFrame with GPS, speed, IMU, vibration, engine state, theft labels.
    """
    if seed is not None:
        np.random.seed(seed + trip_id)
        random.seed(seed + trip_id)

    # timestamps
    ts = [start_time + timedelta(seconds=i) for i in range(duration_seconds)]
    n = len(ts)

    # base GPS anchor
    lat0 = 12.90 + np.random.randn() * 0.002
    lon0 = 80.10 + np.random.randn() * 0.002

    # initialize sensor arrays
    speed = np.zeros(n)
    engine_on = np.zeros(n, dtype=int)
    vibration = np.abs(np.random.randn(n)) * 0.02

    # movement window (normal riding)
    move_start = np.random.randint(int(0.3 * n), int(0.6 * n))
    move_end = min(n - 1, move_start + np.random.randint(int(0.2 * n), int(0.5 * n)))

    for i in range(n):
        if move_start <= i <= move_end:
            engine_on[i] = 1
            speed[i] = np.clip(np.random.normal(12, 3), 1, 25)  # realistic riding speed
            vibration[i] += np.random.rand() * 0.1
        else:
            speed[i] = np.random.rand() * 0.3  # parked / idle noise

    # generate GPS trail from speed & heading
    lat = np.full(n, lat0)
    lon = np.full(n, lon0)
    heading = 2 * np.pi * np.cumsum(np.random.randn(n) * 0.01)

    for i in range(1, n):
        ds = speed[i]

        dlat = (ds * np.cos(heading[i])) / 111320
        dlon = (ds * np.sin(heading[i])) / (111320 * np.cos(np.deg2rad(lat0)))

        lat[i] = lat[i - 1] + dlat
        lon[i] = lon[i - 1] + dlon

    # IMU (accelerometer and gyro)
    accel_x = np.gradient(speed) + np.random.randn(n) * 0.2
    accel_y = np.random.randn(n) * 0.15
    gyro_z  = np.gradient(heading) + np.random.randn(n) * 0.05

    # label = 1 (theft), 0 (normal)
    label = np.zeros(n, dtype=int)

    # ------------------------------
    # Inject synthetic theft anomaly
    # ------------------------------
    if np.random.rand() < theft_probability:

        scenario = random.choice(["towing", "snatch", "engine_kill"])
        s = np.random.randint(int(0.1 * n), int(0.8 * n))
        length = np.random.randint(10, 60)

        if scenario == "towing":
            # vehicle moves with engine OFF
            speed[s:s+length] = np.random.uniform(4, 10, length)
            engine_on[s:s+length] = 0
            vibration[s:s+length] += 0.3
            label[s:s+length] = 1

        elif scenario == "snatch":
            # sudden spike in acceleration, movement without engine
            accel_x[s] += np.random.uniform(5, 12)
            speed[s:s+length] = np.random.uniform(10, 20, length)
            engine_on[s:s+length] = 0
            vibration[s:s+length] += 0.5
            label[s:s+length] = 1

        elif scenario == "engine_kill":
            # engine suddenly goes off but bike still moves
            engine_on[s:s+length] = 0
            vibration[s:s+length] += 0.2
            label[s:s+length] = 1

    # create DataFrame
    df = pd.DataFrame({
        "ts": ts,
        "trip_id": trip_id,
        "lat": lat,
        "lon": lon,
        "speed": speed,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "gyro_z": gyro_z,
        "vibration": vibration,
        "engine_on": engine_on,
        "label": label
    })

    return df


# ---------------------------------------------------------
# Generate multi-trip dataset
# ---------------------------------------------------------

def generate_dataset(n_trips: int = 100, seed: int = 42):
    """
    Generates multiple synthetic trips and returns a full combined dataset.
    """
    start = datetime.now()
    trips = []

    for t in range(n_trips):
        st = start + timedelta(seconds=t * 600)
        df = generate_single_trip(
            trip_id=t,
            start_time=st,
            duration_seconds=np.random.randint(200, 600),
            theft_probability=0.12,
            seed=seed
        )
        trips.append(df)

    return pd.concat(trips, ignore_index=True)
