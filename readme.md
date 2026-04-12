🚨 Motorcycle Theft Intelligence System  
### Machine Learning + Synthetic Sensor Anomaly Detection  

A complete end-to-end system that predicts motorcycle theft risk using sensor analytics, anomaly detection, and a Streamlit dashboard.

---

## 📌 Project Overview

Motorcycle theft often occurs quickly and silently.  
This system uses **synthetic sensor data**, **sliding-window ML features**,  
**Random Forest classification**, and **Isolation Forest anomaly detection**  
to flag unusual or theft-like patterns in motorcycle motion data.

The system works on five key sensor streams:

- Speed  
- Acceleration  
- Gyro  
- Vibration  
- GPS jumps  

These signals are used to detect patterns like:

- Sudden acceleration  
- Forced movement while engine is OFF  
- Abnormal vibration spikes  
- Rapid GPS displacement (push theft)  
- Suspicious movement patterns when parked  

---

## 🎯 Key Features

✔ Synthetic sensor data generator  
✔ Feature engineering with sliding windows  
✔ Random Forest theft classifier  
✔ Isolation Forest anomaly detection  
✔ Streamlit dashboard with risk prediction  
✔ Sensor visualization & anomaly plotting  
✔ Visualization script for reports  
✔ Full modular folder structure  

---

## 📁 Project Structure

Motorcycle-Theft-Intelligence/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_generate_synthetic_data.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_training_eval.ipynb
│
├── src/
│ ├── data_generator.py
│ ├── feature_engineering.py
│ ├── anomaly_detection.py
│ ├── model_training.py
│ └── utils.py
│
├── app/
│ ├── streamlit_app.py
│ ├── components/
│ │ ├── charts.py
│ │ ├── layout.py
│ │ └── maps.py
│ └── assets/
│ └── style.css
│
├── visuals/
│ ├── speed_plots/
│ ├── anomaly_plots/
│ └── feature_plots/
│
├── models/
│ ├── isolation_forest.pkl
│ ├── random_forest.pkl
│ └── scaler.pkl
│
├── configs/
│ └── settings.yaml
│
├── requirements.txt
│
└── README.md

yaml
Copy code

---

## 🧠 Machine Learning Models

### 🔹 1. Feature Engineering
Sliding-window feature extraction:

| Feature | Meaning |
|--------|---------|
| speed_mean | Avg speed |
| speed_std | Speed variation |
| accel_mean | Avg acceleration |
| accel_std | Acceleration variation |
| vib_mean | Vibration intensity |
| gps_jump_mean | Sudden GPS displacement |
| engine_ratio | % of time engine ON |

---

### 🔹 2. Classification Model — Random Forest  
Predicts probability of theft-like behavior.  
Model saved at:

models/random_forest.pkl
models/scaler.pkl

yaml
Copy code

---

### 🔹 3. Unsupervised Model — Isolation Forest  
Detects unexpected patterns using unsupervised anomaly detection.

Saved at:

models/isolation_forest.pkl

yaml
Copy code

---

## 🖥️ Streamlit Dashboard

Run the dashboard:

```bash
cd Motorcycle-Theft-Intelligence
streamlit run app/streamlit_app.py
Pages:
Dashboard

Risk Prediction (ML-based theft scoring)

Sensor Visualization (line charts)

Anomaly Detection (Isolation Forest detection)

📊 Visualizations
Generated using:

bash
Copy code
python visuals/generate_visuals.py
Output includes:

Speed distribution

Acceleration distribution

Vibration analysis

GPS jump distribution

Correlation heatmap

Pairplots

Anomaly scatter

Saved inside:

bash
Copy code
visuals/speed_plots/
visuals/feature_plots/
visuals/anomaly_plots/
🔧 Installation
nginx
Copy code
pip install -r requirements.txt
Python version recommended: 3.10+

🔮 Future Improvements
GPS live tracking

IoT sensor integration

LSTM sequence models

Real-time theft alerting

Mobile app

Cloud API for live monitoring

🏁 Conclusion
This project demonstrates how machine learning,
sensor analytics, and real-time visualization
can be combined to detect motorcycle theft with high accuracy.

A complete academic + practical ML system.

👤 Author
Vignesh S
MSc Data Science
Bishop Heber College