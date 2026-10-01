# Motorcycle Theft Intelligence System

### Machine Learning and Synthetic Sensor Anomaly Detection

An end-to-end machine learning system that identifies **potential motorcycle theft patterns** using synthetic sensor data, sliding-window feature engineering, Random Forest classification, Isolation Forest anomaly detection, and an interactive Streamlit dashboard.

---

## Project Overview

Motorcycle theft can involve sudden movement, unauthorized handling, or suspicious activity while a vehicle is parked. This project explores how sensor analytics and machine learning can help identify movement patterns associated with potential theft.

The system combines **synthetic sensor data generation, feature engineering, supervised classification, and unsupervised anomaly detection** to analyze motorcycle movement and assign theft-risk scores.

### Sensor Streams

* **Speed:** Detects unusual speed changes and unexpected movement.
* **Acceleration:** Captures sudden changes in motion.
* **Gyroscope (Gyro):** Represents rotational movement.
* **Vibration:** Identifies abnormal vibration patterns.
* **GPS Displacement:** Captures sudden changes in recorded location.
* **Engine Status:** Helps identify movement while the engine is switched off.

### Suspicious Patterns Analyzed

* Sudden acceleration or movement.
* Vehicle movement while the engine is OFF.
* Abnormal vibration spikes.
* Rapid GPS displacement that may indicate unauthorized movement.
* Unusual sensor patterns while the motorcycle is parked.

> **Note:** This project uses synthetic sensor data. Its predictions indicate potentially suspicious patterns and do not independently confirm an actual theft.

---

## Key Features

* Synthetic motorcycle sensor data generation.
* Sliding-window feature engineering.
* Random Forest theft-risk classification.
* Isolation Forest unsupervised anomaly detection.
* Interactive Streamlit dashboard.
* Sensor trend visualization and anomaly plotting.
* Model training and evaluation notebooks.
* Modular Python architecture.
* Visualization scripts for analysis and reporting.

---

## Project Structure

```text
Motorcycle-Theft-Intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_generate_synthetic_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training_eval.ipynb
│
├── src/
│   ├── data_generator.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py
│   ├── model_training.py
│   └── utils.py
│
├── app/
│   ├── streamlit_app.py
│   ├── components/
│   │   ├── charts.py
│   │   ├── layout.py
│   │   └── maps.py
│   └── assets/
│       └── style.css
│
├── visuals/
│   ├── generate_visuals.py
│   ├── speed_plots/
│   ├── anomaly_plots/
│   └── feature_plots/
│
├── models/
│   ├── isolation_forest.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── configs/
│   └── settings.yaml
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

The system follows a modular pipeline that transforms synthetic sensor readings into engineered features, model predictions, and visual insights.

1. **Data Generation:** Generate synthetic motorcycle sensor readings representing different movement patterns.
2. **Feature Engineering:** Extract statistical features from sliding windows of sensor data.
3. **Model Training:** Train a Random Forest classifier and an Isolation Forest anomaly detector.
4. **Risk Assessment:** Analyze input features to estimate theft-like behavior and identify unusual sensor patterns.
5. **Visualization:** Display sensor trends, risk predictions, and detected anomalies through the Streamlit dashboard.

---

## Feature Engineering

Sliding-window feature extraction summarizes sensor behavior over a specified time window.

| Feature         | Description                                          |
| --------------- | ---------------------------------------------------- |
| `speed_mean`    | Average speed within the window                      |
| `speed_std`     | Standard deviation of speed                          |
| `accel_mean`    | Average acceleration                                 |
| `accel_std`     | Standard deviation of acceleration                   |
| `vib_mean`      | Average vibration intensity                          |
| `gps_jump_mean` | Average GPS displacement per observation or interval |
| `engine_ratio`  | Proportion of observations with the engine ON        |

These features help capture changes in movement, sensor variability, and engine activity.

---

## Machine Learning Models

### 1. Random Forest Classifier

A supervised machine learning model used to classify sensor patterns associated with theft-like behavior.

**Purpose:**

* Estimate the likelihood of theft-like patterns.
* Generate a risk score for dashboard visualization.
* Learn relationships between engineered sensor features and training labels.

**Model artifacts:**

```text
models/
├── random_forest.pkl
└── scaler.pkl
```

The scaler is required if the training pipeline uses feature scaling and the same transformation is applied during inference.

### 2. Isolation Forest

An unsupervised anomaly detection algorithm used to identify sensor observations that deviate from typical patterns.

**Purpose:**

* Detect unusual sensor combinations.
* Flag unexpected movement or vibration patterns.
* Complement the Random Forest classifier with anomaly-based signals.

**Model artifact:**

```text
models/isolation_forest.pkl
```

The anomaly detector complements the classifier; an anomaly is not necessarily evidence of theft.

---

## Streamlit Dashboard

The interactive dashboard provides a user interface for exploring sensor data and reviewing model outputs.

### Dashboard Modules

* **Dashboard:** Overview of sensor activity and model outputs.
* **Risk Prediction:** ML-based theft-risk scoring.
* **Sensor Visualization:** Line charts for sensor readings over time.
* **Anomaly Detection:** Visualization of observations flagged by Isolation Forest.

### Run the Dashboard

From the project root directory:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will display a local URL that you can open in your browser.

---

## Data Visualization

The project includes a visualization script for generating plots used in analysis and reporting.

### Generate Visualizations

```bash
python visuals/generate_visuals.py
```

### Expected Visualizations

* Speed distribution.
* Acceleration distribution.
* Vibration analysis.
* GPS displacement distribution.
* Feature correlation heatmap.
* Pairwise feature plots.
* Anomaly scatter plots.

### Output Directories

```text
visuals/
├── speed_plots/
├── feature_plots/
└── anomaly_plots/
```

The generated outputs depend on the implementation of the visualization script and the availability of the required input data.

---

## Installation and Setup

### Prerequisites

* Python 3.10 or later.
* pip package manager.
* Git, for cloning the repository.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Motorcycle-Theft-Intelligence
```

Replace the placeholder with the actual repository URL.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
streamlit run app/streamlit_app.py
```

**Alternative workflow:** Run the Jupyter notebooks in sequence to explore synthetic data generation, feature engineering, and model training.

---

## Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Core implementation                   |
| Pandas           | Data manipulation                     |
| NumPy            | Numerical computation                 |
| Scikit-learn     | Random Forest and Isolation Forest    |
| Streamlit        | Interactive dashboard                 |
| Matplotlib       | Data visualization                    |
| Seaborn          | Statistical visualization             |
| Jupyter Notebook | Experimentation and model development |
| PyYAML           | Configuration management              |

*The final dependency list should reflect the libraries actually used in the project.*

---

## Future Enhancements

* **Live GPS Tracking:** Integrate real GPS data for location monitoring.
* **IoT Sensor Integration:** Connect physical sensors to collect motorcycle telemetry.
* **Sequence Models:** Explore LSTM or other time-series models for temporal pattern recognition.
* **Real-Time Alerts:** Add notifications when suspicious movement is detected.
* **Mobile Application:** Provide a mobile interface for monitoring vehicle activity.
* **Cloud API:** Expose prediction services through a deployable API.
* **Real-World Validation:** Evaluate model performance using representative, ethically collected sensor data.
* **Model Monitoring:** Track false positives, false negatives, and prediction drift.

---

## Limitations

* The current system is based on synthetic sensor data and may not generalize to real-world motorcycle theft scenarios.
* Performance depends on the quality of the synthetic data, feature engineering, and training labels.
* Anomaly detection identifies unusual patterns, which may also arise from legitimate activities.
* Real-world deployment would require validation, reliable sensor integration, and evaluation under different operating conditions.

---

## Conclusion

The Motorcycle Theft Intelligence System demonstrates how machine learning, sensor analytics, anomaly detection, and interactive visualization can be combined to analyze potentially suspicious motorcycle movement.

By integrating a Random Forest classifier with Isolation Forest anomaly detection, the project provides an academic and practical foundation for developing sensor-based vehicle monitoring systems.

## Author

**Vignesh S**
MSc Data Science
Bishop Heber College
