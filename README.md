# AIR QUALITY MONITORING AND PREDICTION SYSTEM

## OVERVIEW
The **Air Quality Intelligence & Prediction System** is an advanced end-to-end data science and machine learning platform designed to monitor environmental air quality, predict pollutant concentrations, detect statistical anomalies, evaluate diurnal atmospheric dynamics, and provide actionable health recommendations and policy simulations.

---

## Collaborators
1. **Mahmoud Ashraf**
2. **Shahd hani**
3. **Mahmoud Sadek**
4. **Farah salah**
5. **Abdelrahman Mohamed**

---

## KEY FEATURES
- **Automated Data Cleaning & Imputation**: Handles missing values, sensor faults (-200 markers), and date/time parsing.
- **System Overview & KPI Dashboard**: Real-time summary metrics, sensor feature discovery, and category breakdowns.
- **Executive Intelligence Hub**:
  - *Dynamic Statistical Anomaly Detection*: Identifies acute contamination events using customizable Z-score bounds.
  - *Environmental Policy Impact Simulator*: Models simulated interventions (e.g., green zones, traffic limits) and calculates clean air gain.
  - *Executive Audit Report Generator*: Downloadable formal environmental audit report.
- **24-Hour Diurnal & Temporal Patterns**:
  - Explores morning commute and evening peak cycles, weekend vs. weekday deltas.
  - Interactive 24x7 Day-Hour heatmaps and boundary layer atmospheric science insights.
- **Exploratory Data Analysis & Bivariate Correlations**: Interactive heatmaps and pairwise scatter plots.
- **Predictive Regression Modeling**: Multi-algorithm pollutant prediction (Ridge Regression, Decision Tree, Random Forest) with $R^2$, MAE, and RMSE evaluation.
- **Classification & Health Diagnosis**:
  - Multi-class categorization (*Good*, *Moderate*, *Poor*) with confusion matrix and full classification report.
  - *Sensor Simulator*: Live interactive parameter testing for instant air quality diagnosis and health advice.
  - *Autoregressive Forecasting*: Multi-step forward projection of air quality indicators.
- **Modern Glassmorphism UI**: Dynamic nature-inspired backgrounds and soundscapes.

---

## DATASET
The project utilizes the **Air Quality UCI Dataset**, collected from an array of chemical sensors stationed in an Italian city:
- **CO(GT)**: True hourly averaged carbon monoxide concentration ($mg/m^3$)
- **NMHC(GT)**: Non-methanic hydrocarbons concentration ($mg/m^3$)
- **C6H6(GT)**: True hourly averaged Benzene concentration ($\mu g/m^3$)
- **NOx(GT)**: True hourly averaged Nitrogen oxides concentration ($ppb$)
- **NO2(GT)**: True hourly averaged Nitrogen dioxide concentration ($\mu g/m^3$)
- **PT08_S1 to PT08_S5**: Sensor resistance responses (Tin oxide, Titania, Tungsten oxide, Indium oxide)
- **T / RH / AH**: Temperature (°C), Relative Humidity (%), and Absolute Humidity

---

## MACHINE LEARNING MODELS

### Regression Models:
- **Ridge Regression (L2 Regularized)**
- **Decision Tree Regressor**
- **Random Forest Regressor**

### Classification Models:
- **Random Forest Classifier**

### Forecasting:
- **Autoregressive Multi-step Forecaster**

---

## AIR QUALITY CATEGORIES
- 🟢 **Good**: Safe air quality; low pollutant concentrations.
- 🟡 **Moderate**: Elevated levels; sensitive individuals should take precautions.
- 🔴 **Poor**: Hazardous air quality; high pollutant concentration exceeding safety thresholds.

---

## TECHNOLOGIES USED
- **Core**: Python
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly (Plotly Express & Graph Objects), Matplotlib, Seaborn
- **Dashboard Framework**: Streamlit
- **UI & UX**: Pillow (PIL), HTML5/CSS3 (Glassmorphism & Audio Web Synthesis)

---

## PROJECT STRUCTURE
```text
├── AirQualityUCI.csv          # Cleaned sensor dataset
├── app.py                     # Streamlit Intelligence Dashboard
├── DataAnalysisFile.ipynb     # Exploratory analysis & model experimentation notebook
├── Launch_Dashboard.bat       # Windows 1-click execution script (auto venv setup)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── assets/                    # Dashboard UI assets
    ├── cloud_icon.png         # App favicon/icon
    └── wallpapers/            # Section-specific high-resolution wallpapers
```

---

## INSTALLATION & HOW TO RUN

### Option 1: One-Click Launcher (Windows)
Double-click `Launch_Dashboard.bat`. It will automatically configure the virtual environment, install dependencies, and launch the application in your browser.

### Option 2: Manual Setup
1. **Clone repository:**
   ```bash
   git clone https://github.com/namharledba/Air-Quality-Prediction-Dashboard-.git
   cd Air-Quality-Prediction-Dashboard-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Option 3: Live Cloud App
Access the deployed Streamlit cloud dashboard:
[Air Quality Prediction Dashboard](https://namharledba-air-quality-prediction-dashboard--app-m6lycc.streamlit.app/)

---

## DASHBOARD SECTIONS
1. **Dashboard**: System overview, key telemetry, and data health metrics.
2. **Executive Intelligence Hub**: Statistical anomaly detection, policy simulator, and audit reports.
3. **Daily Patterns**: 24-hour diurnal profile, weekend vs. weekday trends, and atmospheric science insights.
4. **Dataset Explorer**: Raw data viewer and statistical summaries.
5. **Correlation Analysis**: Correlation matrix heatmap and bivariate relationships.
6. **Predictive Modeling**: Algorithm comparison (Ridge, Decision Tree, Random Forest) and metrics ($R^2$, MAE, RMSE).
7. **Classification & Diagnosis**: Multi-class categorization, confusion matrix, sensor simulator, and autoregressive forecasting.

---

## HEALTH RECOMMENDATIONS
- **Good (🟢)**: Outdoor activities are safe. Natural ventilation is encouraged.
- **Moderate (🟡)**: Sensitive individuals (asthma, children, elderly) should limit prolonged outdoor exertion. Consider protective masks.
- **Poor (🔴)**: Avoid outdoor activities. Keep windows closed, operate air filtration/purifiers, and wear protective masks if going outside.
