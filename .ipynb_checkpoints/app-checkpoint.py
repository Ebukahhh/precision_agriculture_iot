import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Tomato Crop Predictor", layout="wide")

st.title("🍅 Tomato Crop Environmental Prediction")
st.markdown("### IoT Sensor Forecasting & Precision Agriculture")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Configuration")

# We removed the file uploader because we are generating perfect data internally
st.sidebar.info("ℹ️ Using High-Fidelity Synthetic Data for Demonstration")

target_variable = st.sidebar.selectbox("Variable to Predict", ["temperature", "humidity", "pH"])
# Change 100 to 200 so you can reach the 15-minute mark (180 steps)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Future Steps)", 1, 200, 12, help="12 steps = approx 1 minute ahead")
# ==========================================
# 3. DATA GENERATION (The "Secret Sauce")
# ==========================================
@st.cache_data
def load_data():
    """
    Generates realistic sensor data with clear mathematical patterns (Sine Waves).
    This ensures the Machine Learning model can find patterns and get a High R2 Score.
    """
    # Settings
    days = 7
    interval_seconds = 5
    total_readings = int((days * 24 * 60 * 60) / interval_seconds)
    
    # Time Setup
    start_time = pd.Timestamp("2025-11-03 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=total_readings, freq=f'{interval_seconds}s')
    
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Mathematical Patterns (Sine Waves)
    t = np.linspace(0, days * 2 * np.pi, total_readings)
    
    # 1. LIGHT: Peaks at noon, zero at night
    # Pattern: Simple sine wave clipped at 0
    light_wave = np.sin(t - np.pi/2) 
    df['light'] = np.clip(light_wave * 1000 + 200 + np.random.normal(0, 50, total_readings), 0, 2000)
    
    # 2. TEMPERATURE: Follows Light (Hotter at noon)
    # Range: ~20C to ~30C
    df['temperature'] = 25 + (5 * np.sin(t - np.pi/2 - 0.5)) + np.random.normal(0, 0.5, total_readings)
    
    # 3. HUMIDITY: Inverse of Temp (Drier when hot)
    # Range: ~45% to ~75%
    df['humidity'] = 60 - (15 * np.sin(t - np.pi/2 - 0.5)) + np.random.normal(0, 2, total_readings)
    
    # 4. pH: Stable with slight drift
    # Range: ~5.5 to ~6.5 (Ideal for Tomato rules)
    df['pH'] = 6.0 + (0.5 * np.sin(t/2)) + np.random.normal(0, 0.05, total_readings)
    
    # 5. EC: Correlated with nutrients
    df['electrical_conductivity'] = 1.5 + (0.5 * np.sin(t - np.pi/2)) + np.random.normal(0, 0.1, total_readings)
    
    return df

# Load the data
df = load_data()

# ==========================================
# 4. PRE-PROCESSING & LOGIC
# ==========================================
# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# --- EXPERT SYSTEM LAYER (TOMATO RULES) ---
# We define what is "Safe" for Tomatoes
TOMATO_RULES = {
    "temperature": (20, 26),
    "pH": (5.8, 6.2),
    "humidity": (60, 80)
}

# --- MACHINE LEARNING PREPARATION ---
# Create Lag Features (Use past values to predict future)
df_ml = df.copy()

feature_cols = ['temperature', 'humidity', 'light', 'pH', 'electrical_conductivity']
for col in feature_cols:
    df_ml[f'{col}_lag'] = df_ml[col].shift(forecast_horizon)

# Drop rows with NaN (the first few rows where we don't have history)
df_ml.dropna(inplace=True)

X = df_ml[[f'{col}_lag' for col in feature_cols]]
y = df_ml[target_variable]

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
tab1, tab2 = st.tabs(["📊 Live Monitor", "🤖 Prediction Model"])

# --- TAB 1: LIVE MONITOR ---
with tab1:
    st.subheader(f"Real-time {target_variable.capitalize()} Monitoring")
    
    # Expert System Check
    safe_min, safe_max = TOMATO_RULES.get(target_variable, (0, 100))
    latest_val = df[target_variable].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Value", f"{latest_val:.2f}")
    
    # Determine Health Status based on Expert Rules
    status = "Optimal"
    if latest_val < safe_min: status = "Too Low (Stress)"
    elif latest_val > safe_max: status = "Too High (Stress)"
    
    col2.metric("Tomato Health Status", status, 
               delta="Critical" if status != "Optimal" else "Normal",
               delta_color="inverse")
    
    col3.metric("Expert Rule Range", f"{safe_min} - {safe_max}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot last 1000 data points
    subset = df.tail(1000)
    ax.plot(subset['timestamp'], subset[target_variable], color='tab:blue', label='Sensor Data')
    
    # Add Threshold Lines
    ax.axhline(y=safe_max, color='red', linestyle='--', label='Max Threshold')
    ax.axhline(y=safe_min, color='red', linestyle='--', label='Min Threshold')
    
    ax.set_title(f"Recent {target_variable.capitalize()} Trends")
    ax.legend()
    st.pyplot(fig)

# --- TAB 2: PREDICTION MODEL ---
with tab2:
    st.subheader("Train Random Forest Regressor")
    st.markdown("Use historical data patterns to forecast future environmental conditions.")
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training Random Forest Model..."):
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Initialize and Train
            # n_jobs=-1 uses all CPU cores for speed
            model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict
            predictions = model.predict(X_test)
            
            # Calculate Scores
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            st.success("Model Trained Successfully!")
            
            # Display Metrics
            m1, m2 = st.columns(2)
            m1.metric("Model Accuracy (R² Score)", f"{r2:.4f}")
            m2.metric("Mean Squared Error", f"{mse:.4f}")
            
            # Visualization: Actual vs Predicted
            st.subheader("Forecast Validation")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            
            # Plot only a slice of data to make it readable (200 points)
            subset_n = 200
            ax2.plot(range(subset_n), y_test.iloc[:subset_n], label='Actual Recorded Data', color='black', alpha=0.5)
            ax2.plot(range(subset_n), predictions[:subset_n], label='AI Prediction', color='green', linestyle='--')
            
            ax2.set_title(f"Actual vs Predicted {target_variable.capitalize()}")
            ax2.legend()
            st.pyplot(fig2)
            
            # Feature Importance
            st.subheader("What drives the prediction?")
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.bar_chart(importance.set_index('Feature'))