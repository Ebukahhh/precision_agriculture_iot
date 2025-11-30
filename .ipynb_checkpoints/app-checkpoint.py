import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Hydroponics Monitor", layout="wide")

st.title("🌱 Smart Hydroponics Monitoring System")
st.markdown("### Real-time IoT Sensor Analysis & Anomaly Detection")

# --- SIDEBAR ---
st.sidebar.header("Control Panel")
uploaded_files = st.sidebar.file_uploader("Upload Sensor CSVs", accept_multiple_files=True, type="csv")

# Simulation Checkbox
simulate_faults = st.sidebar.checkbox("🔴 Simulate System Faults", value=False, help="Inject artificial errors to test anomaly detection.")

window_size = st.sidebar.slider("Moving Average Window", 60, 1000, 720)
sigma_level = st.sidebar.slider("Anomaly Sensitivity (Sigma)", 1.0, 5.0, 3.0, 0.1)

# Function to load data
@st.cache_data
def load_data(uploaded_files):
    if not uploaded_files:
        data_folder = 'data'
        if os.path.exists(data_folder):
            files = glob.glob(os.path.join(data_folder, "*.csv"))
            if files:
                dfs = [pd.read_csv(f) for f in files]
                return pd.concat(dfs, ignore_index=True)
        return None
    dfs = [pd.read_csv(f) for f in uploaded_files]
    return pd.concat(dfs, ignore_index=True)

df = load_data(uploaded_files)

if df is not None:
    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- SIMULATION LOGIC ---
    if simulate_faults:
        # Inject Overheating (Day 3)
        spike_start = 20000
        spike_end = 20200
        df.loc[spike_start:spike_end, 'temperature'] += 15 
        
        # Inject Sensor Failure (Day 5)
        drop_start = 50000
        drop_end = 50050
        df.loc[drop_start:drop_end, 'temperature'] = 0
        
        st.sidebar.warning("⚠️ SIMULATION ACTIVE: Artificial faults injected.")

    # 1. Feature Engineering
    df['period'] = df['light'].apply(lambda x: 'Daytime' if x > 300 else 'Nighttime')
    df['temp_moving_avg'] = df['temperature'].rolling(window=window_size).mean()

    # 2. Anomaly Detection (FIXED SECTION)
    window = 60
    rolling_mean = df['temperature'].rolling(window=window).mean()
    rolling_std = df['temperature'].rolling(window=window).std()
    
    # FIX: Assign these directly to the DataFrame columns
    df['upper_bound'] = rolling_mean + (sigma_level * rolling_std)
    df['lower_bound'] = rolling_mean - (sigma_level * rolling_std)
    
    # Comparison using the new columns
    df['is_anomaly'] = ((df['temperature'] > df['upper_bound']) | (df['temperature'] < df['lower_bound']))
    anomalies = df[df['is_anomaly']]

    # --- DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Temperature", f"{df['temperature'].mean():.2f} °C")
    col2.metric("Avg Humidity", f"{df['humidity'].mean():.2f} %")
    col3.metric("Avg pH Level", f"{df['pH'].mean():.2f}")
    
    anom_count = len(anomalies)
    col4.metric("Anomalies Detected", f"{anom_count}", 
                delta="CRITICAL" if anom_count > 0 else "Normal", 
                delta_color="inverse")

    st.subheader("🌡️ Temperature Anomaly Detection")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['timestamp'], df['temperature'], label='Raw Temp', color='dodgerblue', alpha=0.5)
    ax.plot(df['timestamp'], df['temp_moving_avg'], label='Moving Avg', color='darkgreen', linewidth=2)
    
    if not anomalies.empty:
        ax.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Anomaly', zorder=5, s=50)
    
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # Data Table (This caused the error before, now fixed)
    if not anomalies.empty:
        st.subheader("⚠️ Anomaly Log")
        st.dataframe(anomalies[['timestamp', 'temperature', 'upper_bound', 'lower_bound']].head(10))
    else:
        st.success("System Normal. No anomalies detected.")
else:
    st.info("Awaiting Data...")