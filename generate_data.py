import pandas as pd
import numpy as np

# Settings
days = 7
interval_seconds = 5
total_readings = int((days * 24 * 60 * 60) / interval_seconds)
start_time = pd.Timestamp("2025-11-01 00:00:00")

# Create Time Index
timestamps = pd.date_range(start=start_time, periods=total_readings, freq=f'{interval_seconds}s')
df = pd.DataFrame({'timestamp': timestamps})

# --- GENERATE REALISTIC PATTERNS (SINE WAVES) ---
# Time of day factor (0 to 2pi) for daily cycles
t = np.linspace(0, days * 2 * np.pi, total_readings)

# 1. LIGHT: Peaks at midday, 0 at night (clipped sine wave)
# Pattern: sin(t - pi/2) shifts peak to noon. Clip negative values to 0.
light_wave = np.sin(t - np.pi/2) 
df['light'] = np.clip(light_wave * 1000 + 200 + np.random.normal(0, 50, total_readings), 0, 2000)

# 2. TEMPERATURE: Follows Light (lagged slightly) + Random Noise
# Range: 20C (Night) to 30C (Day)
df['temperature'] = 25 + (5 * np.sin(t - np.pi/2 - 0.5)) + np.random.normal(0, 0.5, total_readings)

# 3. HUMIDITY: Inverse of Temperature (Hot = Dry, Cold = Wet)
# Range: 40% to 80%
df['humidity'] = 60 - (15 * np.sin(t - np.pi/2 - 0.5)) + np.random.normal(0, 2, total_readings)

# 4. pH: Mostly stable with slow drift
df['pH'] = 6.0 + (0.5 * np.sin(t/2)) + np.random.normal(0, 0.05, total_readings)

# 5. EC: Correlated with Temperature (Nutrient uptake)
df['electrical_conductivity'] = 1.5 + (0.5 * np.sin(t - np.pi/2)) + np.random.normal(0, 0.1, total_readings)

# Save to CSV
df.to_csv("data/realistic_sensor_data.csv", index=False)
print("✅ New REALISTIC dataset generated: data/realistic_sensor_data.csv")