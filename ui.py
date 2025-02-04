# ui.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Smart CNC Dashboard")

# 1. Simulate Data
st.header("1) Simulate Data")
base_temperature = st.number_input("Base Temperature (°C)", value=20.0)
cutting_speed = st.number_input("Cutting Speed (m/min)", value=100.0)
feed_rate = st.number_input("Feed Rate (mm/rev)", value=0.2)
material_hardness = st.number_input("Material Hardness (HB)", value=150.0)
cooling_efficiency = st.number_input("Cooling Efficiency", value=1.0)
duration = st.number_input("Duration (seconds)", value=1000.0)
#sample_rate = st.number_input("Sample Rate (Hz)", value=1.0)

if st.button("Simulate Data"):
    payload = {
        "base_temperature": base_temperature,
        "cutting_speed": cutting_speed,
        "feed_rate": feed_rate,
        "material_hardness": material_hardness,
        "cooling_efficiency": cooling_efficiency,
        "duration": duration,
       # "sample_rate": sample_rate
    }
    resp = requests.post(f"{API_URL}/simulate", json=payload)
    if resp.status_code == 200:
        st.success("Data simulation successful!")
        st.json(resp.json())
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")

# 2. Train Model (Param Sweep)
st.header("2) Train Model (Param Sweep)")
n_samples = st.number_input("Number of samples (for training data)", value=1000)
epochs = st.number_input("Number of epochs", value=1000)

if st.button("Train Model (Param Sweep)"):
    payload = {"n_samples": n_samples, "epochs": epochs}
    resp = requests.post(f"{API_URL}/train", json=payload)
    if resp.status_code == 200:
        st.success("Model trained successfully (param sweep)!")
        st.json(resp.json())
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")


# 2B. Train Model On Latest Data
st.header("2B) Train on Latest Data")
train_latest_epochs = st.number_input("Epochs (latest data)", value=1000, key="latest_epochs")
if st.button("Train Model on Latest Data"):
    # Just pass epochs as a query param for simplicity
    resp = requests.post(f"{API_URL}/train_on_latest_data?epochs={train_latest_epochs}")
    if resp.status_code == 200:
        st.success("Trained on latest data!")
        st.json(resp.json())
    else:
        st.error(f"{resp.status_code} {resp.text}")


# 3. Monitor Data (Threshold Based)
st.header("3) Monitor the Data (Threshold)")

temp_threshold = st.number_input("Temperature Threshold (°C)", value=160.0)
vibration_threshold = st.number_input("Vibration Threshold", value=2.0)
wear_threshold = st.number_input("Tool Wear Threshold", value=0.8)

if st.button("Monitor (Threshold-Based)"):
    payload = {
        "temp_threshold": temp_threshold,
        "vibration_threshold": vibration_threshold,
        "wear_threshold": wear_threshold
    }
    resp = requests.post(f"{API_URL}/monitor", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        if "error" in data:
            st.error(data["error"])
        else:
            st.write("**Machine Health**:", "Healthy" if data["is_healthy"] else "Warning")
            st.write("**Temperature Status**:", data["temperature_status"])
            st.write("**Vibration Status**:", data["vibration_status"])
            st.write("**Tool Wear**:", data["tool_wear_status"])
            st.write("**Estimated Maintenance**:", data["estimated_maintenance_time"])
            st.write("**Anomalies**:", data["anomalies"])
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")


# 3B. Monitor with PINN
st.header("3B) Monitor with PINN Predictions")
resid_threshold = st.number_input("Residual Threshold (°C)", value=5.0, step=0.5)
if st.button("Monitor (Model-Based)"):
    resp = requests.get(f"{API_URL}/monitor_with_pinn?residual_threshold={resid_threshold}")
    if resp.status_code == 200:
        data = resp.json()
        if "error" in data:
            st.error(data["error"])
        else:
            st.write("**Number of Anomalies**:", data["num_anomalies"])
            st.write("**Anomaly Rows**:", data["anomaly_rows"])
            st.write("**Max Predicted Temp**:", data["predicted_temp_max"])
            st.write("**High Temp Warning?**:", data["high_temp_warning"])
            st.write("**Data Preview**:")
            st.json(data["head_preview"])
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")


# 4. Predict Temperature
st.header("4) Predict Temperature with PINN")
predict_cut_speed = st.number_input("Cutting Speed (m/min)", value=120.0, key="predict_speed")
predict_feed_rate = st.number_input("Feed Rate (mm/rev)", value=0.25, key="predict_feed")
predict_hardness = st.number_input("Material Hardness (HB)", value=160.0, key="predict_hardness")
predict_time = st.number_input("Time (seconds)", value=500.0, key="predict_time")

if st.button("Predict Temperature"):
    payload = {
        "cutting_speed": predict_cut_speed,
        "feed_rate": predict_feed_rate,
        "material_hardness": predict_hardness,
        "time_in_seconds": predict_time
    }
    resp = requests.post(f"{API_URL}/predict", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        if "error" in data:
            st.error(data["error"])
        else:
            pred_temp = data["predicted_temperature"]
            st.success(f"Predicted Temperature: {pred_temp:.2f} °C")
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")

