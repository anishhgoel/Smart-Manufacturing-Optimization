# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import torch
import numpy as np
from src.data.data_generator import CNCDataGenerator
from src.models.pinn import CNCPINN, prepare_training_data, train_pinn
from src.utils.monitoring import CNCMonitor, MachineStatus
import io
import base64
import matplotlib.pyplot as plt


app = FastAPI(title="Smart CNC System")

# global references (for demo) to store data and model in-memory
# in production, need to store them in a database or a persistent store.
latest_data: Optional[pd.DataFrame] = None
pinn_model: Optional[CNCPINN] = None
monitor = CNCMonitor(temp_threshold=160.0, vibration_threshold=2.0, tool_wear_threshold=0.8)


# pydantic schemas
class SimulationParams(BaseModel):
    base_temperature: float = 20.0
    cutting_speed: float = 100.0
    feed_rate: float = 0.2
    material_hardness: float = 150.0
    cooling_efficiency: float = 1.0
    duration: float = 1000.0
    sample_rate: float = 1.0

class TrainParams(BaseModel):
    n_samples: int = 1000
    epochs: int = 1000

class PredictParams(BaseModel):
    cutting_speed: float
    feed_rate: float
    material_hardness: float
    time_in_seconds: float

class MonitorParams(BaseModel):
    temp_threshold: float = 160.0
    vibration_threshold: float = 2.0
    wear_threshold: float = 0.8


# endpoints for the dashboard

@app.get("/")
def root():
    return {"message": "Smart CNC API is running."}

# simulate / Generate Data
@app.post("/simulate")
def simulate_data(params: SimulationParams):
    """
    to generate synthetic CNC data based on user-specified parameters
    and store it globally (latest_data).
    """
    global latest_data
    generator = CNCDataGenerator(
        base_temperature=params.base_temperature,
        cutting_speed=params.cutting_speed,
        feed_rate=params.feed_rate,
        material_hardness=params.material_hardness,
        cooling_efficiency=params.cooling_efficiency
    )
    df = generator.generate_full_dataset(
        duration=params.duration,
        sample_rate=params.sample_rate
    )
    latest_data = df
    sample_json = df.head(10).to_dict(orient="records")
    return {
        "message": "Data simulation complete",
        "num_rows": len(df),
        "params_used": params.dict(),
        "sample_data": sample_json
    }


#   to train PINN for param sweep approach (on a wide variety of data)

@app.post("/train")
def train_model(train_params: TrainParams):
    global pinn_model
    x_train, y_train = prepare_training_data(CNCDataGenerator, n_samples=train_params.n_samples)
    model = CNCPINN()
    train_pinn(model, x_train, y_train, epochs=train_params.epochs)
    pinn_model = model
    return {"message": f"PINN trained for {train_params.epochs} epochs on param-sweep"}


#  t train PINN on latest generated data

@app.post("/train_on_latest_data")
def train_model_on_latest_data(epochs: int = 1000):
    global latest_data, pinn_model
    if latest_data is None:
        raise HTTPException(status_code=400, detail="No data available. Please POST /simulate first.")
    
    df = latest_data.copy()

    # create or confirm a 'time_in_seconds' column
    if 'timestamp' in df.columns:
        df['time_in_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    else:
        df['time_in_seconds'] = (df.index - df.index[0]).total_seconds()

    x_cols = ['cutting_speed', 'feed_rate', 'material_hardness', 'time_in_seconds']
    y_col = 'temperature'

    x_train = torch.tensor(df[x_cols].values, dtype=torch.float32)
    y_train = torch.tensor(df[y_col].values.reshape(-1, 1), dtype=torch.float32)
    x_train.requires_grad_(True)

    model = CNCPINN()
    train_pinn(model, x_train, y_train, epochs=epochs)
    pinn_model = model
    return {"message": f"PINN trained on latest_data for {epochs} epochs", "rows_used": len(df)}

#  monitor Data threshold based
@app.post("/monitor")
def monitor_data(params: MonitorParams):
    global latest_data, monitor
    if latest_data is None:
        return {"error": "No data available. Please POST /simulate first."}
    
    custom_monitor = CNCMonitor(
        temp_threshold=params.temp_threshold,
        vibration_threshold=params.vibration_threshold,
        tool_wear_threshold=params.wear_threshold
    )
    status = custom_monitor.analyze_data(latest_data)
    return {
        "is_healthy": status.is_healthy,
        "temperature_status": status.temperature_status,
        "vibration_status": status.vibration_status,
        "tool_wear_status": status.tool_wear_status,
        "estimated_maintenance_time": status.estimated_maintenance_time,
        "anomalies": status.anomalies
    }

# monitor Data with PINN Predictions
@app.get("/monitor_with_pinn")
def monitor_data_with_pinn(residual_threshold: float = 5.0):
    global latest_data, pinn_model
    if latest_data is None:
        return {"error": "No data available. Please POST /simulate first."}
    if pinn_model is None:
        return {"error": "No trained PINN. Please train a model first."}

    df = latest_data.copy()
    if 'timestamp' in df.columns:
        df['time_in_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    else:
        df['time_in_seconds'] = (df.index - df.index[0]).total_seconds()

    x_cols = ['cutting_speed', 'feed_rate', 'material_hardness', 'time_in_seconds']
    import torch

    x_tensor = torch.tensor(df[x_cols].values, dtype=torch.float32)
    with torch.no_grad():
        preds = pinn_model(x_tensor).flatten().numpy()

    df['predicted_temp'] = preds
    df['residual'] = df['temperature'] - df['predicted_temp']

    # needto convert anomalies to strings if  timestamps
    # or just store them as row numbers, etc.
    anomaly_idx = df.index[abs(df['residual']) > residual_threshold].tolist()
    # If these are Timestamps, do:
    anomaly_strs = [str(ts) for ts in anomaly_idx]

    # convert to a Python bool, not a numpy.bool_, previosly caused some issue so keep an eye
    high_temp_warning = bool(df['predicted_temp'].max() > 160.0)

    # if timestamp is a Timestamp column, convert it to string in the preview:
    # to_dict() sometimes handles it, but better to be explicit
    df['timestamp'] = df['timestamp'].astype(str)

    head_preview = df[['timestamp','temperature','predicted_temp','residual']].head(5).to_dict(orient="records")

    return {
        "residual_threshold": residual_threshold,
        "num_anomalies": len(anomaly_strs),
        "anomaly_rows": anomaly_strs,
        "predicted_temp_max": float(df['predicted_temp'].max()),
        "high_temp_warning": high_temp_warning,
        "head_preview": head_preview
    }

# to predict Temperature at a single point
@app.post("/predict")
def predict_temperature(req: PredictParams):
    global pinn_model
    if pinn_model is None:
        return {"error": "No trained model. Please POST /train or /train_on_latest_data first."}
    
    x_input = torch.tensor([[
        req.cutting_speed,
        req.feed_rate,
        req.material_hardness,
        req.time_in_seconds
    ]], dtype=torch.float32)
    with torch.no_grad():
        y_pred = pinn_model(x_input).item()
    return {"predicted_temperature": y_pred}
