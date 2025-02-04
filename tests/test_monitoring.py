import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.data_generator import CNCDataGenerator
from src.models.pinn import CNCPINN
from src.utils.monitoring import CNCMonitor


def train_on_generated_data_with_norm(df, epochs=5000):
    """
    training a PINN on df's data, with:
      - normalization,
      - a learning rate scheduler,
      - a lower physics loss weight,
      - more hidden units (100),
      - up to 5000 epochs
    """

    #  convert timestamp -> time_seconds
    df = df.copy()
    if 'timestamp' in df.columns and df['timestamp'].dtype.kind == 'M':
        df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    else:
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

    #  identify columns
    x_cols = ['cutting_speed', 'feed_rate', 'material_hardness', 'time_seconds']
    y_col = 'temperature'

    #   max values for normalization
    max_cs   = df['cutting_speed'].max()
    max_feed = df['feed_rate'].max()
    max_mh   = df['material_hardness'].max()
    max_time = df['time_seconds'].max()
    max_temp = df['temperature'].max()

    #  normalizization
    df['cs_norm']   = df['cutting_speed'] / max_cs
    df['feed_norm'] = df['feed_rate']    / max_feed
    df['mh_norm']   = df['material_hardness'] / max_mh
    df['time_norm'] = df['time_seconds'] / max_time
    df['temp_norm'] = df['temperature']  / max_temp

    #  building tensors
    x_train = torch.tensor(
        df[['cs_norm','feed_norm','mh_norm','time_norm']].values,
        dtype=torch.float32
    )
    y_train = torch.tensor(
        df['temp_norm'].values.reshape(-1, 1),
        dtype=torch.float32
    )
    x_train.requires_grad_(True)

    # 6) constructing the PINN with more hidden units
    model = CNCPINN(input_dim=4, hidden_dim=100, output_dim=1)

    # 7) optimizer + scheduler
    base_lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,   # every 1000 epochs
        gamma=0.5         # multiply LR by 0.5
    )

    # 8) train loop with a  physics weight
    mse_loss = torch.nn.MSELoss()
    physics_weight = 0.01

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred_norm = model(x_train)
        data_loss = mse_loss(y_pred_norm, y_train)

        physics_loss = model.physics_loss(x_train, y_pred_norm)
        total_loss = data_loss + physics_weight * physics_loss

        total_loss.backward()
        optimizer.step()

        # Step the scheduler once per epoch
        scheduler.step()

        # printing output
        if epoch % 500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} | LR={current_lr:.5f} | Data={data_loss.item():.3f} | "
                  f"Phys={physics_loss.item():.3f} | Tot={total_loss.item():.3f}")

    # Return model + scaling
    scaling_factors = {
        'max_cs': max_cs,
        'max_feed': max_feed,
        'max_mh': max_mh,
        'max_time': max_time,
        'max_temp': max_temp
    }
    return model, scaling_factors


def test_full_system():
    # geenerating 1 hour of data at 1 Hz
    generator = CNCDataGenerator(
        cutting_speed=150.0,
        material_hardness=150,
        cooling_efficiency=1.0
    )
    df = generator.generate_full_dataset(duration=3600, sample_rate=1)

    # training with normalization, 5000 epochs, LR scheduler, physics_weight=0.01
    model, scales = train_on_generated_data_with_norm(df, epochs=5000)

    # basic threshold-based monitoring
    monitor = CNCMonitor()
    status = monitor.analyze_data(df)

    # Predict with the model (un-scale the output)
    df = df.copy()
    if 'timestamp' in df.columns and df['timestamp'].dtype.kind == 'M':
        df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    else:
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

    # Scale inputs
    df['cs_norm']   = df['cutting_speed'] / scales['max_cs']
    df['feed_norm'] = df['feed_rate']    / scales['max_feed']
    df['mh_norm']   = df['material_hardness'] / scales['max_mh']
    df['time_norm'] = df['time_seconds'] / scales['max_time']

    x_test = torch.tensor(
        df[['cs_norm','feed_norm','mh_norm','time_norm']].values,
        dtype=torch.float32
    )
    with torch.no_grad():
        y_pred_norm = model(x_test).flatten().numpy()

    df['predicted_temp'] = y_pred_norm * scales['max_temp']
    df['residual'] = df['temperature'] - df['predicted_temp']

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # (a) actual
    axes[0].plot(df['timestamp'], df['temperature'], color='blue', label='Actual Temp')
    axes[0].set_title('Actual Temperature Over Time')
    axes[0].grid(True)
    axes[0].legend()

    # (b) predicted
    axes[1].plot(df['timestamp'], df['predicted_temp'], color='orange', label='Predicted Temp')
    axes[1].set_title('Predicted Temperature Over Time')
    axes[1].grid(True)
    axes[1].legend()

    # (c) compare
    axes[2].plot(df['timestamp'], df['temperature'], color='blue', label='Actual')
    axes[2].plot(df['timestamp'], df['predicted_temp'], color='orange', linestyle='--', label='Predicted')
    axes[2].set_title('Actual vs. Predicted Temperature')
    axes[2].grid(True)
    axes[2].legend()

    # (d) residual
    axes[3].plot(df['timestamp'], df['residual'], color='red', label='Residual (Actual - Predicted)')
    axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[3].set_title('Residual Over Time')
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    # printing threshold-based status
    print("\nSystem Status Report:")
    print(f"Machine Health: {'Healthy' if status.is_healthy else 'Warning'}")
    print(f"Temperature: {status.temperature_status}")
    print(f"Vibration: {status.vibration_status}")
    print(f"Tool Wear: {status.tool_wear_status}")

    if status.anomalies:
        print("\nAnomalies Detected:")
        for param, timestamps in status.anomalies.items():
            print(f"{param}: {len(timestamps)} anomalies detected")

    # final stats
    mean_abs = df['residual'].abs().mean()
    max_abs  = df['residual'].abs().max()
    print(f"\nFinal Model Residuals:\n  Mean Abs Error: {mean_abs:.2f} °C\n  Max Abs Error:  {max_abs:.2f} °C")


if __name__ == "__main__":
    test_full_system()