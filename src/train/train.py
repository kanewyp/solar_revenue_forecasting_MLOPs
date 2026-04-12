"""
PyTorch LSTM training for solar revenue forecasting.
Architecture: sliding window (seq_len days) → LSTM → dense → predicted revenue.

Key MLOps patterns demonstrated:
  - PyTorch Dataset + DataLoader (decoupled from training logic)
  - Adam optimizer + ReduceLROnPlateau scheduler
  - Gradient clipping (prevent exploding gradients in LSTM)
  - Early stopping
  - MLflow: pytorch autolog + manual artifact logging
  - Saves model as mlflow.pytorch artifact for registry
"""
import json
import subprocess
import yaml

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# PyTorch Dataset

class SolarSequenceDataset(Dataset):
    """
    Converts a flat time-series DataFrame into (sequence, target) pairs.
    Each sample: X = features[t-seq_len : t], y = target[t]

    This is the correct pattern from the PDF:
      - decouples data loading from training logic
      - handles windowing/sequence creation
      - standardizes data format
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y
    
# LSTM model

class SolarRevenueLSTM(nn.Module):
    """
    2-layer LSTM with dropout for solar revenue forecasting.
    Input:  (batch, seq_len, input_size)
    Output: (batch, 1) — predicted daily revenue in USD
    """
    def __init__(self, input_size:int, hidden_size:int,
                 num_layers:int, dropout:float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x) # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden).squeeze(-1)
    
    
# MLFLOW PyTorch wrapper for registry

class LSTMWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps the LSTM + scaler into a single MLflow pyfunc model.
    This means inference code (FastAPI) never needs to know it's PyTorch —
    it just calls model.predict(df) the same way as any other MLflow model.
    """
    def __init__(self, model: SolarRevenueLSTM, scaler: StandardScaler,
                 seq_len: int, feature_cols: list[str]):
        self.model = model
        self.scaler = scaler
        self.seq_len = seq_len
        self.feature_cols = feature_cols

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """
        model_input: DataFrame with exactly seq_len rows of features.
        Returns: array of shape (1,) — predicted revenue for the next day.
        """
        if len(model_input) != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} rows, got {len(model_input)}")
        features = model_input[self.feature_cols].values
        scaled = self.scaler.transform(features)
        x = torch.FloatTensor(scaled).unsqueeze(0) # (1, seq_len, n_features)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).item()
        return np.array([pred])
    

# Helpers

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except Exception:
        return "unknown"

def mape(y_true, y_pred) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# Training loop

def train_model():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    cfg = config['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # load + scale data
    df = pd.read_csv("data/processed/features.csv")
    feature_cols = cfg['feature_cols']
    target_col = cfg['target_col']

    # time-based split (NO SHUFFLE)
    split_idx = int(len(df) * (1 - cfg['test_size']))
    X_all = df[feature_cols].values
    y_all = df[target_col].values

    # Fit scaler on training data only - never fit on val/test
    scaler = StandardScaler()
    X_train_raw = X_all[: split_idx]
    X_val_raw = X_all[split_idx:]
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # Also scale targets for stable training, inverse-transform for metrics
    y_mean, y_std = y_all[:split_idx].mean(), y_all[:split_idx].std()
    y_train = (y_all[:split_idx] - y_mean) / y_std
    y_val   = (y_all[split_idx:]  - y_mean) / y_std

    # datasets + dataloaders
    train_ds = SolarSequenceDataset(X_train, y_train, cfg["seq_len"])
    val_ds = SolarSequenceDataset(X_val, y_val, cfg["seq_len"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    print(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

    # model, optimizer, scheduler
    model = SolarRevenueLSTM(
        input_size = len(feature_cols),
        hidden_size = cfg["hidden_size"],
        num_layers = cfg["num_layers"],
        dropout = cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["lr_patience"], factor=cfg["lr_factor"]
    )
    criterion = nn.MSELoss()

    # MLFlow run
    # 1. Set Experiment
    mlflow.set_experiment("solar-revenue-forecasting")
    mlflow.pytorch.autolog(log_models=False)
    # 2. Start Run
    with mlflow.start_run(run_name=f"lstm-h{cfg['hidden_size']}-l{cfg['num_layers']}"):
        # 3. Log static params once
        mlflow.log_params({
            "model_type": "LSTM",
            "seq_len": cfg["seq_len"],
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "grad_clip": cfg["grad_clip"],
            "n_features": len(feature_cols),
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "git_commit": get_git_commit(),
            "device": str(device),
        })
        
        # Training Loop
        # track best checkpoint
        best_val_loss = float("inf")
        best_state = None 
        patience_counter = 0

        # 4. Training Phase
        for epoch in range(cfg['epochs']):
            # Train
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                # Gradient cliping - prevents exploding gradients in LSTMs
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    val_losses.append(criterion(preds, y_batch).item())
                
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            current_lr = optimizer.param_groups[0]["lr"]

            # 5. Log step metrics for each epoch
            mlflow.log_metrics({
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "learning_rate" : current_lr
            }, step=epoch)
            
            # 6. Set Scheduler
            scheduler.step(val_loss)

            
            # track and update the best checkpoint in memory
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else: # Early stopping
                patience_counter += 1
                if patience_counter >= cfg["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} "
                      f"| val_loss={val_loss:.4f} | lr={current_lr:.6f}")

        # Restore best weights once after training loop.
        if best_state is not None:
            model.load_state_dict(best_state)

        # Compute interpretable metrics on original scale
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch.to(device))
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.numpy())

        preds_scaled = np.concatenate(all_preds)
        target_scaled = np.concatenate(all_targets)

        # Inverse-transform to USD
        preds_usd = preds_scaled * y_std + y_mean
        targets_usd = target_scaled * y_std + y_mean

        val_mape = mape(targets_usd, preds_usd)
        val_rmse = rmse(targets_usd, preds_usd)
        val_r2 = float(
            1 - np.sum((targets_usd - preds_usd) ** 2) /
            np.sum((targets_usd - targets_usd.mean()) ** 2)
        )
        
        # log final metrics
        mlflow.log_metrics({
            "val_mape": val_mape,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "best_val_loss": best_val_loss,
        })
        print(f"\nFinal - val_mape={val_mape:.2f}% | val_rmse=${val_rmse:,.0f} | val_r2={val_r2:.3f}")

        # Log model via pyfunc wrapper
        # Logging Artifacts using a 
        wrapper = LSTMWrapper(model.cpu(), scaler, cfg["seq_len"], feature_cols)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            pip_requirements=["torch==2.2.0", "scikit-learn==1.4.0",
                              "pandas==2.1.4", "numpy==1.26.3"],
        )

        # Write metrics.json for DVC
        metrics = {"val_mape": val_mape, "val_rmse": val_rmse, "val_r2": val_r2}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        return run_id, val_mape
        

if __name__ == "__main__":
    run_id, val_mape = train_model()
    print(f"\nDone. Run ID: {run_id} | MAPE: {val_mape:.2f}%")
