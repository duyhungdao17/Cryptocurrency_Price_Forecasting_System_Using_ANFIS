import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import Literal
from .ANFIS_Model import compute_metrics    # Import từ ANFIS_Model.py

"""
=============================================================================
FILE: 03_baseline_models.py
MỤC ĐÍCH: Định nghĩa và huấn luyện LSTM và ANN làm baseline so sánh
MÔI TRƯỜNG: Google Colab
=============================================================================
""" 


# ===========================================================================
# PHẦN 1: MODEL DEFINITIONS
# ===========================================================================

class LSTMModel(nn.Module):
    """
    LSTM cho dự báo time series.
    Input shape: (batch_size, sequence_length, n_features)
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, seq_len, n_features)
        lstm_out, _ = self.lstm(x)           # (B, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]        # Lấy output của timestep cuối
        out = self.dropout(last_out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out).squeeze(-1)      # (B,)
        return out


class ANNModel(nn.Module):
    """
    Feedforward ANN (MLP) cho dự báo.
    Input shape: (batch_size, n_features)
    """

    def __init__(self, n_features: int, hidden_sizes: list = None,
                 dropout: float = 0.5):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def create_sequences(X: np.ndarray, y: np.ndarray,
                     seq_len: int) -> tuple:
    """
    Tạo sequences cho LSTM từ flat array.
    Ví dụ: seq_len=7 → mỗi sample là 7 ngày liên tiếp.

    Args:
        X      : (n_samples, n_features)
        y      : (n_samples,)
        seq_len: Độ dài chuỗi

    Returns:
        X_seq: (n_samples - seq_len, seq_len, n_features)
        y_seq: (n_samples - seq_len,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


# ===========================================================================
# PHẦN 2: TRAINING FUNCTION (chung cho LSTM và ANN)
# ===========================================================================

def train_baseline_model(model: nn.Module,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val:   np.ndarray, y_val:   np.ndarray,
                          model_name: str,
                          epochs: int = 100,
                          lr: float = 0.001,
                          batch_size: int = 64,
                          patience: int = 15,
                          save_dir: str = "/content/drive/MyDrive/CI/checkpoint",
                          scaler_y = None,
                          seq_len: int = None):
    """
    Training loop chung cho LSTM và ANN.

    Notes:
        - Nếu model là LSTM: X phải là 3D (B, seq_len, features)
          → Tự động tạo sequences nếu seq_len được cung cấp
        - Nếu model là ANN: X là 2D (B, features)
        - Lưu checkpoint BEST và LAST (cùng dict structure với ANFIS)

    Args:
        seq_len: Chỉ dùng cho LSTM — độ dài chuỗi (None = X đã là 3D)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Tạo sequences cho LSTM nếu cần
    if seq_len is not None and isinstance(model, LSTMModel):
        X_train, y_train = create_sequences(X_train, y_train, seq_len)
        X_val,   y_val   = create_sequences(X_val,   y_val,   seq_len)
        print(f"[LSTM] Sequences tạo xong | "
              f"Train: {X_train.shape} | Val: {X_val.shape}")

    # Chuyển sang Tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.FloatTensor(y_val).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=False    # shuffle=False cho time series
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": [],
        "train_mae" : [], "val_mae" : [],
        "best_epoch": 0
    }

    best_val_loss = float("inf")
    # Early stopping da bi loai bo: chay du toan bo epochs
    # de co loss curve day du cho visualization.

    print(f"\n[TRAINING] {model_name} | Device: {device}")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {batch_size} (no early stopping)")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    print(f"{'─'*65}")

    for epoch in range(1, epochs + 1):
        # -----------------------------------------------------------------
        # Training phase
        # -----------------------------------------------------------------
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        epoch_loss /= len(X_train)

        # -----------------------------------------------------------------
        # Validation phase
        # -----------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        # Tính metrics
        with torch.no_grad():
            train_pred_np = model(X_train_t).cpu().numpy()
        val_pred_np   = val_pred.cpu().numpy()

        train_metrics = compute_metrics(y_train, train_pred_np, scaler_y)
        val_metrics   = compute_metrics(y_val,   val_pred_np,   scaler_y)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_metrics["RMSE"])
        history["val_rmse"].append(val_metrics["RMSE"])
        history["train_mae"].append(train_metrics["MAE"])
        history["val_mae"].append(val_metrics["MAE"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Val RMSE: {val_metrics['RMSE']:.2f} | "
                  f"Val R²: {val_metrics['R2']:.4f}")

        # -----------------------------------------------------------------
        # Checkpoint saving — cùng structure với ANFIS checkpoint
        # -----------------------------------------------------------------
        checkpoint_last = {
            "model_state_dict"     : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "scheduler_state_dict" : scheduler.state_dict(),
            "epoch"      : epoch,
            "val_loss"   : val_loss,
            "val_rmse"   : val_metrics["RMSE"],
            "val_mae"    : val_metrics["MAE"],
            "val_r2"     : val_metrics["R2"],
            "history"    : history,
        }

        torch.save(checkpoint_last,
                   os.path.join(save_dir, f"{model_name}_last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch

            import copy
            torch.save(copy.deepcopy(checkpoint_last),
                       os.path.join(save_dir, f"{model_name}_best.pt"))

    print(f"\n[DONE] {model_name} | Best epoch: {history['best_epoch']} | "
          f"Best val loss: {best_val_loss:.6f}")

    return model, history