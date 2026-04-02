import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Literal
import warnings
warnings.filterwarnings("ignore")

"""
=============================================================================
FILE: 02_anfis_model.py
MỤC ĐÍCH: Triển khai mô hình ANFIS từ đầu (NumPy + PyTorch)
          - Grid Partitioning clustering
          - Fuzzy C-Means (FCM) clustering
          - Hybrid training: BP (Backpropagation) + LSE (Least Squares)
MÔI TRƯỜNG: Google Colab
=============================================================================
"""


# ===========================================================================
# PHẦN 1: CLUSTERING — Khởi tạo cấu trúc ANFIS trước khi training
# ===========================================================================

class GridPartitioning:
    """
    Phân vùng lưới đơn giản nhất.
    Mỗi input được chia đều thành n_mf vùng → Gaussian MF.
    Số rules = n_mf ^ n_inputs (tổ hợp Cartesian).

    KHUYẾN NGHỊ: Dùng khi n_inputs <= 4 và n_mf <= 3.
    """

    def __init__(self, n_mf: int = 2):
        """
        Args:
            n_mf: Số Membership Functions cho mỗi input (2 = Low/High)
        """
        self.n_mf = n_mf

    def initialize(self, X: np.ndarray):
        """
        Khởi tạo centers và sigmas cho Gaussian MF.

        Args:
            X: Training data (n_samples, n_features), đã normalize [0,1]

        Returns:
            centers: (n_features, n_mf) — vị trí trung tâm MF
            sigmas : (n_features, n_mf) — độ rộng MF
        """
        n_features = X.shape[1]

        # Đặt centers đều đặn trong [min, max] của từng feature
        centers = np.zeros((n_features, self.n_mf))
        sigmas  = np.zeros((n_features, self.n_mf))

        for i in range(n_features):
            x_min = X[:, i].min()
            x_max = X[:, i].max()
            # Chia đều khoảng [x_min, x_max] thành n_mf điểm
            centers[i] = np.linspace(x_min, x_max, self.n_mf)
            # Sigma = 1/2 khoảng cách giữa 2 centers liền kề
            if self.n_mf > 1:
                spacing = (x_max - x_min) / (self.n_mf - 1)
            else:
                spacing = x_max - x_min
            sigmas[i] = np.full(self.n_mf, max(spacing / 2, 0.1))

        n_rules = self.n_mf ** n_features
        print(f"[Grid Partitioning] {n_features} inputs × {self.n_mf} MF/input "
              f"= {n_rules} rules")

        return centers, sigmas


class FuzzyCMeans:
    """
    Fuzzy C-Means Clustering.
    Mỗi cluster center → 1 fuzzy rule.
    Một điểm dữ liệu có thể thuộc về nhiều cluster với mức độ khác nhau.

    KẾT QUẢ PAPER 1: FCM + Backpropagation cho RMSE tốt nhất với BTC.
    """

    def __init__(self, n_clusters: int = 5, m: float = 2.0,
                 max_iter: int = 300, tol: float = 1e-6):
        """
        Args:
            n_clusters: Số clusters C = Số fuzzy rules (thường 4–10)
            m         : Fuzziness exponent (thường = 2.0)
            max_iter  : Số vòng lặp tối đa
            tol       : Ngưỡng hội tụ
        """
        self.n_clusters = n_clusters
        self.m          = m
        self.max_iter   = max_iter
        self.tol        = tol
        self.centers_   = None
        self.u_          = None     # Membership matrix

    def fit(self, X: np.ndarray):
        """
        Huấn luyện FCM trên dữ liệu training.

        Args:
            X: (n_samples, n_features)

        Returns:
            self
        """
        n_samples, n_features = X.shape
        C = self.n_clusters
        m = self.m

        # Khởi tạo ngẫu nhiên membership matrix U
        np.random.seed(42)
        U = np.random.dirichlet(np.ones(C), size=n_samples).T
        # U shape: (C, n_samples), mỗi cột tổng = 1

        for iteration in range(self.max_iter):
            U_old = U.copy()

            # Cập nhật cluster centers
            um = U ** m   # (C, n_samples)
            centers = (um @ X) / (um.sum(axis=1, keepdims=True) + 1e-10)
            # centers shape: (C, n_features)

            # Cập nhật membership matrix U
            distances = np.zeros((C, n_samples))
            for k in range(C):
                diff = X - centers[k]
                distances[k] = np.sqrt((diff ** 2).sum(axis=1)) + 1e-10

            new_U = np.zeros((C, n_samples))
            for k in range(C):
                for j in range(C):
                    new_U[k] += (distances[k] / (distances[j] + 1e-10)) ** (2 / (m - 1))
            U = 1.0 / (new_U + 1e-10)

            # Chuẩn hóa U (tổng membership của mỗi sample = 1)
            U = U / (U.sum(axis=0, keepdims=True) + 1e-10)

            # Kiểm tra hội tụ
            if np.max(np.abs(U - U_old)) < self.tol:
                print(f"  [FCM] Hội tụ sau {iteration + 1} vòng lặp")
                break

        self.centers_ = centers     # (C, n_features)
        self.u_        = U           # (C, n_samples)

        return self

    def initialize_mf(self, X: np.ndarray):
        """
        Chuyển cluster centers thành tham số Gaussian MF.

        Returns:
            centers: (n_features, n_clusters)
            sigmas : (n_features, n_clusters)
        """
        if self.centers_ is None:
            self.fit(X)

        C          = self.n_clusters
        n_features = X.shape[1]

        centers = self.centers_.T   # (n_features, C)

        # Sigma: dựa trên khoảng cách giữa các cluster centers
        sigmas = np.zeros((n_features, C))
        for i in range(n_features):
            for k in range(C):
                # Sigma = trung bình khoảng cách đến cluster lân cận / 2
                dists = [abs(centers[i, k] - centers[i, j])
                         for j in range(C) if j != k]
                sigmas[i, k] = max(np.mean(dists) / 2, 0.05)

        n_rules = C
        print(f"[FCM] {n_features} inputs × {C} clusters = {n_rules} rules")

        return centers, sigmas


# ===========================================================================
# PHẦN 2: ANFIS MODEL (PyTorch)
# ===========================================================================

class GaussianMF(nn.Module):
    """
    Gaussian Membership Function: μ(x) = exp(-((x - c)² / (2σ²)))

    Parameters (learnable):
        centers: vị trí trung tâm (c)
        sigmas : độ rộng (σ)

    Được cập nhật bởi Backpropagation.
    """

    def __init__(self, n_inputs: int, n_mf: int,
                 init_centers: np.ndarray = None,
                 init_sigmas:  np.ndarray = None):
        """
        Args:
            n_inputs    : Số features đầu vào
            n_mf        : Số MF cho mỗi input
            init_centers: (n_inputs, n_mf) — từ clustering
            init_sigmas : (n_inputs, n_mf) — từ clustering
        """
        super().__init__()

        if init_centers is not None:
            c_init = torch.FloatTensor(init_centers)
        else:
            c_init = torch.rand(n_inputs, n_mf)

        if init_sigmas is not None:
            s_init = torch.FloatTensor(init_sigmas).clamp(min=0.01)
        else:
            s_init = torch.ones(n_inputs, n_mf) * 0.5

        # nn.Parameter → PyTorch tự tính gradient và cập nhật bằng BP
        self.centers = nn.Parameter(c_init)
        self.sigmas  = nn.Parameter(s_init)

        self.n_inputs = n_inputs
        self.n_mf     = n_mf

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_inputs)

        Returns:
            mu: (batch_size, n_inputs, n_mf) — membership degrees
        """
        # x: (B, n_inputs) → (B, n_inputs, 1)
        x_expanded = x.unsqueeze(2)

        # centers: (n_inputs, n_mf) → (1, n_inputs, n_mf)
        c = self.centers.unsqueeze(0)
        s = self.sigmas.unsqueeze(0).clamp(min=1e-4)

        # Gaussian MF
        mu = torch.exp(-((x_expanded - c) ** 2) / (2 * s ** 2))
        # mu: (B, n_inputs, n_mf)

        return mu


class ANFIS(nn.Module):
    """
    ANFIS (Adaptive Neuro-Fuzzy Inference System) — Sugeno Type
    Kiến trúc 5 lớp theo Jang (1993).

    Training: Hybrid Algorithm
      - Layer 1 parameters (centers, sigmas): Backpropagation
      - Layer 4 parameters (p, q, r):         LSE (Least Squares Estimation)
    """

    def __init__(self, n_inputs:     int,
                 n_mf:        int,
                 clustering:  Literal["grid", "fcm"] = "fcm",
                 n_clusters:  int = None,
                 X_train:     np.ndarray = None):
        """
        Args:
            n_inputs   : Số features đầu vào
            n_mf       : Số MF/input (cho Grid) hoặc số clusters (cho FCM)
            clustering : "grid" hoặc "fcm"
            n_clusters : Số clusters cho FCM (override n_mf nếu khác None)
            X_train    : Training data để khởi tạo clustering
        """
        super().__init__()

        self.n_inputs   = n_inputs
        self.clustering = clustering

        # -------------------------------------------------------------------
        # BƯỚC 0: Clustering — Khởi tạo cấu trúc
        # -------------------------------------------------------------------
        if clustering == "grid":
            clusterer = GridPartitioning(n_mf=n_mf)
            self.n_rules = n_mf ** n_inputs
            self.n_mf    = n_mf
        elif clustering == "fcm":
            c = n_clusters if n_clusters is not None else n_mf
            clusterer    = FuzzyCMeans(n_clusters=c)
            self.n_rules = c
            self.n_mf    = None   # FCM không dùng n_mf theo nghĩa Grid
        else:
            raise ValueError(f"clustering phải là 'grid' hoặc 'fcm', nhận: {clustering}")

        if X_train is not None:
            init_centers, init_sigmas = clusterer.initialize(X_train) \
                if clustering == "grid" else clusterer.initialize_mf(X_train)
        else:
            init_centers = None
            init_sigmas  = None

        # Với FCM: n_mf trong GaussianMF = số clusters
        actual_n_mf = (n_mf if clustering == "grid"
                       else (n_clusters if n_clusters else n_mf))

        # -------------------------------------------------------------------
        # LAYER 1: Fuzzification
        # -------------------------------------------------------------------
        self.fuzz_layer = GaussianMF(
            n_inputs    = n_inputs,
            n_mf        = actual_n_mf,
            init_centers= init_centers,
            init_sigmas = init_sigmas
        )

        # -------------------------------------------------------------------
        # LAYER 2 & 3: Rule và Normalization
        # Được tính trong forward() — không có learnable parameters
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # LAYER 4: Defuzzification — Consequent Parameters
        # {p_i, q_i, r_i} trong f_i = p_i*x1 + q_i*x2 + ... + r_i
        # Cập nhật bằng LSE (không phải gradient descent)
        # -------------------------------------------------------------------
        # Tham số: n_rules × (n_inputs + 1)
        # +1 cho bias term r_i
        self.consequent_params = nn.Parameter(
            torch.zeros(self.n_rules, n_inputs + 1),
            requires_grad=False   # LSE cập nhật, không cần gradient
        )

        print(f"\n[ANFIS] Khởi tạo:")
        print(f"  Clustering  : {clustering}")
        print(f"  n_inputs    : {n_inputs}")
        print(f"  n_mf/input  : {actual_n_mf}")
        print(f"  n_rules     : {self.n_rules}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()):,}")

    def _build_rule_indices(self):
        """
        Tạo tổ hợp rule indices cho Grid Partitioning.
        Ví dụ: 2 inputs, 2 MF → [(0,0), (0,1), (1,0), (1,1)]
        """
        from itertools import product
        n_mf = self.fuzz_layer.n_mf
        return list(product(range(n_mf), repeat=self.n_inputs))

    def forward(self, x):
        """
        Forward pass qua 5 lớp của ANFIS.

        Args:
            x: (batch_size, n_inputs) — normalized input

        Returns:
            output: (batch_size,) — predicted value
        """
        B = x.shape[0]

        # -------------------------------------------------------------------
        # LAYER 1: Fuzzification
        # mu: (B, n_inputs, n_mf_per_input)
        # -------------------------------------------------------------------
        mu = self.fuzz_layer(x)

        # -------------------------------------------------------------------
        # LAYER 2: Rule Layer — Firing Strength
        # w_i = Π μ_Ai(x_j) cho mỗi rule i
        # -------------------------------------------------------------------
        if self.clustering == "grid":
            # Grid: tổ hợp Cartesian của MF indices
            # FIX: Dùng torch.stack thay vì inplace assignment để tránh
            # "RuntimeError: one of the variables needed for gradient
            # computation has been modified by an inplace operation"
            rule_indices = self._build_rule_indices()
            w_list = []
            for mf_combo in rule_indices:
                # Khởi đầu từ MF của input đầu tiên (không tạo ones trung gian)
                rule_w = mu[:, 0, mf_combo[0]]
                for input_idx in range(1, len(mf_combo)):
                    # Nhân từng MF degree — pure element-wise, KHÔNG inplace
                    rule_w = rule_w * mu[:, input_idx, mf_combo[input_idx]]
                w_list.append(rule_w)
            # Stack thành (B, n_rules) — giữ nguyên computation graph
            w = torch.stack(w_list, dim=1)
        else:
            # FCM: mỗi cluster là 1 rule
            # mu: (B, n_inputs, n_clusters) → prod theo dim=1 → (B, n_clusters)
            w = mu.prod(dim=1)

        # -------------------------------------------------------------------
        # LAYER 3: Normalization
        # w̄_i = w_i / Σw_j
        # -------------------------------------------------------------------
        w_sum = w.sum(dim=1, keepdim=True) + 1e-10
        w_norm = w / w_sum   # (B, n_rules)

        # -------------------------------------------------------------------
        # LAYER 4: Defuzzification
        # f_i = p_i*x1 + q_i*x2 + ... + r_i (linear function)
        # O4_i = w̄_i × f_i
        # -------------------------------------------------------------------
        # Thêm bias term vào x: (B, n_inputs) → (B, n_inputs+1)
        ones  = torch.ones(B, 1, device=x.device)
        x_aug = torch.cat([x, ones], dim=1)   # (B, n_inputs+1)

        # f: (B, n_rules) = x_aug @ consequent_params.T
        f = x_aug @ self.consequent_params.T  # (B, n_rules)

        # Weighted output
        wf = w_norm * f   # (B, n_rules)

        # -------------------------------------------------------------------
        # LAYER 5: Summation — Final Output
        # -------------------------------------------------------------------
        output = wf.sum(dim=1)   # (B,)

        return output, w_norm    # Trả về cả normalized weights cho phân tích

    def get_firing_matrix(self, x):
        """
        Trả về normalized firing strengths — hữu ích để phân tích rules.
        """
        with torch.no_grad():
            _, w_norm = self.forward(x)
        return w_norm.cpu().numpy()


# ===========================================================================
# PHẦN 3: TRAINING — Hybrid Algorithm (BP + LSE)
# ===========================================================================

def lse_update_consequent(anfis_model: ANFIS,
                            X: torch.Tensor,
                            y: torch.Tensor) -> float:
    """
    LSE (Least Squares Estimation) — cập nhật consequent parameters {p, q, r}.

    Giải hệ phương trình: A @ θ = B
      A: ma trận hệ số (n_samples, n_rules × (n_inputs+1))
      θ: consequent parameters vector
      B: target values

    Dùng pseudoinverse: θ = pinv(A) @ B

    Returns:
        lse_loss: MSE sau khi cập nhật
    """
    anfis_model.eval()
    with torch.no_grad():
        B_size, n_inputs = X.shape[0], anfis_model.n_inputs
        n_rules = anfis_model.n_rules

        # Tính w_norm từ forward pass
        _, w_norm = anfis_model(X)   # w_norm: (B, n_rules)

        # Xây dựng ma trận A
        ones  = torch.ones(B_size, 1, device=X.device)
        x_aug = torch.cat([X, ones], dim=1)   # (B, n_inputs+1)

        # A[i, rule*(n_inputs+1) : (rule+1)*(n_inputs+1)] = w_norm[i, rule] * x_aug[i]
        A = torch.zeros(B_size, n_rules * (n_inputs + 1), device=X.device)
        for rule in range(n_rules):
            start = rule * (n_inputs + 1)
            end   = start + (n_inputs + 1)
            A[:, start:end] = w_norm[:, rule:rule+1] * x_aug

        # LSE: θ = pinv(A) @ y
        A_np = A.cpu().numpy()
        y_np = y.cpu().numpy()

        # Regularized pseudoinverse để tránh singular matrix
        theta, _, _, _ = np.linalg.lstsq(A_np, y_np, rcond=None)

        # Cập nhật consequent parameters
        theta_reshaped = torch.FloatTensor(
            theta.reshape(n_rules, n_inputs + 1)
        ).to(X.device)
        anfis_model.consequent_params.data = theta_reshaped

        # Tính loss sau cập nhật
        y_pred, _ = anfis_model(X)
        lse_loss  = ((y_pred - y) ** 2).mean().item()

    return lse_loss


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scaler_y=None) -> dict:
    """
    Tính toán đầy đủ metrics. Nếu scaler_y được cung cấp,
    tính metrics trên giá trị gốc (inverse transform).

    Returns:
        dict: {"RMSE": ..., "MAE": ..., "MAPE": ..., "RMSRE": ..., "R2": ...}
    """
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    eps  = 1e-10
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    rmsre = np.sqrt(np.mean(((y_true - y_pred) / (y_true + eps)) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / (ss_tot + eps)

    return {
        "RMSE" : round(rmse, 6),
        "MAE"  : round(mae,  6),
        "MAPE" : round(mape, 4),
        "RMSRE": round(rmsre, 6),
        "R2"   : round(r2,   4),
    }


def train_anfis(anfis_model: ANFIS,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val:   np.ndarray, y_val:   np.ndarray,
                epochs: int = 100,
                lr: float = 0.001,
                batch_size: int = 64,
                patience: int = 15,
                save_dir: str = "../Checkpoints",
                model_name: str = "anfis",
                scaler_y = None):
    """
    Huấn luyện ANFIS với Hybrid Algorithm: BP + LSE.

    Mỗi epoch:
      1. [LSE]  Cập nhật consequent parameters (Layer 4)
      2. [BP]   Cập nhật premise parameters (centers, sigmas ở Layer 1)

    Args:
        anfis_model: ANFIS instance
        X_train, y_train: Training data (numpy, đã normalized)
        X_val, y_val    : Validation data
        epochs          : Số epochs tối đa
        lr              : Learning rate cho Backpropagation
        batch_size      : Batch size
        patience        : Early stopping patience
        save_dir        : Thư mục lưu checkpoint
        model_name      : Tên file checkpoint
        scaler_y        : Để tính metrics trên giá trị gốc

    Returns:
        history: dict với loss và metrics theo epoch
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anfis_model = anfis_model.to(device)

    # Chuyển sang Tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.FloatTensor(y_val).to(device)

    # DataLoader cho training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # QUAN TRỌNG: shuffle=False cho time series!

    # Optimizer cho BP (chỉ cập nhật premise parameters)
    premise_params = [anfis_model.fuzz_layer.centers,
                      anfis_model.fuzz_layer.sigmas]
    optimizer = optim.Adam(premise_params, lr=lr, weight_decay=1e-5)

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
    print(f"  Training samples: {len(X_train):,} | Val samples: {len(X_val):,}")
    print(f"{'─'*65}")

    for epoch in range(1, epochs + 1):

        anfis_model.train()
        epoch_train_loss = 0.0

        # -----------------------------------------------------------------
        # BƯỚC 1: LSE — Cập nhật toàn bộ training data một lần
        # (LSE dùng toàn bộ batch, không chia nhỏ)
        # -----------------------------------------------------------------
        lse_loss = lse_update_consequent(anfis_model, X_train_t, y_train_t)

        # -----------------------------------------------------------------
        # BƯỚC 2: BP — Cập nhật premise parameters theo mini-batch
        # -----------------------------------------------------------------
        anfis_model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred, _ = anfis_model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # Gradient clipping để tránh exploding gradient
            nn.utils.clip_grad_norm_(premise_params, max_norm=1.0)
            optimizer.step()

            # Clamp sigmas để luôn dương
            with torch.no_grad():
                anfis_model.fuzz_layer.sigmas.data.clamp_(min=1e-4)

            epoch_train_loss += loss.item() * len(X_batch)

        epoch_train_loss /= len(X_train)

        # -----------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------
        anfis_model.eval()
        with torch.no_grad():
            val_pred, _ = anfis_model(X_val_t)
            val_loss    = criterion(val_pred, y_val_t).item()

        # Tính metrics (trên giá trị gốc nếu có scaler)
        # FIX: Wrap trong no_grad để tránh tạo gradient graph không cần thiết
        with torch.no_grad():
            train_pred_np = anfis_model(X_train_t)[0].cpu().numpy()
        val_pred_np   = val_pred.cpu().numpy()
        y_train_np    = y_train
        y_val_np      = y_val

        train_metrics = compute_metrics(y_train_np, train_pred_np, scaler_y)
        val_metrics   = compute_metrics(y_val_np,   val_pred_np,   scaler_y)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_metrics["RMSE"])
        history["val_rmse"].append(val_metrics["RMSE"])
        history["train_mae"].append(train_metrics["MAE"])
        history["val_mae"].append(val_metrics["MAE"])

        # -----------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val RMSE: {val_metrics['RMSE']:.2f} | "
                  f"Val MAE: {val_metrics['MAE']:.2f} | "
                  f"Val R²: {val_metrics['R2']:.4f}")

        # -----------------------------------------------------------------
        # Lưu checkpoint
        # -----------------------------------------------------------------
        # Checkpoint = model state + optimizer state + metadata trong 1 dict
        checkpoint_last = {
            # Model state
            "model_state_dict"     : anfis_model.state_dict(),
            # Optimizer state (để resume training)
            "optimizer_state_dict" : optimizer.state_dict(),
            # Metadata
            "epoch"      : epoch,
            "val_loss"   : val_loss,
            "val_rmse"   : val_metrics["RMSE"],
            "val_mae"    : val_metrics["MAE"],
            "val_r2"     : val_metrics["R2"],
            "history"    : history,
            # Config để tái tạo model
            "model_config": {
                "n_inputs"   : anfis_model.n_inputs,
                "n_rules"    : anfis_model.n_rules,
                "clustering" : anfis_model.clustering,
            }
        }

        # Lưu checkpoint cuối cùng (luôn luôn)
        torch.save(checkpoint_last,
                   os.path.join(save_dir, f"{model_name}_last.pt"))

        # Lưu checkpoint tốt nhất (cùng format)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch

            # FIX: Dùng copy.deepcopy thay vì .copy() (shallow copy)
            # Shallow copy khiến model_state_dict trỏ đến cùng tensor objects
            # → best checkpoint bị ghi đè khi model tiếp tục training
            import copy
            checkpoint_best = copy.deepcopy(checkpoint_last)
            torch.save(checkpoint_best,
                       os.path.join(save_dir, f"{model_name}_best.pt"))

            if epoch % 10 == 0 or epoch == 1:
                print(f"  ✓ Best checkpoint saved (val_loss: {val_loss:.6f})")

    print(f"\n[TRAINING DONE] Best epoch: {history['best_epoch']} | "
          f"Best val loss: {best_val_loss:.6f}")

    return history


def load_checkpoint(path: str, anfis_model: ANFIS,
                    optimizer=None, device: str = "cpu"):
    """
    Load checkpoint để tiếp tục training hoặc inference.

    Args:
        path        : Đường dẫn file .pt
        anfis_model : ANFIS instance (cùng config khi save)
        optimizer   : Optimizer instance (None nếu chỉ inference)
        device      : "cpu" hoặc "cuda"

    Returns:
        checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    anfis_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[LOAD] Checkpoint từ: {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')} | "
          f"Val RMSE: {checkpoint.get('val_rmse', 'N/A'):.4f} | "
          f"Val R²: {checkpoint.get('val_r2', 'N/A'):.4f}")

    return checkpoint


# ===========================================================================
# PHẦN 4: QUICK TEST — Kiểm tra mô hình chạy đúng (dùng Minimal Features)
# ===========================================================================

def quick_test():
    """
    Kiểm tra nhanh ANFIS với dữ liệu giả để verify pipeline.
    Dùng minimal feature set (4 features) — chạy nhanh, kết quả sớm.
    """
    print("\n" + "="*60)
    print("  QUICK TEST — Verify ANFIS Pipeline")
    print("="*60)

    np.random.seed(42)
    n_samples  = 500
    n_features = 4   # Minimal: [close, volume, RSI14, MA7]

    # Dữ liệu giả đã normalize [0, 1]
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.rand(n_samples).astype(np.float32)

    # Split
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # -----------------------------------------------------------------
    # Test với Grid Partitioning (n_inputs=4, n_mf=2 → 16 rules)
    # -----------------------------------------------------------------
    print("\n--- Grid Partitioning (2 MF/input, 16 rules) ---")
    anfis_grid = ANFIS(
        n_inputs   = n_features,
        n_mf       = 2,
        clustering = "grid",
        X_train    = X_train
    )
    history_grid = train_anfis(
        anfis_grid, X_train, y_train, X_val, y_val,
        epochs=20, lr=0.01, batch_size=32, patience=10,
        save_dir="../Checkpoints",
        model_name="anfis_grid_quicktest"
    )

    # -----------------------------------------------------------------
    # Test với FCM (5 clusters = 5 rules)
    # -----------------------------------------------------------------
    print("\n--- FCM Clustering (5 rules) ---")
    anfis_fcm = ANFIS(
        n_inputs   = n_features,
        n_mf       = 5,
        clustering = "fcm",
        n_clusters = 5,
        X_train    = X_train
    )
    history_fcm = train_anfis(
        anfis_fcm, X_train, y_train, X_val, y_val,
        epochs=20, lr=0.01, batch_size=32, patience=10,
        save_dir="../Checkpoints",
        model_name="anfis_fcm_quicktest"
    )

    print("\n[QUICK TEST PASSED] ✓ Cả 2 cấu hình chạy thành công")
    return anfis_grid, anfis_fcm


if __name__ == "__main__":
    quick_test()