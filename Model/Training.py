import numpy as np
import pandas as pd
import torch
import os
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .CONFIG import CONFIG 
from Crawling import create_feature_sets, prepare_dataset
from .ANFIS_Model import ANFIS, train_anfis, compute_metrics, load_checkpoint, lse_update_consequent
from .Base_Model import LSTMModel, ANNModel, train_baseline_model, create_sequences

# ===========================================================================
# PHẦN 1: LOAD DATA
# ===========================================================================

def load_prepared_data(symbol: str, feature_set: str,
                        data_dir: str) -> tuple:
    """
    Load CSV features đã được tạo bởi 01_data_collection.py,
    sau đó chạy lại prepare_dataset để split + scale.
    """
    # from data_collection import create_feature_sets, prepare_dataset

    feat_path = os.path.join(data_dir, f"{symbol}_features.csv")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    feature_sets = create_feature_sets(df)
    feature_cols = feature_sets[feature_set]

    result = prepare_dataset(
        df, feature_cols=feature_cols,
        target_col="target",
        train_ratio=0.70, val_ratio=0.10, scale=True
    )

    X_train, X_val, X_test = result[0], result[1], result[2]
    y_train, y_val, y_test = result[3], result[4], result[5]
    scaler_X, scaler_y     = result[6], result[7]

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_X, scaler_y, feature_cols)


# ===========================================================================
# PHẦN 2: RÀNG BUỘC SỐ RULES CHO GRID PARTITIONING
# ===========================================================================

def check_grid_feasibility(n_inputs: int, n_mf: int,
                            max_rules: int = 1024) -> tuple:
    """
    Kiểm tra xem Grid Partitioning có khả thi không dựa trên số rules.

    Vấn đề gốc rễ: Grid tạo ra n_mf^n_inputs rules (tổ hợp Cartesian).
      - 4  inputs × 2 MF = 2^4  =         16 rules   OK
      - 10 inputs × 2 MF = 2^10 =      1,024 rules   Giới hạn
      - 27 inputs × 2 MF = 2^27 = 134,217,728 rules  Crash OOM
      - 80 inputs × 2 MF = 2^80 = ~1.2 x 10^24       Không thể

    Args:
        n_inputs : Số features đầu vào
        n_mf     : Số MF mỗi input
        max_rules: Ngưỡng tối đa (mặc định 1024)

    Returns:
        (feasible: bool, n_rules: int, message: str)
    """
    n_rules = n_mf ** n_inputs

    if n_rules <= max_rules:
        msg = (f"[Grid] {n_inputs} inputs x {n_mf} MF/input = "
               f"{n_rules:,} rules  OK (ngưỡng: {max_rules:,})")
        return True, n_rules, msg
    else:
        msg = (f"[Grid] {n_inputs} inputs x {n_mf} MF/input = "
               f"{n_rules:,} rules  VƯỢT NGƯỠNG ({max_rules:,})\n"
               f"  -> Bỏ qua Grid Partitioning, chỉ dùng FCM.")
        return False, n_rules, msg


def suggest_safe_n_mf(n_inputs: int, max_rules: int = 1024) -> int:
    """
    Với số inputs cho trước, tìm n_mf lớn nhất sao cho
    n_mf^n_inputs <= max_rules.

    Ví dụ: n_inputs=10, max_rules=1024
      -> n_mf=2 vì 2^10=1024 <= 1024  OK
      -> n_mf=3 vì 3^10=59049 > 1024  không OK

    Returns:
        n_mf_safe: int — trả về 0 nếu ngay cả n_mf=2 cũng vượt ngưỡng
                   (n_mf=1 vô nghĩa vì chỉ tạo ra 1 rule duy nhất)
    """
    for n_mf in range(10, 1, -1):    # Bắt đầu từ 10, xuống đến 2 (không thử 1)
        if n_mf ** n_inputs <= max_rules:
            return n_mf
    # n_mf=2 cũng vượt ngưỡng → không thể dùng Grid
    return 0


# ===========================================================================
# PHẦN 3: EVALUATION
# ===========================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_type: str, scaler_y,
                   seq_len: int = None) -> dict:
    """
    Đánh giá model trên test set, trả về dict metrics đầy đủ.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    if model_type == "lstm" and seq_len is not None:
        X_seq, y_true = create_sequences(X_test, y_test, seq_len)
        X_input = torch.FloatTensor(X_seq).to(device)
    else:
        X_input = torch.FloatTensor(X_test).to(device)
        y_true  = y_test

    with torch.no_grad():
        y_pred_t = model(X_input)[0] if model_type == "anfis" \
                   else model(X_input)

    return compute_metrics(y_true, y_pred_t.cpu().numpy(), scaler_y)


# ===========================================================================
# PHẦN 4: OPTIMIZER — PSO CHO ANFIS
# ===========================================================================

def optimize_anfis_pso(anfis_model: ANFIS,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val:   np.ndarray, y_val:   np.ndarray,
                        n_particles: int = 20,
                        iters: int = 30,
                        save_dir: str = "../Checkpoints",
                        model_name: str = "anfis_pso",
                        scaler_y=None) -> tuple:
    """
    PSO tối ưu premise parameters của ANFIS (đã fix device mismatch).
    """
    try:
        import pyswarms as ps
    except ImportError:
        print("[WARN] Chưa cài pyswarms: pip install pyswarms")
        return anfis_model, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anfis_model = anfis_model.to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.FloatTensor(y_val).to(device)

    with torch.no_grad():
        c_flat = anfis_model.fuzz_layer.centers.data.cpu().numpy().ravel()
        s_flat = anfis_model.fuzz_layer.sigmas.data.cpu().numpy().ravel()

    n_dims    = len(c_flat) + len(s_flat)
    n_centers = len(c_flat)
    bounds    = (np.zeros(n_dims), np.ones(n_dims))

    def pso_objective(swarm):
        costs = []
        for p in swarm:
            pc  = p[:n_centers].reshape(anfis_model.n_inputs, -1)
            ps_ = np.abs(p[n_centers:]).reshape(anfis_model.n_inputs, -1) + 1e-4

            with torch.no_grad():
                anfis_model.fuzz_layer.centers.data = torch.FloatTensor(pc).to(device)
                anfis_model.fuzz_layer.sigmas.data  = torch.FloatTensor(ps_).to(device)

            lse_update_consequent(anfis_model, X_tr, y_tr)

            anfis_model.eval()
            with torch.no_grad():
                vp, _ = anfis_model(X_vl)
                costs.append(((vp - y_vl) ** 2).mean().item())
        return np.array(costs)

    print(f"\n[PSO] {n_particles} particles x {iters} iters | Search space: {n_dims} dims")

    opt = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=n_dims,
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        bounds=bounds
    )
    best_cost, best_params = opt.optimize(pso_objective, iters=iters)

    # === SỬA LỖI Ở ĐÂY ===
    pc  = best_params[:n_centers].reshape(anfis_model.n_inputs, -1)
    ps_ = np.abs(best_params[n_centers:]).reshape(anfis_model.n_inputs, -1) + 1e-4

    with torch.no_grad():
        anfis_model.fuzz_layer.centers.data = torch.FloatTensor(pc).to(device)
        anfis_model.fuzz_layer.sigmas.data  = torch.FloatTensor(ps_).to(device)

    lse_update_consequent(anfis_model, X_tr, y_tr)

    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        vm = compute_metrics(y_val, anfis_model(X_vl)[0].cpu().numpy(), scaler_y)

    ckpt = {
        "model_state_dict"   : anfis_model.state_dict(),
        "val_loss"           : best_cost,
        "val_rmse"           : vm["RMSE"],
        "val_mae"            : vm["MAE"],
        "val_r2"             : vm["R2"],
        "optimizer_type"     : "PSO",
        "pso_config"         : {"n_particles": n_particles, "iters": iters},
        "model_config"       : {
            "n_inputs"  : anfis_model.n_inputs,
            "n_rules"   : anfis_model.n_rules,
            "clustering": anfis_model.clustering,
        }
    }
    torch.save(ckpt, os.path.join(save_dir, f"{model_name}_best.pt"))
    print(f"[PSO DONE] cost={best_cost:.6f} | Val RMSE={vm['RMSE']:.4f}")
    return anfis_model, ckpt


# ===========================================================================
# PHẦN 5: BẢNG SO SÁNH
# ===========================================================================

def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({
            "Model"    : name,
            "RMSE"     : f"{m.get('RMSE',  0):.4f}",
            "MAE"      : f"{m.get('MAE',   0):.4f}",
            "MAPE(%)"  : f"{m.get('MAPE',  0):.2f}",
            "RMSRE"    : f"{m.get('RMSRE', 0):.6f}",
            "R2"       : f"{m.get('R2',    0):.4f}",
        })
    df = pd.DataFrame(rows)
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    best_r = min(results.items(), key=lambda x: x[1].get("RMSE", 999))
    best_2 = max(results.items(), key=lambda x: x[1].get("R2",  -999))
    print(f"\n  Best RMSE: {best_r[0]} ({best_r[1]['RMSE']:.4f})")
    print(f"  Best R2  : {best_2[0]} ({best_2[1]['R2']:.4f})")
    return df


def decision_flow(anfis_metrics: dict, lstm_metrics: dict,
                  ann_metrics: dict) -> str:
    ar, lr, ar2 = anfis_metrics["RMSE"], lstm_metrics["RMSE"], ann_metrics["RMSE"]
    bl, ba = ar < lr, ar < ar2
    print(f"\n[DECISION FLOW]")
    print(f"  ANFIS RMSE: {ar:.4f}")
    print(f"  LSTM  RMSE: {lr:.4f}  | ANFIS {'TOT HON' if bl else 'KEM HON'}")
    print(f"  ANN   RMSE: {ar2:.4f} | ANFIS {'TOT HON' if ba else 'KEM HON'}")
    if bl and ba:
        print("\n  -> ANFIS tot hon ca LSTM va ANN -> Them PSO de tim ANFIS tot nhat")
        return "better"
    elif not bl and not ba:
        print("\n  -> ANFIS kem hon ca -> BAT BUOC them PSO/GA")
        return "worse"
    else:
        print("\n  -> Ket qua hon hop -> Nen them optimizer")
        return "mixed"


# ===========================================================================
# PHẦN 6: VISUALIZATION — 2 BIỂU ĐỒ TRAINING HISTORY
# ===========================================================================

_COLOR_MAP = {
    "ANFIS-Grid" : "#E63946",
    "ANFIS-FCM"  : "#F4A261",
    "LSTM"       : "#2A9D8F",
    "ANN"        : "#457B9D",
    "ANFIS-PSO"  : "#6A0572",
}

def _get_style(name: str):
    for key in _COLOR_MAP:
        if name.startswith(key):
            return _COLOR_MAP[key], "-"
    return "#333333", "-"


def plot_training_loss(histories: dict, symbol: str, plot_dir: str):
    """Chỉ vẽ 1 biểu đồ: Train Loss + Val Loss của 4 mô hình"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"[{symbol}] Training & Validation Loss", fontsize=14, fontweight="bold")

    for ax, loss_key, title in [
        (axes[0], "train_loss", "Training Loss"),
        (axes[1], "val_loss",   "Validation Loss"),
    ]:
        for model_name, hist in histories.items():
            if model_name not in ["ANFIS-Grid", "ANFIS-FCM", "LSTM", "ANN", "ANFIS-PSO"]:
                continue
            values = hist.get(loss_key, [])
            if not values:
                continue
            color, ls = _get_style(model_name)
            ax.plot(range(1, len(values)+1), values,
                    label=model_name, color=color, linestyle=ls, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{symbol}_train_val_loss.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Train/Val Loss → {path}")


# ===========================================================================
# PHẦN 7: MAIN PIPELINE
# ===========================================================================

def run_phase1(symbol: str = "BTCUSDT",
               feature_set: str = "minimal",
               quick_test: bool = False):
    """
    Chạy toàn bộ Phase 1 cho một symbol.

    Luồng:
      1. Load data
      2. ANFIS-Grid  (skip nếu rules > max_grid_rules, tự giảm n_mf)
      3. ANFIS-FCM   (luôn chạy)
      4. LSTM
      5. ANN
      6. So sánh → Decision Flow
      7. PSO Optimization
      8. Biểu đồ 1 (baseline) + Biểu đồ 2 (optimized)
      9. Lưu summary CSV
    """
    cfg = CONFIG.copy()
    if quick_test:
        feature_set     = "minimal"
        cfg["epochs"]   = 20
        # patience khong can override vi early stopping da bi tat

    save_dir = os.path.join(cfg["save_dir"], symbol, feature_set)
    os.makedirs(save_dir, exist_ok=True)

    # ── BƯỚC 1: Load data ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PHASE 1: {symbol} | Feature set: {feature_set}")
    print(f"{'='*65}")

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler_X, scaler_y, feature_cols) = load_prepared_data(
        symbol, feature_set, cfg["data_dir"]
    )

    n_features = X_train.shape[1]
    results    = {}
    histories  = {}    # history dict mỗi model → dùng để visualize

    # ── BƯỚC 2: ANFIS — Grid (có ràng buộc) ─────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  [2.1] ANFIS — Grid Partitioning")
    print(f"{'─'*65}")

    n_mf_cfg = cfg["anfis"]["grid"]["n_mf"]
    feasible, n_rules_grid, grid_msg = check_grid_feasibility(
        n_features, n_mf_cfg, cfg["max_grid_rules"]
    )
    print(grid_msg)

    anfis_grid     = None
    anfis_grid_key = None

    if feasible:
        # Grid với n_mf gốc
        anfis_grid = ANFIS(n_inputs=n_features, n_mf=n_mf_cfg,
                           clustering="grid", X_train=X_train)
        h = train_anfis(
            anfis_grid, X_train, y_train, X_val, y_val,
            epochs=cfg["epochs"], lr=cfg["anfis"]["grid"]["lr"],
            batch_size=cfg["batch_size"], patience=cfg["patience"],
            save_dir=save_dir, model_name=f"anfis_grid_{symbol}",
            scaler_y=scaler_y
        )
        load_checkpoint(
            os.path.join(save_dir, f"anfis_grid_{symbol}_best.pt"), anfis_grid
        )
        results["ANFIS-Grid"]   = evaluate_model(anfis_grid, X_test, y_test,
                                                  "anfis", scaler_y)
        histories["ANFIS-Grid"] = h
        anfis_grid_key          = "ANFIS-Grid"

    else:
        # Tự động thử giảm n_mf về mức an toàn
        safe_mf = suggest_safe_n_mf(n_features, cfg["max_grid_rules"])
        if safe_mf >= 2:
            print(f"  [AUTO] Thu giam n_mf={safe_mf} "
                  f"({safe_mf}^{n_features}={safe_mf**n_features:,} rules <= "
                  f"{cfg['max_grid_rules']:,})")
            label = f"ANFIS-Grid(mf={safe_mf})"
            anfis_grid = ANFIS(n_inputs=n_features, n_mf=safe_mf,
                               clustering="grid", X_train=X_train)
            h = train_anfis(
                anfis_grid, X_train, y_train, X_val, y_val,
                epochs=cfg["epochs"], lr=cfg["anfis"]["grid"]["lr"],
                batch_size=cfg["batch_size"], patience=cfg["patience"],
                save_dir=save_dir,
                model_name=f"anfis_grid_{symbol}_mf{safe_mf}",
                scaler_y=scaler_y
            )
            load_checkpoint(
                os.path.join(save_dir,
                             f"anfis_grid_{symbol}_mf{safe_mf}_best.pt"),
                anfis_grid
            )
            results[label]   = evaluate_model(anfis_grid, X_test, y_test,
                                               "anfis", scaler_y)
            histories[label] = h
            anfis_grid_key   = label
        else:
            # Ngay cả n_mf=2 cũng vượt ngưỡng với số features này
            # Ví dụ: 80 features → 2^80 ≈ 1.2×10^24 rules
            print(f"  [SKIP] Khong the dung Grid Partitioning voi {n_features} inputs "
                  f"(ngay ca n_mf=2 cung tao {2**n_features:,} rules).\n"
                  f"  -> Chi dung FCM clustering.")

    # ── BƯỚC 3: ANFIS — FCM (luôn chạy) ─────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  [2.2] ANFIS — FCM Clustering")
    print(f"{'─'*65}")

    anfis_fcm = ANFIS(
        n_inputs=n_features,
        n_mf=cfg["anfis"]["fcm"]["n_clusters"],
        clustering="fcm",
        n_clusters=cfg["anfis"]["fcm"]["n_clusters"],
        X_train=X_train
    )
    h_fcm = train_anfis(
        anfis_fcm, X_train, y_train, X_val, y_val,
        epochs=cfg["epochs"], lr=cfg["anfis"]["fcm"]["lr"],
        batch_size=cfg["batch_size"], patience=cfg["patience"],
        save_dir=save_dir, model_name=f"anfis_fcm_{symbol}",
        scaler_y=scaler_y
    )
    load_checkpoint(
        os.path.join(save_dir, f"anfis_fcm_{symbol}_best.pt"), anfis_fcm
    )
    results["ANFIS-FCM"]   = evaluate_model(anfis_fcm, X_test, y_test,
                                             "anfis", scaler_y)
    histories["ANFIS-FCM"] = h_fcm

    # ── BƯỚC 4: LSTM ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  [3] LSTM")
    print(f"{'─'*65}")

    lstm = LSTMModel(
        n_features=n_features,
        hidden_size=cfg["lstm"]["hidden_size"],
        n_layers=cfg["lstm"]["n_layers"],
        dropout=cfg["lstm"]["dropout"]
    )
    lstm, h_lstm = train_baseline_model(
        lstm, X_train, y_train, X_val, y_val,
        model_name=f"lstm_{symbol}",
        epochs=cfg["epochs"], lr=cfg["lstm"]["lr"],
        batch_size=cfg["batch_size"], patience=cfg["patience"],
        save_dir=save_dir, scaler_y=scaler_y,
        seq_len=cfg["lstm"]["seq_len"]
    )
    results["LSTM"]   = evaluate_model(lstm, X_test, y_test,
                                        "lstm", scaler_y,
                                        seq_len=cfg["lstm"]["seq_len"])
    histories["LSTM"] = h_lstm

    # ── BƯỚC 5: ANN ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  [4] ANN")
    print(f"{'─'*65}")

    ann = ANNModel(
        n_features=n_features,
        hidden_sizes=cfg["ann"]["hidden_sizes"],
        dropout=cfg["ann"]["dropout"]
    )
    ann, h_ann = train_baseline_model(
        ann, X_train, y_train, X_val, y_val,
        model_name=f"ann_{symbol}",
        epochs=cfg["epochs"], lr=cfg["ann"]["lr"],
        batch_size=cfg["batch_size"], patience=cfg["patience"],
        save_dir=save_dir, scaler_y=scaler_y
    )
    results["ANN"]   = evaluate_model(ann, X_test, y_test, "ann", scaler_y)
    histories["ANN"] = h_ann

    # ── BƯỚC 6: Bảng so sánh + Biểu đồ 1 (Baseline) ─────────────────────────
    compare_models(results)

    print(f"\n[PLOT] Vẽ biểu đồ Train / Val Loss của 4 mô hình ...")
    plot_training_loss(histories, symbol, save_dir)

    # ── BƯỚC 7: Decision Flow ─────────────────────────────────────────────────
    # Tìm ANFIS tốt nhất trong các kết quả hiện tại
    anfis_keys  = [k for k in results if k.startswith("ANFIS")]
    best_anfis_key = min(anfis_keys, key=lambda k: results[k]["RMSE"])
    print(f"\n  -> ANFIS tot nhat trong baseline: {best_anfis_key}")

    decision = decision_flow(
        results[best_anfis_key],
        results["LSTM"],
        results["ANN"]
    )

    # ── BƯỚC 8: PSO Optimization (luôn chạy theo sơ đồ quyết định) ──────────
    if decision in ["worse", "mixed", "better"]:
        print(f"\n{'─'*65}")
        print(f"  [5] ANFIS + PSO Optimization")
        print(f"{'─'*65}")

        # Khởi tạo ANFIS mới và load best FCM checkpoint làm warm start
        anfis_pso = ANFIS(
            n_inputs=n_features,
            n_mf=cfg["anfis"]["fcm"]["n_clusters"],
            clustering="fcm",
            n_clusters=cfg["anfis"]["fcm"]["n_clusters"],
            X_train=X_train
        )
        load_checkpoint(
            os.path.join(save_dir, f"anfis_fcm_{symbol}_best.pt"), anfis_pso
        )

        anfis_pso, pso_ckpt = optimize_anfis_pso(
            anfis_pso, X_train, y_train, X_val, y_val,
            n_particles=20, iters=30,
            save_dir=save_dir,
            model_name=f"anfis_pso_{symbol}",
            scaler_y=scaler_y
        )
        results["ANFIS-PSO"] = evaluate_model(
            anfis_pso, X_test, y_test, "anfis", scaler_y
        )

        compare_models(results)

        # ── Biểu đồ 2 (Optimized): Best ANFIS before PSO + LSTM + ANN ────
        # PSO không có epoch-by-epoch history → hiển thị cùng ANFIS/LSTM/ANN
        # để thấy context, PSO được chú thích riêng dưới dạng axhline.
        # FIX: Dùng best_anfis_key động thay vì hardcode "ANFIS-FCM"
        # → Đúng khi Grid bị skip và best ANFIS là FCM, hoặc ngược lại
        before_pso_label = f"{best_anfis_key} (before PSO)"
        histories_opt = {
            before_pso_label: copy.deepcopy(histories[best_anfis_key]),
            "LSTM"          : copy.deepcopy(histories["LSTM"]),
            "ANN"           : copy.deepcopy(histories["ANN"]),
        }

        # print(f"\n[PLOT] Bieu do 2 — Optimized ANFIS training history ...")
        # plot_training_histories(histories_opt, symbol, save_dir,
        #                         phase="optimized")
        # plot_val_rmse_comparison(histories_opt, symbol, save_dir,
        #                          phase="optimized")

        # ── Ghi chú PSO result lên cả 2 biểu đồ 2 dưới dạng hline ─────────
        # _annotate_pso_on_plots(
        #     pso_ckpt, symbol, save_dir,
        #     results["ANFIS-PSO"],
        #     histories_opt
        # )

    # ── BƯỚC 9: Lưu summary ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(os.path.join("../Summary", "results_summary.csv"))
    print(f"\n[SAVED] Summary -> {save_dir}/results_summary.csv")

    return results, histories


# def _annotate_pso_on_plots(pso_ckpt: dict, symbol: str,
#                             save_dir: str, pso_test_metrics: dict,
#                             histories_ref: dict) -> None:
#     """
#     Tạo biểu đồ Val RMSE chuyên biệt cho phase "optimized":
#       - Đường epoch-by-epoch cho FCM / LSTM / ANN (từ histories_ref)
#       - Đường nằm ngang (axhline) cho ANFIS-PSO best val RMSE
#       - Annotation test RMSE của PSO ở góc

#     Lý do không dùng imread + imshow:
#       imread chỉ tạo ảnh pixel, không thể vẽ thêm line/text có toạ độ
#       theo đơn vị dữ liệu (data coordinates).
#     """
#     pso_val_rmse  = pso_ckpt.get("val_rmse",  None)
#     pso_test_rmse = pso_test_metrics.get("RMSE", None)
#     pso_color     = _COLOR_MAP.get("ANFIS-PSO", "#6A0572")

#     # ── Biểu đồ Val RMSE with PSO horizontal line ────────────────────────────
#     fig, ax = plt.subplots(figsize=(13, 5))
#     ax.set_title(
#         f"[{symbol}] Val RMSE (USD) — ANFIS-FCM / LSTM / ANN  +  ANFIS-PSO (best)",
#         fontsize=13, fontweight="bold"
#     )

#     for model_name, hist in histories_ref.items():
#         rmse_vals = hist.get("val_rmse", [])
#         if not rmse_vals:
#             continue
#         color, ls = _get_style(model_name)
#         ax.plot(range(1, len(rmse_vals) + 1), rmse_vals,
#                 label=model_name, color=color,
#                 linestyle=ls, linewidth=2.0, alpha=0.85)
#         best_ep = int(np.argmin(rmse_vals))
#         ax.scatter(best_ep + 1, rmse_vals[best_ep],
#                    color=color, marker="*", s=160, zorder=6)
#         ax.annotate(f"  {rmse_vals[best_ep]:.1f}\n  (e{best_ep+1})",
#                     xy=(best_ep + 1, rmse_vals[best_ep]),
#                     fontsize=8, color=color, va="center",
#                     bbox=dict(boxstyle="round,pad=0.15",
#                               fc="white", alpha=0.7, ec=color))

#     # PSO: horizontal dashed line (không có epoch loop nên không có line curve)
#     if pso_val_rmse is not None:
#         max_ep = max(
#             (len(h.get("val_rmse", [])) for h in histories_ref.values()),
#             default=10
#         )
#         ax.axhline(y=pso_val_rmse,
#                    color=pso_color, linestyle="--",
#                    linewidth=2.4, alpha=0.92,
#                    label=f"ANFIS-PSO  val RMSE = {pso_val_rmse:.1f}")
#         ax.annotate(
#             f"  PSO best\n  val {pso_val_rmse:.1f} USD\n"
#             f"  test {pso_test_rmse:.1f} USD",
#             xy=(max_ep * 0.65, pso_val_rmse),
#             fontsize=9, color=pso_color,
#             bbox=dict(boxstyle="round,pad=0.25",
#                       fc="white", alpha=0.88, ec=pso_color)
#         )

#     ax.set_xlabel("Epoch", fontsize=11)
#     ax.set_ylabel("Val RMSE (USD)", fontsize=11)
#     ax.legend(fontsize=10, loc="upper right", framealpha=0.85)
#     ax.grid(True, alpha=0.3, linestyle="--")
#     ax.set_xlim(left=1)
#     _clip_yaxis(ax, histories_ref, "val_rmse")

#     path = os.path.join(save_dir, f"{symbol}_optimized_val_rmse_with_pso.png")
#     plt.tight_layout()
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"[PLOT] Val RMSE with PSO line -> {path}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    # Quick test: verify pipeline không crash (minimal features, 20 epochs)
    # results_q, _ = run_phase1("BTCUSDT", quick_test=True)

    print("\n" + "★" * 65)
    print("  BTCUSDT — Training")
    print("★" * 65)
    results_btc, hist_btc = run_phase1(
        "BTCUSDT",
        feature_set=CONFIG["feature_set"],
        quick_test=False
    )

    # print("\n" + "★" * 65)
    # print("  ETHUSDT — Training")
    # print("★" * 65)
    # results_eth, hist_eth = run_phase1(
    #     "ETHUSDT",
    #     feature_set=CONFIG["feature_set"],
    #     quick_test=False
    # )