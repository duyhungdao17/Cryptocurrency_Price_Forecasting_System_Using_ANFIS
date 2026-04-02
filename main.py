"""
=============================================================================
MAIN PIPELINE - Crypto Forecasting Project
=============================================================================
Hàm main chạy toàn bộ quy trình:
  1. Crawl dữ liệu từ Binance API
  2. Làm sạch và feature engineering
  3. Train ANFIS (Grid + FCM)
  4. Train LSTM và ANN baseline
  5. Đánh giá và so sánh kết quả

Chạy: python main.py
=============================================================================
"""

import sys
import os
import argparse

# Fix sys.path để import từ Crawling và Model
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Định nghĩa các đường dẫn tuyệt đối
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "Dataset")
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "Checkpoints")
DEFAULT_PLOT_DIR = os.path.join(ROOT_DIR, "Plot")
DEFAULT_SUMMARY_DIR = os.path.join(ROOT_DIR, "Summary")

import numpy as np
import pandas as pd
import torch
import warnings
import pickle

warnings.filterwarnings("ignore")

# Import từ Crawling package
from Crawling import (
    fetch_binance_klines,
    clean_raw_data,
    add_technical_indicators,
    create_feature_sets,
    prepare_dataset,
)

# Import từ Model package
from Model.CONFIG import CONFIG
from Model.ANFIS_Model import ANFIS, train_anfis, compute_metrics, load_checkpoint
from Model.Base_Model import LSTMModel, ANNModel, train_baseline_model, create_sequences
from Model.Training import (
    check_grid_feasibility,
    suggest_safe_n_mf,
    evaluate_model,
    compare_models,
    decision_flow,
    plot_training_loss,
    optimize_anfis_pso,
)


# ===========================================================================
# PHẦN 1: SETUP và UTILITIES
# ===========================================================================

def setup_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR, DEFAULT_PLOT_DIR, DEFAULT_SUMMARY_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Directory ready: {d}")


def print_header(title: str):
    """In tiêu đề đẹp"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ===========================================================================
# PHẦN 2: CRAWL DATA
# ===========================================================================

def crawl_data_pipeline(symbol: str, interval: str = "1d",
                        start_date: str = "2020-01-01",
                        end_date: str = None) -> pd.DataFrame:
    """
    Crawl dữ liệu từ Binance API → Làm sạch → Thêm technical indicators
    """
    print_header(f"CRAWLING DATA: {symbol}")

    # Crawl
    print(f"[1/3] Fetching data from Binance ({start_date} → {end_date})...")
    df_raw = fetch_binance_klines(symbol, interval, start_date, end_date)

    # Clean
    print(f"[2/3] Cleaning raw data...")
    df_clean = clean_raw_data(df_raw, symbol)

    # Feature engineering
    print(f"[3/3] Adding technical indicators...")
    df_features = add_technical_indicators(df_clean)

    print(f"\n✓ Total rows: {len(df_features):,}")
    print(f"✓ Date range: {df_features.index.min().date()} → {df_features.index.max().date()}")

    return df_features


def save_data(df: pd.DataFrame, symbol: str, data_dir: str):
    """Lưu dữ liệu vào CSV"""
    path = os.path.join(data_dir, f"{symbol}_features.csv")
    df.to_csv(path)
    print(f"✓ Saved: {path}")


# ===========================================================================
# PHẦN 3: TRAINING PIPELINE
# ===========================================================================

def train_models_pipeline(symbol: str = "BTCUSDT",
                          feature_set: str = "minimal",
                          data_dir: str = None,
                          save_dir: str = None,
                          plot_dir: str = None):
    """
    Load data → Train ANFIS (Grid+FCM) → Train LSTM + ANN → Evaluate
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if save_dir is None:
        save_dir = DEFAULT_SAVE_DIR
    if plot_dir is None:
        plot_dir = DEFAULT_PLOT_DIR

    print_header(f"TRAINING PIPELINE: {symbol} | Feature Set: {feature_set}")

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 1: Load data
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading prepared data...")
    feat_path = os.path.join(data_dir, f"{symbol}_features.csv")

    if not os.path.exists(feat_path):
        print(f"✗ File not found: {feat_path}")
        print("  Run crawl_data first!")
        return None

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    print(f"  Shape: {df.shape}")

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 2: Prepare dataset
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[2/7] Preparing dataset (split + scale)...")
    feature_sets = create_feature_sets(df)
    feature_cols = feature_sets[feature_set]

    result = prepare_dataset(
        df, feature_cols=feature_cols,
        target_col="target",
        train_ratio=0.70, val_ratio=0.10, scale=True
    )

    X_train, X_val, X_test = result[0], result[1], result[2]
    y_train, y_val, y_test = result[3], result[4], result[5]
    scaler_X, scaler_y = result[6], result[7]

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    n_inputs = X_train.shape[1]
    print(f"  Features: {n_inputs}")

    histories = {}
    test_metrics = {}

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 3: ANFIS - Grid Partitioning (nếu khả thi)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[3/7] ANFIS - Grid Partitioning...")

    n_mf_grid = CONFIG["anfis"]["grid"]["n_mf"]
    feasible, n_rules, msg = check_grid_feasibility(n_inputs, n_mf_grid)

    if feasible:
        print(f"  {msg}")
        anfis_grid = ANFIS(
            n_inputs=n_inputs,
            n_mf=n_mf_grid,
            clustering="grid",
            X_train=X_train
        )

        hist_grid = train_anfis(
            anfis_grid, X_train, y_train, X_val, y_val,
            epochs=CONFIG["epochs"],
            lr=CONFIG["anfis"]["grid"]["lr"],
            batch_size=CONFIG["batch_size"],
            save_dir=save_dir,
            model_name=f"{symbol}_ANFIS_Grid",
            scaler_y=scaler_y
        )

        histories["ANFIS-Grid"] = hist_grid

        # Evaluate
        metrics_grid = evaluate_model(
            anfis_grid, X_test, y_test, "anfis", scaler_y
        )
        test_metrics["ANFIS-Grid"] = metrics_grid
        print(f"  Test RMSE: {metrics_grid['RMSE']:.4f} USD")

    else:
        print(f"  {msg} → Skipping Grid (use FCM instead)")
        n_mf_safe = suggest_safe_n_mf(n_inputs)
        if n_mf_safe > 0:
            print(f"  Suggested n_mf: {n_mf_safe}")

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 4: ANFIS - Fuzzy C-Means (FCM)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[4/7] ANFIS - Fuzzy C-Means...")

    n_clusters = CONFIG["anfis"]["fcm"]["n_clusters"]
    anfis_fcm = ANFIS(
        n_inputs=n_inputs,
        n_mf=n_clusters,
        clustering="fcm",
        n_clusters=n_clusters,
        X_train=X_train
    )

    hist_fcm = train_anfis(
        anfis_fcm, X_train, y_train, X_val, y_val,
        epochs=CONFIG["epochs"],
        lr=CONFIG["anfis"]["fcm"]["lr"],
        batch_size=CONFIG["batch_size"],
        save_dir=save_dir,
        model_name=f"{symbol}_ANFIS_FCM",
        scaler_y=scaler_y
    )

    histories["ANFIS-FCM"] = hist_fcm

    # Evaluate
    metrics_fcm = evaluate_model(
        anfis_fcm, X_test, y_test, "anfis", scaler_y
    )
    test_metrics["ANFIS-FCM"] = metrics_fcm
    print(f"  Test RMSE: {metrics_fcm['RMSE']:.4f} USD")

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 5: LSTM Baseline
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[5/7] LSTM Baseline Model...")

    lstm_model = LSTMModel(
        n_features=n_inputs,
        hidden_size=CONFIG["lstm"]["hidden_size"],
        n_layers=CONFIG["lstm"]["n_layers"],
        dropout=CONFIG["lstm"]["dropout"]
    )

    lstm_model, hist_lstm = train_baseline_model(
        lstm_model, X_train, y_train, X_val, y_val,
        model_name=f"{symbol}_LSTM",
        epochs=CONFIG["epochs"],
        lr=CONFIG["lstm"]["lr"],
        batch_size=CONFIG["batch_size"],
        save_dir=save_dir,
        scaler_y=scaler_y,
        seq_len=CONFIG["lstm"]["seq_len"]
    )

    histories["LSTM"] = hist_lstm

    # Evaluate
    metrics_lstm = evaluate_model(
        lstm_model, X_test, y_test, "lstm", scaler_y,
        seq_len=CONFIG["lstm"]["seq_len"]
    )
    test_metrics["LSTM"] = metrics_lstm
    print(f"  Test RMSE: {metrics_lstm['RMSE']:.4f} USD")

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 6: ANN Baseline
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[6/7] ANN Baseline Model...")

    ann_model = ANNModel(
        n_features=n_inputs,
        hidden_sizes=CONFIG["ann"]["hidden_sizes"],
        dropout=CONFIG["ann"]["dropout"]
    )

    ann_model, hist_ann = train_baseline_model(
        ann_model, X_train, y_train, X_val, y_val,
        model_name=f"{symbol}_ANN",
        epochs=CONFIG["epochs"],
        lr=CONFIG["ann"]["lr"],
        batch_size=CONFIG["batch_size"],
        save_dir=save_dir,
        scaler_y=scaler_y
    )

    histories["ANN"] = hist_ann

    # Evaluate
    metrics_ann = evaluate_model(
        ann_model, X_test, y_test, "ann", scaler_y
    )
    test_metrics["ANN"] = metrics_ann
    print(f"  Test RMSE: {metrics_ann['RMSE']:.4f} USD")

    # ─────────────────────────────────────────────────────────────────────
    # BƯỚC 7: So sánh và Visualization
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[7/7] Comparing and visualizing results...")

    # Bảng so sánh
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 70)

    comparison_df = compare_models(test_metrics)
    compare_models(test_metrics).to_string()

    # Biểu đồ training loss
    plot_training_loss(histories, symbol, plot_dir)

    # ─────────────────────────────────────────────────────────────────────
    # Save feature sets để tái sử dụng
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[SAVE] Lưu feature sets...")
    feature_sets_pkl = {
        "feature_sets": create_feature_sets(df),
        "feature_cols": feature_cols,
        "n_inputs": n_inputs
    }
    
    for fs_name, fs_cols in feature_sets_pkl["feature_sets"].items():
        pkl_path = os.path.join(data_dir, f"{symbol}_{fs_name}_features.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"feature_cols": fs_cols, "symbol": symbol}, f)

    # ─────────────────────────────────────────────────────────────────────
    # Save summary CSV
    # ─────────────────────────────────────────────────────────────────────
    summary_df = comparison_df.copy()
    summary_df.insert(0, "Symbol", symbol)
    summary_df.insert(1, "Feature_Set", feature_set)
    
    os.makedirs(DEFAULT_SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(DEFAULT_SUMMARY_DIR, f"{symbol}_{feature_set}_results.csv")
    summary_df.to_csv(summary_path, index=True)

    # Decision
    decision = decision_flow(
        test_metrics.get("ANFIS-FCM", {}),
        test_metrics.get("LSTM", {}),
        test_metrics.get("ANN", {})
    )

    print(f"\n RECOMMENDATION: {decision}")

    # ─────────────────────────────────────────────────────────────────────
    # PSO OPTIMIZATION - Luôn optimize ANFIS-FCM (dù tốt hay kém)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  [PSO OPTIMIZATION] Optimizing ANFIS-FCM parameters...")
    print(f"{'=' * 70}")

    try:
        # Run PSO optimization
        anfis_pso, ckpt_pso = optimize_anfis_pso(
            anfis_fcm,
            X_train, y_train,
            X_val, y_val,
            n_particles=15,
            iters=20,
            save_dir=save_dir,
            model_name=f"{symbol}_ANFIS_PSO",
            scaler_y=scaler_y
        )

        # Evaluate PSO model on test set
        metrics_pso_test = evaluate_model(
            anfis_pso, X_test, y_test, "anfis", scaler_y
        )

        # Add to results
        test_metrics["ANFIS-PSO"] = metrics_pso_test
        
        # Create history for PSO
        hist_pso = {
            "val_rmse": [ckpt_pso.get("val_rmse", 0)],
            "val_mae": [ckpt_pso.get("val_mae", 0)],
            "val_r2": [ckpt_pso.get("val_r2", 0)],
        }
        histories["ANFIS-PSO"] = hist_pso

        print(f"\n✓ PSO Optimization Complete")
        print(f"  Val RMSE:  {ckpt_pso.get('val_rmse', 0):.4f}")
        print(f"  Test RMSE: {metrics_pso_test['RMSE']:.4f} USD")

        # Update comparison
        comparison_df = compare_models(test_metrics)

        print(f"\n{'=' * 70}")
        print("UPDATED MODEL COMPARISON (with PSO)")
        print(f"{'=' * 70}")
        compare_models(test_metrics).to_string()

    except Exception as e:
        print(f"\n SO Optimization failed: {e}")
        print("  Continuing without PSO...")

    return {
        "histories": histories,
        "test_metrics": test_metrics,
        "comparison": comparison_df,
        "models": {
            "anfis_fcm": anfis_fcm,
            "lstm": lstm_model,
            "ann": ann_model
        },
        "scalers": {"scaler_X": scaler_X, "scaler_y": scaler_y}
    }


# ===========================================================================
# PHẦN 4: COMMAND LINE ARGUMENTS
# ===========================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Crypto Forecasting Pipeline - ANFIS/LSTM/ANN Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Default (từ CONFIG):
     python main.py

  2. Chế độ nhanh - minimal features, 20 epochs:
     python main.py --feature-set minimal --epochs 20

  3. Training với symbol cụ thể:
     python main.py --symbol BTCUSDT --feature-set set_features_6 --epochs 40

  4. Multiple symbols:
     python main.py --symbols BTCUSDT ETHUSDT --feature-set minimal --epochs 30

  5. Custom LSTM:
     python main.py --lstm-hidden 128 --lstm-layers 3 --lstm-seq-len 21

  6. Chỉ crawl data (không train):
     python main.py --skip-training

  7. Chỉ train (không crawl lại):
     python main.py --skip-crawl --epochs 50

  8. Full control:
     python main.py --symbol ETHUSDT --feature-set full --epochs 100 \
                    --batch-size 32 --lstm-hidden 64 --lstm-layers 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )
    
    # ─── Symbols ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to train (e.g., BTCUSDT). Ghi đè CONFIG[symbols]"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Multiple symbols to train (e.g., BTCUSDT ETHUSDT). Ghi đè CONFIG[symbols]"
    )
    
    # ─── Feature & Training ──────────────────────────────────────────────
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["minimal", "set_features_6", "set_features_7", "set_features_8", "full"],
        default=None,
        help="Feature set: minimal(4) | set_features_6(6) | set_features_7(7) | set_features_8(8) | full(80+)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số epochs training (default: từ CONFIG, thường 40)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: từ CONFIG, thường 64)"
    )
    
    # ─── LSTM Config ─────────────────────────────────────────────────────
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=None,
        help="LSTM hidden size (default: 64)"
    )
    
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=None,
        help="LSTM layers (default: 2)"
    )
    
    parser.add_argument(
        "--lstm-seq-len",
        type=int,
        default=None,
        help="LSTM sequence length (default: 14)"
    )
    
    # ─── Control Flow ────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Bỏ qua crawl data phase (dùng dữ liệu cũ)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Chỉ crawl, không train model"
    )
    
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Bỏ qua tạo directories"
    )
    
    # ─── Verbose ─────────────────────────────────────────────────────────
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Chi tiết thêm thông tin log"
    )
    
    return parser.parse_args()


def apply_args_to_config(args):
    """Apply command line arguments to CONFIG"""
    
    # ─── Symbols ──────────────────────────────────────────────────────────
    if args.symbol:
        CONFIG["symbols"] = [args.symbol]
        print(f"✓ Symbol: {args.symbol}")
    elif args.symbols:
        CONFIG["symbols"] = args.symbols
        print(f"✓ Symbols: {args.symbols}")
    
    # ─── Feature Set ──────────────────────────────────────────────────────
    if args.feature_set:
        CONFIG["feature_set"] = args.feature_set
        print(f"✓ Feature Set: {args.feature_set}")
    
    # ─── Epochs ───────────────────────────────────────────────────────────
    if args.epochs:
        CONFIG["epochs"] = args.epochs
        print(f"✓ Epochs: {args.epochs}")
    
    # ─── Batch Size ───────────────────────────────────────────────────────
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
        print(f"✓ Batch Size: {args.batch_size}")
    
    # ─── LSTM ──────────────────────────────────────────────────────────────
    if args.lstm_hidden:
        CONFIG["lstm"]["hidden_size"] = args.lstm_hidden
        print(f"✓ LSTM Hidden Size: {args.lstm_hidden}")
    
    if args.lstm_layers:
        CONFIG["lstm"]["n_layers"] = args.lstm_layers
        print(f"✓ LSTM Layers: {args.lstm_layers}")
    
    if args.lstm_seq_len:
        CONFIG["lstm"]["seq_len"] = args.lstm_seq_len
        print(f"✓ LSTM Seq Len: {args.lstm_seq_len}")
    
    print()


# ===========================================================================
# PHẦN 5: MAIN ENTRY POINT
# ===========================================================================

def main(args=None):
    """
    Main entry point - Chạy toàn bộ pipeline
    """
    # ─────────────────────────────────────────────────────────────────────
    # Parse arguments
    # ─────────────────────────────────────────────────────────────────────
    if args is None:
        args = parse_args()
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CRYPTO FORECASTING - FULL PIPELINE" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # ─────────────────────────────────────────────────────────────────────
    # Apply command line arguments to CONFIG
    # ─────────────────────────────────────────────────────────────────────
    print("\n[CONFIG OVERRIDES]")
    apply_args_to_config(args)
    
    # Setup directories
    if not args.skip_setup:
        # setup_directories() 
        pass
    else:
        print("Skipping directory setup")

    # Configuration
    symbols = CONFIG["symbols"]
    feature_set = CONFIG["feature_set"]
    
    print(f"\n[FINAL CONFIG]")
    print(f"  Symbols:    {symbols}")
    print(f"  Feature Set: {feature_set}")
    print(f"  Epochs:     {CONFIG['epochs']}")
    print(f"  Batch Size: {CONFIG['batch_size']}")

    try:
        # ═════════════════════════════════════════════════════════════════
        # PHASE 1: DATA COLLECTION
        # ═════════════════════════════════════════════════════════════════
        if not args.skip_crawl:
            print_header("PHASE 1: DATA COLLECTION & FEATURE ENGINEERING")

            for symbol in symbols:
                # Crawl data
                df_features = crawl_data_pipeline(
                    symbol,
                    interval="1d",
                    start_date="2017-08-31",
                    end_date=None  # Latest date
                )

                # Save
                save_data(df_features, symbol, DEFAULT_DATA_DIR)
        else:
            print_header("PHASE 1: SKIPPED (using existing data)")

        # ═════════════════════════════════════════════════════════════════
        # PHASE 2: MODEL TRAINING & EVALUATION
        # ═════════════════════════════════════════════════════════════════
        if not args.skip_training:
            all_results = {}

            for symbol in symbols:
                print(f"\n>>> Training models for {symbol}...")

                results = train_models_pipeline(
                    symbol=symbol,
                    feature_set=feature_set,
                    data_dir=DEFAULT_DATA_DIR,
                    save_dir=DEFAULT_SAVE_DIR,
                    plot_dir=DEFAULT_PLOT_DIR
                )

                all_results[symbol] = results

            # ═════════════════════════════════════════════════════════════════
            # PHASE 3: SUMMARY
            # ═════════════════════════════════════════════════════════════════
            print_header("FINAL SUMMARY")

            for symbol, results in all_results.items():
                print(f"\n {symbol}:")
                print(results["comparison"].to_string())

            return all_results
        
        else:
            print_header("PHASE 2: SKIPPED (training disabled)")
            print("Use --skip-training=False to enable training")
            return None

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    all_results = main()
