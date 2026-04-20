import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.ANFIS_Model import ANFIS
from Crawling.Crawling import (
    fetch_binance_klines,
    clean_raw_data,
    add_technical_indicators,
    prepare_dataset
)

def run_backtest(model_path, symbol="BTCUSDT", initial_balance=10000.0, fee=0.001):
    print("\n" + "="*60)
    print(f"[STARTING BACKTEST] (PROTECTIVE STRATEGY): {symbol}")
    print("="*60)

    # 1. Chuẩn bị dữ liệu
    df_raw = fetch_binance_klines(symbol=symbol, interval="1d", start_str="2023-01-01")
    df_clean = clean_raw_data(df_raw, symbol=symbol)
    df_features = add_technical_indicators(df_clean)
    
    FEATURES = ["close", "MA7", "RSI14", "volume_ratio", "buy_pressure", "ATR14"]
    TARGET = 'target'
    
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, 
     scaler_X, scaler_y, split_dates) = prepare_dataset(
        df_features, feature_cols=FEATURES, target_col=TARGET
    )
    
    test_start_date = split_dates['test'][0]
    df_test_prices = df_features[df_features.index.date >= test_start_date].copy()
    df_test_prices = df_test_prices.iloc[:len(X_test)]

    # 2. Load Model ANFIS-PSO
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    n_inputs = len(FEATURES)
    model = ANFIS(n_inputs=n_inputs, n_mf=6, clustering='fcm')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    # 3. Dự báo
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled, _ = model(X_tensor)
        y_pred_scaled = y_pred_scaled.numpy().flatten()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 4. Mô phỏng với CHIẾN LƯỢC TỰ VỆ (Conservative)
    balance = initial_balance
    position = 0.0 
    history = []
    prices = df_test_prices['close'].values
    
    # Tăng ngưỡng (Threshold) để lọc nhiễu: Chỉ vào lệnh khi dự báo tăng ít nhất 0.8%
    # Thoát lệnh ngay khi dự báo < 0 (không cần chờ giảm mạnh mới thoát)
    buy_threshold = 0.008 
    exit_threshold = 0.002

    print(f"Backtesting with Buy Threshold: {buy_threshold*100:.2f}%")

    for i in range(len(prices) - 1):
        current_price = prices[i]
        pred_change = y_pred[i] 
        
        if pred_change > buy_threshold and position == 0:
            # MUA
            position = (balance * (1 - fee)) / current_price
            balance = 0
            # print(f"  DAY {i}: BUY  at {current_price:,.2f} (Pred: +{pred_change*100:.2f}%)")
            
        elif pred_change < exit_threshold and position > 0:
            # BÁN VÀ ĐỨNG NGOÀI (CASH OUT)
            balance = position * current_price * (1 - fee)
            position = 0
            # print(f"  DAY {i}: SELL at {current_price:,.2f} (Pred: {pred_change*100:.2f}%)")
            
        equity = balance + (position * current_price)
        history.append(equity)

    if position > 0:
        balance = position * prices[-1] * (1 - fee)
        position = 0
    
    final_equity = balance
    total_roi = ((final_equity - initial_balance) / initial_balance) * 100
    bh_roi = ((prices[-1] - prices[0]) / prices[0]) * 100
    
    print("\n" + "="*60)
    print("[BACKTEST RESULTS SUMMARY]")
    print("="*60)
    print(f"Final Balance:    {final_equity:,.2f} USDT")
    print(f"ANFIS Strategy:   {total_roi:+.2f}%")
    print(f"Buy & Hold:       {bh_roi:+.2f}%")
    print(f"Outperformance:   {total_roi - bh_roi:+.2f}%")
    print("="*60)

    # Trực quan hóa kết quả
    plt.figure(figsize=(12, 6))
    plt.plot(history, label='ANFIS (Protective Strategy)', color='green', linewidth=2)
    bh_history = [initial_balance * (p / prices[0]) for p in prices[:len(history)]]
    plt.plot(bh_history, label='Buy & Hold BTC', color='gray', linestyle='--')
    plt.fill_between(range(len(history)), history, bh_history, where=(np.array(history) > np.array(bh_history)), color='green', alpha=0.1)
    plt.title(f"Optimized ANFIS-PSO Backtest - {symbol}")
    plt.ylabel("USDT")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(f"Plot/Backtest_Optimized_{symbol}.png")
    print(f"[CHART] Optimized chart saved to: Plot/Backtest_Optimized_{symbol}.png")

    # LƯU KẾT QUẢ VÀO CSV
    summary_data = {
        'Symbol': [symbol],
        'Strategy_ROI': [total_roi],
        'Buy_and_Hold_ROI': [bh_roi],
        'Outperformance': [total_roi - bh_roi],
        'Final_Balance': [final_equity],
        'Initial_Balance': [initial_balance]
    }
    df_summary = pd.DataFrame(summary_data)
    os.makedirs("Summary", exist_ok=True)
    summary_path = "Summary/Backtest_Performance.csv"
    
    # Ghi đè hoặc thêm mới (append)
    if not os.path.exists(summary_path):
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    else:
        df_summary.to_csv(summary_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    print(f"[STATS] Performance metrics saved to: {summary_path}")

if __name__ == "__main__":
    checkpoint_dir = "Checkpoints"
    # Tìm file PSO tốt nhất vừa train xong
    pso_files = [f for f in os.listdir(checkpoint_dir) if 'pso' in f.lower() and f.endswith('.pt')]
    if pso_files:
        best_model = os.path.join(checkpoint_dir, sorted(pso_files)[-1])
        run_backtest(best_model, symbol="BTCUSDT")
    else:
        print("No PSO model found.")
