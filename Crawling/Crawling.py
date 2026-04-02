import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# PHẦN 1: CRAWL DỮ LIỆU TỪ BINANCE API
# ===========================================================================
 
BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
 
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]
 
def fetch_binance_klines(symbol: str, interval: str,
                          start_str: str, end_str: str = None) -> pd.DataFrame:
    """
    Crawl toàn bộ dữ liệu lịch sử từ Binance API.
    Binance giới hạn 1000 nến/request → tự động phân trang.
 
    Args:
        symbol   : "BTCUSDT" hoặc "ETHUSDT"
        interval : "1d" (daily), "1h" (hourly), ...
        start_str: "2017-08-17"
        end_str  : "2024-12-31" hoặc None (lấy đến hiện tại)
 
    Returns:
        pd.DataFrame với đầy đủ dữ liệu
    """
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts   = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str \
               else int(datetime.now().timestamp() * 1000)
 
    all_data = []
    current_start = start_ts
 
    print(f"[INFO] Bắt đầu crawl {symbol} từ {start_str} ...")
 
    while current_start < end_ts:
        params = {
            "symbol"   : symbol,
            "interval" : interval,
            "startTime": current_start,
            "endTime"  : end_ts,
            "limit"    : 1000           # Giới hạn tối đa của Binance
        }
 
        try:
            response = requests.get(BINANCE_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"[ERROR] Request thất bại: {e}. Thử lại sau 5s...")
            time.sleep(5)
            continue
 
        if not data:
            break
 
        all_data.extend(data)
 
        # Cập nhật timestamp cho batch tiếp theo
        last_close_time = data[-1][6]       # close_time của nến cuối
        current_start   = last_close_time + 1
 
        print(f"  → Đã lấy {len(all_data):,} nến | "
              f"Đến: {pd.Timestamp(data[-1][0], unit='ms').date()}")
 
        # Tránh rate limit của Binance
        time.sleep(0.3)
 
    df = pd.DataFrame(all_data, columns=COLUMNS)
    print(f"[DONE] Tổng cộng: {len(df):,} nến cho {symbol}\n")
    return df
 
 
def clean_raw_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Làm sạch và chuẩn hóa kiểu dữ liệu từ Binance API.
    """
    df = df.copy()
 
    # Chuyển timestamp về datetime
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
 
    # Đặt index
    df.set_index("open_time", inplace=True)
    df.index.name = "date"
 
    # Chuyển kiểu dữ liệu số
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # Xóa các cột không cần thiết
    df.drop(columns=["close_time", "ignore"], inplace=True)
 
    # Thêm cột symbol
    df["symbol"] = symbol
 
    # Loại bỏ duplicate và sắp xếp theo thời gian
    df = df[~df.index.duplicated(keep="first")].sort_index()
 
    print(f"[CLEAN] {symbol}: {len(df):,} rows | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df
 
 
# ===========================================================================
# PHẦN 2: FEATURE ENGINEERING CHO DAILY DATA
# ===========================================================================
 
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm toàn bộ technical indicators và derived features.
    """
    df = df.copy()
 
    # -------------------------------------------------------------------
    # NHÓM 1: PRICE-DERIVED FEATURES
    # -------------------------------------------------------------------
    df["return_1d"]    = df["close"].pct_change(1)
    df["log_return"]   = np.log(df["close"] / df["close"].shift(1))
    df["price_range"]  = df["high"] - df["low"]          # High - Low
    df["body"]         = df["close"] - df["open"]         # Candle body
    df["shadow_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["shadow_lower"] = df[["open", "close"]].min(axis=1) - df["low"]
 
    # -------------------------------------------------------------------
    # NHÓM 2: MOVING AVERAGES (Simple + Exponential)
    # -------------------------------------------------------------------
    for period in [7, 14, 21, 30, 50, 200]:
        df[f"MA{period}"] = df["close"].rolling(period).mean()
 
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["EMA9"]  = df["close"].ewm(span=9,  adjust=False).mean()
 
    # Khoảng cách giữa giá và MA (normalize bằng giá)
    df["close_vs_MA7"]  = (df["close"] - df["MA7"])  / df["MA7"]
    df["close_vs_MA30"] = (df["close"] - df["MA30"]) / df["MA30"]
 
    # -------------------------------------------------------------------
    # NHÓM 3: VOLATILITY INDICATORS
    # -------------------------------------------------------------------
    for period in [7, 14, 21, 30]:
        df[f"std{period}"] = df["close"].rolling(period).std()
 
    # Bollinger Bands (MA20 ± 2σ)
    bb_period = 20
    df["BB_mid"]   = df["close"].rolling(bb_period).mean()
    bb_std         = df["close"].rolling(bb_period).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]   # %B width
    df["BB_pct"]   = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
 
    # ATR (Average True Range)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
 
    # -------------------------------------------------------------------
    # NHÓM 4: MOMENTUM INDICATORS
    # -------------------------------------------------------------------
    # RSI (Relative Strength Index)
    for period in [7, 14, 21]:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-10)   # Tránh chia cho 0
        df[f"RSI{period}"] = 100 - (100 / (1 + rs))
 
    # MACD
    df["MACD"]         = df["EMA12"] - df["EMA26"]
    df["MACD_signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]    = df["MACD"] - df["MACD_signal"]
 
    # ROC (Rate of Change)
    for period in [7, 14, 30]:
        df[f"ROC{period}"] = df["close"].pct_change(period) * 100
 
    # -------------------------------------------------------------------
    # NHÓM 5: VOLUME INDICATORS
    # -------------------------------------------------------------------
    df["volume_MA7"]   = df["volume"].rolling(7).mean()
    df["volume_MA14"]  = df["volume"].rolling(14).mean()
    df["volume_ratio"] = df["volume"] / df["volume_MA14"]       # Volume so với TB 14 ngày
 
    # Buy Pressure từ Binance Taker data
    df["buy_pressure"] = (
        df["taker_buy_base_asset_volume"] / (df["volume"] + 1e-10)
    )
    df["sell_pressure"] = 1 - df["buy_pressure"]
 
    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
 
    # -------------------------------------------------------------------
    # NHÓM 6: LAG FEATURES (Sliding Window — theo Paper 3)
    # -------------------------------------------------------------------
    lag_features = ["close", "volume", "RSI14", "MACD", "buy_pressure"]
    for feature in lag_features:
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
 
    # -------------------------------------------------------------------
    # NHÓM 7: MARKET MICROSTRUCTURE
    # -------------------------------------------------------------------
    df["trade_intensity"] = df["number_of_trades"] / df["volume"]       # Giao dịch/unit volume
    df["avg_trade_size"]  = df["volume"] / (df["number_of_trades"] + 1) # Kích thước GD TB
 
    # -------------------------------------------------------------------
    # OUTPUT TARGET (next day close)
    # -------------------------------------------------------------------
    df["target"]         = df["log_return"].shift(-1)   
    df["target_return"]  = df["return_1d"].shift(-1)  # Return ngày hôm sau (phụ trợ)
 
    return df
 
 
def create_feature_sets(df: pd.DataFrame) -> dict:
    """
    Tạo các bộ features khác nhau cho từng giai đoạn training.
 
    Nguyên tắc thiết kế (dựa trên literature):
    ─────────────────────────────────────────────────────────────────────────
    • ANFIS bị giới hạn bởi "curse of dimensionality":
        - Grid Partitioning: rules = n_mf ^ n_inputs → bùng nổ khi n_inputs > 6
        - FCM: số rules = số clusters C, nhưng nhiều inputs → MF khó hội tụ
      → Research consensus (Jang FAQ, Hindawi 2021, IntechOpen 2018):
        ANFIS hoạt động tốt nhất khi n_inputs ≤ 6.
    • Nguyên tắc "1 feature / 1 chức năng" để tránh multicollinearity:
        1. Price anchor: close (Neo mức giá hiện tại, giúp mô hình nhận biết regime giá)
        2. Trend: MA7 (Đại diện xu hướng ngắn hạn đã được làm mượt, phù hợp với fuzzy rules hơn giá thô)
        3. Momentum: RSI14 (Biểu diễn trạng thái quá mua / quá bán, dễ phân vùng Low–Medium–High cho membership functions)
        4. Participation: volume_ratio (Đo mức độ tham gia của thị trường theo dạng tương đối, ổn định hơn volume thô)
        5. Order-flow: buy_pressure (Phản ánh áp lực mua chủ động từ dữ liệu taker, bổ sung thông tin về hướng dòng lệnh)
        6. Volatility: ATR14 (Đại diện mức biến động thực tế của thị trường theo cách gọn và ít dư thừa)
        7. Short-return proxy: ROC7 (Đại diện chuyển động giá ngắn hạn mà không trùng trực tiếp với target log-return)
        8. Microstructure: trade_intensity (Phản ánh cấu trúc giao dịch trên mỗi đơn vị volume, bổ sung thông tin vi mô của thị trường)
 
    Returns:
        dict với các key:
          - "minimal"        : 4 features — Quick test (Paper 1 config)
          - "set_features_6" : 6 features — ANFIS safe zone (khuyến nghị)
          - "set_features_7" : 7 features — Balanced (thêm volatility)
          - "set_features_8" : 8 features — Aggressive (thêm microstructure)
          - "full"           : Tất cả features (dùng cho LSTM/ANN)
 
    Lưu ý "standard" đã bị loại bỏ vì chứa quá nhiều features tương quan cao
    (open/high/low cùng close, nhiều MA family, nhiều lag) → ANFIS yếu kém.
    """
 
    def validate_features(feat_list, df_ref):
        valid   = [f for f in feat_list if f in df_ref.columns]
        missing = [f for f in feat_list if f not in df_ref.columns]
        if missing:
            print(f"[WARN] Features không tồn tại: {missing}")
        return valid
 
    # -------------------------------------------------------------------
    # BỘ 0: MINIMAL — Quick test, verify pipeline không crash
    # 4 features theo Paper 1 (Mehrban & Ahadian 2024) + Kutlu 2021
    # Grid: 2^4 = 16 rules   FCM: 4–8 rules
    # -------------------------------------------------------------------
    minimal_features = [
        "close",        # Price anchor
        "volume",       # Participation (raw, giữ nguyên theo Paper 1)
        "RSI14",        # Momentum
        "MA7",          # Trend
    ]
 
    # -------------------------------------------------------------------
    # BỘ 1: SET_FEATURES_6 — ANFIS Safe Zone (6 features)
    # Nguyên tắc: 1 feature / 1 chức năng, không features tương quan chéo
    # Grid: 2^6 = 64 rules (còn quản lý được nhưng nên dùng FCM)
    # FCM: 6–8 clusters khuyến nghị
    #
    # Cơ sở lý luận:
    # - Thay "volume" bằng "volume_ratio" (tương đối, ổn định hơn qua các
    #   bull/bear cycle do giá BTC thay đổi hàng chục lần)
    # - Thêm "log_return" thay vì "return_1d" (ít skew hơn, time-additive)
    # - "buy_pressure" từ Taker data của Binance: không có trong các paper
    #   nhưng rất phù hợp ANFIS vì range [0,1] → MF tự nhiên
    # -------------------------------------------------------------------
    set_features_6 = [
        "close",          # [1] Price anchor — neo giá tuyệt đối
        "MA7",            # [2] Trend — xu hướng ngắn hạn (7 ngày)
        "RSI14",          # [3] Momentum — [0,100] rất hợp với MF Gaussian
        "volume_ratio",   # [4] Participation — volume / MA14(volume)
        "buy_pressure",   # [5] Order-flow — taker_buy/volume ∈ [0,1]
        "ATR14",          # [6] Volatility
    ]
 
    # -------------------------------------------------------------------
    # BỘ 2: SET_FEATURES_7 — ANFIS Balanced (7 features)
    # Thêm 1 volatility feature vào bộ safe_6
    # ATR14 được chọn thay vì BB_width vì:
    #   - ATR = True Range trung bình 14 ngày → biến động thực
    #   - BB_width phụ thuộc MA20 (đã có MA7 trong set → dư thừa một phần)
    # FCM với 6–10 clusters, Grid KHÔNG khuyến nghị (2^7 = 128 rules)
    # -------------------------------------------------------------------
    set_features_7 = [
        "close",          # [1] Price anchor
        "MA7",            # [2] Trend
        "RSI14",          # [3] Momentum
        "volume_ratio",   # [4] Participation
        "buy_pressure",   # [5] Order-flow
        "ATR14",          # [6] Volatility — Average True Range 14 ngày
        "ROC7",           # [7] Short-return proxy
    ]
 
    # -------------------------------------------------------------------
    # BỘ 3: SET_FEATURES_8 — ANFIS Aggressive (8 features)
    # Thêm 1 microstructure feature vào bộ balanced_7
    # Chọn "trade_intensity" = number_of_trades / volume:
    #   - Đại diện cho mức độ phân tán giao dịch (nhiều GD nhỏ vs ít GD lớn)
    #   - Tín hiệu về "chất lượng" volume, bổ sung cho buy_pressure
    #   - KHÔNG dùng "price_range" vì ATR14 đã bao phủ biến động trong ngày
    # FCM với 8–12 clusters, Grid TUYỆT ĐỐI KHÔNG dùng (2^8 = 256 rules)
    #
    # CẢNH BÁO: Ở bộ này, PSO/GA optimizer đặc biệt quan trọng để tránh
    # overfitting vì ANFIS có nhiều tham số hơn với 8 inputs.
    # -------------------------------------------------------------------
    set_features_8 = [
        "close",            # [1] Price anchor
        "MA7",              # [2] Trend
        "RSI14",            # [3] Momentum
        "volume_ratio",     # [4] Participation
        "buy_pressure",     # [5] Order-flow
        "ATR14",            # [6] Volatility
        "ROC7",             # [7] Short-return proxy
        "trade_intensity",  # [8] Microstructure — trades/volume
    ]
 
    # -------------------------------------------------------------------
    # BỘ 4: FULL — Tất cả features (dành cho LSTM/ANN, không dùng cho ANFIS)
    # Loại bỏ: target, target_return, symbol, raw Binance fields đã có FE
    # -------------------------------------------------------------------
    _exclude = {
        "target", "target_return", "symbol",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "number_of_trades",
    }
    full_features = [c for c in df.columns if c not in _exclude]
 
    # In tóm tắt
    result = {
        "minimal"       : validate_features(minimal_features,  df),
        "set_features_6": validate_features(set_features_6,    df),
        "set_features_7": validate_features(set_features_7,    df),
        "set_features_8": validate_features(set_features_8,    df),
        "full"          : validate_features(full_features,      df),
    }
 
    print(f"\n[FEATURE SETS SUMMARY]")
    for name, cols in result.items():
        tag = ""
        if name == "minimal":
            tag = "← Quick test"
        elif name == "set_features_6":
            tag = "← ANFIS khuyến nghị"
        elif name == "full":
            tag = "← LSTM/ANN only"
        print(f"  {name:16s}: {len(cols):3d} features  {tag}")
 
    return result
 
 
 
 
 
# ===========================================================================
# PHẦN 2B: DATA-DRIVEN FEATURE SELECTION CHO ANFIS
# ===========================================================================
 
def select_anfis_features_auto(df: pd.DataFrame,
                                n_features: int = 6,
                                target_col: str = "target",
                                method: str = "mutual_info",
                                train_ratio: float = 0.70) -> list:
    """
    Chọn tự động n features tốt nhất cho ANFIS dựa trên dữ liệu thực tế.
 
    Approach: Dùng data-driven methods thay vì chọn tay để tránh thiên kiến.
    Research backing:
    - CMC 2025: MI + RFE + RFI giúp giảm 80-85% features mà không mất performance
    - Kutlu 2021: dùng `exhsrch` trong MATLAB để tìm 4 inputs quan trọng nhất
    - Jang ANFIS FAQ: khuyến nghị chọn inputs dựa trên prediction power
 
    Args:
        df         : DataFrame đã qua add_technical_indicators()
        n_features : Số features muốn chọn (khuyến nghị 4–8 cho ANFIS)
        target_col : Tên cột output
        method     : "mutual_info" | "correlation" | "random_forest"
        train_ratio: Chỉ fit trên training portion (tránh data leakage)
 
    Returns:
        List tên features được chọn, theo thứ tự quan trọng giảm dần
 
    Ví dụ sử dụng:
        >>> feat_df = add_technical_indicators(raw_df)
        >>> auto_features = select_anfis_features_auto(feat_df, n_features=6)
        >>> print(auto_features)
        ['close', 'RSI14', 'MA7', 'ATR14', 'buy_pressure', 'volume_ratio']
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
 
    # Lấy tất cả candidate features (loại target và raw Binance fields)
    exclude = {
        target_col, "target_return", "symbol",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "number_of_trades",
    }
    candidate_cols = [c for c in df.columns if c not in exclude]
 
    # Làm sạch: drop NaN, chỉ dùng training portion để tránh leakage
    data    = df[candidate_cols + [target_col]].dropna()
    n_train = int(len(data) * train_ratio)
    train   = data.iloc[:n_train]
 
    X_train = train[candidate_cols].values.astype(float)
    y_train = train[target_col].values.astype(float)
 
    print(f"\n[AUTO SELECT] Method: {method} | Candidates: {len(candidate_cols)} "
          f"| Target: top {n_features}")
 
    if method == "mutual_info":
        # Mutual Information: đo lường phụ thuộc phi tuyến giữa feature và target
        # Không giả định phân phối → phù hợp với crypto data phi tuyến
        scores = mutual_info_regression(X_train, y_train,
                                         discrete_features=False,
                                         random_state=42)
        score_dict = dict(zip(candidate_cols, scores))
 
    elif method == "correlation":
        # Pearson correlation: nhanh, dễ hiểu nhưng chỉ bắt linear
        score_dict = {
            col: abs(train[col].corr(train[target_col]))
            for col in candidate_cols
        }
 
    elif method == "random_forest":
        # Random Forest feature importance: bắt được phi tuyến, tương tác features
        # Nặng hơn về compute nhưng thường cho kết quả tốt nhất
        rf = RandomForestRegressor(n_estimators=100, max_depth=5,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        score_dict = dict(zip(candidate_cols, rf.feature_importances_))
 
    else:
        raise ValueError(f"method phải là 'mutual_info', 'correlation', "
                         f"hoặc 'random_forest'. Nhận: {method}")
 
    # Sắp xếp và lấy top n
    sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    selected        = [f for f, _ in sorted_features[:n_features]]
 
    # In kết quả
    print(f"  {'Feature':<25s}  Score")
    print(f"  {'─'*40}")
    for feat, score in sorted_features[:n_features]:
        marker = " ← selected"
        print(f"  {feat:<25s}  {score:.6f}{marker}")
    if len(sorted_features) > n_features:
        print(f"  ... (còn {len(sorted_features) - n_features} features không được chọn)")
 
    return selected
 
 
 
def prepare_dataset(df: pd.DataFrame,
                    feature_cols: list,
                    target_col: str = "target",
                    train_ratio: float = 0.70,
                    val_ratio:   float = 0.10,
                    scale: bool = True):
    """
    Chuẩn bị dataset: dropna → split → scale.
 
    QUAN TRỌNG: Dùng temporal split (KHÔNG random split)
    để tránh data leakage từ tương lai.
 
    Args:
        df           : DataFrame đã có FE
        feature_cols : Danh sách feature columns
        target_col   : Tên cột output
        train_ratio  : 0.70 (70% training)
        val_ratio    : 0.10 (10% validation)
        scale        : True → MinMaxScaler
 
    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test, scaler, split_dates)
    """
    # Chọn columns và drop NaN
    data = df[feature_cols + [target_col]].dropna().copy()
 
    n = len(data)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
 
    # Temporal split
    train_df = data.iloc[:train_end]
    val_df   = data.iloc[train_end:val_end]
    test_df  = data.iloc[val_end:]
 
    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values
 
    y_train = train_df[target_col].values
    y_val   = val_df[target_col].values
    y_test  = test_df[target_col].values
 
    scaler_X = None
    scaler_y = None
 
    if scale:
        # Fit scaler CHỈ trên training data
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
 
        X_train = scaler_X.fit_transform(X_train)
        X_val   = scaler_X.transform(X_val)     # KHÔNG fit lại
        X_test  = scaler_X.transform(X_test)    # KHÔNG fit lại
 
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
 
    split_dates = {
        "train": (data.index[0].date(), data.index[train_end - 1].date()),
        "val"  : (data.index[train_end].date(), data.index[val_end - 1].date()),
        "test" : (data.index[val_end].date(), data.index[-1].date()),
    }
 
    print(f"\n[SPLIT] Temporal Split:")
    for split, (s, e) in split_dates.items():
        size = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}[split]
        print(f"  {split:5s}: {s} → {e} | {size:,} samples")
    print(f"  Features: {len(feature_cols)}")
 
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_X, scaler_y, split_dates)
 
 
# ===========================================================================
# PHẦN 3: MAIN — CHẠY TOÀN BỘ PIPELINE
# ===========================================================================
 
def main():
    SAVE_DIR = "../Dataset"
    os.makedirs(SAVE_DIR, exist_ok=True)
 
    SYMBOLS     = ["BTCUSDT", "ETHUSDT"]
    INTERVAL    = "1d"
    START_DATE  = "2017-08-17"    # Ngày đầu tiên Binance có dữ liệu
    END_DATE    = None #"2024-12-31"    # Hoặc None để lấy đến hiện tại
 
    raw_dfs = {}
    feature_dfs = {}
    datasets = {}
 
    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  XỬ LÝ: {symbol}")
        print(f"{'='*60}")
 
        # ---------------------------------------------------------------
        # BƯỚC 1: Crawl dữ liệu
        # ---------------------------------------------------------------
        raw_path = os.path.join(SAVE_DIR, f"{symbol}_raw.csv")
 
        if os.path.exists(raw_path):
            print(f"[CACHE] Load từ cache: {raw_path}")
            raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        else:
            raw  = fetch_binance_klines(symbol, INTERVAL, START_DATE, END_DATE)
            raw_df = clean_raw_data(raw, symbol)
            raw_df.to_csv(raw_path)
            print(f"[SAVE] Raw data → {raw_path}")
 
        raw_dfs[symbol] = raw_df
 
        # ---------------------------------------------------------------
        # BƯỚC 2: Feature Engineering
        # ---------------------------------------------------------------
        feat_path = os.path.join(SAVE_DIR, f"{symbol}_features.csv")
 
        feat_df = add_technical_indicators(raw_df)
        feat_df.to_csv(feat_path)
        print(f"[SAVE] Features ({len(feat_df.columns)} cols) → {feat_path}")
        feature_dfs[symbol] = feat_df
 
        # ---------------------------------------------------------------
        # BƯỚC 3: Tạo các bộ feature sets
        # ---------------------------------------------------------------
        feature_sets = create_feature_sets(feat_df)
        # feature_sets summary already printed by create_feature_sets()
 
        # ---------------------------------------------------------------
        # BƯỚC 4: Chuẩn bị datasets cho từng bộ feature
        # ---------------------------------------------------------------
        datasets[symbol] = {}
        for set_name, feat_cols in feature_sets.items():
            result = prepare_dataset(
                feat_df,
                feature_cols=feat_cols,
                target_col="target",
                train_ratio=0.70,
                val_ratio=0.10,
                scale=True
            )
            datasets[symbol][set_name] = result
 
            # Lưu scaler để dùng khi inference
            import joblib
            scaler_X, scaler_y = result[6], result[7]
            joblib.dump(scaler_X,
                os.path.join(SAVE_DIR, f"{symbol}_{set_name}_scaler_X.pkl"))
            joblib.dump(scaler_y,
                os.path.join(SAVE_DIR, f"{symbol}_{set_name}_scaler_y.pkl"))
 
    print(f"\n{'='*60}")
    print("  HOÀN THÀNH DATA PIPELINE")
    print(f"{'='*60}")
    print(f"[INFO] Dữ liệu đã lưu tại: {SAVE_DIR}")
 
    return raw_dfs, feature_dfs, datasets
 
 
if __name__ == "__main__":
    raw_dfs, feature_dfs, datasets = main()