CONFIG = {
    "data_dir" : "../Dataset",
    "save_dir" : "../Checkpoints",

    # Symbols
    "symbols"  :  ["BTCUSDT"],#, "ETHUSDT"],

    # Feature set: "minimal" | "set_features_6" | "set_features_7" | "set_features_8" | "full"
    "feature_set" : "set_features_6",

    # Training params chung
    "epochs"      : 40,
    # "patience" dinh nghia nhung Early Stopping da bi loai bo.
    # Giu lai tham so nay de khong can sua tat ca cac ham goi.
    "patience"    : 9999,  # Effectively disabled
    "batch_size"  : 64,

    # ── ANFIS Grid ──────────────────────────────────────────────────────────
    # MAX_GRID_RULES: Ngưỡng tối đa số rules cho phép với Grid Partitioning.
    # Nếu n_mf^n_inputs > ngưỡng → bỏ qua Grid, tự động thử n_mf nhỏ hơn.
    # Lý do: 27 inputs × 2 MF = 2^27 ≈ 134 triệu rules → crash OOM/RAM.
    # Với 80+ features thì FCM là lựa chọn bắt buộc.
    "max_grid_rules": 1024,       # Ngưỡng an toàn (2^10)

    "anfis": {
        "grid": {
            "n_mf" : 2,           # 2 MF/input → 2^n_features rules
            "lr"   : 0.001,
        },
        "fcm": {
            "n_clusters": 6,      # Số fuzzy rules (Paper 1 dùng 5–8)
            "lr"        : 0.001,
        },
    },

    # ── LSTM ────────────────────────────────────────────────────────────────
    "lstm": {
        "hidden_size": 64,
        "n_layers"   : 2,
        "dropout"    : 0.5,
        "lr"         : 0.0001,
        "seq_len"    : 14,
    },

    # ── ANN ─────────────────────────────────────────────────────────────────
    "ann": {
        "hidden_sizes": [128, 64, 32],
        "dropout"     : 0.5,
        "lr"          : 0.0001,
    },
}