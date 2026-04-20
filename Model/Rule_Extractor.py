import torch
import numpy as np
import os
import sys
import itertools

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.ANFIS_Model import ANFIS

def extract_rules(model_path, feature_names, output_file=None):
    if not os.path.exists(model_path):
        print(f"ERROR: Checkpoint not found at {model_path}")
        return

    # Lưu stdout gốc để khôi phục sau này
    original_stdout = sys.stdout
    f = None
    if output_file:
        f = open(output_file, 'w', encoding='utf-8-sig')
        sys.stdout = f

    try:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        n_inputs = len(feature_names)
        clustering = 'fcm' if 'fcm' in model_path.lower() or 'pso' in model_path.lower() else 'grid'
        n_mf = 6 if clustering == 'fcm' else 2

        model = ANFIS(n_inputs=n_inputs, n_mf=n_mf, clustering=clustering)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Note: Loading state dict: {e}")
        
        model.eval()

        print("="*60)
        print("[ANFIS FUZZY RULE EXTRACTION REPORT (BTCUSDT)]")
        print("="*60)

        # 1. Layer 1: Fuzzification (fuzz_layer)
        print("\n[STEP 1] INTERPRETING MEMBERSHIP FUNCTIONS")
        # fuzz_layer là một module GaussianMF, có thuộc tính centers và sigmas
        centers = model.fuzz_layer.centers.detach().numpy()
        sigmas = model.fuzz_layer.sigmas.detach().numpy()

        labels = ["Low/Negative", "Medium", "High/Positive"]
        feature_mfs = {}
        
        for i, name in enumerate(feature_names):
            print(f"\nFeature: {name:20}")
            feature_mfs[i] = []
            mf_indices = np.argsort(centers[i])
            for idx, m_idx in enumerate(mf_indices):
                label = labels[idx] if idx < len(labels) else f"Level {idx}"
                c = centers[i][m_idx]
                s = sigmas[i][m_idx]
                feature_mfs[i].append({"label": label, "center": c, "sigma": s, "original_idx": m_idx})
                print(f"  - {label:15}: Center={c:8.4f}, Sigma={s:8.4f}")

        # 2. Layer 4: Rules (consequent_params)
        print("\n" + "="*60)
        print("[STEP 2] EXTRACTING IF-THEN RULES (Top 10 Influential)")
        # consequent_params là một nn.Parameter
        consequent = model.consequent_params.detach().numpy()
        n_rules = consequent.shape[0]

        if clustering == 'grid':
            mf_combinations = list(itertools.product(range(n_mf), repeat=n_inputs))
        else:
            # FCM: 1 input ứng với 1 cluster rule cụ thể
            mf_combinations = [(i,)*n_inputs for i in range(n_rules)]

        for r_idx in range(min(n_rules, 10)):
            rule_parts = []
            for f_idx in range(n_inputs):
                mf_idx = mf_combinations[r_idx][f_idx] if r_idx < len(mf_combinations) else 0
                label = "Unknown"
                for item in feature_mfs[f_idx]:
                    if item['original_idx'] == mf_idx:
                        label = item['label']
                        break
                rule_parts.append(f"({feature_names[f_idx]} is {label})")
            
            condition = " AND ".join(rule_parts)
            # Kết luận Sugeno: y = p*x + r
            # bias là tham số cuối cùng
            bias = consequent[r_idx, -1]
            trend = "UP [UP]" if bias > 0 else "DOWN [DOWN]"
            
            print(f"\nRULE {r_idx+1}:")
            print(f"  IF {condition}")
            print(f"  THEN Price Trend is {trend} (Impact Score: {bias:.6f})")

        print("\n" + "="*60)
        print("[SUCCESS] Extraction Complete.")

    finally:
        if f:
            sys.stdout = original_stdout
            f.close()

if __name__ == "__main__":
    FEATURES = ['close_log_return', 'target_dist_7', 'target_dist_30', 'volatility_7', 'volume_zscore', 'rsi_14']
    
    checkpoint_dir = "Checkpoints"
    best_model = None
    if os.path.exists(checkpoint_dir):
        # Ưu tiên load mô hình PSO vì kết quả tốt nhất
        pso_files = [f for f in os.listdir(checkpoint_dir) if 'pso' in f.lower() and f.endswith('.pt')]
        if pso_files:
            best_model = os.path.join(checkpoint_dir, sorted(pso_files)[-1])
        else:
            fcm_files = [f for f in os.listdir(checkpoint_dir) if 'fcm' in f.lower() and f.endswith('.pt')]
            if fcm_files:
                best_model = os.path.join(checkpoint_dir, sorted(fcm_files)[-1])

    output_report = os.path.join("Summary", "Fuzzy_Rules_Report.txt")
    os.makedirs("Summary", exist_ok=True)

    if best_model:
        extract_rules(best_model, FEATURES, output_file=output_report)
        print(f"Report saved to: {output_report}")
    else:
        print("No checkpoint found.")
