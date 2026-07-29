import os
import sys
import joblib
import pandas as pd

# ==============================================================================
# 🎯 PRODUCTION INFRASTRUCTURE CONFIGURATION
# ==============================================================================
MODEL_PATH = "./best_engineered_rf.joblib"  # Targets the optimized pipeline
ALERT_THRESHOLD = 0.25

def engineer_single_vector(raw_dict):
    """Dynamically converts a 7-feature raw capture into a 15-feature engineered matrix."""
    eps = 1e-6
    c = raw_dict["file_ops_created"]
    m = raw_dict["file_ops_modified"]
    r = raw_dict["file_ops_renamed"]
    d = raw_dict["file_ops_deleted"]
    reg = raw_dict["registry_ops_written"]
    p = raw_dict["process_ops_spawned"]
    ext = raw_dict["unique_extensions_touched"]
    
    total_file_ops = c + m + r + d
    
    # Matches the exact lowercase math logic from your training script
    return pd.DataFrame([{
        "file_ops_created": c,
        "file_ops_modified": m,
        "file_ops_renamed": r,
        "file_ops_deleted": d,
        "registry_ops_written": reg,
        "process_ops_spawned": p,
        "unique_extensions_touched": ext,
        "rename_ratio": r / (c + eps),
        "encryption_proxy": (m + r) / (d + eps),
        "registry_per_proc": reg / (p + eps),
        "extension_churn": ext / (r + eps),
        "file_io_intensity": total_file_ops,
        "del_create_ratio": d / (c + eps),
        "modify_create_ratio": m / (c + eps),
        "proc_file_ratio": p / (total_file_ops + eps)
    }])

def main():
    print("==========================================================")
    print("🛡️  EDR PRODUCTION INFERENCE LIVE AGENT (V2 ENHANCED)      ")
    print("==========================================================")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Optimized pipeline model file '{MODEL_PATH}' not found.")
        print("   ➔ Please run 'python improve_performance.py' first to export the asset.")
        sys.exit(1)
        
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Optimized 15-Feature Pipeline loaded successfully into memory.")

    # ⚡ TEST VECTOR: Let's pass the absolute idle vector that previously failed
    sample_raw_log = {
        "file_ops_created": 0,
        "file_ops_modified": 0,
        "file_ops_renamed": 0,
        "file_ops_deleted": 0,
        "registry_ops_written": 2,
        "process_ops_spawned": 0,
        "unique_extensions_touched": 0
    }
    
    print("\n🔍 Simulating incoming endpoint telemetry capture event...")
    for feat, val in sample_raw_log.items():
        print(f"   ↳ {feat:<26} : {val}")

    # Process and evaluate using full feature space
    processed_df = engineer_single_vector(sample_raw_log)
    malicious_probability = pipeline.predict_proba(processed_df)[0, 1]
    
    is_malicious = malicious_probability >= ALERT_THRESHOLD
    verdict = "🚨 MALICIOUS (RANSOMWARE FOOTPRINT)" if is_malicious else "🟢 BENIGN (SAFE WORKLOAD)"
    
    print("\n==========================================================")
    print("📊 LIVE DETECTION TARGET VERDICT PANEL                    ")
    print("==========================================================")
    print(f"  🧠 Computed Malicious Score : {malicious_probability * 100:.2f}%")
    print(f"  🛡️ EDR Alert Decision Gate : {ALERT_THRESHOLD * 100:.1f}%")
    print(f"   Verdict Status             : {verdict}")
    print("==========================================================")

if __name__ == "__main__":
    main()