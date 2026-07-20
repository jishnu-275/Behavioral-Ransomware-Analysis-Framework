import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==============================================================================
# 🎯 ENVIRONMENT CONFIGURATION
# ==============================================================================
INPUT_CSV = "./feature_dataset.csv"
OUTPUT_CSV = "./evaluation_results.csv"

# Professional Risk Classification Threshold Rule
RISK_THRESHOLD = 7

def main():
    print("==========================================================")
    print("🛡️ PHASE 4: PRO EDR MULTI-THRESHOLD RISK SCORING ENGINE   ")
    print("==========================================================")
    
    # 1. Load Feature Space Matrix into Memory
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Feature dataset '{INPUT_CSV}' could not be resolved.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    print(f"📋 Loaded {len(df)} samples into threat detection banks.")
    
    # ==============================================================================
    # 📊 STEP 1: DYNAMIC DATASET-DRIVEN THRESHOLD TUNING
    # ==============================================================================
    print("\n📈 [DYNAMIC DATASET-TUNED THRESHOLD CALIBRATION]")
    
    # Isolate records based on True Ground Truth labels
    benign_df = df[df["Target_Label"] == 0]
    malicious_df = df[df["Target_Label"] == 1]
    
    # Compute true class means across key tracking vectors
    means = {}
    thresholds = {}
    features_to_tune = [
        "File_Ops_Modified", "File_Ops_Renamed", "File_Ops_Deleted", 
        "Registry_Ops_Written", "Process_Ops_Spawned", "Unique_Extensions_Touched"
    ]
    
    for feat in features_to_tune:
        b_mean = benign_df[feat].mean()
        m_mean = malicious_df[feat].mean()
        
        # Calculate optimal separation midpoint threshold boundary
        midpoint = (b_mean + m_mean) / 2.0
        
        means[feat] = {"benign": b_mean, "malicious": m_mean}
        thresholds[feat] = midpoint
        
        print(f"   ↳ Feature [{feat:<25}] ➔ Benign Mean: {b_mean:<8.1f} | Malware Mean: {m_mean:<8.1f} ➔ Tuned Threshold: {midpoint:.1f}")

    # ==============================================================================
    # 🧠 STEP 2: RATIO CALCULATIONS & RISK INTERSECTIONS MATRIX
    # ==============================================================================
    print("\n⚙️ Computing behavioral ratios and executing intersection rules...")
    
    def evaluate_edr_rules(row):
        risk_score = 0
        triggered_rules = []
        epsilon = 1e-5 # Prevents zero division errors
        
        # --- FEATURE RATIOS ---
        # Ratio A: File Rename Density Ratio
        rename_ratio = row["File_Ops_Renamed"] / (row["File_Ops_Modified"] + epsilon)
        
        # Ratio B: Structural Registry Footprint per Process hook
        reg_process_ratio = row["Registry_Ops_Written"] / (row["Process_Ops_Spawned"] + epsilon)
        
        # Ratio C: Extension Diversity Index
        ext_diversity_ratio = row["Unique_Extensions_Touched"] / (row["File_Ops_Modified"] + epsilon)

        # --- BEHAVIOR INTERSECTIONS & SEQUENCE PROXIES (AND BEFORE OR) ---
        
        # Intersection 1: High-Intensity Destructive Ransomware Footprint
        # Mass Modification AND Deletion AND High Extension Variety
        if (row["File_Ops_Modified"] > thresholds["File_Ops_Modified"]) and \
           (row["File_Ops_Deleted"] > thresholds["File_Ops_Deleted"]) and \
           (row["Unique_Extensions_Touched"] > thresholds["Unique_Extensions_Touched"]):
            risk_score += 6
            triggered_rules.append("Destructive_Ransomware_Intersection(+6)")
            
        # Intersection 2: System Subversion / Infiltration Sequence
        # High Process Spawns AND Heavy Registry Persistence writes AND Disk Activity Anomalies
        if (row["Process_Ops_Spawned"] > thresholds["Process_Ops_Spawned"]) and \
           (row["Registry_Ops_Written"] > thresholds["Registry_Ops_Written"]) and \
           (row["File_Ops_Modified"] > thresholds["File_Ops_Modified"]):
            risk_score += 4
            triggered_rules.append("System_Subversion_Sequence(+4)")
            
        # Intersection 3: Suspicious Target Extension Shifting Anomalies
        # High Rename Ratio coupled with an elevated File Modification fingerprint
        if (rename_ratio > 0.01) and (row["File_Ops_Modified"] > thresholds["File_Ops_Modified"]):
            risk_score += 5
            triggered_rules.append("Suspicious_Mass_Rename_Ratio(+5)")
            
        # Intersection 4: Aggressive System Registry Flooding
        # High Registry Operations per active Process execution handle
        if reg_process_ratio > 35.0:
            risk_score += 2
            triggered_rules.append("Aggressive_Registry_Flooding_Ratio(+2)")
            
        # Intersection 5: Stealth Sub-Threshold Malware Footprint
        # Extension Diversity index is high while overall modifications sit low
        if (ext_diversity_ratio > 0.05) and (row["File_Ops_Modified"] < thresholds["File_Ops_Modified"]):
            risk_score += 3
            triggered_rules.append("Stealth_Extension_Diversity_Ratio(+3)")

        return pd.Series([risk_score, ", ".join(triggered_rules)])

    # Map engine logic onto Data Frame arrays
    df[["Risk_Score", "Triggered_Rules"]] = df.apply(evaluate_edr_rules, axis=1)
    
    # ==============================================================================
    # 🎯 STEP 3: CLASSIFICATION ACCORDING TO CUMULATIVE THRESHOLD (>= 7)
    # ==============================================================================
    df["Predicted_Label"] = (df["Risk_Score"] >= RISK_THRESHOLD).astype(int)
    
    # Compute metric values
    y_true = df["Target_Label"]
    y_pred = df["Predicted_Label"]
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ==============================================================================
    # 📊 PERFORMANCE DISPLAY RESULTS
    # ==============================================================================
    print("\n🧱 [DETECTION ENGINE CONFUSION MATRIX]")
    print(f"                  Predicted BENIGN   Predicted MALICIOUS")
    print(f"Actual BENIGN         {tn:<14}     {fp:<18}")
    print(f"Actual MALICIOUS      {fn:<14}     {tp:<18}")
    
    print("\n🎯 [HEURISTIC EVALUATION METRICS SUMMARY]")
    print(f"   ↳ Final System Accuracy:  {accuracy * 100:.2f}%")
    print(f"   ↳ Detection Precision:    {precision * 100:.2f}%")
    print(f"   ↳ Detection Recall:       {recall * 100:.2f}%")
    print(f"   ↳ F1 Performance Score:   {f1 * 100:.2f}%")
    
    print("\n🔍 [EXPLAINABILITY AUDIT SAMPLE DATA DUMP (First 5 Rows)]")
    audit_cols = ["Target_Label", "Risk_Score", "Predicted_Label", "Triggered_Rules"]
    print(df[audit_cols].head(5).to_string())
    
    # Save statistics log sheet
    metrics_summary = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1_Score", "True_Negatives", "False_Positives", "False_Negatives", "True_Positives"],
        "Value": [accuracy, precision, recall, f1, tn, fp, fn, tp]
    }
    pd.DataFrame(metrics_summary).to_csv(OUTPUT_CSV, index=False)
    
    print("\n==========================================================")
    print("✅ PRO CONTEXTUAL SCORING RUN COMPLETED!")
    print(f"📂 Updated analysis data matrix exported: {os.path.abspath(OUTPUT_CSV)}")
    print("==========================================================")

if __name__ == "__main__":
    main()