"""
improve_performance.py
----------------------
Improves model performance on your existing 98-sample dataset using
four techniques — no additional data required.

UPDATED: Employs an aggressive EDR-style 25% alert threshold during evaluation 
to minimize/eliminate False Negatives (missed malware) at the deployment layer.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.metrics         import (roc_curve, auc, f1_score,
                                     accuracy_score, roc_auc_score,
                                     confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling  import SMOTE
from imblearn.pipeline       import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────
DATASET_PATH    = "feature_dataset.csv"
OUTPUT_DIR      = "./improved_output"
N_FOLDS         = 5
RANDOM_STATE    = 42
ALERT_THRESHOLD = 0.25  # ⚡ DEFENSIVE EDR FIX: Flag as malware if probability >= 25%
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*62)
print("  PERFORMANCE IMPROVEMENT PIPELINE (ZERO-TOLERANCE DEFENSE)")
print("  Ransomware Behavioral Analysis Framework")
print("="*62)


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower()

# Handle both 'label' and 'target_label' column names
label_col = "target_label" if "target_label" in df.columns else "label"

feature_cols = [c for c in df.columns if c != label_col]
X_raw = df[feature_cols].values
y     = df[label_col].values

print(f"\n  Dataset  : {len(df)} samples  |  "
      f"malicious={int(y.sum())}  benign={int((y==0).sum())}")
print(f"  Original features ({len(feature_cols)}): {feature_cols}")


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  — derive 8 new features from existing 7
# ══════════════════════════════════════════════════════════════════════
def engineer_features(X_df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6   # avoid division by zero
    out = X_df.copy()

    total_file_ops = (
        X_df["file_ops_created"] +
        X_df["file_ops_modified"] +
        X_df["file_ops_renamed"] +
        X_df["file_ops_deleted"]
    )

    out["rename_ratio"]        = X_df["file_ops_renamed"]  / (X_df["file_ops_created"]  + eps)
    out["encryption_proxy"]    = (X_df["file_ops_modified"] + X_df["file_ops_renamed"])  / (X_df["file_ops_deleted"] + eps)
    out["registry_per_proc"]  = X_df["registry_ops_written"] / (X_df["process_ops_spawned"] + eps)
    out["extension_churn"]    = X_df["unique_extensions_touched"] / (X_df["file_ops_renamed"] + eps)
    out["file_io_intensity"]  = total_file_ops
    out["del_create_ratio"]    = X_df["file_ops_deleted"]  / (X_df["file_ops_created"]  + eps)
    out["modify_create_ratio"]= X_df["file_ops_modified"] / (X_df["file_ops_created"]  + eps)
    out["proc_file_ratio"]    = X_df["process_ops_spawned"] / (total_file_ops + eps)

    return out

X_df  = pd.DataFrame(X_raw, columns=[c.lower() for c in feature_cols])
X_eng = engineer_features(X_df).values
new_features = [
    "rename_ratio", "encryption_proxy", "registry_per_proc",
    "extension_churn", "file_io_intensity", "del_create_ratio",
    "modify_create_ratio", "proc_file_ratio"
]
all_features = [c.lower() for c in feature_cols] + new_features
print(f"  Engineered features (+8): {new_features}")

# Train/test split (same seed → identical split across all experiments)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_eng, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════
# 3. BASELINE  (original features, no SMOTE, default hyperparams)
# ══════════════════════════════════════════════════════════════════════
baseline_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
    )),
])
baseline_scores = cross_validate(
    baseline_pipe, X_eng[:, :len(feature_cols)], y,
    cv=cv, scoring=["accuracy", "f1", "roc_auc"]
)
baseline_auc = baseline_scores["test_roc_auc"].mean()
print(f"\n  Baseline RF AUC (no improvements) : {baseline_auc:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 4. HYPERPARAMETER TUNING  — Random Forest
# ══════════════════════════════════════════════════════════════════════
print("\n  Tuning Random Forest hyperparameters...")
param_dist = {
    "clf__n_estimators":      [100, 200, 300, 500],
    "clf__max_depth":         [4, 6, 8, 10, None],
    "clf__min_samples_split": [2, 4, 6],
    "clf__min_samples_leaf":  [1, 2, 3],
    "clf__max_features":      ["sqrt", "log2", 0.5, 0.7],
}
rf_tune_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    )),
])
search = RandomizedSearchCV(
    rf_tune_pipe, param_dist,
    n_iter=40, cv=cv, scoring="roc_auc",
    random_state=RANDOM_STATE, n_jobs=-1
)
search.fit(X_eng, y)
best_rf = search.best_estimator_
print(f"  Best RF params : {search.best_params_}")
print(f"  Best RF AUC    : {search.best_score_:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. SMOTE + TUNED RF PIPELINE
# ══════════════════════════════════════════════════════════════════════
print("\n  Building SMOTE + tuned RF pipeline...")
best_params = search.best_params_

smote_rf_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=RANDOM_STATE)),
    ("clf",    RandomForestClassifier(
        n_estimators      = best_params.get("clf__n_estimators", 200),
        max_depth         = best_params.get("clf__max_depth", 8),
        min_samples_split = best_params.get("clf__min_samples_split", 2),
        min_samples_leaf  = best_params.get("clf__min_samples_leaf", 1),
        max_features      = best_params.get("clf__max_features", "sqrt"),
        class_weight      = "balanced",
        random_state      = RANDOM_STATE,
    )),
])
smote_scores = cross_validate(
    smote_rf_pipe, X_eng, y,
    cv=cv, scoring=["accuracy", "f1", "roc_auc"]
)
smote_auc = smote_scores["test_roc_auc"].mean()
print(f"  SMOTE + Tuned RF AUC : {smote_auc:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 6. STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════
print("\n  Building stacking ensemble...")
rf_params = search.best_params_

estimators = [
    ("dt",  Pipeline([
        ("s", StandardScaler()),
        ("c", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
    ])),
    ("rf",  Pipeline([
        ("s", StandardScaler()),
        ("c", RandomForestClassifier(
            n_estimators      = rf_params.get("clf__n_estimators", 200),
            max_depth         = rf_params.get("clf__max_depth", 8),
            min_samples_split = rf_params.get("clf__min_samples_split", 2),
            min_samples_leaf  = rf_params.get("clf__min_samples_leaf", 1),
            max_features      = rf_params.get("clf__max_features", "sqrt"),
            class_weight      = "balanced",
            random_state      = RANDOM_STATE,
        )),
    ])),
    ("lr",  Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        )),
    ])),
    ("svm", Pipeline([
        ("s", StandardScaler()),
        ("c", SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])),
]

stack = StackingClassifier(
    estimators    = estimators,
    final_estimator = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    ),
    cv            = 5,
    passthrough   = False,
    n_jobs        = -1,
)
stack_scores = cross_validate(
    stack, X_eng, y,
    cv=cv, scoring=["accuracy", "f1", "roc_auc"]
)
stack_auc = stack_scores["test_roc_auc"].mean()
print(f"  Stacking Ensemble AUC : {stack_auc:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 7. AVERAGED ROC CURVES  — all improved models
# ══════════════════════════════════════════════════════════════════════
print("\n  Plotting averaged ROC curves...")

pipelines = {
    "Baseline RF (no improvements)":   (baseline_pipe,   X_eng[:, :len(feature_cols)]),
    "Tuned RF + Eng. Features":         (best_rf,         X_eng),
    "SMOTE + Tuned RF":                 (smote_rf_pipe,   X_eng),
    "Stacking Ensemble":                (stack,           X_eng),
}
colors    = ["#9E9E9E", "#FF9800", "#2196F3", "#4CAF50"]
mean_fpr  = np.linspace(0, 1, 200)
fig, ax   = plt.subplots(figsize=(9, 7))

for (label, (pipe, Xp)), color in zip(pipelines.items(), colors):
    tprs, aucs = [], []
    for tr_idx, val_idx in cv.split(Xp, y):
        pipe.fit(Xp[tr_idx], y[tr_idx])
        y_prob = pipe.predict_proba(Xp[val_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[val_idx], y_prob)
        interp = np.interp(mean_fpr, fpr, tpr); interp[0] = 0.
        tprs.append(interp); aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.
    std_tpr  = np.std(tprs,  axis=0)
    mean_auc = np.mean(aucs); std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
            label=f"{label}  (AUC={mean_auc:.3f}±{std_auc:.3f})")
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.10, color=color)

ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.500)")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="Improvement Comparison — Averaged ROC Curves (5-Fold CV)\n"
             "Ransomware Behavioral Analysis Framework")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "improvement_roc_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {roc_path}")


# ══════════════════════════════════════════════════════════════════════
# 8. CONFUSION MATRICES  — best model (stacking) with custom EDR threshold
# ══════════════════════════════════════════════════════════════════════
print(f"\n  Generating confusion matrix with a custom {ALERT_THRESHOLD*100}% alert threshold...")
stack.fit(X_tr, y_tr)

# ⚡ THE FIX: Extract soft raw probabilities instead of default hard 50% cutoff decisions
y_prob_te = stack.predict_proba(X_te)[:, 1]

# ⚡ THE FIX: Enforce zero-tolerance gate logic (Probability >= 0.25 -> Malicious)
y_pred_custom = (y_prob_te >= ALERT_THRESHOLD).astype(int)

cm = confusion_matrix(y_te, y_pred_custom)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Benign","Malicious"]).plot(
    ax=ax, colorbar=False, cmap="Blues"
)
ax.set_title(f"Stacking Ensemble (Threshold={ALERT_THRESHOLD})", fontweight="bold")
cm_path = os.path.join(OUTPUT_DIR, "stacking_confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {cm_path}")


# ══════════════════════════════════════════════════════════════════════
# 9. IMPROVEMENT SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════
report = [
    "=" * 62,
    "  PERFORMANCE IMPROVEMENT SUMMARY",
    "  Ransomware Behavioral Analysis Framework",
    "=" * 62,
    "",
    f"  Dataset : {len(df)} samples  "
    f"(malicious={int(y.sum())}  benign={int((y==0).sum())})",
    f"  Original features : {len(feature_cols)}",
    f"  After engineering : {len(all_features)} (+{len(new_features)} derived)",
    f"  EDR Alert Decision Gate Threshold : {ALERT_THRESHOLD * 100}% (Zero-Tolerance Mode)",
    "",
    "─" * 62,
    f"  {'Technique':<38}  {'AUC':>7}  {'Gain':>6}",
    "─" * 62,
    f"  {'Baseline RF (no improvements)':<38}  {baseline_auc:.3f}     —",
    f"  {'Tuned RF + Engineered Features':<38}  {search.best_score_:.3f}  "
    f"  +{search.best_score_ - baseline_auc:.3f}",
    f"  {'SMOTE + Tuned RF':<38}  {smote_auc:.3f}  "
    f"  +{smote_auc - baseline_auc:.3f}",
    f"  {'Stacking Ensemble (all 4 models)':<38}  {stack_auc:.3f}  "
    f"  +{stack_auc - baseline_auc:.3f}",
    "─" * 62,
    "",
    "  ENGINEERED FEATURES ADDED:",
    "  rename_ratio        = File_Ops_Renamed / File_Ops_Created",
    "  encryption_proxy    = (Modified + Renamed) / Deleted",
    "  registry_per_proc   = Registry_Ops_Written / Process_Ops_Spawned",
    "  extension_churn      = Unique_Extensions / File_Ops_Renamed",
    "  file_io_intensity   = sum of all file operations",
    "  del_create_ratio    = File_Ops_Deleted / File_Ops_Created",
    "  modify_create_ratio = File_Ops_Modified / File_Ops_Created",
    "  proc_file_ratio     = Process_Ops_Spawned / total_file_ops",
    "",
    "  BEST RF HYPERPARAMETERS FOUND:",
    f"  {search.best_params_}",
    "=" * 62,
]
report_text = "\n".join(report)
print("\n" + report_text)

report_path = os.path.join(OUTPUT_DIR, "improvement_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n  Report saved : {report_path}")
print(f"  All outputs  : {OUTPUT_DIR}/\n")

# ══════════════════════════════════════════════════════════════════════
# 10. SERIALIZE CHAMPION MODEL PIPELINE (ADD THIS AT THE END)
# ══════════════════════════════════════════════════════════════════════
import joblib
champion_model_path = "./best_engineered_rf.joblib"
print(f"\n💾 Saving champion Tuned RF + Engineered Features pipeline...")
joblib.dump(best_rf, champion_model_path)
print(f"✅ Saved optimized 15-feature pipeline to: {champion_model_path}")