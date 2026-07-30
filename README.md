\# Behavioral Ransomware Analysis Framework



An automated endpoint threat detection pipeline that analyzes runtime process telemetry to catch ransomware behavior in real time using engineered behavioral indicators and machine learning classifiers.



\## 🎯 Key Highlights

\* \*\*Zero False Negatives:\*\* Calibrated high-recall threshold achieving \*\*100% detection\*\* on test ransomware samples.

\* \*\*Feature Engineering:\*\* Converts raw telemetry into custom behavioral interaction ratios (e.g., file modification bursts vs. execution runtime).

\* \*\*Baseline Comparison:\*\* Benchmarks heuristic rule-based detection against Random Forest and Stacked ensemble models.

\* \*\*Isolated Sandbox Architecture:\*\* Designed to capture dynamic process telemetry within a safe analysis environment.



\## 📁 Repository Structure

```text

├── analysis/

│   ├── feature\_dataset.csv          # 98-row core telemetry dataset

│   ├── rule\_based\_detector.py      # Baseline heuristic engine

│   └── evaluation\_results.csv       # Heuristic benchmark metrics

├── evaluation/

│   ├── improve\_performance.py      # 15-feature engineered ML pipeline

│   ├── best\_engineered\_rf.joblib    # Trained model weights

│   ├── improvement\_report.txt       # Performance logs

│   ├── improvement\_roc\_curves.png   # ROC analysis plots

│   └── stacking\_confusion\_matrix.png # Confusion matrix visualization

├── sandbox/

│   └── README.md                    # Environment architecture docs

├── predict\_live.py                  # Real-time inference script

└── requirements.txt                 # Dependency matrix

