# 📦 Sandbox Telemetry Collection Architecture

This directory defines the isolated execution environment used to safely detonate payloads and extract high-fidelity Windows event telemetry without risking host infrastructure contamination.

### 🖥️ Environment Profile
* **Hypervisor Platform:** VirtualBox
* **Target Operating System:** Windows 10/11 (Isolated Host Build)
* **Telemetry Extraction Tool:** Sysinternals Process Monitor (Procmon)
* **Capture Protocol:** Dedicated runtime windows capturing live process, file system, and registry event dynamics.

### 📊 Dataset Compilation Strategy
The raw event streams were aggregated, filtered for ambient operating system background noise, and condensed into a structured matrix containing 98 samples (40 Benign runs and 58 Malicious ransomware runs) tracking 7 core behavioral features.