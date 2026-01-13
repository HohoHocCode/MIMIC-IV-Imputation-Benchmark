# MIMIC-IV Imputation Benchmark 🏥📊

[![GPU Acceleration](https://img.shields.io/badge/Hardware-NVIDIA_GPU-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Performance](https://img.shields.io/badge/Metric-NRMSE_0.1742-blue.svg)](#key-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance benchmark suite evaluating **23 imputation algorithms** on the **MIMIC-IV** dataset. This project focuses on **Knowledge-based Imputation** and leverages **GPU Tensor Cores** for large-scale clinical data reconstruction.

---

## 🌟 Overview

Missing data is a critical challenge in Electronic Health Records (EHR). This repository provides a comprehensive pipeline to:
- Preprocess **MIMIC-IV** clinical data into large-scale (D1) and high-resolution (D2) subsets.
- Benchmark 23 algorithms across 4 categories: **Global**, **Local**, **Hybrid**, and **Knowledge-based**.
- Integrate **Medical Knowledge Metadata** (History, Reliability, Bounds) to improve accuracy.
- Accelerate computations using **CUDA Tensor Cores**, enabling 500k+ record processing in seconds.

---

## 🏗️ Project Structure

```text
.
├── ImputingLibrary/        # Core C++ & CUDA implementations
│   ├── experiment_runner.hpp # Evaluation logic & Data handling
│   └── run_benchmarks.cu     # GPU-accelerated algorithm wrappers
├── scripts/                # Python pipeline orchestration
│   ├── preprocess_v2.py      # Data cleaning & Holdout generation
│   ├── run_dual_benchmark.sh # Main Slurm/local benchmark runner
│   └── generate_knowledge.py # Medical metadata extractor
├── report/                 # Technical documentation
│   ├── report_v3.tex         # LaTeX source for technical report
│   └── results_table.tex     # Auto-generated results table
└── data_lib/               # Internal library dependencies
```

---

## 📊 Key Results

The benchmark highlights the superiority of **Knowledge-based** methods when provided with historical patient data.

| Algorithm | Category | D1 (Large) NRMSE | D2 (High-Res) NRMSE | Time (D1) |
| :--- | :--- | :---: | :---: | :---: |
| **HAlimpute** | **Knowledge** | 0.9829 | **0.1742** | 14.3s |
| **ARLS** | **Local** | **0.9114** | **0.5996** | 1.0s |
| **SVD** | **Global** | 0.9392 | 0.8312 | 3.7s |

> **Highlight**: `HAlimpute` achieved a remarkable **0.1742 NRMSE** on the high-resolution dataset by effectively injecting patient history metadata.

---

## 🚀 Getting Started

### 1. Requirements
- Linux OS
- NVIDIA GPU with Compute Capability 7.0+ (V100, A100, etc.)
- CUDA Toolkit 11.0+
- Python 3.8+ (with pandas, numpy)

### 2. Run Benchmark
Execute the orchestrated script to handle preprocessing, compilation, and execution:
```bash
./scripts/run_dual_benchmark.sh
```

### 3. Generate Report
To update the LaTeX report with fresh results:
```bash
python3 scripts/generate_dual_report.py
cd report && pdflatex report_v3.tex
```

---

## 🧠 Core Algorithms

This suite includes implementations for:
- **Global**: SVD, BPCA
- **Local**: KNN, LLS, RLSP, ARLS, AMVI, CMVE, BGS, GMC
- **Knowledge**: HAlimpute, iMISS, POCSimpute, WeNNI, metaMISS, GOimpute
- **Hybrid**: LinCmb

Algorithms marked with `[Tensor]` utilize **Mixed Precision Tensor Cores** for maximum throughput.

---

## 📖 Citation

If you use this benchmark in your research, please cite:

```bibtex
@techreport{hoang2026mimic,
  author = {Pham Le Huy Hoang},
  title = {Technical Report: Performance Evaluation of Imputation Algorithms on MIMIC-IV},
  year = {2026},
  institution = {HohoHocCode GitHub Repository}
}
```

---
**Author**: [HohoHocCode](https://github.com/HohoHocCode)
