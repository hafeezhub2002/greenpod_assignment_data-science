# Fruit Ripeness & Condition Classification Computer Vision Suite

A production-grade, multi-paradigm computer vision pipeline implementing and benchmarking four distinct machine learning and deep learning approaches for agricultural fruit quality and condition inspection.

Developed for **Greenpod Labs** Technical Computer Vision Evaluation.

---

## ⚠️ Mandatory Dataset Audit & Disclosure

> **CRITICAL DATASET NOTICE**:  
> The assignment specifications describe an **Avocado Ripening Stage Detection** system (`unripe`, `ripe`, `overripe`). However, forensic inspection of the provided dataset archive (`Orange Dataset.zip`) and associated Google Drive repository revealed that the supplied imagery consists of **1,090 citrus/orange fruit images** categorized into four distinct disease and quality classes:
>
> `['blackspot', 'canker', 'fresh', 'grenning']`
>
> In accordance with scientific integrity and the assignment's Critical Dataset Rule:
> 1. We **explicitly disclose** the mismatch rather than falsely claiming orange condition images are avocados.
> 2. The pipeline **dynamically discovers class names** directly from dataset directory structures without hardcoding static labels.
> 3. All four models are trained and rigorously benchmarked against the genuine visual ground truth of the provided dataset.

---

## 🏆 Key Benchmark Results (Held-Out Test Set, $N=99$)

All four approaches were evaluated on the **identical 99 held-out test images** under identical measurement conditions (Intel CPU, Python 3.11, TensorFlow 2.13):

| Approach | Architecture / Methodology | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | End-to-End Latency | Model-Only Latency | Model File Size |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Approach 4** | **Custom Compact CNN** | **97.98%** | **97.92%** | **97.73%** | **97.72%** | 36.63 ms | 5.57 ms | 49.13 MB |
| **Approach 1** | **Classical HSV + Random Forest** | 96.97% | 97.00% | 96.59% | 96.57% | 37.03 ms | 0.62 ms | **0.37 MB** |
| **Approach 2** | **MobileNetV2 Transfer Learning** | 94.95% | 94.81% | 95.08% | 94.81% | 183.27 ms | 31.04 ms | 20.54 MB |
| **Approach 3** | **OpenCV GrabCut + HOG/LBP + SVM** | 87.88% | 86.46% | 86.36% | 86.34% | 55.00 ms | **0.45 ms** | 1.73 MB |

---

## 📁 Repository Structure

```
greenpod_labs/
├── train/                         # Training images (991 images across 4 classes)
├── test/                          # Held-out testing images (99 images across 4 classes)
├── Orange Dataset/                # Source dataset archive folders
├── src/
│   ├── config.py                  # Central configuration, paths, and hardware detection
│   ├── data_audit.py              # Dataset exploration, duplicate checks, leakage audit
│   ├── preprocessing.py           # Stratified splitting, augmentation & dataset generators
│   ├── classical_rf.py            # Approach 1: HSV Histograms + Random Forest
│   ├── mobilenet_transfer.py      # Approach 2: MobileNetV2 Transfer Learning & Fine-Tuning
│   ├── segmentation_svm.py        # Approach 3: GrabCut + HOG/LBP/Color + SVM
│   ├── custom_cnn.py              # Approach 4: Custom 3-Block CNN from scratch
│   ├── evaluate_all.py            # Unified evaluation benchmark script
│   ├── predict.py                 # Single & batch prediction inference pipeline
│   ├── visualization.py           # Publication-quality figure generation
│   └── generate_pdf_report.py     # Script to generate report.pdf
├── models/
│   ├── class_labels.json          # Bidirectional index-to-class label mapping
│   ├── classical_rf.joblib        # Fitted Random Forest model artifact
│   ├── mobilenet_v2.h5            # Best checkpoint for MobileNetV2
│   ├── segmentation_svm.joblib    # Fitted StandardScaler + RBF-SVM pipeline
│   └── custom_cnn.h5              # Best checkpoint for Custom CNN
├── outputs/
│   ├── figures/                   # Confusion matrices, curves, comparisons, grids
│   ├── metrics/                   # Audit reports, evaluation JSON and CSV summaries
│   └── example_predictions/       # Real single-image demonstration predictions
├── tests/
│   └── test_pipeline.py           # 8 automated unit & integration tests (100% pass)
├── app.py                         # Interactive Streamlit dashboard
├── requirements.txt               # Pinned dependencies
├── README.md                      # Project documentation and guide
├── report.md                      # Comprehensive scientific technical report
└── report.pdf                     # Formatted PDF technical report
```

---

## 🛠️ Installation & Environment Setup

### 1. Prerequisites
- Python 3.10 or 3.11
- Windows, Linux, or macOS (CPU and GPU supported automatically)

### 2. Setup Virtual Environment
```bash
# Using Python 3.11 launcher on Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Reproducibility Guide

Execute the pipeline phases sequentially from the project root:

```bash
# Step 1: Run comprehensive dataset exploration and quality audit
py -3.11 src/data_audit.py

# Step 2: Generate stratified train/validation splits and label mappings
py -3.11 src/preprocessing.py

# Step 3: Train Approach 1 (Classical HSV Color Histogram + Random Forest)
py -3.11 src/classical_rf.py

# Step 4: Train Approach 3 (OpenCV GrabCut + HOG/LBP/Color + RBF-SVM)
py -3.11 src/segmentation_svm.py

# Step 5: Train Approach 4 (Custom Compact CNN From Scratch)
py -3.11 src/custom_cnn.py

# Step 6: Train Approach 2 (MobileNetV2 Transfer Learning & Fine-Tuning)
py -3.11 src/mobilenet_transfer.py

# Step 7: Run fair evaluation benchmark on the held-out test set
py -3.11 src/evaluate_all.py

# Step 8: Generate all comparison figures, confusion matrices, and tradeoff plots
py -3.11 src/visualization.py

# Step 9: Demonstrate single-image prediction pipeline
py -3.11 src/predict.py

# Step 10: Run the automated test suite
py -3.11 -m pytest tests/test_pipeline.py -v

# Step 11: Launch the interactive Streamlit dashboard
py -3.11 -m streamlit run app.py
```

---

## 🌐 Interactive Streamlit Dashboard

Launch the application:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Key Features:
1. **Live Inference Tab**:
   - Upload any custom image (`.jpg`, `.jpeg`, `.png`, `.webp`) or select an image from the held-out test partition.
   - Choose any individual model or run **⚡ All 4 Models Simultaneously**.
   - View predicted class badge, confidence percentages, latency readouts, and ground-truth validation comparison.
2. **Model Benchmarks Tab**:
   - Consolidated metrics comparison table.
   - Side-by-side Accuracy and Macro F1 comparison bar charts.
   - Latency vs. Accuracy trade-off scatter plot.
   - Complete 4-model confusion matrix heatmaps.
3. **Dataset Quality Audit Tab**:
   - Class distribution breakdown across Train and Test splits.
   - Segmentation quality validation samples showing GrabCut and fallback masks.
   - Representative class image gallery.
4. **Architecture & Production Tab**:
   - Algorithmic strengths and weaknesses comparison matrix.
   - Industrial conveyor sortation deployment blueprint.

---

## 🧠 Mandatory Reflection & Interview Answers

### Question 1: In production, how will your model perform?
**Answer**:  
- **Throughput**: Approach 4 (Custom CNN) achieves **36.63 ms/image** end-to-end latency on single-core CPU (~27 FPS). If quantized to ONNX INT8 or TensorRT, throughput will reach **>60–80 FPS**, satisfying commercial fruit conveyor speeds. Approach 1 (Random Forest) is even faster at **0.62 ms model-only latency**.
- **Lighting Rig**: Optical performance relies heavily on consistent illumination. An enclosed diffuse 5000K daylight LED dome rig is mandatory in production to eliminate glare, shadows, and camera white-balance drift.
- **Fail-Safe Monitoring**: Predictions with entropy confidence below 65% are automatically routed to a manual inspection buffer.

### Question 2: What are the drawbacks of your model?
**Answer**:  
- **Dataset Domain Shift**: The models are trained on 1,090 images captured under identical studio conditions. Generalization to muddy or dusty field fruit may degrade performance.
- **GrabCut Latency**: The iterative graph-cut segmentation in Approach 3 adds 55 ms latency without exceeding the CNN accuracy.
- **Color Sensitivity**: Approach 1 relies purely on color frequencies and is blind to lesion geometry and spot shapes.
- **CNN Footprint**: The Custom CNN file is 49.1 MB; pruning its dense flattening layer into Global Average Pooling will reduce this to <5 MB.

### Question 3: If given more time, what would you improve?
**Answer**:  
1. **Multispectral & True Avocado Ripening**: Collect multi-angle avocado imagery paired with destructive penetrometer firmness measurements.
2. **Quantization**: Export Custom CNN and MobileNetV2 to TensorRT FP16/INT8 for sub-5ms inference on edge devices (Jetson Orin).
3. **Hard Negative Mining**: Target ambiguous border samples between `canker` and `blackspot` to boost fine-grained lesion discrimination.
4. **Group-Aware Splitting**: Track fruit identity metadata to guarantee zero cross-split photographic leakage.

### Question 4: Where did you spend the most time while solving the assignment?
**Answer**:  
The majority of engineering time was spent on **Dataset Validation & Preprocessing Integrity**:
1. Performing cryptographic and perceptual audits that revealed the orange disease dataset mismatch and building dynamic label inference.
2. Designing a robust multi-stage segmentation algorithm with Otsu and morphological fallbacks so GrabCut never collapses on irregular images.
3. Stabilizing Custom CNN convergence and spatial dropout regularization on a small dataset without BatchNormalization drift.

---

## 📄 Deliverable Documents
- Full Technical Report: [`report.md`](file:///c:/Users/abdul/OneDrive/Desktop/greenpod_labs/report.md)
- Formatted PDF Report: [`report.pdf`](file:///c:/Users/abdul/OneDrive/Desktop/greenpod_labs/report.pdf)
- Evaluation Metrics: [`outputs/metrics/evaluation_comparison.csv`](file:///c:/Users/abdul/OneDrive/Desktop/greenpod_labs/outputs/metrics/evaluation_comparison.csv)
- Data Audit Findings: [`outputs/metrics/data_audit_report.json`](file:///c:/Users/abdul/OneDrive/Desktop/greenpod_labs/outputs/metrics/data_audit_report.json)
