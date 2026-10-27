"""
Centralized Configuration and Path Management for the Fruit Ripeness / Condition Pipeline.
"""

import os
from pathlib import Path
import tensorflow as tf

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths: check if 'train' exists directly or inside 'Orange Dataset/train'
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "train"
DEFAULT_TEST_DIR = PROJECT_ROOT / "test"

if not DEFAULT_TRAIN_DIR.exists():
    alt_train = PROJECT_ROOT / "Orange Dataset" / "train"
    if alt_train.exists():
        DEFAULT_TRAIN_DIR = alt_train

if not DEFAULT_TEST_DIR.exists():
    alt_test = PROJECT_ROOT / "Orange Dataset" / "test"
    if alt_test.exists():
        DEFAULT_TEST_DIR = alt_test

# Reproducibility
RANDOM_SEED = 42

# Image Resolutions
IMAGE_SIZE_CLASSICAL = (128, 128)
IMAGE_SIZE_CNN = (128, 128)
IMAGE_SIZE_MOBILENET = (224, 224)

# Output Paths
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EXAMPLES_DIR = OUTPUTS_DIR / "example_predictions"

for d in [MODELS_DIR, FIGURES_DIR, METRICS_DIR, EXAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hardware Detection
def get_hardware_info():
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "gpu_available": len(gpus) > 0,
        "gpu_devices": [gpu.name for gpu in gpus],
        "default_device": "/GPU:0" if len(gpus) > 0 else "/CPU:0"
    }

# Model filenames
MODEL_PATHS = {
    "random_forest": MODELS_DIR / "classical_rf.joblib",
    "segmentation_svm": MODELS_DIR / "segmentation_svm.joblib",
    "custom_cnn": MODELS_DIR / "custom_cnn.h5",
    "mobilenet_v2": MODELS_DIR / "mobilenet_v2.h5",
    "class_labels": MODELS_DIR / "class_labels.json",
    "audit_report": METRICS_DIR / "data_audit_report.json",
    "evaluation_metrics": METRICS_DIR / "evaluation_comparison.json",
    "evaluation_csv": METRICS_DIR / "evaluation_comparison.csv",
}
