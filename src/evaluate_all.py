"""
Unified Evaluation and Fair Experimental Comparison Module.
Evaluates all four trained computer vision models on the exact same held-out test dataset (99 images)
under identical hardware and measurement conditions.
Generates metrics comparison tables in JSON and CSV formats.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import joblib
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATHS,
    METRICS_DIR,
    IMAGE_SIZE_CLASSICAL,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_MOBILENET,
    get_hardware_info
)
from src.preprocessing import get_stratified_splits, load_image_bgr
from src.classical_rf import extract_hsv_histogram
from src.segmentation_svm import extract_multimodal_features


def evaluate_all_models(splits: dict = None) -> Dict[str, Any]:
    """
    Run identical benchmark on all 4 models using the held-out test set.
    """
    print("=" * 70)
    print("FAIR EXPERIMENTAL EVALUATION ACROSS ALL 4 APPROACHES")
    print("=" * 70)

    if splits is None:
        splits = get_stratified_splits()

    classes = splits["classes"]
    test_paths = splits["test_paths"]
    test_y = np.array(splits["test_labels"])
    n_test = len(test_paths)
    hw_info = get_hardware_info()

    print(f"[EVAL] Test Images: {n_test} | Classes: {classes}")
    print(f"[EVAL] Hardware: {hw_info['default_device']} (GPU available: {hw_info['gpu_available']})")

    results_comparison = {
        "metadata": {
            "test_sample_count": n_test,
            "classes": classes,
            "hardware": hw_info,
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_subject": "Citrus / Orange Condition (Disease & Quality)",
            "note": "Evaluation performed strictly on held-out test partition with zero leakage."
        },
        "models": {}
    }

    # -------------------------------------------------------------
    # 1. Evaluate Approach 1: Random Forest
    # -------------------------------------------------------------
    print("\n--- Evaluating Approach 1: HSV Color Histogram + Random Forest ---")
    if not MODEL_PATHS["random_forest"].exists():
        print("  Model file not found. Skipping.")
    else:
        rf_pkg = joblib.load(MODEL_PATHS["random_forest"])
        rf_model = rf_pkg["model"]
        rf_size_mb = MODEL_PATHS["random_forest"].stat().st_size / (1024 * 1024)

        # Precompute features for model-only latency
        t0 = time.time()
        rf_features = np.array([extract_hsv_histogram(load_image_bgr(p)) for p in test_paths])
        feat_time = time.time() - t0

        # Model-only inference
        t_model_start = time.time()
        rf_preds = rf_model.predict(rf_features)
        rf_probs = rf_model.predict_proba(rf_features)
        rf_model_latency_ms = ((time.time() - t_model_start) / n_test) * 1000

        # End-to-end latency (image read + HSV hist + inference)
        rf_e2e_latencies = []
        for p in test_paths:
            t_s = time.time()
            img = load_image_bgr(p)
            feat = extract_hsv_histogram(img).reshape(1, -1)
            _ = rf_model.predict(feat)
            rf_e2e_latencies.append(time.time() - t_s)
        rf_e2e_latency_ms = float(np.mean(rf_e2e_latencies)) * 1000

        rf_report = classification_report(test_y, rf_preds, target_names=classes, output_dict=True)
        rf_cm = confusion_matrix(test_y, rf_preds, labels=range(len(classes))).tolist()

        results_comparison["models"]["Approach 1: Random Forest"] = {
            "approach_name": "Approach 1: HSV Histogram + Random Forest",
            "model_type": "Classical CV (HSV 512-D) + Random Forest",
            "accuracy": round(float(accuracy_score(test_y, rf_preds)), 4),
            "macro_precision": round(float(precision_score(test_y, rf_preds, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(test_y, rf_preds, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(test_y, rf_preds, average="macro")), 4),
            "model_only_latency_ms": round(rf_model_latency_ms, 3),
            "end_to_end_latency_ms": round(rf_e2e_latency_ms, 3),
            "model_size_mb": round(rf_size_mb, 3),
            "per_class": {
                cls: {
                    "precision": round(rf_report[cls]["precision"], 4),
                    "recall": round(rf_report[cls]["recall"], 4),
                    "f1": round(rf_report[cls]["f1-score"], 4),
                    "support": int(rf_report[cls]["support"])
                }
                for cls in classes
            },
            "confusion_matrix": rf_cm,
            "test_predictions": rf_preds.tolist(),
            "test_probabilities": rf_probs.tolist()
        }
        print(f"  Acc: {results_comparison['models']['Approach 1: Random Forest']['accuracy']:.4f} | Macro F1: {results_comparison['models']['Approach 1: Random Forest']['macro_f1']:.4f} | Latency: {rf_e2e_latency_ms:.2f}ms")

    # -------------------------------------------------------------
    # 2. Evaluate Approach 2: MobileNetV2 Transfer Learning
    # -------------------------------------------------------------
    print("\n--- Evaluating Approach 2: MobileNetV2 Transfer Learning ---")
    if not MODEL_PATHS["mobilenet_v2"].exists():
        print("  Model file not found. Skipping.")
    else:
        mobilenet_model = load_model(MODEL_PATHS["mobilenet_v2"])
        mob_size_mb = MODEL_PATHS["mobilenet_v2"].stat().st_size / (1024 * 1024)

        # Model-only latency
        batch_inputs = np.array([preprocess_input(load_image_bgr(p, target_size=IMAGE_SIZE_MOBILENET)[:, :, ::-1].astype(np.float32)) for p in test_paths])
        # Warmup
        _ = mobilenet_model(batch_inputs[:2], training=False)

        t_model_start = time.time()
        mob_probs = mobilenet_model.predict(batch_inputs, verbose=0)
        mob_preds = np.argmax(mob_probs, axis=1)
        mob_model_latency_ms = ((time.time() - t_model_start) / n_test) * 1000

        # End-to-end latency (image read + preprocess + predict)
        mob_e2e_latencies = []
        for p in test_paths:
            t_s = time.time()
            img = load_image_bgr(p, target_size=IMAGE_SIZE_MOBILENET)
            img_prep = preprocess_input(img[:, :, ::-1].astype(np.float32))[np.newaxis, ...]
            _ = mobilenet_model(img_prep, training=False)
            mob_e2e_latencies.append(time.time() - t_s)
        mob_e2e_latency_ms = float(np.mean(mob_e2e_latencies)) * 1000

        mob_report = classification_report(test_y, mob_preds, target_names=classes, output_dict=True)
        mob_cm = confusion_matrix(test_y, mob_preds, labels=range(len(classes))).tolist()

        results_comparison["models"]["Approach 2: MobileNetV2"] = {
            "approach_name": "Approach 2: MobileNetV2 Transfer Learning",
            "model_type": "Deep Learning: Pretrained MobileNetV2 + Fine-Tuning",
            "accuracy": round(float(accuracy_score(test_y, mob_preds)), 4),
            "macro_precision": round(float(precision_score(test_y, mob_preds, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(test_y, mob_preds, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(test_y, mob_preds, average="macro")), 4),
            "model_only_latency_ms": round(mob_model_latency_ms, 3),
            "end_to_end_latency_ms": round(mob_e2e_latency_ms, 3),
            "model_size_mb": round(mob_size_mb, 3),
            "per_class": {
                cls: {
                    "precision": round(mob_report[cls]["precision"], 4),
                    "recall": round(mob_report[cls]["recall"], 4),
                    "f1": round(mob_report[cls]["f1-score"], 4),
                    "support": int(mob_report[cls]["support"])
                }
                for cls in classes
            },
            "confusion_matrix": mob_cm,
            "test_predictions": mob_preds.tolist(),
            "test_probabilities": mob_probs.tolist()
        }
        print(f"  Acc: {results_comparison['models']['Approach 2: MobileNetV2']['accuracy']:.4f} | Macro F1: {results_comparison['models']['Approach 2: MobileNetV2']['macro_f1']:.4f} | Latency: {mob_e2e_latency_ms:.2f}ms")

    # -------------------------------------------------------------
    # 3. Evaluate Approach 3: Segmentation + SVM
    # -------------------------------------------------------------
    print("\n--- Evaluating Approach 3: Segmentation + HOG/LBP/Color + SVM ---")
    if not MODEL_PATHS["segmentation_svm"].exists():
        print("  Model file not found. Skipping.")
    else:
        svm_pkg = joblib.load(MODEL_PATHS["segmentation_svm"])
        svm_model = svm_pkg["model"]
        scaler = svm_pkg["scaler"]
        svm_size_mb = MODEL_PATHS["segmentation_svm"].stat().st_size / (1024 * 1024)

        raw_features = np.array([extract_multimodal_features(load_image_bgr(p)) for p in test_paths])
        scaled_features = scaler.transform(raw_features)

        t_model_start = time.time()
        svm_preds = svm_model.predict(scaled_features)
        svm_probs = svm_model.predict_proba(scaled_features)
        svm_model_latency_ms = ((time.time() - t_model_start) / n_test) * 1000

        # End-to-end latency (image read + GrabCut + HOG + LBP + scaler + SVM)
        svm_e2e_latencies = []
        for p in test_paths:
            t_s = time.time()
            img = load_image_bgr(p)
            feat = extract_multimodal_features(img).reshape(1, -1)
            feat_sc = scaler.transform(feat)
            _ = svm_model.predict(feat_sc)
            svm_e2e_latencies.append(time.time() - t_s)
        svm_e2e_latency_ms = float(np.mean(svm_e2e_latencies)) * 1000

        svm_report = classification_report(test_y, svm_preds, target_names=classes, output_dict=True)
        svm_cm = confusion_matrix(test_y, svm_preds, labels=range(len(classes))).tolist()

        results_comparison["models"]["Approach 3: Segmentation + SVM"] = {
            "approach_name": "Approach 3: Segmentation + HOG/LBP/Color + SVM",
            "model_type": "Classical CV: GrabCut + HOG/LBP/HSV + RBF-SVM",
            "accuracy": round(float(accuracy_score(test_y, svm_preds)), 4),
            "macro_precision": round(float(precision_score(test_y, svm_preds, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(test_y, svm_preds, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(test_y, svm_preds, average="macro")), 4),
            "model_only_latency_ms": round(svm_model_latency_ms, 3),
            "end_to_end_latency_ms": round(svm_e2e_latency_ms, 3),
            "model_size_mb": round(svm_size_mb, 3),
            "per_class": {
                cls: {
                    "precision": round(svm_report[cls]["precision"], 4),
                    "recall": round(svm_report[cls]["recall"], 4),
                    "f1": round(svm_report[cls]["f1-score"], 4),
                    "support": int(svm_report[cls]["support"])
                }
                for cls in classes
            },
            "confusion_matrix": svm_cm,
            "test_predictions": svm_preds.tolist(),
            "test_probabilities": svm_probs.tolist()
        }
        print(f"  Acc: {results_comparison['models']['Approach 3: Segmentation + SVM']['accuracy']:.4f} | Macro F1: {results_comparison['models']['Approach 3: Segmentation + SVM']['macro_f1']:.4f} | Latency: {svm_e2e_latency_ms:.2f}ms")

    # -------------------------------------------------------------
    # 4. Evaluate Approach 4: Custom CNN
    # -------------------------------------------------------------
    print("\n--- Evaluating Approach 4: Custom CNN From Scratch ---")
    if not MODEL_PATHS["custom_cnn"].exists():
        print("  Model file not found. Skipping.")
    else:
        cnn_model = load_model(MODEL_PATHS["custom_cnn"])
        cnn_size_mb = MODEL_PATHS["custom_cnn"].stat().st_size / (1024 * 1024)

        cnn_batch_inputs = np.array([(load_image_bgr(p, target_size=IMAGE_SIZE_CNN)[:, :, ::-1].astype(np.float32) / 255.0) for p in test_paths])
        _ = cnn_model(cnn_batch_inputs[:2], training=False)

        t_model_start = time.time()
        cnn_probs = cnn_model.predict(cnn_batch_inputs, verbose=0)
        cnn_preds = np.argmax(cnn_probs, axis=1)
        cnn_model_latency_ms = ((time.time() - t_model_start) / n_test) * 1000

        cnn_e2e_latencies = []
        for p in test_paths:
            t_s = time.time()
            img = load_image_bgr(p, target_size=IMAGE_SIZE_CNN)
            img_norm = (img[:, :, ::-1].astype(np.float32) / 255.0)[np.newaxis, ...]
            _ = cnn_model(img_norm, training=False)
            cnn_e2e_latencies.append(time.time() - t_s)
        cnn_e2e_latency_ms = float(np.mean(cnn_e2e_latencies)) * 1000

        cnn_report = classification_report(test_y, cnn_preds, target_names=classes, output_dict=True)
        cnn_cm = confusion_matrix(test_y, cnn_preds, labels=range(len(classes))).tolist()

        results_comparison["models"]["Approach 4: Custom CNN"] = {
            "approach_name": "Approach 4: Custom CNN From Scratch",
            "model_type": "Deep Learning: Compact Keras CNN (Conv2D+Dropout+Flatten)",
            "accuracy": round(float(accuracy_score(test_y, cnn_preds)), 4),
            "macro_precision": round(float(precision_score(test_y, cnn_preds, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(test_y, cnn_preds, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(test_y, cnn_preds, average="macro")), 4),
            "model_only_latency_ms": round(cnn_model_latency_ms, 3),
            "end_to_end_latency_ms": round(cnn_e2e_latency_ms, 3),
            "model_size_mb": round(cnn_size_mb, 3),
            "per_class": {
                cls: {
                    "precision": round(cnn_report[cls]["precision"], 4),
                    "recall": round(cnn_report[cls]["recall"], 4),
                    "f1": round(cnn_report[cls]["f1-score"], 4),
                    "support": int(cnn_report[cls]["support"])
                }
                for cls in classes
            },
            "confusion_matrix": cnn_cm,
            "test_predictions": cnn_preds.tolist(),
            "test_probabilities": cnn_probs.tolist()
        }
        print(f"  Acc: {results_comparison['models']['Approach 4: Custom CNN']['accuracy']:.4f} | Macro F1: {results_comparison['models']['Approach 4: Custom CNN']['macro_f1']:.4f} | Latency: {cnn_e2e_latency_ms:.2f}ms")

    # -------------------------------------------------------------
    # Save Comparison JSON & CSV
    # -------------------------------------------------------------
    json_path = METRICS_DIR / "evaluation_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results_comparison, f, indent=2)
    print(f"\n[SAVE] Comprehensive evaluation JSON saved to: {json_path}")

    # Build comparison summary table
    summary_rows = []
    for model_key, m_data in results_comparison["models"].items():
        summary_rows.append({
            "Approach": m_data["approach_name"],
            "Model Paradigm": m_data["model_type"],
            "Accuracy": f"{m_data['accuracy'] * 100:.2f}%",
            "Macro Precision": f"{m_data['macro_precision'] * 100:.2f}%",
            "Macro Recall": f"{m_data['macro_recall'] * 100:.2f}%",
            "Macro F1-Score": f"{m_data['macro_f1'] * 100:.2f}%",
            "End-to-End Latency (ms)": f"{m_data['end_to_end_latency_ms']:.2f}",
            "Model-Only Latency (ms)": f"{m_data['model_only_latency_ms']:.2f}",
            "Model Size (MB)": f"{m_data['model_size_mb']:.2f}",
            "Test Image Count": n_test
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = METRICS_DIR / "evaluation_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Summary table exported to: {csv_path}")

    print("\n" + "=" * 70)
    print("CONSOLIDATED COMPARISON TABLE (HELD-OUT TEST SET):")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

    return results_comparison


if __name__ == "__main__":
    evaluate_all_models()
