"""
Approach 1: Classical Computer Vision using HSV Color Histograms + Random Forest Classifier.
Extracts normalized 3D HSV histograms, performs validation-guided hyperparameter tuning,
evaluates on held-out test data, and serializes the trained model.
"""

import sys
import time
import json
from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_SEED,
    IMAGE_SIZE_CLASSICAL,
    MODEL_PATHS,
    METRICS_DIR
)
from src.preprocessing import get_stratified_splits, load_image_bgr


def extract_hsv_histogram(img_bgr: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    """
    Convert image to HSV and compute normalized 3D color histogram.
    Hue: 0-180, Saturation: 0-256, Value: 0-256.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize histogram to sum to 1.0 (L1 norm)
    hist = hist.flatten()
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist.astype(np.float32)


def extract_features_from_paths(paths: List[str], target_size: Tuple[int, int] = IMAGE_SIZE_CLASSICAL) -> np.ndarray:
    """Batch extract HSV histogram features from image file paths."""
    features = []
    for p in paths:
        img_bgr = load_image_bgr(p, target_size=target_size)
        feat = extract_hsv_histogram(img_bgr)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def train_and_evaluate_rf(splits: dict = None) -> dict:
    """
    Train and evaluate Approach 1: HSV Color Histogram + Random Forest.
    Tunes hyperparameters on validation set, evaluates on held-out test set.
    """
    print("\n" + "=" * 60)
    print("APPROACH 1: HSV COLOR HISTOGRAM + RANDOM FOREST")
    print("=" * 60)

    if splits is None:
        splits = get_stratified_splits()

    classes = splits["classes"]
    train_paths, train_y = splits["train_paths"], np.array(splits["train_labels"])
    val_paths, val_y = splits["val_paths"], np.array(splits["val_labels"])
    test_paths, test_y = splits["test_paths"], np.array(splits["test_labels"])

    print(f"[EXTRACT] Extracting HSV histograms (dim = 8x8x8 = 512)...")
    t0 = time.time()
    train_X = extract_features_from_paths(train_paths)
    val_X = extract_features_from_paths(val_paths)
    test_X = extract_features_from_paths(test_paths)
    extraction_time = time.time() - t0
    print(f"[EXTRACT] Features extracted in {extraction_time:.2f}s | Train shape: {train_X.shape}")

    # Hyperparameter tuning on validation set
    param_grid = [
        {"n_estimators": 50, "max_depth": 10, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 15, "class_weight": "balanced"},
        {"n_estimators": 150, "max_depth": None, "class_weight": "balanced"},
        {"n_estimators": 150, "max_depth": 20, "class_weight": None},
        {"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
    ]

    best_val_f1 = -1.0
    best_params = None
    best_rf = None

    print("\n[TUNING] Evaluating candidate hyperparameters on validation split:")
    for params in param_grid:
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight=params["class_weight"],
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf.fit(train_X, train_y)
        val_preds = rf.predict(val_X)
        val_macro_f1 = f1_score(val_y, val_preds, average="macro")
        val_acc = accuracy_score(val_y, val_preds)
        print(f"  Params: {params} -> Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_params = params
            best_rf = rf

    print(f"\n[TUNING] Best Selected Params: {best_params} with Val Macro F1 = {best_val_f1:.4f}")

    # Retrain best configuration on combined Train+Val or keep fitted model
    # Retraining on Train+Val with selected optimal hyperparameters gives production quality
    full_train_X = np.vstack([train_X, val_X])
    full_train_y = np.concatenate([train_y, val_y])

    t_train_start = time.time()
    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        class_weight=best_params["class_weight"],
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    final_model.fit(full_train_X, full_train_y)
    train_time = time.time() - t_train_start

    # Benchmark Test Inference (Model-only and End-to-End)
    # Warmup
    _ = final_model.predict(test_X[:5])

    t_inf_start = time.time()
    test_preds = final_model.predict(test_X)
    test_probs = final_model.predict_proba(test_X)
    model_inference_time = (time.time() - t_inf_start) / len(test_X)

    # End-to-end latency measurement (image decode + feature extraction + inference)
    e2e_latencies = []
    for p in test_paths:
        t_single = time.time()
        img = load_image_bgr(p)
        f_vec = extract_hsv_histogram(img).reshape(1, -1)
        _ = final_model.predict(f_vec)
        e2e_latencies.append(time.time() - t_single)
    avg_e2e_latency = float(np.mean(e2e_latencies))

    # Metrics
    acc = float(accuracy_score(test_y, test_preds))
    macro_f1 = float(f1_score(test_y, test_preds, average="macro"))
    cm = confusion_matrix(test_y, test_preds, labels=range(len(classes))).tolist()
    cls_report = classification_report(test_y, test_preds, target_names=classes, output_dict=True)

    # Save model and artifact package
    model_artifact = {
        "model": final_model,
        "classes": classes,
        "best_params": best_params,
        "image_size": IMAGE_SIZE_CLASSICAL,
        "bins": (8, 8, 8),
        "feature_dim": 512,
        "val_macro_f1": float(best_val_f1)
    }
    joblib.dump(model_artifact, MODEL_PATHS["random_forest"])
    model_size_mb = MODEL_PATHS["random_forest"].stat().st_size / (1024 * 1024)

    results = {
        "approach": "Approach 1: HSV Color Histogram + Random Forest",
        "model_type": "Classical CV + Random Forest",
        "best_params": best_params,
        "accuracy": round(acc, 4),
        "macro_precision": round(cls_report["macro avg"]["precision"], 4),
        "macro_recall": round(cls_report["macro avg"]["recall"], 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            cls: {
                "precision": round(cls_report[cls]["precision"], 4),
                "recall": round(cls_report[cls]["recall"], 4),
                "f1": round(cls_report[cls]["f1-score"], 4),
                "support": int(cls_report[cls]["support"])
            }
            for cls in classes
        },
        "confusion_matrix": cm,
        "training_time_sec": round(train_time, 4),
        "model_only_latency_ms": round(model_inference_time * 1000, 3),
        "end_to_end_latency_ms": round(avg_e2e_latency * 1000, 3),
        "model_size_mb": round(model_size_mb, 3),
        "test_image_count": len(test_y),
        "test_predictions": test_preds.tolist(),
        "test_probabilities": test_probs.tolist()
    }

    print("\n[RESULTS] Held-out Test Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Latency:    {avg_e2e_latency*1000:.2f} ms/image (end-to-end)")
    print(f"  Saved to:   {MODEL_PATHS['random_forest']}")

    return results


if __name__ == "__main__":
    train_and_evaluate_rf()
