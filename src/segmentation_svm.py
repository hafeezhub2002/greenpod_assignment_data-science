"""
Approach 3: Foreground Segmentation + Multi-Modal (HSV, HOG, LBP) Features + SVM Classifier.
Implements OpenCV GrabCut foreground isolation with robust Otsu/morphological fallback,
extracts color, gradient shape (HOG), and micro-texture (LBP) descriptors,
fits StandardScaler strictly on training data, tunes SVM with RBF kernel,
and serializes the full pipeline.
"""

import sys
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_SEED,
    IMAGE_SIZE_CLASSICAL,
    MODEL_PATHS,
    FIGURES_DIR
)
from src.preprocessing import get_stratified_splits, load_image_bgr


def segment_fruit_foreground(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Segment the fruit foreground from the background.
    Uses GrabCut with an enclosing rectangle, with robust Otsu/morphological fallback
    when mask is degenerate (<10% or >95% image area).
    Returns (masked_bgr, binary_mask, method_used).
    """
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # Define bounding box with a 5% margin
    margin_x = max(5, int(w * 0.05))
    margin_y = max(5, int(h * 0.05))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    method = "GrabCut"
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        # GC_FGD (1) and GC_PR_FGD (3) are foreground
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    except Exception:
        binary_mask = np.zeros((h, w), np.uint8)

    fg_pixels = np.count_nonzero(binary_mask)
    fg_ratio = fg_pixels / total_pixels

    # Fallback if GrabCut collapsed or selected almost entire image
    if fg_ratio < 0.10 or fg_ratio > 0.95:
        method = "Otsu_Morphology_Fallback"
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Otsu threshold on inverted grayscale (or saturation)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological close and open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

        # Retain largest contour
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback_mask = np.zeros((h, w), np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(fallback_mask, [largest], -1, 255, thickness=-1)

        if np.count_nonzero(fallback_mask) / total_pixels >= 0.10:
            binary_mask = fallback_mask
        else:
            # Elliptical center mask as ultimate failsafe
            method = "Center_Ellipse_Failsafe"
            binary_mask = np.zeros((h, w), np.uint8)
            cv2.ellipse(binary_mask, (w // 2, h // 2), (int(w * 0.4), int(h * 0.4)), 0, 0, 360, 255, -1)

    masked_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=binary_mask)
    return masked_bgr, binary_mask, method


def extract_multimodal_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract concatenated feature vector:
    1. Segmented HSV Color Histogram (64 dims)
    2. HOG Gradient and Shape Descriptors (skimage hog, 128 dims)
    3. LBP Texture Pattern Histogram (uniform LBP, 18 dims)
    Total feature vector = 210 dimensions.
    """
    masked_bgr, binary_mask, _ = segment_fruit_foreground(img_bgr)

    # 1. HSV Histogram from foreground pixels only
    img_hsv = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([img_hsv], [0, 1, 2], binary_mask, [4, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hsv_sum = hist_hsv.sum()
    if hsv_sum > 0:
        hist_hsv = hist_hsv / hsv_sum

    # 2. HOG Features from grayscale foreground
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    # Resize to fixed 64x64 for compact HOG descriptor
    gray_resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog_feats = hog(
        gray_resized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )

    # 3. LBP Texture Features (local_binary_pattern with 16 neighbors and radius 2)
    radius = 2
    n_points = 16
    lbp = local_binary_pattern(gray_resized, n_points, radius, method="uniform")
    # For uniform LBP with P=16, number of bins = P + 2 = 18
    n_bins = n_points + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # Concatenate features into a unified 1D vector
    feature_vector = np.concatenate([hist_hsv, hog_feats, lbp_hist]).astype(np.float32)
    return feature_vector


def extract_features_dataset(paths: List[str], target_size: Tuple[int, int] = IMAGE_SIZE_CLASSICAL) -> np.ndarray:
    """Extract multi-modal features for all images in paths."""
    features = []
    for p in paths:
        img_bgr = load_image_bgr(p, target_size=target_size)
        feat = extract_multimodal_features(img_bgr)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def save_segmentation_validation_plot(paths: List[str], output_path: Path):
    """Generate and save a visual validation plot showing GrabCut and fallback segmentations."""
    sample_paths = paths[:6]
    n_samples = len(sample_paths)

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    for i, p in enumerate(sample_paths):
        img_bgr = load_image_bgr(p, target_size=(128, 128))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masked_bgr, mask, method = segment_fruit_foreground(img_bgr)
        masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original: {Path(p).name}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"Mask ({method})", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(masked_rgb)
        axes[i, 2].set_title("Segmented Foreground", fontsize=9)
        axes[i, 2].axis("off")

    plt.suptitle("Foreground Segmentation Validation (OpenCV GrabCut + Fallback)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SEGMENTATION] Validation figure saved to {output_path}")


def train_and_evaluate_svm(splits: dict = None) -> dict:
    """
    Train and evaluate Approach 3: Segmentation + HOG/LBP/Color Features + SVM.
    Scales features using StandardScaler fit on training data only.
    Tunes C and gamma on validation set, evaluates on test set.
    """
    print("\n" + "=" * 60)
    print("APPROACH 3: FOREGROUND SEGMENTATION + HOG/LBP/COLOR + SVM")
    print("=" * 60)

    if splits is None:
        splits = get_stratified_splits()

    classes = splits["classes"]
    train_paths, train_y = splits["train_paths"], np.array(splits["train_labels"])
    val_paths, val_y = splits["val_paths"], np.array(splits["val_labels"])
    test_paths, test_y = splits["test_paths"], np.array(splits["test_labels"])

    # Save segmentation validation plot
    seg_plot_path = FIGURES_DIR / "segmentation_validation.png"
    save_segmentation_validation_plot(train_paths, seg_plot_path)

    print("[EXTRACT] Extracting multi-modal (HSV + HOG + LBP) features...")
    t0 = time.time()
    train_X = extract_features_dataset(train_paths)
    val_X = extract_features_dataset(val_paths)
    test_X = extract_features_dataset(test_paths)
    extraction_time = time.time() - t0
    print(f"[EXTRACT] Extracted in {extraction_time:.2f}s | Feature dim: {train_X.shape[1]}")

    # Standard Scaling - FIT ONLY ON TRAINING DATA!
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)

    # Validation Hyperparameter Tuning for SVM
    param_grid = [
        {"C": 0.5, "gamma": "scale", "kernel": "rbf"},
        {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        {"C": 5.0, "gamma": "scale", "kernel": "rbf"},
        {"C": 10.0, "gamma": "scale", "kernel": "rbf"},
        {"C": 10.0, "gamma": 0.01, "kernel": "rbf"},
        {"C": 10.0, "gamma": "auto", "kernel": "rbf"},
    ]

    best_val_f1 = -1.0
    best_params = None
    best_svm = None

    print("\n[TUNING] Evaluating candidate SVM hyperparameters on validation split:")
    for params in param_grid:
        svm = SVC(
            C=params["C"],
            gamma=params["gamma"],
            kernel=params["kernel"],
            probability=True,
            random_state=RANDOM_SEED
        )
        svm.fit(train_X_scaled, train_y)
        val_preds = svm.predict(val_X_scaled)
        val_macro_f1 = f1_score(val_y, val_preds, average="macro")
        val_acc = accuracy_score(val_y, val_preds)
        print(f"  Params: {params} -> Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_params = params
            best_svm = svm

    print(f"\n[TUNING] Best Selected Params: {best_params} with Val Macro F1 = {best_val_f1:.4f}")

    # Combine Train+Val for final fit
    full_train_X = np.vstack([train_X, val_X])
    full_train_y = np.concatenate([train_y, val_y])

    final_scaler = StandardScaler()
    full_train_X_scaled = final_scaler.fit_transform(full_train_X)
    final_test_X_scaled = final_scaler.transform(test_X)

    t_train_start = time.time()
    final_svm = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel=best_params["kernel"],
        probability=True,
        random_state=RANDOM_SEED
    )
    final_svm.fit(full_train_X_scaled, full_train_y)
    train_time = time.time() - t_train_start

    # Benchmark Test Inference (Model-only and End-to-End)
    _ = final_svm.predict(final_test_X_scaled[:5])

    t_inf_start = time.time()
    test_preds = final_svm.predict(final_test_X_scaled)
    test_probs = final_svm.predict_proba(final_test_X_scaled)
    model_inference_time = (time.time() - t_inf_start) / len(test_X)

    # End-to-end latency measurement
    e2e_latencies = []
    for p in test_paths:
        t_single = time.time()
        img = load_image_bgr(p)
        f_vec = extract_multimodal_features(img).reshape(1, -1)
        f_scaled = final_scaler.transform(f_vec)
        _ = final_svm.predict(f_scaled)
        e2e_latencies.append(time.time() - t_single)
    avg_e2e_latency = float(np.mean(e2e_latencies))

    # Test Metrics
    acc = float(accuracy_score(test_y, test_preds))
    macro_f1 = float(f1_score(test_y, test_preds, average="macro"))
    cm = confusion_matrix(test_y, test_preds, labels=range(len(classes))).tolist()
    cls_report = classification_report(test_y, test_preds, target_names=classes, output_dict=True)

    # Save Pipeline Artifact
    pipeline_artifact = {
        "scaler": final_scaler,
        "model": final_svm,
        "classes": classes,
        "best_params": best_params,
        "image_size": IMAGE_SIZE_CLASSICAL,
        "feature_dim": full_train_X.shape[1],
        "val_macro_f1": float(best_val_f1)
    }
    joblib.dump(pipeline_artifact, MODEL_PATHS["segmentation_svm"])
    model_size_mb = MODEL_PATHS["segmentation_svm"].stat().st_size / (1024 * 1024)

    results = {
        "approach": "Approach 3: Segmentation + HOG/LBP/Color + SVM",
        "model_type": "OpenCV Segmentation + Multimodal Features + RBF-SVM",
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
    print(f"  Saved to:   {MODEL_PATHS['segmentation_svm']}")

    return results


if __name__ == "__main__":
    train_and_evaluate_svm()
