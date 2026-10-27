"""
Reusable Prediction and Single-Image Inference Pipeline.
Supports all four trained computer vision models:
1. 'random_forest'
2. 'mobilenet_v2'
3. 'segmentation_svm'
4. 'custom_cnn'
Performs image validation, model-specific preprocessing, confidence score computation,
and latency benchmarking.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import cv2
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATHS,
    IMAGE_SIZE_CLASSICAL,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_MOBILENET,
    EXAMPLES_DIR
)
from src.preprocessing import load_image_bgr
from src.classical_rf import extract_hsv_histogram
from src.segmentation_svm import extract_multimodal_features


class FruitModelPredictor:
    """Production inference wrapper for all four computer vision models."""

    def __init__(self, model_key: str = "custom_cnn"):
        self.model_key = model_key
        self.classes = self._load_classes()
        self.model_artifact = None
        self.loaded_model = None
        self.scaler = None
        self._load_model()

    def _load_classes(self) -> list:
        if not MODEL_PATHS["class_labels"].exists():
            raise FileNotFoundError(f"Class labels file not found: {MODEL_PATHS['class_labels']}")
        with open(MODEL_PATHS["class_labels"], "r") as f:
            data = json.load(f)
        return data["classes"]

    def _load_model(self):
        model_path = MODEL_PATHS.get(self.model_key)
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model file for '{self.model_key}' not found at: {model_path}")

        print(f"[PREDICTOR] Loading '{self.model_key}' from {model_path}...")
        if self.model_key == "random_forest":
            self.model_artifact = joblib.load(model_path)
            self.loaded_model = self.model_artifact["model"]
        elif self.model_key == "segmentation_svm":
            self.model_artifact = joblib.load(model_path)
            self.loaded_model = self.model_artifact["model"]
            self.scaler = self.model_artifact["scaler"]
        elif self.model_key in ["custom_cnn", "mobilenet_v2"]:
            self.loaded_model = load_model(model_path)
        else:
            raise ValueError(f"Unknown model key: {self.model_key}")

    def validate_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Validate and decode image input into a standard BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image path does not exist: {p}")
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise ValueError(f"Corrupted or unsupported image file: {p}")
            return img_bgr
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0 or len(image_input.shape) < 2:
                raise ValueError("Empty or invalid image array provided.")
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run inference on an image and return prediction, confidence scores, and latency.
        """
        t_start = time.time()
        img_bgr = self.validate_image(image_input)
        h, w = img_bgr.shape[:2]

        t_preprocess_start = time.time()
        probs = None

        if self.model_key == "random_forest":
            img_resized = cv2.resize(img_bgr, IMAGE_SIZE_CLASSICAL, interpolation=cv2.INTER_AREA)
            features = extract_hsv_histogram(img_resized).reshape(1, -1)
            t_model_start = time.time()
            probs = self.loaded_model.predict_proba(features)[0]
            model_latency_ms = (time.time() - t_model_start) * 1000

        elif self.model_key == "segmentation_svm":
            img_resized = cv2.resize(img_bgr, IMAGE_SIZE_CLASSICAL, interpolation=cv2.INTER_AREA)
            features = extract_multimodal_features(img_resized).reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            t_model_start = time.time()
            probs = self.loaded_model.predict_proba(scaled_features)[0]
            model_latency_ms = (time.time() - t_model_start) * 1000

        elif self.model_key == "custom_cnn":
            img_resized = cv2.resize(img_bgr, IMAGE_SIZE_CNN, interpolation=cv2.INTER_AREA)
            img_norm = (img_resized[:, :, ::-1].astype(np.float32) / 255.0)[np.newaxis, ...]
            t_model_start = time.time()
            pred_scores = self.loaded_model(img_norm, training=False).numpy()
            probs = pred_scores[0]
            model_latency_ms = (time.time() - t_model_start) * 1000

        elif self.model_key == "mobilenet_v2":
            img_resized = cv2.resize(img_bgr, IMAGE_SIZE_MOBILENET, interpolation=cv2.INTER_AREA)
            img_rgb = img_resized[:, :, ::-1].astype(np.float32)
            img_prep = preprocess_input(img_rgb)[np.newaxis, ...]
            t_model_start = time.time()
            pred_scores = self.loaded_model(img_prep, training=False).numpy()
            probs = pred_scores[0]
            model_latency_ms = (time.time() - t_model_start) * 1000

        total_latency_ms = (time.time() - t_start) * 1000

        pred_idx = int(np.argmax(probs))
        predicted_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        class_probabilities = {
            self.classes[i]: round(float(probs[i]), 4)
            for i in range(len(self.classes))
        }

        result = {
            "model_key": self.model_key,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "score_interpretation": (
                "Empirical posterior probability" if "rf" in self.model_key or "svm" in self.model_key
                else "Softmax model score (uncalibrated neural network output)"
            ),
            "class_probabilities": class_probabilities,
            "model_latency_ms": round(model_latency_ms, 3),
            "total_latency_ms": round(total_latency_ms, 3),
            "original_dimensions": [w, h]
        }

        if ground_truth is not None:
            result["ground_truth"] = ground_truth
            result["is_correct"] = (predicted_class.lower() == ground_truth.lower())

        return result


def demonstrate_single_inference():
    """Demonstrate inference on a real held-out test image and export artifacts."""
    from src.preprocessing import get_stratified_splits
    import argparse
    parser = argparse.ArgumentParser(description="Multi-model Inference CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input image for prediction")
    args = parser.parse_args()

    splits = get_stratified_splits()
    test_paths = splits["test_paths"]
    test_labels = splits["test_labels"]
    classes = splits["classes"]

    if args.image:
        sample_path = Path(args.image)
        # Attempt to infer ground truth from parent folder
        gt_label = sample_path.parent.name if sample_path.parent.name in classes else "unknown"
    else:
        sample_idx = 10
        sample_path = test_paths[sample_idx]
        gt_label = classes[test_labels[sample_idx]]

    print("\n" + "=" * 60)
    print("DEMONSTRATING SINGLE-IMAGE INFERENCE")
    print(f"Sample: {sample_path}")
    print(f"Ground Truth Label: {gt_label}")
    print("=" * 60)

    demo_results = {"sample_path": str(sample_path), "ground_truth": gt_label, "models_predictions": {}}

    for model_key in ["random_forest", "mobilenet_v2", "segmentation_svm", "custom_cnn"]:
        predictor = FruitModelPredictor(model_key=model_key)
        res = predictor.predict(sample_path, ground_truth=gt_label if gt_label != "unknown" else None)
        demo_results["models_predictions"][model_key] = res
        status = "CORRECT" if res["is_correct"] else "INCORRECT" if gt_label != "unknown" else "N/A"
        print(f"[{model_key:18s}] Pred: {res['predicted_class']:10s} | Conf: {res['confidence']*100:.1f}% | Latency: {res['total_latency_ms']:.2f}ms [{status}]")

    output_json = EXAMPLES_DIR / "single_prediction_sample.json"
    with open(output_json, "w") as f:
        json.dump(demo_results, f, indent=2)
    print(f"\n[SAVE] Single-image demonstration exported to: {output_json}")


if __name__ == "__main__":
    demonstrate_single_inference()
