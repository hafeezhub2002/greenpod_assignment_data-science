"""
Unit and Integration Test Suite for the Computer Vision Pipeline.
Validates:
1. Dataset Label Discovery
2. Corrupted and Invalid Image Handling
3. Feature Extraction Output Shapes
4. Class-to-Index Consistency
5. Prediction Output Schema and Latency
6. Empty Input Edge Cases
7. Model Loading and Inference
8. Evaluation Metric Calculations
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_TEST_DIR,
    MODEL_PATHS,
    IMAGE_SIZE_CLASSICAL
)
from src.preprocessing import discover_classes, load_image_bgr
from src.classical_rf import extract_hsv_histogram
from src.segmentation_svm import extract_multimodal_features, segment_fruit_foreground
from src.predict import FruitModelPredictor
from sklearn.metrics import accuracy_score, f1_score


def test_dataset_label_discovery():
    """Verify that class labels are dynamically discovered and saved."""
    classes, class_to_idx, idx_to_class = discover_classes(DEFAULT_TRAIN_DIR)
    assert len(classes) == 4
    assert sorted(classes) == ["blackspot", "canker", "fresh", "grenning"]
    assert len(class_to_idx) == 4
    assert len(idx_to_class) == 4
    for cls in classes:
        assert idx_to_class[class_to_idx[cls]] == cls


def test_corrupted_image_handling(tmp_path):
    """Verify that corrupt or zero-byte files raise appropriate errors."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_A_VALID_JPEG_HEADER")

    predictor = FruitModelPredictor(model_key="random_forest")
    with pytest.raises(ValueError, match="Corrupted or unsupported"):
        predictor.predict(corrupt_file)


def test_empty_input_handling():
    """Verify that empty arrays raise ValueError."""
    predictor = FruitModelPredictor(model_key="random_forest")
    with pytest.raises(ValueError, match="Empty or invalid"):
        predictor.predict(np.array([]))


def test_feature_extraction_shapes():
    """Verify output shapes of HSV histogram, GrabCut, HOG, and LBP features."""
    dummy_bgr = np.full((128, 128, 3), 120, dtype=np.uint8)
    # Put a circle to test segmentation
    import cv2
    cv2.circle(dummy_bgr, (64, 64), 40, (0, 165, 255), -1)

    # 1. HSV Histogram: 8x8x8 = 512
    hsv_feat = extract_hsv_histogram(dummy_bgr)
    assert hsv_feat.shape == (512,)
    assert np.isclose(hsv_feat.sum(), 1.0, atol=1e-3)

    # 2. Segmentation
    masked, mask, method = segment_fruit_foreground(dummy_bgr)
    assert masked.shape == dummy_bgr.shape
    assert mask.shape == (128, 128)

    # 3. Multimodal Features: 64 + 128 + 18 = 210 (or matching config)
    multi_feat = extract_multimodal_features(dummy_bgr)
    assert len(multi_feat.shape) == 1
    assert multi_feat.shape[0] > 0


def test_class_name_mapping_consistency():
    """Verify class_labels.json matches model metadata."""
    assert MODEL_PATHS["class_labels"].exists()
    with open(MODEL_PATHS["class_labels"]) as f:
        meta = json.load(f)
    assert "classes" in meta
    assert len(meta["classes"]) == 4


def test_prediction_output_format():
    """Verify prediction output dictionary has all required keys and score distributions."""
    predictor = FruitModelPredictor(model_key="random_forest")
    dummy_bgr = np.full((128, 128, 3), 180, dtype=np.uint8)

    res = predictor.predict(dummy_bgr, ground_truth="fresh")
    assert "predicted_class" in res
    assert "confidence" in res
    assert "class_probabilities" in res
    assert "total_latency_ms" in res
    assert "is_correct" in res
    assert len(res["class_probabilities"]) == 4
    prob_sum = sum(res["class_probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-2)


def test_model_loading_and_inference():
    """Verify that all four models can be instantiated and produce predictions."""
    dummy_bgr = np.full((128, 128, 3), 150, dtype=np.uint8)

    for m_key in ["random_forest", "segmentation_svm", "custom_cnn", "mobilenet_v2"]:
        if MODEL_PATHS[m_key].exists():
            predictor = FruitModelPredictor(model_key=m_key)
            out = predictor.predict(dummy_bgr)
            assert out["predicted_class"] in predictor.classes
            assert 0.0 <= out["confidence"] <= 1.0


def test_metric_calculation_accuracy():
    """Verify metric calculations for macro F1 and accuracy edge cases."""
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    assert accuracy_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred, average="macro") == 1.0

    y_pred_wrong = [0, 1, 1, 3]
    assert accuracy_score(y_true, y_pred_wrong) == 0.75
