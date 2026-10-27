"""
Dataset Splitting, Preprocessing, and Label Management Module.
Provides stratified data splitting, reproducible dataset generators, and model-specific preprocessing.
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_TEST_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    IMAGE_SIZE_CLASSICAL,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_MOBILENET,
)


def discover_classes(train_dir: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Dynamically discover classes from subdirectories in train_dir.
    Sorts alphabetically for reproducibility and creates bidirectional index mappings.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class subdirectories discovered in {train_dir}")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

    mapping_path = MODELS_DIR / "class_labels.json"
    mapping_payload = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()}
    }
    with open(mapping_path, "w") as f:
        json.dump(mapping_payload, f, indent=2)

    return classes, class_to_idx, idx_to_class


def scan_image_files(directory: Path, class_to_idx: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """Scan all valid image files in a split directory and pair with numeric labels."""
    image_paths = []
    labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = directory / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.iterdir():
            if p.suffix.lower() in valid_extensions:
                image_paths.append(str(p))
                labels.append(cls_idx)

    return image_paths, labels


def get_stratified_splits(
    train_dir: Path = DEFAULT_TRAIN_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    val_ratio: float = 0.2,
    seed: int = RANDOM_SEED
) -> Dict[str, any]:
    """
    Create stratified train and validation splits from the training directory,
    while keeping the official test set held-out intact.
    """
    classes, class_to_idx, idx_to_class = discover_classes(train_dir)

    all_train_paths, all_train_labels = scan_image_files(train_dir, class_to_idx)
    test_paths, test_labels = scan_image_files(test_dir, class_to_idx)

    # Stratified Train/Val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths,
        all_train_labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=all_train_labels
    )

    split_info = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "train_paths": train_paths,
        "train_labels": train_labels,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "test_paths": test_paths,
        "test_labels": test_labels,
        "counts": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
            "total": len(train_paths) + len(val_paths) + len(test_paths)
        }
    }

    print(f"[DATA SPLIT] Discovered {len(classes)} classes: {classes}")
    print(f"[DATA SPLIT] Stratified Train: {len(train_paths)} | Val: {len(val_paths)} | Held-out Test: {len(test_paths)}")

    return split_info


def load_image_bgr(path: str, target_size: Tuple[int, int] = IMAGE_SIZE_CLASSICAL) -> np.ndarray:
    """Load image using OpenCV in BGR format and resize."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Unable to read image at: {path}")
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def create_tf_dataset(
    paths: List[str],
    labels: List[int],
    target_size: Tuple[int, int] = IMAGE_SIZE_CNN,
    batch_size: int = 32,
    is_training: bool = False,
    preprocess_func=None
) -> tf.data.Dataset:
    """
    Build high-throughput tf.data.Dataset pipeline with optional training-only augmentation.
    """
    def _parse_image(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, target_size)
        if preprocess_func is not None:
            img = preprocess_func(img)
        else:
            img = tf.cast(img, tf.float32) / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 1000), seed=RANDOM_SEED)

    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Augmentation without changing hue/color semantics
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
            tf.keras.layers.RandomRotation(0.08, seed=RANDOM_SEED),
            tf.keras.layers.RandomZoom(0.08, seed=RANDOM_SEED),
        ])
        dataset = dataset.map(lambda x, y: (augmentation_layers(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    splits = get_stratified_splits()
    print("Class label mapping saved to models/class_labels.json successfully.")
