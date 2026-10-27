"""
Approach 2: Transfer Learning with Pretrained MobileNetV2.
Leverages ImageNet features, trains a custom classification head on frozen backbone,
fine-tunes upper layers with a gentle learning rate, monitors validation metrics,
and serializes the best performing checkpoint.
"""

import sys
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_SEED,
    IMAGE_SIZE_MOBILENET,
    MODEL_PATHS,
    FIGURES_DIR
)
from src.preprocessing import get_stratified_splits, create_tf_dataset, load_image_bgr


def build_mobilenet_transfer_model(num_classes=4, input_shape=(224, 224, 3)) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Construct MobileNetV2 transfer learning model.
    Backbone is frozen initially.
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    try:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        print("[MOBILENET] Successfully loaded ImageNet pretrained weights.")
    except Exception as e:
        print(f"[MOBILENET WARNING] Failed to load ImageNet weights ({e}), falling back to random init.")
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    # Base model pass (training=False keeps BN in inference mode)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, seed=RANDOM_SEED, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def plot_combined_curves(history_phase1, history_phase2, output_path: Path):
    """Plot combined Phase 1 (frozen) and Phase 2 (fine-tuning) learning curves."""
    loss = history_phase1.history["loss"] + (history_phase2.history["loss"] if history_phase2 else [])
    val_loss = history_phase1.history["val_loss"] + (history_phase2.history["val_loss"] if history_phase2 else [])
    acc = history_phase1.history["accuracy"] + (history_phase2.history["accuracy"] if history_phase2 else [])
    val_acc = history_phase1.history["val_accuracy"] + (history_phase2.history["val_accuracy"] if history_phase2 else [])

    epochs = range(1, len(loss) + 1)
    split_epoch = len(history_phase1.history["loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, loss, "o-", label="Train Loss", color="#2b5c8f")
    ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="#e28743")
    if history_phase2:
        ax1.axvline(x=split_epoch, color="red", linestyle=":", label="Fine-Tuning Start")
    ax1.set_title("MobileNetV2: Crossentropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy plot
    ax2.plot(epochs, acc, "o-", label="Train Accuracy", color="#2b5c8f")
    ax2.plot(epochs, val_acc, "s--", label="Val Accuracy", color="#2ca02c")
    if history_phase2:
        ax2.axvline(x=split_epoch, color="red", linestyle=":", label="Fine-Tuning Start")
    ax2.set_title("MobileNetV2: Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("MobileNetV2 Transfer Learning & Fine-Tuning History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[MOBILENET] Saved learning curves to {output_path}")


def train_and_evaluate_mobilenet(
    splits: dict = None,
    phase1_epochs: int = 10,
    phase2_epochs: int = 8,
    batch_size: int = 32
) -> dict:
    """Train MobileNetV2 with frozen backbone followed by fine-tuning."""
    print("\n" + "=" * 60)
    print("APPROACH 2: MOBILENETV2 TRANSFER LEARNING")
    print("=" * 60)

    if splits is None:
        splits = get_stratified_splits()

    classes = splits["classes"]
    num_classes = len(classes)
    train_paths, train_y = splits["train_paths"], splits["train_labels"]
    val_paths, val_y = splits["val_paths"], splits["val_labels"]
    test_paths, test_y = splits["test_paths"], np.array(splits["test_labels"])

    # Datasets with MobileNetV2 preprocessing
    train_ds = create_tf_dataset(
        train_paths, train_y,
        target_size=IMAGE_SIZE_MOBILENET,
        batch_size=batch_size,
        is_training=True,
        preprocess_func=preprocess_input
    )
    val_ds = create_tf_dataset(
        val_paths, val_y,
        target_size=IMAGE_SIZE_MOBILENET,
        batch_size=batch_size,
        is_training=False,
        preprocess_func=preprocess_input
    )
    test_ds = create_tf_dataset(
        test_paths, test_y,
        target_size=IMAGE_SIZE_MOBILENET,
        batch_size=batch_size,
        is_training=False,
        preprocess_func=preprocess_input
    )

    model, base_model = build_mobilenet_transfer_model(num_classes=num_classes)
    checkpoint_path = MODEL_PATHS["mobilenet_v2"]

    cb_phase1 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    ]

    print(f"\n[PHASE 1] Training classification head with frozen backbone ({phase1_epochs} epochs)...")
    t0 = time.time()
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=cb_phase1,
        verbose=1
    )
    phase1_time = time.time() - t0

    # Phase 2: Fine-Tuning top 30 layers
    print(f"\n[PHASE 2] Fine-tuning top 30 backbone layers ({phase2_epochs} epochs)...")
    base_model.trainable = True
    # Freeze all layers before the 124th layer
    for layer in base_model.layers[:124]:
        layer.trainable = False
    # Keep BatchNormalization layers in inference mode to preserve moving stats
    for layer in base_model.layers[124:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_phase2 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    ]

    t1 = time.time()
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=cb_phase2,
        verbose=1
    )
    phase2_time = time.time() - t1
    total_train_time = phase1_time + phase2_time

    # Plot curves
    curve_path = FIGURES_DIR / "mobilenet_v2_curves.png"
    plot_combined_curves(history_phase1, history_phase2, curve_path)

    # Benchmark Test Inference (Model-only and End-to-End)
    # Warmup
    for batch_x, _ in test_ds.take(1):
        _ = model(batch_x, training=False)

    t_inf_start = time.time()
    test_probs = model.predict(test_ds, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    model_inference_time = (time.time() - t_inf_start) / len(test_y)

    # End-to-end latency measurement
    e2e_latencies = []
    for p in test_paths:
        t_single = time.time()
        img = load_image_bgr(p, target_size=IMAGE_SIZE_MOBILENET)
        # Convert BGR to RGB and apply preprocess_input
        img_rgb = img[:, :, ::-1].astype(np.float32)
        img_prep = preprocess_input(img_rgb)[np.newaxis, ...]
        _ = model(img_prep, training=False)
        e2e_latencies.append(time.time() - t_single)
    avg_e2e_latency = float(np.mean(e2e_latencies))

    # Test Metrics
    acc = float(accuracy_score(test_y, test_preds))
    macro_f1 = float(f1_score(test_y, test_preds, average="macro"))
    cm = confusion_matrix(test_y, test_preds, labels=range(len(classes))).tolist()
    cls_report = classification_report(test_y, test_preds, target_names=classes, output_dict=True)

    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    results = {
        "approach": "Approach 2: MobileNetV2 Transfer Learning",
        "model_type": "MobileNetV2 (ImageNet Pretrained) + Fine-Tuning",
        "best_params": {
            "phase1_epochs": len(history_phase1.history["loss"]),
            "phase2_epochs": len(history_phase2.history["loss"]),
            "phase1_lr": 1e-3,
            "phase2_lr": 1e-5,
            "batch_size": batch_size
        },
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
        "training_time_sec": round(total_train_time, 4),
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
    print(f"  Saved to:   {checkpoint_path}")

    return results


if __name__ == "__main__":
    train_and_evaluate_mobilenet()
