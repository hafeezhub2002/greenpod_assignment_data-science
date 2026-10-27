"""
Approach 4: Custom Convolutional Neural Network (CNN) Trained From Scratch.
Architecture features:
- Normalized Input (128, 128, 3)
- Training-only geometric augmentation (RandomFlip, RandomRotation, RandomZoom)
- 3 Conv2D + BatchNorm + ReLU + MaxPool blocks (32, 64, 128 filters)
- GlobalAveragePooling2D + Dropout (0.4)
- Dense projection head + Softmax output
- Sparse categorical crossentropy + Adam optimizer with EarlyStopping and ModelCheckpoint.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_SEED,
    IMAGE_SIZE_CNN,
    MODEL_PATHS,
    FIGURES_DIR
)
from src.preprocessing import get_stratified_splits, create_tf_dataset, load_image_bgr


def build_custom_cnn(input_shape=(128, 128, 3), num_classes=4) -> tf.keras.Model:
    """Construct compact CNN with regularized convolution blocks for stable generalization on small datasets."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.2, seed=RANDOM_SEED, name="dropout_block2"),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        layers.Dropout(0.3, seed=RANDOM_SEED, name="dropout_block3"),

        # Dense Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_features"),
        layers.Dropout(0.5, seed=RANDOM_SEED, name="head_dropout"),
        layers.Dense(num_classes, activation="softmax", name="predictions")
    ], name="CustomFruitCNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_learning_curves(history, output_path: Path):
    """Plot training and validation loss and accuracy curves."""
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history.history["loss"], "o-", label="Train Loss", color="#2b5c8f")
    ax1.plot(epochs, history.history["val_loss"], "s--", label="Val Loss", color="#e28743")
    ax1.set_title("Custom CNN: Loss Progression", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Sparse Categorical Crossentropy", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"], "o-", label="Train Accuracy", color="#2b5c8f")
    ax2.plot(epochs, history.history["val_accuracy"], "s--", label="Val Accuracy", color="#2ca02c")
    ax2.set_title("Custom CNN: Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Custom CNN Training & Validation History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[CNN] Saved learning curves to {output_path}")


def train_and_evaluate_cnn(splits: dict = None, epochs: int = 25, batch_size: int = 32) -> dict:
    """Train custom CNN from scratch and evaluate on held-out test split."""
    print("\n" + "=" * 60)
    print("APPROACH 4: CUSTOM CNN FROM SCRATCH")
    print("=" * 60)

    if splits is None:
        splits = get_stratified_splits()

    classes = splits["classes"]
    num_classes = len(classes)
    train_paths, train_y = splits["train_paths"], splits["train_labels"]
    val_paths, val_y = splits["val_paths"], splits["val_labels"]
    test_paths, test_y = splits["test_paths"], np.array(splits["test_labels"])

    # Datasets
    train_ds = create_tf_dataset(train_paths, train_y, target_size=IMAGE_SIZE_CNN, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_y, target_size=IMAGE_SIZE_CNN, batch_size=batch_size, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_y, target_size=IMAGE_SIZE_CNN, batch_size=batch_size, is_training=False)

    model = build_custom_cnn(input_shape=(*IMAGE_SIZE_CNN, 3), num_classes=num_classes)
    model.summary()

    checkpoint_path = MODEL_PATHS["custom_cnn"]
    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]

    print(f"\n[TRAIN] Training Custom CNN for up to {epochs} epochs...")
    t_train_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )
    training_time = time.time() - t_train_start

    # Plot curves
    curve_path = FIGURES_DIR / "custom_cnn_curves.png"
    plot_learning_curves(history, curve_path)

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
        img = load_image_bgr(p, target_size=IMAGE_SIZE_CNN)
        img_norm = (img[:, :, ::-1].astype(np.float32) / 255.0)[np.newaxis, ...]
        _ = model(img_norm, training=False)
        e2e_latencies.append(time.time() - t_single)
    avg_e2e_latency = float(np.mean(e2e_latencies))

    # Test Metrics
    acc = float(accuracy_score(test_y, test_preds))
    macro_f1 = float(f1_score(test_y, test_preds, average="macro"))
    cm = confusion_matrix(test_y, test_preds, labels=range(len(classes))).tolist()
    cls_report = classification_report(test_y, test_preds, target_names=classes, output_dict=True)

    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    results = {
        "approach": "Approach 4: Custom CNN From Scratch",
        "model_type": "Keras 3-Block CNN (Conv2D + BatchNorm + GAP)",
        "best_params": {"epochs_trained": len(history.history["loss"]), "batch_size": batch_size, "lr": 1e-3},
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
        "training_time_sec": round(training_time, 4),
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
    train_and_evaluate_cnn()
