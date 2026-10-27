"""
Visualization Module for Model Comparisons, Confusion Matrices, and Predictions.
Generates publication-quality charts for the technical report and dashboard.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    EXAMPLES_DIR,
    MODEL_PATHS
)
from src.preprocessing import get_stratified_splits, load_image_bgr


def plot_confusion_matrices(eval_data: dict, output_path: Path):
    """Plot 2x2 grid of confusion matrices for all four models."""
    classes = eval_data["metadata"]["classes"]
    models_dict = eval_data["models"]
    model_names = list(models_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    cm_cmaps = ["Blues", "Greens", "Purples", "Oranges"]

    for idx, (m_name, cmap) in enumerate(zip(model_names, cm_cmaps)):
        ax = axes[idx]
        m_info = models_dict[m_name]
        cm = np.array(m_info["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{m_name}\nAcc: {m_info['accuracy']*100:.2f}% | Macro F1: {m_info['macro_f1']*100:.2f}%",
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="bold")
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.suptitle("Held-Out Test Set Confusion Matrices Across All 4 Approaches", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[VIZ] Saved confusion matrices to {output_path}")


def plot_model_comparison_bars(eval_data: dict, output_path: Path):
    """Plot bar chart comparing Accuracy and Macro F1."""
    models_dict = eval_data["models"]
    names = [m["approach_name"].split(":")[0] + "\n" + m["model_type"].split(":")[0][:18] for m in models_dict.values()]
    accs = [m["accuracy"] * 100 for m in models_dict.values()]
    f1s = [m["macro_f1"] * 100 for m in models_dict.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#1f77b4', edgecolor='black', alpha=0.9)
    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1 (%)', color='#2ca02c', edgecolor='black', alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
    ax.set_title("Model Performance Benchmark on Held-out Test Set (N=99)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax.set_ylim(70, 105)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[VIZ] Saved performance comparison bars to {output_path}")


def plot_latency_accuracy_tradeoff(eval_data: dict, output_path: Path):
    """Scatter plot of End-to-End Latency vs Macro F1 Score."""
    models_dict = eval_data["models"]
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
    markers = ['o', 's', '^', 'D']

    for idx, (m_key, m) in enumerate(models_dict.items()):
        latency = m["end_to_end_latency_ms"]
        f1 = m["macro_f1"] * 100
        size = m["model_size_mb"] * 25 + 100  # Size reflects model size

        ax.scatter(latency, f1, s=size, color=colors[idx], marker=markers[idx],
                   edgecolors='black', linewidth=1.5, alpha=0.85, label=f"{m_key} ({m['model_size_mb']:.1f} MB)")
        ax.annotate(
            f"{m_key.split(':')[1].strip()}\n({latency:.1f}ms, {f1:.1f}%)",
            (latency, f1),
            textcoords="offset points",
            xytext=(10, 5),
            ha='left',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_title("Accuracy vs. Latency Trade-off Across Approaches", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("End-to-End Latency per Image (ms) [Lower is Better]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Held-out Macro F1 Score (%) [Higher is Better]", fontsize=12, fontweight='bold')
    ax.set_ylim(80, 102)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Model & Serialized Size", fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[VIZ] Saved latency vs accuracy tradeoff to {output_path}")


def plot_example_predictions(splits: dict, eval_data: dict, output_path: Path):
    """Save grid of real test images with predictions from all 4 models."""
    classes = eval_data["metadata"]["classes"]
    test_paths = splits["test_paths"]
    test_y = splits["test_labels"]

    # Select one representative sample per class from test set
    sample_indices = []
    for c_idx in range(len(classes)):
        match_idxs = [i for i, y in enumerate(test_y) if y == c_idx]
        if match_idxs:
            sample_indices.append(match_idxs[0])

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3.8 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, s_idx in enumerate(sample_indices):
        img_path = test_paths[s_idx]
        true_label = classes[test_y[s_idx]]
        img = Image.open(img_path)

        # Image view
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"Ground Truth: {true_label}\nFile: {Path(img_path).name}", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis("off")

        # Predictions breakdown
        pred_texts = []
        for m_name, m_info in eval_data["models"].items():
            pred_idx = m_info["test_predictions"][s_idx]
            pred_label = classes[pred_idx]
            conf = m_info["test_probabilities"][s_idx][pred_idx] * 100
            correct = (pred_idx == test_y[s_idx])
            mark = "CORRECT" if correct else "WRONG"
            pred_texts.append(f"{m_name.split(':')[0]}: {pred_label} ({conf:.1f}% conf) [{mark}]")

        axes[row_idx, 1].text(
            0.05, 0.5,
            "\n\n".join(pred_texts),
            fontsize=11,
            va="center",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da")
        )
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title("Multi-Model Predictions Breakdown", fontsize=11, fontweight='bold')

    plt.suptitle("Real Held-Out Predictions Across All 4 Models", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[VIZ] Saved example predictions grid to {output_path}")


def run_visualizations():
    """Execute all visualizations from saved evaluation metrics."""
    json_path = METRICS_DIR / "evaluation_comparison.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation metrics JSON not found: {json_path}")

    with open(json_path, "r") as f:
        eval_data = json.load(f)

    splits = get_stratified_splits()

    cm_path = FIGURES_DIR / "confusion_matrices.png"
    plot_confusion_matrices(eval_data, cm_path)

    bars_path = FIGURES_DIR / "model_comparison_bars.png"
    plot_model_comparison_bars(eval_data, bars_path)

    tradeoff_path = FIGURES_DIR / "latency_vs_accuracy.png"
    plot_latency_accuracy_tradeoff(eval_data, tradeoff_path)

    example_path = FIGURES_DIR / "example_predictions_grid.png"
    plot_example_predictions(splits, eval_data, example_path)


if __name__ == "__main__":
    run_visualizations()
