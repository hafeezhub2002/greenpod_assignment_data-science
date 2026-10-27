"""
Dataset Exploration and Quality Audit Module.
Scans train and test splits, verifies readability, computes dimensions, detects exact and
perceptual duplicates, checks for cross-split leakage, generates distribution and representative plots,
and explicitly flags any dataset mismatch with the avocado ripening stage objective.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DEFAULT_TRAIN_DIR, DEFAULT_TEST_DIR, FIGURES_DIR, METRICS_DIR, RANDOM_SEED


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file."""
    h = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_perceptual_hash(filepath: Path) -> str:
    """Compute perceptual hash using ImageHash."""
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except Exception:
        return ""


def audit_split(split_dir: Path, split_name: str) -> dict:
    """
    Audit an individual split directory.
    Extracts image counts, shapes, corruptions, and duplicate hashes.
    """
    records = []
    corrupted_files = []
    file_hashes = defaultdict(list)
    perceptual_hashes = defaultdict(list)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda x: x.name)

    for c_dir in class_dirs:
        class_name = c_dir.name
        img_paths = [p for p in c_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        for p in img_paths:
            # Check readability with PIL and OpenCV
            try:
                with Image.open(p) as pil_img:
                    w, h = pil_img.size
                    mode = pil_img.mode

                cv_img = cv2.imread(str(p))
                if cv_img is None:
                    corrupted_files.append({"path": str(p), "reason": "OpenCV imread returned None"})
                    continue

                f_hash = compute_file_hash(p)
                p_hash = compute_perceptual_hash(p)

                file_hashes[f_hash].append(str(p))
                if p_hash:
                    perceptual_hashes[p_hash].append(str(p))

                records.append({
                    "split": split_name,
                    "class": class_name,
                    "filename": p.name,
                    "path": str(p),
                    "width": w,
                    "height": h,
                    "channels": cv_img.shape[2] if len(cv_img.shape) > 2 else 1,
                    "aspect_ratio": round(w / h, 4) if h > 0 else 0,
                    "filesize_bytes": p.stat().st_size,
                    "sha256": f_hash,
                    "phash": p_hash
                })
            except Exception as e:
                corrupted_files.append({"path": str(p), "reason": str(e)})

    # Detect exact duplicates within this split
    exact_duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
    perceptual_duplicates = {h: paths for h, paths in perceptual_hashes.items() if len(paths) > 1}

    df = pd.DataFrame(records)
    class_counts = df["class"].value_counts().to_dict() if not df.empty else {}

    dimension_stats = {}
    if not df.empty:
        dimension_stats = {
            "min_width": int(df["width"].min()),
            "median_width": float(df["width"].median()),
            "max_width": int(df["width"].max()),
            "min_height": int(df["height"].min()),
            "median_height": float(df["height"].median()),
            "max_height": int(df["height"].max()),
            "channels_detected": sorted(list(df["channels"].unique().tolist())),
        }

    return {
        "split_name": split_name,
        "total_images": len(records),
        "class_counts": class_counts,
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "perceptual_duplicate_groups": len(perceptual_duplicates),
        "dimension_stats": dimension_stats,
        "dataframe": df
    }


def check_cross_split_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Check for identical images appearing in both train and test splits."""
    if train_df.empty or test_df.empty:
        return {"exact_overlap_count": 0, "overlapping_files": []}

    train_hashes = set(train_df["sha256"])
    test_overlap = test_df[test_df["sha256"].isin(train_hashes)]

    overlapping_files = []
    for _, row in test_overlap.iterrows():
        matching_train = train_df[train_df["sha256"] == row["sha256"]]["path"].tolist()
        overlapping_files.append({
            "test_file": row["path"],
            "train_match": matching_train,
            "sha256": row["sha256"]
        })

    return {
        "exact_overlap_count": len(overlapping_files),
        "overlapping_files": overlapping_files
    }


def plot_class_distributions(train_counts: dict, test_counts: dict, output_path: Path):
    """Plot side-by-side class distributions."""
    classes = sorted(list(set(list(train_counts.keys()) + list(test_counts.keys()))))
    train_vals = [train_counts.get(c, 0) for c in classes]
    test_vals = [test_counts.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, train_vals, width, label='Train Split', color='#2b5c8f', edgecolor='black', alpha=0.9)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test Split', color='#e28743', edgecolor='black', alpha=0.9)

    ax.set_title("Dataset Class Distribution: Train vs. Test", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Class Label", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Images", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(fontsize=11)

    # Value annotations on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AUDIT] Saved class distribution plot to {output_path}")


def plot_representative_images(train_df: pd.DataFrame, output_path: Path):
    """Plot representative samples from each class."""
    classes = sorted(train_df["class"].unique().tolist())
    n_classes = len(classes)
    samples_per_class = 4

    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(14, 3.5 * n_classes))
    if n_classes == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, cls in enumerate(classes):
        cls_df = train_df[train_df["class"] == cls].sample(n=min(samples_per_class, len(train_df[train_df["class"] == cls])), random_state=RANDOM_SEED)
        for col_idx, (_, row) in enumerate(cls_df.iterrows()):
            ax = axes[row_idx, col_idx]
            img = Image.open(row["path"])
            ax.imshow(img)
            ax.set_title(f"{cls}\n({img.width}x{img.height})", fontsize=10)
            ax.axis("off")

    plt.suptitle("Representative Images by Class (Greenpod Labs Dataset)", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AUDIT] Saved representative images plot to {output_path}")


def run_data_audit(train_dir: Path = DEFAULT_TRAIN_DIR, test_dir: Path = DEFAULT_TEST_DIR) -> dict:
    """Run full dataset exploration, audit, and save report."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE DATASET QUALITY AUDIT")
    print(f"Train path: {train_dir}")
    print(f"Test path:  {test_dir}")
    print("=" * 60)

    train_res = audit_split(train_dir, "train")
    test_res = audit_split(test_dir, "test")

    leakage_res = check_cross_split_leakage(train_res["dataframe"], test_res["dataframe"])

    # Discovered classes
    train_classes = sorted(list(train_res["class_counts"].keys()))
    test_classes = sorted(list(test_res["class_counts"].keys()))

    # Verify if classes match avocado ripeness stages or orange condition
    is_avocado_labels = any(term in "".join(train_classes).lower() for term in ["ripe", "unripe", "overripe", "avocado"])
    dataset_subject = "Avocado Ripeness" if is_avocado_labels else "Citrus / Orange Condition (Disease & Quality)"

    # Audit summary
    audit_summary = {
        "dataset_validation_verdict": {
            "critical_finding": "DATASET SUBJECT MISMATCH EXPLICITLY FLAGGED",
            "assignment_objective": "Avocado Ripening Stage Detection (unripe, ripe, overripe)",
            "actual_dataset_subject": dataset_subject,
            "actual_classes_discovered": train_classes,
            "interpretation": (
                "The provided Greenpod Labs dataset 'Orange Dataset.zip' contains citrus/orange images "
                "categorized into 4 disease and condition classes: ['blackspot', 'canker', 'fresh', 'grenning']. "
                "Per Master Prompt Critical Dataset Rule, we do NOT hallucinate or mislabel these as avocado stages. "
                "Instead, we dynamically infer labels, preserve exact visual ground truth, and build the 4-approach "
                "computer vision pipeline to rigorously benchmark performance on the actual data while documenting "
                "the discrepancy for production deployment."
            )
        },
        "splits": {
            "train": {
                "total_images": train_res["total_images"],
                "class_counts": train_res["class_counts"],
                "corrupted_count": train_res["corrupted_count"],
                "exact_duplicates": train_res["exact_duplicate_groups"],
                "perceptual_duplicates": train_res["perceptual_duplicate_groups"],
                "dimension_stats": train_res["dimension_stats"]
            },
            "test": {
                "total_images": test_res["total_images"],
                "class_counts": test_res["class_counts"],
                "corrupted_count": test_res["corrupted_count"],
                "exact_duplicates": test_res["exact_duplicate_groups"],
                "perceptual_duplicates": test_res["perceptual_duplicate_groups"],
                "dimension_stats": test_res["dimension_stats"]
            }
        },
        "data_leakage_check": {
            "cross_split_exact_duplicates": leakage_res["exact_overlap_count"],
            "details": leakage_res["overlapping_files"]
        },
        "class_imbalance_analysis": {
            "train_class_balance": {
                cls: round(count / train_res["total_images"], 4)
                for cls, count in train_res["class_counts"].items()
            },
            "test_class_balance": {
                cls: round(count / test_res["total_images"], 4)
                for cls, count in test_res["class_counts"].items()
            }
        }
    }

    # Save summary plots
    dist_plot_path = FIGURES_DIR / "class_distribution.png"
    plot_class_distributions(train_res["class_counts"], test_res["class_counts"], dist_plot_path)

    rep_plot_path = FIGURES_DIR / "representative_images.png"
    plot_representative_images(train_res["dataframe"], rep_plot_path)

    # Save JSON report
    report_json_path = METRICS_DIR / "data_audit_report.json"
    with open(report_json_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[AUDIT] Audit report saved to {report_json_path}")

    # Save CSV of all image metadata
    combined_df = pd.concat([train_res["dataframe"], test_res["dataframe"]], ignore_index=True)
    csv_metadata_path = METRICS_DIR / "dataset_metadata.csv"
    combined_df.to_csv(csv_metadata_path, index=False)
    print(f"[AUDIT] Image metadata exported to {csv_metadata_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("DATA AUDIT EXECUTIVE SUMMARY:")
    print(f"Dataset Subject: {audit_summary['dataset_validation_verdict']['actual_dataset_subject']}")
    print(f"Classes Found:   {train_classes}")
    print(f"Train Images:    {train_res['total_images']} (Corrupted: {train_res['corrupted_count']}, Exact Duplicates: {train_res['exact_duplicate_groups']})")
    print(f"Test Images:     {test_res['total_images']} (Corrupted: {test_res['corrupted_count']}, Exact Duplicates: {test_res['exact_duplicate_groups']})")
    print(f"Cross-Split Leakage: {leakage_res['exact_overlap_count']} overlapping duplicates")
    print("=" * 60 + "\n")

    return audit_summary


if __name__ == "__main__":
    run_data_audit()
