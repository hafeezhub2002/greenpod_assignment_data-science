"""
Interactive Streamlit Dashboard for Fruit Ripeness and Condition Classification.
Integrates all four trained computer vision models:
1. Approach 1: Classical HSV Color Histogram + Random Forest
2. Approach 2: MobileNetV2 Transfer Learning
3. Approach 3: Foreground Segmentation + Multi-Modal Features + SVM
4. Approach 4: Custom Compact CNN From Scratch
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from src.config import (
    MODEL_PATHS,
    FIGURES_DIR,
    METRICS_DIR,
    DEFAULT_TEST_DIR,
    get_hardware_info
)
from src.predict import FruitModelPredictor

# Page Setup
st.set_page_config(
    page_title="Fruit Ripeness & Condition Computer Vision Suite",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .banner-warning {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        color: #2b1100;
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 20px;
        border-left: 6px solid #d9534f;
    }
    .badge-class {
        display: inline-block;
        padding: 6px 14px;
        font-size: 14px;
        font-weight: 700;
        border-radius: 20px;
        color: white;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-correct { background-color: #28a745; }
    .badge-wrong { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_predictors():
    """Cache and load all available trained predictors."""
    predictors = {}
    for key in ["random_forest", "mobilenet_v2", "segmentation_svm", "custom_cnn"]:
        if MODEL_PATHS[key].exists():
            try:
                predictors[key] = FruitModelPredictor(model_key=key)
            except Exception as e:
                st.warning(f"Could not load {key}: {e}")
    return predictors


@st.cache_data
def load_audit_and_metrics():
    """Load precomputed audit report and evaluation comparison data."""
    audit_data, eval_data, eval_df = None, None, None

    if MODEL_PATHS["audit_report"].exists():
        with open(MODEL_PATHS["audit_report"], "r") as f:
            audit_data = json.load(f)

    if MODEL_PATHS["evaluation_metrics"].exists():
        with open(MODEL_PATHS["evaluation_metrics"], "r") as f:
            eval_data = json.load(f)

    if MODEL_PATHS["evaluation_csv"].exists():
        eval_df = pd.read_csv(MODEL_PATHS["evaluation_csv"])

    return audit_data, eval_data, eval_df


def main():
    predictors = load_all_predictors()
    audit_data, eval_data, eval_df = load_audit_and_metrics()
    hw_info = get_hardware_info()

    # Sidebar
    st.sidebar.title("🍊 Vision Pipeline")
    st.sidebar.caption("Avocado Ripening & Citrus Condition Classifier")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Active Models")
    model_status_map = {
        "Approach 1: Random Forest": "random_forest" in predictors,
        "Approach 2: MobileNetV2": "mobilenet_v2" in predictors,
        "Approach 3: GrabCut + SVM": "segmentation_svm" in predictors,
        "Approach 4: Custom CNN": "custom_cnn" in predictors,
    }
    for m_name, is_ready in model_status_map.items():
        status_icon = "🟢 Ready" if is_ready else "🔴 Missing"
        st.sidebar.write(f"**{m_name}**: {status_icon}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Hardware")
    st.sidebar.write(f"**Execution Device**: `{hw_info['default_device']}`")
    st.sidebar.write(f"**GPU Accelerated**: `{'Yes' if hw_info['gpu_available'] else 'No (CPU Mode)'}`")

    # Header
    st.title("🥑 Avocado Ripening Stage & Citrus Condition Detection Suite")
    st.markdown(
        "A rigorous, multi-paradigm computer vision pipeline comparing **Classical HSV Histograms + Random Forest**, "
        "**Pretrained MobileNetV2 Transfer Learning**, **OpenCV GrabCut Segmentation + SVM**, and a **Custom CNN from Scratch**."
    )

    # Mandatory Critical Dataset Disclosure Banner
    st.markdown("""
    <div class="banner-warning">
        ⚠️ <strong>MANDATORY DATASET AUDIT DISCLOSURE:</strong><br>
        The assignment specifies Avocado Ripening Stage Detection (unripe, ripe, overripe). However, the actual dataset provided 
        by Greenpod Labs (<code>Orange Dataset.zip</code>) consists of <strong>1,090 citrus/orange images</strong> labeled into 4 conditions: 
        <code>['blackspot', 'canker', 'fresh', 'grenning']</code>. Per Master Prompt integrity rules, this pipeline dynamically infers 
        the true dataset classes and evaluates all 4 algorithms on real ground truth without mislabeling orange condition images as avocados.
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Inference",
        "📊 Model Benchmarks",
        "🔬 Dataset Audit & Quality",
        "🧠 Architecture & Production"
    ])

    # =========================================================================
    # TAB 1: LIVE INFERENCE
    # =========================================================================
    with tab1:
        st.subheader("Interactive Fruit Image Classification")
        col_input, col_pred = st.columns([1, 1.2])

        with col_input:
            st.markdown("#### 1. Provide an Image")
            input_mode = st.radio("Choose Input Mode:", ["Upload Image File", "Select Held-Out Test Sample"], horizontal=True)

            selected_image = None
            ground_truth_label = None

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Upload fruit image (JPG, JPEG, PNG, WEBP)",
                    type=["jpg", "jpeg", "png", "webp"]
                )
                if uploaded_file is not None:
                    try:
                        selected_image = Image.open(uploaded_file)
                        st.image(selected_image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error opening image file: {e}")
            else:
                # Test sample picker
                if DEFAULT_TEST_DIR.exists():
                    test_classes = sorted([d.name for d in DEFAULT_TEST_DIR.iterdir() if d.is_dir()])
                    pick_class = st.selectbox("Select Ground-Truth Class Folder:", test_classes)
                    sample_files = list((DEFAULT_TEST_DIR / pick_class).glob("*.*"))
                    if sample_files:
                        pick_file = st.selectbox("Select Image Sample:", sample_files, format_func=lambda p: p.name)
                        ground_truth_label = pick_class
                        selected_image = Image.open(pick_file)
                        st.image(selected_image, caption=f"Sample: {pick_file.name} | True Label: {ground_truth_label}", use_column_width=True)
                else:
                    st.warning("Test directory not found on disk.")

            st.markdown("#### 2. Model Selection")
            model_options = ["Run All 4 Models Simultaneously"] + list(predictors.keys())
            model_choice = st.selectbox(
                "Select Approach to Run:",
                model_options,
                format_func=lambda x: {
                    "Run All 4 Models Simultaneously": "⚡ Run All 4 Models Simultaneously",
                    "random_forest": "Approach 1: HSV Color Histogram + Random Forest",
                    "mobilenet_v2": "Approach 2: MobileNetV2 Transfer Learning",
                    "segmentation_svm": "Approach 3: Segmentation + HOG/LBP + SVM",
                    "custom_cnn": "Approach 4: Custom CNN From Scratch",
                }.get(x, x)
            )

        with col_pred:
            st.markdown("#### 3. Inference Results")
            if selected_image is None:
                st.info("👈 Please upload an image or select a sample from the test partition to begin inference.")
            else:
                if model_choice == "Run All 4 Models Simultaneously":
                    st.markdown("##### ⚡ Multi-Model Comparative Run")
                    res_cards = st.container()

                    comparison_list = []
                    for k in ["random_forest", "mobilenet_v2", "segmentation_svm", "custom_cnn"]:
                        if k in predictors:
                            pred_res = predictors[k].predict(selected_image, ground_truth=ground_truth_label)
                            correct = pred_res.get("is_correct", None)
                            badge_color = "badge-correct" if correct else ("badge-wrong" if correct is False else "")
                            comparison_list.append({
                                "Approach": k.replace("_", " ").title(),
                                "Predicted Class": pred_res["predicted_class"],
                                "Confidence": f"{pred_res['confidence']*100:.1f}%",
                                "End-to-End Latency": f"{pred_res['total_latency_ms']:.2f} ms",
                                "Verdict": "✅ Correct" if correct else ("❌ Incorrect" if correct is False else "N/A")
                            })

                    comp_df = pd.DataFrame(comparison_list)
                    st.dataframe(comp_df, use_container_width=True)

                    # Multi-model probability bar chart
                    st.markdown("##### Class Probability Comparison")
                    chart_data = {}
                    for k in ["random_forest", "mobilenet_v2", "segmentation_svm", "custom_cnn"]:
                        if k in predictors:
                            pred_res = predictors[k].predict(selected_image, ground_truth=ground_truth_label)
                            chart_data[k.replace("_", " ").title()] = pred_res["class_probabilities"]

                    prob_plot_df = pd.DataFrame(chart_data)
                    st.bar_chart(prob_plot_df)

                else:
                    pred_res = predictors[model_choice].predict(selected_image, ground_truth=ground_truth_label)
                    correct = pred_res.get("is_correct", None)

                    # Result Card
                    pred_cls = pred_res["predicted_class"]
                    conf = pred_res["confidence"] * 100
                    lat = pred_res["total_latency_ms"]

                    st.success(f"### Predicted Class: **{pred_cls.upper()}** ({conf:.1f}% Confidence)")

                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Predicted Class", pred_cls)
                    metric_cols[1].metric("Confidence", f"{conf:.1f}%")
                    metric_cols[2].metric("Latency", f"{lat:.1f} ms")

                    if ground_truth_label is not None:
                        is_match = (pred_cls.lower() == ground_truth_label.lower())
                        badge = "✅ CORRECT MATCH" if is_match else "❌ MISCLASSIFICATION"
                        st.info(f"**Ground Truth Validation:** True Class = `{ground_truth_label}` -> Result: **{badge}**")

                    st.caption(f"Score Nature: {pred_res['score_interpretation']}")

                    # Probability distribution
                    st.markdown("##### Class Score Distribution")
                    st.bar_chart(pd.Series(pred_res["class_probabilities"], name="Score"))

    # =========================================================================
    # TAB 2: BENCHMARKS & EVALUATION
    # =========================================================================
    with tab2:
        st.subheader("Rigorous Experimental Evaluation (Held-Out Test Set, N=99)")

        if eval_df is not None:
            st.markdown("#### Consolidated Benchmark Table")
            st.dataframe(eval_df, use_container_width=True)

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            st.markdown("#### Model Performance Metrics")
            fig_bars = FIGURES_DIR / "model_comparison_bars.png"
            if fig_bars.exists():
                st.image(str(fig_bars), use_column_width=True)

        with col_fig2:
            st.markdown("#### Latency vs. Accuracy Trade-off")
            fig_tradeoff = FIGURES_DIR / "latency_vs_accuracy.png"
            if fig_tradeoff.exists():
                st.image(str(fig_tradeoff), use_column_width=True)

        st.markdown("---")
        st.markdown("#### Confusion Matrices (All 4 Approaches)")
        fig_cm = FIGURES_DIR / "confusion_matrices.png"
        if fig_cm.exists():
            st.image(str(fig_cm), use_column_width=True)

        st.markdown("---")
        st.markdown("#### Example Predictions Grid on Test Set")
        fig_ex = FIGURES_DIR / "example_predictions_grid.png"
        if fig_ex.exists():
            st.image(str(fig_ex), use_column_width=True)

    # =========================================================================
    # TAB 3: DATA AUDIT & QUALITY
    # =========================================================================
    with tab3:
        st.subheader("Comprehensive Dataset Exploration & Integrity Audit")

        if audit_data:
            s_train = audit_data["splits"]["train"]
            s_test = audit_data["splits"]["test"]

            m_cols = st.columns(4)
            m_cols[0].metric("Total Dataset Images", s_train["total_images"] + s_test["total_images"])
            m_cols[1].metric("Train Split", s_train["total_images"])
            m_cols[2].metric("Held-Out Test Split", s_test["total_images"])
            m_cols[3].metric("Corrupted Files", s_train["corrupted_count"] + s_test["corrupted_count"])

            st.markdown("#### Data Split Counts & Integrity")
            c_df = pd.DataFrame({
                "Class Label": list(s_train["class_counts"].keys()),
                "Train Count": list(s_train["class_counts"].values()),
                "Test Count": [s_test["class_counts"].get(k, 0) for k in s_train["class_counts"].keys()]
            })
            c_df["Total"] = c_df["Train Count"] + c_df["Test Count"]
            st.dataframe(c_df, use_container_width=True)

        col_audit1, col_audit2 = st.columns(2)
        with col_audit1:
            st.markdown("#### Class Distributions")
            fig_dist = FIGURES_DIR / "class_distribution.png"
            if fig_dist.exists():
                st.image(str(fig_dist), use_column_width=True)

        with col_audit2:
            st.markdown("#### Segmentation Quality Validation")
            fig_seg = FIGURES_DIR / "segmentation_validation.png"
            if fig_seg.exists():
                st.image(str(fig_seg), use_column_width=True)

        st.markdown("#### Representative Class Samples")
        fig_rep = FIGURES_DIR / "representative_images.png"
        if fig_rep.exists():
            st.image(str(fig_rep), use_column_width=True)

    # =========================================================================
    # TAB 4: ARCHITECTURE & PRODUCTION
    # =========================================================================
    with tab4:
        st.subheader("Methodology, Trade-offs & Production Deployment")

        st.markdown("""
        ### Four Computer Vision Approaches Compared

        | Approach | Feature Extraction / Architecture | Classifier | Strengths | Drawbacks |
        |---|---|---|---|---|
        | **Approach 1** | Normalized 3D HSV Color Histogram (512 dims) | Random Forest (50 trees, balanced weights) | Ultra-fast inference (0.6ms model-only), tiny size (0.37 MB), 96.97% test accuracy. | Ignores spatial geometry and spot morphology; sensitive to background illumination shifts. |
        | **Approach 2** | Pretrained MobileNetV2 (ImageNet weights, 154 layers) | Fine-Tuned Dense Head (Softmax) | Strong semantic feature extraction, 94.95% accuracy, smooth convergence. | Higher CPU latency (183 ms end-to-end), larger model memory footprint (20.5 MB). |
        | **Approach 3** | GrabCut Foreground Mask + HSV + HOG + LBP (370 dims) | RBF Support Vector Machine (C=5, scale) | Captures gradient shape and micro-surface texture; robust fallback prevents empty masks. | GrabCut computational overhead (55 ms latency); sensitive to segmentation edge artifacts (87.88% acc). |
        | **Approach 4** | Custom 3-Block CNN (Conv2D + MaxPool + Dropout + Dense) | Softmax Classification Head | Highest measured test accuracy (**97.98%**) and Macro F1 (**97.72%**), fast inference (36.6 ms). | Requires training from scratch; larger parameter footprint without GAP pruning (49.1 MB). |

        ---

        ### Production Deployment Architecture
        1. **Deployment Architecture**: Containerized FastAPI service running on edge sortation nodes or cloud inference endpoints.
        2. **Camera & Lighting Rig**: Industrial controlled diffuse dome lighting (5000K daylight LED) to eliminate specularity and shadows.
        3. **Throughput**: Approach 1 achieves >25 FPS on single-core CPU; Custom CNN achieves ~27 FPS; MobileNetV2 requires batching or ONNX Runtime quantization for high conveyor line speeds (>60 FPS).
        4. **Handling Data Shift / Avocado Adaptation**: When transitioning from this citrus proof-of-concept to genuine avocado sorting:
           - Ingest multi-angle avocado imagery (calyx, cheek, tip).
           - Incorporate firmness sensor telemetry (destructive penetrometer ground truth).
           - Implement automated drift detection using population stability index (PSI) on prediction distributions.
        """)


if __name__ == "__main__":
    main()
