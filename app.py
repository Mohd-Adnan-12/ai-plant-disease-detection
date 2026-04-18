"""
============================================================
Smart Plant Disease Detection System
app.py — Streamlit Web Application
============================================================
Features:
  - Drag-and-drop leaf image upload
  - Real-time CNN prediction (MobileNetV2 transfer learning)
  - Confidence score with animated progress bar
  - Per-class probability breakdown (bar chart)
  - Disease description, symptoms, treatment & prevention
  - Model performance dashboard (accuracy/loss curves + confusion matrix)
  - Sidebar history of recent predictions
  - Sample image testing without dataset
============================================================
Run: streamlit run app.py
"""

import os
import json
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tensorflow as tf

# Import our prediction engine
from predict import load_model_and_labels, predict, DISEASE_INFO

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #1a6b3c 0%, #2d9b5c 50%, #56c785 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(26, 107, 60, 0.3);
    }
    .hero-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-header p  { font-size: 1rem; opacity: 0.9; margin-top: 0.4rem; }

    /* ── Cards ── */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #2d9b5c;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .result-card.warning { border-left-color: #f59e0b; }
    .result-card.danger  { border-left-color: #ef4444; }

    /* ── Metric tiles ── */
    .metric-tile {
        background: #f0fdf4;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #bbf7d0;
    }
    .metric-tile .value { font-size: 2rem; font-weight: 700; color: #15803d; }
    .metric-tile .label { font-size: 0.8rem; color: #6b7280; margin-top: 0.2rem; }

    /* ── Severity badge ── */
    .badge-none    { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-moderate{ background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; }
    .badge-high    { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; }

    /* ── Upload zone ── */
    .upload-hint {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        padding: 0.5rem;
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a6b3c;
        border-bottom: 2px solid #dcfce7;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Streamlit element overrides */
    .stButton button {
        background-color: #1a6b3c;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton button:hover { background-color: #15803d; }

    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    """Load model once and cache for the session."""
    return load_model_and_labels(model_dir="models")

# ── Session State ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # List of recent prediction results

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Plant Disease Detector")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔬 Detect Disease", "📊 Model Performance", "📖 Disease Guide", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Recent predictions history
    if st.session_state.history:
        st.markdown("**🕑 Recent Predictions**")
        for item in reversed(st.session_state.history[-5:]):
            icon = DISEASE_INFO.get(item['class'], {}).get('icon', '🌱')
            conf = item['confidence']
            st.markdown(f"{icon} **{item['class']}** — {conf:.1%}")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small>🔧 Model: MobileNetV2<br>"
        "📦 Classes: Healthy / Powdery / Rust<br>",
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: Disease Detection
# ════════════════════════════════════════════════════════════════════════════
if "Detect" in page:

    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>🌿 Smart Plant Disease Detection</h1>
        <p>Upload a leaf image — our CNN model (MobileNetV2) detects disease with confidence scores,
        full disease details, and personalised treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1.3], gap="large")

    # ── Upload Column ──────────────────────────────────────────────────────
    with col_upload:
        st.markdown('<div class="section-header">📷 Upload Leaf Image</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="upload-hint">
            Supported formats: JPG, JPEG, PNG, WEBP<br>
            Tip: Use a clear, close-up photo of a single leaf for best results
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Uploaded Image", use_container_width=True)

            # Image metadata
            w, h = img.size
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f'<div class="metric-tile"><div class="value">{w}</div><div class="label">Width px</div></div>', unsafe_allow_html=True)
            col_b.markdown(f'<div class="metric-tile"><div class="value">{h}</div><div class="label">Height px</div></div>', unsafe_allow_html=True)
            col_c.markdown(f'<div class="metric-tile"><div class="value">RGB</div><div class="label">Mode</div></div>', unsafe_allow_html=True)

            analyze_btn = st.button("🔬 Analyze Leaf", use_container_width=True)
        else:
            st.info("⬆️ Upload an image to get started")
            analyze_btn = False

    # ── Result Column ──────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

        if uploaded_file and analyze_btn:
            # Load model
            with st.spinner("🧠 Loading AI model..."):
                try:
                    model, label_map = get_model()
                    model_loaded = True
                except Exception as e:
                    st.error(f"⚠️ Model not found. Please run `train.py` first.\n\n`{e}`")
                    model_loaded = False

            if model_loaded:
                # Inference with timing
                with st.spinner("🔬 Analyzing leaf..."):
                    start = time.time()
                    result = predict(model, label_map, img)
                    elapsed = time.time() - start

                disease   = result["predicted_class"]
                conf      = result["confidence"]
                all_probs = result["all_probs"]
                info      = result["disease_info"]

                # Save to history
                st.session_state.history.append({
                    "class": disease,
                    "confidence": conf
                })

                # ── Top Prediction Card ────────────────────────────────
                icon     = info.get("icon", "🌱")
                severity = info.get("severity", "Unknown")
                sev_cls  = {"None": "badge-none", "Moderate": "badge-moderate", "High": "badge-high"}.get(severity, "badge-none")

                card_cls = {"None": "", "Moderate": "warning", "High": "danger"}.get(severity, "")
                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:0.8rem;color:#6b7280;">DETECTED CONDITION</div>
                            <div style="font-size:1.8rem;font-weight:700;color:#111;">{icon} {disease}</div>
                        </div>
                        <span class="{sev_cls}">⚠️ {severity} Severity</span>
                    </div>
                    <div style="margin-top:0.8rem;font-size:0.85rem;color:#6b7280;">
                        ⚡ Inference time: {elapsed*1000:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Confidence Gauge ───────────────────────────────────
                st.markdown("**Confidence Score**")
                conf_color = "#22c55e" if conf > 0.85 else "#f59e0b" if conf > 0.65 else "#ef4444"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf * 100,
                    number={"suffix": "%", "font": {"size": 32}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": conf_color},
                        "steps": [
                            {"range": [0,  65], "color": "#fee2e2"},
                            {"range": [65, 85], "color": "#fef3c7"},
                            {"range": [85,100], "color": "#d1fae5"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.8,
                            "value": conf * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=200,
                    margin=dict(t=20, b=20, l=20, r=20),
                    font={"family": "Inter"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── All Class Probabilities ────────────────────────────
                st.markdown("**Probability Breakdown**")
                sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
                classes = [c for c, _ in sorted_probs]
                values  = [v * 100 for _, v in sorted_probs]
                colors  = ["#1a6b3c" if c == disease else "#d1fae5" for c in classes]

                fig_bar = go.Figure(go.Bar(
                    x=values, y=classes, orientation='h',
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in values],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    height=160,
                    margin=dict(t=10, b=10, l=10, r=60),
                    xaxis=dict(range=[0, 115], showticklabels=False),
                    yaxis=dict(tickfont=dict(size=13)),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        elif not uploaded_file:
            # Placeholder state
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
                <div style="font-size:4rem;">🍃</div>
                <div style="font-size:1.1rem;margin-top:1rem;">Upload a leaf image to get started</div>
                <div style="font-size:0.85rem;margin-top:0.5rem;">
                    Supports Healthy, Powdery Mildew, and Rust disease detection
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Disease Details (below, full width) ──────────────────────────────────
    if uploaded_file and analyze_btn and 'result' in dir():
        st.markdown("---")
        st.markdown('<div class="section-header">🔍 Disease Details & Recommendations</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📋 Description & Symptoms", "💊 Treatment", "🛡️ Prevention"])

        with tab1:
            col_desc, col_symp = st.columns(2)
            with col_desc:
                st.markdown("**About this condition**")
                st.info(info.get("description", "No description available."))
            with col_symp:
                st.markdown("**Common Symptoms**")
                for s in info.get("symptoms", []):
                    st.markdown(f"- {s}")

        with tab2:
            st.markdown("**Recommended Actions**")
            for t in info.get("treatment", ["No treatment info available."]):
                st.markdown(f"- {t}")

        with tab3:
            st.markdown("**Prevention Strategies**")
            for p in info.get("prevention", ["No prevention info available."]):
                st.markdown(f"- {p}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown("""
    <div class="hero-header">
        <h1>📊 Model Performance Dashboard</h1>
        <p>Training metrics, accuracy/loss curves, and confusion matrix from the trained CNN model.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Training History Charts ───────────────────────────────────────────
    history_path = "models/training_history.json"

    if os.path.exists(history_path):
        with open(history_path) as f:
            hist = json.load(f)

        epochs = list(range(1, len(hist['accuracy']) + 1))
        best_val_acc = max(hist['val_accuracy'])
        best_val_loss = min(hist['val_loss'])

        # KPI Tiles
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best Val Accuracy", f"{best_val_acc:.2%}")
        k2.metric("Best Val Loss", f"{best_val_loss:.4f}")
        k3.metric("Total Epochs", len(epochs))
        k4.metric("Architecture", "MobileNetV2")

        st.markdown("---")
        col_acc, col_loss = st.columns(2)

        with col_acc:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=epochs, y=hist['accuracy'],
                                         name='Train', line=dict(color='#2d9b5c', width=2.5)))
            fig_acc.add_trace(go.Scatter(x=epochs, y=hist['val_accuracy'],
                                          name='Validation', line=dict(color='#f59e0b', width=2.5, dash='dot')))
            fig_acc.update_layout(
                title="Accuracy over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                legend=dict(x=0.02, y=0.02),
                height=360
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=hist['loss'],
                                           name='Train', line=dict(color='#2d9b5c', width=2.5)))
            fig_loss.add_trace(go.Scatter(x=epochs, y=hist['val_loss'],
                                           name='Validation', line=dict(color='#ef4444', width=2.5, dash='dot')))
            fig_loss.update_layout(
                title="Loss over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend=dict(x=0.7, y=0.95),
                height=360
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    else:
        st.warning(
            "⚠️ No training history found. Run `train.py` to generate it.\n\n"
            "Once trained, `models/training_history.json` will appear here automatically."
        )

    # ── Confusion Matrix Image ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔢 Confusion Matrix</div>', unsafe_allow_html=True)

    cm_path = "assets/confusion_matrix.png"
    if os.path.exists(cm_path):
        col_cm, col_cr = st.columns([1, 1])
        with col_cm:
            st.image(cm_path, caption="Confusion Matrix — Test Set", use_container_width=True)
        with col_cr:
            cr_path = "assets/classification_report.txt"
            if os.path.exists(cr_path):
                with open(cr_path) as f:
                    st.markdown("**Classification Report**")
                    st.code(f.read())
    else:
        st.info(
            "Confusion matrix will appear here after training. "
            "Run `python train.py` to generate it."
        )

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: Disease Guide
# ════════════════════════════════════════════════════════════════════════════
elif "Guide" in page:
    st.markdown("""
    <div class="hero-header">
        <h1>📖 Plant Disease Reference Guide</h1>
        <p>Detailed information about the three conditions detected by our model.</p>
    </div>
    """, unsafe_allow_html=True)

    for disease_name, info in DISEASE_INFO.items():
        icon      = info["icon"]
        severity  = info["severity"]
        sev_cls   = {"None": "badge-none", "Moderate": "badge-moderate", "High": "badge-high"}.get(severity, "badge-none")
        card_cls  = {"None": "", "Moderate": "warning", "High": "danger"}.get(severity, "")

        with st.expander(f"{icon} {disease_name}  —  Severity: {severity}", expanded=(disease_name == "Healthy")):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Description**")
                st.write(info["description"])
                st.markdown("**Symptoms**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("**Treatment**")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")
                st.markdown("**Prevention**")
                for p in info["prevention"]:
                    st.markdown(f"- {p}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: About
# ════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div class="hero-header">
        <h1>ℹ️ About This Project</h1>
        <p>Smart Plant Disease Detection System — Built with MobileNetV2 + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Overview")
        st.write("""
        This system uses **Transfer Learning** with MobileNetV2 (pretrained on ImageNet)
        to classify plant leaf images into three categories:

        - 🌿 **Healthy** — No disease present
        - 🍂 **Powdery** — Powdery mildew fungal infection
        - 🍁 **Rust** — Rust fungal infection

        The model achieves high accuracy by leveraging pretrained convolutional features
        and fine-tuning on domain-specific plant disease images.
        """)

        st.markdown("### 🏗️ Architecture")
        st.code("""
MobileNetV2 (ImageNet pretrained, frozen)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(256, relu)
    └── Dropout(0.4)
    └── Dense(128, relu)
    └── Dropout(0.2)
    └── Dense(3, softmax)   ← Healthy / Powdery / Rust
        """, language="text")

    with col2:
        st.markdown("### ⚙️ Tech Stack")
        tech = {
            "Framework": "TensorFlow / Keras 3.x",
            "Base Model": "MobileNetV2 (transfer learning)",
            "UI": "Streamlit",
            "Visualizations": "Plotly",
            "Image Processing": "PIL / OpenCV",
            "Dataset": "PlantVillage (Healthy / Powdery / Rust)"
        }
        for k, v in tech.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("### 📁 Folder Structure")
        st.code("""
smart_plant_disease/
├── app.py              ← Streamlit app (this file)
├── train.py            ← Model training script
├── predict.py          ← Prediction engine + disease info
├── requirements.txt    ← Dependencies
├── models/
│   ├── plant_disease_model.keras
│   ├── plant_disease_model.h5
│   ├── class_labels.json
│   └── training_history.json
└── assets/
    ├── training_curves.png
    └── confusion_matrix.png
        """, language="text")

  